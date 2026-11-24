"""CT rungs on real LIDC crops: --rung 1 = planted corner marker (PROTOCOL.md section D),
--rung 2 = faint, partly correlated stripe pattern (section E), --rung 2b = fainter 80/20 stripes
(amp 0.01) on size-normalised crops, fixed-epoch training (sections F3, G; final setting G11). Default: rung 1.

Stages (run in order):
  zoo        train 3 marker + 3 control models, shortcut check, opaque IDs
  serialize  weights -> text (serialization v1), token estimates
  tokens     exact token counts via the API (needs ANTHROPIC_API_KEY in .env)
  audit      5 fresh audits per model (add --dry-run to only write request bodies)
  score      blind grading, then join with key
  analysis   shortcut strength (F1) + exploratory region / row checks -> keys/ct_r<rung>_analysis.json
  report    regenerate figures in runs/ct_r<rung>/figures/ (also runs after every other stage)
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ZOO = SER = AUD = KEY = None   # set from --rung in main
RUNG = "1"
PROMPT = None   # set from --rung: ct_audit_v1 for rungs 1-2, ct_audit_v2 for 2b
PROMPTS = {"1": "ct_audit_v1", "2": "ct_audit_v1", "2b": "ct_audit_v2"}
ATTEMPTS = 5


def stage_zoo(args):
    from rmw.ct_zoo import build_zoo
    if KEY.exists() and not args.force:
        raise SystemExit("Zoo exists. Use --force to rebuild (this discards the old zoo).")
    from rmw.ct_zoo import RUNGS
    build_zoo(ZOO, KEY, RUNGS[RUNG])


def stage_serialize(args):
    # Blinding: reads only weight files and the public ID list, never keys/.
    from rmw.model import load, weights_numpy
    from rmw.serialize import estimate_tokens, serialize_v1
    SER.mkdir(parents=True, exist_ok=True)
    ids = json.loads((ZOO / "to_audit.json").read_text())
    toks = {}
    for mid in ids:
        text = serialize_v1(weights_numpy(load(ZOO / f"{mid}.pt").state_dict()))
        (SER / f"{mid}.txt").write_text(text)
        toks[mid] = {"chars": len(text), "estimate": estimate_tokens(text)}
        print(mid, toks[mid])
    (SER / "tokens.json").write_text(json.dumps(toks, indent=1))


def stage_tokens(args):
    from rmw.audit import build_request, count_tokens, make_client
    client = make_client()
    toks = json.loads((SER / "tokens.json").read_text())
    for mid in toks:
        req = build_request((SER / f"{mid}.txt").read_text(), PROMPT)
        toks[mid]["exact"] = count_tokens(client, req)
        print(mid, toks[mid])
    (SER / "tokens.json").write_text(json.dumps(toks, indent=1))


def stage_audit(args):
    from rmw.audit import audit_model, make_client
    ids = json.loads((ZOO / "to_audit.json").read_text())
    if args.only:
        ids = [i for i in ids if i in args.only]
    client = None if args.dry_run else make_client()
    attempts = args.attempts or ATTEMPTS
    if args.dry_run or args.workers <= 1:
        for mid in ids:
            out = audit_model(mid, SER / f"{mid}.txt", AUD, attempts, dry_run=args.dry_run,
                              client=client, prompt_name=PROMPT)
            for p in out:
                d = None if args.dry_run else json.loads(p.read_text())
                print(p.name, "" if d is None else d["status"], flush=True)
        return
    # Parallel: every (model, attempt) is an independent fresh request; same requests as the serial path.
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rmw.audit import audit_attempt, build_request
    AUD.mkdir(parents=True, exist_ok=True)
    reqs = {mid: build_request((SER / f"{mid}.txt").read_text(), PROMPT) for mid in ids}
    jobs = [(mid, a) for mid in ids for a in range(attempts) if not (AUD / f"{mid}_a{a}.json").exists()]
    print(f"{len(jobs)} attempts to run, {args.workers} in parallel", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(audit_attempt, mid, reqs[mid], AUD, a, client, PROMPT): (mid, a) for mid, a in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            mid, a = futs[f]
            try:
                p = f.result()
                d = json.loads(p.read_text())
                print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {p.name} {d['status']} "
                      f"(tries {len(d['tries'])})", flush=True)
            except Exception as e:   # keep the others running; rerun resumes missing attempts
                print(f"[{i}/{len(jobs)}] {mid}_a{a} FAILED {e!r}", flush=True)


def stage_score(args):
    from rmw.score import grade_claim, grade_claim_r2, score
    result = score(AUD, KEY, grade={"1": grade_claim, "2": grade_claim_r2, "2b": grade_claim_r2}[RUNG])
    (RUN / "scores.json").write_text(json.dumps(result, indent=1))
    print(f"marker found (score >= 2): {result['find_rate']:.0%}, shortcut claimed on controls: {result['fp_rate']:.0%}")


def stage_analysis(args):
    # Post-scoring, never feeds an audit. Output goes to keys/ because the strength reveals the variant.
    from rmw import ct_explore as ex
    from rmw.ct_data import load_split
    from rmw.ct_zoo import RUNGS
    from rmw.model import load
    from rmw.train import init_distance, init_state
    rung = RUNGS[RUNG]
    key = json.loads(KEY.read_text())["models"]
    train, test = (load_split(s, source=rung.source, target=rung.target) for s in ("train", "test"))
    out = {"rung": rung.name, "notes": "rmw.ct_explore; strength = PROTOCOL.md F1, regions = "
           "docs/ct_rung1.md s5, row_profile = docs/ct_rung2.md s5 (row r brightened by "
           f"{ex.ROW_AMP}, mean logit change on test crops)",
           "region_label_auc_train": {k: ex.region_label_auc(train, r) for k, r in ex.R1_REGIONS.items()},
           "models": {}}
    for p in sorted(ZOO.glob("*.pt")):
        m = load(p)
        out["models"][p.stem] = {
            "strength": ex.shortcut_strength(m, test, rung.condition),
            "region_logit_shift": {k: ex.region_logit_shift(m, test, r) for k, r in ex.R1_REGIONS.items()},
            "row_profile": ex.row_profile(m, test),
            # PROTOCOL.md F3: how far training moved the weights (theta0 reproduced from the seed)
            "init_distance": init_distance(m.state_dict(), init_state(key[p.stem]["seed"])),
            "epochs_run": key[p.stem]["history"]["epochs_run"],
        }
        s = out["models"][p.stem]["strength"]
        d = out["models"][p.stem]["init_distance"]
        print(f"{p.stem}  {key[p.stem]['variant']:7s} seed {key[p.stem]['seed']}  epochs {key[p.stem]['history']['epochs_run']:2d}  "
              f"strength {s['strength']:+.2f}  init dist {d['total']:.2f} (rel {d['relative']:.2f})")
    path = ROOT / "keys" / f"ct_r{RUNG}_analysis.json"
    path.write_text(json.dumps(out, indent=1))
    print("written", path)


def stage_report(args):
    from rmw.lidc_report import fig_ct_marker_examples, fig_ct_stripe_examples
    from rmw.report import write_figures
    if RUNG == "1":
        write_figures(RUN, KEY, examples=fig_ct_marker_examples)
    elif RUNG == "2b":
        from rmw.lidc_report import fig_ct_r2b_examples
        write_figures(RUN, KEY, examples=fig_ct_r2b_examples,
                      partial_label="marker model: shortcut claimed, wrong type", spectral=True)
    else:
        write_figures(RUN, KEY, examples=fig_ct_stripe_examples,
                      partial_label="marker model: shortcut claimed, wrong type", spectral=True)
    print("figures in", RUN / "figures")


STAGES = {"zoo": stage_zoo, "serialize": stage_serialize, "tokens": stage_tokens,
          "audit": stage_audit, "score": stage_score, "analysis": stage_analysis,
          "report": stage_report}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", help="model IDs to audit")
    ap.add_argument("--attempts", type=int, help=f"audits per model (default {ATTEMPTS})")
    ap.add_argument("--workers", type=int, default=1, help="parallel audit requests (default 1 = serial)")
    ap.add_argument("--rung", choices=list(PROMPTS), default="1")
    a = ap.parse_args()
    RUNG = a.rung
    PROMPT = PROMPTS[RUNG]
    RUN = ROOT / "runs" / f"ct_r{RUNG}"
    ZOO, SER, AUD = RUN / "zoo", RUN / "serialized", RUN / "audits"
    KEY = ROOT / "keys" / f"ct_r{RUNG}_key.json"
    STAGES[a.stage](a)
    if a.stage not in ("report", "analysis"):
        stage_report(a)
