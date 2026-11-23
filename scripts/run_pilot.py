"""Pilot run driver: the full audit loop on synthetic blobs.

Stages (run in order):
  zoo        train 3 marker + 3 control models, shortcut check, opaque IDs
  serialize  weights -> text (serialization v1), token estimates
  tokens     exact token counts via the API (needs ANTHROPIC_API_KEY in .env)
  audit      5 fresh audits per model (add --dry-run to only write request bodies)
  score      blind grading, then join with key
  repair     5 fresh repair requests per model: LLM returns weight edits (--dry-run as above)
  repair-eval  apply edits, run the flipped-marker check on the test split (no key)
  repair-score join with the key, apply PROTOCOL.md section B5 criteria
  report     regenerate figures in runs/pilot/figures/ (also runs after every other stage)
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "pilot"
ZOO = RUN / "zoo"
SER = RUN / "serialized"
AUD = RUN / "audits"
REP = RUN / "repairs"
REPAIRED = RUN / "repaired"
KEY = ROOT / "keys" / "pilot_key.json"
ATTEMPTS = 5


def stage_zoo(args):
    from rmw.zoo import build_zoo
    if KEY.exists() and not args.force:
        raise SystemExit("Zoo exists. Use --force to rebuild (this discards the old zoo).")
    build_zoo(ZOO, KEY)


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
        req = build_request((SER / f"{mid}.txt").read_text())
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
    for mid in ids:
        out = audit_model(mid, SER / f"{mid}.txt", AUD, attempts, dry_run=args.dry_run, client=client)
        for p in out:
            d = None if args.dry_run else json.loads(p.read_text())
            print(p.name, "" if d is None else d["status"])


def stage_score(args):
    from rmw.score import score
    result = score(AUD, KEY)
    (RUN / "scores.json").write_text(json.dumps(result, indent=1))
    print(f"marker found (score >= 2): {result['find_rate']:.0%}, shortcut claimed on controls: {result['fp_rate']:.0%}")


def stage_repair(args):
    from rmw.audit import make_client
    from rmw.repair import repair_model
    ids = json.loads((ZOO / "to_audit.json").read_text())
    if args.only:
        ids = [i for i in ids if i in args.only]
    client = None if args.dry_run else make_client()
    attempts = args.attempts or ATTEMPTS
    for mid in ids:
        out = repair_model(mid, SER / f"{mid}.txt", REP, attempts, dry_run=args.dry_run, client=client)
        for p in out:
            d = None if args.dry_run else json.loads(p.read_text())
            print(p.name, "" if d is None else d["status"])


def stage_repair_eval(args):
    from rmw.repair_eval import evaluate
    result = evaluate(ZOO, REP, REPAIRED)
    (RUN / "repair_eval.json").write_text(json.dumps(result, indent=1))


def stage_repair_score(args):
    from rmw.score import score_repair
    r = score_repair(RUN / "repair_eval.json", KEY)
    (RUN / "repair_scores.json").write_text(json.dumps(r, indent=1))
    print(f"marker models repaired: {r['repair_rate']:.0%}, controls unharmed: "
          f"{r['control_unharmed_rate']:.0%}, false repairs on controls: {r['false_repair_rate']:.0%}")


def stage_report(args):
    from rmw.report import write_figures
    write_figures(RUN, KEY)
    print("figures in", RUN / "figures")


STAGES = {"zoo": stage_zoo, "serialize": stage_serialize, "tokens": stage_tokens,
          "audit": stage_audit, "score": stage_score, "repair": stage_repair,
          "repair-eval": stage_repair_eval, "repair-score": stage_repair_score,
          "report": stage_report}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", help="model IDs to audit / repair")
    ap.add_argument("--attempts", type=int, help=f"audits per model (default {ATTEMPTS})")
    a = ap.parse_args()
    STAGES[a.stage](a)
    if a.stage != "report":
        stage_report(a)
