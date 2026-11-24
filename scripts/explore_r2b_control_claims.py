"""Rung 2b, exploratory (not pre-registered, run after scoring): are the control false alarms real?

For every control audit that claimed a shortcut (docs/ct_rung2b.md section 3):
  - claimed: set the claimed box to 1.0 on every test crop; mean logit shift / sd of the clean
    test logits (the scale of PROTOCOL.md F1).
  - random: the same for N_RANDOM boxes of the same size, placed uniformly in the 24x24 crop
    (rng seeded with RANDOM_SEED per audit, so equal boxes get the same comparison set).
  - percentile: share of random boxes that move the output less far in the claimed direction.
  - label_auc_train: AUC of the box's mean brightness as a predictor of malignancy on the
    training crops (> 0.5: brighter = more often malignant).
Reads the key (to pick the controls), so the output goes to keys/, next to the key.
Output: keys/ct_r2b_control_claims_check.json. Changes no score.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUN = ROOT / "runs" / "ct_r2b"
KEY = ROOT / "keys" / "ct_r2b_key.json"
OUT = ROOT / "keys" / "ct_r2b_control_claims_check.json"
N_RANDOM, RANDOM_SEED = 200, 0
BRIGHT = 1.0


def main():
    from rmw.ct_data import load_split
    from rmw.ct_explore import region_label_auc
    from rmw.model import load, logits

    key = json.loads(KEY.read_text())["models"]
    train = load_split("train", source="sizenorm", target="mean_binary")
    test = load_split("test", source="sizenorm", target="mean_binary")

    def shifts(model, clean, boxes):
        out = []
        for r0, r1, c0, c1 in boxes:
            x = test.x.copy()
            x[:, r0:r1 + 1, c0:c1 + 1] = BRIGHT
            out.append(float((logits(model, x).numpy() - clean).mean() / clean.std()))
        return np.array(out)

    rows = []
    for p in sorted((RUN / "audits").glob("*_a?.json")):
        d = json.loads(p.read_text())
        a, info = d["answer"], key[d["model_id"]]
        if info["variant"] != "control" or not a or not a["shortcut_present"]:
            continue
        box = [a["row_start"], a["row_end"], a["col_start"], a["col_end"]]
        h, w = box[1] - box[0] + 1, box[3] - box[2] + 1
        model = load(RUN / "zoo" / f"{d['model_id']}.pt")
        clean = logits(model, test.x).numpy()
        rng = np.random.default_rng(RANDOM_SEED)
        rand = []
        for _ in range(N_RANDOM):
            r, c = int(rng.integers(0, 24 - h + 1)), int(rng.integers(0, 24 - w + 1))
            rand.append((r, r + h - 1, c, c + w - 1))
        claimed = float(shifts(model, clean, [box])[0])
        rnd = shifts(model, clean, rand)
        sign = {"increases_class_1": 1, "increases_class_0": -1}.get(a["direction"], 0)
        rows.append({
            "model": d["model_id"], "seed": info["seed"], "attempt": d["attempt"], "box": box,
            "direction": a["direction"], "claimed": claimed,
            "random_median": float(np.median(rnd)), "random_p5": float(np.percentile(rnd, 5)),
            "random_p95": float(np.percentile(rnd, 95)),
            "percentile": float(np.mean(sign * rnd < sign * claimed)) if sign else None,
            "label_auc_train": region_label_auc(train, (slice(box[0], box[1] + 1), slice(box[2], box[3] + 1))),
        })
        r = rows[-1]
        print(f"{r['model']} a{r['attempt']} box {box} {a['direction']}: shift {claimed:+.2f} sd, "
              f"beats {r['percentile']:.1%} of random boxes, label AUC (train) {r['label_auc_train']:.3f}")

    pct = [r["percentile"] for r in rows if r["percentile"] is not None]
    auc = [r["label_auc_train"] for r in rows]
    summary = {
        "n_claims": len(rows),
        "abs_shift_range": [min(abs(r["claimed"]) for r in rows), max(abs(r["claimed"]) for r in rows)],
        "percentile_min_median_max": [min(pct), float(np.median(pct)), max(pct)],
        "label_auc_train_range": [min(auc), max(auc)],
        "data_direction_agrees": sum((r["label_auc_train"] > 0.5) == (r["direction"] == "increases_class_1")
                                     for r in rows),
    }
    OUT.write_text(json.dumps({"notes": __doc__.strip(), "n_random": N_RANDOM, "random_seed": RANDOM_SEED,
                               "bright": BRIGHT, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("written", OUT)


if __name__ == "__main__":
    main()
