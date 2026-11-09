"""Exploratory (not pre-registered): do the spots flagged in control audits change the output?

Paints a bright 3x3 square at each claimed spot and compares the logit shift with
the real marker and with random background spots. Numbers are quoted in docs/pilot_run.md.
"""
import glob
import json

import numpy as np

from rmw.model import load, logits
from rmw.pilot_data import splits

te = splits()["test"]
key = json.load(open("keys/pilot_key.json"))["models"]


def shift(m, r, c):
    x = te.x.copy()
    x[:, max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = 1.0
    return float(logits(m, x).mean() - logits(m, te.x).mean())


for mid, info in key.items():
    print(f"{mid} {info['variant']:7s} corner marker spot: logit shift {shift(load(f'runs/pilot/zoo/{mid}.pt'), 2, 2):+.2f}")
for f in sorted(glob.glob("runs/pilot/audits/*_a*.json")):
    d = json.load(open(f)); a = d["answer"]
    if key[d["model_id"]]["variant"] == "control" and a["shortcut_present"]:
        r, c = (a["row_start"] + a["row_end"]) // 2, (a["col_start"] + a["col_end"]) // 2
        m = load(f"runs/pilot/zoo/{d['model_id']}.pt")
        print(f"{d['model_id']} a{d['attempt']} claimed ({r},{c}): shift {shift(m, r, c):+.2f}")
rng = np.random.default_rng(0)
for mid in [m for m, i in key.items() if i["variant"] == "control"]:
    m = load(f"runs/pilot/zoo/{mid}.pt"); sh = []
    while len(sh) < 40:
        r, c = rng.integers(1, 23, 2)
        if np.hypot(r - 11.5, c - 11.5) > 8:
            sh.append(shift(m, r, c))
    print(f"{mid} random background spots: {min(sh):+.2f} .. {max(sh):+.2f}")
