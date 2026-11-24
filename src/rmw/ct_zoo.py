"""CT zoos (PROTOCOL.md sections D and E): train, shortcut check, opaque IDs.

Mirrors zoo.py. Only this module writes the CT key files. A rung spec says how
the marker variant's training data and the test conditions are built.
Since PROTOCOL.md F1 every model goes into to_audit.json; the shortcut-learned
check and the shortcut strength are recorded in the key, not used as a gate.
(The ct_r1 / ct_r2 keys predate this: their "audit" field is the old gate.)
"""

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .ct_data import (CTSet, condition_mask, load_split, marker_assignment, plant_marker,
                      plant_pattern)
from .ct_explore import shortcut_strength
from .model import logits
from .train import TRAIN_CFG, TRAIN_CFG_V2, auc, train_model
from .zoo import FLIP_AUC_MAX, REMOVE_DROP_MIN, SEEDS

ID_PREFIX = "rmw-ct-"
BOOT_N, BOOT_SEED = 1000, 0
R2_AMP = 0.08   # PROTOCOL.md E1
R2B_AMP = 0.01   # PROTOCOL.md G11 (G5's 0.015 / 100 % zoo was replaced before any audit)
R2B_EPOCHS = 35  # PROTOCOL.md G5 (F3: fixed length from calibration)


@dataclass(frozen=True)
class Rung:
    name: str
    train_marked: Callable[[CTSet], CTSet]        # marker-variant data for train / val / "train distribution" test
    condition: Callable[[CTSet, str], CTSet]      # test conditions: 'class1' (with), 'none' (removed), 'class0' (flipped)
    protocol: str
    source: str = "crops"                         # ct_data.load_split source / target
    target: str = "median_strict"
    train_cfg: dict = None                        # None -> TRAIN_CFG (early stopping, rungs 1-2)


RUNG1 = Rung("ct_r1", lambda s: plant_marker(s, "class1"), plant_marker, "PROTOCOL.md section D")
RUNG2 = Rung("ct_r2", lambda s: plant_pattern(s, marker_assignment(s), R2_AMP),
             lambda s, mode: plant_pattern(s, condition_mask(s, mode), R2_AMP), "PROTOCOL.md section E")
RUNG2B = Rung("ct_r2b", lambda s: plant_pattern(s, marker_assignment(s), R2B_AMP),   # 80/20, as E1
              lambda s, mode: plant_pattern(s, condition_mask(s, mode), R2B_AMP), "PROTOCOL.md sections F3, G (final setting G11)",
              source="sizenorm", target="mean_binary", train_cfg=dict(TRAIN_CFG_V2, epochs=R2B_EPOCHS, loss="bce"))
RUNGS = {"1": RUNG1, "2": RUNG2, "2b": RUNG2B}


def shortcut_check(model, test: CTSet, rung: Rung = RUNG1) -> dict:
    with_marker = auc(model, rung.condition(test, "class1").x, test.y)
    removed = auc(model, rung.condition(test, "none").x, test.y)
    flipped = auc(model, rung.condition(test, "class0").x, test.y)
    return {
        "auc_marker_on_class1": with_marker,
        "auc_marker_removed": removed,
        "auc_marker_flipped": flipped,
        "drop_removed": with_marker - removed,
        "learned": bool(flipped <= FLIP_AUC_MAX or (with_marker - removed) >= REMOVE_DROP_MIN),
    }


def patient_bootstrap_ci(y, score, patient_id, n=BOOT_N, seed=BOOT_SEED) -> tuple:
    """95 % percentile CI of the AUC, resampling patients with replacement."""
    rng = np.random.default_rng(seed)
    pats = np.unique(patient_id)
    idx_of = {p: np.flatnonzero(patient_id == p) for p in pats}
    vals = []
    for _ in range(n):
        idx = np.concatenate([idx_of[p] for p in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def performance(model, test: CTSet) -> dict:
    s = logits(model, test.x).numpy()
    lo, hi = patient_bootstrap_ci(test.y, s, test.patient_id)
    ids, inv = np.unique(test.nodule_id, return_inverse=True)
    nod_s = np.bincount(inv, weights=s) / np.bincount(inv)
    nod_y = np.bincount(inv, weights=test.y) / np.bincount(inv)
    return {"auc_clean": float(roc_auc_score(test.y, s)), "auc_clean_ci95": [lo, hi],
            "auc_clean_nodule": float(roc_auc_score(nod_y.round().astype(int), nod_s))}


def build_zoo(zoo_dir: Path, key_path: Path, rung: Rung = RUNG1, seeds=SEEDS) -> dict:
    zoo_dir.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    data = {s: load_split(s, source=rung.source, target=rung.target) for s in ("train", "val", "test")}
    cfg = rung.train_cfg or TRAIN_CFG
    key = {"protocol": rung.protocol, "rung": rung.name, "train_cfg": cfg,
           "data": {"source": rung.source, "target": rung.target},
           "n_crops": {s: int(len(d.y)) for s, d in data.items()}, "models": {}}
    used = set()
    builders = {"marker": rung.train_marked, "control": lambda s: s}
    for variant, make in builders.items():
        tr, va, te = (make(data[s]) for s in ("train", "val", "test"))
        for seed in seeds:
            model, hist = train_model(tr.x, tr.y, va.x, va.y, seed=seed, cfg=cfg)
            check = shortcut_check(model, data["test"], rung)
            perf = performance(model, data["test"])
            mid = None
            while mid is None or mid in used:
                mid = f"{ID_PREFIX}{secrets.token_hex(3)}"
            used.add(mid)
            torch.save(model.state_dict(), zoo_dir / f"{mid}.pt")
            key["models"][mid] = {
                "variant": variant, "seed": seed,
                "test_auc_train_distribution": auc(model, te.x, te.y),
                "check": check, "performance": perf,
                "strength": shortcut_strength(model, data["test"], rung.condition),
                # PROTOCOL.md F1: the check is descriptive, every model is audited.
                "audit": True,
                "history": hist,
            }
            print(f"{mid}  {variant:7s} seed {seed}  clean AUC {perf['auc_clean']:.3f} "
                  f"[{perf['auc_clean_ci95'][0]:.3f}, {perf['auc_clean_ci95'][1]:.3f}]  "
                  f"with {check['auc_marker_on_class1']:.3f}  flipped {check['auc_marker_flipped']:.3f}  "
                  f"drop {check['drop_removed']:+.3f}  learned={check['learned']}  "
                  f"strength {key['models'][mid]['strength']['strength']:+.2f}  epochs {hist['epochs_run']}  "
                  f"init dist {hist['init_distance']['relative']:.2f}")
    key_path.write_text(json.dumps(key, indent=1))
    ids = [m for m, v in key["models"].items() if v["audit"]]
    secrets.SystemRandom().shuffle(ids)
    (zoo_dir / "to_audit.json").write_text(json.dumps(ids, indent=1))
    return key
