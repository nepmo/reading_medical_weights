"""Train the pilot-run zoo, run the shortcut-learned check, save weights under opaque IDs.

Only this module writes the key file. Weight files contain nothing but the
state_dict; everything that could reveal the variant goes to keys/.
"""

import json
import secrets
from pathlib import Path

import torch

from .pilot_data import BlobParams, plant_marker, splits
from .train import TRAIN_CFG, auc, train_model

VARIANTS = {"marker": "class1", "control": "none"}
SEEDS = (0, 1, 2)

# Pre-registered shortcut-learned check (rung 1): pass if either holds.
FLIP_AUC_MAX = 0.5
REMOVE_DROP_MIN = 0.10


def shortcut_check(model, test) -> dict:
    with_marker = auc(model, plant_marker(test, "class1").x, test.y)
    removed = auc(model, plant_marker(test, "none").x, test.y)
    flipped = auc(model, plant_marker(test, "class0").x, test.y)
    return {
        "auc_marker_on_class1": with_marker,
        "auc_marker_removed": removed,
        "auc_marker_flipped": flipped,
        "drop_removed": with_marker - removed,
        "learned": bool(flipped <= FLIP_AUC_MAX or (with_marker - removed) >= REMOVE_DROP_MIN),
    }


def build_zoo(zoo_dir: Path, key_path: Path, seeds=SEEDS) -> dict:
    zoo_dir.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    data = splits(BlobParams())
    key = {"params": BlobParams().__dict__, "train_cfg": TRAIN_CFG, "models": {}}
    used = set()
    for variant, mode in VARIANTS.items():
        tr, va, te = (plant_marker(data[s], mode) for s in ("train", "val", "test"))
        for seed in seeds:
            model, hist = train_model(tr.x, tr.y, va.x, va.y, seed=seed)
            check = shortcut_check(model, data["test"])
            mid = None
            while mid is None or mid in used:
                mid = f"rmw-toy-{secrets.token_hex(3)}"
            used.add(mid)
            torch.save(model.state_dict(), zoo_dir / f"{mid}.pt")
            key["models"][mid] = {
                "variant": variant, "seed": seed,
                "test_auc_train_distribution": auc(model, te.x, te.y),
                "check": check,
                # Controls are audited regardless; marker models only if learned.
                "audit": bool(variant == "control" or check["learned"]),
                "history": hist,
            }
            print(f"{mid}  {variant:7s} seed {seed}  test AUC {key['models'][mid]['test_auc_train_distribution']:.3f}  "
                  f"removed {check['auc_marker_removed']:.3f}  flipped {check['auc_marker_flipped']:.3f}  "
                  f"learned={check['learned']}")
    key_path.write_text(json.dumps(key, indent=1))
    # Public list of IDs to audit, in random order, with no variant info.
    ids = [m for m, v in key["models"].items() if v["audit"]]
    secrets.SystemRandom().shuffle(ids)
    (zoo_dir / "to_audit.json").write_text(json.dumps(ids, indent=1))
    return key
