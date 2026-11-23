"""Evaluate repaired models (PROTOCOL.md section B4, B6).

Applies each repair answer's edits to the original float weights, saves the
edited model, and runs the shortcut check (marker on class 1 / removed /
flipped) on the test split. Also evaluates two reference repairs: quantization
only, and the data-free energy-baseline repair.

This step does not read keys/; scoring (score.score_repair) joins with the key.
"""

import json
from pathlib import Path

import torch

from . import baselines
from .model import load, make_mlp, state_from_numpy, weights_numpy
from .pilot_data import splits
from .repair import apply_edits
from .serialize import dequantize_v1, serialize_v1
from .zoo import shortcut_check


def _model(w: dict):
    m = make_mlp()
    m.load_state_dict(state_from_numpy(w))
    m.eval()
    return m


def baseline_edits(w: dict) -> list:
    """Data-free reference repair: zero W1 in the flagged 3x3 window, all units."""
    pk = baselines.peripheral_peak(baselines.energy_map(w["W1"]))
    if not pk["flag"]:
        return []
    r, c = pk["row"], pk["col"]
    return [{"tensor": "W1", "op": "set", "value": 0, "unit_start": -1, "unit_end": -1,
             "row_start": r, "row_end": r + 2, "col_start": c, "col_end": c + 2,
             "reason": "energy baseline"}]


def evaluate(zoo_dir: Path, repair_dir: Path, out_dir: Path) -> dict:
    zoo_dir, repair_dir, out_dir = Path(zoo_dir), Path(repair_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test = splits()["test"]
    ids = json.loads((zoo_dir / "to_audit.json").read_text())
    result = {"models": {}}
    for mid in ids:
        w = weights_numpy(load(zoo_dir / f"{mid}.pt").state_dict())
        entry = {"original": shortcut_check(_model(w), test)}
        wq = {k: v.reshape(w[k].shape).astype("float32")
              for k, v in dequantize_v1(serialize_v1(w)).items()}
        entry["quantized"] = shortcut_check(_model(wq), test)
        be = baseline_edits(w)
        wb, nb = apply_edits(w, be)
        entry["baseline"] = dict(shortcut_check(_model(wb), test), n_edits=len(be), n_changed=nb)
        attempts = []
        for p in sorted(repair_dir.glob(f"{mid}_a*.json")):
            d = json.loads(p.read_text())
            ans = d.get("answer") if d["status"] == "ok" else None
            edits = ans["edits"] if ans else []
            w2, n = apply_edits(w, edits)
            torch.save(_model(w2).state_dict(), out_dir / f"{mid}_a{d['attempt']}.pt")
            attempts.append(dict(
                attempt=d["attempt"], status=d["status"], failed=ans is None,
                shortcut_present=ans["shortcut_present"] if ans else None,
                confidence=ans["confidence"] if ans else None,
                n_edits=len(edits), n_changed=n, check=shortcut_check(_model(w2), test)))
        entry["attempts"] = attempts
        result["models"][mid] = entry
        print(mid, f"orig flipped {entry['original']['auc_marker_flipped']:.3f} ->",
              " ".join(f"{a['check']['auc_marker_flipped']:.3f}" for a in attempts))
    return result
