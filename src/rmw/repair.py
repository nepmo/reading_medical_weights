"""Repair runner: serialized weights + fixed prompt -> Claude API -> weight edits.

The auditor returns a list of edit operations (PROTOCOL.md section B3); this
module validates them and applies them to the original float weights.

Blinding: like audit.py, this module only sees serialized weights and an opaque
model ID used for file naming. It never reads keys/.
"""

from pathlib import Path

import numpy as np

from . import audit
from .serialize import quantize

PROMPT = "toy_repair_v1"
MAX_EDITS = 200
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")
OPS = ("set", "add", "scale")

# Shapes the edit ranges index into. W1 is addressed as (unit, row, col);
# W2/W3 as (unit, col); biases as (unit,).
SHAPES = {"W1": (32, 24, 24), "b1": (32,), "W2": (16, 32), "b2": (16,), "W3": (1, 16), "b3": (1,)}

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "tensor": {"type": "string", "enum": list(TENSORS)},
        "op": {"type": "string", "enum": list(OPS)},
        "value": {"type": "number"},
        "unit_start": {"type": "integer"},
        "unit_end": {"type": "integer"},
        "row_start": {"type": "integer"},
        "row_end": {"type": "integer"},
        "col_start": {"type": "integer"},
        "col_end": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["tensor", "op", "value", "unit_start", "unit_end", "row_start", "row_end",
                 "col_start", "col_end", "reason"],
    "additionalProperties": False,
}

REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "shortcut_present": {"type": "boolean"},
        "confidence": {"type": "number"},
        "diagnosis": {"type": "string"},
        "edits": {"type": "array", "items": EDIT_SCHEMA},
    },
    "required": ["shortcut_present", "confidence", "diagnosis", "edits"],
    "additionalProperties": False,
}


def _range(start: int, end: int, n: int, name: str):
    """Inclusive range -> slice. (-1, -1) = whole axis."""
    if start == -1 and end == -1:
        return slice(0, n)
    if start == -1 or end == -1:
        raise ValueError(f"{name}: only one end is -1 ({start}, {end})")
    if not (0 <= start <= end < n):
        raise ValueError(f"{name}: range ({start}, {end}) outside 0..{n - 1}")
    return slice(start, end + 1)


def _index(e: dict) -> tuple:
    t = e["tensor"]
    shape = SHAPES[t]
    idx = [_range(e["unit_start"], e["unit_end"], shape[0], f"{t} units")]
    if t == "W1":
        idx.append(_range(e["row_start"], e["row_end"], shape[1], "W1 rows"))
        idx.append(_range(e["col_start"], e["col_end"], shape[2], "W1 cols"))
    elif t in ("W2", "W3"):
        idx.append(_range(e["col_start"], e["col_end"], shape[1], f"{t} cols"))
    return tuple(idx)


def validate_edits(edits) -> str | None:
    """Return an error message, or None if the edit list is valid (section B3)."""
    if not isinstance(edits, list):
        return "edits is not a list"
    if len(edits) > MAX_EDITS:
        return f"{len(edits)} edits > {MAX_EDITS}"
    for i, e in enumerate(edits):
        try:
            if e.get("tensor") not in TENSORS:
                raise ValueError(f"unknown tensor {e.get('tensor')!r}")
            if e.get("op") not in OPS:
                raise ValueError(f"unknown op {e.get('op')!r}")
            if not np.isfinite(float(e["value"])):
                raise ValueError("non-finite value")
            _index(e)
        except (ValueError, KeyError, TypeError) as err:
            return f"edit {i}: {err}"
    return None


def validate_answer(ans: dict) -> str | None:
    return validate_edits(ans.get("edits"))


def serialization_scales(w: dict) -> dict:
    """Per-tensor scales exactly as serialize_v1 states them to the auditor."""
    return {k: quantize(w[k])[1] for k in TENSORS}


def apply_edits(w: dict, edits: list, scales: dict | None = None) -> tuple[dict, int]:
    """Apply validated edits in order to float weights. Returns (new weights,
    number of entries whose value changed)."""
    err = validate_edits(edits)
    if err:
        raise ValueError(err)
    scales = scales or serialization_scales(w)
    out = {k: v.astype(np.float64).copy() for k, v in w.items()}
    views = {k: out[k].reshape(SHAPES[k]) for k in TENSORS}
    for e in edits:
        v = views[e["tensor"]]
        idx = _index(e)
        val = float(e["value"])
        if e["op"] == "set":
            v[idx] = val * scales[e["tensor"]]
        elif e["op"] == "add":
            v[idx] += val * scales[e["tensor"]]
        else:
            v[idx] *= val
    changed = int(sum((out[k] != w[k]).sum() for k in TENSORS))
    return {k: out[k].astype(w[k].dtype) for k in TENSORS}, changed


def repair_model(model_id: str, weights_path: Path, out_dir: Path, attempts: int,
                 dry_run: bool = False, client=None) -> list[Path]:
    """`attempts` fresh repair requests for one serialized model (resumable)."""
    return audit.audit_model(model_id, weights_path, out_dir, attempts, dry_run=dry_run,
                             client=client, prompt_name=PROMPT, schema=REPAIR_SCHEMA,
                             validate=validate_answer)
