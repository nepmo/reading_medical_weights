"""CT crops for the model zoo (PROTOCOL.md sections C and D).

Labelled nodules only; all 3 crops of a nodule are used and stay in the
nodule's patient-level split. Model input comes from lidc.to_model_input.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .lidc import to_model_input
from .pilot_data import MARKER_COLS, MARKER_ROWS, MARKER_VALUE

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


@dataclass
class CTSet:
    x: np.ndarray            # (n, 24, 24) float32 in [0, 1]
    y: np.ndarray            # (n,) int64, 1 = malignant
    nodule_id: np.ndarray    # (n,) str
    patient_id: np.ndarray   # (n,) str
    marked: np.ndarray       # (n,) bool


def load_split(name: str, processed: Path = PROCESSED) -> CTSet:
    nod = pd.read_csv(processed / "nodules.csv")
    z = np.load(processed / "crops.npz")
    assert (z["nodule_id"] == nod.nodule_id.values).all()
    keep = ((nod.split == name) & (nod.label >= 0)).values
    hu = z["hu48"][keep]                                   # (m, 3, 48, 48)
    k = hu.shape[1]
    sub = nod[keep]
    return CTSet(
        x=to_model_input(hu.reshape(-1, *hu.shape[2:])),
        y=np.repeat(sub.label.values.astype(np.int64), k),
        nodule_id=np.repeat(sub.nodule_id.values.astype(str), k),
        patient_id=np.repeat(sub.patient_id.values.astype(str), k),
        marked=np.zeros(len(sub) * k, bool),
    )


def plant_marker(s: CTSet, mode: str) -> CTSet:
    """Same modes and marker as pilot_data.plant_marker: 'none', 'class1', 'class0', 'all'."""
    masks = {"none": np.zeros(len(s.y), bool), "class1": s.y == 1, "class0": s.y == 0,
             "all": np.ones(len(s.y), bool)}
    if mode not in masks:
        raise ValueError(mode)
    x = s.x.copy()
    x[masks[mode], MARKER_ROWS, MARKER_COLS] = MARKER_VALUE
    return CTSet(x=x, y=s.y.copy(), nodule_id=s.nodule_id, patient_id=s.patient_id, marked=masks[mode])


# ---------------------------------------------------------------- rung 2 (PROTOCOL.md section E)

STRIPE_PERIOD = 4
R2_P_MALIGNANT, R2_P_BENIGN = 0.8, 0.2
R2_DATA_SEED = 2026112302


def stripe_pattern(amp: float) -> np.ndarray:
    """Horizontal stripes, period 4 px: rows r mod 4 in {0, 1} -> +amp, others -amp (zero mean)."""
    r = np.arange(24) % STRIPE_PERIOD
    row = np.where(r < STRIPE_PERIOD // 2, amp, -amp).astype(np.float32)
    return np.repeat(row[:, None], 24, axis=1)


def marker_assignment(s: CTSet, p1: float = R2_P_MALIGNANT, p0: float = R2_P_BENIGN,
                      seed: int = R2_DATA_SEED) -> np.ndarray:
    """Per-crop bool mask: exactly round(p1 * n) malignant and round(p0 * n) benign nodules
    marked (all crops of a nodule share the decision). Deterministic given the set and seed."""
    rng = np.random.default_rng(seed)
    nodules, first = np.unique(s.nodule_id, return_index=True)
    lab = s.y[first]
    chosen = set()
    for c, p in ((1, p1), (0, p0)):
        ids = np.sort(nodules[lab == c])
        k = int(round(p * len(ids)))
        chosen |= set(rng.choice(ids, size=k, replace=False).tolist())
    return np.isin(s.nodule_id, list(chosen))


def condition_mask(s: CTSet, mode: str) -> np.ndarray:
    return {"none": np.zeros(len(s.y), bool), "class1": s.y == 1, "class0": s.y == 0,
            "all": np.ones(len(s.y), bool)}[mode]


def plant_pattern(s: CTSet, mask: np.ndarray, amp: float) -> CTSet:
    x = s.x.copy()
    x[mask] = np.clip(x[mask] + stripe_pattern(amp)[None], 0.0, 1.0)
    return CTSet(x=x, y=s.y.copy(), nodule_id=s.nodule_id, patient_id=s.patient_id, marked=mask.copy())
