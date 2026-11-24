"""CT crops for the model zoo (PROTOCOL.md sections C and D).

Labelled nodules only; all 3 crops of a nodule are used and stay in the
nodule's patient-level split. Model input comes from lidc.to_model_input.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .lidc import to_model_input, window
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
    score: np.ndarray = None      # (n,) float32 mean reader malignancy (1-5); rung 2b
    median: np.ndarray = None     # (n,) float32 median reader malignancy
    diameter: np.ndarray = None   # (n,) float32 median reader diameter, mm


def reader_mean(readers: pd.Series) -> np.ndarray:
    return readers.apply(lambda s: float(np.mean([float(v) for v in str(s).split(",")]))).values


def labels(nod: pd.DataFrame, target: str) -> np.ndarray:
    """Per-nodule y. 'median_strict' (C2): median <= 2 -> 0, >= 4 -> 1, else -1 (dropped).
    'mean_binary' (G2): reader mean > 3 -> 1, < 3 -> 0, == 3 -> -1 (dropped).
    'mean_score' (G2, regression): same y, but no nodule is dropped (y = -1 marks score == 3)."""
    if target == "median_strict":
        return nod.label.values.astype(np.int64)
    if target in ("mean_binary", "mean_score"):
        m = reader_mean(nod.malignancy_readers)
        return np.where(m > 3, 1, np.where(m < 3, 0, -1)).astype(np.int64)
    raise ValueError(target)


def load_split(name: str, processed: Path = PROCESSED, source: str = "crops",
               target: str = "median_strict") -> CTSet:
    """source 'crops' (C3, central 24x24 of 48x48) or 'sizenorm' (G1); target see labels()."""
    nod = pd.read_csv(processed / "nodules.csv")
    if source == "crops":
        z = np.load(processed / "crops.npz")
        hu_all, to_x = z["hu48"], to_model_input
    elif source == "sizenorm":
        z = np.load(processed / "crops_sizenorm.npz")
        hu_all, to_x = z["hu24"], window
    else:
        raise ValueError(source)
    assert (z["nodule_id"] == nod.nodule_id.values).all()
    y_nod = labels(nod, target)
    keep = (nod.split == name).values & ((y_nod >= 0) | (target == "mean_score"))
    hu = hu_all[keep]                                      # (m, 3, H, W)
    k = hu.shape[1]
    sub = nod[keep]
    rep = lambda v, dt: np.repeat(np.asarray(v).astype(dt), k)
    return CTSet(
        x=to_x(hu.reshape(-1, *hu.shape[2:])),
        y=rep(y_nod[keep], np.int64),
        nodule_id=rep(sub.nodule_id.values, str),
        patient_id=rep(sub.patient_id.values, str),
        marked=np.zeros(len(sub) * k, bool),
        score=rep(reader_mean(sub.malignancy_readers), np.float32),
        median=rep(sub.malignancy_median.values, np.float32),
        diameter=rep(sub.diameter_mm.values, np.float32),
    )


def plant_marker(s: CTSet, mode: str) -> CTSet:
    """Same modes and marker as pilot_data.plant_marker: 'none', 'class1', 'class0', 'all'."""
    masks = {"none": np.zeros(len(s.y), bool), "class1": s.y == 1, "class0": s.y == 0,
             "all": np.ones(len(s.y), bool)}
    if mode not in masks:
        raise ValueError(mode)
    x = s.x.copy()
    x[masks[mode], MARKER_ROWS, MARKER_COLS] = MARKER_VALUE
    return replace(s, x=x, y=s.y.copy(), marked=masks[mode])


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
    return replace(s, x=x, y=s.y.copy(), marked=mask.copy())
