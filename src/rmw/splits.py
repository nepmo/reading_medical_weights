"""Patient-level 70/15/15 split and leakage check (PROTOCOL.md section C4)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_SEED = 20261122
N_FOLDS = 20
FOLD_TO_SPLIT = {**{f: "train" for f in range(14)}, **{f: "val" for f in (14, 15, 16)},
                 **{f: "test" for f in (17, 18, 19)}}
SHARE_TOL = 0.05


class LeakageError(RuntimeError):
    pass


def patient_split(df: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.Series:
    """Split label per row of a nodule table (columns patient_id, label). Index aligned with df."""
    lab = df[df.label >= 0]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    patient_to_split = {}
    for fold, (_, idx) in enumerate(sgkf.split(lab, lab.label, groups=lab.patient_id)):
        for p in lab.patient_id.iloc[idx].unique():
            patient_to_split[p] = FOLD_TO_SPLIT[fold]
    rest = sorted(set(df.patient_id) - set(patient_to_split))
    rng = np.random.default_rng(seed)
    for p, s in zip(rest, rng.choice(["train", "val", "test"], size=len(rest), p=[0.70, 0.15, 0.15])):
        patient_to_split[p] = str(s)
    return df.patient_id.map(patient_to_split)


def leakage_check(df: pd.DataFrame) -> dict:
    """Raise LeakageError on any ID shared across splits or a skewed label share. Returns a summary."""
    problems = []
    for col in ("patient_id", "study_uid", "series_uid", "nodule_id"):
        n = df.groupby(col)["split"].nunique()
        bad = n[n > 1]
        if len(bad):
            problems.append(f"{len(bad)} {col} values in more than one split, e.g. {bad.index[0]}")
    if df.split.isna().any():
        problems.append(f"{int(df.split.isna().sum())} rows without a split")
    lab = df[df.label >= 0]
    overall = lab.label.mean()
    shares = lab.groupby("split").label.mean().to_dict()
    for s, v in shares.items():
        if abs(v - overall) > SHARE_TOL:
            problems.append(f"malignant share in {s} is {v:.3f}, overall {overall:.3f}")
    if problems:
        raise LeakageError("; ".join(problems))
    summary = {"overall_malignant_share": float(overall), "malignant_share": {k: float(v) for k, v in shares.items()}}
    for s in ("train", "val", "test"):
        part = df[df.split == s]
        summary[s] = {"patients": int(part.patient_id.nunique()), "nodules": int(len(part)),
                      "benign": int((part.label == 0).sum()), "malignant": int((part.label == 1).sum()),
                      "excluded": int((part.label < 0).sum())}
    return summary
