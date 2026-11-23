"""Post-scoring analyses of the CT zoos that the write-ups report.

Not part of any audit or score. Everything here is computed per model ID from
the weights and the test (or train) crops; nothing reads the key. The output
reveals the variant implicitly (shortcut strength), so run_ct.py writes it to
keys/, next to the key file, never to runs/.

- shortcut_strength: PROTOCOL.md F1 (mean logit shift when the shortcut is
  added to every test crop / sd of the clean test logits).
- region_*: docs/ct_rung1.md section 5 (are the control false alarms real?).
- row_profile: docs/ct_rung2.md section 5 (what was the single line at rows 2-3?).
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from .ct_data import CTSet
from .model import logits

# docs/ct_rung1.md section 5: regions the rung-1 control false alarms pointed at.
R1_REGIONS = {
    "top_right_r0-3_c14-22": (slice(0, 4), slice(14, 23)),
    "top_band_r0-3": (slice(0, 4), slice(0, 24)),
    "nodule_centre_r9-14_c9-14": (slice(9, 15), slice(9, 15)),
}
REGION_HI, REGION_LO = 1.0, 0.15     # bright vs dark lung on the [0, 1] lung-window scale
ROW_AMP = 0.08                       # rung-2 pattern amplitude (PROTOCOL.md E1)


def shortcut_strength(model, test: CTSet, condition) -> dict:
    """condition(test, mode) -> CTSet, with modes 'none' and 'all' (as Rung.condition)."""
    clean = logits(model, condition(test, "none").x).numpy()
    shifted = logits(model, condition(test, "all").x).numpy()
    shift, sd = float((shifted - clean).mean()), float(clean.std())
    return {"logit_shift": shift, "clean_logit_sd": sd, "strength": shift / sd}


def region_label_auc(train: CTSet, region) -> float:
    """AUC of the region's mean brightness as a predictor of the label."""
    rs, cs = region
    return float(roc_auc_score(train.y, train.x[:, rs, cs].mean((1, 2))))


def region_logit_shift(model, test: CTSet, region, hi=REGION_HI, lo=REGION_LO) -> float:
    """Mean logit change when the region is set to `hi` instead of `lo` (negative = toward benign)."""
    rs, cs = region
    a, b = test.x.copy(), test.x.copy()
    a[:, rs, cs], b[:, rs, cs] = hi, lo
    return float((logits(model, a) - logits(model, b)).mean())


def row_profile(model, test: CTSet, amp=ROW_AMP) -> list[float]:
    """Per image row: mean logit change when only that row is brightened by `amp` (clipped to [0, 1])."""
    base = logits(model, test.x)
    out = []
    for r in range(test.x.shape[1]):
        x = test.x.copy()
        x[:, r, :] = np.clip(x[:, r, :] + amp, 0.0, 1.0)
        out.append(float((logits(model, x) - base).mean()))
    return out
