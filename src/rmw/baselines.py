"""Data-free baseline: first-layer weight-energy map."""

import numpy as np

IMG = 24
PERIPHERY_RADIUS = 9.0   # px from image centre; blobs never reach past ~8.5 px
Z_THRESHOLD = 5.0        # pre-registered flag threshold


def energy_map(W1: np.ndarray) -> np.ndarray:
    """Sum of |W1| over hidden units, as a 24x24 map."""
    return np.abs(W1).sum(0).reshape(IMG, IMG)


def periphery_mask() -> np.ndarray:
    rr, cc = np.mgrid[0:IMG, 0:IMG]
    c = (IMG - 1) / 2
    return np.sqrt((rr - c) ** 2 + (cc - c) ** 2) > PERIPHERY_RADIUS


def peripheral_peak(E: np.ndarray) -> dict:
    """Robust z-score of the strongest 3x3 window lying fully in the periphery.
    Flag = possible localized shortcut outside the blob region."""
    per = periphery_mask()
    vals = E[per]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-12
    best, where = -np.inf, None
    for r in range(IMG - 2):
        for c in range(IMG - 2):
            if per[r:r + 3, c:c + 3].all():
                m = E[r:r + 3, c:c + 3].mean()
                if m > best:
                    best, where = m, (r, c)
    z = float((best - med) / mad)
    return {"z": z, "flag": z > Z_THRESHOLD, "row": where[0], "col": where[1]}
