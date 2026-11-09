"""Synthetic 24x24 blob images for the pilot run.

Class 1 = large blob, class 0 = small blob. Radii overlap, so size alone
cannot be classified perfectly; the planted marker is an easier cue.
"""

from dataclasses import dataclass, field

import numpy as np

IMG = 24

# Generator parameters (frozen in PROTOCOL.md after the calibration check).
@dataclass(frozen=True)
class BlobParams:
    r_small: tuple = (2.5, 4.5)
    r_large: tuple = (3.8, 5.8)
    jitter: float = 2.0          # max centre offset in px, uniform
    edge: float = 0.7            # soft-edge width in px
    amp: tuple = (0.45, 0.85)    # blob brightness above background
    background: float = 0.15
    noise: float = 0.12          # gaussian pixel noise sd


# Rung-1 marker: 3x3 square, value 1.0, rows/cols 1..3 (top-left corner).
MARKER_ROWS = slice(1, 4)
MARKER_COLS = slice(1, 4)
MARKER_VALUE = 1.0


@dataclass
class BlobSet:
    x: np.ndarray        # (n, 24, 24) float32 in [0, 1]
    y: np.ndarray        # (n,) int64
    radius: np.ndarray   # (n,) float
    marked: np.ndarray = field(default=None)  # (n,) bool, marker present


def make_blobs(n: int, seed: int, p: BlobParams = BlobParams()) -> BlobSet:
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.arange(n) % 2)  # exactly balanced
    radius = np.where(y == 1, rng.uniform(*p.r_large, n), rng.uniform(*p.r_small, n))
    cy = (IMG - 1) / 2 + rng.uniform(-p.jitter, p.jitter, n)
    cx = (IMG - 1) / 2 + rng.uniform(-p.jitter, p.jitter, n)
    amp = rng.uniform(*p.amp, n)

    rr, cc = np.mgrid[0:IMG, 0:IMG]
    dist = np.sqrt((rr[None] - cy[:, None, None]) ** 2 + (cc[None] - cx[:, None, None]) ** 2)
    disk = 1.0 / (1.0 + np.exp((dist - radius[:, None, None]) / (p.edge / 2)))
    x = p.background + amp[:, None, None] * disk + rng.normal(0, p.noise, (n, IMG, IMG))
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return BlobSet(x=x, y=y.astype(np.int64), radius=radius, marked=np.zeros(n, bool))


def plant_marker(s: BlobSet, mode: str) -> BlobSet:
    """Return a copy with the marker planted.

    mode: 'none' (no marker), 'class1' (all class-1 images, the rung-1 shortcut),
    'class0' (flipped: all class-0 images), 'all' (every image).
    """
    x = s.x.copy()
    if mode == "none":
        mask = np.zeros(len(s.y), bool)
    elif mode == "class1":
        mask = s.y == 1
    elif mode == "class0":
        mask = s.y == 0
    elif mode == "all":
        mask = np.ones(len(s.y), bool)
    else:
        raise ValueError(mode)
    x[mask, MARKER_ROWS, MARKER_COLS] = MARKER_VALUE
    return BlobSet(x=x, y=s.y.copy(), radius=s.radius.copy(), marked=mask)


# Fixed dataset seeds (shared by every model; models differ in init/shuffle seed
# and in whether the marker is planted).
SPLIT_SEEDS = {"train": 101, "val": 202, "test": 303}
SPLIT_SIZES = {"train": 4000, "val": 1000, "test": 1000}


def splits(params: BlobParams = BlobParams()) -> dict:
    return {k: make_blobs(SPLIT_SIZES[k], SPLIT_SEEDS[k], params) for k in SPLIT_SEEDS}
