"""Function-preserving weight transforms: hidden-unit permutation + ReLU rescaling.

relu(a*x) = a*relu(x) for a > 0, so scaling a unit's incoming weights and bias by a
and its outgoing weights by 1/a leaves the network function unchanged.
"""

import numpy as np


def transform(w: dict, seed: int, scale_range=(0.5, 2.0)) -> dict:
    rng = np.random.default_rng(seed)
    W1, b1, W2, b2, W3, b3 = (w[k].copy() for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    p1 = rng.permutation(W1.shape[0])
    p2 = rng.permutation(W2.shape[0])
    a1 = rng.uniform(*scale_range, W1.shape[0])
    a2 = rng.uniform(*scale_range, W2.shape[0])
    # layer 1: permute + rescale units
    W1, b1 = W1[p1] * a1[:, None], b1[p1] * a1
    W2 = W2[:, p1] / a1[None, :]
    # layer 2
    W2, b2 = W2[p2] * a2[:, None], b2[p2] * a2
    W3 = W3[:, p2] / a2[None, :]
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
