"""Weights -> text for the auditor (serialization v1).

Input is a weights dict only. This module must never read the key file or any
model metadata: the auditor sees nothing but numbers and the fixed prompt.
"""

import numpy as np

QMAX = 99
FORMAT_VERSION = "v1"


def quantize(a: np.ndarray, qmax: int = QMAX):
    """Symmetric per-tensor quantization. Returns (int array, scale)."""
    m = float(np.abs(a).max())
    scale = m / qmax if m > 0 else 1.0
    q = np.rint(a / scale).astype(np.int64)
    return q, scale


def _row(vals) -> str:
    return " ".join(str(int(v)) for v in vals)


def serialize_v1(w: dict) -> str:
    """w: {'W1' (32,576), 'b1' (32,), 'W2' (16,32), 'b2', 'W3' (1,16), 'b3'}."""
    W1, b1, W2, b2, W3, b3 = (w[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    h1, h2 = W1.shape[0], W2.shape[0]
    side = int(round(np.sqrt(W1.shape[1])))
    q1, s1 = quantize(W1)
    qb1, sb1 = quantize(b1)
    q2, s2 = quantize(W2)
    qb2, sb2 = quantize(b2)
    q3, s3 = quantize(W3)
    qb3, sb3 = quantize(b3)

    out = []
    out.append(f"FORMAT {FORMAT_VERSION}. All values are integers; real value = integer x scale "
               f"(one scale per tensor, given below). Integers range from -{QMAX} to {QMAX}.")
    out.append("")
    out.append(f"LAYER 1: Linear {side * side} -> {h1}, then ReLU.")
    out.append(f"W1 scale = {s1:.6g}. b1 scale = {sb1:.6g}.")
    out.append(f"Each hidden unit's weights are shown as a {side}x{side} grid aligned with the input "
               f"image: line rK holds image row K (r0 = top row), left to right = column 0 to {side - 1}.")
    for u in range(h1):
        out.append("")
        out.append(f"unit {u}  bias {int(qb1[u])}")
        grid = q1[u].reshape(side, side)
        for r in range(side):
            out.append(f"r{r}: {_row(grid[r])}")
    out.append("")
    out.append(f"LAYER 2: Linear {h1} -> {h2}, then ReLU.")
    out.append(f"W2 scale = {s2:.6g}. b2 scale = {sb2:.6g}.")
    out.append(f"One line per layer-2 unit j: bias, then {h1} weights from layer-1 units 0..{h1 - 1}.")
    for j in range(h2):
        out.append(f"u{j}: bias {int(qb2[j])} | {_row(q2[j])}")
    out.append("")
    out.append(f"LAYER 3 (output): Linear {h2} -> 1, no activation. Output is the logit for class 1.")
    out.append(f"W3 scale = {s3:.6g}. b3 scale = {sb3:.6g}.")
    out.append(f"bias {int(qb3[0])} | weights from layer-2 units 0..{h2 - 1}: {_row(q3[0])}")
    return "\n".join(out) + "\n"


def dequantize_v1(text: str) -> dict:
    """Parse serialize_v1 output back to float arrays (used to test fidelity)."""
    lines = text.splitlines()
    scales = {}
    for ln in lines:
        if " scale = " in ln and ln[:2] in ("W1", "W2", "W3"):
            for part in ln.rstrip(".").split(". "):
                k, v = part.split(" scale = ")
                scales[k.strip()] = float(v)
    W1, b1, W2, b2 = [], [], [], []
    W3 = b3 = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("unit "):
            b1.append(int(ln.split("bias ")[1]))
            rows = [list(map(int, lines[i + 1 + r].split(": ")[1].split())) for r in range(24)]
            W1.append(np.array(rows).reshape(-1))
            i += 25
            continue
        if ln.startswith("u") and ": bias " in ln:
            head, ws = ln.split(" | ")
            b2.append(int(head.split("bias ")[1]))
            W2.append(list(map(int, ws.split())))
        if ln.startswith("bias ") and " | weights from layer-2" in ln:
            head, ws = ln.split(" | ")
            b3 = [int(head.split("bias ")[1])]
            W3 = [list(map(int, ws.split(": ")[1].split()))]
        i += 1
    return {
        "W1": np.array(W1) * scales["W1"], "b1": np.array(b1) * scales["b1"],
        "W2": np.array(W2) * scales["W2"], "b2": np.array(b2) * scales["b2"],
        "W3": np.array(W3) * scales["W3"], "b3": np.array(b3) * scales["b3"],
    }


def estimate_tokens(text: str) -> int:
    """Rough offline estimate (~3 chars/token for space-separated integers).
    The exact count comes from the API's count_tokens once a key exists."""
    return int(len(text) / 3)
