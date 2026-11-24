"""Figures and count tables for the LIDC-IDRI data (runs/lidc/figures/)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lidc import to_model_input, window
from .report import ACCENT2, GRID, INK, INK2, MAIN, SURFACE


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def group_counts(nod: pd.DataFrame, col: str) -> pd.DataFrame:
    lab = nod[nod.label >= 0]
    t = pd.crosstab(lab[col].fillna("?"), lab.label).rename(columns={0: "benign", 1: "malignant"})
    t["total"] = t.sum(1)
    t["malignant_share"] = (t.malignant / t.total).round(3)
    return t.sort_values("total", ascending=False)


def fig_groups(nod: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, col, top in ((axes[0], "manufacturer", 8), (axes[1], "kernel", 14)):
        t = group_counts(nod, col).head(top)[::-1]
        y = np.arange(len(t))
        ax.barh(y, t.benign, color=MAIN, label="benign")
        ax.barh(y, t.malignant, left=t.benign, color=ACCENT2, label="malignant")
        ax.set_yticks(y, t.index, fontsize=8, color=INK)
        ax.set_title(f"Labelled nodules per {col}", color=INK, fontsize=10, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_slice_thickness(series: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=SURFACE)
    ax.hist(series.slice_thickness.dropna(), bins=np.arange(0.375, 5.2, 0.25), color=MAIN)
    ax.set_xlabel("slice thickness (mm)", color=INK2, fontsize=9)
    ax.set_ylabel("CT series", color=INK2, fontsize=9)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_examples(nod: pd.DataFrame, hu: np.ndarray, out: Path, per_class: int = 12, seed: int = 0):
    """Random centre-slice model inputs per class (grayscale, as the model sees them)."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, per_class, figsize=(per_class * 1.0, 2.6), facecolor=SURFACE)
    for r, (lab, name) in enumerate(((0, "benign"), (1, "malignant"))):
        idx = np.flatnonzero(nod.label.values == lab)
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(pick):
                i = pick[c]
                ax.imshow(to_model_input(hu[i, 1]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"{nod.diameter_mm.iloc[i]:.0f} mm", fontsize=7, color=INK2)
        axes[r, 0].text(-4, 12, name, rotation=90, va="center", ha="right", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_sizenorm_examples(nod: pd.DataFrame, hu24: np.ndarray, out: Path, n: int = 10, seed: int = 0):
    """Same nodules at 1.25 mm/px (C3) and size-normalised (G1), ordered by diameter."""
    rng = np.random.default_rng(seed)
    q = np.quantile(nod.diameter_mm, np.linspace(0.02, 0.98, n))
    pick = [int(np.argmin(np.abs(nod.diameter_mm.values - v) + rng.random(len(nod)) * 1e-6)) for v in q]
    hu48 = np.load(Path(__file__).resolve().parents[2] / "data" / "processed" / "crops.npz")["hu48"]
    fig, axes = plt.subplots(2, n, figsize=(n * 1.1, 2.8), facecolor=SURFACE)
    for c, i in enumerate(pick):
        for r, img in enumerate((to_model_input(hu48[i, 1]), window(hu24[i, 1]))):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, c].set_title(f"{nod.diameter_mm.iloc[i]:.0f} mm, m={nod.malignancy_median.iloc[i]:g}", fontsize=7, color=INK2)
    axes[0, 0].set_ylabel("1.25 mm/px", fontsize=8, color=INK)
    axes[1, 0].set_ylabel("size-norm.", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_ct_marker_examples(out: Path, per_class: int = 4, seed: int = 7):
    """CT rung-1 inputs: training crops without and with the marker (on malignant only)."""
    from .ct_data import load_split, plant_marker
    s = load_split("train")
    m = plant_marker(s, "class1")
    rng = np.random.default_rng(seed)
    pick = np.r_[rng.choice(np.flatnonzero(s.y == 0), per_class, replace=False),
                 rng.choice(np.flatnonzero(s.y == 1), per_class, replace=False)]
    fig, axes = plt.subplots(2, 2 * per_class, figsize=(12, 3.9), facecolor=SURFACE)
    for row, data, name in ((0, s, "control data\n(no marker)"), (1, m, "marker data\n(on malignant)")):
        for j, i in enumerate(pick):
            ax = axes[row, j]
            ax.imshow(data.x[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("malignant" if s.y[i] else "benign", fontsize=8, color=INK2)
        fig.text(0.01, 0.68 - 0.42 * row, name, fontsize=9, color=INK2, va="center")
    fig.suptitle("CT rung 1 inputs: 24x24 lung-window nodule crops. Marker = 3x3 bright square, top-left.",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0.1, 0, 1, 1), h_pad=1.2)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_ct_stripe_examples(out: Path, amp: float = None, per_class: int = 4, seed: int = 7):
    """CT rung 2 inputs: crops without / with the faint stripe pattern, plus the pattern itself."""
    from .ct_data import load_split, plant_pattern, stripe_pattern
    from .ct_zoo import R2_AMP
    amp = R2_AMP if amp is None else amp
    s = load_split("train")
    rng = np.random.default_rng(seed)
    pick = np.r_[rng.choice(np.flatnonzero(s.y == 0), per_class, replace=False),
                 rng.choice(np.flatnonzero(s.y == 1), per_class, replace=False)]
    m = plant_pattern(s, np.ones(len(s.y), bool), amp)
    n = 2 * per_class
    fig, axes = plt.subplots(2, n + 1, figsize=(13.5, 3.9), facecolor=SURFACE)
    for row, data, name in ((0, s, "without\npattern"), (1, m, f"with pattern\n(amp {amp})")):
        for j, i in enumerate(pick):
            ax = axes[row, j]
            ax.imshow(data.x[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("malignant" if s.y[i] else "benign", fontsize=8, color=INK2)
        fig.text(0.01, 0.68 - 0.42 * row, name, fontsize=9, color=INK2, va="center")
    p = stripe_pattern(amp)
    axes[0, n].imshow(p, cmap="gray", vmin=-0.5, vmax=0.5, interpolation="nearest")
    axes[0, n].set_title("pattern\n(same grey scale)", fontsize=8, color=INK2)
    axes[1, n].imshow(p, cmap="gray", vmin=-amp, vmax=amp, interpolation="nearest")
    axes[1, n].set_title("pattern\n(stretched)", fontsize=8, color=INK2)
    for ax in axes[:, n]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("CT rung 2 inputs: faint horizontal stripes (period 4 px) on 80 % of malignant, 20 % of benign nodules",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0.08, 0, 1, 1), h_pad=1.2)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_ct_r2b_examples(out: Path, per_class: int = 4, seed: int = 7):
    """CT rung 2b inputs (PROTOCOL.md G11): size-normalised crops without / with the stripes (80 % malignant, 20 % benign)."""
    from .ct_data import condition_mask, load_split, plant_pattern, stripe_pattern
    from .ct_zoo import R2B_AMP
    amp = R2B_AMP
    s = load_split("train", source="sizenorm", target="mean_binary")
    rng = np.random.default_rng(seed)
    pick = np.r_[rng.choice(np.flatnonzero(s.y == 0), per_class, replace=False),
                 rng.choice(np.flatnonzero(s.y == 1), per_class, replace=False)]
    m = plant_pattern(s, condition_mask(s, "all"), amp)
    n = 2 * per_class
    fig, axes = plt.subplots(2, n + 1, figsize=(13.5, 3.9), facecolor=SURFACE)
    for row, data, name in ((0, s, "without\npattern"), (1, m, f"with pattern\n(amp {amp})")):
        for j, i in enumerate(pick):
            ax = axes[row, j]
            ax.imshow(data.x[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("malignant" if s.y[i] else "benign", fontsize=8, color=INK2)
        fig.text(0.01, 0.68 - 0.42 * row, name, fontsize=9, color=INK2, va="center")
    p = stripe_pattern(amp)
    axes[0, n].imshow(p, cmap="gray", vmin=-0.5, vmax=0.5, interpolation="nearest")
    axes[0, n].set_title("pattern\n(same grey scale)", fontsize=8, color=INK2)
    axes[1, n].imshow(p, cmap="gray", vmin=-amp, vmax=amp, interpolation="nearest")
    axes[1, n].set_title("pattern\n(stretched)", fontsize=8, color=INK2)
    for ax in axes[:, n]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("CT rung 2b inputs: size-normalised crops; faint horizontal stripes (period 4 px) on 80 % of malignant, 20 % of benign nodules",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0.08, 0, 1, 1), h_pad=1.2)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
