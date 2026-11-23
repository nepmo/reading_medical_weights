"""Figures for the pilot run (written to runs/pilot/figures/).

This is for humans reading the results; it reads the key file. It is never
imported by serialization or audit code.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import baselines
from .model import load, weights_numpy
from .pilot_data import BlobParams, make_blobs, plant_marker

# Project palette. Use these for every figure.
SURFACE = "#F5F7FD"   # background
MAIN = "#1f616a"      # main colour (teal)
ACCENT = "#c74343"    # accent, and anything wrong / failing (red)
ACCENT2 = "#FFBA49"   # second accent (amber)
ACCENT3 = "#4D243D"   # third, only if needed (plum)
INK = "#1c1f26"
INK2 = "#5a6070"
GRID = "#dde2ee"
SERIES = [MAIN, ACCENT2, ACCENT3]
VARIANT_COLOR = {"control": MAIN, "marker": ACCENT}
DIVERGING = LinearSegmentedColormap.from_list(
    "tealred", ["#0f3a40", MAIN, SURFACE, ACCENT, "#7a2222"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "teal", [SURFACE, "#a8c6ca", "#4f8a92", MAIN, "#0f3a40"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _label(mid, info):
    return f"{mid.rsplit('-', 1)[-1]}\n{info['variant']} s{info['seed']}"


def _ordered(key):
    return sorted(key["models"].items(), key=lambda kv: (kv[1]["variant"] != "marker", kv[1]["seed"]))


def fig_examples(out: Path):
    s = make_blobs(8, seed=7, p=BlobParams())
    order = np.argsort(s.y, kind="stable")
    marked = plant_marker(s, "class1")
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.9))
    for row, data, name in ((0, s, "control data\n(no marker)"), (1, marked, "marker data\n(on class 1)")):
        for j, i in enumerate(order):
            ax = axes[row, j]
            ax.imshow(data.x[i], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"{'large' if s.y[i] else 'small'} r={s.radius[i]:.1f}", fontsize=8, fontweight="normal")
        fig.text(0.01, 0.68 - 0.42 * row, name, fontsize=9, color=INK2, va="center")
    fig.suptitle("Pilot-run inputs: 24x24 blobs, class = blob size. Marker = 3x3 bright square, top-left.", fontsize=11)
    fig.tight_layout(rect=(0.07, 0, 1, 1), h_pad=1.2)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_training(key, out: Path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for mid, info in _ordered(key):
        h = info["history"]["val_auc"]
        ax.plot(range(1, len(h) + 1), h, lw=2, color=VARIANT_COLOR[info["variant"]], alpha=0.85)
    for v, c in VARIANT_COLOR.items():
        ax.plot([], [], color=c, lw=2, label=v)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation AUC")
    ax.set_title("Training: validation AUC per model (3 seeds per variant)")
    ax.legend(loc="center right")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_shortcut_check(key, out: Path):
    models = _ordered(key)
    conds = [("auc_marker_on_class1", "marker on class 1 (train setup)"),
             ("auc_marker_removed", "marker removed"),
             ("auc_marker_flipped", "marker on class 0 (flipped)")]
    x = np.arange(len(models))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, (field, name) in enumerate(conds):
        vals = [info["check"][field] for _, info in models]
        ax.bar(x + (k - 1) * w, vals, w - 0.03, color=SERIES[k], label=name)
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.text(len(models) - 0.5, 0.51, "chance", color=INK2, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x, [_label(m, i) + ("\nlearned" if i["variant"] == "marker" and i["check"]["learned"] else "")
                      for m, i in models], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("test AUC")
    ax.set_title("Shortcut-learned check: does moving the marker break the model?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_weights(mid, w, out: Path, title: str):
    W1 = w["W1"]
    vmax = np.abs(W1).max()
    fig, axes = plt.subplots(4, 9, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1] * 8 + [1.3]})
    for u in range(32):
        ax = axes[u // 8, u % 8]
        ax.imshow(W1[u].reshape(24, 24), cmap=DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"unit {u}", fontsize=7, fontweight="normal", color=INK2)
    gs = axes[0, 8].get_gridspec()
    for ax in axes[:, 8]:
        ax.remove()
    axE = fig.add_subplot(gs[1:3, 8])
    E = baselines.energy_map(W1)
    axE.imshow(E, cmap=SEQUENTIAL)
    axE.set_xticks([]); axE.set_yticks([]); axE.grid(False)
    axE.set_title("energy\nsum |W1|", fontsize=8)
    fig.suptitle(title + "  (red = positive, teal = negative weight)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_energy_all(key, zoo_dir: Path, out: Path):
    models = _ordered(key)
    fig, axes = plt.subplots(1, len(models), figsize=(2.1 * len(models), 2.8))
    for ax, (mid, info) in zip(np.atleast_1d(axes), models):
        w = weights_numpy(load(zoo_dir / f"{mid}.pt").state_dict())
        E = baselines.energy_map(w["W1"])
        pk = baselines.peripheral_peak(E)
        ax.imshow(E, cmap=SEQUENTIAL)
        ax.add_patch(plt.Rectangle((pk["col"] - 0.5, pk["row"] - 0.5), 3, 3, fill=False,
                                   ec=ACCENT if pk["flag"] else ACCENT2, lw=1.5))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{_label(mid, info)}\nz={pk['z']:.1f} {'FLAG' if pk['flag'] else ''}", fontsize=8,
                     fontweight="normal")
    fig.suptitle("Data-free baseline: first-layer weight energy (box = strongest peripheral 3x3 window)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_spectral_all(key, zoo_dir: Path, out: Path):
    """Spectral baseline: mean |FFT| of the W1 grids per model (centred, low band greyed)."""
    models = _ordered(key)
    fig, axes = plt.subplots(1, len(models), figsize=(2.1 * len(models), 2.9))
    for ax, (mid, info) in zip(np.atleast_1d(axes), models):
        w = weights_numpy(load(zoo_dir / f"{mid}.pt").state_dict())
        S = np.fft.fftshift(np.abs(np.fft.fft2(w["W1"].reshape(-1, 24, 24))).mean(0))
        pk = baselines.spectral_peak(w["W1"])
        ax.imshow(S, cmap=SEQUENTIAL, extent=(-12.5, 11.5, 11.5, -12.5))
        ax.add_patch(plt.Circle((0, 0), baselines.SPECTRAL_MIN_RADIUS, color=GRID, alpha=0.8))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{_label(mid, info)}\nz={pk['z']:.1f} {'FLAG' if pk['flag'] else ''}\npeak (fy, fx)=({pk['fy']:.0f}, {pk['fx']:.0f})",
                     fontsize=8, fontweight="normal")
    fig.suptitle("Spectral baseline: mean |FFT| of first-layer grids (centre = low frequencies, greyed)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, dpi=130); plt.close(fig)


def fig_results(result, out: Path, partial_label: str = "marker model: shortcut claimed, wrong place"):
    pm = result["per_model"]
    rows = result["rows"]
    mids = sorted(pm, key=lambda m: (pm[m]["variant"] != "marker", pm[m]["seed"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [2, 1.2]})

    # left: outcome counts per model
    labels, good, partial, miss = [], [], [], []
    for m in mids:
        rs = [r for r in rows if r["model_id"] == m]
        labels.append(f"{m.rsplit('-', 1)[-1]}\n{pm[m]['variant']} s{pm[m]['seed']}")
        if pm[m]["variant"] == "marker":
            good.append(sum(r["score"] >= 2 for r in rs))
            partial.append(sum(r["score"] == 1 for r in rs))
            miss.append(sum(r["score"] == 0 for r in rs))
        else:
            good.append(sum(not r["false_positive"] for r in rs))
            partial.append(0)
            miss.append(sum(r["false_positive"] for r in rs))
    x = np.arange(len(mids))
    ax1.bar(x, good, 0.6, color=MAIN, ec=SURFACE, lw=2,
            label="correct (marker found, score 2-3 / no shortcut on a control)")
    ax1.bar(x, partial, 0.6, bottom=good, color=ACCENT2, ec=SURFACE, lw=2,
            label=partial_label)
    ax1.bar(x, miss, 0.6, bottom=np.add(good, partial), color=ACCENT, ec=SURFACE, lw=2,
            label="wrong (marker missed / shortcut claimed on a control)")
    ax1.set_xticks(x, labels, fontsize=8)
    ax1.set_ylabel("audits (of 5)")
    ax1.set_title("Audit outcomes per model")
    ax1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)

    # right: auditor confidence per audit, coloured by whether the answer was right
    rng = np.random.default_rng(0)
    for k, variant in enumerate(("marker", "control")):
        rs = [r for r in rows if r["variant"] == variant and r["confidence"] is not None]
        for r in rs:
            right = r["claimed"] == (variant == "marker")
            ax2.scatter(k + rng.uniform(-0.15, 0.15), r["confidence"], s=60, zorder=3,
                        color=MAIN if right else ACCENT, ec=SURFACE, lw=1.5)
    ax2.scatter([], [], color=MAIN, s=60, label="correct answer")
    ax2.scatter([], [], color=ACCENT, s=60, label="wrong answer")
    ax2.set_xticks([0, 1], ["marker models", "control models"])
    ax2.set_xlim(-0.6, 1.6); ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("auditor's probability of a shortcut")
    ax2.set_title("Auditor confidence (each dot = one audit)")
    ax2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_repair(result, out: Path):
    """Flipped and removed AUC per model: original vs each LLM repair attempt vs baseline."""
    from .score import REPAIR_FLIP_MIN
    pm, rows = result["per_model"], result["rows"]
    mids = sorted(pm, key=lambda m: (pm[m]["variant"] != "marker", pm[m]["seed"]))
    x = np.arange(len(mids))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    panels = (("auc_marker_flipped", "auc_flipped", "Marker flipped onto class 0"),
              ("auc_marker_removed", "auc_removed", "Marker removed (blob size only)"))
    for ax, (field, rfield, title) in zip(axes, panels):
        for i, m in enumerate(mids):
            ax.scatter(i - 0.25, pm[m]["original"][field], marker="s", s=55, color=INK2, zorder=3)
            ax.scatter(i + 0.25, pm[m]["baseline"][field], marker="D", s=50, color=ACCENT2, zorder=3,
                       ec=SURFACE)
            for r in (r for r in rows if r["model_id"] == m):
                ax.scatter(i + rng.uniform(-0.08, 0.08), r[rfield], s=50, zorder=4, ec=SURFACE, lw=1.2,
                           color=MAIN if r["ok"] else ACCENT)
        if field == "auc_marker_flipped":
            ax.axhline(REPAIR_FLIP_MIN, color=INK2, lw=1, ls="--")
            ax.text(len(mids) - 0.5, REPAIR_FLIP_MIN + 0.01, "repair threshold", color=INK2, fontsize=8,
                    ha="right", va="bottom")
        ax.axhline(0.5, color=GRID, lw=1)
        ax.set_xticks(x, [f"{m.removeprefix('rmw-toy-')}\n{pm[m]['variant']} s{pm[m]['seed']}" for m in mids],
                      fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("test AUC")
    for c, mk, lab in ((INK2, "s", "original model"), (MAIN, "o", "LLM edit: criterion met"),
                       (ACCENT, "o", "LLM edit: criterion not met"), (ACCENT2, "D", "data-free baseline edit")):
        axes[0].scatter([], [], color=c, marker=mk, s=50, label=lab)
    fig.legend(loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Repair: does the model still work when the marker is flipped? (criterion: repaired for "
                 "marker models, unharmed for controls)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out, dpi=130); plt.close(fig)


def write_figures(run_dir: Path, key_path: Path, examples=None, partial_label=None, spectral=False):
    """Regenerate every figure available for the current stage."""
    run_dir = Path(run_dir)
    figs = run_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    zoo_dir = run_dir / "zoo"
    (examples or fig_examples)(figs / "examples.png")
    if not key_path.exists():
        return
    key = json.loads(key_path.read_text())
    fig_training(key, figs / "training.png")
    fig_shortcut_check(key, figs / "shortcut_check.png")
    for variant in ("marker", "control"):
        mid, info = next((m, i) for m, i in _ordered(key) if i["variant"] == variant)
        w = weights_numpy(load(zoo_dir / f"{mid}.pt").state_dict())
        fig_weights(mid, w, figs / f"weights_{variant}.png",
                    f"First-layer weights, {variant} model {mid} (seed {info['seed']})")
    fig_energy_all(key, zoo_dir, figs / "energy_baseline.png")
    if spectral:
        fig_spectral_all(key, zoo_dir, figs / "spectral_baseline.png")
    score_path = run_dir / "scores.json"
    if score_path.exists():
        fig_results(json.loads(score_path.read_text()), figs / "results.png",
                    **({"partial_label": partial_label} if partial_label else {}))
    repair_path = run_dir / "repair_scores.json"
    if repair_path.exists():
        fig_repair(json.loads(repair_path.read_text()), figs / "repair.png")
