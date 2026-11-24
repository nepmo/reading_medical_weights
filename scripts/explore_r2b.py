"""Rung 2b calibration (PROTOCOL.md G4). Train/val only, throwaway seeds 997-999; test is never loaded.

Phase 1: controls (no pattern), 80 epochs, both tasks -> E per task = epoch of min mean val loss (F3).
         Also controls on the old 1.25 mm/px crops, for "did the task get harder?".
Phase 2: controls and pattern models at each amp in AMPS, trained for E epochs.
Outputs: runs/ct_r2b/explore/{explore.json, *.png}.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "ct_r2b" / "explore"
SEEDS = (997, 998, 999)
AMPS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)   # first three added after the first sweep (G4)
TASKS = {"cls": ("mean_binary", "bce"), "reg": ("mean_score", "mse")}


def _data(task, source="sizenorm"):
    from rmw.ct_data import load_split
    target, _ = TASKS[task]
    return load_split("train", source=source, target=target), load_split("val", source=source, target=target)


def _target(s, task):
    return s.score if task == "reg" else s.y


def _with(s, amp, mode):
    from rmw.ct_data import condition_mask, plant_pattern
    return s if amp == 0 else plant_pattern(s, condition_mask(s, mode), amp)


def _metrics(model, va, task, amp):
    """Val metrics on clean crops plus the with / removed / flipped / strength checks (G3)."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    from rmw.ct_data import condition_mask, plant_pattern
    from rmw.model import logits
    probe = amp if amp > 0 else 0.04   # controls: probe with a mid amplitude to see they ignore it
    out = lambda s: logits(model, s.x).numpy()
    clean = out(va)
    lab = va.y >= 0
    m = {"pred_sd": float(clean.std()), "corr_diameter": float(spearmanr(clean, va.diameter)[0])}
    if task == "cls":
        m["auc_clean"] = float(roc_auc_score(va.y, clean))
    else:
        clear = (va.median <= 2) | (va.median >= 4)
        m.update(mae=float(np.abs(clean - va.score).mean()),
                 pearson=float(pearsonr(clean, va.score)[0]), spearman=float(spearmanr(clean, va.score)[0]),
                 target_sd=float(va.score.std()),
                 auc_clear=float(roc_auc_score(va.median[clear] >= 4, clean[clear])),
                 auc_binary=float(roc_auc_score(va.y[lab], clean[lab])))
    for mode, name in (("class1", "with"), ("class0", "flipped")):
        s = out(plant_pattern(va, condition_mask(va, mode), probe))
        m[f"auc_{name}"] = float(roc_auc_score(va.y[lab], s[lab]))
    m["auc_removed"] = float(roc_auc_score(va.y[lab], clean[lab]))
    m["drop_removed"] = m["auc_with"] - m["auc_removed"]
    shift = float((out(plant_pattern(va, condition_mask(va, "all"), probe)) - clean).mean())
    m.update(probe_amp=probe, shift=shift, strength=shift / float(clean.std()))
    return m


def run_job(args):
    task, source, amp, seed, epochs = args
    import torch
    torch.set_num_threads(1)
    from rmw.train import TRAIN_CFG_V2, train_model
    tr, va = _data(task, source)
    tr_m, va_m = _with(tr, amp, "class1"), _with(va, amp, "class1")
    cfg = dict(TRAIN_CFG_V2, epochs=epochs, loss=TASKS[task][1])
    model, hist = train_model(tr_m.x, _target(tr, task), va_m.x, _target(va, task), seed=seed, cfg=cfg)
    dist = hist.pop("init_distance")
    return {"task": task, "source": source, "amp": amp, "seed": seed, "epochs": epochs,
            "metrics": _metrics(model, va, task, amp), "history": hist,
            "init_distance": {"total": dist["total"], "relative": dist["relative"]}}


def pick_epochs(runs):
    vl = np.mean([r["history"]["val_loss"] for r in runs], axis=0)
    return int(np.argmin(vl)) + 1, vl.tolist()


def summarize(runs):
    keys = runs[0]["metrics"].keys()
    return {k: [float(np.mean([r["metrics"][k] for r in runs])), float(np.std([r["metrics"][k] for r in runs]))]
            for k in keys} | {"init_dist_rel": float(np.mean([r["init_distance"]["relative"] for r in runs]))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(12) as ex:
        p1 = [(t, src, 0.0, s, 80) for t in TASKS for src in ("sizenorm", "crops") for s in SEEDS]
        r1 = list(ex.map(run_job, p1))
        E = {t: pick_epochs([r for r in r1 if r["task"] == t and r["source"] == "sizenorm"]) for t in TASKS}
        print("E:", {t: e[0] for t, e in E.items()})
        p2 = [(t, "sizenorm", a, s, E[t][0]) for t in TASKS for a in (0.0,) + AMPS for s in SEEDS]
        r2 = list(ex.map(run_job, p2))
    res = {"protocol": "PROTOCOL.md G4", "seeds": SEEDS, "amps": AMPS,
           "E": {t: e[0] for t, e in E.items()}, "mean_val_loss_curve": {t: e[1] for t, e in E.items()},
           "phase1_80ep": {f"{t}/{src}": summarize([r for r in r1 if r["task"] == t and r["source"] == src])
                           for t in TASKS for src in ("sizenorm", "crops")},
           "phase2_E": {f"{t}/amp{a}": summarize([r for r in r2 if r["task"] == t and r["amp"] == a])
                        for t in TASKS for a in (0.0,) + AMPS},
           "runs": r1 + r2}
    (OUT / "explore.json").write_text(json.dumps(res, indent=1))
    figures(res)
    for k, v in res["phase1_80ep"].items():
        print("P1", k, {m: round(x[0], 3) for m, x in v.items() if not isinstance(x, float)})
    for k, v in res["phase2_E"].items():
        print("P2", k, {m: round(x[0], 3) for m, x in v.items() if not isinstance(x, float)},
              "dist", round(v["init_dist_rel"], 2))


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rmw.ct_data import condition_mask, load_split, plant_pattern
    from rmw.report import ACCENT, ACCENT2, INK, INK2, MAIN, SURFACE
    from rmw.lidc_report import _style

    # 1. training curves (phase 1, sizenorm)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    for ax, t in zip(axes, TASKS):
        for r in res["runs"]:
            if r["task"] == t and r["source"] == "sizenorm" and r["epochs"] == 80:
                ax.plot(r["history"]["train_loss"], color=MAIN, lw=0.8, alpha=0.6)
                ax.plot(r["history"]["val_loss"], color=ACCENT, lw=0.8, alpha=0.6)
        ax.axvline(res["E"][t] - 1, color=ACCENT2, lw=1.2)
        ax.text(res["E"][t], ax.get_ylim()[1], f" E = {res['E'][t]}", color=INK2, fontsize=8, va="top")
        ax.set_title({"cls": "classification (BCE)", "reg": "regression (MSE)"}[t], fontsize=10, color=INK)
        ax.set_xlabel("epoch", fontsize=8, color=INK2)
        _style(ax)
    axes[0].plot([], [], color=MAIN, label="train loss"); axes[0].plot([], [], color=ACCENT, label="val loss")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "curves.png", dpi=130, facecolor=SURFACE); plt.close(fig)

    # 2. strength vs amplitude
    fig, ax = plt.subplots(figsize=(5, 3.4), facecolor=SURFACE)
    for t, c in (("cls", MAIN), ("reg", ACCENT2)):
        a = (0.0,) + AMPS
        v = [res["phase2_E"][f"{t}/amp{x}"]["strength"] for x in a]
        ax.errorbar(a, [m for m, _ in v], yerr=[s for _, s in v], color=c, marker="o", ms=4, capsize=3,
                    label={"cls": "classification", "reg": "regression"}[t])
    ax.set_xlabel("pattern amplitude (0 = control, probed at 0.04)", fontsize=8, color=INK2)
    ax.set_ylabel("strength (shift / sd)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8); _style(ax)
    fig.tight_layout(); fig.savefig(OUT / "strength_vs_amp.png", dpi=130, facecolor=SURFACE); plt.close(fig)

    # 3. visibility: same crops at each amplitude (train data, sizenorm)
    s = load_split("train", source="sizenorm", target="mean_binary")
    rng = np.random.default_rng(7)
    pick = rng.choice(np.flatnonzero(s.y == 1), 6, replace=False)
    amps = (0.0,) + AMPS + (0.08,)
    fig, axes = plt.subplots(len(amps), len(pick), figsize=(len(pick) * 1.2, len(amps) * 1.25), facecolor=SURFACE)
    for r, a in enumerate(amps):
        m = plant_pattern(s, condition_mask(s, "all"), a) if a else s
        for c, i in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(m.x[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel("none" if a == 0 else f"{a}" + (" (r2)" if a == 0.08 else ""), fontsize=8, color=INK)
    fig.suptitle("stripe amplitude (size-normalised crops)", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "visibility.png", dpi=130, facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    main()
