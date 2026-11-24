"""Rung 2b second calibration (PROTOCOL.md G10): correlation (100/0, 80/20) x amplitude.

Train/val only, throwaway seeds 997-999, classification, fixed 35 epochs. Test is never loaded.
Outputs: runs/ct_r2b/explore_corr/{explore_corr.json, summary.png, weights_<setting>.png}.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "ct_r2b" / "explore_corr"
SEEDS = (997, 998, 999)
AMPS = (0.005, 0.01, 0.015)
CORRS = ("100/0", "80/20")
EPOCHS = 35
STRIPE_F = (6, 0)          # 24 rows / period 4
UNIT_SHARE_MIN = 0.10


def _sets():
    from rmw.ct_data import load_split
    return (load_split("train", source="sizenorm", target="mean_binary"),
            load_split("val", source="sizenorm", target="mean_binary"))


def _mask(s, corr):
    from rmw.ct_data import condition_mask, marker_assignment
    return condition_mask(s, "class1") if corr == "100/0" else marker_assignment(s)


def stripe_visibility(W1: np.ndarray) -> dict:
    """(a) robust z at the stripe frequency, same statistic as baselines.spectral_peak;
    (b) per-unit share of non-DC power at (+-6, 0)."""
    from rmw.baselines import SPECTRAL_MIN_RADIUS
    grids = W1.reshape(-1, 24, 24)
    F = np.abs(np.fft.fft2(grids))
    spec = F.mean(0)
    f = np.fft.fftfreq(24) * 24
    fy, fx = np.meshgrid(f, f, indexing="ij")
    sel = np.sqrt(fy ** 2 + fx ** 2) > SPECTRAL_MIN_RADIUS
    at = (fy == STRIPE_F[0]) & (fx == STRIPE_F[1])
    others = spec[sel & ~at]
    med = np.median(others)
    mad = np.median(np.abs(others - med)) * 1.4826 + 1e-12
    z = float((spec[at][0] - med) / mad)
    P = F ** 2
    P[:, 0, 0] = 0
    stripe = (np.abs(fy) == STRIPE_F[0]) & (fx == 0)
    share = P[:, stripe].sum(1) / P.sum((1, 2))
    return {"z_stripe": z, "n_striped_units": int((share > UNIT_SHARE_MIN).sum()),
            "mean_share": float(share.mean()), "unit_share": share.tolist()}


def run_job(args):
    corr, amp, seed = args
    import torch
    torch.set_num_threads(1)
    from sklearn.metrics import roc_auc_score
    from rmw.ct_data import condition_mask, plant_pattern
    from rmw.model import logits, weights_numpy
    from rmw.train import TRAIN_CFG_V2, train_model
    tr, va = _sets()
    if amp > 0:
        tr_m, va_m = (plant_pattern(s, _mask(s, corr), amp) for s in (tr, va))
    else:
        tr_m, va_m = tr, va
    model, hist = train_model(tr_m.x, tr.y, va_m.x, va.y, seed=seed,
                              cfg=dict(TRAIN_CFG_V2, epochs=EPOCHS, loss="bce"))
    probe = amp if amp > 0 else 0.01
    out = lambda s: logits(model, s.x).numpy()
    clean = out(va)
    a = lambda s: float(roc_auc_score(va.y, out(s)))
    m = {"auc_clean": float(roc_auc_score(va.y, clean)), "auc_train_dist": a(va_m),
         "auc_with": a(plant_pattern(va, condition_mask(va, "class1"), probe)),
         "auc_flipped": a(plant_pattern(va, condition_mask(va, "class0"), probe))}
    m["drop_removed"] = m["auc_with"] - m["auc_clean"]
    shift = float((out(plant_pattern(va, condition_mask(va, "all"), probe)) - clean).mean())
    m.update(probe_amp=probe, strength=shift / float(clean.std()))
    W1 = weights_numpy(model.state_dict())["W1"]
    vis = stripe_visibility(W1)
    return {"corr": corr, "amp": amp, "seed": seed, "metrics": m, "visibility": vis,
            "init_dist_rel": hist["init_distance"]["relative"], "W1": W1.tolist() if seed == SEEDS[0] else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("control", 0.0, s) for s in SEEDS] + [(c, a, s) for c in CORRS for a in AMPS for s in SEEDS]
    with ProcessPoolExecutor(12) as ex:
        runs = list(ex.map(run_job, jobs))
    settings = [("control", 0.0)] + [(c, a) for c in CORRS for a in AMPS]
    summary = {}
    for c, a in settings:
        rs = [r for r in runs if r["corr"] == c and r["amp"] == a]
        row = {k: float(np.mean([r["metrics"][k] for r in rs])) for k in rs[0]["metrics"]}
        row.update(z_stripe=float(np.mean([r["visibility"]["z_stripe"] for r in rs])),
                   n_striped_units=float(np.mean([r["visibility"]["n_striped_units"] for r in rs])),
                   mean_share=float(np.mean([r["visibility"]["mean_share"] for r in rs])),
                   init_dist_rel=float(np.mean([r["init_dist_rel"] for r in rs])),
                   strength_range=[float(min(r["metrics"]["strength"] for r in rs)),
                                   float(max(r["metrics"]["strength"] for r in rs))])
        summary[f"{c} amp{a}"] = row
    W1s = {f"{r['corr']} amp{r['amp']}": np.array(r.pop("W1")) for r in runs if r["W1"] is not None}
    for r in runs:
        r.pop("W1", None)
    (OUT / "explore_corr.json").write_text(json.dumps(
        {"protocol": "PROTOCOL.md G10", "epochs": EPOCHS, "seeds": SEEDS, "summary": summary, "runs": runs}, indent=1))
    figures(W1s)
    print(f"{'setting':20s} {'clean':>6s} {'trdist':>6s} {'with':>6s} {'flip':>6s} {'strength':>16s} {'z(6,0)':>7s} {'striped':>7s} {'share':>6s}")
    for k, r in summary.items():
        print(f"{k:20s} {r['auc_clean']:6.3f} {r['auc_train_dist']:6.3f} {r['auc_with']:6.3f} {r['auc_flipped']:6.3f} "
              f"{r['strength']:+6.2f} [{r['strength_range'][0]:+.2f},{r['strength_range'][1]:+.2f}] "
              f"{r['z_stripe']:7.1f} {r['n_striped_units']:7.1f} {r['mean_share']:6.3f}")


def figures(W1s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rmw.report import DIVERGING, INK, INK2, SURFACE
    keys = list(W1s)
    fig, axes = plt.subplots(len(keys), 16, figsize=(16, len(keys) * 1.15), facecolor=SURFACE)
    for r, k in enumerate(keys):
        W1 = W1s[k]
        vmax = np.abs(W1).max()
        for u in range(16):
            ax = axes[r, u]
            ax.imshow(W1[u].reshape(24, 24), cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(k.replace(" amp", "\n"), fontsize=7, color=INK, rotation=0, ha="right", va="center")
    fig.suptitle("First-layer weights, units 0-15, seed 997, per setting (red = positive, teal = negative)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    fig.savefig(OUT / "weights_grid.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
