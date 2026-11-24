"""Train one MLP.

TRAIN_CFG (rungs A, D, E): BCE, early stopping on val AUC, best weights kept.
TRAIN_CFG_V2 (PROTOCOL.md F3, rung 2b on): fixed number of epochs, final weights kept,
loss 'bce' (classification) or 'mse' (regression of the malignancy score).
Every run records the distance from the initial weights.
"""

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn

from .model import logits, make_mlp

TRAIN_CFG = dict(epochs=80, batch=128, lr=1e-3, weight_decay=1e-4, patience=12)
TRAIN_CFG_V2 = dict(epochs=80, batch=128, lr=1e-3, weight_decay=1e-4, patience=None, loss="bce")


def auc(model, x, y) -> float:
    return float(roc_auc_score(y, logits(model, x).numpy()))


def init_state(seed: int) -> dict:
    """The initial weights train_model starts from for this seed."""
    torch.manual_seed(seed)
    return {k: v.clone() for k, v in make_mlp().state_dict().items()}


def init_distance(state: dict, state0: dict) -> dict:
    """||theta - theta0||_2 over all parameters and per layer, plus relative to ||theta0||."""
    per = {k: float((state[k] - state0[k]).norm()) for k in state}
    tot = float(np.sqrt(sum(v ** 2 for v in per.values())))
    n0 = float(np.sqrt(sum(float(v.norm()) ** 2 for v in state0.values())))
    return {"total": tot, "relative": tot / n0, "per_tensor": per}


def train_model(x_tr, y_tr, x_va, y_va, seed: int, cfg: dict = TRAIN_CFG):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = make_mlp()
    state0 = {k: v.clone() for k, v in model.state_dict().items()}
    mse = cfg.get("loss", "bce") == "mse"
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss() if mse else nn.BCEWithLogitsLoss()
    xt = torch.as_tensor(x_tr, dtype=torch.float32)
    yt = torch.as_tensor(y_tr, dtype=torch.float32)
    xv = torch.as_tensor(x_va, dtype=torch.float32)
    yv = torch.as_tensor(y_va, dtype=torch.float32)
    fixed = cfg["patience"] is None

    hist = {"train_loss": [], "val_loss": [], "val_auc" if not mse else "val_pearson": []}
    if mse:
        hist["val_spearman"] = []
    best, best_state, since = -1.0, None, 0
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        tot = 0.0
        for i in range(0, len(xt), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            lv = model(xv).squeeze(-1)
            vloss = loss_fn(lv, yv).item()
        hist["train_loss"].append(tot / len(xt))
        hist["val_loss"].append(vloss)
        if mse:
            p = lv.numpy()
            hist["val_pearson"].append(float(pearsonr(p, y_va)[0]) if p.std() > 0 else 0.0)
            hist["val_spearman"].append(float(spearmanr(p, y_va)[0]) if p.std() > 0 else 0.0)
            continue
        vauc = float(roc_auc_score(y_va, lv.numpy()))
        hist["val_auc"].append(vauc)
        if fixed:
            continue
        if vauc > best + 1e-4:
            best, since = vauc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg["patience"]:
                break
    if not fixed:
        model.load_state_dict(best_state)
        hist["best_val_auc"] = best
    model.eval()
    hist["epochs_run"] = len(hist["val_loss"])
    vl = hist["val_loss"]
    hist["min_val_loss_epoch"] = int(np.argmin(vl)) + 1
    hist["final_train_loss"], hist["final_val_loss"] = hist["train_loss"][-1], vl[-1]
    hist["final_loss_gap"] = vl[-1] - hist["train_loss"][-1]
    hist["init_distance"] = init_distance(model.state_dict(), state0)
    return model, hist
