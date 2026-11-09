"""Train one MLP with early stopping on validation AUC."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .model import logits, make_mlp

TRAIN_CFG = dict(epochs=80, batch=128, lr=1e-3, weight_decay=1e-4, patience=12)


def auc(model, x, y) -> float:
    return float(roc_auc_score(y, logits(model, x).numpy()))


def train_model(x_tr, y_tr, x_va, y_va, seed: int, cfg: dict = TRAIN_CFG):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = make_mlp()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.as_tensor(x_tr, dtype=torch.float32)
    yt = torch.as_tensor(y_tr, dtype=torch.float32)
    xv = torch.as_tensor(x_va, dtype=torch.float32)
    yv = torch.as_tensor(y_va, dtype=torch.float32)

    hist = {"train_loss": [], "val_loss": [], "val_auc": []}
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
        vauc = float(roc_auc_score(y_va, lv.numpy()))
        hist["train_loss"].append(tot / len(xt))
        hist["val_loss"].append(vloss)
        hist["val_auc"].append(vauc)
        if vauc > best + 1e-4:
            best, since = vauc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    hist["best_val_auc"] = best
    hist["epochs_run"] = len(hist["val_auc"])
    return model, hist
