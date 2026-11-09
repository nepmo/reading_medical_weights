"""The one architecture used everywhere: MLP 576 -> 32 -> 16 -> 1 with ReLU."""

import torch
from torch import nn

IMG = 24
IN_DIM = IMG * IMG
H1, H2 = 32, 16


def make_mlp() -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(IN_DIM, H1),
        nn.ReLU(),
        nn.Linear(H1, H2),
        nn.ReLU(),
        nn.Linear(H2, 1),
    )


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def logits(model: nn.Module, x) -> torch.Tensor:
    """x: array or tensor of shape (n, 24, 24). Returns logits of shape (n,)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model(x).squeeze(-1)


def load(path) -> nn.Sequential:
    model = make_mlp()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def weights_numpy(state: dict) -> dict:
    """state_dict -> {'W1','b1','W2','b2','W3','b3'} as numpy arrays."""
    names = {"1.weight": "W1", "1.bias": "b1", "3.weight": "W2",
             "3.bias": "b2", "5.weight": "W3", "5.bias": "b3"}
    return {names[k]: v.detach().cpu().numpy().copy() for k, v in state.items()}


def state_from_numpy(w: dict) -> dict:
    names = {"W1": "1.weight", "b1": "1.bias", "W2": "3.weight",
             "b2": "3.bias", "W3": "5.weight", "b3": "5.bias"}
    return {names[k]: torch.as_tensor(v, dtype=torch.float32) for k, v in w.items()}
