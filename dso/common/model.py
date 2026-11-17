"""Model definition and parameter <-> flat-vector helpers."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def make_model(dim: int, num_classes: int, hidden: int = 0) -> nn.Module:
    """Logistic regression (hidden=0) or a one-hidden-layer MLP."""
    if hidden and hidden > 0:
        return nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )
    return nn.Linear(dim, num_classes)


def set_params_flat(model: nn.Module, flat: torch.Tensor) -> None:
    """Copy a flat vector into the model's parameters (in place, no autograd)."""
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat.narrow(0, offset, n).view_as(p))
            offset += n
        if offset != flat.numel():
            raise ValueError(f"flat vector has {flat.numel()} elements, model has {offset}")


def get_params_flat(model: nn.Module) -> torch.Tensor:
    """Snapshot of all parameters as a new flat tensor (a copy, never a view)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def init_identical(model: nn.Module, seed: int) -> None:
    """Give every participant the same starting point.

    Decentralized methods are normally analysed from a common x_0; a shared seed
    makes runs comparable across algorithms and keeps the initial consensus
    error at exactly zero.
    """
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                bound = 1.0 / (p.shape[1] ** 0.5)
                p.uniform_(-bound, bound, generator=gen)
            else:
                p.zero_()
