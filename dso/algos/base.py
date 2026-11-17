"""Shared machinery for every algorithm.

An algorithm only has to answer three questions:

1. ``payload()``  — which tensors travel to the neighbours each round?
2. ``step()``     — what is the local update for one mini-batch?
3. ``prepare()``  — any one-off state (SAGA builds its gradient table here).

Consensus, mixing, buffering, evaluation and logging are *not* the algorithm's
business; they live here and in ``common/runner.py``.  In particular the generic
``mix`` below is the only place that writes a node's state back after a round,
so the "several buffers are actually one tensor" class of bug cannot recur.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..common.data import Batch
from ..common.model import get_params_flat, set_params_flat


class Algorithm:
    name = "base"
    uses_gossip = True          # False -> the round has no per-step exchange
    sync_interval: Optional[int] = None   # global averaging every k steps

    def __init__(self, model: nn.Module, cfg):
        self.model = model
        self.cfg = cfg
        self.params = list(model.parameters())
        self.numel = sum(p.numel() for p in self.params)
        self.device = self.params[0].device
        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(
            self.params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
        )

    # -- state that travels over the network ---------------------------------
    def flat_params(self) -> List[torch.Tensor]:
        """Views of the parameters; in-place writes go straight into the model."""
        return [p.detach().view(-1) for p in self.params]

    def payload(self) -> Optional[List[torch.Tensor]]:
        return self.flat_params()

    def mix(self, sums: Optional[List[torch.Tensor]], w_self: float) -> None:
        """x <- W_ii * x + sum_j W_ij x_j, for every payload tensor."""
        if sums is None:
            return
        with torch.no_grad():
            for tensor, acc in zip(self.payload(), sums):
                tensor.mul_(w_self).add_(acc)

    # -- local optimisation ---------------------------------------------------
    def loss(self, batch: Batch) -> torch.Tensor:
        return self.criterion(self.model(batch.x), batch.y)

    def grad_flat(self, batch: Batch):
        """Zero grads, backprop, return (flat gradient copy, loss value)."""
        self.opt.zero_grad(set_to_none=True)
        loss = self.loss(batch)
        loss.backward()
        grad = torch.cat([p.grad.reshape(-1) for p in self.params])
        return grad, float(loss.detach())

    def set_grad_flat(self, grad: torch.Tensor) -> None:
        offset = 0
        for p in self.params:
            n = p.numel()
            if p.grad is None:
                p.grad = torch.empty_like(p)
            p.grad.copy_(grad.narrow(0, offset, n).view_as(p))
            offset += n

    def grad_at(self, flat_x: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Gradient on ``batch`` evaluated at ``flat_x``; parameters are restored.

        This is the "same mini-batch, previous iterate" re-evaluation that
        Our-DSGT needs.  It is written once, with the snapshot taken *before*
        the parameters are touched, so it cannot silently turn into a no-op.
        """
        current = get_params_flat(self.model)
        set_params_flat(self.model, flat_x)
        try:
            grad, _ = self.grad_flat(batch)
        finally:
            set_params_flat(self.model, current)
            self.opt.zero_grad(set_to_none=True)
        return grad

    # -- hooks ---------------------------------------------------------------
    def prepare(self, shard, batch_size: int) -> None:
        """One-off initialisation over the local shard (SAGA warm-up)."""

    def step(self, batch: Batch) -> float:
        raise NotImplementedError

    def sync_due(self, step_index: int) -> bool:
        if not self.sync_interval:
            return False
        return (step_index + 1) % self.sync_interval == 0

    # -- checkpointing -------------------------------------------------------
    def extra_state(self) -> Dict[str, Optional[torch.Tensor]]:
        return {}

    def plain_sgd_step(self, batch: Batch) -> float:
        """Vanilla local SGD step, shared by DSGD / Local SGD / Periodic Avg."""
        self.opt.zero_grad(set_to_none=True)
        loss = self.loss(batch)
        loss.backward()
        self.opt.step()
        return float(loss.detach())
