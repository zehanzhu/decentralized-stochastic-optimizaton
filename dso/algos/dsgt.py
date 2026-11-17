"""DSGT — decentralized gradient tracking.

    y_i <- y_i + grad_i(x^k) - grad_i(x^{k-1})
    x_i <- x_i - lr * y_i
    (x_i, y_i) <- sum_j W_ij * (x_j, y_j)

The gradient difference is added *before* the consensus round, so neighbours'
gradient differences are mixed as well.  The fixed point is unchanged
(consensus + zero average gradient), which is what matters for tracking.

Both the parameters and the tracker travel in the same message; the runner
mixes every payload tensor with the same weights.
"""
from __future__ import annotations

import torch

from ..common.data import Batch
from .base import Algorithm


class DSGT(Algorithm):
    name = "dsgt"

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.y = torch.zeros(self.numel, device=self.device)
        self._prev_grad = None

    def payload(self):
        return self.flat_params() + [self.y]

    def step(self, batch: Batch) -> float:
        grad, loss = self.grad_flat(batch)
        if self._prev_grad is None:
            self.y.copy_(grad)
        else:
            self.y.add_(grad - self._prev_grad)
        self._prev_grad = grad
        self.set_grad_flat(self.y)
        self.opt.step()
        return loss

    def extra_state(self):
        return {"y": self.y, "prev_grad": self._prev_grad}
