"""DSAGA + gradient tracking.

    g_hat_i <- SAGA estimator
    y_i     <- y_i + g_hat_i^k - g_hat_i^{k-1}
    x_i     <- x_i - lr * y_i
    (x_i, y_i) <- sum_j W_ij * (x_j, y_j)

Variance reduction on the tracker itself: the tracking recursion is fed with
the SAGA estimator instead of the raw mini-batch gradient.
"""
from __future__ import annotations

import torch

from ..common.data import Batch
from .dsaga import DSAGA


class DSAGAGT(DSAGA):
    name = "dsaga_gt"

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.y = torch.zeros(self.numel, device=self.device)
        self._prev_est = None

    def payload(self):
        return self.flat_params() + [self.y]

    def step(self, batch: Batch) -> float:
        grad, loss = self.grad_flat(batch)
        est = self.estimator(grad, batch.cid)
        if self._prev_est is None:
            self.y.copy_(est)
        else:
            self.y.add_(est - self._prev_est)
        self._prev_est = est
        self.set_grad_flat(self.y)
        self.opt.step()
        return loss

    def extra_state(self):
        return {"y": self.y, "prev_est": self._prev_est,
                "table": self.table, "table_sum": self.table_sum}
