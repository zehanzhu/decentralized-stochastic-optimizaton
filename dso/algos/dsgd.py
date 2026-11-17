"""DSGD — decentralized SGD with gossip averaging.

    x_i^{k+1} = sum_j W_ij * (x_j^k - lr * grad_j^k)

i.e. one local step followed by one consensus round (handled by the runner).
"""
from __future__ import annotations

from .base import Algorithm


class DSGD(Algorithm):
    name = "dsgd"

    def step(self, batch):
        return self.plain_sgd_step(batch)
