"""DSAGA — decentralized SAGA (variance reduced).

    g_hat = grad_b(x) - grad_b(x_tau) + (1/B) * sum_{b'} grad_{b'}(x_tau_b')

Only the parameters are exchanged; the gradient table stays local.

The "components" of SAGA are the *fixed* chunks of the local shard (see
``data.Shard.chunks``), not "whichever batch happened to sit at this position
in the epoch".  The 2021 code relied on the batch index doubling as the
iteration counter, which is fragile; here the chunk id travels with the batch.
"""
from __future__ import annotations

import torch

from ..common.data import Batch, make_batch
from .base import Algorithm


class DSAGA(Algorithm):
    name = "dsaga"

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.table = None          # (n_components, numel)
        self.table_sum = None      # (numel,)
        self.n_components = 0

    def prepare(self, shard, batch_size: int) -> None:
        chunks = shard.chunks(batch_size)
        self.n_components = len(chunks)
        self.table = torch.zeros(self.n_components, self.numel, device=self.device)
        self.table_sum = torch.zeros(self.numel, device=self.device)
        for cid, chunk in enumerate(chunks):
            grad, _ = self.grad_flat(make_batch(shard, chunk, self.device, cid))
            self.table[cid].copy_(grad)
            self.table_sum.add_(grad)
        self.opt.zero_grad(set_to_none=True)

    def estimator(self, grad: torch.Tensor, cid: int) -> torch.Tensor:
        old = self.table[cid]
        mean = self.table_sum / self.n_components
        est = grad - old + mean
        self.table_sum.add_(grad - old)
        self.table[cid].copy_(grad)
        return est

    def step(self, batch: Batch) -> float:
        grad, loss = self.grad_flat(batch)
        self.set_grad_flat(self.estimator(grad, batch.cid))
        self.opt.step()
        return loss

    def extra_state(self):
        return {"table": self.table, "table_sum": self.table_sum}
