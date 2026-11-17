"""Periodic global averaging on top of DSGD.

Same per-step gossip as DSGD, plus an *exact* global average every ``f`` steps.
Useful to see how much of the consensus error one strong synchronisation
removes; compare against local_sgd (no gossip at all) and dsgd (no resync).
"""
from __future__ import annotations

from .base import Algorithm


class PeriodicAvg(Algorithm):
    name = "periodic_avg"

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.sync_interval = cfg.sync_every

    def step(self, batch):
        return self.plain_sgd_step(batch)
