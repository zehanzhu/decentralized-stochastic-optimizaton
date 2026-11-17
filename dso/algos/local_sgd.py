"""Local SGD — no gossip, exact averaging of the parameters every ``f`` steps.

Baseline for "how little communication do we get away with".  Note that the
participants start from a common x_0 and the averaging is over the whole world,
so this is the communication-compressed counterpart of parallel SGD.
"""
from __future__ import annotations

from .base import Algorithm


class LocalSGD(Algorithm):
    name = "local_sgd"
    uses_gossip = False

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.sync_interval = cfg.sync_every

    def payload(self):
        return None

    def step(self, batch):
        return self.plain_sgd_step(batch)
