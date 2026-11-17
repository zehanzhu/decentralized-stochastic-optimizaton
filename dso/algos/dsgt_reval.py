"""DSGT-REVAL — gradient tracking with same-batch gradient re-evaluation.

    y_i <- y_i + grad_i(x^k; b) - grad_i(x^{k-1}; b)

The second term re-evaluates the *previous iterate on the current mini-batch*
instead of keeping the previously computed gradient.  Both terms then come from
the same batch, so their difference is a path integral along the segment
[x^{k-1}, x^k] and is much less noisy than the difference of two independent
batch gradients.  This is the SARAH-style recursive estimator

    y_k = y_{k-1} + grad f(x_k; b) - grad f(x_{k-1}; b)

fed into a gradient-tracking consensus step; it stays unbiased because the
telescoping increments have zero mean.

Cost: one extra forward+backward per step (≈2x the compute of DSGT), same
state size (x^{k-1} instead of grad^{k-1}).

Naming: the 2021 file was called ``our_dsgt``.  "our" only means something
relative to the author's paper, so the registry keeps it as a deprecated alias
(see ``dso.algos.ALIASES``) while the canonical name describes the mechanism.

Note: ``grad_at`` restores the parameters, so the "go back one step" is a real
round-trip.  In the 2021 version ``last_model_parameters`` and
``present_model_parameters`` were the same tensor, which turned the restore
into a no-op and left the parameter vector in the gradient slot.
"""
from __future__ import annotations

import torch

from ..common.data import Batch
from ..common.model import get_params_flat
from .base import Algorithm


class DSGTReval(Algorithm):
    name = "dsgt_reval"

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.y = torch.zeros(self.numel, device=self.device)
        self._x_prev = None

    def payload(self):
        return self.flat_params() + [self.y]

    def step(self, batch: Batch) -> float:
        grad, loss = self.grad_flat(batch)                 # grad_i(x^k; b)
        if self._x_prev is None:
            self.y.copy_(grad)
        else:
            grad_prev = self.grad_at(self._x_prev, batch)  # grad_i(x^{k-1}; b)
            self.y.add_(grad - grad_prev)
        # snapshot x^k *before* the local step and before the mixing round
        self._x_prev = get_params_flat(self.model)
        self.set_grad_flat(self.y)
        self.opt.step()
        return loss

    def extra_state(self):
        return {"y": self.y, "x_prev": self._x_prev}
