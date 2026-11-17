"""Communication layer.

Two transports with *identical* semantics, so every algorithm and the training
loop are written once:

``DistTransport``
    One process per participant (one GPU each).  The gossip round is a
    point-to-point exchange inside the default process group using
    ``batch_isend_irecv`` + ``wait()``.

``LocalTransport``
    N virtual participants inside a single process.  A faithful synchronous
    simulation of the same rounds, used by the test-suite and for laptop smoke
    runs (``--transport local``).

Both expose

    exchange(payloads)   -> [sum_j W[r, j] * payload_j  (None if isolated)]
    all_average(groups)  -> exact global averaging, in place
    all_gather(groups)   -> (world_size, numel) stacked tensor
    self_weight(rank)    -> W[rank, rank]
    barrier()

Why point-to-point instead of one process group per edge
--------------------------------------------------------
The 2021 code created a ``new_group`` per directed edge (16 groups for 8 nodes).
On NCCL that is 16 communicators to build at start-up, and every ``broadcast``
must be matched exactly.  P2P keeps a single communicator, addresses peers by
global rank, and — crucially — returns a ``Work`` handle that we explicitly
``wait()`` on.  The old code dropped that handle while the send buffer was a
temporary tensor, i.e. the buffer could be freed while still in flight.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten(flat: torch.Tensor, tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out, offset = [], 0
    for t in tensors:
        n = t.numel()
        out.append(flat.narrow(0, offset, n).view_as(t))
        offset += n
    return out


def _edges(w: np.ndarray, rank: int):
    """(peer, coefficient) lists.  in: row non-zeros, out: column non-zeros."""
    in_edges = [(int(j), float(w[rank, j]))
                for j in np.nonzero(w[rank])[0] if int(j) != rank]
    out_edges = [(int(j), float(w[j, rank]))
                 for j in np.nonzero(w[:, rank])[0] if int(j) != rank]
    return in_edges, out_edges


class Transport:
    world_size: int
    device: torch.device

    def exchange(self, payloads: List[Optional[List[torch.Tensor]]]):
        raise NotImplementedError

    def all_average(self, groups: List[List[torch.Tensor]]) -> None:
        raise NotImplementedError

    def all_gather(self, groups: List[List[torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def self_weight(self, rank: int) -> float:
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# real distributed execution
# --------------------------------------------------------------------------- #
class DistTransport(Transport):
    def __init__(self, rank: int, world_size: int, weight_matrix: np.ndarray,
                 device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self._w = np.asarray(weight_matrix, dtype=np.float64)
        self._diag = np.diag(self._w).copy()
        self.in_edges, self.out_edges = _edges(self._w, rank)

    # -- gossip -------------------------------------------------------------
    def exchange(self, payloads):
        if len(payloads) != 1:
            raise ValueError("DistTransport handles exactly one local participant")
        payload = payloads[0]
        if not self.in_edges and not self.out_edges:
            return [None]

        flat = flatten(payload).contiguous()
        ops, keep, recv = [], [], []
        for dst, _ in self.out_edges:
            buf = flat.clone()          # private buffer, alive until wait()
            keep.append(buf)
            ops.append(dist.P2POp(dist.isend, buf, dst))
        for src, _ in self.in_edges:
            buf = torch.empty_like(flat)
            recv.append(buf)
            ops.append(dist.P2POp(dist.irecv, buf, src))

        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()

        acc = torch.zeros_like(flat)
        for (_src, coeff), buf in zip(self.in_edges, recv):
            acc.add_(buf, alpha=coeff)
        return [unflatten(acc, payload)]

    # -- collectives (always on the default group) --------------------------
    def all_average(self, groups) -> None:
        if len(groups) != 1:
            raise ValueError("DistTransport handles exactly one local participant")
        if self.world_size == 1:
            return
        for t in groups[0]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for t in groups[0]:
            t.div_(self.world_size)

    def all_gather(self, groups) -> torch.Tensor:
        flat = flatten(groups[0]).contiguous()
        out = [torch.empty_like(flat) for _ in range(self.world_size)]
        dist.all_gather(out, flat)
        return torch.stack(out)

    def self_weight(self, rank: int) -> float:
        return float(self._diag[rank])

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# single-process simulation
# --------------------------------------------------------------------------- #
class LocalTransport(Transport):
    def __init__(self, world_size: int, weight_matrix: np.ndarray, device: torch.device):
        self.world_size = world_size
        self.device = device
        self._w = np.asarray(weight_matrix, dtype=np.float64)
        self._diag = np.diag(self._w).copy()
        self._edges = [_edges(self._w, r) for r in range(world_size)]

    def exchange(self, payloads):
        if len(payloads) != self.world_size:
            raise ValueError(f"expected {self.world_size} payloads, got {len(payloads)}")
        flats = [flatten(p).contiguous() if p is not None else None for p in payloads]
        out = []
        for r in range(self.world_size):
            in_edges, _ = self._edges[r]
            if not in_edges:
                out.append(None)
                continue
            acc = torch.zeros_like(flats[r])
            for src, coeff in in_edges:
                acc.add_(flats[src], alpha=coeff)
            out.append(unflatten(acc, payloads[r]))
        return out

    def all_average(self, groups) -> None:
        if len(groups) != self.world_size:
            raise ValueError(f"expected {self.world_size} groups, got {len(groups)}")
        if self.world_size == 1:
            return
        flats = [flatten(g) for g in groups]
        mean = torch.stack(flats).mean(dim=0)
        for group, flat in zip(groups, flats):
            offset = 0
            for t in group:
                n = t.numel()
                t.copy_(mean.narrow(0, offset, n).view_as(t))
                offset += n

    def all_gather(self, groups) -> torch.Tensor:
        if len(groups) != self.world_size:
            raise ValueError(f"expected {self.world_size} groups, got {len(groups)}")
        return torch.stack([flatten(g) for g in groups])

    def self_weight(self, rank: int) -> float:
        return float(self._diag[rank])
