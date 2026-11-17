"""CIFAR-10 / synthetic data, partitioning, and batching.

Design notes
------------
* Everything is float32 and **flattened once** at materialisation time; the
  original code kept float64 (18 GiB across 8 workers) and reshaped per batch.
* A worker only ever materialises its *own* rows, so the whole dataset is never
  replicated per process.
* ``Shard.chunks`` defines a **fixed** partition of the local dataset.  Those
  chunks are the SAGA components, so a component id keeps its meaning across
  epochs no matter how the iteration order is shuffled.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

CIFAR_TRAIN_FILES = tuple(f"data_batch_{i}" for i in range(1, 6))
CIFAR_TEST_FILE = "test_batch"


# --------------------------------------------------------------------------- #
# CIFAR-10
# --------------------------------------------------------------------------- #
def _cifar_path(root: str, name: str) -> str:
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"CIFAR-10 file not found: {path}\n"
            f"Download the python version of CIFAR-10 and point --data-root at the "
            f"directory that contains {CIFAR_TRAIN_FILES[0]} ... {CIFAR_TEST_FILE}."
        )
    return path


def _load_cifar_raw(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        payload = pickle.load(f, encoding="bytes")
    x = np.asarray(payload[b"data"], dtype=np.float32).reshape(-1, 3 * 32 * 32)
    y = np.asarray(payload[b"labels"], dtype=np.int64)
    return x, y


def train_labels(root: str) -> np.ndarray:
    """Labels of the whole training set (data arrays are released immediately)."""
    labels = []
    for name in CIFAR_TRAIN_FILES:
        x, y = _load_cifar_raw(_cifar_path(root, name))
        del x
        labels.append(y)
    return np.concatenate(labels)


def load_train_rows(root: str, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Materialise only the requested training rows (one batch file at a time)."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    n = idx.size
    out_x = np.empty((n, 3 * 32 * 32), dtype=np.float32)
    out_y = np.empty(n, dtype=np.int64)
    filled = np.zeros(n, dtype=bool)

    for b, name in enumerate(CIFAR_TRAIN_FILES):
        base = b * 10000
        rel = idx - base
        sel = (rel >= 0) & (rel < 10000)
        if not sel.any():
            continue
        x, y = _load_cifar_raw(_cifar_path(root, name))
        out_x[sel] = x[rel[sel]]
        out_y[sel] = y[rel[sel]]
        filled |= sel
        del x

    if not filled.all():
        missing = idx[~filled][:5]
        raise IndexError(f"training indices out of range, e.g. {missing.tolist()}")
    return out_x, out_y


def load_test(root: str) -> tuple[np.ndarray, np.ndarray]:
    return _load_cifar_raw(_cifar_path(root, CIFAR_TEST_FILE))


# --------------------------------------------------------------------------- #
# Synthetic data (smoke tests / CI, no download required)
# --------------------------------------------------------------------------- #
def synthetic_centres(dim: int, classes: int, seed: int, sep: float) -> torch.Tensor:
    """Class centres depend only on the seed, so train and test share them."""
    gen = torch.Generator().manual_seed(seed * 7919 + 17)
    return torch.randn(classes, dim, generator=gen) * sep


def make_synthetic(n: int, dim: int, classes: int, seed: int,
                   sep: float = 2.0, noise: float = 1.0,
                   split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian class centres + isotropic noise: well-posed and cheap.

    ``split`` only changes the *sampling* stream; the centres stay the same, so
    the test set is drawn from the same distribution as the training set.
    """
    centres = synthetic_centres(dim, classes, seed, sep)
    gen = torch.Generator().manual_seed(seed * 104729 + (0 if split == "train" else 1))
    y = torch.randint(0, classes, (n,), generator=gen)
    x = centres[y] + noise * torch.randn(n, dim, generator=gen)
    return x.to(torch.float32), y.to(torch.int64)


# --------------------------------------------------------------------------- #
# Partitioning
# --------------------------------------------------------------------------- #
def partition_iid(labels: np.ndarray, parts: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labels))
    return [np.sort(chunk) for chunk in np.array_split(perm, parts)]


def partition_dirichlet(labels: np.ndarray, parts: int, alpha: float, seed: int,
                        min_size: int = 25) -> List[np.ndarray]:
    """Per-class Dirichlet split; small alpha -> strongly non-IID shards."""
    rng = np.random.default_rng(seed)
    buckets: List[List[np.ndarray]] = [[] for _ in range(parts)]
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        sizes = rng.multinomial(len(idx), rng.dirichlet(np.full(parts, alpha)))
        offset = 0
        for k, size in enumerate(sizes):
            buckets[k].append(idx[offset:offset + size])
            offset += size

    out = [np.sort(np.concatenate(b)) if b else np.empty(0, dtype=np.int64) for b in buckets]
    return _rebalance(out, min_size)


def _rebalance(parts: List[np.ndarray], min_size: int) -> List[np.ndarray]:
    """Move samples from the largest shard to any shard below ``min_size``."""
    if min_size <= 0 or len(parts) < 2:
        return parts
    for _ in range(min_size * len(parts)):
        small = min(range(len(parts)), key=lambda k: len(parts[k]))
        if len(parts[small]) >= min_size:
            break
        big = max(range(len(parts)), key=lambda k: len(parts[k]))
        if big == small or len(parts[big]) <= min_size:
            break
        parts[small] = np.append(parts[small], parts[big][-1])
        parts[big] = parts[big][:-1]
    return [np.sort(p) for p in parts]


def make_partitions(cfg) -> List[np.ndarray]:
    if cfg.dataset == "cifar10":
        labels = train_labels(cfg.data_root)
    elif cfg.dataset == "synthetic":
        labels = make_synthetic(cfg.synthetic_samples, cfg.synthetic_dim,
                                cfg.synthetic_classes, cfg.seed)[1].numpy()
    else:
        raise ValueError(f"unknown dataset {cfg.dataset!r}")

    if cfg.world_size > len(labels):
        raise ValueError(f"world_size={cfg.world_size} exceeds dataset size {len(labels)}")

    if cfg.partition == "iid":
        return partition_iid(labels, cfg.world_size, cfg.seed)
    if cfg.partition == "dirichlet":
        return partition_dirichlet(labels, cfg.world_size, cfg.alpha, cfg.seed,
                                   cfg.min_partition_size)
    raise ValueError(f"unknown partition {cfg.partition!r}")


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    cid: int          # fixed chunk id == SAGA component id


@dataclass
class Shard:
    x: torch.Tensor   # (n, dim) float32, already flattened
    y: torch.Tensor   # (n,) int64

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def chunks(self, batch_size: int, drop_last: bool = True) -> List[torch.Tensor]:
        return split_indices(len(self), batch_size, drop_last=drop_last)


def split_indices(n: int, batch_size: int, drop_last: bool = True) -> List[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    bounds = list(range(0, n - batch_size + 1, batch_size))
    chunks = [torch.arange(s, s + batch_size, dtype=torch.long) for s in bounds]
    if not drop_last and n % batch_size:
        chunks.append(torch.arange(len(bounds) * batch_size, n, dtype=torch.long))
    if not chunks:
        raise ValueError(f"shard of {n} samples is smaller than batch_size={batch_size}")
    return chunks


def make_batch(shard: Shard, chunk: torch.Tensor, device, cid: int = 0) -> Batch:
    """Gather one fixed chunk and move it to the compute device."""
    return Batch(shard.x.index_select(0, chunk).to(device, non_blocking=True),
                 shard.y.index_select(0, chunk).to(device, non_blocking=True),
                 cid)


# --------------------------------------------------------------------------- #
# Materialisation
# --------------------------------------------------------------------------- #
def materialize(cfg, indices: Sequence[int]) -> Shard:
    if cfg.dataset == "cifar10":
        x, y = load_train_rows(cfg.data_root, indices)
        return Shard(torch.from_numpy(x), torch.from_numpy(y))
    if cfg.dataset == "synthetic":
        xs, ys = make_synthetic(cfg.synthetic_samples, cfg.synthetic_dim,
                                cfg.synthetic_classes, cfg.seed,
                                cfg.synthetic_sep, cfg.synthetic_noise)
        sel = torch.as_tensor(np.asarray(indices, dtype=np.int64))
        return Shard(xs[sel], ys[sel])
    raise ValueError(f"unknown dataset {cfg.dataset!r}")


def test_shard(cfg) -> Shard:
    if cfg.dataset == "cifar10":
        x, y = load_test(cfg.data_root)
        shard = Shard(torch.from_numpy(x), torch.from_numpy(y))
    elif cfg.dataset == "synthetic":
        xs, ys = make_synthetic(cfg.synthetic_samples, cfg.synthetic_dim,
                                cfg.synthetic_classes, cfg.seed,
                                cfg.synthetic_sep, cfg.synthetic_noise,
                                split="test")
        shard = Shard(xs, ys)
    else:
        raise ValueError(f"unknown dataset {cfg.dataset!r}")

    if cfg.eval_samples and cfg.eval_samples < len(shard):
        return Shard(shard.x[: cfg.eval_samples], shard.y[: cfg.eval_samples])
    return shard
