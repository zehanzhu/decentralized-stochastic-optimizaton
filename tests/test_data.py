"""Data pipeline: partitioning, chunking, synthetic splits, CIFAR errors."""
import numpy as np
import pytest
import torch

from dso.common import data
from dso.common.config import RunConfig


def _labels(n=1000, classes=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, classes, n)


def test_iid_partition_is_a_disjoint_cover():
    labels = _labels()
    parts = data.partition_iid(labels, 8, seed=0)
    all_idx = np.concatenate(parts)
    assert len(all_idx) == len(labels)
    assert len(np.unique(all_idx)) == len(labels)
    assert {len(p) for p in parts} == {125}


def test_dirichlet_is_non_iid_and_respects_min_size():
    labels = _labels(2000)
    parts = data.partition_dirichlet(labels, 8, alpha=0.1, seed=0, min_size=25)
    assert len(np.concatenate(parts)) == len(labels)
    assert min(len(p) for p in parts) >= 25

    # alpha=0.1 must produce visibly skewed shards
    shares = [np.bincount(labels[p], minlength=10) for p in parts]
    spread = max(s.max() for s in shares)
    assert spread > 0.25 * max(len(p) for p in parts)


def test_partitions_are_deterministic():
    labels = _labels()
    a = data.partition_dirichlet(labels, 4, 0.3, seed=7)
    b = data.partition_dirichlet(labels, 4, 0.3, seed=7)
    c = data.partition_dirichlet(labels, 4, 0.3, seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_chunks_cover_the_shard():
    shard = data.Shard(torch.randn(100, 4), torch.randint(0, 2, (100,)))
    chunks = shard.chunks(25)
    assert len(chunks) == 4
    idx = torch.cat(chunks).sort().values
    assert torch.equal(idx, torch.arange(100))


def test_chunks_drop_last_and_guard_small_shards():
    shard = data.Shard(torch.randn(103, 4), torch.randint(0, 2, (103,)))
    assert len(shard.chunks(25)) == 4
    assert len(shard.chunks(25, drop_last=False)) == 5

    tiny = data.Shard(torch.randn(10, 4), torch.randint(0, 2, (10,)))
    with pytest.raises(ValueError, match="smaller than batch_size"):
        tiny.chunks(25)


def test_make_batch_shapes():
    shard = data.Shard(torch.randn(50, 7), torch.randint(0, 3, (50,)))
    batch = data.make_batch(shard, shard.chunks(25)[0], torch.device("cpu"), cid=3)
    assert batch.x.shape == (25, 7) and batch.y.shape == (25,)
    assert batch.x.dtype == torch.float32 and batch.y.dtype == torch.int64
    assert batch.cid == 3


def test_synthetic_train_and_test_share_class_centres():
    sep = 1.5
    x, y = data.make_synthetic(600, 16, 4, seed=0, sep=sep)
    xt, yt = data.make_synthetic(300, 16, 4, seed=0, sep=sep, split="test")
    centres = data.synthetic_centres(16, 4, 0, sep)
    pred = ((xt[:, None, :] - centres[None]) ** 2).sum(-1).argmin(dim=1)
    assert float((pred == yt).float().mean()) > 0.95
    assert not torch.equal(x[:300], xt)      # different samples, same distribution


def test_missing_cifar_reports_the_expected_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="data_batch_1"):
        data.load_train_rows(str(tmp_path), [0])


def test_partitions_match_world_size():
    cfg = RunConfig(dataset="synthetic", synthetic_samples=400, synthetic_dim=8,
                    synthetic_classes=4, world_size=4, partition="iid")
    parts = data.make_partitions(cfg)
    assert len(parts) == 4
    assert sum(len(p) for p in parts) == 400
