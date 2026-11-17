"""Transport semantics: the single-process simulation must do exact gossip."""
import numpy as np
import pytest
import torch

from dso.common import topology
from dso.common.comm import LocalTransport, flatten, unflatten

CPU = torch.device("cpu")


def test_flatten_unflatten_round_trip():
    tensors = [torch.randn(3, 4), torch.randn(5)]
    flat = flatten(tensors)
    assert flat.numel() == 17
    back = unflatten(flat, tensors)
    for original, restored in zip(tensors, back):
        assert restored.shape == original.shape
        assert torch.allclose(restored, original)


def test_exchange_matches_the_weight_matrix():
    w = topology.ring(3)
    transport = LocalTransport(3, w, CPU)
    payloads = [[torch.tensor([float(r)])] for r in range(3)]
    sums = transport.exchange(payloads)

    for r in range(3):
        expected = sum(w[r, j] * j for j in range(3) if j != r)
        assert sums[r][0].item() == pytest.approx(expected)


def test_exchange_returns_views_that_mix_can_write_through():
    w = topology.complete(2)
    transport = LocalTransport(2, w, CPU)
    own = [torch.tensor([0.0, 0.0])]
    sums = transport.exchange([own, [torch.tensor([2.0, 4.0])]])
    own[0].mul_(w[0, 0]).add_(sums[0][0])
    assert torch.allclose(own[0], torch.tensor([1.0, 2.0]))   # average of 0 and 2/4


def test_isolated_node_has_no_neighbours():
    transport = LocalTransport(1, np.array([[1.0]]), CPU)
    assert transport.exchange([[torch.zeros(3)]]) == [None]


def test_all_average_is_exact():
    transport = LocalTransport(3, topology.ring(3), CPU)
    groups = [[torch.tensor([float(r), 1.0])] for r in range(3)]
    transport.all_average(groups)
    for group in groups:
        assert torch.allclose(group[0], torch.tensor([1.0, 1.0]))


def test_all_gather_stacks_every_node():
    transport = LocalTransport(3, topology.ring(3), CPU)
    groups = [[torch.tensor([float(r), 2.0 * r])] for r in range(3)]
    stacked = transport.all_gather(groups)
    assert stacked.shape == (3, 2)
    assert torch.allclose(stacked[:, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_world_size_mismatch_is_rejected():
    transport = LocalTransport(3, topology.ring(3), CPU)
    with pytest.raises(ValueError, match="expected 3"):
        transport.exchange([[torch.zeros(2)]])


def test_gossip_contracts_the_consensus_error():
    """Repeated mixing must shrink the spread by ~|lambda2| per round."""
    w = topology.regular_directed(6, 2)
    transport = LocalTransport(6, w, CPU)
    gen = torch.Generator().manual_seed(0)
    groups = [[torch.randn(8, generator=gen)] for _ in range(6)]

    def spread():
        stacked = torch.stack([g[0] for g in groups])
        return float((stacked - stacked.mean(dim=0)).norm())

    before = spread()
    for _ in range(12):
        sums = transport.exchange(groups)
        for r, (group, acc) in enumerate(zip(groups, sums)):
            group[0].mul_(w[r, r]).add_(acc[0])

    assert spread() < 0.05 * before
