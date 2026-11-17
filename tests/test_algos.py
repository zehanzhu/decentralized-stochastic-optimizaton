"""Algorithm updates, payloads, and the no-aliasing regression."""
import pytest
import torch

from dso.algos import ALIASES, ALGO_NAMES, REGISTRY, build_algo, canonical_name
from dso.common.config import RunConfig
from dso.common.data import Shard, make_batch, make_synthetic
from dso.common.model import get_params_flat, make_model, set_params_flat

CPU = torch.device("cpu")

TRACKING = {"dsgt", "dsgt_reval", "dsaga_gt"}


def _setup(name, n=600, dim=16, classes=4, batch=50, lr=0.1):
    x, y = make_synthetic(n, dim, classes, seed=0)
    shard = Shard(x, y)
    model = make_model(dim, classes)
    cfg = RunConfig(algo=name, lr=lr, weight_decay=0.0, batch_size=batch)
    algo = build_algo(name, model, cfg)
    algo.prepare(shard, batch)
    return algo, shard, model, cfg


@pytest.mark.parametrize("name", list(REGISTRY))
def test_loss_decreases(name):
    algo, shard, model, cfg = _setup(name)
    chunks = shard.chunks(cfg.batch_size)
    first = last = None
    for _ in range(8):
        for cid, chunk in enumerate(chunks):
            loss = algo.step(make_batch(shard, chunk, CPU, cid))
            if first is None:
                first = loss
            last = loss
    assert last < 0.8 * first, f"{name}: loss went {first:.4f} -> {last:.4f}"


@pytest.mark.parametrize("name", ["dsgd", "dsaga", "periodic_avg", "dsgt",
                                  "dsgt_reval", "dsaga_gt"])
def test_payload_carries_params_and_the_tracker(name):
    algo, *_ = _setup(name)
    n_params = len(list(algo.model.parameters()))
    tracking = name in TRACKING
    payload = algo.payload()

    assert len(payload) == n_params + (1 if tracking else 0)
    assert sum(t.numel() for t in payload) == algo.numel * (2 if tracking else 1)


def test_local_sgd_has_no_gossip():
    algo, *_ = _setup("local_sgd")
    assert algo.payload() is None
    assert algo.uses_gossip is False
    assert algo.sync_interval == algo.cfg.sync_every
    assert algo.sync_due(algo.sync_interval - 1)
    assert not algo.sync_due(algo.sync_interval - 2)


@pytest.mark.parametrize("name", ["dsgt", "dsgt_reval", "dsaga_gt", "dsaga"])
def test_state_tensors_never_alias(name):
    """Regression: in the 2021 code several logical buffers were one tensor,
    so `y -= last_y` silently zeroed the tracker and `x_prev` became a no-op."""
    algo, shard, model, cfg = _setup(name)
    for cid, chunk in enumerate(shard.chunks(cfg.batch_size)[:2]):
        algo.step(make_batch(shard, chunk, CPU, cid))

    params = algo.flat_params()
    param_ptrs = {t.data_ptr() for t in params}
    assert len(param_ptrs) == len(params), "parameters share storage"

    state = []
    for attr in ("y", "_prev_grad", "_x_prev", "_prev_est", "table", "table_sum"):
        value = getattr(algo, attr, None)
        if torch.is_tensor(value):
            state.append((attr, value))

    assert state, f"{name} exposes no state tensors"
    for attr, value in state:
        assert value.data_ptr() not in param_ptrs, f"{name}.{attr} aliases a parameter"
    for i, (name_a, tensor_a) in enumerate(state):
        for name_b, tensor_b in state[:i]:
            assert tensor_a.data_ptr() != tensor_b.data_ptr(), \
                f"{name}.{name_a} aliases {name}.{name_b}"


def test_mix_writes_through_to_the_parameters():
    algo, *_ = _setup("dsgd")
    before = get_params_flat(algo.model).clone()
    neighbour_sum = [t.clone().fill_(2.0) for t in algo.payload()]
    algo.mix(neighbour_sum, w_self=0.0)
    assert torch.allclose(get_params_flat(algo.model), torch.full_like(before, 2.0))
    assert not torch.allclose(get_params_flat(algo.model), before)


def test_mix_is_a_no_op_without_neighbours():
    algo, *_ = _setup("dsgd")
    before = get_params_flat(algo.model).clone()
    algo.mix(None, w_self=0.5)
    assert torch.equal(get_params_flat(algo.model), before)


def test_dsgt_reval_restores_the_previous_iterate_exactly():
    algo, shard, model, cfg = _setup("dsgt_reval")
    batch = make_batch(shard, shard.chunks(cfg.batch_size)[0], CPU, 0)

    here = get_params_flat(model).clone()
    grad_here, _ = algo.grad_flat(batch)

    # a *random* perturbation: shifting every weight by the same constant would
    # leave the softmax unchanged and the gradients identical
    other = here + 0.1 * torch.randn_like(here)
    grad_there = algo.grad_at(other, batch)

    assert torch.allclose(get_params_flat(model), here)      # restored
    assert not torch.allclose(grad_here, grad_there)         # genuinely re-evaluated
    assert algo.params[0].grad is None                       # grads left clean


def test_dsgt_reval_snapshots_the_iterate_before_the_step():
    algo, shard, model, cfg = _setup("dsgt_reval")
    x0 = get_params_flat(model).clone()
    algo.step(make_batch(shard, shard.chunks(cfg.batch_size)[0], CPU, 0))

    assert torch.allclose(algo._x_prev, x0)
    assert algo._x_prev.data_ptr() != model.weight.data_ptr()
    assert not torch.allclose(get_params_flat(model), x0)     # the step happened


def test_dsgt_tracker_initialises_to_the_first_gradient():
    """First round: y_0 = grad(x_0), so the step direction is the true gradient."""
    algo, shard, model, cfg = _setup("dsgt")
    batch = make_batch(shard, shard.chunks(cfg.batch_size)[0], CPU, 0)
    x0 = get_params_flat(model).clone()
    algo.step(batch)

    set_params_flat(model, x0)
    grad, _ = algo.grad_flat(batch)
    assert torch.allclose(algo.y, grad)


def test_saga_table_holds_the_mean_gradient_at_x0():
    algo, shard, model, cfg = _setup("dsaga")
    model.zero_grad(set_to_none=True)
    chunks = shard.chunks(cfg.batch_size)
    for cid, chunk in enumerate(chunks):
        algo.loss(make_batch(shard, chunk, CPU, cid)).backward()
    mean = torch.cat([p.grad.reshape(-1) for p in model.parameters()]) / len(chunks)
    assert torch.allclose(algo.table_sum / algo.n_components, mean, atol=1e-6)


def test_saga_updates_only_the_sampled_component():
    algo, shard, model, cfg = _setup("dsaga")
    chunks = shard.chunks(cfg.batch_size)
    # move x away from x_0 first, otherwise the gradient equals the stored one
    algo.step(make_batch(shard, chunks[0], CPU, 0))
    before = algo.table.clone()
    algo.step(make_batch(shard, chunks[3], CPU, 3))
    changed = [i for i in range(len(chunks))
               if not torch.allclose(algo.table[i], before[i])]
    assert changed == [3]


def test_legacy_alias_still_resolves():
    """`our_dsgt` was the 2021 name; old command lines must keep working."""
    assert canonical_name("our_dsgt") == "dsgt_reval"
    algo = build_algo("our_dsgt", make_model(4, 3), RunConfig(algo="our_dsgt"))
    assert algo.name == "dsgt_reval"
    assert type(algo) is REGISTRY["dsgt_reval"]


def test_aliases_are_not_advertised_as_algorithms():
    """`--algo all` iterates ALGO_NAMES, so it must not contain retired names."""
    assert not set(ALIASES) & set(REGISTRY)
    assert set(REGISTRY) == set(ALGO_NAMES)
