"""End-to-end runs through the real training loop (single-process transport)."""
import csv
import os

import pytest

from dso.algos import REGISTRY
from dso.common import runner
from dso.common.config import RunConfig


def _cfg(name: str, tmp_path, **overrides) -> RunConfig:
    values = dict(
        algo=name, world_size=3, epochs=3, batch_size=50, lr=0.1, weight_decay=1e-3,
        dataset="synthetic", synthetic_samples=1200, synthetic_dim=16,
        synthetic_classes=4, topology="ring", device="cpu", transport="local",
        out_dir=str(tmp_path), eval_samples=400, sync_every=4, log_every=0,
    )
    values.update(overrides)
    return RunConfig(**values)


def _rows(summary):
    path = os.path.join(summary["run_dir"], "metrics.csv")
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("name", list(REGISTRY))
def test_every_algorithm_trains(name, tmp_path):
    summary = runner.run_experiment(_cfg(name, tmp_path))
    rows = _rows(summary)

    assert len(rows) == 3
    assert float(rows[-1]["test_loss"]) < float(rows[0]["test_loss"])
    assert summary["test_acc"] > 0.5
    assert summary["numel"] == 16 * 4 + 4
    assert os.path.exists(os.path.join(summary["run_dir"], "ckpt", "rank0.pt"))
    assert os.path.exists(os.path.join(summary["run_dir"], "config.json"))


def test_gossip_keeps_nodes_closer_than_no_communication(tmp_path):
    """The whole point of the gossip round: same steps, less divergence."""
    talkative = runner.run_experiment(
        _cfg("dsgd", tmp_path / "gossip", epochs=3, world_size=5, topology="regular"))
    silent = runner.run_experiment(
        _cfg("local_sgd", tmp_path / "silent", epochs=3, world_size=5,
             topology="regular", sync_every=10 ** 9))
    assert float(_rows(talkative)[-1]["consensus_rel"]) * 10 < \
        float(_rows(silent)[-1]["consensus_rel"])


def test_periodic_averaging_resets_the_consensus_error(tmp_path):
    """After an exact average the participants must be bit-identical."""
    cfg = _cfg("local_sgd", tmp_path, epochs=1, sync_every=4)
    summary = runner.run_experiment(cfg)
    rows = _rows(summary)
    # the epoch ends on a sync step only if steps_per_epoch % sync_every == 0
    steps = summary.get("steps_per_epoch", 8)
    if steps % cfg.sync_every == 0:
        assert float(rows[-1]["consensus_rel"]) < 1e-5


def test_plan_for_eight_nodes(tmp_path):
    cfg = RunConfig(algo="dsgd", world_size=8, topology="paper8", device="cpu",
                    dataset="synthetic", synthetic_samples=800, synthetic_dim=8,
                    synthetic_classes=4, batch_size=50, out_dir=str(tmp_path))
    plan = runner.make_plan(cfg)
    assert plan.steps_per_epoch == 2
    assert len(plan.partitions) == 8
    assert plan.topology_stats["nodes"] == 8


def test_custom_matrix_decides_the_communication_pattern(tmp_path):
    """The sparsity of the supplied matrix is the set of p2p messages sent."""
    import numpy as np

    from dso.common import topology

    asymmetric = topology.regular_directed(3, 1)      # one out-neighbour per node
    path = tmp_path / "custom_w.npy"
    np.save(path, asymmetric)

    cfg = _cfg("dsgd", tmp_path, epochs=2, world_size=3,
               topology="complete", topology_matrix=str(path))

    plan = runner.make_plan(cfg)
    assert plan.topology_stats["mean_degree"] == 1.0        # from the matrix, not "complete"
    assert cfg.topology.startswith("custom_")

    summary = runner.run_experiment(cfg)
    rows = _rows(summary)
    assert float(rows[-1]["test_loss"]) < float(rows[0]["test_loss"])


def test_bad_custom_matrix_fails_before_training(tmp_path):
    import numpy as np

    import pytest as _pytest

    bad = np.array([[0.5, 0.5, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])
    path = tmp_path / "bad_w.npy"
    np.save(path, bad)

    cfg = _cfg("dsgd", tmp_path, world_size=3, topology_matrix=str(path))
    with _pytest.raises(ValueError, match="column sums"):
        runner.make_plan(cfg)


def test_matrix_can_be_injected_from_code(tmp_path):
    """The 2021 API was My_Graph(..., weight_matrix=matrix); that still works."""
    import numpy as np

    from dso.common import topology

    asymmetric = topology.regular_directed(3, 1)
    cfg = _cfg("dsgd", tmp_path, epochs=2, world_size=3)

    plan = runner.make_plan(cfg, weight_matrix=asymmetric)
    assert plan.topology_stats["mean_degree"] == 1.0     # edge count comes from W
    assert cfg.topology == "custom"

    summary = runner.run_experiment(cfg, weight_matrix=asymmetric)
    rows = _rows(summary)
    assert float(rows[-1]["test_loss"]) < float(rows[0]["test_loss"])


def test_injected_matrix_must_match_the_world_size(tmp_path):
    import numpy as np

    import pytest as _pytest

    cfg = _cfg("dsgd", tmp_path, world_size=4)
    with _pytest.raises(ValueError, match="world_size=4"):
        runner.make_plan(cfg, weight_matrix=np.eye(3))
