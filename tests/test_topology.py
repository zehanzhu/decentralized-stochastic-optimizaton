"""Mixing matrices: shapes, double stochasticity, connectivity, semantics."""
import numpy as np
import pytest

from dso.common import topology as topo

BUILDERS = [
    ("paper8", 8),
    ("ring", 3),
    ("ring", 8),
    ("complete", 5),
    ("regular", 8),
    ("grid2x2", 4),
    ("grid4x2", 8),
    ("grid", 9),
]


@pytest.mark.parametrize("name,n", BUILDERS)
def test_builders_are_valid(name, n):
    stats = topo.validate(topo.build(name, n))
    assert stats["nodes"] == n
    assert 0.0 <= stats["lambda2"] < 1.0 + 1e-9
    assert stats["self_weight"] > 0.0


@pytest.mark.parametrize("name,n", BUILDERS)
def test_doubly_stochastic(name, n):
    w = topo.build(name, n)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-9)
    assert (w >= 0).all()


def test_paper8_matches_the_2021_matrix():
    expected = np.array([[1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3, 0],
                         [0, 1 / 3, 0, 0, 1 / 3, 0, 1 / 3, 0],
                         [0, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0, 0],
                         [0, 1 / 3, 0, 1 / 3, 0, 1 / 3, 0, 0],
                         [1 / 3, 0, 1 / 3, 0, 1 / 3, 0, 0, 0],
                         [0, 0, 0, 1 / 3, 0, 1 / 3, 0, 1 / 3],
                         [0, 1 / 3, 0, 0, 0, 0, 1 / 3, 1 / 3],
                         [1 / 3, 0, 1 / 3, 0, 0, 0, 0, 1 / 3]])
    np.testing.assert_allclose(topo.paper8(), expected)


def test_neighbour_semantics_paper8():
    """Row = who I listen to, column = who listens to me."""
    w = topo.paper8()
    assert topo.in_neighbors(w, 0) == [3, 6]
    assert topo.out_neighbors(w, 0) == [4, 7]
    # every off-diagonal entry shows up as exactly one (in, out) pair
    for i in range(8):
        for j in topo.in_neighbors(w, i):
            assert i in topo.out_neighbors(w, j)


def test_grid_is_symmetric_with_metropolis_weights():
    w = topo.grid(4, 2)
    np.testing.assert_allclose(w, w.T)
    assert (np.diag(w) > 0).all()


def test_rejects_bad_matrices():
    with pytest.raises(ValueError, match="column sums"):
        topo.validate(np.array([[0.5, 0.5], [1.0, 0.0]]))

    with pytest.raises(ValueError, match="row sums"):
        topo.validate(np.array([[0.7, 0.5], [0.5, 0.5]]))

    with pytest.raises(ValueError, match="strongly connected"):
        topo.validate(np.eye(4))

    with pytest.raises(ValueError, match="negative"):
        topo.validate(np.array([[1.5, -0.5], [0.5, 0.5]]))


def test_paper8_requires_eight_nodes():
    with pytest.raises(ValueError, match="8 nodes"):
        topo.build("paper8", 4)


def test_reference_spectral_gaps():
    # sanity for the numbers quoted in README.md's topology table: paper8 mixes
    # much faster than a plain ring, complete mixing is exact in one round
    paper = topo.describe(topo.paper8())["lambda2"]
    ring = topo.describe(topo.ring(8))["lambda2"]
    assert paper == pytest.approx(1 / 3, abs=1e-6)
    assert ring == pytest.approx((1 + 2 * np.cos(np.pi / 4)) / 3, abs=1e-6)
    assert paper < ring
    assert topo.describe(topo.complete(8))["lambda2"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# user supplied matrices
# --------------------------------------------------------------------------- #
def test_load_matrix_from_all_supported_formats(tmp_path):
    w = topo.ring(4)

    npy = tmp_path / "w.npy"
    np.save(npy, w)
    np.testing.assert_allclose(topo.load_matrix(str(npy), expected_n=4), w)

    csv = tmp_path / "w.csv"
    np.savetxt(csv, w, delimiter=",")
    np.testing.assert_allclose(topo.load_matrix(str(csv), expected_n=4), w)

    txt = tmp_path / "w.txt"
    np.savetxt(txt, w)
    np.testing.assert_allclose(topo.load_matrix(str(txt)), w)

    npz = tmp_path / "w.npz"
    np.savez(npz, W=topo.paper8())
    np.testing.assert_allclose(topo.load_matrix(str(npz)), topo.paper8())


def test_load_matrix_checks_the_shape(tmp_path):
    not_square = tmp_path / "bad.npy"
    np.save(not_square, np.ones((3, 4)))
    with pytest.raises(ValueError, match="square"):
        topo.load_matrix(str(not_square))

    wrong_size = tmp_path / "small.npy"
    np.save(wrong_size, np.eye(3))
    with pytest.raises(ValueError, match="world_size=5"):
        topo.load_matrix(str(wrong_size), expected_n=5)


def test_resolve_prefers_the_matrix_file(tmp_path):
    from dso.common.config import RunConfig

    custom = topo.regular_directed(3, 1)
    path = tmp_path / "custom.npy"
    np.save(path, custom)

    cfg = RunConfig(world_size=3, topology="complete", topology_matrix=str(path))
    np.testing.assert_allclose(topo.resolve(cfg), custom)

    preset = RunConfig(world_size=3, topology="complete")
    np.testing.assert_allclose(topo.resolve(preset), topo.complete(3))
