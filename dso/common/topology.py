"""Mixing matrices and their validation.

Convention
----------
``W[i, j]`` is the weight that node ``i`` puts on what node ``j`` sends, i.e.
``x_i <- sum_j W[i, j] * x_j``.  Therefore:

* in-neighbours of ``i`` (nodes ``i`` listens to)  = non-zeros of row ``i``
* out-neighbours of ``i`` (nodes that listen to i) = non-zeros of column ``i``

Every builder returns a **doubly stochastic** matrix (rows *and* columns sum to
one) built on a strongly connected graph.  ``validate`` refuses anything else,
because a violation silently breaks consensus while training still "runs".
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

__all__ = [
    "build", "complete", "ring", "grid", "regular_directed", "paper8",
    "validate", "describe", "in_neighbors", "out_neighbors",
    "load_matrix", "resolve",
]


def complete(n: int) -> np.ndarray:
    return np.full((n, n), 1.0 / n, dtype=np.float64)


def ring(n: int) -> np.ndarray:
    """Symmetric ring: self + two neighbours, each 1/3."""
    if n <= 2:
        return complete(n)
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        w[i, (i - 1) % n] = 1.0 / 3.0
        w[i, i] = 1.0 / 3.0
        w[i, (i + 1) % n] = 1.0 / 3.0
    return w


def grid(rows: int, cols: int) -> np.ndarray:
    """2-D grid with Metropolis-Hastings weights.

    Boundary nodes have smaller degree, so a plain row-normalisation would break
    the column sums.  Metropolis weights ``1 / (1 + max(d_i, d_j))`` are
    symmetric and leave a positive self weight, hence doubly stochastic.
    """
    n = rows * cols
    adj: List[List[int]] = [[] for _ in range(n)]
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if r > 0:
                adj[i].append((r - 1) * cols + c)
            if r + 1 < rows:
                adj[i].append((r + 1) * cols + c)
            if c > 0:
                adj[i].append(r * cols + c - 1)
            if c + 1 < cols:
                adj[i].append(r * cols + c + 1)

    deg = [len(a) for a in adj]
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in adj[i]:
            w[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        w[i, i] = 1.0 - w[i].sum()
    return w


def regular_directed(n: int, degree: int = 2) -> np.ndarray:
    """Circulant digraph: node i listens to i+1 ... i+degree (plus itself).

    Row i has degree+1 entries, column j is fed by j-1 ... j-degree, so with
    uniform weights the matrix is doubly stochastic.  Shift 1 alone already
    makes the graph strongly connected.
    """
    if n <= 2:
        return complete(n)
    degree = max(1, min(degree, n - 1))
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        w[i, i] = 1.0
        for k in range(1, degree + 1):
            w[i, (i + k) % n] += 1.0
    return w / w.sum(axis=1, keepdims=True)


def paper8() -> np.ndarray:
    """The exact 8-node matrix used by the 2021 experiments (see legacy/)."""
    return np.array([[1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3, 0],
                     [0, 1 / 3, 0, 0, 1 / 3, 0, 1 / 3, 0],
                     [0, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0, 0],
                     [0, 1 / 3, 0, 1 / 3, 0, 1 / 3, 0, 0],
                     [1 / 3, 0, 1 / 3, 0, 1 / 3, 0, 0, 0],
                     [0, 0, 0, 1 / 3, 0, 1 / 3, 0, 1 / 3],
                     [0, 1 / 3, 0, 0, 0, 0, 1 / 3, 1 / 3],
                     [1 / 3, 0, 1 / 3, 0, 0, 0, 0, 1 / 3]], dtype=np.float64)


def build(name: str, n: int) -> np.ndarray:
    name = name.lower()
    if name == "paper8":
        if n != 8:
            raise ValueError(f"topology 'paper8' is fixed at 8 nodes, got world_size={n}")
        return paper8()
    if name == "complete":
        return complete(n)
    if name == "ring":
        return ring(n)
    if name == "regular":
        return regular_directed(n)
    if name.startswith("grid"):
        body = name[len("grid"):]
        if not body:
            side = int(round(n ** 0.5))
            if side * side != n:
                raise ValueError(f"'grid' needs a square world size, got {n}; use 'gridRxC'")
            rows = cols = side
        else:
            parts = body.split("x")
            if len(parts) == 1:
                rows = cols = int(parts[0])
            elif len(parts) == 2:
                rows, cols = int(parts[0]), int(parts[1])
            else:
                raise ValueError("grid topology must be 'grid', 'gridN' or 'gridRxC'")
        if rows * cols != n:
            raise ValueError(f"{name} has {rows * cols} nodes, world_size={n}")
        return grid(rows, cols)
    raise ValueError(f"unknown topology {name!r}")


def in_neighbors(w: np.ndarray, i: int) -> List[int]:
    return [int(j) for j in np.nonzero(w[i])[0] if int(j) != i]


def out_neighbors(w: np.ndarray, i: int) -> List[int]:
    return [int(j) for j in np.nonzero(w[:, i])[0] if int(j) != i]


# --------------------------------------------------------------------------- #
# user supplied matrices
# --------------------------------------------------------------------------- #
def load_matrix(path: str, expected_n: int = 0) -> np.ndarray:
    """Read a mixing matrix from ``.npy`` / ``.npz`` / ``.csv`` / whitespace text.

    Zero entries mean "no message": the transport derives one point-to-point
    edge per non-zero off-diagonal entry, so the communication pattern is
    exactly the sparsity of the matrix you pass in.
    """
    path = str(path)
    lower = path.lower()
    if lower.endswith(".npy"):
        w = np.load(path, allow_pickle=False)
    elif lower.endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            key = "W" if "W" in data.files else data.files[0]
            w = data[key]
    else:
        w = np.loadtxt(path, delimiter="," if lower.endswith(".csv") else None)

    w = np.atleast_2d(np.asarray(w, dtype=np.float64))
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"{path}: expected a square matrix, got shape {w.shape}")
    if expected_n and w.shape[0] != expected_n:
        raise ValueError(f"{path}: matrix is {w.shape[0]}x{w.shape[1]} but "
                         f"world_size={expected_n}; pass --world-size {w.shape[0]} "
                         f"or supply a matrix that matches")
    return w


def resolve(cfg) -> np.ndarray:
    """The mixing matrix for this run: a file if one was given, else the preset."""
    path = getattr(cfg, "topology_matrix", "")
    if path:
        return load_matrix(path, expected_n=cfg.world_size)
    return build(cfg.topology, cfg.world_size)


def _strongly_connected(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    if n == 1:
        return True
    for start in (0, 1):
        seen = {start}
        stack = [start]
        graph = adj if start == 0 else adj.T
        while stack:
            u = stack.pop()
            for v in np.nonzero(graph[u])[0]:
                if v not in seen:
                    seen.add(int(v))
                    stack.append(int(v))
        if len(seen) != n:
            return False
    return True


def validate(w: np.ndarray, tol: float = 1e-6) -> Dict[str, float]:
    """Raise on anything that would silently prevent consensus."""
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[0]
    if w.shape != (n, n):
        raise ValueError(f"weight matrix must be square, got {w.shape}")
    if np.any(w < -tol):
        raise ValueError("weight matrix has negative entries")
    if not np.allclose(w.sum(axis=1), 1.0, atol=tol):
        raise ValueError(f"row sums are not 1: {np.round(w.sum(axis=1), 6).tolist()}")
    if not np.allclose(w.sum(axis=0), 1.0, atol=tol):
        raise ValueError(f"column sums are not 1 (not doubly stochastic): "
                         f"{np.round(w.sum(axis=0), 6).tolist()}")

    off = w.copy()
    np.fill_diagonal(off, 0.0)
    adj = off > 0
    if n > 1 and not _strongly_connected(adj):
        raise ValueError("communication graph is not strongly connected")

    return describe(w)


def describe(w: np.ndarray) -> Dict[str, float]:
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[0]
    off = w.copy()
    np.fill_diagonal(off, 0.0)
    deg = (off > 0).sum(axis=1)
    eig = np.sort(np.abs(np.linalg.eigvals(w))) if n > 1 else np.array([1.0])
    return {
        "nodes": n,
        "symmetric": bool(np.allclose(w, w.T)),
        "mean_degree": float(deg.mean()),
        "lambda2": float(eig[-2]) if n > 1 else 0.0,
        "self_weight": float(np.diag(w).mean()),
    }


def format_stats(stats: Dict[str, float]) -> str:
    return (f"nodes={int(stats['nodes'])} symmetric={stats['symmetric']} "
            f"mean_degree={stats['mean_degree']:.2f} "
            f"|lambda2|={stats['lambda2']:.4f} self_weight={stats['self_weight']:.4f}")
