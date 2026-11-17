# Mixing matrices

Any `N x N` mixing matrix can be supplied instead of a preset:

```bash
python experiments/run.py --algo dsgd --world-size 4 \
    --topology-matrix examples/directed_ring4.csv
```

## Format

* `.npy` — a plain square array
* `.npz` — array named `W` (or the only array in the archive)
* `.csv` — comma separated, one row per line
* `.txt` — whitespace separated

`N` must equal `--world-size`. Validate a matrix without training:

```bash
python experiments/run.py --algo dsgd --world-size 4 \
    --topology-matrix examples/directed_ring4.csv --dry-run
```

## What the matrix means

`W[i, j]` is the weight that node `i` puts on what node `j` sends, i.e.
`x_i <- sum_j W[i, j] x_j`.

* **row i** = the nodes that `i` listens to → `i` receives one point-to-point
  message from each non-zero `j != i`
* **column i** = the nodes that listen to `i` → `i` sends one message to each

A **zero entry means no message at all**: the transport derives one p2p edge per
non-zero off-diagonal entry, so the sparsity of the matrix *is* the
communication pattern (and `bytes_sent_per_step` is out-degree x payload).
Isolated nodes (empty row and column) simply never communicate.

## Requirements

Checked by `topology.validate` before the run starts — a violation normally does
not crash training, it silently prevents consensus, so failing early matters:

* non-negative entries
* each row sums to 1 (node `i`'s mixing weights)
* each column sums to 1 (**doubly stochastic** — this is what keeps the average
  invariant, and hence what makes the algorithms converge to the consensus mean)
* the graph is **strongly connected** (otherwise information cannot reach
  everyone; use `--topology complete` or `ring` for dense/simple alternatives)

## Example: `directed_ring4.csv`

```
0.5,0.5,0,0        node 0 listens to itself and to 1  (0 -> 1 is *not* implied)
0,0.5,0.5,0
0,0,0.5,0.5
0.5,0,0,0.5        node 3 listens to itself and to 0
```

A directed cycle: 4 nodes, out-degree 1 each, 4 point-to-point messages per
round instead of 12 for a complete graph, asymmetric (this is a legitimate
directed consensus matrix, not a bug). Its second eigenvalue modulus is
`|lambda2| = 0.707`, i.e. the consensus error shrinks by ~30% per round —
compare that with the `paper8` value of `1/3` or a plain ring's `0.805`.
