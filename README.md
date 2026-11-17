# Decentralized Stochastic Optimization

Reference implementations of consensus / variance-reduced / gradient-tracking
methods for **decentralized stochastic optimization**, all sharing one training
loop, one data pipeline and one communication layer so that the *only*
difference between two runs is the algorithm.

Originally a set of seven copy-pasted scripts (2021, kept byte-for-byte in
[`legacy/`](legacy/README.md)); now refactored, with the bugs of the original
fixed. Why the 2021 numbers cannot be used is summarised in
[`legacy/README.md`](legacy/README.md).

## Algorithms

| name | class | update |
| --- | --- | --- |
| `dsgd` | `DSGD` | `x <- W(x - lr*g)` — gossip averaging after every local step |
| `dsaga` | `DSAGA` | SAGA estimator `g_b - g_b(x_tau) + mean(table)`, then gossip |
| `dsgt` | `DSGT` | gradient tracking: `y += grad(x^k) - grad(x^{k-1})`, `x -= lr*y`, then mix `(x, y)` |
| `dsgt_reval` | `DSGTReval` | tracking, but the previous gradient is **re-evaluated on the current batch** at `x^{k-1}` (same-batch difference ⇒ lower variance, 2× compute) |
| `dsaga_gt` | `DSAGAGT` | the SAGA estimator fed into the tracking recursion |
| `local_sgd` | `LocalSGD` | no gossip, exact global averaging every `--sync-every` steps |
| `periodic_avg` | `PeriodicAvg` | DSGD plus an exact global average every `--sync-every` steps |

`W` is the mixing matrix: `W[i, j]` is the weight node `i` puts on node `j`.
Every topology builder returns a **doubly stochastic** matrix on a **strongly
connected** graph, and `topology.validate` refuses anything else — a broken `W`
does not crash training, it just silently prevents consensus.

`dsgt_reval` was called `our_dsgt` in the 2021 code. The old name is kept as a
registry alias, so `--algo our_dsgt` still works and old run directories stay
readable.

## Install

```bash
pip install -r requirements.txt      # torch, numpy
```

Python ≥ 3.9. Training runs on one GPU per participant (NCCL); a CPU-only
single-process simulation is available for tests and smoke runs.

## Quick start (8 GPUs, CIFAR-10)

```bash
# check the configuration first: partitions, topology, steps/epoch
python experiments/run.py --algo dsgd --world-size 8 --dry-run

# every algorithm, three seeds
python experiments/run.py --algo all --world-size 8 --epochs 120 --seeds 0 1 2

# non-IID shards, ring topology, DSGD vs the proposed method
python experiments/run.py --algo dsgd,dsgt_reval --topology ring \
    --partition dirichlet --alpha 0.1 --seeds 0 1 2
```

CIFAR-10 (python version) must be unpacked at `--data-root` (default
`cifar-10-batches-py`), i.e. that directory must contain `data_batch_1` …
`data_batch_5` and `test_batch`.

### Smoke test without a GPU or a dataset

```bash
python experiments/run.py --algo all --transport local --device cpu \
    --dataset synthetic --topology ring --world-size 4 \
    --epochs 3 --synthetic-samples 2000
```

`--transport local` runs the same rounds as N virtual participants in one
process; it validates the algorithms, not the network.

## Topologies

| `--topology` | graph | `|lambda2|` (n=8) |
| --- | --- | --- |
| `paper8` | the exact 8-node directed graph of the 2021 experiments | 0.333 |
| `ring` | symmetric ring, self + 2 neighbours | 0.805 |
| `regular` | circulant digraph, node `i` listens to `i+1 … i+degree` | 0.805 |
| `grid2x2`, `grid4x2`, `grid` | 2-D grid with Metropolis–Hastings weights | 0.854 / 0.767 |
| `complete` | full averaging every round (one-step consensus) | 0.000 |

Any **custom matrix** can be supplied instead of a preset; zero entries mean
"no message", so the sparsity of your matrix *is* the point-to-point
communication pattern:

```bash
python experiments/run.py --algo dsgd --world-size 4 \
    --topology-matrix examples/directed_ring4.csv
```

`--topology-matrix` accepts `.npy` / `.npz` / `.csv` / whitespace text, must be
`world_size x world_size`, doubly stochastic and strongly connected — see
[`examples/README.md`](examples/README.md). Validate it with `--dry-run` before
burning GPU time.

From code, the matrix can be handed in directly (validated the same way):

```python
from dso.common import runner
from dso.common.config import RunConfig

cfg = RunConfig(algo="dsgd", world_size=4, device="cuda", dataset="cifar10")
runner.run_experiment(cfg, weight_matrix=my_W)
```

`|lambda2|` is the per-round contraction factor of the consensus error, so
smaller is faster mixing. Reproduce the numbers with
`python -c "from dso.common import topology as t; print(t.format_stats(t.describe(t.build('ring', 8))))"`.

## Output

Each run writes a directory `runs/<algo>_<topology>_w<N>_s<seed>_<data>_<partition>/`:

```
config.json      full configuration + topology stats + shard sizes
metrics.csv      one row per evaluated epoch
summary.json     final metrics + metadata of the run
ckpt/rankK.pt    final model and algorithm state of participant K
```

`metrics.csv` columns:

| column | meaning |
| --- | --- |
| `train_loss` | mean local training loss over the epoch |
| `test_loss`, `test_acc` | mean over participants (each evaluates **its own** model) |
| `test_acc_std` | spread of accuracy across participants |
| `consensus_rel` | `‖x_i − x̄‖ / ‖x̄‖` over all participants — how far the models are apart |
| `param_norm` | `‖x̄‖` |
| `elapsed_s` | wall clock since the start of the run |

`runs/summary.csv` collects the final row of every run, plus
`bytes_sent_per_step` (gossip payload × out-degree, plus the amortised cost of
periodic averaging) so that communication cost can be put on the x-axis.

## Design

```
dso/
  common/
    config.py     RunConfig (picklable, dumped to JSON next to the metrics)
    data.py       CIFAR-10 / synthetic, IID & Dirichlet sharding, fixed chunks
    model.py      logistic regression / 1-hidden-layer MLP, flat-vector helpers
    topology.py   mixing matrices + validation (double stochasticity, connectivity)
    comm.py       Transport: DistTransport (NCCL p2p) | LocalTransport (in-process)
    runner.py     Participant, training loop, evaluation, logging, process spawn
  algos/
    base.py       Algorithm: payload / step / prepare / mix
    dsgd.py  dsaga.py  dsgt.py  dsgt_reval.py  dsaga_gt.py  local_sgd.py  periodic_avg.py
experiments/
  run.py          CLI, sweeps, summary table
tests/            pytest suite (runs on CPU)
```

Three properties the old code did not have:

1. **One place owns the state.** The generic `Algorithm.mix` is the only code
   that writes a participant's state back after a round, so the "several
   buffers are actually one tensor" bug cannot come back. There is an explicit
   regression test for it (`tests/test_algos.py::test_state_tensors_never_alias`).
2. **Explicit communication lifetime.** The gossip round posts all sends and
   receives with `batch_isend_irecv`, keeps every send buffer alive, then
   `wait()`s. Each directed edge has its own message; the payload is
   `cat(parameters, tracker)`, so parameters and tracker always travel together.
3. **Everything is configurable and measured.** Model size, world size, batch
   size, epochs, learning rate, topology and partitioning are arguments; no
   `30730`, no `world_size=8`, no `range(120)`.

## Comparison protocol

To make a claim like "dsgt_reval is better" supportable:

* **non-IID shards** — `--partition dirichlet --alpha 0.1`. With IID shards the
  variance-reduced and tracking methods have little to win.
* **equal communication** — `dsgt`/`dsgt_reval`/`dsaga_gt` send two payloads per
  round (`bytes_sent_per_step` is 2× `dsgd`), and `dsgt_reval` also costs 2× compute.
  Plot accuracy against bytes (and against wall clock), not only against epochs.
* **several seeds** — `--seeds 0 1 2`; report mean ± std.
* **baselines** — `local_sgd` with `--sync-every 1 5 20 50`, and `complete`
  topology as the fully-connected limit.

SAGA-based algorithms spend one extra epoch of forward/backward at start-up to
fill the gradient table (`prepare`); that cost is not part of the epoch loop.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q      # 83 tests, CPU only, no dataset needed
```

The suite covers the mixing matrices, partitioning, the transport algebra, each
algorithm's update on toy tensors, and end-to-end runs of all seven algorithms
through the real training loop.

## Limitations

* Models are linear/MLP by design (`--hidden`); the algorithms assume a
  single global model with no batch-norm statistics and no momentum state.
  `--momentum` is accepted for experimentation but the consensus analysis does
  not cover the momentum buffers.
* `dsgt_reval` performs an extra forward+backward per step; that is the point,
  but it must be accounted for in any comparison.
* `DistTransport` uses one process per GPU and point-to-point messages; it has
  been written for and tested on the NCCL path only.
