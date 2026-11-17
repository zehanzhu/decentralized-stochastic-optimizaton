#!/usr/bin/env python
"""Experiment driver for the decentralized stochastic optimization suite.

Examples
--------
8-GPU sweep over every algorithm, three seeds (this is the paper's protocol)::

    python experiments/run.py --algo all --world-size 8 --epochs 120 --seeds 0 1 2

Non-IID shards on a ring, DSGD vs the proposed method::

    python experiments/run.py --algo dsgd,dsgt_reval --topology ring \
        --partition dirichlet --alpha 0.1

Laptop smoke test — no GPU, no dataset download, single process::

    python experiments/run.py --algo all --transport local --device cpu \
        --dataset synthetic --world-size 4 --epochs 3 --synthetic-samples 4000

Validate a configuration without training anything::

    python experiments/run.py --algo dsgd --world-size 8 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dso.algos import ALGO_NAMES, DESCRIPTIONS, canonical_name  # noqa: E402
from dso.common import runner  # noqa: E402
from dso.common.config import RunConfig  # noqa: E402

SUMMARY_FIELDS = ["algo", "seed", "world_size", "topology", "dataset", "partition",
                  "epochs", "test_acc", "test_loss", "consensus_rel", "elapsed_s",
                  "bytes_sent_per_step", "run_dir"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decentralized stochastic optimization experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo", default="dsgd",
                   help=f"comma separated, or 'all'. one of: {', '.join(ALGO_NAMES)} "
                        f"(alias: our_dsgt -> dsgt_reval)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="random seeds")
    p.add_argument("--tag", default="", help="extra suffix for the run directory")

    g = p.add_argument_group("distributed setup")
    g.add_argument("--world-size", type=int, default=8, help="number of participants")
    g.add_argument("--transport", choices=["dist", "local"], default="dist",
                   help="dist: one process per GPU; local: single-process simulation")
    g.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    g.add_argument("--backend", default="", help="nccl/gloo (default: nccl on cuda)")
    g.add_argument("--dist-url", default="", help="master address (default: 127.0.0.1)")

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=120)
    g.add_argument("--batch-size", type=int, default=50)
    g.add_argument("--lr", type=float, default=0.05)
    g.add_argument("--weight-decay", type=float, default=1e-3)
    g.add_argument("--momentum", type=float, default=0.0)
    g.add_argument("--hidden", type=int, default=0, help="0 = logistic regression")
    g.add_argument("--sync-every", type=int, default=20,
                   help="periodic global averaging interval (local_sgd, periodic_avg)")

    g = p.add_argument_group("graph")
    g.add_argument("--topology", default="paper8",
                   help="paper8 | ring | complete | regular | gridRxC")
    g.add_argument("--topology-matrix", default="",
                   help="load the mixing matrix from .npy/.npz/.csv/.txt instead of a "
                        "preset (overrides --topology). Must be square, doubly "
                        "stochastic and strongly connected; zero entries mean "
                        "'no message', so its sparsity is the communication pattern")

    g = p.add_argument_group("data")
    g.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    g.add_argument("--data-root", default="cifar-10-batches-py")
    g.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    g.add_argument("--alpha", type=float, default=0.1, help="Dirichlet concentration")
    g.add_argument("--synthetic-samples", type=int, default=6000)
    g.add_argument("--synthetic-dim", type=int, default=64)
    g.add_argument("--synthetic-classes", type=int, default=10)

    g = p.add_argument_group("evaluation / output")
    g.add_argument("--eval-every", type=int, default=1)
    g.add_argument("--eval-samples", type=int, default=0, help="0 = whole test set")
    g.add_argument("--log-every", type=int, default=1, help="print every N epochs")
    g.add_argument("--out-dir", default="runs")
    g.add_argument("--save-dir", default="")
    g.add_argument("--dry-run", action="store_true",
                   help="build the plan (partitions, topology) and exit")
    return p


def make_config(args, algo: str, seed: int) -> RunConfig:
    return RunConfig(
        algo=algo, seed=seed, world_size=args.world_size,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, momentum=args.momentum, hidden=args.hidden,
        topology=args.topology, topology_matrix=args.topology_matrix,
        dataset=args.dataset, data_root=args.data_root,
        partition=args.partition, alpha=args.alpha,
        synthetic_samples=args.synthetic_samples, synthetic_dim=args.synthetic_dim,
        synthetic_classes=args.synthetic_classes,
        device=args.device, backend=args.backend, transport=args.transport,
        dist_url=args.dist_url, sync_every=args.sync_every,
        eval_every=args.eval_every, eval_samples=args.eval_samples,
        log_every=args.log_every, out_dir=args.out_dir, save_dir=args.save_dir,
        tag=args.tag,
    )


def resolve_algos(spec: str):
    """Canonicalise names (``our_dsgt`` -> ``dsgt_reval``) and drop duplicates."""
    if spec.strip() == "all":
        return list(ALGO_NAMES)
    algos = [canonical_name(a.strip()) for a in spec.split(",") if a.strip()]
    unknown = [a for a in algos if a not in ALGO_NAMES]
    if unknown:
        raise SystemExit(f"unknown algorithm(s): {unknown}; available: {list(ALGO_NAMES)}")
    return list(dict.fromkeys(algos))


def write_summary(out_dir: str, results) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return path


def print_table(results) -> None:
    if not results:
        return
    header = f"{'algo':<12}{'seed':>5}{'acc':>9}{'loss':>9}{'consensus':>12}{'MB/step':>10}{'time':>9}"
    print("\n" + header)
    print("-" * len(header))
    for row in results:
        print(f"{row['algo']:<12}{row['seed']:>5}{row['test_acc'] * 100:>8.2f}%"
              f"{row['test_loss']:>9.4f}{row['consensus_rel']:>12.2e}"
              f"{row['bytes_sent_per_step'] / 2 ** 20:>10.2f}{row['elapsed_s']:>8.0f}s")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    algos = resolve_algos(args.algo)
    configs = [make_config(args, algo, seed) for algo in algos for seed in args.seeds]

    print(f"{len(configs)} run(s): {', '.join(DESCRIPTIONS[a] for a in algos)}", flush=True)
    results = []
    for index, cfg in enumerate(configs, 1):
        # the run directory name is only final after make_plan() (a custom matrix
        # renames the topology), so describe_plan() below prints it
        print(f"\n===== run {index}/{len(configs)}: algo={cfg.algo} seed={cfg.seed} =====",
              flush=True)
        if args.dry_run:
            plan = runner.make_plan(cfg)
            print(runner.describe_plan(plan), flush=True)
            continue
        results.append(runner.run_experiment(cfg))

    if args.dry_run or not results:
        return 0

    path = write_summary(args.out_dir, results)
    print_table(results)
    print(f"\nper-epoch metrics: {args.out_dir}/<run>/metrics.csv")
    print(f"summary: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
