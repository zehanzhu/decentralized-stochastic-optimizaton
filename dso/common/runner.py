"""Training loop, evaluation, logging and process orchestration."""
from __future__ import annotations

import csv
import datetime
import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..algos import build_algo
from . import data, topology
from .comm import DistTransport, LocalTransport
from .config import RunConfig
from .model import init_identical, make_model

METRIC_FIELDS = ["epoch", "train_loss", "test_loss", "test_acc", "test_acc_std",
                 "consensus_rel", "param_norm", "elapsed_s"]


# --------------------------------------------------------------------------- #
# one participant
# --------------------------------------------------------------------------- #
class Participant:
    """A single decentralized learner: model + algorithm + its own shard."""

    def __init__(self, rank: int, cfg: RunConfig, shard: data.Shard,
                 test: data.Shard, device: torch.device):
        self.rank = rank
        self.cfg = cfg
        self.device = device
        self.shard = shard
        self.test = test

        dim = int(shard.x.shape[1])
        num_classes = int(max(int(shard.y.max()), int(test.y.max()))) + 1
        self.model = make_model(dim, num_classes, cfg.hidden).to(device)
        init_identical(self.model, cfg.seed)
        self.algo = build_algo(cfg.algo, self.model, cfg)

        self.chunks = shard.chunks(cfg.batch_size)
        self.order = np.arange(len(self.chunks))
        self.steps = 0

    # -- lifecycle -----------------------------------------------------------
    def prepare(self) -> None:
        self.algo.prepare(self.shard, self.cfg.batch_size)

    def start_epoch(self, seed: int, epoch: int, steps: int) -> None:
        rng = np.random.default_rng((seed * 100003 + self.rank * 1009 + epoch) % (2 ** 63))
        self.order = rng.permutation(len(self.chunks))
        self.steps = min(steps, len(self.chunks))

    # -- one step ------------------------------------------------------------
    def batch(self, t: int) -> data.Batch:
        cid = int(self.order[t])
        return data.make_batch(self.shard, self.chunks[cid], self.device, cid)

    def step(self, t: int) -> float:
        return self.algo.step(self.batch(t))

    def payload(self):
        return self.algo.payload()

    def param_payload(self) -> List[torch.Tensor]:
        return self.algo.flat_params()

    def mix(self, sums, w_self: float) -> None:
        self.algo.mix(sums, w_self)

    def sync_due(self, t: int) -> bool:
        return self.algo.sync_due(t)

    @property
    def sync_interval(self) -> Optional[int]:
        return self.algo.sync_interval

    @property
    def numel(self) -> int:
        return self.algo.numel

    # -- evaluation ----------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, batch_size: int):
        was_training = self.model.training
        self.model.eval()
        total = correct = 0
        loss_sum = 0.0
        n = len(self.test)
        for start in range(0, n, batch_size):
            xb = self.test.x[start:start + batch_size].to(self.device)
            yb = self.test.y[start:start + batch_size].to(self.device)
            logits = self.model(xb)
            loss_sum += float(self.algo.criterion(logits, yb)) * xb.shape[0]
            correct += int((logits.argmax(dim=1) == yb).sum())
            total += xb.shape[0]
        if was_training:
            self.model.train()
        return loss_sum / max(total, 1), correct / max(total, 1)


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
@dataclass
class RunPlan:
    config: RunConfig
    partitions: List[np.ndarray]
    weight_matrix: np.ndarray
    steps_per_epoch: int
    run_dir: str
    topology_stats: Dict[str, float]


def make_plan(cfg: RunConfig, weight_matrix: Optional[np.ndarray] = None) -> RunPlan:
    """Build the run plan.

    ``weight_matrix`` takes precedence over ``cfg.topology_matrix`` and over the
    preset name, so a matrix can be injected straight from code:

        runner.run_experiment(cfg, weight_matrix=my_W)
    """
    if cfg.world_size < 1:
        raise ValueError("world_size must be >= 1")

    if weight_matrix is not None:
        weight = np.asarray(weight_matrix, dtype=np.float64)
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError(f"weight_matrix must be square, got shape {weight.shape}")
        if weight.shape[0] != cfg.world_size:
            raise ValueError(f"weight_matrix is {weight.shape[0]}x{weight.shape[1]} but "
                             f"world_size={cfg.world_size}")
        if not cfg.topology_matrix:
            cfg.topology = "custom"
    else:
        # a matrix file defines the graph, so it also names the run
        if cfg.topology_matrix:
            cfg.topology = _custom_tag(cfg.topology_matrix)
        weight = topology.resolve(cfg)

    stats = topology.validate(weight)
    partitions = data.make_partitions(cfg)

    try:
        steps = min(len(data.split_indices(len(p), cfg.batch_size)) for p in partitions)
    except ValueError as exc:
        sizes = [int(len(p)) for p in partitions]
        raise ValueError(f"a shard is too small for batch_size={cfg.batch_size}: "
                         f"shard sizes {sizes} ({exc})") from None

    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({**cfg.to_dict(),
                   "topology_stats": stats,
                   "steps_per_epoch": steps,
                   "shard_sizes": [int(len(p)) for p in partitions]}, fh, indent=2)
    return RunPlan(cfg, partitions, weight, steps, run_dir, stats)


def describe_plan(plan: RunPlan) -> str:
    cfg = plan.config
    sizes = [int(len(p)) for p in plan.partitions]
    graph = cfg.topology
    if cfg.topology_matrix:
        graph += f" (matrix={cfg.topology_matrix})"
    return (f"algo={cfg.algo} world={cfg.world_size} topology={graph} "
            f"({topology.format_stats(plan.topology_stats)})\n"
            f"          shards={sizes} steps/epoch={plan.steps_per_epoch} "
            f"batch={cfg.batch_size} lr={cfg.lr} wd={cfg.weight_decay}\n"
            f"          run_dir={plan.run_dir}")


def _custom_tag(path: str) -> str:
    """Readable, filesystem-safe tag for a matrix loaded from a file."""
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in stem)
    return f"custom_{safe[:24]}" if safe else "custom"


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def train(plan: RunPlan, nodes: List[Participant], transport, is_master: bool) -> dict:
    cfg = plan.config
    started = time.time()
    for node in nodes:
        node.prepare()

    logger = None
    if is_master:
        logger = _CsvLogger(os.path.join(plan.run_dir, "metrics.csv"))
    last: dict = {}

    try:
        for epoch in range(1, cfg.epochs + 1):
            for node in nodes:
                node.start_epoch(cfg.seed, epoch, plan.steps_per_epoch)

            train_loss = 0.0
            for t in range(plan.steps_per_epoch):
                train_loss += sum(node.step(t) for node in nodes) / len(nodes)

                payloads = [node.payload() for node in nodes]
                if payloads[0] is not None:
                    sums = transport.exchange(payloads)
                    for node, acc in zip(nodes, sums):
                        node.mix(acc, transport.self_weight(node.rank))

                if nodes[0].sync_due(t):
                    transport.all_average([node.param_payload() for node in nodes])

            train_loss /= plan.steps_per_epoch
            if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
                last = _log_epoch(plan, nodes, transport, epoch, train_loss,
                                  logger, is_master, started)
    finally:
        if logger is not None:
            logger.close()

    _save_checkpoints(plan, nodes)
    last.update(_run_metadata(plan, nodes))
    if is_master:
        with open(os.path.join(plan.run_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(last, fh, indent=2)
    return last


def _log_epoch(plan: RunPlan, nodes: List[Participant], transport, epoch: int,
               train_loss: float, logger: Optional["_CsvLogger"], is_master: bool,
               started: float) -> dict:
    cfg = plan.config
    device = nodes[0].device

    evals = [node.evaluate(cfg.eval_batch_size) for node in nodes]
    meta = torch.tensor(evals, dtype=torch.float32, device=device)
    gathered = transport.all_gather([[meta[i]] for i in range(len(nodes))])

    params = transport.all_gather([node.param_payload() for node in nodes])
    mean = params.mean(dim=0)
    consensus = float((params - mean).norm() / (mean.norm() + 1e-12))

    acc = gathered[:, 1]
    row = {
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "test_loss": round(float(gathered[:, 0].mean()), 6),
        "test_acc": round(float(acc.mean()), 6),
        "test_acc_std": round(float(acc.std(unbiased=False)) if acc.numel() > 1 else 0.0, 6),
        "consensus_rel": consensus,
        "param_norm": round(float(mean.norm()), 6),
        "elapsed_s": round(time.time() - started, 2),
    }

    if logger is not None:
        logger.write(row)
    if is_master and cfg.log_every and (epoch % cfg.log_every == 0 or epoch == cfg.epochs):
        print(f"[{cfg.algo:10s}] epoch {epoch:3d}/{cfg.epochs}  "
              f"train {row['train_loss']:.4f}  test {row['test_loss']:.4f}  "
              f"acc {row['test_acc'] * 100:5.2f}%  consensus {consensus:.2e}  "
              f"{row['elapsed_s']:.0f}s", flush=True)
    return row


def _run_metadata(plan: RunPlan, nodes: List[Participant]) -> dict:
    cfg = plan.config
    numel = nodes[0].numel
    off = plan.weight_matrix.copy()
    np.fill_diagonal(off, 0.0)
    out_degree = float((off > 0).sum(axis=0).mean())
    gossip_bytes = out_degree * numel * 4
    sync_bytes = 0.0
    if nodes[0].sync_interval:
        sync_bytes = 2.0 * numel * 4 / nodes[0].sync_interval
    return {
        "algo": cfg.algo,
        "seed": cfg.seed,
        "world_size": cfg.world_size,
        "topology": cfg.topology,
        "dataset": cfg.dataset,
        "partition": cfg.partition,
        "epochs": cfg.epochs,
        "run_dir": plan.run_dir,
        "numel": numel,
        "bytes_sent_per_step": round(gossip_bytes + sync_bytes, 1),
    }


def _save_checkpoints(plan: RunPlan, nodes: List[Participant]) -> None:
    ckpt_dir = plan.config.save_dir or os.path.join(plan.run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    for node in nodes:
        extra = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                 for k, v in node.algo.extra_state().items()}
        torch.save({"model": node.model.state_dict(), "algo": extra,
                    "config": node.cfg.to_dict()},
                   os.path.join(ckpt_dir, f"rank{node.rank}.pt"))


class _CsvLogger:
    def __init__(self, path: str):
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=METRIC_FIELDS)
        self._writer.writeheader()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- #
# execution backends
# --------------------------------------------------------------------------- #
def run_experiment(cfg: RunConfig,
                   weight_matrix: Optional[np.ndarray] = None) -> dict:
    plan = make_plan(cfg, weight_matrix)
    print(describe_plan(plan), flush=True)
    if cfg.transport == "local":
        return _run_local(plan)
    return _run_distributed(plan)


def _run_local(plan: RunPlan) -> dict:
    cfg = plan.config
    device = _resolve_device(cfg, 0)
    transport = LocalTransport(cfg.world_size, plan.weight_matrix, device)
    test = data.test_shard(cfg)
    nodes = [Participant(rank, cfg, data.materialize(cfg, plan.partitions[rank]), test, device)
             for rank in range(cfg.world_size)]
    return train(plan, nodes, transport, is_master=True)


def _run_distributed(plan: RunPlan) -> dict:
    port = _free_port()
    mp.spawn(_dist_entry, nprocs=plan.config.world_size, args=(plan, port), join=True)
    with open(os.path.join(plan.run_dir, "summary.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _dist_entry(rank: int, plan: RunPlan, port: int) -> None:
    cfg = plan.config
    device = _resolve_device(cfg, rank)
    backend = cfg.backend or ("nccl" if device.type == "cuda" else "gloo")
    init_method = cfg.dist_url or f"tcp://127.0.0.1:{port}"

    dist.init_process_group(backend=backend, init_method=init_method,
                            world_size=cfg.world_size, rank=rank,
                            timeout=datetime.timedelta(minutes=cfg.timeout_min))
    transport = DistTransport(rank, cfg.world_size, plan.weight_matrix, device)
    try:
        node = Participant(rank, cfg, data.materialize(cfg, plan.partitions[rank]),
                           data.test_shard(cfg), device)
        train(plan, [node], transport, is_master=(rank == 0))
    finally:
        # no barrier here on purpose: if one rank died, a barrier would block the
        # survivors until the process-group timeout instead of failing fast
        transport.close()


def _resolve_device(cfg: RunConfig, rank: int) -> torch.device:
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is unavailable")
        if rank >= torch.cuda.device_count():
            raise RuntimeError(f"rank {rank} needs a GPU but only "
                               f"{torch.cuda.device_count()} are visible")
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)
    return torch.device("cpu")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
