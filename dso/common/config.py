"""Run configuration.

Kept as a plain dataclass so it can be pickled to worker processes and dumped to
JSON next to every run's metrics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RunConfig:
    # --- algorithm / optimisation ------------------------------------------------
    algo: str = "dsgd"
    seed: int = 0
    world_size: int = 8
    epochs: int = 120
    batch_size: int = 50
    lr: float = 0.05
    weight_decay: float = 1e-3
    momentum: float = 0.0
    hidden: int = 0                   # 0 -> logistic regression, >0 -> one hidden layer

    # --- graph -------------------------------------------------------------------
    topology: str = "paper8"
    topology_matrix: str = ""         # .npy/.npz/.csv/.txt; overrides `topology`

    # --- data --------------------------------------------------------------------
    dataset: str = "cifar10"          # cifar10 | synthetic
    data_root: str = "cifar-10-batches-py"
    partition: str = "iid"            # iid | dirichlet
    alpha: float = 0.1                # Dirichlet concentration
    min_partition_size: int = 25
    synthetic_samples: int = 6000
    synthetic_dim: int = 64
    synthetic_classes: int = 10
    synthetic_sep: float = 2.0
    synthetic_noise: float = 1.0

    # --- execution ---------------------------------------------------------------
    device: str = "cuda"              # cuda | cpu | auto
    backend: str = ""                 # "" -> nccl on cuda, gloo on cpu
    transport: str = "dist"           # dist | local (single-process simulation)
    dist_url: str = ""                # "" -> tcp://127.0.0.1:<free port>
    timeout_min: int = 30             # process-group timeout for collectives

    # --- logging -----------------------------------------------------------------
    sync_every: int = 20              # periodic global averaging interval
    eval_every: int = 1
    eval_batch_size: int = 1000
    eval_samples: int = 0             # 0 -> whole test set
    log_every: int = 1                # print every N epochs (0 -> only last)
    out_dir: str = "runs"
    save_dir: str = ""                # "" -> <run_dir>/ckpt
    tag: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def run_name(self) -> str:
        parts = [self.algo, self.topology, f"w{self.world_size}", f"s{self.seed}",
                 self.dataset, self.partition]
        if self.partition == "dirichlet":
            parts.append(f"a{self.alpha:g}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)
