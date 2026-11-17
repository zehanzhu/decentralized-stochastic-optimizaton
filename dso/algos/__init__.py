"""Algorithm registry.

``REGISTRY`` holds the canonical names; ``ALIASES`` maps retired names (e.g. the
2021 ``our_dsgt``) onto them so old command lines and run directories keep
working.  Only canonical names are exposed by ``ALGO_NAMES`` so that
``--algo all`` never runs the same algorithm twice.
"""
from __future__ import annotations

from .base import Algorithm
from .dsgd import DSGD
from .dsaga import DSAGA
from .dsgt import DSGT
from .dsgt_reval import DSGTReval
from .dsaga_gt import DSAGAGT
from .local_sgd import LocalSGD
from .periodic_avg import PeriodicAvg

REGISTRY = {
    DSGD.name: DSGD,
    DSAGA.name: DSAGA,
    DSGT.name: DSGT,
    DSGTReval.name: DSGTReval,
    DSAGAGT.name: DSAGAGT,
    LocalSGD.name: LocalSGD,
    PeriodicAvg.name: PeriodicAvg,
}

ALIASES = {
    "our_dsgt": DSGTReval.name,      # 2021 name, kept for compatibility
}

ALGO_NAMES = tuple(REGISTRY)

DESCRIPTIONS = {
    "dsgd": "decentralized SGD (gossip averaging)",
    "dsaga": "decentralized SAGA (variance reduction)",
    "dsgt": "decentralized gradient tracking",
    "dsgt_reval": "gradient tracking + same-batch gradient re-evaluation",
    "dsaga_gt": "SAGA estimator fed into gradient tracking",
    "local_sgd": "local SGD, periodic exact averaging",
    "periodic_avg": "DSGD + periodic exact averaging",
}


def canonical_name(name: str) -> str:
    """Map a (possibly retired) algorithm name onto its canonical one."""
    return ALIASES.get(name, name)


def build_algo(name: str, model, cfg) -> Algorithm:
    key = canonical_name(name)
    try:
        cls = REGISTRY[key]
    except KeyError:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"unknown algorithm {name!r}; choose from {available}") from None
    return cls(model, cfg)
