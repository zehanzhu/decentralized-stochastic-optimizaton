"""Decentralized stochastic optimization — refactored package.

Layout
------
dso.common   infrastructure: data, model, topology, communication, training loop
dso.algos    one file per algorithm, each only implementing its update rule
"""
from .common.config import RunConfig

__version__ = "2.0.0"
__all__ = ["RunConfig", "__version__"]
