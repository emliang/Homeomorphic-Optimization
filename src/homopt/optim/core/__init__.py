"""Stable public optimizer-core contracts.

Implementation helpers stay in their owning submodules.  Optimizers import
them explicitly so dependency edges are visible in source.
"""

from .result import (
    OptimizerRunResult,
    ParametricOptimizerRunResult,
    _collect_optimizer_timing_metrics,
    _normalize_optimizer_result,
)
from .updates import AdamOptimizer, GDOptimizer, NormalizedGDOptimizer

__all__ = [
    "AdamOptimizer",
    "GDOptimizer",
    "NormalizedGDOptimizer",
    "OptimizerRunResult",
    "ParametricOptimizerRunResult",
    "_collect_optimizer_timing_metrics",
    "_normalize_optimizer_result",
]
