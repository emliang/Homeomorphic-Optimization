"""Hom-ALM experiment entrypoints."""

from __future__ import annotations

from .benchmarks.convex import convex_algorithm_comparison
from .benchmarks.convex_eq_2d import CONVEX_EQ_2D_ALGORITHMS, TOY_PROBLEM_OVERRIDES, convex_eq_2d_benchmark
from .benchmarks.jcc import jcc_algorithm_comparison, jcc_baseline_solver_sweep
from .benchmarks.stiefel import stiefel_algorithm_comparison
from .benchmarks.stiefel.compare import STIEFEL_PLOT_ALGORITHM_ORDER, STIEFEL_PLOT_LABELS

__all__ = [
    "CONVEX_EQ_2D_ALGORITHMS",
    "STIEFEL_PLOT_ALGORITHM_ORDER",
    "STIEFEL_PLOT_LABELS",
    "TOY_PROBLEM_OVERRIDES",
    "convex_algorithm_comparison",
    "convex_eq_2d_benchmark",
    "jcc_algorithm_comparison",
    "jcc_baseline_solver_sweep",
    "stiefel_algorithm_comparison",
]
