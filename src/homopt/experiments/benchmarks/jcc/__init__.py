"""JCC benchmark family."""

from __future__ import annotations

from .linear import jcc_linear_solver_benchmark, jcc_problem_benchmark
from .opf import (
    _JCCINNPGDAdapter,
    _build_jcc_problem,
    _normalize_jcc_inn_configs,
    _run_jcc_inn_pgd_variant,
    _run_jcc_solver_suite,
    jcc_algorithm_comparison,
    jcc_baseline_solver_sweep,
)

__all__ = [
    "_JCCINNPGDAdapter",
    "_build_jcc_problem",
    "_normalize_jcc_inn_configs",
    "_run_jcc_inn_pgd_variant",
    "_run_jcc_solver_suite",
    "jcc_algorithm_comparison",
    "jcc_baseline_solver_sweep",
    "jcc_linear_solver_benchmark",
    "jcc_problem_benchmark",
]
