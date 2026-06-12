"""Shared helpers for single-problem benchmark workflows."""

from __future__ import annotations

from .config import merged
from .method_specs import (
    base_common_params,
    build_convex_penalty_method_kwargs,
    build_first_order_method_params,
    build_penalty_method_params,
)


SINGLE_PROBLEM_BENCHMARKS = (
    "jcc_baseline_solver_sweep",
    "poly_star_benchmark",
    "socp_hompgd_benchmark",
    "convex_algorithm_comparison",
    "maxcut_algorithm_comparison",
    "jcc_algorithm_comparison",
)


def _ineq_algorithm_params(
    seed,
    max_iterations,
    max_running_time,
    learning_rate,
    stepsize_rule,
    lr_decay,
    min_lr,
    common_config=None,
):
    common = base_common_params(seed, max_iterations, max_running_time, learning_rate, stepsize_rule, lr_decay, min_lr)
    common = merged(common, common_config)
    effective_max_iterations = int(common["max_iterations"])
    return {
        "common": common,
        **build_first_order_method_params(
            common,
            projection_outer_iterations=5,
            projection_inner_iterations=10,
            linearization_outer_iterations=5,
            linearization_inner_iterations=10,
        ),
        "ALM": build_penalty_method_params(
            common,
            **build_convex_penalty_method_kwargs(
                common,
                outer_iterations=effective_max_iterations,
                inner_iterations=10,
                dual_learning_rate=5e-3,
                proximal_space="x",
            ),
        ),
    }
