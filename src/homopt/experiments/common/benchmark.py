"""Public facade for shared benchmark helpers."""

from __future__ import annotations

from .convex_reference import (
    build_convex_hom_map,
    build_convex_problem,
    build_convex_problem_config,
    normalize_convex_problem_type,
    normalize_hom_origin_constraints,
    prepare_convex_reference_context,
)
from .problem_reports import (
    constraint_violation_summary,
    convex_solution_diagnostics,
    make_single_instance_solver_row,
    objective_summary,
    summarize_single_problem_run_record,
)
from .run_records import (
    ensure_record_violation_split,
    run_algorithm_suite,
    summarize_run_record,
)

__all__ = [
    "build_convex_hom_map",
    "build_convex_problem",
    "build_convex_problem_config",
    "constraint_violation_summary",
    "convex_solution_diagnostics",
    "ensure_record_violation_split",
    "make_single_instance_solver_row",
    "normalize_convex_problem_type",
    "normalize_hom_origin_constraints",
    "objective_summary",
    "prepare_convex_reference_context",
    "run_algorithm_suite",
    "summarize_run_record",
    "summarize_single_problem_run_record",
]
