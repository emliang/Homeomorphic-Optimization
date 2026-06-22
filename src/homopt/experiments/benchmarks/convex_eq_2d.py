"""2D convex-equality Hom-ALM visualization benchmark."""

from __future__ import annotations

import numpy as np

from homopt.records.artifacts import load_incremental_comparison_artifacts
from homopt.experiments.common.runtime import resolve_runtime
from homopt.experiments.benchmarks.convex import convex_algorithm_comparison
from homopt.solvers.references import (
    build_convex_hom_map,
    build_convex_problem,
    build_convex_problem_config,
)
from homopt.viz import save_convex_2d_visualizations


CONVEX_EQ_2D_ALGORITHMS = ["ALM", "Hom-ALM"]

TOY_LINEAR_A = np.asarray(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [2.0, -1.0],
        [-1.0, 2.0],
        [1.0, -2.0],
        [0.5, 1.0],
    ],
    dtype=float,
)
TOY_LINEAR_B = np.asarray([1.5, 0.8, 1.2, 0.6, 1.8, 1.0, 2.2, 1.5, 1.8, 1.3], dtype=float)
TOY_QUAD_Q_RAW = np.asarray([[0.3, 0.1], [0.2, 2.0]], dtype=float)
TOY_QUAD_Q = 0.5 * (TOY_QUAD_Q_RAW + TOY_QUAD_Q_RAW.T)
TOY_QUAD_P = np.asarray([-0.1, -0.2], dtype=float)
TOY_OBJECTIVE_Q = np.asarray([[1.0, 0.35], [0.35, 1.0]], dtype=float)
TOY_OBJECTIVE_CENTER = np.asarray([1.2, -0.2], dtype=float)
TOY_EQUALITY_A = np.asarray([[1.0, 1.0]], dtype=float)
TOY_EQUALITY_B = np.asarray([1.0], dtype=float)
EXPECTED_BOUNDARY_OPTIMUM = np.asarray([16.0 / 15.0, -1.0 / 15.0], dtype=float)
TOY_QUAD_B = np.asarray([0.2], dtype=float)

TOY_PROBLEM_OVERRIDES = {
    "Q": TOY_OBJECTIVE_Q,
    "p": -TOY_OBJECTIVE_CENTER @ TOY_OBJECTIVE_Q,
    "A": TOY_LINEAR_A,
    "b": TOY_LINEAR_B,
    "A_eq": TOY_EQUALITY_A,
    "b_eq": TOY_EQUALITY_B,
    "Qq": TOY_QUAD_Q.reshape(1, 2, 2),
    "pq": TOY_QUAD_P.reshape(1, 2),
    "bq": TOY_QUAD_B,
    "G": None,
    "h": None,
    "C": None,
    "d": None,
    "L": np.asarray([-2.0, -2.0], dtype=float),
    "U": np.asarray([2.0, 2.0], dtype=float),
}


def _build_problem(params):
    runtime_device, runtime_dtype = resolve_runtime(
        params.get("device"),
        params.get("dtype"),
        default_device="cpu",
    )
    config = build_convex_problem_config(
        seed=params["seed"],
        n_var=params["n_var"],
        n_linear_cons=params["n_linear_cons"],
        n_soc_cons=params["n_soc_cons"],
        n_qua_cons=params["n_qua_cons"],
        n_lin_eq=params["n_lin_eq"],
        obj=params["obj"],
        x_lower=params["x_lower"],
        x_upper=params["x_upper"],
        margin_scale=params["margin_scale"],
        anchor_interior_ratio=params["anchor_interior_ratio"],
        quad_diag_lower=params["quad_diag_lower"],
        quad_diag_upper=params["quad_diag_upper"],
        low_rank_quad_ridge=params["low_rank_quad_ridge"],
        explicit_config=params["problem_config"],
    )
    return build_convex_problem(
        config=config,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        problem_type=params["problem_type"],
    )


def _build_hom_map(params, problem, payload):
    x_origin = payload.get("reference_origin")
    if x_origin is None:
        return None
    runtime_device, runtime_dtype = resolve_runtime(
        params.get("device"),
        params.get("dtype"),
        default_device="cpu",
    )
    common_params = params["common_config"]
    return build_convex_hom_map(
        problem,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        x_origin=np.asarray(x_origin, dtype=float),
        smooth=common_params["smooth"],
        hom_p_norm=common_params["hom_p_norm"],
        hom_map_explicit_gradient_rule=common_params["hom_map_explicit_gradient_rule"],
        hom_map_smooth_tie_tol=common_params["hom_map_smooth_tie_tol"],
        hom_map_smooth_temperature=common_params["hom_map_smooth_temperature"],
    )


def _load_records(output_dir):
    records, _, _, _ = load_incremental_comparison_artifacts(
        output_dir,
        algorithms=CONVEX_EQ_2D_ALGORITHMS,
    )
    return records


def convex_eq_2d_benchmark(output_dir=None, **params):
    benchmark_params = dict(params)
    show_constraint_notation = bool(benchmark_params.pop("show_constraint_notation"))
    payload = convex_algorithm_comparison(output_dir=output_dir, **benchmark_params)
    problem = _build_problem(params)
    hom_map = _build_hom_map(params, problem, payload)
    records = _load_records(output_dir)
    viz_artifacts = save_convex_2d_visualizations(
        problem=problem,
        records=records,
        algorithms=CONVEX_EQ_2D_ALGORITHMS,
        output_dir=output_dir,
        hom_map=hom_map,
        payload=payload,
        prefix="hal_2d",
        include_individual=True,
        show_constraint_notation=show_constraint_notation,
        violation_y_min=float(params["common_config"]["convergence_threshold"]),
    )
    payload["artifacts"] = {**payload.get("artifacts", {}), **viz_artifacts}
    payload["visualization_artifacts"] = sorted(viz_artifacts)
    return payload


__all__ = [
    "CONVEX_EQ_2D_ALGORITHMS",
    "EXPECTED_BOUNDARY_OPTIMUM",
    "TOY_PROBLEM_OVERRIDES",
    "convex_eq_2d_benchmark",
]

