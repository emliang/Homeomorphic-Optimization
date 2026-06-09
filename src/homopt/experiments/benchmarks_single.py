"""Single-problem benchmark entrypoints."""

from __future__ import annotations

import json
import numpy as np
import torch
import warnings
from pathlib import Path
from time import perf_counter

from ._benchmark_common import (
    apply_param_overrides,
    apply_alm_common_configs,
    artifact_mapping,
    artifact_ref,
    artifact_root,
    as_builtin,
    base_common_params,
    build_benchmark_payload,
    build_convex_penalty_method_kwargs,
    build_convex_problem,
    build_convex_problem_config,
    build_eq_cvxpy_baseline_params,
    build_first_order_method_params,
    build_hom_penalty_method_params,
    build_penalty_method_params,
    build_visualize_only_benchmark_payload,
    EQ_CVXPY_BASELINE_SPECS,
    convex_solution_diagnostics,
    constraint_violation_summary,
    ensure_int_list,
    load_incremental_comparison_artifacts,
    make_single_instance_solver_row,
    merged,
    normalize_convex_problem_type,
    normalize_jcc_lagrangian_config,
    normalize_jcc_problem_config,
    normalize_jcc_solver_configs,
    normalize_single_common_config,
    normalize_single_problem_config,
    prepare_convex_reference_context,
    resolve_runtime,
    save_json,
    save_incremental_comparison_artifacts,
    save_numpy,
    summarize_single_problem_run_record,
    summarize_run_record,
    write_csv,
)
from ._comparison_helpers import build_comparison_views, summarize_comparison_rows_by_method
from ._jcc_helpers import (
    build_jcc_comparison_metrics,
    build_jcc_iterative_method_rows,
    build_jcc_solver_method_row,
    build_jcc_solver_record,
)
from ._inn_baselines import normalize_inn_lagrangian_baselines, run_inn_lagrangian_baseline
from ._naming import labeled_artifact_prefix
from homopt.mappings import GaugeMap, GaugeMapMaxCut, PolyStarMap
from homopt.mappings.star import StarMap
from homopt.models import INNPGDOptimizer
from homopt.experiments._benchmark_parametric_common import (
    default_qcqp_inn_args,
    default_qcqp_optimizer_config,
    load_or_train_inn_mapping,
)
from homopt.optim import run_algorithm
from homopt.problems import BMSDP, ConvexOpt, JCCDCOPFProblem, MaxCutSDP, PolyStarOpt, ToyStarOpt, bind_singleton_problem_instance, create_maxcut_problem, create_test_problem
from homopt.solvers import ConvexSolver, JCCDCOPFCVaRSolver, JCCDCOPFRobustScenarioSolver, JCCDCOPFSolver, MaxCutSolver, solve_exact_result
from homopt.utils import cast_tensors_to_dtype, set_global_seed
from homopt.viz import save_comparison_visualizations, save_convex_2d_visualizations


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
    common_config=None,
):
    common = base_common_params(seed, max_iterations, max_running_time, learning_rate, stepsize_rule, lr_decay)
    common = merged(common, common_config)
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
                outer_iterations=max_iterations,
                inner_iterations=10,
                dual_learning_rate=5e-3,
                proximal_space="x",
            ),
        ),
    }


def _default_convex_algorithms(problem_type):
    if problem_type == "socp_eq":
        return ["Penalty", "Prox-Penalty", "ALM", "Prox-ALM", *EQ_CVXPY_BASELINE_SPECS, "Hom-ALM", "Prox-Hom-ALM"]
    return ["PGD", "FW", "RD", "Hom-PGD"]


def _origin_polytope_config(poly_config, *, seed):
    config = dict(poly_config or {})
    n_linear = int(config.get("n_linear_cons", 6))
    x_lower = float(config.get("x_lower", -2.0))
    x_upper = float(config.get("x_upper", 2.0))
    if x_lower >= 0 or x_upper <= 0:
        raise ValueError("poly_star intersection requires poly_config bounds to contain the origin.")
    radius = float(config.get("poly_radius", 1.0))
    radius_jitter = float(config.get("poly_radius_jitter", 0.0))
    if radius <= 0:
        raise ValueError("poly_radius must be positive.")
    if radius_jitter < 0:
        raise ValueError("poly_radius_jitter must be nonnegative.")

    rng = np.random.RandomState(config.get("seed", seed))
    if n_linear > 0:
        angle_offset = float(config.get("angle_offset", rng.uniform(0.0, 2.0 * np.pi / max(n_linear, 1))))
        angles = angle_offset + np.linspace(0.0, 2.0 * np.pi, n_linear, endpoint=False)
        a_matrix = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        jitter = rng.uniform(-radius_jitter, radius_jitter, size=n_linear)
        b_vector = np.maximum(radius * (1.0 + jitter), 1e-6)
    else:
        a_matrix = None
        b_vector = None

    return {
        "Q": np.eye(2),
        "p": np.zeros(2),
        "U": np.full(2, x_upper),
        "L": np.full(2, x_lower),
        "A": a_matrix,
        "b": b_vector,
        "Qq": None,
        "pq": None,
        "bq": None,
        "G": None,
        "h": None,
        "C": None,
        "d": None,
    }


def poly_star_benchmark(
    algorithms=None,
    common_overrides=None,
    algorithm_params=None,
    common_config=None,
    common_config_overrides=None,
    algorithm_config=None,
    algorithm_config_overrides=None,
    alpha=0.3,
    num_star=4,
    problem_type="star",
    poly_config=None,
    max_iterations=25,
    max_running_time=5,
    learning_rate=1e-2,
    stepsize_rule="constant",
    lr_decay=0.999,
    seed=2025,
    output_dir=None,
    device=None,
    dtype=None,
    visualize=True,
    visualize_only=False,
    show_constraint_notation=False,
):
    """Small package-native toy benchmark for polytope, star, or intersection sets."""

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    problem_type = str(problem_type).lower()
    if problem_type not in {"poly", "star", "poly_star"}:
        raise ValueError("problem_type must be 'poly', 'star', or 'poly_star'")
    algorithms = list(algorithms or ["Hom-PGD"])
    effective_common_config = normalize_single_common_config(
        common_overrides=common_overrides,
        common_config=common_config,
        common_config_overrides=common_config_overrides,
    )
    params = _ineq_algorithm_params(
        seed,
        max_iterations,
        max_running_time,
        learning_rate,
        stepsize_rule,
        lr_decay,
        common_config=effective_common_config,
    )
    params = apply_param_overrides(
        params,
        algorithm_params=algorithm_params,
        algorithm_config=algorithm_config,
        algorithm_config_overrides=algorithm_config_overrides,
    )
    hom_p_norm = params.get("Hom-PGD", {}).get("hom_p_norm", 2)
    if problem_type == "star":
        problem = cast_tensors_to_dtype(
            ToyStarOpt(alpha=alpha, num_star=num_star).to_device(runtime_device),
            runtime_dtype,
        )
        hom_map = cast_tensors_to_dtype(StarMap(problem, p_norm=hom_p_norm).to_device(runtime_device), runtime_dtype)
    elif problem_type == "poly_star":
        poly_problem = cast_tensors_to_dtype(
            ConvexOpt(_origin_polytope_config(poly_config, seed=seed)).to_device(runtime_device),
            runtime_dtype,
        )
        star_problem = cast_tensors_to_dtype(
            ToyStarOpt(alpha=alpha, num_star=num_star).to_device(runtime_device),
            runtime_dtype,
        )
        problem = cast_tensors_to_dtype(PolyStarOpt(poly_problem, star_problem).to_device(runtime_device), runtime_dtype)
        poly_map = cast_tensors_to_dtype(
            GaugeMap(poly_problem, p_norm=hom_p_norm, x_origin=np.zeros(2)).to_device(runtime_device),
            runtime_dtype,
        )
        star_map = cast_tensors_to_dtype(
            StarMap(star_problem, p_norm=hom_p_norm).to_device(runtime_device),
            runtime_dtype,
        )
        hom_map = cast_tensors_to_dtype(
            PolyStarMap(poly_map, star_map, p_norm=hom_p_norm).to_device(runtime_device),
            runtime_dtype,
        )
    else:
        default_poly_config = {
            "obj": "quad",
            "seed": seed,
            "n_var": 2,
            "n_linear_cons": 4,
            "n_soc_cons": 0,
            "n_qua_cons": 0,
            "x_lower": -2.0,
            "x_upper": 2.0,
            "margin_scale": 0.2,
        }
        config = create_test_problem(merged(default_poly_config, poly_config or {}))
        problem = cast_tensors_to_dtype(ConvexOpt(config).to_device(runtime_device), runtime_dtype)
        x_origin = ConvexSolver(problem.prob_para).solve("central_ip")
        hom_map = cast_tensors_to_dtype(
            GaugeMap(problem, p_norm=hom_p_norm, x_origin=x_origin).to_device(runtime_device),
            runtime_dtype,
        )

    existing_artifacts = {}
    if visualize_only:
        records, summaries, previous_result, _ = load_incremental_comparison_artifacts(
            output_dir,
            algorithms=algorithms,
        )
        existing_artifacts = dict(previous_result.get("artifacts", {}) or {})
    else:
        records = {}
        summaries = {}
        for algorithm in algorithms:
            result = run_algorithm(algorithm, problem, params, hom_map=hom_map, init_point=[-1.0, -1.0])
            records[algorithm] = result
            summaries[algorithm] = summarize_run_record(result, include_total_iter_time=True)

    artifacts, _, records, summaries = save_incremental_comparison_artifacts(
        output_dir,
        records=records,
        summaries=summaries,
        algorithm_order=algorithms,
        manifest_metadata={"problem_family": f"toy_{problem_type}"},
    )
    artifacts = {**existing_artifacts, **artifacts}

    if visualize and output_dir is not None:
        viz_artifacts = save_convex_2d_visualizations(
            problem=problem,
            records=records,
            algorithms=algorithms,
            output_dir=output_dir,
            hom_map=hom_map,
            payload={},
            prefix=f"toy_{problem_type}",
            include_individual=True,
            show_constraint_notation=show_constraint_notation,
            violation_y_min=float(params["common"]["convergence_threshold"]),
        )
        artifacts = {**artifacts, **viz_artifacts}

    primary = algorithms[0]
    return build_benchmark_payload(
        objective=summaries[primary]["final_objective"],
        feasible=summaries[primary]["final_violation"] <= 1e-5,
        artifacts=artifacts,
        algorithms=list(records.keys()),
        benchmarks=summaries,
        nvar=int(problem.nvar),
        problem=f"toy_{problem_type}",
        seed=seed,
    )


def socp_hompgd_benchmark(
    common_overrides=None,
    problem_overrides=None,
    algorithm_params=None,
    common_config=None,
    common_config_overrides=None,
    problem_config=None,
    problem_config_overrides=None,
    algorithm_config=None,
    algorithm_config_overrides=None,
    seed=2025,
    n_var=4,
    n_soc_cons=1,
    n_linear_cons=0,
    n_qua_cons=0,
    x_lower=-2.0,
    x_upper=2.0,
    margin_scale=0.1,
    anchor_interior_ratio=0.8,
    objective_quadratic_type="diagonal",
    constraint_quadratic_type="diagonal",
    quad_diag_lower=1e-2,
    quad_diag_upper=1.0,
    low_rank_quad_ridge=1e-2,
    max_iterations=50,
    max_running_time=10,
    learning_rate=1e-2,
    stepsize_rule="constant",
    lr_decay=0.999,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Package-native Hom-PGD smoke for convex problems with SOC constraints."""

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    effective_problem_config = normalize_single_problem_config(
        problem_overrides=problem_overrides,
        problem_config=problem_config,
        problem_config_overrides=problem_config_overrides,
    )
    config = build_convex_problem_config(
        seed=seed,
        n_var=n_var,
        n_linear_cons=n_linear_cons,
        n_soc_cons=n_soc_cons,
        n_qua_cons=n_qua_cons,
        obj="quad",
        x_lower=x_lower,
        x_upper=x_upper,
        margin_scale=margin_scale,
        anchor_interior_ratio=anchor_interior_ratio,
        objective_quadratic_type=objective_quadratic_type,
        constraint_quadratic_type=constraint_quadratic_type,
        quad_diag_lower=quad_diag_lower,
        quad_diag_upper=quad_diag_upper,
        low_rank_quad_ridge=low_rank_quad_ridge,
        problem_overrides=effective_problem_config,
    )
    problem = build_convex_problem(
        config=config,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        problem_type="socp",
    )
    params = _ineq_algorithm_params(
        seed=seed,
        max_iterations=max_iterations,
        max_running_time=max_running_time,
        learning_rate=learning_rate,
        stepsize_rule=stepsize_rule,
        lr_decay=lr_decay,
        common_config=normalize_single_common_config(
            common_overrides=common_overrides,
            common_config=common_config,
            common_config_overrides=common_config_overrides,
        ),
    )
    params = apply_param_overrides(
        params,
        problem_config=effective_problem_config,
        algorithm_params=algorithm_params,
        algorithm_config=algorithm_config,
        algorithm_config_overrides=algorithm_config_overrides,
    )
    reference = prepare_convex_reference_context(
        problem,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        hom_p_norm=params.get("Hom-PGD", {}).get("hom_p_norm", 2),
        hom_map_explicit_gradient_rule=params.get("common", {}).get("hom_map_explicit_gradient_rule", "polynomial"),
        hom_map_smooth_tie_tol=params.get("common", {}).get("hom_map_smooth_tie_tol", 1e-7),
        hom_map_smooth_temperature=params.get("common", {}).get("hom_map_smooth_temperature", 1e-4),
        need_opt=False,
        ip_mode="central_ip",
    )
    result = run_algorithm("Hom-PGD", problem, params, hom_map=reference["hom_map"], init_point=[-1.0] * problem.nvar)

    summary = {
        **summarize_single_problem_run_record(problem, result, include_total_iter_time=True),
        "n_soc_cons": int(n_soc_cons),
        "n_var": int(problem.nvar),
    }
    final_objective = summary["final_objective"]
    final_violation = summary["final_violation"]
    records = {"Hom-PGD": result}
    summaries = {"Hom-PGD": summary}
    artifacts, comparison_views, _, _ = save_incremental_comparison_artifacts(
        output_dir,
        records=records,
        summaries=summaries,
        algorithm_order=records.keys(),
    )

    return build_benchmark_payload(
        objective=final_objective,
        feasible=final_violation <= 1e-5,
        artifacts=artifacts,
        algorithm="Hom-PGD",
        algorithms=["Hom-PGD"],
        results=summaries,
        final_violation=final_violation,
        final_full_violation=summary.get("final_full_violation"),
        final_inequality_violation=summary.get("final_inequality_violation"),
        iterations=int(len(result["iter_time"])),
        n_soc_cons=int(n_soc_cons),
        n_var=int(problem.nvar),
        instance_rows=comparison_views["instance_rows"],
        method_summary_rows=comparison_views["method_summary_rows"],
        comparison_rows=comparison_views["comparison_rows"],
        comparison_summary=comparison_views["comparison_summary"],
        comparison_summary_rows=comparison_views["comparison_summary_rows"],
        best_method=comparison_views["best_method"],
        seed=seed,
    )


def convex_algorithm_comparison(
    algorithms=None,
    common_overrides=None,
    problem_overrides=None,
    outer_common=None,
    inner_solver_common=None,
    algorithm_params=None,
    common_config=None,
    common_config_overrides=None,
    problem_config=None,
    problem_config_overrides=None,
    algorithm_config=None,
    algorithm_config_overrides=None,
    problem_type="socp",
    seed=2025,
    expr_id=0,
    n_var=10,
    n_linear_cons=1,
    n_soc_cons=1,
    n_qua_cons=0,
    n_lin_eq=0,
    x_lower=-5.0,
    x_upper=5.0,
    margin_scale=0.1,
    anchor_interior_ratio=0.8,
    objective_quadratic_type="diagonal",
    constraint_quadratic_type="diagonal",
    quad_diag_lower=1e-2,
    quad_diag_upper=1.0,
    low_rank_quad_ridge=1e-2,
    obj="quad",
    max_iterations=200,
    max_running_time=30,
    learning_rate=1e-2,
    stepsize_rule="adaptive",
    lr_decay=0.999,
    warm_start=False,
    acceleration_method="none",
    acceleration_space="z",
    smooth=False,
    momentum=0.0,
    verbose=False,
    verbose_interval=50,
    include_reference_solver=None,
    reference_need_opt=True,
    reference_cache=True,
    hom_origin_constraints="full",
    hom_origin_ip_mode="ip",
    hom_origin_ip_eps=1e-3,
    hom_origin=None,
    output_dir=None,
    device=None,
    dtype=None,
    visualize=False,
    visualize_only=False,
    visualization_prefix=None,
    plot_algorithm_order=None,
    method_labels=None,
    reference_label="MOSEK",
    show_convergence_legend=True,
    scale_label=None,
):
    """Package-owned convex comparison runner for SOCP and SOCP-EQ instances."""

    del expr_id
    requested_algorithm_filter = None if algorithms is None else list(algorithms)
    problem_type = normalize_convex_problem_type(problem_type)
    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    effective_problem_config = normalize_single_problem_config(
        problem_overrides=problem_overrides,
        problem_config=problem_config,
        problem_config_overrides=problem_config_overrides,
    )
    is_eq = problem_type == "socp_eq"
    has_override_equality = bool(
        effective_problem_config
        and effective_problem_config.get("A_eq") is not None
        and effective_problem_config.get("b_eq") is not None
    )
    if is_eq and int(n_lin_eq or 0) <= 0 and not has_override_equality:
        n_lin_eq = 1
    config = build_convex_problem_config(
        seed=seed,
        n_var=n_var,
        n_linear_cons=n_linear_cons,
        n_soc_cons=n_soc_cons,
        n_qua_cons=n_qua_cons,
        n_lin_eq=n_lin_eq,
        obj=obj,
        x_lower=x_lower,
        x_upper=x_upper,
        margin_scale=margin_scale,
        anchor_interior_ratio=anchor_interior_ratio,
        objective_quadratic_type=objective_quadratic_type,
        constraint_quadratic_type=constraint_quadratic_type,
        quad_diag_lower=quad_diag_lower,
        quad_diag_upper=quad_diag_upper,
        low_rank_quad_ridge=low_rank_quad_ridge,
        problem_overrides=effective_problem_config,
    )
    problem = build_convex_problem(
        config=config,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        problem_type=problem_type,
    )
    problem = bind_singleton_problem_instance(problem, seed=seed)

    common = base_common_params(seed, max_iterations, max_running_time, learning_rate, stepsize_rule, lr_decay)
    common = merged(
        {
            **common,
            "warm_start": warm_start,
            "acceleration_method": acceleration_method,
            "acceleration_space": acceleration_space,
            "smooth": smooth,
            "momentum": momentum,
            "verbose": verbose,
            "verbose_interval": verbose_interval,
        },
        normalize_single_common_config(
            common_overrides=common_overrides,
            common_config=common_config,
            common_config_overrides=common_config_overrides,
        ),
    )
    effective_warm_start = bool(common.get("warm_start", warm_start))
    effective_smooth = bool(common.get("smooth", smooth))
    effective_visualize = bool(common.get("visualize", visualize))
    params = {
        "prob": {
            "obj": obj,
            "seed": seed,
            "n_var": n_var,
            "n_linear_cons": n_linear_cons,
            "n_qua_cons": n_qua_cons,
            "n_soc_cons": n_soc_cons,
            "n_lin_eq": n_lin_eq,
            "x_lower": x_lower,
            "x_upper": x_upper,
            "margin_scale": margin_scale,
            "anchor_interior_ratio": anchor_interior_ratio,
            "objective_quadratic_type": objective_quadratic_type,
            "constraint_quadratic_type": constraint_quadratic_type,
            "quad_diag_lower": quad_diag_lower,
            "quad_diag_upper": quad_diag_upper,
            "low_rank_quad_ridge": low_rank_quad_ridge,
        },
        "common": common,
        **build_first_order_method_params(
            common,
            projection_outer_iterations=10,
            projection_inner_iterations=100,
            linearization_outer_iterations=10,
            linearization_inner_iterations=100,
            hom_p_norm=2,
        ),
        "Penalty": build_penalty_method_params(
            common,
            **build_convex_penalty_method_kwargs(
                common,
                outer_iterations=max_iterations,
                dual_learning_rate=1e-1 if is_eq else 5e-3,
                proximal_space="x",
                use_lagrangian=False,
                use_penalty=True,
            ),
        ),
        "Prox-Penalty": build_penalty_method_params(
            common,
            **build_convex_penalty_method_kwargs(
                common,
                outer_iterations=max_iterations,
                dual_learning_rate=1e-1 if is_eq else 5e-3,
                proximal_space="x",
                use_lagrangian=False,
                use_penalty=True,
                use_proximal=True,
            ),
        ),
        "ALM": build_penalty_method_params(
            common,
            **build_convex_penalty_method_kwargs(
                common,
                outer_iterations=max_iterations,
                dual_learning_rate=1e-1 if is_eq else 5e-3,
                proximal_space="x",
            ),
        ),
        "Prox-ALM": build_penalty_method_params(
            common,
            **build_convex_penalty_method_kwargs(
                common,
                outer_iterations=max_iterations,
                dual_learning_rate=1e-1 if is_eq else 5e-3,
                proximal_space="x",
                use_proximal=True,
            ),
        ),
        **build_eq_cvxpy_baseline_params(common, outer_iterations=max_iterations),
        "Hom-ALM": build_hom_penalty_method_params(
            common,
            **build_convex_penalty_method_kwargs(
                common,
                outer_iterations=max_iterations,
                dual_learning_rate=1e-1,
                proximal_space="z",
            ),
        ),
        "Prox-Hom-ALM": build_hom_penalty_method_params(
            common,
            **build_convex_penalty_method_kwargs(
                common,
                outer_iterations=max_iterations,
                dual_learning_rate=1e-1,
                proximal_space="z",
                use_proximal=True,
            ),
        ),
    }
    params = apply_alm_common_configs(
        params,
        outer_common=outer_common,
        inner_solver_common=inner_solver_common,
    )
    params = apply_param_overrides(
        params,
        problem_config=effective_problem_config,
        algorithm_params=algorithm_params,
        algorithm_config=algorithm_config,
        algorithm_config_overrides=algorithm_config_overrides,
    )

    if algorithms is None:
        algorithms = _default_convex_algorithms(problem_type)
    algorithms = list(algorithms)
    effective_visualization_prefix = labeled_artifact_prefix(
        problem_type,
        {"scale_label": scale_label},
        explicit_prefix=visualization_prefix,
    )

    if bool(visualize_only):
        records, summaries, previous_result, manifest = load_incremental_comparison_artifacts(
            output_dir,
            algorithms=requested_algorithm_filter,
        )
        previous_metrics = dict(previous_result.get("metrics", {}) or {})
        previous_artifacts = dict(previous_result.get("artifacts", {}) or {})
        previous_algorithms = list(records.keys()) or list(manifest.get("algorithms") or algorithms)
        if plot_algorithm_order is None:
            visualize_algorithms = previous_algorithms
        else:
            visualize_algorithms = [
                algorithm for algorithm in plot_algorithm_order if algorithm in previous_algorithms
            ]
            visualize_algorithms.extend(
                algorithm for algorithm in previous_algorithms if algorithm not in visualize_algorithms
            )
        artifacts = dict(previous_artifacts)
        if effective_visualize and output_dir is not None:
            artifacts.update(
                save_comparison_visualizations(
                    problem=problem,
                    records=records,
                    algorithms=visualize_algorithms,
                    output_dir=output_dir,
                    prefix=effective_visualization_prefix,
                    reference_objective=previous_metrics.get("reference_objective"),
                    reference_label=reference_label,
                    method_labels=method_labels,
                    violation_y_min=float(common["convergence_threshold"]),
                    show_convergence_legend=show_convergence_legend,
                )
            )
        return build_visualize_only_benchmark_payload(
            previous_result=previous_result,
            summaries=summaries,
            artifacts=artifacts,
            requested_algorithms=algorithms,
            algorithms=previous_algorithms,
        )

    if include_reference_solver is True:
        reference_need_opt = True
    if verbose and reference_need_opt:
        print("Preparing ConvexSolver reference")
    reference_context_start = perf_counter()
    reference = prepare_convex_reference_context(
        problem,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        hom_p_norm=common["hom_p_norm"],
        smooth=effective_smooth,
        hom_map_explicit_gradient_rule=common.get("hom_map_explicit_gradient_rule", "polynomial"),
        hom_map_smooth_tie_tol=common.get("hom_map_smooth_tie_tol", 1e-7),
        hom_map_smooth_temperature=common.get("hom_map_smooth_temperature", 1e-4),
        need_opt=reference_need_opt,
        ip_mode=hom_origin_ip_mode,
        ip_eps=hom_origin_ip_eps,
        hom_origin_constraints=hom_origin_constraints,
        hom_origin=hom_origin,
        output_dir=output_dir,
        cache_reference=reference_cache,
    )
    reference_context_time = perf_counter() - reference_context_start
    if verbose and reference_need_opt:
        if reference.get("reference_cache_hit", False):
            print("Loaded ConvexSolver reference from cache")
        else:
            print("Computed ConvexSolver reference")
    obj_opt = reference["objective_opt"]
    solver_time = reference["solver_time"]
    solver_violation = reference["solver_violation"]
    ip_solver_time = reference["ip_solver_time"]
    origin_method = reference.get("origin_method", str(hom_origin_ip_mode))
    init_point = reference["x_origin"] if effective_warm_start else None

    records = {}
    summaries = {}
    for algorithm in algorithms:
        record = run_algorithm(algorithm, problem, params, reference["hom_map"], init_point)
        if str(algorithm).startswith("Hom-") or str(algorithm).startswith("Prox-Hom-"):
            record["initial_ip_time"] = ip_solver_time
        records[algorithm] = record
        summaries[algorithm] = summarize_single_problem_run_record(
            problem,
            record,
            reference_objective=obj_opt,
            include_total_iter_time=True,
        )

    artifacts, comparison_views, records, summaries = save_incremental_comparison_artifacts(
        output_dir,
        records=records,
        summaries=summaries,
        algorithm_order=algorithms,
        manifest_metadata={
            "problem_type": problem_type,
            "scale_label": scale_label,
            "reference_label": reference_label,
            "reference_objective": obj_opt,
        },
    )
    reference_cache_path = reference.get("reference_cache_path")
    if reference_cache_path is not None:
        artifacts["reference_cache"] = artifact_ref(output_dir, reference_cache_path)
    visualization_time = 0.0
    if effective_visualize and output_dir is not None:
        visualization_start = perf_counter()
        stored_algorithms = list(records.keys())
        if plot_algorithm_order is None:
            visualize_algorithms = stored_algorithms
        else:
            visualize_algorithms = [
                algorithm for algorithm in plot_algorithm_order if algorithm in stored_algorithms
            ]
            visualize_algorithms.extend(
                algorithm for algorithm in stored_algorithms if algorithm not in visualize_algorithms
            )
        artifacts.update(
            save_comparison_visualizations(
                problem=problem,
                records=records,
                algorithms=visualize_algorithms,
                output_dir=output_dir,
                prefix=effective_visualization_prefix,
                reference_objective=obj_opt,
                reference_label=reference_label,
                method_labels=method_labels,
                violation_y_min=float(common["convergence_threshold"]),
                show_convergence_legend=show_convergence_legend,
            )
        )
        visualization_time = perf_counter() - visualization_start
    if include_reference_solver is None:
        include_reference_solver = is_eq and reference_need_opt
    reference_solver_split = constraint_violation_summary(problem, reference["x_opt"])
    reference_solution_diagnostics = convex_solution_diagnostics(problem, reference["x_opt"])
    reference_solver_full_violation = reference_solver_split.get("final_full_violation", solver_violation)
    if include_reference_solver and obj_opt is not None:
        solver_row = make_single_instance_solver_row(
            method="ConvexSolver",
            objective=obj_opt,
            violation=reference_solver_full_violation,
            runtime_total=solver_time,
            total_wall_time=solver_time,
            status="optimal",
            full_violation=reference_solver_full_violation,
            equality_violation=reference_solver_split.get("final_equality_violation"),
            inequality_violation=reference_solver_split.get("final_inequality_violation"),
        )
        comparison_rows = [solver_row, *comparison_views["comparison_rows"]]
        comparison_summary_rows = summarize_comparison_rows_by_method(comparison_rows)
        comparison_views = build_comparison_views(comparison_rows, summary_rows=comparison_summary_rows)

    return build_benchmark_payload(
        objective=comparison_views["objective"],
        feasible=comparison_views["feasible"],
        artifacts=artifacts,
        algorithms=algorithms,
        plot_algorithm_order=plot_algorithm_order,
        method_labels=dict(method_labels or {}),
        reference_label=reference_label,
        scale_label=scale_label,
        problem_type=problem_type,
        reference_objective=obj_opt,
        reference_solver_time=solver_time,
        reference_solver_violation=reference_solver_full_violation,
        reference_need_opt=reference_need_opt,
        reference_solver_equality_violation=reference_solver_split.get("final_equality_violation"),
        reference_solver_inequality_violation=reference_solver_split.get("final_inequality_violation"),
        reference_solution_diagnostics=reference_solution_diagnostics,
        reference_ip_time=ip_solver_time,
        reference_context_time=reference_context_time,
        reference_opt_solver_time=solver_time,
        reference_origin_solver_time=ip_solver_time,
        reference_origin_time=ip_solver_time,
        reference_origin_method=origin_method,
        reference_cache_enabled=reference_cache,
        reference_cache_hit=reference.get("reference_cache_hit", False),
        reference_cache_path=artifact_ref(output_dir, reference_cache_path),
        reference_cache_load_time=reference.get("reference_cache_load_time", 0.0),
        reference_cached_opt_solver_time=reference.get("cached_solver_time"),
        reference_cached_origin_solver_time=reference.get("cached_ip_solver_time"),
        visualization_time=visualization_time,
        timing_breakdown={
            "algorithm_iteration_time_by_method": {
                name: summaries[name].get("total_iter_time")
                for name in algorithms
                if name in summaries
            },
            "algorithm_wall_time_by_method": {
                name: summaries[name].get("total_wall_time")
                for name in algorithms
                if name in summaries
            },
            "algorithm_initial_transform_time_by_method": {
                name: summaries[name].get("initial_transform_time")
                for name in algorithms
                if name in summaries and summaries[name].get("initial_transform_time") is not None
            },
            "algorithm_final_transform_time_by_method": {
                name: summaries[name].get("final_transform_time")
                for name in algorithms
                if name in summaries and summaries[name].get("final_transform_time") is not None
            },
            "cvxpy_reference_opt_time": solver_time,
            "cvxpy_reference_origin_time": ip_solver_time
            if origin_method in {"ip", "central_ip", "geometric_central_ip", "analytical_ip"}
            else None,
            "reference_origin_method": origin_method,
            "reference_origin_time": ip_solver_time,
            "reference_cache_load_time": reference.get("reference_cache_load_time", 0.0),
            "reference_context_time": reference_context_time,
            "visualization_time": visualization_time,
        },
        reference_origin=np.asarray(reference["x_origin"]).reshape(-1).tolist() if reference["x_origin"] is not None else None,
        hom_origin_constraints=reference["hom_origin_constraints"],
        reference_origin_equality_violation=reference["x_origin_eq_violation"],
        results=summaries,
        instance_rows=comparison_views["instance_rows"],
        method_summary_rows=comparison_views["method_summary_rows"],
        comparison_rows=comparison_views["comparison_rows"],
        comparison_summary=comparison_views["comparison_summary"],
        comparison_summary_rows=comparison_views["comparison_summary_rows"],
        best_method=comparison_views["best_method"],
        seed=seed,
    )


def maxcut_algorithm_comparison(
    algorithms=None,
    common=None,
    common_overrides=None,
    algorithm_params=None,
    common_config=None,
    common_config_overrides=None,
    algorithm_config=None,
    algorithm_config_overrides=None,
    seed=2025,
    expr_id=0,
    n=10,
    alpha=0.5,
    max_iterations=200,
    max_running_time=30,
    output_dir=None,
    device=None,
    dtype=None,
    visualize=True,
    visualize_only=False,
    visualization_prefix="maxcut_sdp",
    method_labels=None,
    reference_label="SDP",
    show_convergence_legend=True,
):
    """Package-owned MaxCut-SDP comparison runner."""

    del expr_id
    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    algorithms = list(algorithms or ["PGD", "ALM-log", "ALM-bp", "RD", "Hom-PGD"])

    common_params = {
        "seed": seed,
        "learning_rate": 1e-3,
        "convergence_threshold": 1e-6,
        "max_iterations": max_iterations,
        "max_running_time": max_running_time,
        "opt": "gd",
        "stepsize_rule": "adaptive",
        "warm_start": False,
        "acceleration_method": "none",
        "acceleration_space": "z",
        "smooth": True,
        "momentum": 0.0,
        "lr_decay": 0.999,
    }
    common_params = normalize_single_common_config(
        common=common_params,
        common_overrides=common_overrides,
        common_config=normalize_single_common_config(
            common=common,
            common_config=common_config,
            common_config_overrides=common_config_overrides,
        ),
    )
    params = {
        "common": common_params,
        **build_first_order_method_params(
            common_params,
            projection_outer_iterations=10,
            projection_inner_iterations=100,
            include_fw=False,
            include_rd=True,
            hom_p_norm=2,
            hom_momentum=0.99,
        ),
        "ALM": build_penalty_method_params(
            common_params,
            outer_iterations=max_iterations,
            inner_iterations=100,
            dual_learning_rate=0.1,
            penalty_coef=10.0,
            penalty_growth=1.1,
            proximal_coef=0.1,
            proximal_space="x",
            outer_stepsize_rule=common_params["stepsize_rule"],
            inner_stepsize_rule="constant",
            max_penalty=1e2,
            max_dual=1e2,
            use_lagrangian=True,
            use_penalty=True,
            use_proximal=False,
        ),
    }
    params["Hom-PGD"]["hom_map_gradient"] = "autograd"
    params = apply_param_overrides(
        params,
        algorithm_params=algorithm_params,
        algorithm_config=algorithm_config,
        algorithm_config_overrides=algorithm_config_overrides,
    )

    config = create_maxcut_problem({"seed": seed, "n": n, "alpha": alpha})
    base_problem = cast_tensors_to_dtype(MaxCutSDP(config).to_device(runtime_device), runtime_dtype)
    base_problem = bind_singleton_problem_instance(base_problem, seed=seed)
    solver = MaxCutSolver(base_problem.prob_para)
    solver_result = solve_exact_result(solver, "opt")
    x_opt = solver_result["solution"]
    x_opt_tensor = torch.tensor(
        x_opt[base_problem.upper_triangle_index[:, 0], base_problem.upper_triangle_index[:, 1]],
        dtype=runtime_dtype,
    ).view(1, -1)
    obj_opt = float(base_problem.objective_x(x_opt_tensor).detach().cpu().view(-1)[0].item())
    hom_map = GaugeMapMaxCut(
        base_problem,
        p_norm=params["Hom-PGD"].get("hom_p_norm", 2),
        smooth=bool(common_params.get("smooth", True)),
    ).to_device(runtime_device)
    hom_map = cast_tensors_to_dtype(hom_map, runtime_dtype)

    if visualize_only:
        records, summaries, previous_result, manifest = load_incremental_comparison_artifacts(
            output_dir,
            algorithms=algorithms,
        )
        artifacts = dict(previous_result.get("artifacts", {}) or {})
        stored_algorithms = list(records.keys()) or list(manifest.get("algorithms") or algorithms)
        if visualize and output_dir is not None:
            artifacts.update(
                save_comparison_visualizations(
                    problem=base_problem,
                    records=records,
                    algorithms=stored_algorithms,
                    output_dir=output_dir,
                    prefix=visualization_prefix,
                    reference_objective=previous_result.get("metrics", {}).get("reference_objective", obj_opt),
                    reference_label=reference_label,
                    method_labels=method_labels,
                    violation_y_min=float(common_params["convergence_threshold"]),
                    show_convergence_legend=show_convergence_legend,
                )
            )
        return build_visualize_only_benchmark_payload(
            previous_result=previous_result,
            summaries=summaries,
            artifacts=artifacts,
            requested_algorithms=algorithms,
            algorithms=stored_algorithms,
        )

    def _run_maxcut_record(algorithm):
        if algorithm.startswith("ALM-"):
            rank_key = algorithm.split("-", 1)[1]
            if rank_key == "log":
                rank = int(np.ceil(np.log(base_problem.num_node)))
            elif rank_key == "bp":
                rank = int(np.ceil(np.sqrt(2 * base_problem.num_node)))
            else:
                rank = base_problem.num_node
            problem = cast_tensors_to_dtype(BMSDP(config, rank=rank).to_device(runtime_device), runtime_dtype)
            problem = bind_singleton_problem_instance(problem, seed=seed)
            init_point = np.random.randn(1, problem.nvar) / problem.num_node
            record = run_algorithm("ALM", problem, params, hom_map=hom_map, init_point=init_point)
            record["violation_scope"] = "equality"
            return record
        problem = cast_tensors_to_dtype(MaxCutSDP(config).to_device(runtime_device), runtime_dtype)
        problem = bind_singleton_problem_instance(problem, seed=seed)
        init_point = np.zeros((1, problem.nvar))
        record = run_algorithm(algorithm, problem, params, hom_map=hom_map, init_point=init_point)
        record["violation_scope"] = "inequality"
        return record

    records = {}
    summaries = {}
    for algorithm in algorithms:
        record = _run_maxcut_record(algorithm)
        records[algorithm] = record
        summaries[algorithm] = summarize_run_record(record, include_total_iter_time=True)

    artifacts, comparison_views, records, summaries = save_incremental_comparison_artifacts(
        output_dir,
        records=records,
        summaries=summaries,
        algorithm_order=records.keys(),
        manifest_metadata={
            "problem_type": "maxcut_sdp",
            "reference_label": reference_label,
            "reference_objective": obj_opt,
        },
    )
    if visualize and output_dir is not None:
        artifacts.update(
            save_comparison_visualizations(
                problem=base_problem,
                records=records,
                algorithms=algorithms,
                output_dir=output_dir,
                prefix=visualization_prefix,
                reference_objective=obj_opt,
                reference_label=reference_label,
                method_labels=method_labels,
                violation_y_min=float(common_params["convergence_threshold"]),
                show_convergence_legend=show_convergence_legend,
            )
        )

    return build_benchmark_payload(
        objective=comparison_views["objective"],
        feasible=comparison_views["feasible"],
        artifacts=artifacts,
        algorithms=algorithms,
        method_labels=dict(method_labels or {}),
        reference_label=reference_label,
        reference_objective=obj_opt,
        results=summaries,
        instance_rows=comparison_views["instance_rows"],
        method_summary_rows=comparison_views["method_summary_rows"],
        comparison_rows=comparison_views["comparison_rows"],
        comparison_summary=comparison_views["comparison_summary"],
        comparison_summary_rows=comparison_views["comparison_summary_rows"],
        best_method=comparison_views["best_method"],
        n=int(n),
        alpha=float(alpha),
        seed=seed,
    )


def _print_jcc_solver_row(row):
    obj = "None" if row["objective"] is None else f"{row['objective']:.8f}"
    feas = "None" if row["feasibility_rate"] is None else f"{row['feasibility_rate']:.4f}"
    print(
        f"  {row['solver']:<14}"
        f"| status={row['status']:<18}"
        f"| time={row['runtime_sec']:>8.3f}s"
        f"| obj={obj:>14}"
        f"| feas={feas:>7}"
        f"| chance_ok={str(row['chance_satisfied']):<5}"
    )


def _resolve_enabled_baselines(enabled_baselines):
    supported = ("mixed_integer", "cvar", "scenario")
    enabled = enabled_baselines if enabled_baselines is not None else supported
    if isinstance(enabled, str):
        enabled = [enabled]
    enabled = [str(name).strip().lower() for name in enabled]
    invalid = [name for name in enabled if name not in supported]
    if invalid:
        raise ValueError(f"Unsupported baseline names: {invalid}. Supported: {list(supported)}")
    return [name for name in supported if name in enabled]


def _jcc_solver_baseline_specs(
    *,
    mixed_integer_config=None,
    cvar_config=None,
    scenario_config=None,
):
    return {
        "mixed_integer": {
            "solver_factory": JCCDCOPFSolver,
            "solver_config": dict(mixed_integer_config or {}),
        },
        "cvar": {
            "solver_factory": JCCDCOPFCVaRSolver,
            "solver_config": dict(cvar_config or {}),
        },
        "scenario": {
            "solver_factory": JCCDCOPFRobustScenarioSolver,
            "solver_config": dict(scenario_config or {}),
        },
    }


def _run_jcc_baseline_solver(problem, solver_name, *, solver_factory, solver_config=None):
    solver = solver_factory(problem, solver_config=dict(solver_config or {}))
    result = solve_exact_result(solver)
    extras = result.get("extras", {})
    return {
        "num_bus": int(problem.n_bus),
        "n_scenarios": int(problem.n_scenarios),
        "solver": solver_name,
        "status": str(result.get("status")),
        "runtime_sec": float(result.get("runtime_total", float("nan"))),
        "objective_value": as_builtin(result.get("objective")),
        "objective": as_builtin(result.get("objective")),
        "feasibility_rate": as_builtin(extras.get("feasibility_rate")),
        "chance_satisfied": bool(
            result.get("feasible")
            if result.get("feasible") is not None
            else extras.get("chance_constraint_satisfied", False)
        ),
        "chance_constraint_satisfied": bool(
            result.get("feasible")
            if result.get("feasible") is not None
            else extras.get("chance_constraint_satisfied", False)
        ),
        "x_optimal": result.get("solution"),
    }


def _run_jcc_solver_suite(
    *,
    problem,
    enabled,
    baseline_specs,
    verbose,
):
    solver_results = {}
    records = []
    comparison_rows = []
    for baseline_name in enabled:
        spec = baseline_specs[baseline_name]
        result = _run_jcc_baseline_solver(
            problem,
            baseline_name,
            solver_factory=spec["solver_factory"],
            solver_config=spec["solver_config"],
        )
        solver_results[baseline_name] = result
        record = build_jcc_solver_record(result)
        records.append(record)
        comparison_rows.append(build_jcc_solver_method_row(problem, result))
        if verbose:
            _print_jcc_solver_row(record)
    return solver_results, records, comparison_rows


def _build_jcc_problem(*, num_bus, n_scenarios, epsilon, demand_std, seed, runtime_device, runtime_dtype):
    problem_cfg = {
        "n_scenarios": int(n_scenarios),
        "epsilon": float(epsilon),
        "demand_std": float(demand_std),
        "seed": int(seed),
    }
    problem = JCCDCOPFProblem(num_bus=int(num_bus), config=problem_cfg).to_device(runtime_device)
    problem = cast_tensors_to_dtype(problem, runtime_dtype)
    return bind_singleton_problem_instance(problem, seed=seed)


def jcc_baseline_solver_sweep(
    num_bus_list,
    n_scenarios,
    epsilon=0.1,
    demand_std=0.05,
    seed=2025,
    solver_configs=None,
    solver_config_overrides=None,
    enabled_baselines=None,
    output_dir=None,
    device=None,
    dtype=None,
    verbose=True,
):
    """Run solver-only JCC baseline sweeps with structured artifact output."""

    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    enabled = _resolve_enabled_baselines(enabled_baselines)
    n_scenarios_list = ensure_int_list(n_scenarios)
    records = []
    comparison_rows = []

    effective_config = {
        "num_bus_list": [int(v) for v in num_bus_list],
        "n_scenarios": n_scenarios_list,
        "epsilon": float(epsilon),
        "demand_std": float(demand_std),
        "seed": int(seed),
        "device": str(runtime_device),
        "dtype": str(runtime_dtype).replace("torch.", ""),
        "solver_configs": normalize_jcc_solver_configs(
            solver_configs=solver_configs,
            solver_config_overrides=solver_config_overrides,
        ),
        "enabled_baselines": enabled,
    }
    if verbose:
        print("\n" + "=" * 88)
        print("[jcc-baseline] Effective Config")
        print("=" * 88)
        print(json.dumps(effective_config, indent=2, ensure_ascii=False))

    solver_cfgs = effective_config["solver_configs"]
    for num_bus in num_bus_list:
        for scenario_count in n_scenarios_list:
            if verbose:
                print("\n" + "-" * 88)
                print(f"[jcc-baseline] Case: num_bus={int(num_bus)}, n_scenarios={int(scenario_count)}")
                print("-" * 88)
            problem = _build_jcc_problem(
                num_bus=num_bus,
                n_scenarios=scenario_count,
                epsilon=epsilon,
                demand_std=demand_std,
                seed=seed,
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
            )
            _, case_records, case_comparison_rows = _run_jcc_solver_suite(
                problem=problem,
                enabled=enabled,
                baseline_specs=_jcc_solver_baseline_specs(
                    mixed_integer_config=solver_cfgs["mixed_integer"],
                    cvar_config=solver_cfgs["cvar"],
                    scenario_config=solver_cfgs["scenario"],
                ),
                verbose=verbose,
            )
            records.extend(case_records)
            comparison_rows.extend(case_comparison_rows)
            if verbose:
                print("-" * 88)

    root = artifact_root(output_dir)
    artifacts = {}
    if root is not None:
        json_path = save_json(root / "baseline_metrics.json", records)
        csv_path = write_csv(
            root / "baseline_metrics.csv",
            records,
            fieldnames=[
                "num_bus",
                "n_scenarios",
                "solver",
                "status",
                "runtime_sec",
                "objective",
                "feasibility_rate",
                "chance_satisfied",
            ],
        )
    comparison_summary_rows = summarize_comparison_rows_by_method(comparison_rows)
    comparison_views = build_comparison_views(comparison_rows, summary_rows=comparison_summary_rows)
    if root is not None:
        comparison_json = save_json(root / "baseline_comparison_summary.json", comparison_summary_rows)
        comparison_csv = write_csv(
            root / "baseline_comparison_summary.csv",
            comparison_summary_rows,
            fieldnames=[
                "row_kind",
                "route",
                "method",
                "objective",
                "feasible",
                "violation",
                "runtime",
                "objective_mean",
                "feasibility_rate",
                "violation_mean",
                "runtime_mean",
                "runtime_total",
                "num_instances",
            ],
        )
        artifacts = {
            "baseline_metrics_json": artifact_ref(output_dir, json_path),
            "baseline_metrics_csv": artifact_ref(output_dir, csv_path),
            "comparison_summary_json": artifact_ref(output_dir, comparison_json),
            "comparison_summary_csv": artifact_ref(output_dir, comparison_csv),
        }

    return {
        "objective": comparison_views["objective"],
        "feasible": comparison_views["feasible"],
        "num_cases": int(len(list(num_bus_list)) * len(n_scenarios_list)),
        "num_records": int(len(records)),
        "enabled_baselines": enabled,
        "instance_rows": comparison_views["instance_rows"],
        "method_summary_rows": comparison_views["method_summary_rows"],
        "comparison_rows": comparison_views["comparison_rows"],
        "comparison_summary": comparison_views["comparison_summary"],
        "comparison_summary_rows": comparison_views["comparison_summary_rows"],
        "best_method": comparison_views["best_method"],
        "all_records_feasible": all(bool(row["chance_satisfied"]) for row in records),
        "artifacts": artifacts,
    }


class _JCCINNPGDAdapter:
    """Adapter exposing the shared INN-PGD training contract for JCC-DC-OPF."""

    def __init__(self, problem):
        self.problem = problem
        self.nvar = int(problem.nvar)
        self.n_bus = int(problem.n_bus)
        self.n_scenarios = int(problem.n_scenarios)
        self.npara = self.n_bus
        self.n_qua = self.n_scenarios
        self.device = getattr(problem, "device", torch.device("cpu"))
        self.dtype = torch.float32
        self._sample_counter = 0

    def to_device(self, device):
        self.problem.to_device(device)
        self.device = device
        return self

    def generate_problem_samples_torch(self, n_samples=1, sample_obj=False, device=None, dtype=None):
        del sample_obj
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        seed = int(self.problem.config.get("seed", 2025)) + self._sample_counter
        self._sample_counter += 1
        samples = self.problem.sample_instances(n_samples, seed=seed)
        return samples.to(device=device, dtype=dtype), None

    def sample_instance_batch(self, n_instances=1, seed=2025, **kwargs):
        return self.problem.sample_instance_batch(n_instances=n_instances, seed=seed, **kwargs)

    def scale(self, input_batch, x):
        del input_batch
        p_min = self.problem.P_min.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        p_max = self.problem.P_max.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        return (x + 1.0) * 0.5 * (p_max - p_min) + p_min

    def complete_partial(self, input_batch, x_scaled):
        del input_batch
        return x_scaled

    def ineq_resid(self, input_batch, x, clip=True):
        return self.problem.constraint_residual_xy(input_batch, x, clip=clip)

    def check_feasibility(self, input_batch, x):
        return self.ineq_resid(input_batch, x, clip=True)

    def violations(self, input_batch, x):
        return self.check_feasibility(input_batch, x)

    def objective(self, x):
        return self.problem.objective_x(x)


def _default_jcc_inn_model_config():
    return {
        **default_qcqp_inn_args(),
        "inv_type": "made",
        "bilip": False,
        "L": 2,
        "h_dim": 128,
        "w_penalty": 10.0,
        "w_distortion": 0.01,
        "c_samples": 10000,
        "lr_decay_step": 1000,
        "ema_decay": 0.9,
        "resultsSaveFreq": 2000,
    }


def _default_jcc_inn_train_config():
    return {
        "n_samples": 10000,
        "batch_size": 256,
        "total_iteration": 10000,
    }


def _default_jcc_inn_optimizer_config():
    return {
        **default_qcqp_optimizer_config(),
        "learning_rate": 0.1,
        "max_iterations": 300,
        "max_running_time": 600,
        "convergence_threshold": 1e-5,
        "lr_decay": 0.999,
        "step_size": 0.5,
        "proj_max_steps": 10,
        "feasibility_eps": 1e-5,
        "opt": "gd",
        "momentum": 0.0,
        "stepsize_rule": "adaptive",
    }


def _reject_jcc_model_training_keys(config, *, context):
    training_keys = {"n_samples", "batch_size", "total_iteration"}
    invalid = sorted(training_keys.intersection((config or {}).keys()))
    if invalid:
        raise ValueError(
            f"{context} contains training keys {invalid}; "
            "use train_config or train_config_overrides instead."
        )


def _normalize_jcc_inn_configs(
    *,
    model_config=None,
    model_config_overrides=None,
    optimizer_config=None,
    optimizer_config_overrides=None,
    train_config=None,
    train_config_overrides=None,
):
    _reject_jcc_model_training_keys(model_config, context="model_config")
    _reject_jcc_model_training_keys(model_config_overrides, context="model_config_overrides")
    model_cfg = merged(_default_jcc_inn_model_config(), model_config)
    model_cfg = merged(model_cfg, model_config_overrides)
    train_cfg = _default_jcc_inn_train_config()
    train_cfg = merged(train_cfg, train_config)
    train_cfg = merged(train_cfg, train_config_overrides)
    optimizer_cfg = merged(_default_jcc_inn_optimizer_config(), optimizer_config)
    optimizer_cfg = merged(optimizer_cfg, optimizer_config_overrides)
    model_payload = {
        **model_cfg,
        "n_samples": int(train_cfg["n_samples"]),
        "batch_size": int(train_cfg["batch_size"]),
        "total_iteration": int(train_cfg["total_iteration"]),
    }
    return model_payload, optimizer_cfg, train_cfg


def _jcc_inn_initial_latent(adapter, input_params, *, mode="center", radius=1.0, seed=2025):
    mode = str(mode).strip().lower()
    batch_size = int(input_params.shape[0])
    if mode == "center":
        return torch.zeros(batch_size, adapter.nvar, device=adapter.device, dtype=adapter.dtype)
    if mode == "sphere_boundary_random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        directions = torch.randn(batch_size, adapter.nvar, generator=generator, dtype=torch.float32)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (float(radius) * directions).to(device=adapter.device, dtype=adapter.dtype)
    raise ValueError(
        "Unsupported initial_latent_mode: "
        f"{mode}. Use 'center' or 'sphere_boundary_random'."
    )


def _run_jcc_inn_pgd_variant(
    problem,
    *,
    output_dir,
    seed,
    retrain,
    num_test_instance,
    model_config=None,
    model_config_overrides=None,
    optimizer_config=None,
    optimizer_config_overrides=None,
    train_config=None,
    train_config_overrides=None,
):
    adapter = _JCCINNPGDAdapter(problem)
    if adapter.nvar < 2:
        raise ValueError(
            "JCC INN-PGD requires at least two active decision variables for the current INN flow. "
            "Use a larger OPF case such as 57-bus or disable run_inn_pgd for 30-bus smoke runs."
        )
    model_cfg, optimizer_cfg, train_cfg = _normalize_jcc_inn_configs(
        model_config=model_config,
        model_config_overrides=model_config_overrides,
        optimizer_config=optimizer_config,
        optimizer_config_overrides=optimizer_config_overrides,
        train_config=train_config,
        train_config_overrides=train_config_overrides,
    )
    artifact_dir = artifact_root(output_dir)
    save_dir = (
        artifact_dir / "jcc_inn_pgd"
        if artifact_dir is not None
        else Path("results") / "_scratch" / "jcc_inn_pgd"
    )
    model_args = {"model_config": model_cfg}
    model, training_record, model_path, record_path = load_or_train_inn_mapping(
        adapter,
        model_args,
        save_dir,
        retrain=retrain,
    )
    instance_batch = adapter.sample_instance_batch(n_instances=int(num_test_instance), seed=seed)
    inputs = instance_batch.inputs.to(device=adapter.device, dtype=adapter.dtype)
    initial_latent = _jcc_inn_initial_latent(
        adapter,
        inputs,
        mode=optimizer_cfg.get("initial_latent_mode", "center"),
        radius=optimizer_cfg.get("initial_latent_radius", 1.0),
        seed=seed,
    )
    optimizer = INNPGDOptimizer(problem=adapter, paras=optimizer_cfg, model=model)
    run_start_time = perf_counter()
    x_opt, decision_traj, latent_traj, obj_traj, cons_traj, per_iter_time = optimizer.optimize(
        initial_point=initial_latent,
        input_params=inputs,
        seed=seed,
    )
    total_wall_time = perf_counter() - run_start_time
    final_objective = float(obj_traj.detach().cpu()[-1].reshape(-1).mean().item())
    final_violation = float(cons_traj.detach().cpu()[-1].reshape(-1).max().item())
    with torch.inference_mode():
        scenario_feasibility = problem.compute_scenario_feasibility(x_opt, scenario_batch=inputs)
        chance_feasibility_rate = float(scenario_feasibility.mean().item())
        violation = max(0.0, (1.0 - chance_feasibility_rate) - float(problem.config["epsilon"]))
    return {
        "x_opt": x_opt.detach().cpu().numpy(),
        "obj_traj": obj_traj.detach().cpu().numpy(),
        "cons_traj": cons_traj.detach().cpu().numpy(),
        "iter_time": list(per_iter_time),
        "total_wall_time": total_wall_time,
        "final_objective": final_objective,
        "final_violation": final_violation,
        "chance_feasibility_rate": chance_feasibility_rate,
        "violation": violation,
        "feasible": violation <= 1e-12,
        "decision_trajectory": decision_traj.detach().cpu().numpy() if hasattr(decision_traj, "detach") else [],
        "latent_trajectory": latent_traj.detach().cpu().numpy() if hasattr(latent_traj, "detach") else [],
        "input_params": inputs.detach().cpu().numpy(),
        "initial_latent_mode": str(optimizer_cfg.get("initial_latent_mode", "center")),
        "initial_latent_radius": float(optimizer_cfg.get("initial_latent_radius", 1.0)),
        "training_record_path": str(record_path),
        "model_path": str(model_path),
        "train_config": train_cfg,
    }


def _save_jcc_comparison_artifacts(output_dir, results, metrics, comparison_summary_rows=None):
    root = artifact_root(output_dir)
    if root is None:
        return {}
    summary_path = save_json(root / "jcc_compare_summary.json", metrics)
    comparison_json = comparison_csv = None
    if comparison_summary_rows is not None:
        comparison_json = save_json(root / "jcc_comparison_summary.json", comparison_summary_rows)
        comparison_csv = write_csv(
            root / "jcc_comparison_summary.csv",
            comparison_summary_rows,
            fieldnames=[
                "row_kind",
                "route",
                "method",
                "objective",
                "feasible",
                "violation",
                "runtime",
                "objective_mean",
                "feasibility_rate",
                "violation_mean",
                "runtime_mean",
                "runtime_total",
                "num_instances",
            ],
        )
    for name, payload in results.items():
        save_numpy(root / f"{name}_record.npy", payload)
    return artifact_mapping(
        output_dir,
        **{f"{name}_record": root / f"{name}_record.npy" for name in results},
        summary=summary_path,
        comparison_summary_json=comparison_json,
        comparison_summary_csv=comparison_csv,
    )


def jcc_algorithm_comparison(
    num_bus=30,
    n_scenarios=5,
    epsilon=0.1,
    demand_std=0.05,
    seed=2025,
    run_inn_pgd=False,
    retrain=False,
    num_test_instance=1,
    solver_configs=None,
    solver_config_overrides=None,
    lagrangian_baselines=None,
    lagrangian_baseline_config=None,
    lagrangian_baseline_config_overrides=None,
    model_config=None,
    model_config_overrides=None,
    optimizer_config=None,
    optimizer_config_overrides=None,
    train_config=None,
    train_config_overrides=None,
    problem_config=None,
    problem_config_overrides=None,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Package-owned JCC comparison using the current problem and optimizer stack."""

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    problem_cfg = normalize_jcc_problem_config(
        n_scenarios=n_scenarios,
        epsilon=epsilon,
        demand_std=demand_std,
        seed=seed,
        problem_config=problem_config,
        problem_config_overrides=problem_config_overrides,
    )
    solver_cfgs = normalize_jcc_solver_configs(
        solver_configs=solver_configs,
        solver_config_overrides=solver_config_overrides,
    )
    lag_args = normalize_jcc_lagrangian_config(
        lagrangian_baseline_config=lagrangian_baseline_config,
        lagrangian_baseline_config_overrides=lagrangian_baseline_config_overrides,
    )
    enabled_lagrangian_baselines = normalize_inn_lagrangian_baselines(
        ("ALM", "Penalty", "Prox-Penalty") if lagrangian_baselines is None else lagrangian_baselines
    )

    problem = _build_jcc_problem(
        num_bus=num_bus,
        n_scenarios=problem_cfg["n_scenarios"],
        epsilon=problem_cfg["epsilon"],
        demand_std=problem_cfg["demand_std"],
        seed=problem_cfg["seed"],
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
    )
    midpoint = ((problem.P_min + problem.P_max) / 2).view(1, -1)
    results = {}

    if run_inn_pgd:
        print("Preparing JCC INN-PGD")
        results["INN-PGD"] = _run_jcc_inn_pgd_variant(
            problem,
            output_dir=output_dir,
            seed=seed,
            retrain=retrain,
            num_test_instance=num_test_instance,
            model_config=model_config,
            model_config_overrides=model_config_overrides,
            optimizer_config=optimizer_config,
            optimizer_config_overrides=optimizer_config_overrides,
            train_config=train_config,
            train_config_overrides=train_config_overrides,
        )

    print("Running JCC exact solver baselines")
    solver_results, _, solver_comparison_rows = _run_jcc_solver_suite(
        problem=problem,
        enabled=["mixed_integer", "cvar", "scenario"],
        baseline_specs=_jcc_solver_baseline_specs(
            mixed_integer_config=solver_cfgs["mixed_integer"],
            cvar_config=solver_cfgs["cvar"],
            scenario_config=solver_cfgs["scenario"],
        ),
        verbose=False,
    )
    solver_result = solver_results["mixed_integer"]
    cvar_result = solver_results["cvar"]
    scenario_result = solver_results["scenario"]

    results.update({"solver": solver_result, "CVaR": cvar_result, "scenario": scenario_result})
    for method in enabled_lagrangian_baselines:
        print(f"Running JCC iterative baseline: {method}")
        results[method] = run_inn_lagrangian_baseline(problem, midpoint, lag_args, method=method, seed=seed)
    comparison_rows = [*solver_comparison_rows, *build_jcc_iterative_method_rows(problem, results)]
    comparison_summary_rows = summarize_comparison_rows_by_method(comparison_rows)
    comparison_views = build_comparison_views(comparison_rows, summary_rows=comparison_summary_rows)
    metrics = build_jcc_comparison_metrics(
        num_bus=num_bus,
        n_scenarios=problem.n_scenarios,
        solver_result=solver_result,
        cvar_result=cvar_result,
        scenario_result=scenario_result,
        results=results,
    )
    artifacts = _save_jcc_comparison_artifacts(output_dir, results, metrics, comparison_summary_rows)

    return build_benchmark_payload(
        objective=comparison_views["objective"],
        feasible=comparison_views["feasible"],
        artifacts=artifacts,
        instance_rows=comparison_views["instance_rows"],
        method_summary_rows=comparison_views["method_summary_rows"],
        comparison_rows=comparison_views["comparison_rows"],
        comparison_summary=comparison_views["comparison_summary"],
        comparison_summary_rows=comparison_views["comparison_summary_rows"],
        best_method=comparison_views["best_method"],
        **metrics,
    )


__all__ = [
    "SINGLE_PROBLEM_BENCHMARKS",
    "convex_algorithm_comparison",
    "jcc_algorithm_comparison",
    "jcc_baseline_solver_sweep",
    "maxcut_algorithm_comparison",
    "poly_star_benchmark",
    "socp_hompgd_benchmark",
]
