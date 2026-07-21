"""Convex inequality and equality single-problem benchmarks."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from time import perf_counter

from homopt.records.artifacts import (
    artifact_ref,
    build_benchmark_payload,
    build_visualize_only_benchmark_payload,
    load_visualize_only_comparison_artifacts,
    save_incremental_comparison_artifacts,
)
from homopt.experiments.common.comparison import build_comparison_views, summarize_comparison_rows_by_method
from homopt.experiments.common.config import (
    align_outer_iterations_with_max_iterations,
    apply_alm_common_configs,
    apply_config_groups,
    enforce_alm_outer_iteration_budget,
    merged,
    normalize_single_common_config,
    normalize_single_problem_config,
    reject_algorithm_iteration_budget_keys,
)
from homopt.experiments.method_specs import (
    EQ_CVXPY_BASELINE_SPECS,
    base_common_params,
    build_convex_penalty_method_kwargs,
    build_eq_cvxpy_baseline_params,
    build_first_order_method_params,
    build_hom_penalty_method_params,
    build_penalty_method_params,
)
from homopt.experiments.common.naming import labeled_artifact_prefix
from homopt.experiments.common.convex_reference import (
    build_convex_problem,
    build_convex_problem_config,
    normalize_convex_problem_type,
    prepare_convex_reference_context,
)
from homopt.experiments.common.reports import (
    convex_solution_diagnostics,
    constraint_violation_summary,
    make_single_instance_solver_row,
    summarize_single_problem_run_record,
)
from homopt.experiments.common.run_records import ensure_record_violation_split, summarize_run_record
from homopt.experiments.common.runtime import resolve_runtime
from homopt.experiments.common.single_problem import _ineq_algorithm_params
from homopt.optim import run_algorithm
from homopt.utils import set_global_seed
from homopt.viz import save_comparison_visualizations


def _default_convex_algorithms(problem_type):
    if problem_type == "socp_eq":
        return ["Penalty", "Prox-Penalty", "ALM", "Prox-ALM", *EQ_CVXPY_BASELINE_SPECS, "Hom-ALM", "Prox-Hom-ALM"]
    return ["PGD", "FW", "RD", "Hom-PGD"]


def socp_hompgd_benchmark(
    common_config=None,
    problem_config=None,
    algorithm_config=None,
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
    min_lr=1e-6,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Package-native Hom-PGD smoke for convex problems with SOC constraints."""

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    reject_algorithm_iteration_budget_keys(algorithm_config, context="algorithm_config")
    effective_problem_config = normalize_single_problem_config(
        problem_config=problem_config,
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
        explicit_config=effective_problem_config,
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
        min_lr=min_lr,
        common_config=normalize_single_common_config(
            common_config=common_config,
        ),
    )
    params = apply_config_groups(
        params,
        problem_config=effective_problem_config,
        algorithm_config=algorithm_config,
    )
    effective_max_iterations = int(params["common"]["max_iterations"])
    params = enforce_alm_outer_iteration_budget(params, max_iterations=effective_max_iterations)
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
    ensure_record_violation_split(problem, result)

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


def _resolve_single_problem_initial_point(problem, reference, mode, *, seed):
    """Return the explicit shared initial point for single-problem comparisons."""

    normalized = str(mode or "gauge_center").lower()
    if normalized == "gauge_center":
        return reference["x_origin"]
    if normalized == "random":
        rng = np.random.default_rng(seed)
        return rng.standard_normal((1, int(problem.nvar)))
    raise ValueError(f"Unsupported initial_point_mode: {mode}")


def convex_algorithm_comparison(
    algorithms=None,
    outer_common=None,
    inner_solver_common=None,
    common_config=None,
    problem_config=None,
    algorithm_config=None,
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
    min_lr=1e-6,
    initial_point_mode="gauge_center",
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
    reject_algorithm_iteration_budget_keys(algorithm_config, context="algorithm_config")
    effective_problem_config = normalize_single_problem_config(
        problem_config=problem_config,
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
        explicit_config=effective_problem_config,
    )
    problem = build_convex_problem(
        config=config,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        problem_type=problem_type,
    )

    common = base_common_params(seed, max_iterations, max_running_time, learning_rate, stepsize_rule, lr_decay, min_lr)
    common = merged(
        {
            **common,
            "initial_point_mode": initial_point_mode,
            "acceleration_method": acceleration_method,
            "acceleration_space": acceleration_space,
            "smooth": smooth,
            "momentum": momentum,
            "verbose": verbose,
            "verbose_interval": verbose_interval,
        },
        normalize_single_common_config(
            common_config=common_config,
        ),
    )
    effective_max_iterations = int(common["max_iterations"])
    effective_outer_common = align_outer_iterations_with_max_iterations(
        outer_common,
        max_iterations=effective_max_iterations,
    )
    enforce_alm_outer_iteration_budget(
        algorithm_config or {},
        max_iterations=effective_max_iterations,
        context="algorithm_config",
    )
    effective_initial_point_mode = common.get("initial_point_mode", initial_point_mode)
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
                outer_iterations=effective_max_iterations,
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
                outer_iterations=effective_max_iterations,
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
                outer_iterations=effective_max_iterations,
                dual_learning_rate=1e-1 if is_eq else 5e-3,
                proximal_space="x",
            ),
        ),
        "Prox-ALM": build_penalty_method_params(
            common,
            **build_convex_penalty_method_kwargs(
                common,
                outer_iterations=effective_max_iterations,
                dual_learning_rate=1e-1 if is_eq else 5e-3,
                proximal_space="x",
                use_proximal=True,
            ),
        ),
        **build_eq_cvxpy_baseline_params(common, outer_iterations=effective_max_iterations),
        "Hom-ALM": build_hom_penalty_method_params(
            common,
            **build_convex_penalty_method_kwargs(
                common,
                outer_iterations=effective_max_iterations,
                dual_learning_rate=1e-1,
                proximal_space="z",
            ),
        ),
        "Prox-Hom-ALM": build_hom_penalty_method_params(
            common,
            **build_convex_penalty_method_kwargs(
                common,
                outer_iterations=effective_max_iterations,
                dual_learning_rate=1e-1,
                proximal_space="z",
                use_proximal=True,
            ),
        ),
    }
    params = apply_alm_common_configs(
        params,
        outer_common=effective_outer_common,
        inner_solver_common=inner_solver_common,
    )
    params = apply_config_groups(
        params,
        problem_config=effective_problem_config,
        algorithm_config=algorithm_config,
    )
    params = enforce_alm_outer_iteration_budget(params, max_iterations=effective_max_iterations)

    if algorithms is None:
        algorithms = _default_convex_algorithms(problem_type)
    algorithms = list(algorithms)
    effective_visualization_prefix = labeled_artifact_prefix(
        problem_type,
        {"scale_label": scale_label},
        explicit_prefix=visualization_prefix,
    )

    if bool(visualize_only):
        visual_state = load_visualize_only_comparison_artifacts(
            output_dir,
            algorithms=requested_algorithm_filter,
            plot_algorithm_order=plot_algorithm_order,
        )
        records = visual_state["records"]
        summaries = visual_state["summaries"]
        previous_result = visual_state["previous_result"]
        previous_metrics = visual_state["metrics"]
        previous_algorithms = visual_state["stored_algorithms"]
        artifacts = visual_state["artifacts"]
        if effective_visualize and output_dir is not None:
            artifacts.update(
                save_comparison_visualizations(
                    problem=problem,
                    records=records,
                    algorithms=visual_state["plot_algorithms"],
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
    init_point = _resolve_single_problem_initial_point(
        problem,
        reference,
        effective_initial_point_mode,
        seed=seed,
    )

    records = {}
    summaries = {}
    for algorithm in algorithms:
        record = run_algorithm(algorithm, problem, params, reference["hom_map"], init_point)
        ensure_record_violation_split(problem, record)
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


__all__ = ["convex_algorithm_comparison", "socp_hompgd_benchmark"]
