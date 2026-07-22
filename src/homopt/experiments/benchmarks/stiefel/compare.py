"""Stiefel benchmark comparison runner."""

from __future__ import annotations

import copy
import warnings

import numpy as np

from homopt.records.artifacts import (
    artifact_ref,
    build_benchmark_payload,
    build_visualize_only_benchmark_payload,
    load_visualize_only_comparison_artifacts,
    save_incremental_comparison_artifacts,
)
from homopt.experiments.common.reports import (
    constraint_violation_summary,
    convex_solution_diagnostics,
    make_single_instance_solver_row,
    objective_summary,
    summarize_single_problem_run_record,
)
from homopt.experiments.common.comparison import build_comparison_views, summarize_comparison_rows_by_method
from homopt.experiments.common.config import (
    align_outer_iterations_with_max_iterations,
    enforce_alm_outer_iteration_budget,
    merged,
    normalize_algorithm_config_group,
    normalize_alm_common_config_groups,
    normalize_single_common_config,
)
from homopt.experiments.common.labels import STIEFEL_METHOD_LABELS
from homopt.experiments.common.naming import labeled_artifact_prefix, scale_label
from homopt.optim import (
    StiefelALMEQPGDOptimizer,
    StiefelRetractionALMOptimizer,
    StiefelRetractionOptimizer,
    run_algorithm,
)
from homopt.solvers import StiefelPyomoIPOPTSolver
from homopt.viz import save_comparison_visualizations

from .setup import (
    build_stiefel_hom_map,
    build_stiefel_initial_point,
    build_stiefel_problem,
    load_stiefel_reference_cache,
    save_stiefel_reference_cache,
    stiefel_reference_cache_key,
    stiefel_reference_cache_path,
)


STIEFEL_PLOT_ALGORITHM_ORDER = [
    "Penalty",
    "Prox-Penalty",
    "ALM",
    "Prox-ALM",
    "ALM-EQ-PGD",
    "StiefelRetraction",
    "StiefelRetractionPenalty",
    "StiefelRetractionALM",
    "Hom-ALM",
    "Prox-Hom-ALM",
]
STIEFEL_PLOT_LABELS = STIEFEL_METHOD_LABELS
STIEFEL_OPTIMIZER_ALGORITHMS = {
    "Penalty",
    "Prox-Penalty",
    "ALM",
    "Prox-ALM",
    "Hom-ALM",
    "Prox-Hom-ALM",
}
STIEFEL_HOM_OPTIMIZER_ALGORITHMS = {"Hom-ALM", "Prox-Hom-ALM"}


def _deepcopy_dict(value):
    return copy.deepcopy(value or {})


def _algorithm_config(params):
    return normalize_algorithm_config_group(
        algorithm_config=params.get("algorithm_config"),
    )


def _outer_config(params):
    common = normalize_single_common_config(common_config=params["common_config"])
    outer_defaults = {
        "learning_rate": common["learning_rate"],
        "outer_stepsize_rule": common["stepsize_rule"],
        "outer_lr_decay": common["lr_decay"],
        "min_lr": common["min_lr"],
    }
    outer_common = align_outer_iterations_with_max_iterations(
        params["outer_common"],
        max_iterations=params["max_iterations"],
    )
    outer, _ = normalize_alm_common_config_groups(
        outer_common=outer_common,
        inner_solver_common=params.get("inner_solver_common"),
    )
    return merged(outer_defaults, outer)


def _inner_config(params):
    _, inner = normalize_alm_common_config_groups(
        outer_common=align_outer_iterations_with_max_iterations(
            params["outer_common"],
            max_iterations=params["max_iterations"],
        ),
        inner_solver_common=params["inner_solver_common"],
    )
    return inner


def _visualization_prefix(params):
    return labeled_artifact_prefix("stiefel", params, explicit_prefix=params.get("visualization_prefix"))


def _plot_stiefel_records(problem, records, algorithms, output_dir, params, *, reference_objective=None):
    if not params.get("visualize", False) or output_dir is None:
        return {}
    plot_algorithms = [algorithm for algorithm in STIEFEL_PLOT_ALGORITHM_ORDER if algorithm in records]
    plot_algorithms.extend(algorithm for algorithm in algorithms if algorithm in records and algorithm not in plot_algorithms)
    return save_comparison_visualizations(
        problem=problem,
        records=records,
        algorithms=plot_algorithms,
        output_dir=output_dir,
        prefix=_visualization_prefix(params),
        reference_objective=reference_objective,
        reference_label="IPOPT",
        method_labels=STIEFEL_PLOT_LABELS,
        runtime_metric="outer_per_iter_time",
        violation_y_min=float(_outer_config(params)["convergence_threshold"]),
        show_convergence_legend=bool(params.get("show_convergence_legend", True)),
    )


def _build_alm_params(params):
    outer = _outer_config(params)
    inner = _inner_config(params)
    base = {
        "max_running_time": outer["max_running_time"],
        "outer_iterations": outer["outer_iterations"],
        "convergence_threshold": outer["convergence_threshold"],
        "learning_rate": outer["learning_rate"],
        "outer_stepsize_rule": outer["outer_stepsize_rule"],
        "outer_lr_decay": outer["outer_lr_decay"],
        "min_lr": outer["min_lr"],
        "inner_lr_decay": inner["inner_lr_decay"],
        "inner_min_lr": inner["inner_min_lr"],
        "dual_learning_rate": outer["dual_learning_rate"],
        "penalty_coef": outer["penalty_coef"],
        "penalty_growth": outer["penalty_growth"],
        "proximal_coef": outer["proximal_coef"],
        "max_penalty": outer["max_penalty"],
        "max_dual": outer["max_dual"],
        "return_best_violation": outer["return_best_violation"],
        "verbose_interval": params.get("verbose_interval", 50),
        "use_lagrangian": True,
        "use_penalty": True,
    }
    base.update(inner)

    algorithm_config = _algorithm_config(params)
    penalty = _deepcopy_dict(base)
    penalty.update(algorithm_config.get("Penalty", {}))
    prox_penalty = _deepcopy_dict(base)
    prox_penalty.update(algorithm_config.get("Penalty", {}))
    prox_penalty.update(algorithm_config.get("Prox-Penalty", {}))
    alm = _deepcopy_dict(base)
    alm.update(algorithm_config.get("ALM", {}))
    prox_alm = _deepcopy_dict(base)
    prox_alm.update(algorithm_config.get("ALM", {}))
    prox_alm.update(algorithm_config.get("Prox-ALM", {}))
    hom_alm = _deepcopy_dict(base)
    hom_alm.update(algorithm_config.get("Hom-ALM", {}))
    prox_hom_alm = _deepcopy_dict(base)
    prox_hom_alm.update(algorithm_config.get("Hom-ALM", {}))
    prox_hom_alm.update(algorithm_config.get("Prox-Hom-ALM", {}))
    penalty["opt_type"] = "Penalty"
    penalty["use_lagrangian"] = False
    penalty["use_penalty"] = True
    prox_penalty["opt_type"] = "Prox-Penalty"
    prox_penalty["use_lagrangian"] = False
    prox_penalty["use_penalty"] = True
    prox_penalty["use_proximal"] = True
    alm["opt_type"] = "ALM"
    prox_alm["opt_type"] = "Prox-ALM"
    prox_alm["use_proximal"] = True
    hom_alm["opt_type"] = "Hom-ALM"
    prox_hom_alm["opt_type"] = "Prox-Hom-ALM"
    prox_hom_alm["use_proximal"] = True
    alm_params = {
        "common": {
            "seed": int(params["seed"]),
            "verbose": bool(params.get("verbose", False)),
        },
        "Penalty": penalty,
        "Prox-Penalty": prox_penalty,
        "ALM": alm,
        "Prox-ALM": prox_alm,
        "Hom-ALM": hom_alm,
        "Prox-Hom-ALM": prox_hom_alm,
    }
    enforce_alm_outer_iteration_budget(
        alm_params,
        max_iterations=params["max_iterations"],
        context="stiefel algorithm_config",
    )
    return alm_params


def _effective_stiefel_inner_iterations(params):
    inner = _inner_config(params)
    rule = str(inner["inner_stopping_rule"]).lower()
    if rule == "adaptive":
        return int(inner["inner_iterations_max"])
    return int(inner["inner_iterations"])


def _stiefel_nested_solver_base_params(params):
    outer = _outer_config(params)
    inner = _inner_config(params)
    return {
        "outer_iterations": int(outer["outer_iterations"]),
        "inner_iterations": _effective_stiefel_inner_iterations(params),
        "max_running_time": outer["max_running_time"],
        "convergence_threshold": outer["convergence_threshold"],
        "learning_rate": outer["learning_rate"],
        "stepsize_rule": inner["inner_stepsize_rule"],
        "lr_decay": inner["inner_lr_decay"],
        "min_lr": inner["inner_min_lr"],
        "dual_learning_rate": outer["dual_learning_rate"],
        "penalty_coef": outer["penalty_coef"],
        "penalty_growth": outer["penalty_growth"],
        "max_penalty": outer["max_penalty"],
        "max_dual": outer["max_dual"],
        "verbose": bool(params.get("verbose", False)),
        "verbose_interval": params.get("verbose_interval", 50),
    }


def _stiefel_penalty_solver_base_params(params):
    outer = _outer_config(params)
    inner = _inner_config(params)
    return {
        "outer_iterations": int(outer["outer_iterations"]),
        "inner_iterations": _effective_stiefel_inner_iterations(params),
        "max_running_time": outer["max_running_time"],
        "convergence_threshold": outer["convergence_threshold"],
        "learning_rate": outer["learning_rate"],
        "stepsize_rule": inner["inner_stepsize_rule"],
        "lr_decay": inner["inner_lr_decay"],
        "min_lr": inner["inner_min_lr"],
        "penalty_coef": outer["penalty_coef"],
        "penalty_growth": outer["penalty_growth"],
        "max_penalty": outer["max_penalty"],
        "verbose": bool(params.get("verbose", False)),
        "verbose_interval": params.get("verbose_interval", 50),
    }


def _relative_objective_gap(objective, reference_objective):
    if objective is None or reference_objective is None:
        return None
    if abs(float(reference_objective)) <= 1e-12:
        return None
    return float(abs(float(objective) - float(reference_objective)) / abs(float(reference_objective)))


def _solver_summary(problem, result, *, reference_objective=None):
    solution = result.get("solution")
    split = constraint_violation_summary(problem, solution)
    objective = objective_summary(problem, solution).get("final_objective", result.get("objective"))
    full_violation = split.get("final_full_violation", result.get("violation"))
    extras = result.get("extras", {})
    summary = {
        "final_objective": objective,
        "final_violation": full_violation,
        "final_full_violation": full_violation,
        "final_equality_violation": split.get("final_equality_violation"),
        "final_inequality_violation": split.get("final_inequality_violation"),
        "total_wall_time": result.get("runtime_total", 0.0),
        "iterations": int(len(extras.get("objective_traj", []))),
        "status": result.get("status"),
    }
    outer_iter_time = np.asarray(extras.get("outer_iter_time", []), dtype=float).reshape(-1)
    if outer_iter_time.size:
        summary["outer_iter_time"] = outer_iter_time.tolist()
        summary["total_outer_iter_time"] = float(outer_iter_time.sum())
        summary["mean_outer_iter_time"] = float(outer_iter_time.mean())
        summary["total_iter_time"] = float(outer_iter_time.sum())
    lag_gap = extras.get("final_first_order_lagrangian_gap")
    if lag_gap is None:
        lag_gap_traj = np.asarray(extras.get("first_order_lagrangian_gap_traj", []), dtype=float).reshape(-1)
        if lag_gap_traj.size:
            lag_gap = float(lag_gap_traj[-1])
            summary["first_order_lagrangian_gap_traj"] = lag_gap_traj.tolist()
    if lag_gap is not None:
        summary["final_first_order_lagrangian_gap"] = float(lag_gap)
    gap = _relative_objective_gap(objective, reference_objective)
    if gap is not None:
        summary["objective_gap"] = gap
    return summary


def _enforce_stiefel_solver_outer_budget(solver_params, params, algorithm):
    if "outer_iterations" not in solver_params:
        return solver_params
    max_iter = int(params["max_iterations"])
    if int(solver_params["outer_iterations"]) != max_iter:
        raise ValueError(
            f"Conflicting stiefel algorithm_config parameters for {algorithm}: "
            f"max_iterations={max_iter!r} and outer_iterations={solver_params['outer_iterations']!r}. "
            "Use max_iterations as the shared outer-loop budget."
        )
    return solver_params


def _run_stiefel_retraction_penalty(problem, params, init_point, algorithm="StiefelRetractionPenalty"):
    solver = StiefelRetractionOptimizer(problem)
    algorithm_config = _algorithm_config(params)
    solver_params = _stiefel_penalty_solver_base_params(params)
    solver_params.update(_deepcopy_dict(algorithm_config.get(algorithm)))
    if algorithm == "StiefelRetraction":
        solver_params.update(_deepcopy_dict(algorithm_config.get("StiefelRetractionPenalty", {})))
    solver_params.setdefault("seed", int(params["seed"]))
    _enforce_stiefel_solver_outer_budget(solver_params, params, algorithm)
    return solver.solve_result(x_init=init_point, **solver_params)


def _run_stiefel_retraction_alm(problem, params, init_point):
    solver = StiefelRetractionALMOptimizer(problem)
    algorithm_config = _algorithm_config(params)
    solver_params = _stiefel_nested_solver_base_params(params)
    solver_params.update(_deepcopy_dict(algorithm_config.get("StiefelRetractionALM", {})))
    solver_params.setdefault("seed", int(params["seed"]))
    _enforce_stiefel_solver_outer_budget(solver_params, params, "StiefelRetractionALM")
    return solver.solve_result(x_init=init_point, **solver_params)


def _run_stiefel_alm_eq_pgd(problem, params, init_point):
    solver = StiefelALMEQPGDOptimizer(problem)
    algorithm_config = _algorithm_config(params)
    solver_params = _stiefel_nested_solver_base_params(params)
    solver_params.update(_deepcopy_dict(algorithm_config.get("ALM-EQ-PGD", {})))
    solver_params.setdefault("seed", int(params["seed"]))
    _enforce_stiefel_solver_outer_budget(solver_params, params, "ALM-EQ-PGD")
    return solver.solve_result(x_init=init_point, **solver_params)


def _run_stiefel_ipopt(problem, params, init_point):
    solver = StiefelPyomoIPOPTSolver(problem)
    algorithm_config = _algorithm_config(params)
    solver_params = _deepcopy_dict(algorithm_config.get("StiefelIPOPT", {}))
    solver_params.setdefault("seed", int(params["seed"]))
    return solver.solve_result(x_init=init_point, **solver_params)


STIEFEL_SOLVER_RUNNERS = {
    "StiefelRetraction": lambda problem, params, init_point: _run_stiefel_retraction_penalty(
        problem,
        params,
        init_point,
        algorithm="StiefelRetraction",
    ),
    "StiefelRetractionPenalty": _run_stiefel_retraction_penalty,
    "StiefelRetractionALM": _run_stiefel_retraction_alm,
    "ALM-EQ-PGD": _run_stiefel_alm_eq_pgd,
}


def _prepare_stiefel_reference_result(problem, params, init_point, output_dir):
    cache_reference = bool(params.get("reference_cache", True))
    cache_path = stiefel_reference_cache_path(output_dir, cache_reference)
    cache_key = stiefel_reference_cache_key(params) if cache_path is not None else None
    cached_payload, cache_load_time = load_stiefel_reference_cache(cache_path, cache_key)
    if cached_payload is not None:
        cached_result = cached_payload.get("result", {})
        if cached_result.get("solution") is not None:
            result = copy.deepcopy(cached_result)
            return result, {
                "reference_cache_enabled": cache_reference,
                "reference_cache_hit": True,
                "reference_cache_path": cache_path,
                "reference_cache_key": cache_key,
                "reference_cache_load_time": cache_load_time,
                "reference_cached_solver_time": result.get("runtime_total"),
                "reference_cached_opt_solver_time": result.get("runtime_total"),
            }
        warnings.warn(
            f"Cached Stiefel IPOPT reference at {cache_path} has no solution; recomputing IPOPT.",
            RuntimeWarning,
        )

    result = _run_stiefel_ipopt(problem, params, init_point)
    if result.get("solution") is not None:
        save_stiefel_reference_cache(cache_path, cache_key, {"result": result})
    return result, {
        "reference_cache_enabled": cache_reference,
        "reference_cache_hit": False,
        "reference_cache_path": cache_path,
        "reference_cache_key": cache_key,
        "reference_cache_load_time": cache_load_time,
        "reference_cached_solver_time": None,
        "reference_cached_opt_solver_time": None,
    }


def _reference_solver_row(name, summary):
    return make_single_instance_solver_row(
        method=name,
        objective=summary["final_objective"],
        violation=summary["final_violation"],
        runtime_total=summary["total_wall_time"],
        total_wall_time=summary["total_wall_time"],
        status=summary.get("status"),
        full_violation=summary.get("final_full_violation"),
        equality_violation=summary.get("final_equality_violation"),
        inequality_violation=summary.get("final_inequality_violation"),
        iterations=summary.get("iterations"),
    )

def stiefel_algorithm_comparison(
    *,
    algorithms=None,
    output_dir=None,
    **params,
):
    requested_algorithms = list(algorithms or params["algorithms"])
    reference_solver_name = "StiefelIPOPT"
    reference_requested = reference_solver_name in requested_algorithms
    algorithms = [algorithm for algorithm in requested_algorithms if algorithm != reference_solver_name]
    if bool(params.get("visualize", True)) and bool(params.get("visualize_only", False)):
        problem, _, _ = build_stiefel_problem(params)
        requested_record_algorithms = algorithms or None
        visual_state = load_visualize_only_comparison_artifacts(
            output_dir,
            algorithms=requested_record_algorithms,
        )
        records = visual_state["records"]
        summaries = visual_state["summaries"]
        previous_result = visual_state["previous_result"]
        previous_algorithms = visual_state["stored_algorithms"]
        reference_objective = visual_state["metrics"].get("reference_objective")
        artifacts = visual_state["artifacts"]
        artifacts.update(
            _plot_stiefel_records(
                problem,
                records,
                previous_algorithms,
                output_dir,
                params,
                reference_objective=reference_objective,
            )
        )
        return build_visualize_only_benchmark_payload(
            previous_result=previous_result,
            summaries=summaries,
            artifacts=artifacts,
            requested_algorithms=requested_algorithms,
            algorithms=previous_algorithms,
        )
    include_reference_solver = params.get("include_reference_solver")
    reference_need_opt = params.get("reference_need_opt")
    if include_reference_solver is None:
        include_reference_solver = reference_requested
    if include_reference_solver is True:
        reference_need_opt = True
    if reference_need_opt is None:
        reference_need_opt = bool(reference_requested or include_reference_solver)
    problem, _, runtime_dtype = build_stiefel_problem(params)
    hom_map = build_stiefel_hom_map(problem, params, runtime_dtype)
    optimizer_params = _build_alm_params(params)
    init_point = build_stiefel_initial_point(problem, params, runtime_dtype)

    reference_result = None
    reference_summary = None
    reference_objective = None
    reference_solver_time = None
    reference_solver_violation = None
    reference_solution_diagnostics = {"available": False}
    reference_cache_metadata = {
        "reference_cache_enabled": bool(params.get("reference_cache", True)),
        "reference_cache_hit": False,
        "reference_cache_path": None,
        "reference_cache_key": None,
        "reference_cache_load_time": 0.0,
        "reference_cached_solver_time": None,
    }
    if reference_need_opt:
        if params.get("verbose", False):
            print("Preparing StiefelIPOPT reference")
        reference_result, reference_cache_metadata = _prepare_stiefel_reference_result(
            problem,
            params,
            init_point,
            output_dir,
        )
        if reference_result.get("solution") is None and not bool(params.get("allow_missing_reference", False)):
            status = reference_result.get("status", "unknown")
            message = reference_result.get("message") or reference_result.get("error") or "no solution returned"
            raise RuntimeError(
                "StiefelIPOPT reference solve failed while reference_need_opt=True. "
                f"status={status!r}, message={message!r}. "
                "Set allow_missing_reference=True only for runs where a missing reference is intentional."
            )
        if params.get("verbose", False):
            if reference_cache_metadata.get("reference_cache_hit", False):
                print("Loaded StiefelIPOPT reference from cache")
            else:
                print("Computed StiefelIPOPT reference")
        reference_summary = _solver_summary(problem, reference_result)
        reference_objective = reference_summary.get("final_objective")
        reference_solver_time = reference_summary.get("total_wall_time")
        reference_solver_violation = reference_summary.get("final_full_violation")
        reference_solution_diagnostics = convex_solution_diagnostics(problem, reference_result.get("solution"))

    records = {}
    summaries = {}
    rows = []
    for algorithm in algorithms:
        if algorithm in STIEFEL_OPTIMIZER_ALGORITHMS:
            record = run_algorithm(
                algorithm,
                problem,
                optimizer_params,
                hom_map if algorithm in STIEFEL_HOM_OPTIMIZER_ALGORITHMS else None,
                init_point,
            )
            records[algorithm] = record
            summaries[algorithm] = summarize_single_problem_run_record(
                problem,
                record,
                reference_objective=reference_objective,
                include_total_iter_time=True,
            )
            continue
        solver_runner = STIEFEL_SOLVER_RUNNERS.get(algorithm)
        if solver_runner is not None:
            record = solver_runner(problem, params, init_point)
            records[algorithm] = record
            summaries[algorithm] = _solver_summary(problem, record, reference_objective=reference_objective)
            continue
        raise ValueError(f"Unknown Stiefel comparison algorithm: {algorithm}")

    if records:
        artifacts, iterative_views, records, summaries = save_incremental_comparison_artifacts(
            output_dir,
            records=records,
            summaries=summaries,
            algorithm_order=algorithms,
            run_identity={
                "benchmark": "stiefel_algorithm_comparison",
                "seed": params["seed"],
                "problem_config": params["problem_config"],
                "mapping": params["mapping"],
                "initialization": params.get("initialization"),
                "common_config": params["common_config"],
                "outer_common": _outer_config(params),
                "inner_solver_common": _inner_config(params),
                "runtime": {"device": str(params.get("device")), "dtype": str(params.get("dtype"))},
                "reference": {
                    "include_reference_solver": include_reference_solver,
                    "need_opt": reference_need_opt,
                    "allow_missing": params.get("allow_missing_reference", False),
                    "solver_config": _algorithm_config(params).get("StiefelIPOPT", {}),
                },
            },
            manifest_metadata={
                "problem_family": "stiefel",
                "scale_label": scale_label(params),
                "reference_label": "IPOPT",
                "reference_objective": reference_objective,
            },
        )
        stored_algorithms = list(records.keys())
        artifacts.update(
            _plot_stiefel_records(
                problem,
                records,
                stored_algorithms,
                output_dir,
                params,
                reference_objective=reference_objective,
            )
        )
        iterative_rows = iterative_views["comparison_rows"]
    else:
        artifacts = {}
        stored_algorithms = []
        iterative_rows = []
    reference_cache_path = reference_cache_metadata.get("reference_cache_path")
    if reference_cache_path is not None:
        artifacts["reference_cache"] = artifact_ref(output_dir, reference_cache_path)
    if include_reference_solver and reference_summary is not None and reference_summary.get("final_objective") is not None:
        rows.append(_reference_solver_row(reference_solver_name, reference_summary))
    for row in iterative_rows:
        if row["method"] in STIEFEL_OPTIMIZER_ALGORITHMS:
            rows.append(row)
    for solver_name in STIEFEL_SOLVER_RUNNERS:
        if solver_name not in summaries:
            continue
        solver_summary = summaries[solver_name]
        rows.append(
            make_single_instance_solver_row(
                method=solver_name,
                objective=solver_summary["final_objective"],
                violation=solver_summary["final_violation"],
                runtime_total=solver_summary["total_wall_time"],
                total_wall_time=solver_summary["total_wall_time"],
                status=solver_summary.get("status"),
                full_violation=solver_summary.get("final_full_violation"),
                equality_violation=solver_summary.get("final_equality_violation"),
                inequality_violation=solver_summary.get("final_inequality_violation"),
                first_order_lagrangian_gap=solver_summary.get("final_first_order_lagrangian_gap"),
                iterations=solver_summary.get("iterations"),
            )
        )
    comparison_summary_rows = summarize_comparison_rows_by_method(rows)
    comparison_views = build_comparison_views(rows, summary_rows=comparison_summary_rows)

    return build_benchmark_payload(
        objective=comparison_views["objective"],
        feasible=comparison_views["feasible"],
        artifacts=artifacts,
        problem_family="stiefel",
        scale_label=scale_label(params),
        algorithms=stored_algorithms,
        requested_algorithms=requested_algorithms,
        problem_config=copy.deepcopy(params["problem_config"]),
        reference_objective=reference_objective,
        reference_solver_time=reference_solver_time,
        reference_solver_violation=reference_solver_violation,
        reference_need_opt=reference_need_opt,
        reference_cache_enabled=reference_cache_metadata.get("reference_cache_enabled", False),
        reference_cache_hit=reference_cache_metadata.get("reference_cache_hit", False),
        reference_cache_path=artifact_ref(output_dir, reference_cache_path),
        reference_cache_load_time=reference_cache_metadata.get("reference_cache_load_time", 0.0),
        reference_cached_solver_time=reference_cache_metadata.get("reference_cached_solver_time"),
        reference_cached_opt_solver_time=reference_cache_metadata.get("reference_cached_opt_solver_time"),
        reference_solver_status=None if reference_summary is None else reference_summary.get("status"),
        reference_solver_equality_violation=None
        if reference_summary is None
        else reference_summary.get("final_equality_violation"),
        reference_solver_inequality_violation=None
        if reference_summary is None
        else reference_summary.get("final_inequality_violation"),
        reference_solution_diagnostics=reference_solution_diagnostics,
        results=summaries,
        instance_rows=comparison_views["instance_rows"],
        method_summary_rows=comparison_views["method_summary_rows"],
        comparison_rows=comparison_views["comparison_rows"],
        comparison_summary=comparison_views["comparison_summary"],
        comparison_summary_rows=comparison_views["comparison_summary_rows"],
        best_method=comparison_views["best_method"],
        seed=int(params["seed"]),
    )
