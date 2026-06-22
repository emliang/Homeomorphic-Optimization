"""Parametric JCC benchmark entrypoints."""

from __future__ import annotations

import numpy as np

from homopt.records.artifacts import artifact_mapping, artifact_ref, artifact_root, build_benchmark_payload, save_json, save_numpy, save_table_artifacts
from homopt.experiments.common.comparison import build_comparison_views, make_comparison_row, summarize_comparison_rows_by_method
from homopt.experiments.common.runtime import resolve_runtime
from homopt.problems import JCCDCOPFProblem, JCCIMProblem, JCCLinearProblem, bind_problem_instance, bind_singleton_problem_instance
from homopt.solvers import JCCLinearCVaRSolver, JCCLinearRobustScenarioSolver, JCCLinearSolver, solve_exact_result
from homopt.utils import cast_tensors_to_dtype, set_global_seed

from .setup import normalize_jcc_linear_problem_config, normalize_jcc_linear_solver_configs


JCC_LINEAR_SOLVER_RESULT_FIELDS = (
    "row_kind",
    "route",
    "method",
    "instance_id",
    "status",
    "objective",
    "feasible",
    "violation",
    "runtime",
    "objective_mean",
    "feasibility_rate",
    "violation_mean",
    "runtime_mean",
    "runtime_total",
)


def _save_jcc_linear_solver_artifacts(output_dir, *, metrics, rows, summary_rows):
    artifact_dir = artifact_root(output_dir)
    if artifact_dir is None:
        return {}
    summary_path = save_json(artifact_dir / "jcc_linear_solver_summary.json", metrics)
    row_artifacts, _, _, _ = save_table_artifacts(
        output_dir,
        base_name="jcc_linear_solver_rows",
        rows=rows,
        fieldnames=list(JCC_LINEAR_SOLVER_RESULT_FIELDS),
        json_key="rows_json",
        csv_key="rows_csv",
        markdown_key="unused_markdown",
    )
    summary_artifacts, _, _, _ = save_table_artifacts(
        output_dir,
        base_name="jcc_linear_solver_method_summary",
        rows=summary_rows,
        fieldnames=[
            "route",
            "method",
            "objective_mean",
            "feasibility_rate",
            "violation_mean",
            "runtime_mean",
            "runtime_total",
            "num_instances",
        ],
        json_key="method_summary_json",
        csv_key="method_summary_csv",
        markdown_key="unused_method_markdown",
    )
    return {
        **row_artifacts,
        **summary_artifacts,
        "summary": artifact_ref(output_dir, summary_path),
    }


def jcc_problem_benchmark(
    num_bus=30,
    n_scenarios=3,
    demand_std=0.05,
    seed=2025,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Package-native JCC smoke that records scenario feasibility data."""

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    problem = JCCDCOPFProblem(
        num_bus=num_bus,
        config={
            "n_scenarios": n_scenarios,
            "demand_std": demand_std,
            "seed": seed,
        },
    ).to_device(runtime_device)
    problem = cast_tensors_to_dtype(problem, runtime_dtype)
    problem = bind_singleton_problem_instance(problem, seed=seed)
    midpoint = ((problem.P_min + problem.P_max) / 2).view(1, -1)
    objective = float(problem.objective_x(midpoint).item())
    chance_violation = float(problem.constraint_x(midpoint).item())
    robust_violation = float(problem.robust_constraint_x(midpoint).item())
    scenario_feasibility = problem.compute_scenario_feasibility(midpoint).detach().cpu().numpy()
    feasibility_rate = float(np.mean(scenario_feasibility))

    artifact_dir = artifact_root(output_dir)
    artifacts = {}
    if artifact_dir is not None:
        scenario_path = save_numpy(artifact_dir / "jcc_scenarios.npy", scenario_feasibility)
        summary_path = save_json(
            artifact_dir / "jcc_summary.json",
            {
                "feasibility_rate": feasibility_rate,
                "objective": objective,
                "chance_violation": chance_violation,
                "robust_violation": robust_violation,
                "n_gen_vars": int(problem.n_gen_vars),
                "n_scenarios": int(problem.n_scenarios),
            },
        )
        artifacts = artifact_mapping(output_dir, scenario_feasibility=scenario_path, summary=summary_path)

    return build_benchmark_payload(
        objective=objective,
        feasible=chance_violation <= 1e-5,
        artifacts=artifacts,
        chance_violation=chance_violation,
        robust_violation=robust_violation,
        feasibility_rate=feasibility_rate,
        n_gen_vars=int(problem.n_gen_vars),
        n_scenarios=int(problem.n_scenarios),
        num_bus=int(problem.n_bus),
        seed=seed,
    )


def jcc_linear_solver_benchmark(
    seed=2025,
    problem_family="jcc_linear",
    n_var=8,
    n_input_dim=4,
    n_ineq=4,
    n_scenarios=20,
    epsilon=0.1,
    n_samples=8,
    problem_config=None,
    solver_configs=None,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Parametric JCC linear benchmark comparing the three exact solver baselines."""

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    problem_args = normalize_jcc_linear_problem_config(
        problem_family=problem_family,
        seed=seed,
        n_var=n_var,
        n_input_dim=n_input_dim,
        n_ineq=n_ineq,
        n_scenarios=n_scenarios,
        epsilon=epsilon,
        problem_config=problem_config,
    )
    solver_cfgs = normalize_jcc_linear_solver_configs(
        solver_configs=solver_configs,
    )
    resolved_family = str(problem_args.get("problem_family", "jcc_linear")).strip().lower()
    problem_cls = JCCLinearProblem if resolved_family == "jcc_linear" else JCCIMProblem
    problem = problem_cls(config=problem_args).to_device(runtime_device).to_dtype(runtime_dtype)
    instance_batch = problem.sample_instance_batch(n_instances=n_samples, seed=seed)

    solver_specs = (
        ("mixed_integer", JCCLinearSolver, solver_cfgs["mixed_integer"]),
        ("cvar", JCCLinearCVaRSolver, solver_cfgs["cvar"]),
        ("scenario", JCCLinearRobustScenarioSolver, solver_cfgs["scenario"]),
    )
    rows = []
    for idx in range(len(instance_batch)):
        bound_problem = bind_problem_instance(problem, instance_batch.get_instance(idx))
        for method, solver_cls, cfg in solver_specs:
            solver = solver_cls(bound_problem, solver_config=cfg)
            result = solve_exact_result(solver)
            rows.append(
                {
                    **make_comparison_row(
                        route="solver",
                        method=method,
                        objective=result.get("objective"),
                        feasible=bool(result.get("feasible")),
                        violation=result.get("violation"),
                        runtime_total=float(result.get("runtime_total") if result.get("runtime_total") is not None else 0.0),
                        total_wall_time=result.get("runtime_total"),
                    ),
                    "instance_id": int(idx),
                    "status": str(result.get("status", "unknown")),
                }
            )

    summary_rows = summarize_comparison_rows_by_method(rows)
    comparison_views = build_comparison_views(rows, summary_rows=summary_rows)
    solver_summary = {
        row["method"]: {
            "objective_mean": row["objective_mean"],
            "feasibility_rate": row["feasibility_rate"],
            "violation_mean": row["violation_mean"],
            "runtime_mean": row["runtime_mean"],
            "runtime_total": row["runtime_total"],
            "num_instances": row["num_instances"],
        }
        for row in summary_rows
    }
    metrics = {
        "seed": int(seed),
        "problem_family": resolved_family,
        "n_samples": int(n_samples),
        "n_var": int(problem.nvar),
        "n_input_dim": int(problem.n_input_dim),
        "n_ineq": int(problem.n_ineq),
        "n_scenarios": int(problem.n_scenarios),
        "epsilon": float(problem.epsilon),
        "enabled_solvers": [name for name, _, _ in solver_specs],
        "results": comparison_views["comparison_rows"],
        "instance_rows": comparison_views["instance_rows"],
        "method_summary_rows": comparison_views["method_summary_rows"],
        "comparison_rows": comparison_views["comparison_rows"],
        "comparison_summary": comparison_views["comparison_summary"],
        "comparison_summary_rows": comparison_views["comparison_summary_rows"],
        "best_method": comparison_views["best_method"],
        "solver_summary": solver_summary,
    }
    artifacts = _save_jcc_linear_solver_artifacts(
        output_dir,
        metrics=metrics,
        rows=rows,
        summary_rows=summary_rows,
    )
    return build_benchmark_payload(
        objective=comparison_views["objective"],
        feasible=comparison_views["feasible"],
        artifacts=artifacts,
        **metrics,
    )


__all__ = [
    "jcc_linear_solver_benchmark",
    "jcc_problem_benchmark",
]
