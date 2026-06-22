"""JCC benchmark comparison entrypoints."""

from __future__ import annotations

import json

from homopt.records.artifacts import (
    artifact_mapping,
    artifact_ref,
    artifact_root,
    build_benchmark_payload,
    save_json,
    save_numpy,
    write_csv,
)
from homopt.experiments.common.comparison import build_comparison_views, summarize_comparison_rows_by_method
from homopt.experiments.common.config import normalize_jcc_problem_config, normalize_jcc_solver_configs
from homopt.experiments.common.inn_baselines import normalize_inn_lagrangian_baselines, run_inn_lagrangian_baseline
from homopt.experiments.method_specs import normalize_jcc_lagrangian_config
from homopt.experiments.common.runtime import ensure_int_list, resolve_runtime
from homopt.utils import set_global_seed

from .methods import (
    _jcc_solver_baseline_specs,
    _resolve_enabled_baselines,
    _run_jcc_inn_pgd_variant,
    _run_jcc_solver_suite,
)
from .reports import build_jcc_comparison_metrics, build_jcc_iterative_method_rows
from .setup import _build_jcc_problem


def jcc_baseline_solver_sweep(
    num_bus_list,
    n_scenarios,
    epsilon=0.1,
    demand_std=0.05,
    seed=2025,
    solver_configs=None,
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

    return build_benchmark_payload(
        objective=comparison_views["objective"],
        feasible=comparison_views["feasible"],
        artifacts=artifacts,
        num_cases=int(len(list(num_bus_list)) * len(n_scenarios_list)),
        num_records=int(len(records)),
        enabled_baselines=enabled,
        instance_rows=comparison_views["instance_rows"],
        method_summary_rows=comparison_views["method_summary_rows"],
        comparison_rows=comparison_views["comparison_rows"],
        comparison_summary=comparison_views["comparison_summary"],
        comparison_summary_rows=comparison_views["comparison_summary_rows"],
        best_method=comparison_views["best_method"],
        all_records_feasible=all(bool(row["chance_satisfied"]) for row in records),
    )


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
    max_iterations=300,
    num_test_instance=1,
    solver_configs=None,
    lagrangian_baselines=None,
    lagrangian_baseline_config=None,
    model_config=None,
    optimizer_config=None,
    train_config=None,
    problem_config=None,
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
    )
    solver_cfgs = normalize_jcc_solver_configs(
        solver_configs=solver_configs,
    )
    enabled_lagrangian_baselines = normalize_inn_lagrangian_baselines(
        ("ALM", "Penalty", "Prox-Penalty") if lagrangian_baselines is None else lagrangian_baselines
    )
    lag_args = None
    if enabled_lagrangian_baselines:
        lag_args = normalize_jcc_lagrangian_config(
            max_iterations=max_iterations,
            lagrangian_baseline_config=lagrangian_baseline_config,
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
            max_iterations=max_iterations,
            retrain=retrain,
            num_test_instance=num_test_instance,
            model_config=model_config,
            optimizer_config=optimizer_config,
            train_config=train_config,
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


__all__ = ["jcc_algorithm_comparison", "jcc_baseline_solver_sweep"]
