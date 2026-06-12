"""JCC OPF benchmark entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import torch

from homopt.experiments.common.artifacts import (
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
from .reports import (
    build_jcc_comparison_metrics,
    build_jcc_iterative_method_rows,
    build_jcc_solver_method_row,
    build_jcc_solver_record,
)
from homopt.experiments.common.method_specs import normalize_jcc_lagrangian_config
from homopt.experiments.common.runtime import as_builtin, ensure_int_list, resolve_runtime
from homopt.experiments.common.inn_baselines import normalize_inn_lagrangian_baselines, run_inn_lagrangian_baseline
from homopt.experiments.common.inn_training import load_or_train_inn_mapping
from homopt.optim import INNPGDOptimizer
from homopt.problems import JCCDCOPFProblem, bind_singleton_problem_instance
from homopt.solvers import JCCDCOPFCVaRSolver, JCCDCOPFRobustScenarioSolver, JCCDCOPFSolver, solve_exact_result
from homopt.utils import cast_tensors_to_dtype, set_global_seed


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


class _JCCINNPGDAdapter:
    """Adapter exposing the shared INN-PGD training contract for JCC-DC-OPF."""

    training_constraint_surrogate = "scenario_max_residual_v1"

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
        scenario_batch = self.problem._coerce_scenario_batch(input_batch, x.device, batch_size=x.shape[0])
        if scenario_batch is None:
            return self.problem.robust_constraint_x(x, clip=clip)
        scenario_max_residuals = [
            self.problem.compute_scenario_k_residual(scenario_batch[:, scenario_idx, :], x, clip=clip).max(
                dim=1,
                keepdim=True,
            )[0]
            for scenario_idx in range(self.n_scenarios)
        ]
        return torch.cat(scenario_max_residuals, dim=1)

    def chance_violation(self, input_batch, x, clip=True):
        return self.problem.constraint_residual_xy(input_batch, x, clip=clip)

    def check_feasibility(self, input_batch, x):
        return self.ineq_resid(input_batch, x, clip=True)

    def violations(self, input_batch, x):
        return self.chance_violation(input_batch, x, clip=True)

    def objective(self, x):
        return self.problem.objective_x(x)

    def training_record_metadata(self):
        return {"constraint_surrogate": self.training_constraint_surrogate}


def _reject_jcc_model_training_keys(config, *, context):
    training_keys = {"n_samples", "batch_size", "total_iteration"}
    invalid = sorted(training_keys.intersection((config or {}).keys()))
    if invalid:
        raise ValueError(
            f"{context} contains training keys {invalid}; "
            "use train_config instead."
        )


def _normalize_jcc_inn_configs(
    *,
    max_iterations=None,
    model_config=None,
    optimizer_config=None,
    train_config=None,
):
    _reject_jcc_model_training_keys(model_config, context="model_config")
    if model_config is None:
        raise ValueError("model_config must be provided explicitly by the experiment script.")
    if optimizer_config is None:
        raise ValueError("optimizer_config must be provided explicitly by the experiment script.")
    if train_config is None:
        raise ValueError("train_config must be provided explicitly by the experiment script.")
    model_cfg = dict(model_config)
    train_cfg = dict(train_config)
    optimizer_cfg = dict(optimizer_config)
    if "max_iterations" in optimizer_cfg:
        raise ValueError("optimizer_config must not set max_iterations; use top-level max_iterations.")
    if max_iterations is not None:
        optimizer_cfg["max_iterations"] = int(max_iterations)
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
    max_iterations,
    retrain,
    num_test_instance,
    model_config=None,
    optimizer_config=None,
    train_config=None,
):
    adapter = _JCCINNPGDAdapter(problem)
    if adapter.nvar < 2:
        raise ValueError(
            "JCC INN-PGD requires at least two active decision variables for the current INN flow. "
            "Use a larger OPF case such as 57-bus or disable run_inn_pgd for 30-bus smoke runs."
        )
    model_cfg, optimizer_cfg, train_cfg = _normalize_jcc_inn_configs(
        max_iterations=max_iterations,
        model_config=model_config,
        optimizer_config=optimizer_config,
        train_config=train_config,
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
    expected_surrogate = getattr(adapter, "training_constraint_surrogate", None)
    if (not retrain) and expected_surrogate is not None and training_record.get("constraint_surrogate") != expected_surrogate:
        print(
            "[HomOPT] Ignoring stale JCC INN checkpoint: "
            "training constraint surrogate changed; retraining."
        )
        model, training_record, model_path, record_path = load_or_train_inn_mapping(
            adapter,
            model_args,
            save_dir,
            retrain=True,
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
    final_objective_tensor = obj_traj.detach().cpu()[-1].reshape(-1)
    final_violation_tensor = cons_traj.detach().cpu()[-1].reshape(-1)
    finite_solution = bool(torch.isfinite(x_opt).all().item())
    finite_objective = bool(torch.isfinite(final_objective_tensor).all().item())
    finite_violation = bool(torch.isfinite(final_violation_tensor).all().item())
    final_objective = float(final_objective_tensor.mean().item()) if finite_objective else None
    final_violation = float(final_violation_tensor.max().item()) if finite_violation else None
    with torch.inference_mode():
        if finite_solution:
            scenario_feasibility = problem.compute_scenario_feasibility(x_opt, scenario_batch=inputs)
            chance_feasibility_rate = float(scenario_feasibility.mean().item())
        else:
            chance_feasibility_rate = 0.0
        violation = max(0.0, (1.0 - chance_feasibility_rate) - float(problem.config["epsilon"]))
    status = "completed" if finite_solution and finite_objective and finite_violation else "failed_nonfinite"
    nonfinite_reason = None
    if status != "completed":
        failed_parts = []
        if not finite_solution:
            failed_parts.append("x_opt")
        if not finite_objective:
            failed_parts.append("objective")
        if not finite_violation:
            failed_parts.append("violation")
        nonfinite_reason = "non-finite " + ", ".join(failed_parts)
    return {
        "x_opt": x_opt.detach().cpu().numpy() if finite_solution else None,
        "obj_traj": obj_traj.detach().cpu().numpy(),
        "cons_traj": cons_traj.detach().cpu().numpy(),
        "iter_time": list(per_iter_time),
        "total_wall_time": total_wall_time,
        "final_objective": final_objective,
        "final_violation": final_violation,
        "chance_feasibility_rate": chance_feasibility_rate,
        "violation": violation,
        "feasible": status == "completed" and violation <= 1e-12,
        "status": status,
        "failure_reason": nonfinite_reason,
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
    lag_args = normalize_jcc_lagrangian_config(
        max_iterations=max_iterations,
        lagrangian_baseline_config=lagrangian_baseline_config,
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


__all__ = [
    "_JCCINNPGDAdapter",
    "_build_jcc_problem",
    "_normalize_jcc_inn_configs",
    "_run_jcc_inn_pgd_variant",
    "_run_jcc_solver_suite",
    "jcc_algorithm_comparison",
    "jcc_baseline_solver_sweep",
]
