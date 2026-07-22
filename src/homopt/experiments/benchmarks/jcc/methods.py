"""JCC benchmark method, solver, and INN adapter helpers."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from homopt.records.artifacts import artifact_root
from homopt.experiments.common.runtime import as_builtin
from homopt.learning.training import prepare_learning_mapping
from homopt.optim import INNPGDOptimizer
from homopt.solvers import JCCDCOPFCVaRSolver, JCCDCOPFRobustScenarioSolver, JCCDCOPFSolver, solve_exact_result

from .reports import build_jcc_solver_method_row, build_jcc_solver_record


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

class _JCCINNPGDAdapter:
    """Adapter exposing the shared INN-PGD training contract for JCC-DC-OPF."""

    # The INN training penalty remains a smooth scenario-wise surrogate, but
    # optimizer repair and bisection must enforce the actual sampled chance
    # constraint used for final comparison.
    training_constraint_surrogate = "scenario_max_residual_v1"
    bisection_feasibility_metric = "sampled_chance_constraint_v1"
    final_evaluation_metric = "sampled_chance_feasibility_rate_v1"

    def __init__(self, problem):
        self.problem = problem
        self.nvar = int(problem.nvar)
        self.n_bus = int(problem.n_bus)
        self.n_scenarios = int(problem.n_scenarios)
        self.npara = self.n_bus
        self.n_qua = self.n_scenarios
        self.device = getattr(problem, "device", torch.device("cpu"))
        # The mapping, latent coordinates, and sampled scenarios must follow
        # the runtime dtype selected for the underlying JCC problem.
        self.dtype = getattr(problem, "dtype", None)
        if self.dtype is None:
            self.dtype = getattr(getattr(problem, "P_min", None), "dtype", torch.float32)
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
        scenario_batch = self.problem._coerce_scenario_batch(
            input_batch,
            x.device,
            batch_size=x.shape[0],
            dtype=x.dtype,
        )
        scenario_feasibility = self.problem.compute_scenario_feasibility(x, scenario_batch=scenario_batch)
        feasibility_rate = scenario_feasibility.mean(dim=1, keepdim=True)
        residual = (1.0 - float(self.problem.config["epsilon"])) - feasibility_rate
        return torch.clamp(residual, min=0) if clip else residual

    def check_feasibility(self, input_batch, x):
        return self.chance_violation(input_batch, x, clip=True)

    def violations(self, input_batch, x):
        return self.chance_violation(input_batch, x, clip=True)

    def objective(self, x):
        return self.problem.objective_x(x)

    def training_record_metadata(self):
        return {
            "constraint_surrogate": self.training_constraint_surrogate,
            "bisection_feasibility_metric": self.bisection_feasibility_metric,
            "final_evaluation_metric": self.final_evaluation_metric,
            "chance_epsilon": float(self.problem.config["epsilon"]),
        }


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
    return model_cfg, optimizer_cfg, train_cfg


def _jcc_inn_initial_latent(adapter, input_params, *, mode="center", radius=1.0, seed=2025):
    mode = str(mode).strip().lower()
    batch_size = int(input_params.shape[0])
    if mode == "center":
        return torch.zeros(batch_size, adapter.nvar, device=adapter.device, dtype=adapter.dtype)
    if mode == "sphere_boundary_random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        directions = torch.randn(batch_size, adapter.nvar, generator=generator, dtype=adapter.dtype)
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
    problem_args = {
        "problem_family": "jcc_opf",
        "n_bus": int(adapter.n_bus),
        "n_scenarios": int(adapter.n_scenarios),
        "pglib_case_name": problem.config.get("pglib_case_name"),
        "pglib_data_dir": problem.config.get("pglib_data_dir"),
    }
    training_resource = prepare_learning_mapping(
        "inn",
        adapter,
        problem_args=problem_args,
        model_args=model_cfg,
        train_args=train_cfg,
        save_dir=save_dir,
        retrain=retrain,
        runtime_device=adapter.device,
        runtime_dtype=adapter.dtype,
        ensure_results_save_freq=True,
    )
    model = training_resource["model"]
    training_record = training_resource["training_record"]
    model_path = training_resource["model_path"]
    record_path = training_resource["record_path"]
    expected_surrogate = getattr(adapter, "training_constraint_surrogate", None)
    recorded_surrogate = training_record.get("constraint_surrogate")
    if (
        (not retrain)
        and expected_surrogate is not None
        and recorded_surrogate is not None
        and recorded_surrogate != expected_surrogate
    ):
        print(
            "[HomOPT] Ignoring stale JCC INN checkpoint: "
            "training constraint surrogate changed; retraining."
        )
        training_resource = prepare_learning_mapping(
            "inn",
            adapter,
            problem_args=problem_args,
            model_args=model_cfg,
            train_args=train_cfg,
            save_dir=save_dir,
            retrain=True,
            runtime_device=adapter.device,
            runtime_dtype=adapter.dtype,
            ensure_results_save_freq=True,
        )
        model = training_resource["model"]
        training_record = training_resource["training_record"]
        model_path = training_resource["model_path"]
        record_path = training_resource["record_path"]
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
    chance_epsilon = float(problem.config["epsilon"])
    per_instance_chance_feasibility_rates = None
    worst_instance_chance_feasibility_rate = 0.0
    with torch.inference_mode():
        if finite_solution:
            scenario_feasibility = problem.compute_scenario_feasibility(x_opt, scenario_batch=inputs)
            per_instance_rates = scenario_feasibility.mean(dim=1)
            per_instance_chance_feasibility_rates = per_instance_rates.detach().cpu().tolist()
            chance_feasibility_rate = float(per_instance_rates.mean().item())
            worst_instance_chance_feasibility_rate = float(per_instance_rates.min().item())
            chance_violation = torch.clamp((1.0 - chance_epsilon) - per_instance_rates, min=0)
            violation = float(chance_violation.max().item())
        else:
            chance_feasibility_rate = 0.0
            violation = 1.0 - chance_epsilon
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
        "per_instance_chance_feasibility_rates": per_instance_chance_feasibility_rates,
        "worst_instance_chance_feasibility_rate": worst_instance_chance_feasibility_rate,
        "violation": violation,
        "feasible": status == "completed" and violation <= 1e-12,
        "training_constraint_surrogate": getattr(adapter, "training_constraint_surrogate", None),
        "bisection_feasibility_metric": getattr(adapter, "bisection_feasibility_metric", None),
        "final_evaluation_metric": getattr(adapter, "final_evaluation_metric", None),
        "chance_epsilon": chance_epsilon,
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


__all__ = [
    "_JCCINNPGDAdapter",
    "_jcc_solver_baseline_specs",
    "_normalize_jcc_inn_configs",
    "_resolve_enabled_baselines",
    "_run_jcc_inn_pgd_variant",
    "_run_jcc_solver_suite",
]
