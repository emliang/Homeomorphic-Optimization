"""JCC benchmark problem construction helpers."""

from __future__ import annotations

from homopt.experiments.common.config import merged, normalize_jcc_problem_config
from homopt.problems import JCCDCOPFProblem, bind_singleton_problem_instance


def normalize_jcc_linear_problem_config(
    *,
    problem_family="jcc_linear",
    seed,
    n_var,
    n_input_dim,
    n_ineq,
    n_scenarios,
    epsilon,
    problem_config=None,
):
    merged_problem = merged(None, problem_config)
    resolved_family = str(
        merged_problem.pop("problem_family", merged_problem.pop("family", problem_family))
    ).strip().lower()
    if resolved_family not in {"jcc_linear", "jccim"}:
        raise ValueError(f"Unsupported JCC chance problem family: {resolved_family}")
    return {
        "problem_family": resolved_family,
        "n_var": int(n_var),
        "n_input_dim": int(n_input_dim),
        "n_ineq": int(n_ineq),
        "n_scenarios": int(n_scenarios),
        "epsilon": float(epsilon),
        "seed": int(seed),
        **dict(merged_problem or {}),
    }


def _normalize_jcc_linear_solver_group(default_solver_name, default_config, payload):
    translated_payload = merged({}, payload)
    invalid = sorted({"name", "options"}.intersection(translated_payload))
    if invalid:
        raise ValueError(
            f"Unsupported solver_configs keys: {invalid}. "
            "Use solver or solver_name with solver_options."
        )
    cfg = merged(default_config, translated_payload)
    solver_name = cfg.pop("solver_name", cfg.get("solver", default_solver_name))
    cfg["solver"] = solver_name
    cfg["solver_options"] = dict(cfg.get("solver_options") or {})
    return cfg


def normalize_jcc_linear_solver_configs(
    *,
    solver_configs=None,
):
    canonical = {
        "mixed_integer": _normalize_jcc_linear_solver_group(
            "GUROBI",
            {"solver_options": {}, "verbose": False, "M": 1000.0},
            None,
        ),
        "cvar": _normalize_jcc_linear_solver_group(
            "MOSEK",
            {"solver_options": {}, "verbose": False},
            None,
        ),
        "scenario": _normalize_jcc_linear_solver_group(
            "MOSEK",
            {"solver_options": {}, "verbose": False},
            None,
        ),
    }
    if solver_configs:
        for name, cfg in dict(solver_configs).items():
            if name not in canonical:
                raise ValueError(f"Unsupported JCC linear solver config group: {name}")
            canonical[name] = _normalize_jcc_linear_solver_group(
                canonical[name].get("solver", "MOSEK"),
                canonical[name],
                cfg,
            )
    return canonical


def _build_jcc_problem(
    *,
    num_bus,
    n_scenarios,
    epsilon,
    demand_std,
    seed,
    runtime_device,
    runtime_dtype,
    problem_config=None,
):
    """Build one JCC-DC-OPF instance with the requested runtime and data source."""

    problem_cfg = normalize_jcc_problem_config(
        n_scenarios=n_scenarios,
        epsilon=epsilon,
        demand_std=demand_std,
        seed=seed,
        problem_config=problem_config,
    )
    problem = JCCDCOPFProblem(num_bus=int(num_bus), config=problem_cfg).with_runtime(
        device=runtime_device,
        dtype=runtime_dtype,
    )
    return bind_singleton_problem_instance(problem, seed=seed)

__all__ = [
    "_build_jcc_problem",
    "normalize_jcc_linear_problem_config",
    "normalize_jcc_linear_solver_configs",
]
