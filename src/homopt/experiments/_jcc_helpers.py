"""Shared JCC comparison helpers reused across benchmark entrypoints."""

from __future__ import annotations

import torch

from homopt.experiments._comparison_helpers import make_comparison_row, timing_extras_from_iter_time


def build_jcc_method_row(problem, *, route, method, objective, x_opt, runtime_total, status, **extras):
    if extras.get("chance_feasibility_rate") is not None:
        chance_feasibility_rate = float(extras.pop("chance_feasibility_rate"))
        violation = extras.pop("violation", max(0.0, (1.0 - chance_feasibility_rate) - float(problem.config["epsilon"])))
        feasible = bool(extras.pop("feasible", violation <= 1e-12))
        return make_comparison_row(
            route=route,
            method=method,
            objective=objective,
            feasible=feasible,
            violation=violation,
            runtime_total=runtime_total,
            status=status,
            chance_feasibility_rate=chance_feasibility_rate,
            **extras,
        )
    if x_opt is None:
        chance_feasibility_rate = 0.0
        violation = None
        feasible = False
    else:
        device = getattr(problem, "device", None)
        x_tensor = torch.as_tensor(x_opt, dtype=torch.float32, device=device).view(1, -1)
        chance_feasibility_rate = float(problem.compute_scenario_feasibility(x_tensor).mean().item())
        violation = max(0.0, (1.0 - chance_feasibility_rate) - float(problem.config["epsilon"]))
        feasible = violation <= 1e-12
    return make_comparison_row(
        route=route,
        method=method,
        objective=objective,
        feasible=feasible,
        violation=violation,
        runtime_total=runtime_total,
        status=status,
        chance_feasibility_rate=chance_feasibility_rate,
        **extras,
    )


def build_jcc_solver_record(result):
    return {
        "num_bus": int(result["num_bus"]),
        "n_scenarios": int(result["n_scenarios"]),
        "solver": str(result["solver"]),
        "status": str(result["status"]),
        "runtime_sec": float(result["runtime_sec"]),
        "objective": result["objective"],
        "feasibility_rate": result["feasibility_rate"],
        "chance_satisfied": bool(result["chance_satisfied"]),
    }


def build_jcc_solver_method_row(problem, result):
    if result.get("x_optimal") is not None and hasattr(problem, "compute_scenario_feasibility"):
        return build_jcc_method_row(
            problem,
            route="solver",
            method=result["solver"],
            objective=result.get("objective_value"),
        x_opt=result.get("x_optimal"),
        runtime_total=result.get("runtime_sec", 0.0),
        status=result.get("status"),
        total_wall_time=result.get("runtime_sec"),
    )
    return make_comparison_row(
        route="solver",
        method=result["solver"],
        objective=result.get("objective"),
        feasible=result.get("chance_satisfied", result.get("chance_constraint_satisfied", False)),
        violation=max(0.0, (1.0 - float(result.get("feasibility_rate", 0.0) or 0.0)) - float(problem.config["epsilon"]))
        if hasattr(problem, "config")
        else None,
        runtime_total=result.get("runtime_sec", 0.0),
        status=result.get("status"),
        total_wall_time=result.get("runtime_sec"),
        chance_feasibility_rate=float(result.get("feasibility_rate", 0.0) or 0.0),
        num_bus=int(result["num_bus"]),
        n_scenarios=int(result["n_scenarios"]),
    )


def build_jcc_iterative_method_rows(problem, results):
    return [
        build_jcc_method_row(
            problem,
            route="iterative",
            method=method,
            objective=payload.get("final_objective"),
            x_opt=payload.get("x_opt"),
            runtime_total=payload.get("total_wall_time", sum(payload.get("iter_time", []))),
            status="completed",
            **timing_extras_from_iter_time(
                payload.get("iter_time", []),
                total_wall_time=payload.get("total_wall_time"),
                inner_iter_time=payload.get("inner_iter_time"),
            ),
        )
        for method in ("INN-PGD", "ALM", "Penalty", "Prox-Penalty")
        if method in results
        for payload in (results[method],)
    ]


def build_jcc_comparison_metrics(*, num_bus, n_scenarios, solver_result, cvar_result, scenario_result, results):
    metrics = {
        "num_bus": int(num_bus),
        "n_scenarios": int(n_scenarios),
        "solver_status": solver_result["status"],
        "solver_feasibility_rate": solver_result["feasibility_rate"],
        "cvar_status": cvar_result["status"],
        "cvar_feasibility_rate": cvar_result["feasibility_rate"],
        "cvar_objective": cvar_result["objective_value"],
        "scenario_status": scenario_result["status"],
        "scenario_feasibility_rate": scenario_result["feasibility_rate"],
        "scenario_objective": scenario_result["objective_value"],
        "enabled_lagrangian_baselines": [
            method for method in ("ALM", "Penalty", "Prox-Penalty") if method in results
        ],
        **(
            {
                "inn_pgd_final_violation": results["INN-PGD"]["final_violation"],
                "inn_pgd_objective": results["INN-PGD"]["final_objective"],
                "inn_pgd_feasibility_rate": results["INN-PGD"]["chance_feasibility_rate"],
            }
            if "INN-PGD" in results
            else {}
        ),
    }
    if "ALM" in results:
        metrics["alm_final_violation"] = results["ALM"]["final_violation"]
    if "Penalty" in results:
        metrics["penalty_final_violation"] = results["Penalty"]["final_violation"]
    if "Prox-Penalty" in results:
        metrics["prox_penalty_final_violation"] = results["Prox-Penalty"]["final_violation"]
    return metrics
