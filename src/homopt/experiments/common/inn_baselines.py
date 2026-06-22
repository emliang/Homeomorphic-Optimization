"""Shared INN-PGD baseline helpers.

The INN-PGD experiments compare the learned feasible-map route against three
fixed-instance Lagrangian/penalty baselines:

* ALM: Lagrangian multiplier update plus quadratic penalty.
* Penalty: external penalty method without multiplier update.
* Prox-Penalty: proximal penalty method without multiplier update.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from homopt.optim import LagrangianOptimizer

from .config import (
    align_outer_iterations_with_max_iterations,
    reject_iteration_budget_keys,
)
from .methods import build_penalty_method_params


INN_LAGRANGIAN_BASELINE_ORDER = ("ALM", "Penalty", "Prox-Penalty")


def normalize_inn_lagrangian_baselines(baselines):
    resolved = []
    for name in baselines or []:
        label = str(name)
        if label not in INN_LAGRANGIAN_BASELINE_ORDER:
            raise ValueError(
                f"Unsupported INN-PGD iterative baseline: {label}. "
                f"Supported: {list(INN_LAGRANGIAN_BASELINE_ORDER)}"
            )
        if label not in resolved:
            resolved.append(label)
    return resolved


def inn_lagrangian_baseline_flags(method):
    method = str(method)
    if method == "ALM":
        return {"use_lagrangian": True, "use_penalty": True, "use_proximal": False}
    if method == "Penalty":
        return {"use_lagrangian": False, "use_penalty": True, "use_proximal": False}
    if method == "Prox-Penalty":
        return {"use_lagrangian": False, "use_penalty": True, "use_proximal": True}
    raise ValueError(
        f"Unsupported INN-PGD iterative baseline: {method}. "
        f"Supported: {list(INN_LAGRANGIAN_BASELINE_ORDER)}"
    )


def build_inn_lagrangian_common_config(seed, config):
    config = dict(config or {})
    invalid = sorted({"outer_lr_decay", "outer_stepsize_rule", "outer_learning_rate"}.intersection(config))
    if invalid:
        raise ValueError(
            f"Unsupported INN-PGD lagrangian_baseline_config keys: {invalid}. "
            "Use learning_rate, lr_decay, and stepsize_rule; baseline builders map them to outer-loop fields."
        )
    return {
        "seed": int(seed),
        "learning_rate": float(config["learning_rate"]),
        "inner_learning_rate": float(config["inner_learning_rate"]),
        "max_running_time": float(config["max_running_time"]),
        "convergence_threshold": float(config["convergence_threshold"]),
        "opt": str(config["opt"]),
        "momentum": float(config["momentum"]),
        "lr_decay": float(config["lr_decay"]),
        "outer_lr_decay": float(config["lr_decay"]),
        "inner_lr_decay": float(config["inner_lr_decay"]),
        "min_lr": float(config["min_lr"]),
        "inner_min_lr": float(config["inner_min_lr"]),
        "stepsize_rule": str(config["stepsize_rule"]),
        "outer_stepsize_rule": str(config["stepsize_rule"]),
        "verbose": bool(config["verbose"]),
        "verbose_interval": int(config["verbose_interval"]),
        "proximal_coef": float(config["proximal_coef"]),
    }


def build_inn_lagrangian_method_params(seed, baseline_config, *, max_iterations=None):
    reject_iteration_budget_keys(baseline_config, context="INN-PGD lagrangian_baseline_config")
    baseline_config = align_outer_iterations_with_max_iterations(
        baseline_config,
        max_iterations=max_iterations,
        context="INN-PGD lagrangian_baseline_config",
    )
    common = build_inn_lagrangian_common_config(seed, baseline_config)
    outer_iterations = int(baseline_config["outer_iterations"])
    inner_iterations = int(baseline_config["inner_iterations"])
    kwargs = {
        "outer_iterations": outer_iterations,
        "inner_iterations": inner_iterations,
        "dual_learning_rate": float(baseline_config["dual_learning_rate"]),
        "penalty_coef": float(baseline_config["penalty_coef"]),
        "penalty_growth": float(baseline_config["penalty_growth"]),
        "proximal_coef": float(baseline_config["proximal_coef"]),
        "max_penalty": float(baseline_config["max_penalty"]),
        "max_dual": float(baseline_config["max_dual"]),
        "inner_stepsize_rule": str(baseline_config["inner_stepsize_rule"]),
        "inner_solver": str(baseline_config["inner_solver"]),
        "lagrangian_gradient": str(baseline_config["lagrangian_gradient"]),
    }
    return {
        "common": common,
        **{
            method: build_penalty_method_params(common, **inn_lagrangian_baseline_flags(method), **kwargs)
            for method in INN_LAGRANGIAN_BASELINE_ORDER
        },
    }


def inn_lagrangian_feasibility_tol(config):
    return float((config or {}).get("feasibility_tol", (config or {}).get("convergence_threshold", 1e-5)))


def _numpy_or_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _trajectory_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    values = list(value or [])
    if not values:
        return np.empty((0,))
    arrays = []
    for item in values:
        if torch.is_tensor(item):
            item = item.detach().cpu().numpy()
        arrays.append(np.asarray(item))
    return np.asarray(arrays)


def _sequence_or_empty(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return list(value)


def _final_sequence_scalar(value):
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        array = value.detach().reshape(-1).cpu()
        return float(array[-1].item()) if array.numel() else float("nan")
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[-1]) if array.size else float("nan")


def run_inn_lagrangian_baseline(problem, initial_point, base_args, *, method, seed):
    params = {
        **base_args,
        **inn_lagrangian_baseline_flags(method),
    }
    optimizer = LagrangianOptimizer(problem, params)
    run_start_time = perf_counter()
    x_opt, x_traj, obj_traj, cons_traj, per_iter_time = optimizer.optimize(
        initial_point=initial_point,
        verbose=bool(params.get("verbose", False)),
        seed=seed,
    )
    total_wall_time = perf_counter() - run_start_time
    x_np = _numpy_or_value(x_opt)
    return {
        "x_opt": x_np,
        "x_solved": x_np,
        "x_traj": _trajectory_numpy(x_traj),
        "obj_traj": _numpy_or_value(obj_traj),
        "cons_traj": _numpy_or_value(cons_traj),
        "iter_time": list(per_iter_time),
        "total_wall_time": total_wall_time,
        "outer_iter_time": _sequence_or_empty(getattr(optimizer, "last_outer_iter_time", [])),
        "inner_iter_time": _sequence_or_empty(getattr(optimizer, "last_inner_iter_time", [])),
        "final_objective": _final_sequence_scalar(obj_traj),
        "final_violation": _final_sequence_scalar(cons_traj),
        "final_first_order_lagrangian_gap": getattr(optimizer, "last_final_first_order_lagrangian_gap", None),
        "first_order_lagrangian_gap_traj": _sequence_or_empty(
            getattr(optimizer, "last_first_order_lagrangian_gap_traj", [])
        ),
        "lagrangian_gradient": params.get("lagrangian_gradient"),
    }
