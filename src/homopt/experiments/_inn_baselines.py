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

from ._benchmark_common import build_penalty_method_params


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
    return {
        "seed": int(seed),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
        "inner_learning_rate": float(config.get("inner_learning_rate", config.get("learning_rate", 1e-3))),
        "max_running_time": float(config.get("max_running_time", 300)),
        "convergence_threshold": float(config.get("convergence_threshold", 1e-6)),
        "opt": str(config.get("opt", "gd")),
        "momentum": float(config.get("momentum", 0.0)),
        "lr_decay": float(config.get("lr_decay", 0.9)),
        "outer_lr_decay": float(config.get("outer_lr_decay", config.get("lr_decay", 0.9))),
        "inner_lr_decay": float(config.get("inner_lr_decay", config.get("lr_decay", 0.9))),
        "stepsize_rule": str(config.get("stepsize_rule", "adaptive")),
        "outer_stepsize_rule": str(config.get("outer_stepsize_rule", config.get("stepsize_rule", "adaptive"))),
        "verbose": bool(config.get("verbose", False)),
        "verbose_interval": int(config.get("verbose_interval", 50)),
        "proximal_coef": float(config.get("proximal_coef", 0.1)),
    }


def build_inn_lagrangian_method_params(seed, baseline_config):
    baseline_config = dict(baseline_config or {})
    if "max_iterations" in baseline_config:
        raise ValueError("INN-PGD lagrangian baseline config uses outer_iterations; remove max_iterations.")
    common = build_inn_lagrangian_common_config(seed, baseline_config)
    outer_iterations = int(baseline_config.get("outer_iterations", 50))
    inner_iterations = int(baseline_config.get("inner_iterations", 10))
    kwargs = {
        "outer_iterations": outer_iterations,
        "inner_iterations": inner_iterations,
        "dual_learning_rate": float(baseline_config.get("dual_learning_rate", 1e-1)),
        "penalty_coef": float(baseline_config.get("penalty_coef", 10.0)),
        "penalty_growth": float(baseline_config.get("penalty_growth", 1.1)),
        "proximal_coef": float(baseline_config.get("proximal_coef", common["proximal_coef"])),
        "max_penalty": float(baseline_config.get("max_penalty", 1e2)),
        "max_dual": float(baseline_config.get("max_dual", 1e2)),
        "inner_stepsize_rule": str(baseline_config.get("inner_stepsize_rule", "constant")),
        "inner_solver": str(baseline_config.get("inner_solver", "gd")),
        "lagrangian_gradient": str(baseline_config.get("lagrangian_gradient", "explicit")),
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
