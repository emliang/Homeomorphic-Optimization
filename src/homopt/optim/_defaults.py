"""Shared optimizer configuration defaults."""

from __future__ import annotations


def build_first_order_subproblem_defaults(common, *, outer_iterations, inner_iterations):
    """Default x-space Lagrangian subproblem config for PGD/FW fallbacks."""

    return {
        "learning_rate": 1e-3,
        "max_running_time": common["max_running_time"],
        "outer_iterations": outer_iterations,
        "inner_iterations": inner_iterations,
        "convergence_threshold": common["convergence_threshold"],
        "outer_stepsize_rule": common.get("outer_stepsize_rule", common["stepsize_rule"]),
        "inner_stepsize_rule": "constant",
        "inner_solver": "gd",
        "inner_proximal_update_iterations": 1,
        "inner_proximal_coef": 0.0,
        "inner_proximal_space": "x",
        "inner_restart": False,
        "acceleration_method": "none",
        "acceleration_space": "x",
        "outer_lr_decay": common.get("outer_lr_decay", common["lr_decay"]),
        "inner_lr_decay": common.get("inner_lr_decay", common["lr_decay"]),
        "dual_learning_rate": 1e-2,
        "penalty_coef": 10.0,
        "penalty_growth": 1.1,
        "proximal_coef": 0.1,
        "proximal_space": "x",
        "max_penalty": 1e2,
        "max_dual": 1e2,
        "use_lagrangian": True,
        "use_penalty": True,
        "use_proximal": False,
        "momentum": common.get("momentum", 0.0),
        "opt": common["opt"],
    }
