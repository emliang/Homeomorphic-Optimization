"""Penalty and proximal-control helpers for ALM-family optimizers."""

from __future__ import annotations

import torch

from .config import _uses_inner_proximal_stages, _validate_inner_solver, _validate_positive_int, _validate_proximal_space


def _quadratic_proximal_point(step_origin, descent_direction, proximal_center, step_size, proximal_coef):
    gamma = float(proximal_coef)
    return (
        step_origin
        - step_size * descent_direction
        + step_size * gamma * proximal_center
    ) / (1.0 + step_size * gamma)


def _resolve_penalty_inner_solver_config(
    params,
    *,
    inner_iterations,
    default_proximal_space,
    inner_iteration_budget=None,
):
    inner_solver = _validate_inner_solver(params.get('inner_solver', 'gd'))
    inner_proximal_update_iterations = _validate_positive_int(
        params.get('inner_proximal_update_iterations', 1),
        name='inner_proximal_update_iterations',
    )
    budget = int(inner_iterations if inner_iteration_budget is None else inner_iteration_budget)
    if _uses_inner_proximal_stages(inner_solver) and inner_proximal_update_iterations > budget:
        raise ValueError(
            f"inner_proximal_update_iterations must be <= active inner iteration budget for {inner_solver}."
        )
    return {
        "inner_solver": inner_solver,
        "uses_inner_proximal_stages": _uses_inner_proximal_stages(inner_solver),
        "inner_proximal_update_iterations": inner_proximal_update_iterations,
        "inner_proximal_steps": max(1, budget // inner_proximal_update_iterations),
        "inner_proximal_coef": params.get('inner_proximal_coef', 0.0),
        "inner_proximal_space": _validate_proximal_space(
            params.get('inner_proximal_space', default_proximal_space)
        ),
    }


def _get_max_penalty(params):
    return params.get("max_penalty", 1e2)


def _update_penalty_coefficient(current_penalty, penalty_growth, max_penalty):
    return min(current_penalty * penalty_growth, max_penalty)


def _should_increase_penalty(previous_violation, current_violation, tolerance=0.999):
    previous = float(previous_violation.detach().max().item())
    current = float(current_violation.detach().max().item())
    return current >= previous * float(tolerance)


def _init_penalty_family_controls(owner, params, *, proximal_coef_default):
    owner.use_lagrangian = params.get('use_lagrangian', True)
    owner.use_penalty = params.get('use_penalty', True)
    owner.use_proximal = params.get('use_proximal', False)
    owner.dual_learning_rate = params['dual_learning_rate']
    owner.penalty_coef = params.get('penalty_coef', 1.0) if owner.use_penalty else None
    owner.penalty_growth = params.get('penalty_growth', 1.1)
    owner.max_penalty = _get_max_penalty(params)
    owner.max_dual = params.get('max_dual', 1e2)
    owner.proximal_coef = (
        params.get('proximal_coef', proximal_coef_default)
        if owner.use_proximal
        else None
    )
    owner.return_best_violation = bool(params.get('return_best_violation', False))


def _maybe_update_penalty(owner, previous_violation, current_violation):
    if (
        owner.use_penalty
        and owner.penalty_coef is not None
        and _should_increase_penalty(previous_violation, current_violation)
    ):
        owner.penalty_coef = _update_penalty_coefficient(
            owner.penalty_coef,
            owner.penalty_growth,
            owner.max_penalty,
        )
    return current_violation.detach()


def _clamp_dual_state(dual_state, *, max_dual, eq_cons=None, ineq_cons=None):
    if eq_cons is None and ineq_cons is None:
        return torch.clamp(dual_state, min=-max_dual, max=max_dual)
    if eq_cons is not None:
        dual_state[:, eq_cons] = torch.clamp(dual_state[:, eq_cons], min=-max_dual, max=max_dual)
    if ineq_cons is not None:
        dual_state[:, ineq_cons] = torch.clamp(dual_state[:, ineq_cons], min=0, max=max_dual)
    return dual_state


def _dual_update_scale(owner):
    penalty_coef = owner.penalty_coef if owner.penalty_coef is not None else 1.0
    return owner.dual_learning_rate * penalty_coef


def _update_penalty_inner_learning_rate(
    base_lr,
    current_lr,
    *,
    stepsize_rule,
    lr_decay,
    min_lr,
    current_error,
    best_error,
    iteration,
):
    if stepsize_rule == "adaptive" and current_error > best_error:
        return max(current_lr * lr_decay, min_lr)
    if stepsize_rule == "diminish":
        return max(base_lr / (iteration + 1) ** 0.5, min_lr)
    return current_lr


def _update_penalty_outer_learning_rate(
    current_lr,
    recorder,
    *,
    stepsize_rule,
    lr_decay,
    min_lr,
):
    if stepsize_rule == "adaptive" and recorder.violation_traj[-1].mean() > recorder.violation_traj[-2].mean():
        return max(current_lr * lr_decay, min_lr)
    return current_lr


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
