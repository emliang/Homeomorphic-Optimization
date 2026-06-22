"""Shared helpers for learning-route refiners."""

from __future__ import annotations

import torch

from homopt.problems import normalize_constraint_violation


def max_violation(problem, input_params, y):
    residual = normalize_constraint_violation(problem.constraint_residual_xy(input_params, y, clip=False))
    return residual.max(dim=1, keepdim=True)[0]


def infeasible_mask(problem, input_params, y, tol):
    return (max_violation(problem, input_params, y) > float(tol)).view(-1)


def inverse_scale_decision(problem_family, input_params, y):
    if hasattr(problem_family, "inverse_scale"):
        return problem_family.inverse_scale(input_params, y)
    return inverse_scale_fixed_box(problem_family, y)


def inverse_scale_fixed_box(problem_family, y):
    lower = torch.as_tensor(problem_family.fixed_L, dtype=y.dtype, device=y.device)
    upper = torch.as_tensor(problem_family.fixed_U, dtype=y.dtype, device=y.device)
    span = torch.clamp(upper - lower, min=1e-8)
    return 2 * (y - lower) / span - 1


__all__ = ["infeasible_mask", "inverse_scale_decision", "inverse_scale_fixed_box", "max_violation"]
