"""Constraint residual and violation helpers for optimizers."""

from __future__ import annotations

import torch

from .config import _supports_keyword


def _constraint_layout(problem):
    ncon = getattr(problem, "ncon", 0)
    n_eq = getattr(problem, "n_eq", 0)
    ineq_cons = getattr(problem, "ineq_cons", None)
    eq_cons = getattr(problem, "eq_cons", None)
    if ineq_cons is None and eq_cons is not None:
        dual_dim = n_eq if n_eq else ncon
    elif eq_cons is None:
        dual_dim = ncon
    else:
        dual_dim = ncon + n_eq
    return dual_dim, ineq_cons, eq_cons


def _constraint_residual(problem, x, *, equality_only=False):
    if equality_only:
        eq_constraint = getattr(problem, "eq_constraint_x", None)
        if eq_constraint is not None:
            return eq_constraint(x)
    constraint_x = problem.constraint_x
    if _supports_keyword(constraint_x, "eq_cons"):
        return constraint_x(x, clip=False, eq_cons=True)
    return constraint_x(x, clip=False)


def _constraint_violation(problem, x, *, equality_only=False):
    residual = _constraint_residual(problem, x, equality_only=equality_only)
    if residual.numel() == 0:
        return torch.zeros(1, 1, device=x.device, dtype=x.dtype)
    if equality_only:
        return residual.abs().max().view(1, -1)

    _, ineq_cons, eq_cons = _constraint_layout(problem)
    violation_terms = []
    if ineq_cons is not None:
        ineq_idx = list(ineq_cons)
        if ineq_idx:
            violation_terms.append(torch.clamp(residual[:, ineq_idx], min=0))
    elif eq_cons is None:
        violation_terms.append(torch.clamp(residual, min=0))

    if eq_cons is not None:
        eq_idx = list(eq_cons)
        if eq_idx:
            violation_terms.append(residual[:, eq_idx].abs())

    if not violation_terms:
        return torch.zeros(1, 1, device=x.device, dtype=x.dtype)
    return torch.cat(violation_terms, dim=1).max().view(1, -1)


def _constraint_violation_split(problem, x):
    residual = _constraint_residual(problem, x, equality_only=False)
    batch_size = int(x.shape[0]) if x.ndim > 1 else 1
    if residual.ndim == 0:
        residual = residual.view(1, 1)
    elif residual.ndim == 1:
        residual = residual.view(1, -1)
    zero = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    if residual.numel() == 0:
        return {
            "eq_violation": zero,
            "ineq_violation": zero,
            "full_violation": zero,
        }

    _, ineq_cons, eq_cons = _constraint_layout(problem)
    if ineq_cons is not None:
        ineq_idx = list(ineq_cons)
        if ineq_idx:
            if max(ineq_idx) >= residual.shape[1]:
                raise IndexError(
                    f"Inequality constraint indices {ineq_idx} exceed residual width {residual.shape[1]} "
                    f"for {type(problem).__name__}."
                )
            ineq_res = residual[:, ineq_idx]
            ineq_violation = torch.clamp(ineq_res, min=0).max(dim=1, keepdim=True).values
        else:
            ineq_violation = zero
    elif eq_cons is None:
        ineq_violation = torch.clamp(residual, min=0).max(dim=1, keepdim=True).values
    else:
        ineq_violation = zero

    if eq_cons is not None:
        eq_idx = list(eq_cons)
        if eq_idx:
            if max(eq_idx) < residual.shape[1]:
                eq_res = residual[:, eq_idx]
            else:
                eq_constraint = getattr(problem, "eq_constraint_x", None)
                if eq_constraint is None:
                    raise IndexError(
                        f"Equality constraint indices {eq_idx} exceed residual width {residual.shape[1]} "
                        f"for {type(problem).__name__}, and no eq_constraint_x is available."
                    )
                eq_res = eq_constraint(x)
                if eq_res.ndim == 0:
                    eq_res = eq_res.view(1, 1)
                elif eq_res.ndim == 1:
                    eq_res = eq_res.view(1, -1)
            eq_violation = eq_res.abs().max(dim=1, keepdim=True).values
        else:
            eq_violation = zero
    else:
        eq_violation = zero

    return {
        "eq_violation": eq_violation,
        "ineq_violation": ineq_violation,
        "full_violation": torch.maximum(eq_violation, ineq_violation),
    }


def _constraint_violation_summary(problem, x):
    split = _constraint_violation_split(problem, x)
    eq_violation = float(split["eq_violation"].detach().max().item())
    ineq_violation = float(split["ineq_violation"].detach().max().item())
    return eq_violation, ineq_violation, max(eq_violation, ineq_violation)


def _is_better_candidate(violation, objective, best_violation, best_objective, *, violation_tolerance=1e-12):
    return violation < best_violation or (
        violation <= best_violation + float(violation_tolerance)
        and objective < best_objective
    )


_CONSTRAINT_SPLIT_TRACK_NAMES = ("eq_violation", "ineq_violation", "full_violation")


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
