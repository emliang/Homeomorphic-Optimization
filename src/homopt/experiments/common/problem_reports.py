"""Problem-level objective, violation, and comparison row reports."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from homopt.experiments.common.comparison import make_comparison_row
from homopt.experiments.common.run_records import summarize_run_record


def _supports_keyword(callable_obj, keyword):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _problem_tensor_kwargs(problem):
    for attr_name in ("Q", "p", "L", "A_obj"):
        tensor = getattr(problem, attr_name, None)
        if torch.is_tensor(tensor):
            return {"device": tensor.device, "dtype": tensor.dtype}
    return {"device": getattr(problem, "device", torch.device("cpu")), "dtype": torch.float32}


def _coerce_decision_tensor(problem, decision):
    if decision is None:
        return None
    kwargs = _problem_tensor_kwargs(problem)
    x = torch.as_tensor(decision, **kwargs)
    if x.ndim == 1:
        x = x.view(1, -1)
    if x.ndim != 2 or x.shape[-1] != int(getattr(problem, "nvar", x.shape[-1])):
        return None
    return x


def _call_problem_constraint(problem, x, *, include_equalities, clip):
    constraint_x = problem.constraint_x
    kwargs = {}
    if _supports_keyword(constraint_x, "clip"):
        kwargs["clip"] = clip
    if _supports_keyword(constraint_x, "eq_cons"):
        kwargs["eq_cons"] = bool(include_equalities)
    return constraint_x(x, **kwargs)


def _max_or_zero(tensor):
    if tensor is None or tensor.numel() == 0:
        return 0.0
    return float(tensor.max().detach().cpu().item())


def constraint_violation_summary(problem, decision):
    """Compute reporting metrics with a consistent full/ineq/eq split."""

    x = _coerce_decision_tensor(problem, decision)
    if x is None:
        return {}

    residual = _call_problem_constraint(problem, x, include_equalities=True, clip=False)
    if not torch.is_tensor(residual):
        residual = torch.as_tensor(residual, **_problem_tensor_kwargs(problem))
    if residual.ndim == 1:
        residual = residual.view(1, -1)

    ineq_cons = getattr(problem, "ineq_cons", None)
    eq_cons = getattr(problem, "eq_cons", None)
    if ineq_cons is None and eq_cons is None:
        inequality_residual = residual
        equality_residual = None
    else:
        ineq_idx = [] if ineq_cons is None else list(ineq_cons)
        eq_idx = [] if eq_cons is None else list(eq_cons)
        inequality_residual = residual[:, ineq_idx] if ineq_idx else residual.new_zeros((residual.shape[0], 0))
        equality_residual = residual[:, eq_idx] if eq_idx else None

    inequality_violation = _max_or_zero(torch.clamp(inequality_residual, min=0))
    equality_violation = None if equality_residual is None else _max_or_zero(equality_residual.abs())
    split_values = [inequality_violation]
    if equality_violation is not None:
        split_values.append(equality_violation)
    return {
        "final_full_violation": max(split_values),
        "final_inequality_violation": inequality_violation,
        "final_equality_violation": equality_violation,
    }


def _residual_block_diagnostics(residual, active_tol):
    residual = residual.detach()
    if residual.ndim == 1:
        residual = residual.view(1, -1)
    if residual.numel() == 0:
        return {"total": 0, "active": 0, "violated": 0, "max_residual": 0.0}
    flat = residual.reshape(-1)
    return {
        "total": int(flat.numel()),
        "active": int((flat >= -active_tol).sum().detach().cpu().item()),
        "violated": int((flat > active_tol).sum().detach().cpu().item()),
        "max_residual": float(flat.max().detach().cpu().item()),
    }


def _empty_residual_block_diagnostics(x, active_tol):
    return _residual_block_diagnostics(x.new_zeros((1, 0)), active_tol)


def _sum_block_diagnostics(*blocks):
    nonempty = [block for block in blocks if block is not None]
    if not nonempty:
        return {"total": 0, "active": 0, "violated": 0, "max_residual": 0.0}
    total = int(sum(block.get("total", 0) for block in nonempty))
    max_values = [
        float(block.get("max_residual", 0.0))
        for block in nonempty
        if block.get("total", 0)
    ]
    return {
        "total": total,
        "active": int(sum(block.get("active", 0) for block in nonempty)),
        "violated": int(sum(block.get("violated", 0) for block in nonempty)),
        "max_residual": max(max_values) if max_values else 0.0,
    }


def _stiefel_group_norm_diagnostics(problem, x, active_tol):
    groups = getattr(problem, "group_norm_constraints", None) or ()
    if not groups:
        return None
    residuals = []
    for group in groups:
        indices = torch.as_tensor(group["flat_indices"], device=x.device, dtype=torch.long)
        residuals.append(torch.norm(x[:, indices], dim=-1, p=2) - float(group["budget"]))
    return _residual_block_diagnostics(torch.stack(residuals, dim=1), active_tol)


def convex_solution_diagnostics(problem, decision, *, active_tol=1e-5):
    """Report active-set structure at a ConvexOpt decision.

    Inequality residuals follow the convention residual <= 0. A constraint is
    counted as active when residual >= -active_tol and violated when residual >
    active_tol. Equalities are reported separately; all equality rows are active
    by definition and `violated` counts rows with |residual| > active_tol.
    """

    x = _coerce_decision_tensor(problem, decision)
    if x is None:
        return {"available": False, "active_tol": float(active_tol)}
    if x.shape[0] != 1:
        x = x[:1]

    diagnostics = {
        "available": True,
        "active_tol": float(active_tol),
        "nvar": int(getattr(problem, "nvar", x.shape[-1])),
        "counts_by_type": {},
    }
    counts = diagnostics["counts_by_type"]

    if getattr(problem, "A", None) is not None:
        counts["linear"] = _residual_block_diagnostics(torch.matmul(x, problem.A.T) - problem.b, active_tol)
    else:
        counts["linear"] = _empty_residual_block_diagnostics(x, active_tol)

    if getattr(problem, "Qq", None) is not None:
        q = 0.5 * torch.einsum("bi,mij,bj->bm", x, problem.Qq, x)
        p = torch.matmul(x, problem.pq.T)
        counts["quadratic"] = _residual_block_diagnostics(q + p - problem.bq, active_tol)
    else:
        counts["quadratic"] = _empty_residual_block_diagnostics(x, active_tol)

    if getattr(problem, "G", None) is not None:
        q = torch.norm(torch.einsum("mkn,bn->bmk", problem.G, x) + problem.h.unsqueeze(0), dim=-1, p=2)
        p = torch.matmul(x, problem.C.T) + problem.d
        counts["soc"] = _residual_block_diagnostics(q - p, active_tol)
    else:
        counts["soc"] = _empty_residual_block_diagnostics(x, active_tol)

    if getattr(problem, "L", None) is not None and getattr(problem, "U", None) is not None:
        counts["box_lower"] = _residual_block_diagnostics(problem.L - x, active_tol)
        counts["box_upper"] = _residual_block_diagnostics(x - problem.U, active_tol)
    else:
        counts["box_lower"] = _empty_residual_block_diagnostics(x, active_tol)
        counts["box_upper"] = _empty_residual_block_diagnostics(x, active_tol)
    box_counts = _sum_block_diagnostics(counts["box_lower"], counts["box_upper"])
    diagnostics["active_box_total"] = box_counts["active"]
    diagnostics["box_total"] = box_counts["total"]
    diagnostics["violated_box_total"] = box_counts["violated"]
    diagnostics["box_max_residual"] = box_counts["max_residual"]

    group_counts = _stiefel_group_norm_diagnostics(problem, x, active_tol)
    if group_counts is not None:
        diagnostics["active_group_total"] = group_counts["active"]
        diagnostics["group_total"] = group_counts["total"]
        diagnostics["violated_group_total"] = group_counts["violated"]
        diagnostics["group_max_residual"] = group_counts["max_residual"]

    eq_residual = None
    if getattr(problem, "n_eq", 0) > 0 and hasattr(problem, "eq_constraint_x"):
        eq_residual = problem.eq_constraint_x(x).detach()
    if eq_residual is None or eq_residual.numel() == 0:
        equality = {"total": 0, "active": 0, "violated": 0, "max_abs_residual": 0.0}
    else:
        eq_flat = eq_residual.abs().reshape(-1)
        equality = {
            "total": int(eq_flat.numel()),
            "active": int(eq_flat.numel()),
            "violated": int((eq_flat > active_tol).sum().detach().cpu().item()),
            "max_abs_residual": float(eq_flat.max().detach().cpu().item()),
        }
    counts["equality"] = equality

    inequality_blocks = [value for key, value in counts.items() if key != "equality"]
    diagnostics["active_inequality_total"] = int(sum(block["active"] for block in inequality_blocks))
    diagnostics["inequality_total"] = int(sum(block["total"] for block in inequality_blocks))
    diagnostics["violated_inequality_total"] = int(sum(block["violated"] for block in inequality_blocks))
    diagnostics["active_total"] = diagnostics["active_inequality_total"] + equality["active"]
    diagnostics["constraint_total"] = diagnostics["inequality_total"] + equality["total"]
    diagnostics["violated_total"] = diagnostics["violated_inequality_total"] + equality["violated"]
    return diagnostics


def objective_summary(problem, decision):
    """Compute the reported objective at the same decision used for violation."""

    x = _coerce_decision_tensor(problem, decision)
    if x is None:
        return {}

    objective = problem.objective_x(x)
    if not torch.is_tensor(objective):
        objective = torch.as_tensor(objective, **_problem_tensor_kwargs(problem))
    return {"final_objective": float(objective.detach().reshape(-1).mean().cpu().item())}


def summarize_single_problem_run_record(
    problem,
    record,
    *,
    reference_objective=None,
    include_total_iter_time=False,
):
    """Summarize a fixed-problem run using full violation for reporting.

    Optimizers may record an internal violation trajectory tailored to their
    update rule. For example, Hom-ALM records equality-only violation because
    the homeomorphism handles inequalities. Benchmark rows should still compare
    all methods using the same full/equality/inequality residual split.
    """

    summary = summarize_run_record(
        record,
        reference_objective=reference_objective,
        include_total_iter_time=include_total_iter_time,
    )
    objective_at_solution = objective_summary(problem, record.get("x_solved"))
    if objective_at_solution:
        summary["optimizer_reported_final_objective"] = summary["final_objective"]
        summary.update(objective_at_solution)
        if reference_objective is not None:
            if abs(float(reference_objective)) <= 1e-12:
                summary["objective_gap"] = None
            else:
                summary["objective_gap"] = float(
                    abs(summary["final_objective"] - float(reference_objective)) / abs(float(reference_objective))
                )
    split = constraint_violation_summary(problem, record.get("x_solved"))
    if split:
        summary["optimizer_reported_final_violation"] = summary["final_violation"]
        summary.update(split)
        summary["final_violation"] = split["final_full_violation"]
    if record.get("hom_map_forward_mode") is not None:
        summary["hom_map_forward_mode"] = record.get("hom_map_forward_mode")
    if record.get("hom_map_forward_mode_counts") is not None:
        summary["hom_map_forward_mode_counts"] = record.get("hom_map_forward_mode_counts")
    for key in ("hom_map_smooth", "hom_map_smooth_tie_tol", "hom_map_smooth_temperature"):
        if record.get(key) is not None:
            summary[key] = record.get(key)
    if record.get("final_first_order_lagrangian_gap") is not None:
        summary["final_first_order_lagrangian_gap"] = float(record["final_first_order_lagrangian_gap"])
    if record.get("first_order_lagrangian_gap_traj") is not None:
        values = np.asarray(record.get("first_order_lagrangian_gap_traj"), dtype=float).reshape(-1)
        if values.size:
            summary["first_order_lagrangian_gap_traj"] = values.tolist()
            summary["final_first_order_lagrangian_gap"] = float(values[-1])
    if record.get("lagrangian_gradient") is not None:
        summary["lagrangian_gradient"] = record.get("lagrangian_gradient")
    if record.get("hom_map_gradient") is not None:
        summary["hom_map_gradient"] = record.get("hom_map_gradient")
    return summary


def make_single_instance_solver_row(*, method, objective, violation, runtime_total, feasibility_tol=1e-5, **extras):
    extras.setdefault("total_wall_time", runtime_total)
    return make_comparison_row(
        route="solver",
        method=method,
        objective=objective,
        feasible=(violation is not None and float(violation) <= float(feasibility_tol)),
        violation=violation,
        runtime_total=runtime_total,
        **extras,
    )
