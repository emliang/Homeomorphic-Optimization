"""Trace normalization and metric extraction for visualization."""

from __future__ import annotations

import numpy as np
import torch

from .evaluation import problem_tensor_kwargs


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def record_trajectory(record, *, key="x_traj", fallback_key="x_solved", decision_dim=2):
    traj = as_numpy(record.get(key, []))
    if traj.size == 0:
        solved = as_numpy(record.get(fallback_key, []))
        return solved.reshape(1, -1) if solved.size else np.empty((0, decision_dim))
    return traj.reshape(-1, traj.shape[-1])


def record_time_axis(record, n_values):
    per_iter = np.asarray(record.get("iter_time", []), dtype=float).reshape(-1)
    if per_iter.size == n_values - 1:
        return np.concatenate([[0.0], np.cumsum(per_iter)])
    if per_iter.size == n_values:
        return np.cumsum(per_iter)
    if per_iter.size == 0 and n_values == 1:
        return np.zeros(1, dtype=float)
    raise ValueError(
        f"Cannot build time axis: {n_values} metric values but {per_iter.size} iter_time values. "
        "Rerun the experiment to regenerate explicit timing records."
    )


def problem_has_equalities(problem):
    eq_cons = getattr(problem, "eq_cons", None)
    if eq_cons is not None:
        return len(list(eq_cons)) > 0
    return int(getattr(problem, "n_eq", 0) or 0) > 0


def split_problem_metrics(problem, x_values):
    """Evaluate objective and split constraint violations along decisions."""

    if x_values.size == 0:
        empty = np.asarray([], dtype=float)
        return {
            "objective": empty,
            "equality_violation": empty,
            "inequality_violation": empty,
            "full_violation": empty,
        }

    x_tensor = torch.as_tensor(x_values, **problem_tensor_kwargs(problem))
    with torch.no_grad():
        objective = problem.objective_x(x_tensor).detach().reshape(-1).cpu().numpy()
        constraint_x = problem.constraint_x
        kwargs = {"clip": False}
        if getattr(problem, "eq_cons", None) is not None:
            kwargs["eq_cons"] = True
        residual = constraint_x(x_tensor, **kwargs)
        if residual.ndim == 1:
            residual = residual.view(1, -1)
        ineq_idx = list(getattr(problem, "ineq_cons", range(residual.shape[1])) or [])
        eq_idx = list(getattr(problem, "eq_cons", []) or [])
        ineq_res = residual[:, ineq_idx] if ineq_idx else residual.new_zeros((residual.shape[0], 0))
        eq_res = residual[:, eq_idx] if eq_idx else residual.new_zeros((residual.shape[0], 0))
        ineq = (
            torch.clamp(ineq_res, min=0).max(dim=1).values
            if ineq_res.numel()
            else residual.new_zeros(residual.shape[0])
        )
        eq = eq_res.abs().max(dim=1).values if eq_res.numel() else residual.new_zeros(residual.shape[0])
    ineq_np = ineq.detach().cpu().numpy()
    eq_np = eq.detach().cpu().numpy()
    return {
        "objective": objective,
        "equality_violation": eq_np,
        "inequality_violation": ineq_np,
        "full_violation": np.maximum(eq_np, ineq_np),
    }


def build_convex_2d_traces(problem, records, algorithms):
    traces = {}
    for method in algorithms:
        record = records[method]
        trajectory = record_trajectory(record, key="x_traj", fallback_key="x_solved", decision_dim=2)
        z_trajectory = record_trajectory(record, key="z_traj", fallback_key=None, decision_dim=2)
        metrics = split_problem_metrics(problem, trajectory)
        traces[method] = {
            "trajectory": trajectory,
            "z_trajectory": z_trajectory,
            "time": record_time_axis(record, len(metrics["objective"])),
            **metrics,
        }
    return traces


def trace_payload(traces):
    return {
        method: {
            "time": trace["time"].tolist(),
            "objective": trace["objective"].tolist(),
            "equality_violation": trace["equality_violation"].tolist(),
            "inequality_violation": trace["inequality_violation"].tolist(),
            "full_violation": trace["full_violation"].tolist(),
            "trajectory": trace["trajectory"].tolist(),
            "z_trajectory": trace.get("z_trajectory", np.empty((0, 2))).tolist(),
        }
        for method, trace in traces.items()
    }


def reference_objective_from_payload(payload):
    if payload.get("reference_objective") is not None:
        return payload.get("reference_objective")
    for row in payload.get("comparison_rows", []):
        if row.get("method") == "ConvexSolver":
            return row.get("objective")
    return None


__all__ = [
    "as_numpy",
    "build_convex_2d_traces",
    "problem_has_equalities",
    "record_time_axis",
    "record_trajectory",
    "reference_objective_from_payload",
    "split_problem_metrics",
    "trace_payload",
]
