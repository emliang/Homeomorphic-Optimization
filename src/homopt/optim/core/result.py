"""Optimizer result packing and metric collection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class OptimizerRunResult:
    """Canonical optimizer-loop result before benchmark serialization."""

    final_decision: Any
    decision_trajectory: Any
    objective_trajectory: Any
    violation_trajectory: Any
    per_iter_time: Any
    last_trans_time: float = 0.0
    latent_trajectory: Any | None = None
    extra_metrics: dict[str, Any] | None = None
    include_transform_time: bool = False

    def __iter__(self):
        yield self.final_decision
        yield self.decision_trajectory
        yield self.objective_trajectory
        yield self.violation_trajectory
        yield self.per_iter_time
        if self.include_transform_time:
            yield self.last_trans_time


@dataclass
class ParametricOptimizerRunResult:
    """Canonical result for batched parametric optimizers with latent trajectories."""

    final_decision: Any
    decision_trajectory: Any
    latent_trajectory: Any
    objective_trajectory: Any
    violation_trajectory: Any
    per_iter_time: Any

    def __iter__(self):
        yield self.final_decision
        yield self.decision_trajectory
        yield self.latent_trajectory
        yield self.objective_trajectory
        yield self.violation_trajectory
        yield self.per_iter_time


def _pack_run_result(
    final_decision,
    decision_trajectory,
    objective_trajectory,
    violation_trajectory,
    per_iter_time,
    *,
    last_trans_time=0.0,
    latent_trajectory=None,
    extra_metrics=None,
):
    result = {
        'x_traj': decision_trajectory,
        'obj_traj': objective_trajectory.detach().cpu().numpy(),
        'cons_traj': violation_trajectory.detach().cpu().numpy(),
        'iter_time': per_iter_time,
        'last_trans_time': last_trans_time,
        'x_solved': final_decision.detach().cpu().numpy(),
    }
    if latent_trajectory is not None:
        result['z_traj'] = latent_trajectory
    if extra_metrics:
        result.update(extra_metrics)
    return result


def _pack_optimizer_run_result(run_result, *, extra_metrics=None):
    metrics = {}
    if run_result.extra_metrics:
        metrics.update(run_result.extra_metrics)
    if extra_metrics:
        metrics.update(extra_metrics)
    return _pack_run_result(
        run_result.final_decision,
        run_result.decision_trajectory,
        run_result.objective_trajectory,
        run_result.violation_trajectory,
        run_result.per_iter_time,
        last_trans_time=run_result.last_trans_time,
        latent_trajectory=run_result.latent_trajectory,
        extra_metrics=metrics,
    )


def _normalize_optimizer_result(raw_result, *, optimizer, extra_metrics=None):
    """Normalize optimizer outputs into the benchmark record dictionary."""

    del optimizer
    if isinstance(raw_result, dict):
        if extra_metrics:
            raw_result = dict(raw_result)
            raw_result.update(extra_metrics)
        return raw_result
    if not isinstance(raw_result, OptimizerRunResult):
        raise TypeError("Optimizer.optimize must return OptimizerRunResult or a benchmark record dict.")
    return _pack_optimizer_run_result(raw_result, extra_metrics=extra_metrics)


def _sequence_as_metric(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return list(value)


def _last_sequence_scalar(value):
    if value is None or len(value) == 0:
        return None
    if torch.is_tensor(value):
        return float(value.reshape(-1)[-1].detach().cpu().item())
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return None
    return float(array[-1])


def _store_first_order_lagrangian_gap_metrics(owner, payload):
    trajectory = payload.get("first_order_lagrangian_gap")
    owner.last_first_order_lagrangian_gap_traj = trajectory
    owner.last_final_first_order_lagrangian_gap = _last_sequence_scalar(trajectory)


def _store_constraint_violation_split_metrics(owner, payload):
    owner.last_eq_violation_traj = payload.get("eq_violation")
    owner.last_ineq_violation_traj = payload.get("ineq_violation")
    owner.last_full_violation_traj = payload.get("full_violation")


_EQ_CVXPY_ALGORITHMS = ('Penalty-EQ', 'Prox-Penalty-EQ', 'ALM-EQ', 'Prox-ALM-EQ')
_OPTIMIZER_SEQUENCE_METRICS = (
    ("last_outer_iter_time", "outer_iter_time"),
    ("last_inner_iter_time", "inner_iter_time"),
    ("last_solver_iter_time", "solver_iter_time"),
    ("last_first_order_lagrangian_gap_traj", "first_order_lagrangian_gap_traj"),
    ("last_eq_violation_traj", "eq_violation_traj"),
    ("last_ineq_violation_traj", "ineq_violation_traj"),
    ("last_full_violation_traj", "full_violation_traj"),
)
_OPTIMIZER_SCALAR_METRICS = (
    ("last_initial_transform_time", "initial_transform_time"),
    ("last_final_transform_time", "final_transform_time"),
    ("last_final_first_order_lagrangian_gap", "final_first_order_lagrangian_gap"),
)
_OPTIMIZER_ROUTE_ATTRS = (
    "lagrangian_gradient",
    "hom_map_gradient",
)


def _collect_optimizer_timing_metrics(optimizer):
    metrics = {}
    for attr_name, metric_name in _OPTIMIZER_SEQUENCE_METRICS:
        value = getattr(optimizer, attr_name, None)
        if value is not None:
            metrics[metric_name] = _sequence_as_metric(value)
    for attr_name, metric_name in _OPTIMIZER_SCALAR_METRICS:
        value = getattr(optimizer, attr_name, None)
        if value is not None:
            metrics[metric_name] = float(value)
    for attr_name in _OPTIMIZER_ROUTE_ATTRS:
        value = getattr(optimizer, attr_name, None)
        if value is not None:
            metrics[attr_name] = value
    return metrics


__all__ = [
    "OptimizerRunResult",
    "ParametricOptimizerRunResult",
    *[name for name in globals() if name.startswith("_") and not name.startswith("__")],
]
