"""Trace normalization helpers for INN visualizations."""

from __future__ import annotations

import numpy as np


def _as_numpy_1d(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=float).reshape(-1)


def _trajectory_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    array = np.asarray(value, dtype=float)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim == 1:
        return np.empty((0, 2))
    if array.ndim == 2 and array.shape[1] >= 2:
        return array[:, :2]
    return np.empty((0, 2))


def _batched_history_for_instance(value, n_samples, instance_idx):
    array = _as_numpy_1d(value)
    if n_samples <= 1:
        return array
    if array.size % int(n_samples) != 0:
        raise ValueError(
            f"Batched history length {array.size} is not divisible by num_test_instance={int(n_samples)}."
        )
    return array.reshape(array.size // int(n_samples), int(n_samples))[:, int(instance_idx)]


def _inn_record_for_instance(record, n_samples, instance_idx):
    x_traj = record.get("x_trajectory", [])
    z_traj = record.get("z_trajectory", [])
    if hasattr(x_traj, "detach"):
        x_traj = x_traj.detach().cpu()
    if hasattr(z_traj, "detach"):
        z_traj = z_traj.detach().cpu()
    x_array = np.asarray(x_traj, dtype=float)
    z_array = np.asarray(z_traj, dtype=float)
    if x_array.ndim >= 2 and n_samples > 1:
        if x_array.shape[0] % int(n_samples) != 0:
            raise ValueError(
                f"INN x trajectory length {x_array.shape[0]} is not divisible by num_test_instance={int(n_samples)}."
            )
        x_array = x_array.reshape(x_array.shape[0] // int(n_samples), int(n_samples), *x_array.shape[1:])[:, int(instance_idx)]
    if z_array.ndim >= 2 and n_samples > 1:
        if z_array.shape[0] % int(n_samples) != 0:
            raise ValueError(
                f"INN z trajectory length {z_array.shape[0]} is not divisible by num_test_instance={int(n_samples)}."
            )
        z_array = z_array.reshape(z_array.shape[0] // int(n_samples), int(n_samples), *z_array.shape[1:])[:, int(instance_idx)]
    if x_array.ndim == 3 and x_array.shape[1] == 1:
        x_array = x_array[:, 0, :]
    if z_array.ndim == 3 and z_array.shape[1] == 1:
        z_array = z_array[:, 0, :]
    obj = _batched_history_for_instance(record.get("obj_trajectory", []), n_samples, instance_idx)
    cons = _batched_history_for_instance(record.get("cons_trajectory", []), n_samples, instance_idx)
    iter_time = _as_numpy_1d(record.get("per_iter_time", record.get("iter_time", [])))
    return {
        "x_traj": x_array,
        "z_traj": z_array,
        "obj_traj": obj,
        "cons_traj": cons,
        "iter_time": iter_time,
        "total_wall_time": record.get("total_wall_time"),
    }


def _method_trace(record):
    obj = _as_numpy_1d(record.get("obj_traj", record.get("obj_trajectory", [])))
    cons = _as_numpy_1d(record.get("cons_traj", record.get("cons_trajectory", [])))
    iter_time = _as_numpy_1d(record.get("iter_time", record.get("per_iter_time", record.get("outer_iter_time", []))))
    n_values = max(obj.size, cons.size)
    if n_values == 0:
        return None
    if iter_time.size == n_values - 1:
        time = np.concatenate([[0.0], np.cumsum(iter_time)])
    elif iter_time.size == n_values:
        time = np.cumsum(iter_time)
    else:
        raise ValueError(
            f"Record timing length {iter_time.size} is incompatible with {n_values} metric values."
        )
    if obj.size == 0:
        obj = np.full(n_values, np.nan)
    if cons.size == 0:
        cons = np.full(n_values, np.nan)
    return {
        "time": time,
        "objective": obj,
        "violation": np.maximum(cons, 0.0),
    }


def _best_known_reference_objective(traces):
    finals = []
    for trace in traces.values():
        objective = np.asarray(trace.get("objective", []), dtype=float).reshape(-1)
        finite = objective[np.isfinite(objective)]
        if finite.size:
            finals.append(float(finite[-1]))
    return min(finals) if finals else None


def _add_objective_gap_metrics(traces, *, reference_objective=None):
    if reference_objective is None:
        reference_objective = _best_known_reference_objective(traces)
    if reference_objective is None or not np.isfinite(reference_objective):
        return {}, None
    denom = max(abs(float(reference_objective)), 1e-12)
    gap_traces = {}
    for method, trace in traces.items():
        objective = np.asarray(trace.get("objective", []), dtype=float).reshape(-1)
        if objective.size == 0:
            continue
        gap = np.abs(objective - float(reference_objective)) / denom
        violation = np.asarray(trace.get("violation", np.zeros_like(gap)), dtype=float).reshape(-1)
        if violation.size != gap.size:
            limit = min(violation.size, gap.size)
            gap = gap[:limit]
            violation = violation[:limit]
        gap_traces[method] = {
            **trace,
            "time": np.asarray(trace.get("time", []), dtype=float).reshape(-1)[: gap.size],
            "objective_gap": gap,
            "objective_gap_plus_violation": gap + np.maximum(violation, 0.0),
        }
    return gap_traces, float(reference_objective)


def _outer_per_iter_time_stats(record):
    for key in ("outer_iter_time", "per_iter_time", "iter_time"):
        values = _as_numpy_1d(record.get(key, []))
        values = values[np.isfinite(values)]
        if values.size:
            return float(values.mean()), float(values.std(ddof=0))
    runtime = record.get("total_wall_time", record.get("runtime_total"))
    if runtime is None:
        return float("nan"), float("nan")
    trace = _method_trace(record)
    if trace is None:
        return float("nan"), float("nan")
    iterations = max(int(np.asarray(trace["objective"]).reshape(-1).size) - 1, 1)
    return float(runtime) / float(iterations), 0.0
