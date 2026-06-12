"""Run-record normalization and summaries for experiment benchmarks."""

from __future__ import annotations

import numpy as np

from homopt.experiments.common.runtime import final_scalar


SCALAR_TIMING_FIELDS = (
    "total_wall_time",
    "last_trans_time",
    "initial_transform_time",
    "final_transform_time",
    "initial_ip_time",
)
SEQUENCE_TIMING_FIELDS = ("outer_iter_time", "inner_iter_time", "solver_iter_time")

def _add_timing_summary(summary, record):
    for key in SCALAR_TIMING_FIELDS:
        if record.get(key) is not None:
            summary[key] = float(record.get(key) or 0.0)
    for key in SEQUENCE_TIMING_FIELDS:
        values = np.asarray(record.get(key, []), dtype=float).reshape(-1)
        if values.size:
            summary[key] = values.tolist()
            summary[f"total_{key}"] = float(values.sum())
            summary[f"mean_{key}"] = float(values.mean())


def summarize_run_record(record, *, reference_objective=None, include_total_iter_time=False):
    final_objective = final_scalar(record["obj_traj"])
    final_violation = final_scalar(record["cons_traj"])
    summary = {
        "final_objective": final_objective,
        "final_violation": final_violation,
        "iterations": int(len(record["iter_time"])),
    }
    if include_total_iter_time:
        summary["total_iter_time"] = float(sum(record["iter_time"]))
    _add_timing_summary(summary, record)
    if reference_objective is not None:
        if abs(float(reference_objective)) <= 1e-12:
            summary["objective_gap"] = None
        else:
            summary["objective_gap"] = float(abs(final_objective - float(reference_objective)) / abs(float(reference_objective)))
    return summary


def ensure_record_violation_split(problem, record):
    """Add explicit split violation trajectories when scalar scope is unambiguous."""

    if record.get("eq_violation_traj") is not None and record.get("ineq_violation_traj") is not None:
        return record
    extras = record.get("extras", {}) or {}
    if extras.get("eq_violation_traj") is not None and extras.get("ineq_violation_traj") is not None:
        return record
    full = record.get("cons_traj", extras.get("full_violation_traj", record.get("violation")))
    if full is None:
        return record
    full = np.asarray(full, dtype=float).reshape(-1)
    eq_cons = getattr(problem, "eq_cons", None)
    ineq_cons = getattr(problem, "ineq_cons", None)
    has_eq = bool(list(eq_cons or [])) if eq_cons is not None else int(getattr(problem, "n_eq", 0) or 0) > 0
    if ineq_cons is None and hasattr(problem, "ineq_cons"):
        has_ineq = False
    else:
        has_ineq = bool(list(ineq_cons or [])) if ineq_cons is not None else int(getattr(problem, "ncon", 0) or 0) > int(getattr(problem, "n_eq", 0) or 0)
    if has_ineq and not has_eq:
        record["ineq_violation_traj"] = full
        record["eq_violation_traj"] = np.zeros_like(full)
        record["violation_scope"] = "inequality"
    elif has_eq and not has_ineq:
        record["eq_violation_traj"] = full
        record["ineq_violation_traj"] = np.zeros_like(full)
        record["violation_scope"] = "equality"
    return record


def run_algorithm_suite(algorithms, run_record, summarize_record):
    records = {}
    summaries = {}
    for algorithm in algorithms:
        record = run_record(algorithm)
        records[algorithm] = record
        summaries[algorithm] = summarize_record(record)
    return records, summaries
