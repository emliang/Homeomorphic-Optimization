"""Shared comparison-row helpers reused across experiment mainlines."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


COMPARISON_ROW_FIELDS = (
    "row_kind",
    "route",
    "method",
    "objective",
    "feasible",
    "violation",
    "runtime",
    "iterations",
    "avg_outer_iter_time",
    "mean_inner_iter_time",
    "mean_solver_iter_time",
    "initial_ip_time",
    "initial_transform_time",
    "final_transform_time",
    "last_trans_time",
    "equality_violation",
    "inequality_violation",
    "objective_mean",
    "feasibility_rate",
    "violation_mean",
    "runtime_mean",
    "runtime_total",
)
COMPARISON_METRIC_FIELDS = (
    "objective_mean",
    "feasibility_rate",
    "violation_mean",
    "runtime_mean",
    "runtime_total",
)
INSTANCE_TIMING_FIELDS = (
    "total_wall_time",
    "total_iter_time",
    "total_outer_iter_time",
    "mean_outer_iter_time",
    "total_inner_iter_time",
    "mean_inner_iter_time",
    "total_solver_iter_time",
    "mean_solver_iter_time",
    "initial_ip_time",
    "initial_transform_time",
    "final_transform_time",
    "last_trans_time",
)
SUMMARY_TIMING_MEAN_FIELDS = (
    "total_wall_time",
    "avg_outer_iter_time",
    "mean_inner_iter_time",
    "mean_solver_iter_time",
    "initial_ip_time",
    "initial_transform_time",
    "final_transform_time",
    "last_trans_time",
)


def timing_extras_from_iter_time(
    iter_time=None,
    *,
    total_wall_time=None,
    inner_iter_time=None,
    solver_iter_time=None,
):
    """Build shared timing extras for comparison rows from per-iteration lists."""

    extras = {}
    iter_values = np.asarray(iter_time if iter_time is not None else [], dtype=float).reshape(-1)
    if iter_values.size:
        extras["total_iter_time"] = float(iter_values.sum())
        extras["total_outer_iter_time"] = float(iter_values.sum())
        extras["avg_outer_iter_time"] = float(iter_values.mean())
    if total_wall_time is not None:
        extras["total_wall_time"] = float(total_wall_time)
    for key, values in (
        ("inner_iter_time", inner_iter_time),
        ("solver_iter_time", solver_iter_time),
    ):
        array = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
        if array.size:
            extras[f"total_{key}"] = float(array.sum())
            extras[f"mean_{key}"] = float(array.mean())
    return extras


def make_comparison_row(
    *,
    route,
    method,
    objective,
    feasible,
    violation,
    runtime_total,
    runtime_mean=None,
    **extras,
):
    objective_value = None if objective is None else float(objective)
    feasible_value = bool(feasible)
    violation_value = None if violation is None else float(violation)
    runtime_value = float(runtime_total)
    return {
        "row_kind": "instance",
        "route": str(route),
        "method": str(method),
        "objective": objective_value,
        "feasible": feasible_value,
        "violation": violation_value,
        "runtime": runtime_value,
        "objective_mean": objective_value,
        "feasibility_rate": 1.0 if feasible_value else 0.0,
        "violation_mean": violation_value,
        "runtime_mean": runtime_value if runtime_mean is None else float(runtime_mean),
        "runtime_total": runtime_value,
        **extras,
    }


def comparison_metric_source(summary):
    if isinstance(summary, dict) and isinstance(summary.get("metrics"), dict):
        return summary["metrics"]
    return summary


def extract_comparison_metrics(summary):
    source = comparison_metric_source(summary)
    return {name: source.get(name) for name in COMPARISON_METRIC_FIELDS}


def select_best_comparison_row(rows):
    rows = list(rows)
    feasible_rows = [row for row in rows if (row.get("feasibility_rate") or 0.0) > 0.0]
    primary_rows = feasible_rows or rows

    def _objective_key(row):
        objective = row.get("objective_mean")
        if objective is None:
            return float("inf")
        objective = float(objective)
        return objective if np.isfinite(objective) else float("inf")

    return min(
        primary_rows,
        key=_objective_key,
    )


def summarize_comparison_rows(rows):
    best_row = select_best_comparison_row(rows)
    return {
        "objective": best_row.get("objective", best_row.get("objective_mean")),
        "feasible": bool(best_row.get("feasible", (best_row.get("feasibility_rate") or 0.0) > 0.0)),
        "best_method": {
            "route": best_row["route"],
            "method": best_row["method"],
        },
    }


def build_comparison_views(rows, *, summary_rows=None):
    rows = list(rows)
    if summary_rows is None:
        summary_rows = summarize_comparison_rows_by_method(rows)
    summary_rows = list(summary_rows)
    comparison_summary = summarize_comparison_rows(rows)
    views = {
        "objective": comparison_summary["objective"],
        "feasible": comparison_summary["feasible"],
        "instance_rows": rows,
        "method_summary_rows": summary_rows,
        "comparison_rows": rows,
        "comparison_summary": comparison_summary,
        "best_method": comparison_summary["best_method"],
    }
    views["comparison_summary_rows"] = summary_rows
    return views


def make_single_instance_summary_row(
    *,
    route,
    method,
    summary,
    objective_key="final_objective",
    violation_key="final_violation",
    runtime_key="total_wall_time",
    feasibility_tol=1e-5,
    **extras,
):
    objective = summary.get(objective_key)
    violation = summary.get(violation_key)
    runtime_total_value = summary.get(runtime_key)
    if runtime_total_value is None:
        raise ValueError(f"summary for {method} is missing required runtime key: {runtime_key}")
    runtime_total = float(runtime_total_value)
    feasible = False if violation is None else float(violation) <= float(feasibility_tol)
    row_extras = {
        "full_violation": summary.get("final_full_violation"),
        "equality_violation": summary.get("final_equality_violation"),
        "inequality_violation": summary.get("final_inequality_violation"),
        "optimizer_reported_objective": summary.get("optimizer_reported_final_objective"),
        "optimizer_reported_violation": summary.get("optimizer_reported_final_violation"),
        "iterations": summary.get("iterations"),
        "hom_map_forward_mode": summary.get("hom_map_forward_mode"),
        "hom_map_smooth": summary.get("hom_map_smooth"),
        "hom_map_smooth_tie_tol": summary.get("hom_map_smooth_tie_tol"),
        "hom_map_smooth_temperature": summary.get("hom_map_smooth_temperature"),
        "first_order_lagrangian_gap": summary.get("final_first_order_lagrangian_gap"),
    }
    row_extras.update({key: summary.get(key) for key in INSTANCE_TIMING_FIELDS})
    total_iter_time = summary.get("total_iter_time")
    iterations = summary.get("iterations")
    if total_iter_time is not None and iterations not in (None, 0):
        row_extras["avg_outer_iter_time"] = float(total_iter_time) / float(iterations)
    row_extras = {key: value for key, value in row_extras.items() if value is not None}
    row_extras.update(extras)
    return make_comparison_row(
        route=route,
        method=method,
        objective=objective,
        feasible=feasible,
        violation=violation,
        runtime_total=runtime_total,
        **row_extras,
    )


def build_single_instance_comparison_views(
    results,
    *,
    route,
    summary_rows=None,
    objective_key="final_objective",
    violation_key="final_violation",
    runtime_key="total_wall_time",
    feasibility_tol=1e-5,
):
    rows = [
        make_single_instance_summary_row(
            route=route,
            method=method,
            summary=summary,
            objective_key=objective_key,
            violation_key=violation_key,
            runtime_key=runtime_key,
            feasibility_tol=feasibility_tol,
        )
        for method, summary in dict(results or {}).items()
    ]
    return build_comparison_views(rows, summary_rows=summary_rows)


def summarize_comparison_rows_by_method(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("route")), str(row.get("method")))].append(row)

    summary_rows = []
    for (route, method), method_rows in grouped.items():
        objectives = [row["objective_mean"] for row in method_rows if row.get("objective_mean") is not None]
        violations = [row["violation_mean"] for row in method_rows if row.get("violation_mean") is not None]
        missing_runtime = [row for row in method_rows if row.get("runtime_total") is None]
        if missing_runtime:
            raise ValueError(f"comparison rows for {route}/{method} are missing runtime_total.")
        runtimes = [float(row["runtime_total"]) for row in method_rows]
        summary = {
            "route": route,
            "method": method,
            "objective_mean": (sum(objectives) / len(objectives)) if objectives else None,
            "feasibility_rate": sum(float(row.get("feasibility_rate", 0.0)) for row in method_rows) / len(method_rows),
            "violation_mean": (sum(violations) / len(violations)) if violations else None,
            "runtime_mean": (sum(runtimes) / len(runtimes)) if runtimes else 0.0,
            "runtime_total": sum(runtimes),
            "num_instances": len(method_rows),
        }
        for key in SUMMARY_TIMING_MEAN_FIELDS:
            values = [float(row[key]) for row in method_rows if row.get(key) is not None]
            if values:
                summary[f"{key}_mean"] = sum(values) / len(values)
        summary_rows.append(summary)
    return summary_rows
