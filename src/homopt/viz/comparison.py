"""Generic comparison visualizations for non-2D solver runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import relative_artifacts, save_figure
from .convergence import save_metric_convergence_bundle
from .style import (
    ALGORITHM_COLORS,
    PAPER_STYLE,
    apply_paper_axis_style,
    require_matplotlib,
    set_axis_labels,
)
from .traces import as_numpy, problem_has_equalities, record_time_axis, split_problem_metrics


def _record_scalar_metrics(record, *, method=None):
    extras = dict(record.get("extras", {}) or {})
    objective = extras.get("objective_traj", record.get("obj_traj", record.get("objective")))
    full = record.get("cons_traj", extras.get("full_violation_traj", record.get("violation")))
    eq = record.get(
        "eq_violation_traj",
        extras.get("eq_violation_traj", extras.get("equality_violation_traj", extras.get("eq_violation"))),
    )
    ineq = record.get(
        "ineq_violation_traj",
        extras.get("ineq_violation_traj", extras.get("inequality_violation_traj", extras.get("ineq_violation"))),
    )
    if objective is None:
        raise ValueError(f"Record for {method or '<unknown>'} is missing objective trajectory.")
    if full is None and (eq is None or ineq is None):
        raise ValueError(
            f"Record for {method or '<unknown>'} is missing violation trajectories. "
            "Rerun the experiment to regenerate explicit violation records."
        )
    objective = np.asarray(objective, dtype=float).reshape(-1)
    full = None if full is None else np.asarray(full, dtype=float).reshape(-1)
    violation_scope = record.get("violation_scope", extras.get("violation_scope"))
    violation_scope = None if violation_scope is None else str(violation_scope).lower()
    if violation_scope in {"equality", "eq", "equality_only"}:
        if eq is None and full is None:
            raise ValueError(f"Record for {method or '<unknown>'} is missing equality violation trajectory.")
        eq = full if eq is None else np.asarray(eq, dtype=float).reshape(-1)
        ineq = np.zeros_like(eq) if ineq is None else np.asarray(ineq, dtype=float).reshape(-1)
    elif violation_scope in {"inequality", "ineq", "inequality_only"}:
        if ineq is None and full is None:
            raise ValueError(f"Record for {method or '<unknown>'} is missing inequality violation trajectory.")
        ineq = full if ineq is None else np.asarray(ineq, dtype=float).reshape(-1)
        eq = np.zeros_like(ineq) if eq is None else np.asarray(eq, dtype=float).reshape(-1)
    elif violation_scope in {None, "full"}:
        if eq is None or ineq is None:
            missing = []
            if eq is None:
                missing.append("eq_violation_traj")
            if ineq is None:
                missing.append("ineq_violation_traj")
            raise ValueError(
                f"Record for {method or '<unknown>'} has scalar full violation but no explicit "
                f"{' and '.join(missing)}. Rerun the experiment to regenerate split violation records."
            )
        eq = np.asarray(eq, dtype=float).reshape(-1)
        ineq = np.asarray(ineq, dtype=float).reshape(-1)
    else:
        raise ValueError(f"Record for {method or '<unknown>'} has unsupported violation_scope={violation_scope!r}.")
    if eq.size != ineq.size:
        raise ValueError(
            f"Record for {method or '<unknown>'} has mismatched equality/inequality violation lengths: "
            f"{eq.size} vs {ineq.size}."
        )
    if full is None or (full.size != max(eq.size, ineq.size) and eq.size == ineq.size and eq.size):
        full = np.maximum(eq, ineq)
    elif full.size != eq.size:
        raise ValueError(
            f"Record for {method or '<unknown>'} has mismatched full/split violation lengths: "
            f"{full.size} vs {eq.size}."
        )
    return {
        "objective": objective,
        "equality_violation": eq,
        "inequality_violation": ineq,
        "full_violation": full,
    }


def _record_values(record, key):
    extras = record.get("extras", {}) or {}
    value = record.get(key, extras.get(key))
    return np.asarray(value if value is not None else [], dtype=float).reshape(-1)


def build_comparison_traces(problem, records, algorithms):
    traces = {}
    for method in algorithms:
        if method not in records:
            continue
        record = records[method]
        trajectory = as_numpy(record.get("x_traj", []))
        if trajectory.size and trajectory.shape[-1] == int(getattr(problem, "nvar", trajectory.shape[-1])):
            metrics = split_problem_metrics(problem, trajectory)
        else:
            metrics = None
        if metrics is None:
            metrics = _record_scalar_metrics(record, method=method)
        n_values = max((len(values) for values in metrics.values()), default=0)
        if n_values == 0:
            continue
        time = record_time_axis(record, n_values)
        traces[method] = {"time": time, **metrics}
    return traces


def _json_safe_trace_payload(traces):
    return {
        method: {
            "time": np.asarray(trace["time"], dtype=float).tolist(),
            "objective": np.asarray(trace["objective"], dtype=float).tolist(),
            "equality_violation": np.asarray(trace["equality_violation"], dtype=float).tolist(),
            "inequality_violation": np.asarray(trace["inequality_violation"], dtype=float).tolist(),
            "full_violation": np.asarray(trace["full_violation"], dtype=float).tolist(),
        }
        for method, trace in traces.items()
    }


def _runtime_value(record):
    value = record.get("runtime_total", record.get("total_wall_time"))
    if value is not None:
        return float(value)
    return float(_record_values(record, "iter_time").sum())


def _outer_per_iter_time_value(record):
    values = _record_values(record, "outer_iter_time")
    if values.size:
        return float(values.mean())
    extras = record.get("extras", {}) or {}
    inner_iterations = extras.get("inner_iterations", record.get("inner_iterations"))
    outer_iterations = extras.get("outer_iterations", record.get("outer_iterations"))
    if inner_iterations not in (None, 0, 1) and outer_iterations not in (None, 0):
        raise ValueError(
            "outer_per_iter_time requires explicit outer_iter_time for nested solvers; "
            "rerun the experiment to regenerate timing records."
        )
    values = _record_values(record, "per_iter_time")
    if values.size:
        return float(values.mean())
    values = _record_values(record, "iter_time")
    if values.size:
        return float(values.mean())
    runtime = record.get("runtime_total", record.get("total_wall_time"))
    iterations = record.get("iterations")
    if iterations is None:
        iterations = extras.get("outer_iterations")
    if runtime is not None and iterations not in (None, 0):
        return float(runtime) / float(iterations)
    return _runtime_value(record)


def _runtime_metric_value(record, runtime_metric):
    if runtime_metric == "total_runtime":
        return _runtime_value(record)
    if runtime_metric == "outer_per_iter_time":
        return _outer_per_iter_time_value(record)
    raise ValueError("runtime_metric must be one of: 'total_runtime', 'outer_per_iter_time'.")


def _runtime_metric_error(record, runtime_metric):
    if runtime_metric == "total_runtime":
        return 0.0
    if runtime_metric != "outer_per_iter_time":
        raise ValueError("runtime_metric must be one of: 'total_runtime', 'outer_per_iter_time'.")
    for key in ("outer_iter_time", "per_iter_time", "iter_time"):
        values = _record_values(record, key)
        values = values[np.isfinite(values)]
        if values.size:
            return float(values.std(ddof=0))
    return 0.0


def plot_runtime_summary(records, algorithms, path, *, method_labels=None, runtime_metric="total_runtime"):
    plt = require_matplotlib()
    labels = [method for method in algorithms if method in records]
    if not labels:
        return None
    method_labels = dict(method_labels or {})
    totals = [_runtime_metric_value(records[method], runtime_metric) for method in labels]
    errors = [_runtime_metric_error(records[method], runtime_metric) for method in labels]
    fig, ax = plt.subplots(1, 1, figsize=PAPER_STYLE["runtime_figsize"])
    colors = [ALGORITHM_COLORS.get(label, "#4C78A8") for label in labels]
    ax.bar(
        np.arange(len(labels)),
        totals,
        yerr=errors,
        color=colors,
        alpha=0.9,
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#333333"},
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([method_labels.get(label, label) for label in labels], rotation=25, ha="right")
    ylabel = "Outer per-iteration time (s)" if runtime_metric == "outer_per_iter_time" else "Total running time (s)"
    set_axis_labels(ax, "Method", ylabel)
    apply_paper_axis_style(ax)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)
    return path


def save_comparison_visualizations(
    *,
    problem,
    records,
    algorithms,
    output_dir,
    prefix,
    reference_objective=None,
    reference_label="ConvexSolver",
    method_labels=None,
    runtime_metric="total_runtime",
    violation_y_min=None,
    show_convergence_legend=True,
    include_equality_convergence=None,
):
    output_dir = Path(output_dir)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    traces = build_comparison_traces(problem, records, algorithms)
    paths = {
        "runtime_summary": artifact_dir / f"{prefix}_runtime_summary.pdf",
        "metric_traces": artifact_dir / f"{prefix}_metric_traces.json",
    }
    if include_equality_convergence is None:
        include_equality_convergence = problem_has_equalities(problem)
    paths.update(
        save_metric_convergence_bundle(
            traces,
            artifact_dir,
            prefix,
            reference_objective=reference_objective,
            reference_label=reference_label,
            method_labels=method_labels,
            violation_y_min=violation_y_min,
            show_legend=show_convergence_legend,
            include_equality=include_equality_convergence,
        )
    )
    plot_runtime_summary(
        records,
        algorithms,
        paths["runtime_summary"],
        method_labels=method_labels,
        runtime_metric=runtime_metric,
    )
    paths["metric_traces"].write_text(json.dumps(_json_safe_trace_payload(traces), indent=2), encoding="utf-8")
    return relative_artifacts(output_dir, paths)


__all__ = [
    "build_comparison_traces",
    "plot_runtime_summary",
    "save_comparison_visualizations",
]
