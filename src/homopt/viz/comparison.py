"""Generic comparison visualizations for non-2D solver runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import relative_artifacts, save_figure
from .primitives import plot_metric_convergence
from .style import (
    ALGORITHM_COLORS,
    PAPER_STYLE,
    apply_paper_axis_style,
    require_matplotlib,
    set_axis_labels,
)
from .traces import as_numpy, problem_has_equalities


def _problem_tensor_kwargs(problem):
    for attr_name in ("Q", "p", "L", "A_obj"):
        tensor = getattr(problem, attr_name, None)
        if torch.is_tensor(tensor):
            return {"device": tensor.device, "dtype": tensor.dtype}
    return {"device": getattr(problem, "device", torch.device("cpu")), "dtype": torch.float32}


def _split_problem_metrics(problem, x_values):
    x_array = np.asarray(x_values)
    if x_array.size == 0:
        return None
    x = torch.as_tensor(x_array.reshape(-1, x_array.shape[-1]), **_problem_tensor_kwargs(problem))
    with torch.no_grad():
        objective = problem.objective_x(x).reshape(-1).detach().cpu().numpy()
        residual = problem.constraint_x(x, clip=False, eq_cons=True)
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
    eq_np = eq.detach().cpu().numpy()
    ineq_np = ineq.detach().cpu().numpy()
    return {
        "objective": objective,
        "equality_violation": eq_np,
        "inequality_violation": ineq_np,
        "full_violation": np.maximum(eq_np, ineq_np),
    }


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
    default_scope = "equality" if method in {"Hom-ALM", "Prox-Hom-ALM"} else "full"
    violation_scope = str(record.get("violation_scope", extras.get("violation_scope", default_scope))).lower()
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
    else:
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
            metrics = _split_problem_metrics(problem, trajectory)
        else:
            metrics = None
        if metrics is None:
            metrics = _record_scalar_metrics(record, method=method)
        n_values = max((len(values) for values in metrics.values()), default=0)
        if n_values == 0:
            continue
        iter_time_values = _record_values(record, "iter_time")
        if iter_time_values.size == n_values - 1:
            time = np.concatenate([[0.0], np.cumsum(iter_time_values)])
        elif iter_time_values.size == n_values:
            time = np.cumsum(iter_time_values)
        elif iter_time_values.size == 0 and n_values == 1:
            time = np.zeros(1, dtype=float)
        else:
            raise ValueError(
                f"Record for {method} has {n_values} metric values but {iter_time_values.size} iter_time values. "
                "Rerun the experiment to regenerate explicit timing records."
            )
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
    objective_paths = plot_metric_convergence(
        traces,
        "objective",
        "Objective value",
        artifact_dir / f"{prefix}_objective_convergence.pdf",
        reference_value=reference_objective,
        reference_label=reference_label,
        method_labels=method_labels,
        show_legend=show_convergence_legend,
    )
    paths.update({f"objective_convergence_{name}": metric_path for name, metric_path in objective_paths.items()})
    if include_equality_convergence is None:
        include_equality_convergence = problem_has_equalities(problem)
    if include_equality_convergence:
        equality_paths = plot_metric_convergence(
            traces,
            "equality_violation",
            "Equality violation",
            artifact_dir / f"{prefix}_equality_violation_convergence.pdf",
            log_y=True,
            method_labels=method_labels,
            y_min_clip=violation_y_min,
            show_legend=show_convergence_legend,
        )
        paths.update({f"equality_violation_convergence_{name}": metric_path for name, metric_path in equality_paths.items()})
    inequality_paths = plot_metric_convergence(
        traces,
        "inequality_violation",
        "Inequality violation",
        artifact_dir / f"{prefix}_inequality_violation_convergence.pdf",
        log_y=True,
        method_labels=method_labels,
        y_min_clip=violation_y_min,
        show_legend=show_convergence_legend,
    )
    paths.update({f"inequality_violation_convergence_{name}": metric_path for name, metric_path in inequality_paths.items()})
    full_paths = plot_metric_convergence(
        traces,
        "full_violation",
        "Full violation",
        artifact_dir / f"{prefix}_full_violation_convergence.pdf",
        log_y=True,
        method_labels=method_labels,
        y_min_clip=violation_y_min,
        show_legend=show_convergence_legend,
    )
    paths.update({f"full_violation_convergence_{name}": metric_path for name, metric_path in full_paths.items()})
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
