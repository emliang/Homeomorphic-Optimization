"""Shared convergence-curve rendering helpers."""

from __future__ import annotations

import numpy as np

from .artifacts import figure_output_path, save_figure
from .style import (
    ALGORITHM_COLORS,
    ALGORITHM_LINE_STYLES,
    PAPER_STYLE,
    apply_paper_axis_style,
    require_matplotlib,
    set_axis_labels,
    style_legend,
)


def draw_metric_convergence(
    traces,
    metric_name,
    ylabel,
    path,
    *,
    reference_value=None,
    log_y=False,
    colors=None,
    line_styles=None,
    reference_label="ConvexSolver",
    method_labels=None,
    y_min_clip=None,
    show_legend=True,
):
    """Draw separate iteration/time convergence figures for one scalar metric."""

    plt = require_matplotlib()
    colors = colors or ALGORITHM_COLORS
    line_styles = line_styles or ALGORITHM_LINE_STYLES
    method_labels = dict(method_labels or {})
    if y_min_clip is not None:
        y_min_clip = float(y_min_clip)
        if y_min_clip <= 0:
            raise ValueError("y_min_clip must be positive when provided.")
    path = figure_output_path(path)
    output_paths = {
        "iteration": path.with_name(f"{path.stem}_by_iter{path.suffix}"),
        "iteration_logx": path.with_name(f"{path.stem}_by_iter_logx{path.suffix}"),
        "time": path.with_name(f"{path.stem}_by_time{path.suffix}"),
        "time_logx": path.with_name(f"{path.stem}_by_time_logx{path.suffix}"),
    }

    def _log_time_floor():
        positive = [
            value
            for trace in traces.values()
            for value in np.asarray(trace.get("time", []), dtype=float).reshape(-1)
            if np.isfinite(value) and value > 0
        ]
        if not positive:
            return 1e-12
        return max(min(positive) * 0.1, 1e-12)

    def _draw_panel(x_key, xlabel, output_path, *, log_x=False):
        fig, ax = plt.subplots(1, 1, figsize=PAPER_STYLE["convergence_figsize"])
        log_time_floor = _log_time_floor() if log_x and x_key == "time" else None
        for method, trace in traces.items():
            data = np.asarray(trace[metric_name], dtype=float)
            if data.size == 0:
                continue
            if y_min_clip is not None:
                y_data = np.maximum(data, y_min_clip)
            elif log_y:
                y_data = data + 1e-12
            else:
                y_data = data
            style = line_styles.get(method, "-")
            color = colors.get(method)
            if x_key == "iteration":
                x_data = np.arange(1, data.size + 1) if log_x else np.arange(data.size)
            else:
                x_data = np.asarray(trace["time"], dtype=float)
                if log_x:
                    x_data = np.where(x_data > 0, x_data, log_time_floor)
                    keep = np.isfinite(x_data) & (x_data > 0)
                    x_data = x_data[keep]
                    y_data = y_data[keep]
                    if x_data.size == 0:
                        continue
            ax.plot(
                x_data,
                y_data,
                linestyle=style,
                color=color,
                linewidth=PAPER_STYLE["curve_linewidth"],
                alpha=0.92,
                label=method_labels.get(method, method),
            )
        if reference_value is not None:
            ax.axhline(
                reference_value,
                color="black",
                linestyle=":",
                linewidth=PAPER_STYLE["reference_linewidth"],
                alpha=0.72,
            )
            if reference_label:
                ax.annotate(
                    reference_label,
                    xy=(0.985, reference_value),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, -5),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=PAPER_STYLE["compact_legend_fontsize"],
                    color="black",
                    alpha=0.82,
                )
        if log_y:
            ax.set_yscale("log")
            if y_min_clip is not None:
                ax.set_ylim(bottom=y_min_clip)
        if log_x:
            ax.set_xscale("log")
        apply_paper_axis_style(ax, grid=True)
        if show_legend:
            legend = ax.legend(fontsize=PAPER_STYLE["compact_legend_fontsize"])
            style_legend(legend)
        set_axis_labels(ax, xlabel, ylabel)
        fig.tight_layout()
        save_figure(fig, output_path)
        plt.close(fig)

    _draw_panel("iteration", "Iteration", output_paths["iteration"])
    _draw_panel("time", "Running time (s)", output_paths["time"])
    _draw_panel("iteration", "Iteration", output_paths["iteration_logx"], log_x=True)
    _draw_panel("time", "Running time (s)", output_paths["time_logx"], log_x=True)
    return output_paths


plot_metric_convergence = draw_metric_convergence


def save_metric_convergence_bundle(
    traces,
    artifact_dir,
    prefix,
    *,
    reference_objective=None,
    reference_label="ConvexSolver",
    method_labels=None,
    violation_y_min=None,
    show_legend=True,
    include_equality=False,
):
    """Save the standard objective/equality/inequality/full convergence bundle."""

    metric_specs = [
        ("objective", "Objective value", False, reference_objective),
        ("inequality_violation", "Inequality violation", True, None),
        ("full_violation", "Full violation", True, None),
    ]
    if include_equality:
        metric_specs.insert(1, ("equality_violation", "Equality violation", True, None))

    paths = {}
    for metric_name, ylabel, log_y, reference_value in metric_specs:
        metric_paths = draw_metric_convergence(
            traces,
            metric_name,
            ylabel,
            artifact_dir / f"{prefix}_{metric_name}_convergence.pdf",
            reference_value=reference_value,
            reference_label=reference_label if reference_value is not None else "ConvexSolver",
            log_y=log_y,
            method_labels=method_labels,
            y_min_clip=violation_y_min if log_y else None,
            show_legend=show_legend,
        )
        paths.update({f"{metric_name}_convergence_{name}": metric_path for name, metric_path in metric_paths.items()})
    return paths


__all__ = [
    "draw_metric_convergence",
    "plot_metric_convergence",
    "save_metric_convergence_bundle",
]
