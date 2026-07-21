"""Shared 2D convex-problem visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import relative_artifacts, save_figure
from .convergence import save_metric_convergence_bundle
from .landscapes import convex_landscape_grid, convex_z_landscape_grid, plot_bounds
from .primitives import (
    draw_constraint_notation,
    draw_convex_landscape,
    draw_convex_z_landscape,
    draw_trajectory,
)
from .style import (
    ALGORITHM_COLORS,
    ALGORITHM_LINE_STYLES,
    PAPER_STYLE,
    apply_paper_axis_style,
    configure_matplotlib_cache,
    require_matplotlib,
    style_legend as _style_legend,
)
from .traces import (
    as_numpy,
    build_convex_2d_traces,
    reference_objective_from_payload,
    problem_has_equalities,
    split_problem_metrics,
    trace_payload,
)


def plot_individual_trajectories(problem, traces, artifact_dir, *, show_constraint_notation=False):
    plt = require_matplotlib()
    low, high = plot_bounds(problem, traces)
    X, Y, objective, inequality, equality, eq_resid = convex_landscape_grid(problem, low, high)
    paths = {}
    for method, trace in traces.items():
        traj = trace["trajectory"]
        if traj.size == 0:
            continue
        fig, ax = plt.subplots(1, 1, figsize=PAPER_STYLE["trajectory_figsize"])
        draw_convex_landscape(ax, X, Y, objective, inequality, equality, eq_resid)
        if show_constraint_notation:
            draw_constraint_notation(ax, space="x")
        draw_trajectory(
            ax,
            traj,
            color=PAPER_STYLE["trajectory_color"],
            label=None,
        )
        fig.tight_layout()
        path = Path(artifact_dir) / f"trajectory_{method.replace('-', '_')}.pdf"
        save_figure(fig, path)
        plt.close(fig)
        paths[f"trajectory_{method.replace('-', '_').lower()}"] = path
    return paths


def plot_individual_z_trajectories(problem, hom_map, traces, artifact_dir, *, show_constraint_notation=False):
    plt = require_matplotlib()
    Z1, Z2, objective, eq_resid, inside, p_norm = convex_z_landscape_grid(problem, hom_map)
    paths = {}
    for method, trace in traces.items():
        traj = trace.get("z_trajectory", np.empty((0, 2)))
        if traj.size == 0:
            continue
        fig, ax = plt.subplots(1, 1, figsize=PAPER_STYLE["trajectory_figsize"])
        draw_convex_z_landscape(ax, Z1, Z2, objective, eq_resid, inside, p_norm)
        if show_constraint_notation:
            draw_constraint_notation(ax, space="z")
        draw_trajectory(
            ax,
            traj,
            color=PAPER_STYLE["trajectory_color"],
            label=None,
            zorder=5,
        )
        fig.tight_layout()
        path = Path(artifact_dir) / f"trajectory_z_{method.replace('-', '_')}.pdf"
        save_figure(fig, path)
        plt.close(fig)
        paths[f"trajectory_z_{method.replace('-', '_').lower()}"] = path
    return paths


def plot_trajectory_legend(traces, path, *, line_styles=None, method_labels=None):
    """Save trajectory legend as a separate compact artifact."""

    plt = require_matplotlib()
    from matplotlib.lines import Line2D

    line_styles = line_styles or ALGORITHM_LINE_STYLES
    method_labels = dict(method_labels or {})
    methods = [method for method, trace in traces.items() if trace["trajectory"].size]
    if not methods:
        return None
    handles = [
        Line2D(
            [0],
            [0],
            color=PAPER_STYLE["trajectory_color"],
            linestyle=line_styles.get(method, "-"),
            linewidth=PAPER_STYLE["trajectory_linewidth"],
            label=method_labels.get(method, method),
        )
        for method in methods
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=PAPER_STYLE["start_marker_color"],
                markeredgecolor="black",
                markeredgewidth=PAPER_STYLE["marker_edge_width"],
                markersize=PAPER_STYLE["start_marker_size"],
                label="start",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markerfacecolor=PAPER_STYLE["final_marker_color"],
                markeredgecolor="black",
                markeredgewidth=PAPER_STYLE["marker_edge_width"],
                markersize=PAPER_STYLE["final_marker_size"],
                label="final",
            ),
        ]
    )
    fig = plt.figure(figsize=(max(3.0, 1.35 * len(handles)), 0.55))
    legend = fig.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=True,
        fontsize=PAPER_STYLE["compact_legend_fontsize"],
        handlelength=1.8,
        columnspacing=1.0,
    )
    _style_legend(legend)
    save_figure(fig, path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def save_convex_2d_visualizations(
    *,
    problem,
    records,
    algorithms,
    output_dir,
    hom_map=None,
    payload=None,
    prefix="convex_2d",
    include_individual=True,
    show_constraint_notation=False,
    method_labels=None,
    reference_label=None,
    include_equality_convergence=None,
    violation_y_min=None,
):
    output_dir = Path(output_dir)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    traces = build_convex_2d_traces(problem, records, algorithms)
    if method_labels is None:
        method_labels = (payload or {}).get("method_labels")
    method_labels = dict(method_labels or {})
    reference_label = reference_label or (payload or {}).get("reference_label", "MOSEK")
    paths = {
        "trajectory_legend": artifact_dir / f"{prefix}_trajectory_legend.pdf",
        "metric_traces": artifact_dir / f"{prefix}_metric_traces.json",
    }
    ref_obj = reference_objective_from_payload(payload or {})
    if include_equality_convergence is None:
        include_equality_convergence = problem_has_equalities(problem)
    paths.update(
        save_metric_convergence_bundle(
            traces,
            artifact_dir,
            prefix,
            reference_objective=ref_obj,
            reference_label=reference_label,
            method_labels=method_labels,
            violation_y_min=violation_y_min,
            include_equality=include_equality_convergence,
        )
    )
    legend_path = plot_trajectory_legend(traces, paths["trajectory_legend"], method_labels=method_labels)
    if legend_path is None:
        del paths["trajectory_legend"]
    if include_individual:
        paths.update(
            plot_individual_trajectories(
                problem,
                traces,
                artifact_dir,
                show_constraint_notation=show_constraint_notation,
            )
        )
        if hom_map is not None:
            paths.update(
                plot_individual_z_trajectories(
                    problem,
                    hom_map,
                    traces,
                    artifact_dir,
                    show_constraint_notation=show_constraint_notation,
                )
            )
    paths["metric_traces"].write_text(json.dumps(trace_payload(traces), indent=2), encoding="utf-8")
    return relative_artifacts(output_dir, paths)


__all__ = [
    "ALGORITHM_COLORS",
    "ALGORITHM_LINE_STYLES",
    "PAPER_STYLE",
    "apply_paper_axis_style",
    "as_numpy",
    "build_convex_2d_traces",
    "configure_matplotlib_cache",
    "problem_has_equalities",
    "save_convex_2d_visualizations",
    "split_problem_metrics",
]
