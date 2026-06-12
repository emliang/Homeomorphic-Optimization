"""INN training visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from .artifacts import save_figure
from .primitives import draw_metric_convergence, draw_objective_residual_landscape, draw_trajectory
from .inn_records import (
    _add_objective_gap_metrics,
    _as_numpy_1d,
    _inn_record_for_instance,
    _method_trace,
    _outer_per_iter_time_stats,
    _trajectory_array,
)
from .qcqp_landscape import (
    QCQP_LATENT_TRAJECTORY_VIEW_LIM,
    _compute_latent_landscape,
    _compute_qcqp_landscape,
    _draw_latent_landscape_from_data,
)
from .style import (
    ALGORITHM_COLORS,
    ALGORITHM_LINE_STYLES,
    INN_PGD_METHOD_LABELS,
    PAPER_STYLE,
    apply_paper_axis_style,
    require_matplotlib,
    set_axis_labels,
)


SENSITIVITY_COLORS = [
    "#2C7BB6",
    "#D7191C",
    "#2CA25F",
    "#F28E2B",
    "#9467BD",
    "#8C564B",
]

QCQP_TRAJECTORY_GRID_SIZE = 520
QCQP_LATENT_TRAJECTORY_GRID_SIZE = 320
QCQP_TRAJECTORY_EVAL_BATCH_SIZE = None


def _artifact_ref(output_dir, path):
    return str(Path(path).relative_to(Path(output_dir)))


def _safe_method_name(method):
    return str(method).replace("-", "_").replace(" ", "_")


def _display_method_label(method):
    return INN_PGD_METHOD_LABELS.get(method, method)


def _instance_artifact_root(artifact_root_path, instance_idx):
    artifact_root_path = Path(artifact_root_path)
    if instance_idx is None:
        return artifact_root_path
    return artifact_root_path / "instances" / f"inst{int(instance_idx)}"


def _plot_outer_per_iter_runtime(method_records, path, *, colors=None, method_labels=None):
    methods = []
    values = []
    errors = []
    colors = dict(colors or {})
    method_labels = dict(method_labels or {})
    for method, record in method_records.items():
        value, error = _outer_per_iter_time_stats(record)
        if np.isfinite(value):
            methods.append(method)
            values.append(value)
            errors.append(error if np.isfinite(error) else 0.0)
    if not methods:
        return None
    plt = require_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=PAPER_STYLE["runtime_figsize"])
    x = np.arange(len(methods))
    bar_colors = [colors.get(method, ALGORITHM_COLORS.get(method, "#4C78A8")) for method in methods]
    ax.bar(
        x,
        values,
        yerr=errors,
        color=bar_colors,
        alpha=0.92,
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#333333"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(method, _display_method_label(method)) for method in methods], rotation=25, ha="right")
    set_axis_labels(ax, "Method", "Outer per-iteration time (s)")
    apply_paper_axis_style(ax)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)
    return path


def _sensitivity_styles(labels):
    labels = [str(label) for label in labels]
    colors = {label: SENSITIVITY_COLORS[i % len(SENSITIVITY_COLORS)] for i, label in enumerate(labels)}
    line_styles = {label: "-" for label in labels}
    method_labels = {label: label for label in labels}
    return colors, line_styles, method_labels

def save_qcqp_2d_comparison_visualizations(
    output_dir,
    artifact_root_path,
    *,
    problem,
    model=None,
    input_params,
    objective_params=None,
    inn_record,
    iterative_records,
    n_samples,
    instance_idx=0,
    reference_objective=None,
    show_convergence_legend=True,
    violation_y_min=1e-8,
    plot_convergence=True,
    plot_objective_gap=True,
    plot_gap_plus_violation=True,
    plot_runtime_summary=True,
):
    """Save 2D trajectories and convergence overlays for INN-PGD baselines."""

    plt = require_matplotlib()
    output_dir = Path(output_dir)
    artifact_root_path = Path(artifact_root_path)
    artifacts = {}
    instance_idx = int(instance_idx)
    n_samples = int(n_samples)
    if instance_idx < 0 or instance_idx >= n_samples:
        raise ValueError(f"instance_idx={instance_idx} is outside available range [0, {n_samples - 1}].")
    artifact_root_path = _instance_artifact_root(artifact_root_path, instance_idx)

    method_records = {
        "INN-PGD": _inn_record_for_instance(inn_record, int(n_samples), instance_idx),
    }
    for method, rows in dict(iterative_records or {}).items():
        if not rows:
            continue
        if instance_idx >= len(rows):
            raise ValueError(
                f"Record for {method} has {len(rows)} instances, cannot plot instance {instance_idx}."
            )
        row = rows[instance_idx]
        method_records[method] = row.get("record", {})

    trajectories = {
        method: _trajectory_array(record.get("x_traj", record.get("x_trajectory", [])))
        for method, record in method_records.items()
    }
    nonempty_trajectories = [traj for traj in trajectories.values() if traj.size]
    x_landscape = None
    if nonempty_trajectories:
        combined_traj = np.vstack(nonempty_trajectories)
        x_landscape = _compute_qcqp_landscape(
            problem,
            input_params,
            combined_traj,
            objective_params=objective_params,
            grid_size=QCQP_TRAJECTORY_GRID_SIZE,
            eval_batch_size=QCQP_TRAJECTORY_EVAL_BATCH_SIZE,
        )
    for method, traj in trajectories.items():
        if not traj.size:
            continue
        fig, ax = plt.subplots(1, 1, figsize=PAPER_STYLE["trajectory_figsize"])
        if x_landscape is not None:
            draw_objective_residual_landscape(ax, **x_landscape, objective_levels=12)
        draw_trajectory(
            ax,
            traj,
            marker=".",
            label=_display_method_label(method),
            zorder=6,
        )
        set_axis_labels(ax, r"$x_1$", r"$x_2$", fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        apply_paper_axis_style(ax)
        fig.tight_layout()
        path = artifact_root_path / f"qcqp_2d_trajectory_x_{_safe_method_name(method)}.pdf"
        save_figure(fig, path)
        plt.close(fig)
        artifacts[f"qcqp_2d_trajectory_x_{_safe_method_name(method).lower()}"] = _artifact_ref(output_dir, path)

    z_traj = _trajectory_array(method_records["INN-PGD"].get("z_traj", []))
    if z_traj.size:
        fig, ax = plt.subplots(1, 1, figsize=PAPER_STYLE["trajectory_figsize"])
        latent_landscape = _compute_latent_landscape(
            problem,
            model,
            input_params,
            z_traj,
            objective_params=objective_params,
            grid_size=QCQP_LATENT_TRAJECTORY_GRID_SIZE,
            eval_batch_size=QCQP_TRAJECTORY_EVAL_BATCH_SIZE,
        )
        _draw_latent_landscape_from_data(ax, latent_landscape)
        draw_trajectory(
            ax,
            z_traj,
            marker=".",
            label=_display_method_label("INN-PGD"),
            zorder=6,
        )
        lim = float(latent_landscape.get("lim", QCQP_LATENT_TRAJECTORY_VIEW_LIM))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        fig.tight_layout()
        path = artifact_root_path / "qcqp_2d_trajectory_z_INN_PGD.pdf"
        save_figure(fig, path)
        plt.close(fig)
        artifacts["qcqp_2d_trajectory_z_inn_pgd"] = _artifact_ref(output_dir, path)

    traces = {}
    for method, record in method_records.items():
        trace = _method_trace(record)
        if trace is not None:
            traces[method] = trace
    if traces and bool(plot_convergence):
        obj_paths = draw_metric_convergence(
            traces,
            "objective",
            "Objective value",
            artifact_root_path / "qcqp_2d_objective.pdf",
            reference_value=reference_objective,
            reference_label="IPOPT" if reference_objective is not None else None,
            colors=ALGORITHM_COLORS,
            line_styles=ALGORITHM_LINE_STYLES,
            method_labels=INN_PGD_METHOD_LABELS,
            show_legend=show_convergence_legend,
        )
        vio_paths = draw_metric_convergence(
            traces,
            "violation",
            "Inequality violation",
            artifact_root_path / "qcqp_2d_violation.pdf",
            log_y=True,
            y_min_clip=violation_y_min,
            colors=ALGORITHM_COLORS,
            line_styles=ALGORITHM_LINE_STYLES,
            method_labels=INN_PGD_METHOD_LABELS,
            show_legend=show_convergence_legend,
        )
        artifacts.update({f"qcqp_2d_objective_{key}": _artifact_ref(output_dir, path) for key, path in obj_paths.items()})
        artifacts.update({f"qcqp_2d_violation_{key}": _artifact_ref(output_dir, path) for key, path in vio_paths.items()})
        gap_traces, _ = _add_objective_gap_metrics(traces, reference_objective=reference_objective)
        if gap_traces and bool(plot_objective_gap):
            gap_paths = draw_metric_convergence(
                gap_traces,
                "objective_gap",
                "Relative optimality gap",
                artifact_root_path / "qcqp_2d_objective_gap.pdf",
                log_y=True,
                y_min_clip=violation_y_min,
                reference_label=None,
                colors=ALGORITHM_COLORS,
                line_styles=ALGORITHM_LINE_STYLES,
                method_labels=INN_PGD_METHOD_LABELS,
                show_legend=show_convergence_legend,
            )
            artifacts.update({f"qcqp_2d_objective_gap_{key}": _artifact_ref(output_dir, path) for key, path in gap_paths.items()})
        if gap_traces and bool(plot_gap_plus_violation):
            solution_paths = draw_metric_convergence(
                gap_traces,
                "objective_gap_plus_violation",
                "Opt. gap + violation",
                artifact_root_path / "qcqp_2d_gap_plus_violation.pdf",
                log_y=True,
                y_min_clip=violation_y_min,
                colors=ALGORITHM_COLORS,
                line_styles=ALGORITHM_LINE_STYLES,
                method_labels=INN_PGD_METHOD_LABELS,
                show_legend=show_convergence_legend,
            )
            artifacts.update(
                {f"qcqp_2d_gap_plus_violation_{key}": _artifact_ref(output_dir, path) for key, path in solution_paths.items()}
            )
        if bool(plot_runtime_summary):
            runtime_path = _plot_outer_per_iter_runtime(
                method_records,
                artifact_root_path / "qcqp_2d_outer_per_iter_time.pdf",
            )
            if runtime_path is not None:
                artifacts["qcqp_2d_outer_per_iter_time"] = _artifact_ref(output_dir, runtime_path)

    return artifacts


def save_qcqp_sensitivity_convergence_visualizations(
    output_dir,
    artifact_root_path,
    *,
    sweep_name,
    cases,
    n_samples,
    visualize_instance_idx=0,
    show_convergence_legend=True,
    violation_y_min=1e-8,
    plot_objective_gap=True,
    plot_gap_plus_violation=True,
    plot_runtime_summary=True,
):
    """Overlay test convergence curves for all cases in one sensitivity sweep."""

    output_dir = Path(output_dir)
    artifact_root_path = Path(artifact_root_path) / "sensitivity" / str(sweep_name)
    artifacts = {}
    labels = [str(case["label"]) for case in cases if case.get("record") is not None]
    if not labels:
        return artifacts
    colors, line_styles, method_labels = _sensitivity_styles(labels)
    n_samples = max(1, int(n_samples))
    if visualize_instance_idx is None:
        raw_indices = [0]
    elif isinstance(visualize_instance_idx, str) and visualize_instance_idx.strip().lower() == "all":
        raw_indices = range(n_samples)
    elif isinstance(visualize_instance_idx, (list, tuple, set)):
        raw_indices = list(visualize_instance_idx)
    else:
        raw_indices = [visualize_instance_idx]
    indices = []
    for raw in raw_indices:
        idx = int(raw)
        if idx < 0 or idx >= n_samples:
            raise ValueError(f"visualize_instance_idx={idx} is outside available range [0, {n_samples - 1}].")
        if idx not in indices:
            indices.append(idx)

    for instance_idx in indices or [0]:
        instance_root = artifact_root_path / "instances" / f"inst{instance_idx}"
        traces = {}
        method_records = {}
        for case in cases:
            record = case.get("record")
            if record is None:
                continue
            label = str(case["label"])
            instance_record = _inn_record_for_instance(record, n_samples, instance_idx)
            trace = _method_trace(instance_record)
            if trace is None:
                continue
            traces[label] = trace
            method_records[label] = instance_record
        if not traces:
            continue

        obj_paths = draw_metric_convergence(
            traces,
            "objective",
            "Objective value",
            instance_root / f"qcqp_sensitivity_{sweep_name}_objective.pdf",
            colors=colors,
            line_styles=line_styles,
            method_labels=method_labels,
            show_legend=show_convergence_legend,
        )
        vio_paths = draw_metric_convergence(
            traces,
            "violation",
            "Inequality violation",
            instance_root / f"qcqp_sensitivity_{sweep_name}_violation.pdf",
            log_y=True,
            y_min_clip=violation_y_min,
            colors=colors,
            line_styles=line_styles,
            method_labels=method_labels,
            show_legend=show_convergence_legend,
        )
        suffix = f"_inst{instance_idx}" if len(indices) > 1 else ""
        artifacts.update(
            {f"qcqp_sensitivity_{sweep_name}_objective_{key}{suffix}": _artifact_ref(output_dir, path) for key, path in obj_paths.items()}
        )
        artifacts.update(
            {f"qcqp_sensitivity_{sweep_name}_violation_{key}{suffix}": _artifact_ref(output_dir, path) for key, path in vio_paths.items()}
        )

        gap_traces, _ = _add_objective_gap_metrics(traces, reference_objective=None)
        if gap_traces and bool(plot_objective_gap):
            gap_paths = draw_metric_convergence(
                gap_traces,
                "objective_gap",
                "Relative optimality gap",
                instance_root / f"qcqp_sensitivity_{sweep_name}_objective_gap.pdf",
                log_y=True,
                y_min_clip=violation_y_min,
                colors=colors,
                line_styles=line_styles,
                method_labels=method_labels,
                show_legend=show_convergence_legend,
            )
            artifacts.update(
                {
                    f"qcqp_sensitivity_{sweep_name}_objective_gap_{key}{suffix}": _artifact_ref(output_dir, path)
                    for key, path in gap_paths.items()
                }
            )
        if gap_traces and bool(plot_gap_plus_violation):
            solution_paths = draw_metric_convergence(
                gap_traces,
                "objective_gap_plus_violation",
                "Opt. gap + violation",
                instance_root / f"qcqp_sensitivity_{sweep_name}_gap_plus_violation.pdf",
                log_y=True,
                y_min_clip=violation_y_min,
                colors=colors,
                line_styles=line_styles,
                method_labels=method_labels,
                show_legend=show_convergence_legend,
            )
            artifacts.update(
                {
                    f"qcqp_sensitivity_{sweep_name}_gap_plus_violation_{key}{suffix}": _artifact_ref(output_dir, path)
                    for key, path in solution_paths.items()
                }
            )
        if bool(plot_runtime_summary):
            runtime_path = _plot_outer_per_iter_runtime(
                method_records,
                instance_root / f"qcqp_sensitivity_{sweep_name}_outer_per_iter_time.pdf",
                colors=colors,
                method_labels=method_labels,
            )
            if runtime_path is not None:
                artifacts[f"qcqp_sensitivity_{sweep_name}_outer_per_iter_time{suffix}"] = _artifact_ref(output_dir, runtime_path)

    return artifacts


def save_inn_training_visualizations(
    output_dir,
    artifact_root_path,
    training_record,
    obj_traj,
    cons_traj,
    decision_traj,
    problem=None,
    input_params=None,
    objective_params=None,
    instance_idx=None,
    plot_optimizer_diagnostics=True,
):
    """Save INN training and INN-PGD diagnostic figures."""

    plt = require_matplotlib()
    output_dir = Path(output_dir)
    artifact_root_path = Path(artifact_root_path)
    artifacts = {}
    instance_artifact_root_path = _instance_artifact_root(artifact_root_path, instance_idx)

    penalty = np.asarray(training_record.get("penalty_list", []), dtype=float).reshape(-1)
    volume = np.asarray(training_record.get("volume_list", []), dtype=float).reshape(-1)
    distortion = np.asarray(training_record.get("dist_list", []), dtype=float).reshape(-1)
    lipschitz = np.asarray(training_record.get("lipschitz_list", []), dtype=float).reshape(-1)
    if penalty.size > 0 or volume.size > 0 or distortion.size > 0 or lipschitz.size > 0:
        series = [
            (volume, "Log-Volume Term", "Log-Volume", "#4C78A8"),
            (penalty, "Constraint Penalty Term", "Constraint Penalty", "#F58518"),
            (distortion, "Log-Lipschitz Term", "Log-Lipschitz", "#E83947"),
        ]
        if lipschitz.size > 0:
            series.append((lipschitz, "Max Log-Lipschitz Term", "Max Log-Lipschitz", "#8E44AD"))
        figsize = PAPER_STYLE["three_panel_figsize"] if len(series) == 3 else (18.0, 3.8)
        fig, axes = plt.subplots(1, len(series), figsize=figsize)
        axes = np.atleast_1d(axes)
        for ax, (values, title, ylabel, color) in zip(axes, series):
            if values.size > 0:
                ax.plot(
                    np.arange(1, values.size + 1),
                    values,
                    color=color,
                    linewidth=PAPER_STYLE["curve_linewidth"],
                )
            ax.set_title(title, fontsize=PAPER_STYLE["title_fontsize"])
            set_axis_labels(ax, "Iteration", ylabel)
            apply_paper_axis_style(ax)
        fig.tight_layout()
        path = artifact_root_path / "inn_training_metrics.pdf"
        save_figure(fig, path)
        plt.close(fig)
        artifacts["training_metrics"] = _artifact_ref(output_dir, path)

    if not bool(plot_optimizer_diagnostics):
        return artifacts

    obj_vals = _as_numpy_1d(obj_traj)
    cons_vals = _as_numpy_1d(cons_traj)
    if obj_vals.size or cons_vals.size:
        fig, axes = plt.subplots(1, 2, figsize=PAPER_STYLE["two_panel_figsize"])
        axes[0].plot(
            np.arange(1, obj_vals.size + 1),
            obj_vals,
            linewidth=PAPER_STYLE["curve_linewidth"],
            color=ALGORITHM_COLORS["INN-PGD"],
        )
        axes[0].set_title(r"Hom-PGD$^+$ objective", fontsize=PAPER_STYLE["title_fontsize"])
        set_axis_labels(axes[0], "Iteration", "Objective")
        axes[1].plot(
            np.arange(1, cons_vals.size + 1),
            np.maximum(cons_vals, 1e-12),
            linewidth=PAPER_STYLE["curve_linewidth"],
            color=ALGORITHM_COLORS["INN-PGD"],
        )
        axes[1].set_yscale("log")
        axes[1].set_title(r"Hom-PGD$^+$ violation", fontsize=PAPER_STYLE["title_fontsize"])
        set_axis_labels(axes[1], "Iteration", "Max violation")
        for ax in axes:
            apply_paper_axis_style(ax)
        fig.tight_layout()
        path = instance_artifact_root_path / "inn_pgd_convergence.pdf"
        save_figure(fig, path)
        plt.close(fig)
        artifacts["inn_pgd_convergence"] = _artifact_ref(output_dir, path)

    if hasattr(decision_traj, "detach"):
        traj = np.asarray(decision_traj.detach().cpu(), dtype=float)
        if traj.ndim == 2 and traj.shape[1] >= 2 and traj.shape[0] > 1:
            fig, ax = plt.subplots(1, 1, figsize=PAPER_STYLE["trajectory_figsize"])
            if problem is not None and int(getattr(problem, "nvar", traj.shape[1])) == 2:
                landscape = _compute_qcqp_landscape(
                    problem,
                    input_params,
                    traj,
                    objective_params=objective_params,
                    grid_size=QCQP_TRAJECTORY_GRID_SIZE,
                    eval_batch_size=QCQP_TRAJECTORY_EVAL_BATCH_SIZE,
                )
                draw_objective_residual_landscape(ax, **landscape, objective_levels=12)
            draw_trajectory(
                ax,
                traj[:, :2],
                marker=".",
                label=_display_method_label("INN-PGD"),
                start_label="Start",
                final_label="Final",
            )
            set_axis_labels(ax, r"$x_1$", r"$x_2$", fontweight="bold")
            ax.set_aspect("equal", adjustable="box")
            apply_paper_axis_style(ax)
            fig.tight_layout()
            path = instance_artifact_root_path / "decision_trajectory_2d.pdf"
            save_figure(fig, path)
            plt.close(fig)
            artifacts["decision_trajectory_2d"] = _artifact_ref(output_dir, path)

    return artifacts


__all__ = [
    "save_inn_training_visualizations",
    "save_qcqp_2d_comparison_visualizations",
    "save_qcqp_sensitivity_convergence_visualizations",
]
