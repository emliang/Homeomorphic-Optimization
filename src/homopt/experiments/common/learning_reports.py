"""Artifact and plot writers for parametric learning reports."""

from __future__ import annotations

import numpy as np

from homopt.records.artifacts import (
    artifact_ref,
    artifact_root,
    save_json,
    save_numpy,
    save_table_artifacts,
)
from homopt.experiments.common.reports import PARAMETRIC_LEARNING_RESULT_FIELDS


def learning_method_label(name):
    labels = {
        "predict_only": "Predict",
        "predict_ray_bisection": "Ray-Bisect",
        "predict_diff_projection": "D-Proj",
        "predict_exact_projection": "Proj",
        "predict_initialized_solver": "Init-Solver",
        "predict_initialized_opt": "Init-Opt",
        "predict_homeomorphic_projection": "H-Proj",
        "predict_ip_bisection": "B-Proj",
    }
    return labels.get(str(name), str(name))


def save_parametric_learning_artifacts(
    output_dir,
    *,
    metrics,
    rows,
    prediction_payload,
    artifact_prefix,
    plot_prefix=None,
):
    artifact_dir = artifact_root(output_dir)
    if artifact_dir is None:
        return {}
    plot_prefix = artifact_prefix if plot_prefix is None else plot_prefix
    summary_path = save_json(artifact_dir / f"{artifact_prefix}_summary.json", metrics)
    artifacts, _, _, _ = save_table_artifacts(
        output_dir,
        base_name=f"{artifact_prefix}_baselines",
        rows=rows,
        fieldnames=list(PARAMETRIC_LEARNING_RESULT_FIELDS),
        json_key="baselines_json",
        csv_key="baselines_csv",
        markdown_key="unused_markdown",
    )
    predictions_path = save_numpy(artifact_dir / f"{artifact_prefix}_predictions.npy", prediction_payload)
    plot_artifacts = save_parametric_learning_plots(output_dir, rows, plot_prefix=plot_prefix)
    return {
        **artifacts,
        **plot_artifacts,
        "summary": artifact_ref(output_dir, summary_path),
        "predictions": artifact_ref(output_dir, predictions_path),
    }


def _learning_bar_plot(output_dir, rows, *, key, ylabel, filename, positive_log=False):
    values = []
    labels = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if positive_log:
            value = max(value, 1e-12)
        labels.append(str(row["baseline"]))
        values.append(value)
    if not values:
        return None

    from homopt.viz.style import ALGORITHM_COLORS, PAPER_STYLE, apply_paper_axis_style, require_matplotlib, set_axis_labels
    from homopt.viz.artifacts import save_figure

    plt = require_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=PAPER_STYLE["runtime_figsize"])
    colors = [ALGORITHM_COLORS.get(label, "#4C78A8") for label in labels]
    ax.bar(np.arange(len(labels)), values, color=colors, alpha=0.9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([learning_method_label(label) for label in labels], rotation=25, ha="right")
    if positive_log:
        ax.set_yscale("log")
    set_axis_labels(ax, "Method", ylabel)
    apply_paper_axis_style(ax)
    fig.tight_layout()
    path = artifact_root(output_dir) / filename
    save_figure(fig, path)
    plt.close(fig)
    return path


def save_parametric_learning_plots(output_dir, rows, *, plot_prefix):
    if output_dir is None:
        return {}
    plot_specs = (
        ("learning_feasibility", "feasibility_rate", "Feasibility rate", f"{plot_prefix}_feasibility.pdf", False),
        ("learning_violation", "violation_mean", "Mean violation", f"{plot_prefix}_violation.pdf", True),
        ("learning_runtime", "runtime_mean", "Running time per instance (s)", f"{plot_prefix}_runtime.pdf", False),
        ("learning_objective_gap", "objective_gap_mean", "Relative optimality gap", f"{plot_prefix}_objective_gap.pdf", True),
        ("learning_solution_mse", "solution_mse", "Solution MSE", f"{plot_prefix}_solution_mse.pdf", True),
    )
    artifacts = {}
    for artifact_name, key, ylabel, filename, positive_log in plot_specs:
        path = _learning_bar_plot(output_dir, rows, key=key, ylabel=ylabel, filename=filename, positive_log=positive_log)
        if path is not None:
            artifacts[artifact_name] = artifact_ref(output_dir, path)
    return artifacts



__all__ = [
    "learning_method_label",
    "save_parametric_learning_artifacts",
    "save_parametric_learning_plots",
]
