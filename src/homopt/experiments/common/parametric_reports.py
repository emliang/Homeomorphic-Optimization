"""Shared reporting helpers for parametric learning benchmarks."""

from __future__ import annotations

import numpy as np

from homopt.experiments.common.artifacts import (
    artifact_ref,
    artifact_root,
    save_json,
    save_numpy,
    save_table_artifacts,
)


PARAMETRIC_LEARNING_RESULT_FIELDS = (
    "baseline",
    "predictor",
    "refiner",
    "objective_mean",
    "objective_gap_mean",
    "objective_improvement_mean",
    "feasibility_rate",
    "violation_mean",
    "violation_reduction_mean",
    "solution_mse",
    "solution_mae",
    "runtime_mean",
    "runtime_total",
    "predictor_train_time_sec",
    "postprocess_train_time_sec",
    "refine_time_sec",
    "raw_objective_mean",
    "raw_objective_gap_mean",
    "raw_feasibility_rate",
    "raw_violation_mean",
    "raw_solution_mse",
    "raw_solution_mae",
)


def make_parametric_learning_result_row(
    *,
    baseline,
    predictor,
    refiner,
    objective_mean,
    feasibility_rate,
    violation_mean,
    runtime_total,
    n_samples,
    raw_objective_mean,
    raw_feasibility_rate,
    raw_violation_mean,
    objective_gap_mean=None,
    objective_improvement_mean=None,
    violation_reduction_mean=None,
    solution_mse=None,
    solution_mae=None,
    raw_objective_gap_mean=None,
    raw_solution_mse=None,
    raw_solution_mae=None,
    predictor_train_time_sec=0.0,
    postprocess_train_time_sec=0.0,
    refine_time_sec=None,
):
    n_samples = max(int(n_samples), 1)
    runtime_total = float(runtime_total)
    return {
        "baseline": str(baseline),
        "predictor": str(predictor),
        "refiner": str(refiner),
        "objective_mean": None if objective_mean is None else float(objective_mean),
        "objective_gap_mean": None if objective_gap_mean is None else float(objective_gap_mean),
        "objective_improvement_mean": None if objective_improvement_mean is None else float(objective_improvement_mean),
        "feasibility_rate": float(feasibility_rate),
        "violation_mean": None if violation_mean is None else float(violation_mean),
        "violation_reduction_mean": None if violation_reduction_mean is None else float(violation_reduction_mean),
        "solution_mse": None if solution_mse is None else float(solution_mse),
        "solution_mae": None if solution_mae is None else float(solution_mae),
        "runtime_mean": runtime_total / n_samples,
        "runtime_total": runtime_total,
        "predictor_train_time_sec": float(predictor_train_time_sec),
        "postprocess_train_time_sec": float(postprocess_train_time_sec),
        "refine_time_sec": runtime_total if refine_time_sec is None else float(refine_time_sec),
        "raw_objective_mean": float(raw_objective_mean),
        "raw_objective_gap_mean": None if raw_objective_gap_mean is None else float(raw_objective_gap_mean),
        "raw_feasibility_rate": float(raw_feasibility_rate),
        "raw_violation_mean": float(raw_violation_mean),
        "raw_solution_mse": None if raw_solution_mse is None else float(raw_solution_mse),
        "raw_solution_mae": None if raw_solution_mae is None else float(raw_solution_mae),
    }


def build_parametric_learning_baseline_output(
    *,
    baseline,
    summary,
    runtime_sec,
    n_samples,
    raw_predictions,
    refined_predictions,
    predictor_train_time=0.0,
    postprocess_train_time=0.0,
):
    row = make_parametric_learning_result_row(
        baseline=baseline,
        predictor=summary["predictor"],
        refiner=summary["refiner"],
        objective_mean=summary["refined_objective_mean"],
        feasibility_rate=summary["refined_feasibility_rate"],
        violation_mean=summary["refined_violation_mean"],
        runtime_total=runtime_sec,
        n_samples=n_samples,
        raw_objective_mean=summary["raw_objective_mean"],
        raw_feasibility_rate=summary["raw_feasibility_rate"],
        raw_violation_mean=summary["raw_violation_mean"],
        objective_gap_mean=summary.get("refined_objective_gap_mean"),
        objective_improvement_mean=summary.get("objective_improvement_mean"),
        violation_reduction_mean=summary.get("violation_reduction_mean"),
        solution_mse=summary.get("refined_solution_mse"),
        solution_mae=summary.get("refined_solution_mae"),
        raw_objective_gap_mean=summary.get("raw_objective_gap_mean"),
        raw_solution_mse=summary.get("raw_solution_mse"),
        raw_solution_mae=summary.get("raw_solution_mae"),
        predictor_train_time_sec=predictor_train_time,
        postprocess_train_time_sec=postprocess_train_time,
        refine_time_sec=runtime_sec,
    )
    result = {
        **summary,
        "objective_mean": row["objective_mean"],
        "feasibility_rate": row["feasibility_rate"],
        "violation_mean": row["violation_mean"],
        "runtime_mean": row["runtime_mean"],
        "runtime_total": row["runtime_total"],
        "runtime_sec": runtime_sec,
        "total_wall_time": runtime_sec,
        "runtime_per_instance_sec": row["runtime_mean"],
        "objective_gap_mean": row["objective_gap_mean"],
        "objective_improvement_mean": row["objective_improvement_mean"],
        "violation_reduction_mean": row["violation_reduction_mean"],
        "solution_mse": row["solution_mse"],
        "solution_mae": row["solution_mae"],
        "predictor_train_time_sec": row["predictor_train_time_sec"],
        "postprocess_train_time_sec": row["postprocess_train_time_sec"],
        "refine_time_sec": row["refine_time_sec"],
    }
    prediction_entry = {
        "raw_predictions": raw_predictions,
        "refined_predictions": refined_predictions,
    }
    return row, result, prediction_entry


def select_primary_learning_result(results, enabled_baselines):
    priority = (
        "predict_initialized_solver",
        "predict_exact_projection",
        "predict_diff_projection",
        "predict_ray_bisection",
        "predict_only",
    )
    enabled = list(enabled_baselines)
    for name in priority:
        if name in results and name in enabled:
            return name, results[name]
    primary_name = enabled[0]
    return primary_name, results[primary_name]


def build_parametric_learning_metrics(
    *,
    results,
    enabled_baselines,
    seed,
    n_samples,
    prediction_value=None,
    metadata=None,
):
    primary_name, primary = select_primary_learning_result(results, enabled_baselines)
    payload = {
        "predictor": primary["predictor"],
        "refiner": primary["refiner"],
        "seed": int(seed),
        "n_samples": int(n_samples),
        "primary_baseline": primary_name,
        "enabled_baselines": list(enabled_baselines),
        "results": results,
        "objective_mean": primary["objective_mean"],
        "feasibility_rate": primary["feasibility_rate"],
        "violation_mean": primary["violation_mean"],
        "runtime_mean": primary["runtime_mean"],
        "runtime_total": primary["runtime_total"],
        "raw_objective_mean": primary["raw_objective_mean"],
        "refined_objective_mean": primary["objective_mean"],
        "raw_feasibility_rate": primary["raw_feasibility_rate"],
        "refined_feasibility_rate": primary["feasibility_rate"],
        "raw_violation_mean": primary["raw_violation_mean"],
        "refined_violation_mean": primary["violation_mean"],
        "runtime_sec": primary["runtime_sec"],
        "runtime_per_instance_sec": primary["runtime_per_instance_sec"],
        "objective_gap_mean": primary.get("objective_gap_mean"),
        "objective_improvement_mean": primary.get("objective_improvement_mean"),
        "violation_reduction_mean": primary.get("violation_reduction_mean"),
        "solution_mse": primary.get("solution_mse"),
        "solution_mae": primary.get("solution_mae"),
        "raw_objective_gap_mean": primary.get("raw_objective_gap_mean"),
        "raw_solution_mse": primary.get("raw_solution_mse"),
        "raw_solution_mae": primary.get("raw_solution_mae"),
        "predictor_train_time_sec": primary.get("predictor_train_time_sec", 0.0),
        "postprocess_train_time_sec": primary.get("postprocess_train_time_sec", 0.0),
        "refine_time_sec": primary.get("refine_time_sec", primary["runtime_sec"]),
    }
    if prediction_value is not None:
        payload["prediction_value"] = float(prediction_value)
    payload.update(dict(metadata or {}))
    return payload


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
    "PARAMETRIC_LEARNING_RESULT_FIELDS",
    "build_parametric_learning_baseline_output",
    "build_parametric_learning_metrics",
    "make_parametric_learning_result_row",
    "save_parametric_learning_artifacts",
    "save_parametric_learning_plots",
    "select_primary_learning_result",
]
