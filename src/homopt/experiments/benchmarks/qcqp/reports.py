"""QCQP rows, metrics, and comparison views."""

from __future__ import annotations

import numpy as np

from homopt.records.artifacts import load_incremental_comparison_artifacts
from homopt.experiments.common.comparison import (
    build_comparison_views,
    COMPARISON_METRIC_FIELDS,
    COMPARISON_ROW_FIELDS,
    comparison_metric_source,
    extract_comparison_metrics,
    make_comparison_row,
)


QCQP_INN_METHOD_NAME = "INN-PGD"
QCQP_LEARNING_BASELINES = ("predict_only", "predict_ray_bisection")
QCQP_POSTPROCESS_BASELINES = (
    "predict_diff_projection",
    "predict_exact_projection",
    "predict_homeomorphic_projection",
    "predict_ip_bisection",
    "predict_initialized_opt",
)
QCQP_LEARNING_CANONICAL_BASELINES = QCQP_LEARNING_BASELINES + QCQP_POSTPROCESS_BASELINES
QCQP_LEARNING_RESULT_FIELDS = (
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
QCQP_ROUTE_COMPARISON_FIELDS = COMPARISON_ROW_FIELDS
QCQP_ROUTE_METRIC_FIELDS = COMPARISON_METRIC_FIELDS


def normalize_qcqp_learning_baseline_name(name):
    canonical = str(name).strip()
    if canonical not in QCQP_LEARNING_CANONICAL_BASELINES:
        raise ValueError(f"Unsupported QCQP learning baseline: {name}")
    return canonical


def normalize_qcqp_learning_baselines(baselines):
    normalized = []
    seen = set()
    for name in baselines:
        name = normalize_qcqp_learning_baseline_name(name)
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def load_qcqp_inn_record(output_dir):
    records, _, _, _ = load_incremental_comparison_artifacts(
        output_dir,
        algorithms=[QCQP_INN_METHOD_NAME],
    )
    return records[QCQP_INN_METHOD_NAME]


def extract_qcqp_inn_instance_metrics_from_record(record, n_samples):
    obj_hist = np.asarray(record.get("obj_trajectory", []), dtype=float).reshape(-1)
    cons_hist = np.asarray(record.get("cons_trajectory", []), dtype=float).reshape(-1)
    per_iter = np.asarray(record.get("per_iter_time", []), dtype=float).reshape(-1)
    if obj_hist.size < n_samples or cons_hist.size < n_samples:
        raise ValueError(
            "QCQP INN record does not contain enough final instance metrics: "
            f"obj={obj_hist.size}, cons={cons_hist.size}, num_test_instance={int(n_samples)}."
        )
    final_obj = obj_hist[-n_samples:]
    final_cons = cons_hist[-n_samples:]
    total_runtime = float(record.get("total_wall_time")) if record.get("total_wall_time") is not None else (
        float(per_iter.sum()) if per_iter.size > 0 else float("nan")
    )
    per_sample_runtime = total_runtime / n_samples if n_samples > 0 and np.isfinite(total_runtime) else float("nan")
    return [
        {
            "index": idx,
            "objective": float(final_obj[idx]),
            "violation": float(final_cons[idx]),
            "feasible": bool(final_cons[idx] <= 1e-5),
            "runtime_est": float(per_sample_runtime),
        }
        for idx in range(n_samples)
    ]


def extract_qcqp_inn_instance_metrics(payload, output_dir, n_samples):
    del payload
    if output_dir is None:
        raise ValueError("QCQP INN metrics require output_dir so the comparison record manifest can be loaded.")
    record = load_qcqp_inn_record(output_dir)
    return extract_qcqp_inn_instance_metrics_from_record(record, n_samples)


def summarize_qcqp_inn_record(record, n_samples):
    rows = extract_qcqp_inn_instance_metrics_from_record(record, n_samples)
    obj = [r["objective"] for r in rows if r.get("objective") is not None]
    vio = [r["violation"] for r in rows if r.get("violation") is not None]
    rt = [r["runtime_est"] for r in rows if np.isfinite(r.get("runtime_est", np.nan))]
    return {
        "objective_mean": float(np.mean(obj)) if obj else None,
        "objective_best": float(np.min(obj)) if obj else None,
        "feasibility_rate": float(np.mean([1.0 if r.get("feasible", False) else 0.0 for r in rows])) if rows else 0.0,
        "violation_mean": float(np.mean(vio)) if vio else None,
        "runtime_mean": float(np.mean(rt)) if rt else None,
        "runtime_total": float(np.sum(rt)) if rt else None,
    }


def summarize_qcqp_inn_result(payload, output_dir, n_samples):
    del payload
    return summarize_qcqp_inn_record(load_qcqp_inn_record(output_dir), n_samples)

def make_qcqp_learning_result_row(
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


def summarize_qcqp_learning_result(baseline, summary, runtime_total, n_samples):
    """Normalize a QCQP learning baseline result to a comparison-friendly row."""

    return make_qcqp_learning_result_row(
        baseline=baseline,
        predictor=summary["predictor"],
        refiner=summary["refiner"],
        objective_mean=summary["refined_objective_mean"],
        feasibility_rate=summary["refined_feasibility_rate"],
        violation_mean=summary["refined_violation_mean"],
        runtime_total=runtime_total,
        n_samples=n_samples,
        raw_objective_mean=summary["raw_objective_mean"],
        raw_feasibility_rate=summary["raw_feasibility_rate"],
        raw_violation_mean=summary["raw_violation_mean"],
    )


def build_qcqp_learning_baseline_output(
    *,
    baseline,
    summary,
    runtime_sec,
    n_samples,
    raw_predictions,
    refined_predictions,
    predictor=None,
    refiner=None,
    objective_mean=None,
    feasibility_rate=None,
    violation_mean=None,
    predictor_train_time=0.0,
    postprocess_train_time=0.0,
):
    """Build the shared row/result/artifact payload for one QCQP learning baseline."""

    predictor_name = summary["predictor"] if predictor is None else predictor
    refiner_name = summary["refiner"] if refiner is None else refiner
    final_objective = summary["refined_objective_mean"] if objective_mean is None else objective_mean
    final_feasibility = summary["refined_feasibility_rate"] if feasibility_rate is None else feasibility_rate
    final_violation = summary["refined_violation_mean"] if violation_mean is None else violation_mean

    row = make_qcqp_learning_result_row(
        baseline=baseline,
        predictor=predictor_name,
        refiner=refiner_name,
        objective_mean=final_objective,
        feasibility_rate=final_feasibility,
        violation_mean=final_violation,
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
        "predictor": predictor_name,
        "refiner": refiner_name,
        "refined_objective_mean": row["objective_mean"],
        "refined_feasibility_rate": row["feasibility_rate"],
        "refined_violation_mean": row["violation_mean"],
        "runtime_sec": runtime_sec,
        "total_wall_time": runtime_sec,
        "runtime_per_instance_sec": runtime_sec / max(int(n_samples), 1),
        "objective_gap_mean": row["objective_gap_mean"],
        "objective_improvement_mean": row["objective_improvement_mean"],
        "violation_reduction_mean": row["violation_reduction_mean"],
        "solution_mse": row["solution_mse"],
        "solution_mae": row["solution_mae"],
        "objective_mean": row["objective_mean"],
        "feasibility_rate": row["feasibility_rate"],
        "violation_mean": row["violation_mean"],
        "runtime_mean": row["runtime_mean"],
        "runtime_total": row["runtime_total"],
        "predictor_train_time_sec": row["predictor_train_time_sec"],
        "postprocess_train_time_sec": row["postprocess_train_time_sec"],
        "refine_time_sec": row["refine_time_sec"],
    }
    prediction_entry = {
        "raw_predictions": raw_predictions,
        "refined_predictions": refined_predictions,
    }
    return row, result, prediction_entry


def summarize_qcqp_warmstart_solver_rows(solver_rows):
    obj = [row["objective"] for row in solver_rows if row.get("objective") is not None]
    vio = [row["max_violation"] for row in solver_rows if row.get("max_violation") is not None]
    feasibility_rate = float(sum(1.0 if row.get("feasible", False) else 0.0 for row in solver_rows) / len(solver_rows)) if solver_rows else 0.0
    return {
        "objective_mean": float(sum(obj) / len(obj)) if obj else None,
        "feasibility_rate": feasibility_rate,
        "violation_mean": float(sum(vio) / len(vio)) if vio else None,
    }


def build_qcqp_route_rows(*, iterative_payload, learning_results):
    rows = [
        make_qcqp_route_result_row(
            route="iterative",
            method="inn_pgd",
            summary=iterative_payload,
        )
    ]
    for method, result in dict(learning_results or {}).items():
        rows.append(make_qcqp_route_result_row(route="learning", method=method, summary=result))
    return rows


def select_qcqp_learning_primary_baseline(results, enabled_baselines):
    """Select the shared top-level baseline view for QCQP learning benchmarks."""
    priority = (
        "predict_homeomorphic_projection",
        "predict_ip_bisection",
        "predict_initialized_opt",
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


def build_qcqp_learning_benchmark_metrics(
    *,
    results,
    enabled_baselines,
    seed,
    n_var,
    n_qua_cons,
    n_linear_cons,
    n_samples,
    prediction_value,
):
    """Build the shared top-level metric view for the QCQP learning benchmark."""

    primary_name, primary = select_qcqp_learning_primary_baseline(results, enabled_baselines)
    return {
        "predictor": primary["predictor"],
        "refiner": primary["refiner"],
        "seed": int(seed),
        "n_var": int(n_var),
        "n_qua_cons": int(n_qua_cons),
        "n_linear_cons": int(n_linear_cons),
        "n_samples": int(n_samples),
        "prediction_value": float(prediction_value),
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


def _objective_sort_value(row, objective_key):
    value = row.get(objective_key)
    return float("inf") if value is None else float(value)


def _select_best_qcqp_row(rows, *, objective_key, is_feasible):
    rows = list(rows)
    feasible_rows = [row for row in rows if is_feasible(row)]
    return min(
        feasible_rows or rows,
        key=lambda row: _objective_sort_value(row, objective_key),
    )


def make_qcqp_route_result_row(*, route, method, summary):
    """Normalize a QCQP route summary to the shared cross-route comparison row."""

    route_metrics = extract_qcqp_route_metrics(summary)
    source = comparison_metric_source(summary)
    timing_extras = {
        key: source.get(key)
        for key in (
            "total_wall_time",
            "total_iter_time",
            "total_outer_iter_time",
            "avg_outer_iter_time",
            "mean_inner_iter_time",
            "mean_solver_iter_time",
            "initial_ip_time",
            "last_trans_time",
        )
        if source.get(key) is not None
    }
    return make_comparison_row(
        route=route,
        method=method,
        objective=route_metrics["objective_mean"],
        feasible=(route_metrics["feasibility_rate"] or 0.0) > 0.0,
        violation=route_metrics["violation_mean"],
        runtime_total=route_metrics["runtime_total"],
        runtime_mean=route_metrics["runtime_mean"],
        **timing_extras,
    )


def extract_qcqp_route_metrics(summary):
    """Extract the shared comparison metrics from a route-level summary payload."""

    return extract_comparison_metrics(summary)


def summarize_qcqp_sweep_rows(rows):
    """Build shared top-level summary fields for a QCQP sweep comparison."""

    best_row = _select_best_qcqp_row(
        rows,
        objective_key="objective",
        is_feasible=lambda row: bool(row.get("feasible")),
    )
    return {
        "objective": best_row.get("objective"),
        "feasible": all(bool(row.get("feasible")) for row in rows),
        "best_case": {
            "n_var": int(best_row["n_var"]),
            "n_qua_cons": int(best_row["n_qua_cons"]),
        },
    }


def summarize_qcqp_learning_route_metrics(results):
    """Build a shared metric-only view over learning-route baseline results."""

    return {
        str(method): extract_qcqp_route_metrics(summary)
        for method, summary in dict(results or {}).items()
    }


def build_qcqp_route_comparison_views(*, rows, iterative_payload, learning_results):
    """Build the shared result views exposed by the QCQP route comparison wrapper."""

    shared_views = build_comparison_views(rows)
    return {
        "objective": shared_views["objective"],
        "feasible": shared_views["feasible"],
        "results": shared_views["instance_rows"],
        "instance_rows": shared_views["instance_rows"],
        "method_summary_rows": shared_views["method_summary_rows"],
        "comparison_rows": shared_views["comparison_rows"],
        "comparison_summary": shared_views["comparison_summary"],
        "comparison_summary_rows": shared_views["comparison_summary_rows"],
        "iterative": extract_qcqp_route_metrics(iterative_payload),
        "learning": learning_results,
        "learning_summary": summarize_qcqp_learning_route_metrics(learning_results),
        "best_method": shared_views["best_method"],
    }
