"""Convex-parametric predictor plus post-processing benchmarks."""

from __future__ import annotations

from pathlib import Path

from homopt.experiments.common.artifacts import artifact_mapping, artifact_root, build_benchmark_payload
from homopt.experiments.common.config import require_explicit_config
from homopt.experiments.common.learning_training import build_predictor_training_payload, load_or_train_decision_predictor
from homopt.experiments.common.parametric_reports import (
    build_parametric_learning_baseline_output,
    build_parametric_learning_metrics,
    save_parametric_learning_artifacts,
)
from homopt.experiments.common.parametric_runtime import run_prediction_refinement_route, sample_parametric_instances
from homopt.experiments.common.runtime import resolve_runtime
from homopt.experiments.common.inn_training import training_time_from_record
from homopt.learning import ConstantDecisionPredictor, DiffProjectionRefiner, IdentityRefiner, NeuralDecisionPredictor
from homopt.problems import ParametricConvexQCQP, ParametricQP, ParametricSDP, ParametricSOCP
from homopt.utils import ensure_dir, set_global_seed


CONVEX_PARAMETRIC_LEARNING_BASELINES = (
    "predict_only",
    "predict_diff_projection",
)


def _normalize_baselines(baselines):
    supported = set(CONVEX_PARAMETRIC_LEARNING_BASELINES)
    normalized = []
    seen = set()
    for name in baselines:
        name = str(name).strip()
        if name not in supported:
            raise ValueError(f"Unsupported convex-parametric learning baseline: {name}")
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _build_convex_parametric_problem(
    *,
    problem_type,
    seed,
    n_var,
    n_eq,
    n_ineq,
    matrix_dim,
    problem_config,
    device,
    dtype,
):
    config = dict(problem_config or {})
    problem_type = str(problem_type).strip().lower()
    common = {
        "n_eq": int(n_eq),
        "seed": int(seed),
        "device": device,
        "dtype": dtype,
        **config,
    }
    if problem_type == "qp":
        return ParametricQP(n_var=int(n_var), n_ineq=int(n_ineq), **common)
    if problem_type in {"qcqp", "convex_qcqp"}:
        return ParametricConvexQCQP(n_var=int(n_var), n_ineq=int(n_ineq), **common)
    if problem_type == "socp":
        return ParametricSOCP(n_var=int(n_var), n_ineq=int(n_ineq), **common)
    if problem_type == "sdp":
        return ParametricSDP(matrix_dim=int(matrix_dim), **common)
    raise ValueError(f"Unsupported convex-parametric problem_type: {problem_type}")


def _make_refiner(baseline, projection_config):
    if baseline == "predict_only":
        return IdentityRefiner()
    if baseline == "predict_diff_projection":
        cfg = dict(projection_config or {})
        return DiffProjectionRefiner(
            steps=int(cfg.get("proj_max_steps", cfg.get("steps", 30))),
            lr=float(cfg.get("corr_lr", cfg.get("lr", 1e-3))),
            momentum=float(cfg.get("corr_momentum", cfg.get("momentum", 0.5))),
            tol=float(cfg.get("proj_eps", cfg.get("tol", 1e-8))),
        )
    raise ValueError(f"Unsupported convex-parametric baseline: {baseline}")


def _prepare_predictor(
    *,
    predictor_type,
    prediction_value,
    seed,
    n_samples,
    runtime_device,
    runtime_dtype,
    problem,
    problem_args,
    save_dir,
    output_dir,
    retrain,
    predictor_model_config,
    predictor_train_config,
):
    predictor_key = str(predictor_type).strip().lower()
    if predictor_key in {"constant", "constant_decision"}:
        return {
            "predictor": ConstantDecisionPredictor(problem.nvar, value=prediction_value),
            "train_time": 0.0,
        }, {}
    if predictor_key not in {"nn", "nn_decision", "neural", "neural_decision"}:
        raise ValueError(f"Unsupported convex-parametric predictor_type: {predictor_type}")

    model_cfg = {
        **require_explicit_config(predictor_model_config, "predictor_model_config"),
        "seed": int(seed),
        "device": str(runtime_device),
        "dtype": runtime_dtype,
    }
    train_cfg = {
        "n_samples": max(int(n_samples), 64),
        "batch_size": min(64, max(int(n_samples), 1)),
        "total_iteration": 1000,
        **dict(predictor_train_config or {}),
    }
    predictor_args = build_predictor_training_payload(
        problem_args,
        model_cfg,
        train_cfg,
        ensure_results_save_freq=True,
    )
    model, record, model_path, record_path = load_or_train_decision_predictor(
        problem,
        predictor_args,
        save_dir / "decision_predictor",
        retrain=retrain,
    )
    return {
        "predictor": NeuralDecisionPredictor(model, problem),
        "train_time": training_time_from_record(record),
    }, artifact_mapping(
        output_dir,
        predictor_model=model_path,
        predictor_training_record=record_path,
    )


def convex_parametric_learning_benchmark(
    seed=2026,
    problem_type="qp",
    n_var=10,
    n_eq=2,
    n_ineq=10,
    matrix_dim=4,
    n_samples=8,
    predictor_type="constant",
    prediction_value=0.0,
    baselines=None,
    problem_config=None,
    predictor_model_config=None,
    predictor_train_config=None,
    projection_config=None,
    retrain=False,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Run predictor/postprocess baselines for convex parametric problem families."""

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    enabled = _normalize_baselines(CONVEX_PARAMETRIC_LEARNING_BASELINES if baselines is None else baselines)
    if not enabled:
        raise ValueError("convex_parametric_learning_benchmark requires at least one baseline.")

    problem = _build_convex_parametric_problem(
        problem_type=problem_type,
        seed=seed,
        n_var=n_var,
        n_eq=n_eq,
        n_ineq=n_ineq,
        matrix_dim=matrix_dim,
        problem_config=problem_config,
        device=runtime_device,
        dtype=runtime_dtype,
    )
    instance_batch = sample_parametric_instances(problem, n_instances=n_samples, seed=seed, sample_obj=False)
    input_samples = instance_batch.inputs
    problem_args = {
        "problem_family": "convex_parametric",
        "problem_type": str(problem_type),
        "n_var": int(problem.nvar),
        "n_eq": int(n_eq),
        "n_ineq": int(n_ineq),
        "matrix_dim": int(matrix_dim),
        "seed": int(seed),
    }

    artifact_dir = artifact_root(output_dir)
    save_dir = artifact_dir if artifact_dir is not None else ensure_dir(Path("results") / "_scratch" / "convex_parametric_learning")
    predictor_resource, predictor_artifacts = _prepare_predictor(
        predictor_type=predictor_type,
        prediction_value=prediction_value,
        seed=seed,
        n_samples=n_samples,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        problem=problem,
        problem_args=problem_args,
        save_dir=save_dir,
        output_dir=output_dir,
        retrain=retrain,
        predictor_model_config=predictor_model_config,
        predictor_train_config=predictor_train_config,
    )

    rows = []
    results = {}
    prediction_payload = {}
    tol = float((projection_config or {}).get("proj_eps", (projection_config or {}).get("tol", 1e-8)))
    for baseline in enabled:
        refiner = _make_refiner(baseline, projection_config)
        summary, runtime_sec = run_prediction_refinement_route(
            problem=problem,
            input_samples=input_samples,
            predictor=predictor_resource["predictor"],
            refiner=refiner,
            tol=tol,
            objective_batch=instance_batch.objectives,
            include_predictions=True,
        )
        raw_predictions = summary.pop("raw_predictions")
        refined_predictions = summary.pop("refined_predictions")
        row, result, prediction_entry = build_parametric_learning_baseline_output(
            baseline=baseline,
            summary=summary,
            runtime_sec=runtime_sec,
            n_samples=n_samples,
            raw_predictions=raw_predictions,
            refined_predictions=refined_predictions,
            predictor_train_time=predictor_resource["train_time"],
            postprocess_train_time=0.0,
        )
        rows.append(row)
        results[baseline] = result
        prediction_payload[baseline] = prediction_entry

    metrics = build_parametric_learning_metrics(
        results=results,
        enabled_baselines=enabled,
        seed=seed,
        n_samples=n_samples,
        prediction_value=prediction_value,
        metadata=problem_args,
    )
    artifacts = save_parametric_learning_artifacts(
        output_dir,
        metrics=metrics,
        rows=rows,
        prediction_payload=prediction_payload,
        artifact_prefix="convex_parametric_learning",
    )
    artifacts.update(predictor_artifacts)
    return build_benchmark_payload(
        objective=float(metrics["refined_objective_mean"]),
        feasible=metrics["refined_feasibility_rate"] >= 1.0 - 1e-12,
        artifacts=artifacts,
        **metrics,
    )


__all__ = ["CONVEX_PARAMETRIC_LEARNING_BASELINES", "convex_parametric_learning_benchmark"]
