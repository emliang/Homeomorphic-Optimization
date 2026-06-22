"""Shared predictor/post-processing benchmark runner for parametric problems."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from homopt.records.artifacts import artifact_mapping, artifact_root, build_benchmark_payload
from homopt.experiments.common.config import require_explicit_config
from homopt.experiments.common.io import ensure_dir
from homopt.experiments.common.parametric_runtime import run_prediction_refinement_route, sample_parametric_instances
from homopt.experiments.common.reports import (
    build_parametric_learning_baseline_output,
    build_parametric_learning_metrics,
)
from homopt.experiments.common.learning_reports import save_parametric_learning_artifacts
from homopt.experiments.common.runtime import resolve_runtime
from homopt.learning import ConstantDecisionPredictor, NeuralDecisionPredictor
from homopt.learning.training import prepare_decision_predictor
from homopt.learning.training.cache import training_time_from_record
from homopt.utils import set_global_seed


@dataclass(frozen=True)
class ParametricLearningSpec:
    family: str
    artifact_prefix: str
    supported_baselines: tuple[str, ...]
    build_problem: Callable
    make_refiner: Callable
    metadata: Callable


def normalize_learning_baselines(baselines, supported_baselines, *, family):
    supported = set(supported_baselines)
    normalized = []
    seen = set()
    for name in baselines:
        name = str(name).strip()
        if name not in supported:
            raise ValueError(f"Unsupported {family} learning baseline: {name}")
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def prepare_parametric_predictor(
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
    family,
):
    predictor_key = str(predictor_type).strip().lower()
    if predictor_key in {"constant", "constant_decision"}:
        return {
            "predictor": ConstantDecisionPredictor(problem.nvar, value=prediction_value),
            "train_time": 0.0,
        }, {}
    if predictor_key not in {"nn", "nn_decision", "neural", "neural_decision"}:
        raise ValueError(f"Unsupported {family} predictor_type: {predictor_type}")

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
    predictor_resource = prepare_decision_predictor(
        problem,
        problem_args=problem_args,
        model_args=model_cfg,
        train_args=train_cfg,
        save_dir=save_dir / "decision_predictor",
        retrain=retrain,
        ensure_results_save_freq=True,
    )
    model = predictor_resource["model"]
    record = predictor_resource["training_record"]
    model_path = predictor_resource["model_path"]
    record_path = predictor_resource["record_path"]
    return {
        "predictor": NeuralDecisionPredictor(model, problem),
        "train_time": training_time_from_record(record),
    }, artifact_mapping(
        output_dir,
        predictor_model=model_path,
        predictor_training_record=record_path,
    )


def run_parametric_learning_benchmark(
    *,
    spec: ParametricLearningSpec,
    seed=2026,
    n_samples=8,
    predictor_type="constant",
    prediction_value=0.0,
    baselines=None,
    predictor_model_config=None,
    predictor_train_config=None,
    projection_config=None,
    retrain=False,
    output_dir=None,
    device=None,
    dtype=None,
    problem_kwargs=None,
):
    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    enabled = normalize_learning_baselines(
        spec.supported_baselines if baselines is None else baselines,
        spec.supported_baselines,
        family=spec.family,
    )
    if not enabled:
        raise ValueError(f"{spec.family} learning benchmark requires at least one baseline.")

    problem_kwargs = dict(problem_kwargs or {})
    data_start = perf_counter()
    problem = spec.build_problem(
        seed=seed,
        device=runtime_device,
        dtype=runtime_dtype,
        **problem_kwargs,
    )
    instance_batch = sample_parametric_instances(problem, n_instances=n_samples, seed=seed, sample_obj=False)
    data_collection_time = float(perf_counter() - data_start)
    problem_args = spec.metadata(problem, seed=seed, **problem_kwargs)

    artifact_dir = artifact_root(output_dir)
    save_dir = artifact_dir if artifact_dir is not None else ensure_dir(Path("results") / "_scratch" / spec.artifact_prefix)
    predictor_resource, predictor_artifacts = prepare_parametric_predictor(
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
        family=spec.family,
    )

    rows = []
    results = {}
    prediction_payload = {}
    tol = float((projection_config or {}).get("proj_eps", (projection_config or {}).get("tol", 1e-8)))
    for baseline in enabled:
        refiner = spec.make_refiner(baseline, projection_config)
        summary, runtime_sec = run_prediction_refinement_route(
            problem=problem,
            input_samples=instance_batch.inputs,
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
    metrics["workflow_stages"] = [
        "data_collection",
        "predictor_training_or_loading",
        "postprocess_evaluation",
    ]
    metrics["data_collection_time_sec"] = data_collection_time
    artifacts = save_parametric_learning_artifacts(
        output_dir,
        metrics=metrics,
        rows=rows,
        prediction_payload=prediction_payload,
        artifact_prefix=spec.artifact_prefix,
    )
    artifacts.update(predictor_artifacts)
    return build_benchmark_payload(
        objective=float(metrics["refined_objective_mean"]),
        feasible=metrics["refined_feasibility_rate"] >= 1.0 - 1e-12,
        artifacts=artifacts,
        **metrics,
    )


__all__ = [
    "ParametricLearningSpec",
    "normalize_learning_baselines",
    "prepare_parametric_predictor",
    "run_parametric_learning_benchmark",
]
