"""Small orchestration helpers for experiment execution."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .results import ExperimentResult, save_result


def _normalize_payload_dict(name, payload):
    allow_top_level_metrics = bool(payload.pop("_allow_top_level_metrics", False))
    allowed_top_level = {"name", "objective", "feasible", "artifacts", "metrics", "context"}
    unknown_keys = sorted(set(payload) - allowed_top_level)
    if unknown_keys and not allow_top_level_metrics:
        raise ValueError(f"Experiment payload has unknown top-level fields: {unknown_keys}. Put metrics under 'metrics'.")
    result_name = payload.get("name", name)
    artifacts = dict(payload.get("artifacts", {}))
    context = dict(payload.get("context", {}))

    if "metrics" in payload:
        nested_metrics = payload.get("metrics", {})
        if nested_metrics is None:
            nested_metrics = {}
        if not isinstance(nested_metrics, dict):
            raise TypeError("payload['metrics'] must be a dict when provided.")
    else:
        nested_metrics = {}

    metrics = {}
    if allow_top_level_metrics:
        for key, value in payload.items():
            if key in allowed_top_level:
                continue
            metrics[key] = value
    metrics.update(dict(nested_metrics))
    return ExperimentResult(
        name=result_name,
        objective=payload.get("objective", None),
        feasible=payload.get("feasible", None),
        metrics=metrics,
        artifacts=artifacts,
        context=context,
    )


def run_and_record(name, run_fn, output_dir, config=None, context=None):
    start = perf_counter()
    payload = run_fn()
    elapsed = perf_counter() - start

    if isinstance(payload, ExperimentResult):
        result = payload
    elif isinstance(payload, dict):
        result = _normalize_payload_dict(name=name, payload=dict(payload))
    else:
        raise TypeError('run_fn must return ExperimentResult or dict')

    if context is not None:
        result.context = dict(context)
    result.metrics['elapsed_sec'] = elapsed
    save_result(result, Path(output_dir), config=config)
    return result
