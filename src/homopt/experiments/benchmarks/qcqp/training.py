"""Shared helpers for QCQP INN benchmark workflows."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from homopt.experiments.common.artifacts import artifact_root
from homopt.experiments.common.parametric import (
    extract_qcqp_inn_instance_metrics,
    load_qcqp_inn_record,
    normalize_qcqp_inn_benchmark_config,
    normalize_qcqp_train_config,
)
from homopt.experiments.common.config import require_explicit_config
from homopt.experiments.common.inn_training import build_qcqp_inn_training_payload, load_or_train_inn_mapping
from .problem import build_qcqp_problem_context
from homopt.experiments.common.runtime import resolve_runtime
from homopt.utils import ensure_dir, set_global_seed


def _load_qcqp_inn_record_from_result(result, case_output_dir):
    del result
    if case_output_dir is None:
        return None
    return load_qcqp_inn_record(case_output_dir)


def _resolve_qcqp_num_test_instance(params):
    params = dict(params)
    if "n_samples" in params:
        raise ValueError("QCQP INN tests use num_test_instance; train_config['n_samples'] controls INN training.")
    if "batch_size" in params or "total_iteration" in params:
        raise ValueError("QCQP INN training controls use train_config, not top-level batch_size/total_iteration.")
    params["num_test_instance"] = int(params.get("num_test_instance", 1))
    return params


def _stable_qcqp_training_cache_label(*, problem_args, model_args, train_args, seed, dtype):
    base = (
        f"qcqp_n{int(problem_args['n_var'])}"
        f"_q{int(problem_args['n_qua_cons'])}"
        f"_seed{int(seed)}"
    )
    payload = {
        "version": 1,
        "seed": int(seed),
        "dtype": str(dtype),
        "problem": problem_args,
        "model": model_args,
        "train": train_args,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._")
    return f"{safe_base}_{digest}"


def _qcqp_training_cache_dir(output_dir, *, problem_args, model_args, train_args, seed, dtype):
    label = _stable_qcqp_training_cache_label(
        problem_args=problem_args,
        model_args=model_args,
        train_args=train_args,
        seed=seed,
        dtype=dtype,
    )
    if output_dir is None:
        root = Path("results") / "_scratch" / "qcqp" / "training_cache"
    else:
        root = Path(output_dir).parent / "training_cache"
    return ensure_dir(root / label)


def _select_batched_history(value, n_samples, instance_idx=0):
    if n_samples <= 1:
        return value
    if hasattr(value, "detach"):
        total = int(value.shape[0])
        if total % int(n_samples) != 0:
            raise ValueError(
                f"Batched history length {total} is not divisible by num_test_instance={int(n_samples)}."
            )
        return value.reshape(total // int(n_samples), int(n_samples), *value.shape[1:])[:, int(instance_idx)]
    array = np.asarray(value)
    if array.shape[0] % int(n_samples) != 0:
        raise ValueError(
            f"Batched history length {array.shape[0]} is not divisible by num_test_instance={int(n_samples)}."
        )
    return array.reshape(array.shape[0] // int(n_samples), int(n_samples), *array.shape[1:])[:, int(instance_idx)]


def _visualize_instance_indices(value, n_samples):
    n_samples = max(1, int(n_samples))
    if value is None:
        raw_indices = [0]
    elif isinstance(value, str) and value.strip().lower() == "all":
        raw_indices = range(n_samples)
    elif isinstance(value, (list, tuple, set)):
        raw_indices = list(value)
    else:
        raw_indices = [value]
    indices = []
    for raw in raw_indices:
        idx = int(raw)
        if idx < 0 or idx >= n_samples:
            raise ValueError(f"visualize_instance_idx={idx} is outside available range [0, {n_samples - 1}].")
        if idx not in indices:
            indices.append(idx)
    return indices or [0]


def _merge_instance_visualization_artifacts(target, source, instance_idx, multiple):
    if not multiple:
        target.update(source)
        return
    for key, value in source.items():
        target[f"{key}_inst{int(instance_idx)}"] = value


def _build_qcqp_inn_case_context(output_dir, params):
    params = _resolve_qcqp_num_test_instance(dict(params))
    seed = int(params.get("seed", 2025))
    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(
        params.get("device"),
        params.get("dtype"),
        default_device="auto",
    )
    artifact_dir = artifact_root(output_dir)
    normalized = normalize_qcqp_inn_benchmark_config(
        seed=seed,
        n_var=int(params.get("n_var", 2)),
        n_qua_cons=int(params.get("n_qua_cons", 3)),
        n_linear_cons=int(params.get("n_linear_cons", 0)),
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        n_samples=10000,
        batch_size=256,
        total_iteration=10000,
        base_inn_args={},
        base_optimizer_config={},
        problem_config=params.get("problem_config"),
        model_config=require_explicit_config(params.get("model_config"), "model_config"),
        optimizer_config=require_explicit_config(params.get("optimizer_config"), "optimizer_config"),
        train_config=require_explicit_config(params.get("train_config"), "train_config"),
    )
    problem_args = normalized["problem"]
    problem_context = build_qcqp_problem_context(
        seed=seed,
        problem_args=problem_args,
        device=runtime_device,
        dtype=runtime_dtype,
    )
    data = problem_context["qc_problem"]
    training_payload = build_qcqp_inn_training_payload(
        problem_args,
        normalized["model"],
        normalized["train"],
        ensure_results_save_freq=True,
    )
    save_dir = _qcqp_training_cache_dir(
        output_dir,
        problem_args=problem_args,
        model_args=normalized["model"],
        train_args=normalized["train"],
        seed=seed,
        dtype=runtime_dtype,
    )
    return {
        "params": params,
        "seed": seed,
        "runtime_device": runtime_device,
        "runtime_dtype": runtime_dtype,
        "artifact_dir": artifact_dir,
        "save_dir": save_dir,
        "normalized": normalized,
        "problem_args": problem_args,
        "model_args": normalized["model"],
        "optimizer_args": normalized["optimizer"],
        "train_args": normalized["train"],
        "data": data,
        "training_payload": training_payload,
    }


def _load_or_train_qcqp_inn_case(case_context, *, retrain):
    return load_or_train_inn_mapping(
        case_context["data"],
        case_context["training_payload"],
        case_context["save_dir"],
        retrain=bool(retrain),
    )


def _sample_qcqp_inn_test_instances(case_context):
    """Sample the test instance batch once and pass it to every compared solver."""

    return case_context["data"].sample_instance_batch(
        n_instances=int(case_context["params"]["num_test_instance"]),
        seed=int(case_context["seed"]),
        sample_obj=True,
        device=case_context["runtime_device"],
        dtype=case_context["runtime_dtype"],
    )


def _prepare_qcqp_inn_training_case(output_dir, params):
    """Prepare the INN mapping for one QCQP case without running test baselines."""

    case_context = _build_qcqp_inn_case_context(output_dir, params)
    return _load_or_train_qcqp_inn_case(case_context, retrain=case_context["params"].get("retrain", False))


def _prepare_qcqp_inn_training_context(case_context):
    """Prepare the INN mapping from an already-built QCQP case context."""

    return _load_or_train_qcqp_inn_case(
        case_context,
        retrain=case_context["params"].get("retrain", False),
    )


def inn_training_benchmark(
    seed=2025,
    n_samples=16,
    batch_size=4,
    total_iteration=4,
    max_iterations=None,
    problem_config=None,
    model_config=None,
    optimizer_config=None,
    train_config=None,
    visualize=False,
    visualize_mdh_mapping=False,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Package-native INN training smoke with artifact output."""

    from .sweep import qcqp_inn_experiment

    model_config = require_explicit_config(model_config, "model_config")
    optimizer_config = require_explicit_config(optimizer_config, "optimizer_config")
    train_args = normalize_qcqp_train_config(
        n_samples=n_samples,
        batch_size=batch_size,
        total_iteration=total_iteration,
        train_config=train_config,
    )
    params = {
        "seed": seed,
        "n_var": int((problem_config or {}).get("n_var", 2)),
        "n_qua_cons": int((problem_config or {}).get("n_qua_cons", 1)),
        "n_linear_cons": int((problem_config or {}).get("n_linear_cons", 0)),
        "num_test_instance": 1,
        "max_iterations": 5 if max_iterations is None else int(max_iterations),
        "problem_config": problem_config,
        "model_config": model_config,
        "optimizer_config": optimizer_config,
        "train_config": train_args,
        "retrain": True,
        "visualize": visualize,
        "visualize_mdh_mapping": visualize_mdh_mapping,
        "lagrangian_baselines": [],
        "compare_ipopt_baseline": False,
        "output_dir": output_dir,
        "device": device,
        "dtype": dtype,
    }
    return qcqp_inn_experiment(**params)


__all__ = [
    "_build_qcqp_inn_case_context",
    "_load_or_train_qcqp_inn_case",
    "_load_qcqp_inn_record_from_result",
    "_merge_instance_visualization_artifacts",
    "_prepare_qcqp_inn_training_case",
    "_prepare_qcqp_inn_training_context",
    "_qcqp_training_cache_dir",
    "_resolve_qcqp_num_test_instance",
    "_sample_qcqp_inn_test_instances",
    "_select_batched_history",
    "_stable_qcqp_training_cache_label",
    "_visualize_instance_indices",
    "inn_training_benchmark",
]
