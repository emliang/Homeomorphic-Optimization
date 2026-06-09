"""Shared helpers for parametric benchmark entrypoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._benchmark_common import load_incremental_comparison_artifacts, merged, normalize_solver_run_config, save_numpy
from ._qcqp_helpers import build_qcqp_problem_args
from homopt.models import (
    load_decision_predictor,
    load_inn_mapping,
    load_ipnn_mapping,
    save_decision_predictor,
    save_inn_mapping,
    save_ipnn_mapping,
    train_decision_predictor,
    train_ipnn_mapping,
    train_mdh_mapping,
)


def default_qcqp_inn_args():
    return {
        "num_layer": 3,
        "inv_type": "coupling",
        "h_dim": 8,
        "Con_type": "PI",
        "w_penalty": 1.0,
        "w_distortion": 0.1,
        "w_lipschitz": 0.0,
        "center": 0.0,
        "lr": 1e-3,
        "lr_decay_step": 2,
        "lr_decay": 0.9,
        "ema_decay": 0.99,
    }


def default_qcqp_optimizer_config():
    return {
        "learning_rate": 1e-2,
        "max_running_time": 30,
        "convergence_threshold": 1e-6,
        "lr_decay": 0.9,
        "feasibility_eps": 1e-6,
        "opt": "gd",
        "momentum": 0.0,
        "stepsize_rule": "constant",
        "initial_latent_mode": "center",
        "initial_latent_radius": 1.0,
    }


def default_qcqp_ipnn_args():
    return {
        "num_layer": 3,
        "h_dim": 64,
        "fixed_margin": False,
        "gamma": 1e-3,
        "noise_type": "add",
        "outact": "tanh",
        "lr": 1e-4,
        "lr_decay_step": 1000,
        "lr_decay": 0.9,
        "resultsSaveFreq": 1000,
    }


def default_qcqp_projection_args():
    return {
        "proj_eps": 1e-5,
        "proj_max_steps": 30,
        "step_size": 0.5,
        "corr_lr": 1e-3,
        "corr_momentum": 0.5,
    }


def default_qcqp_predictor_args():
    return {
        "approach": "supervise",
        "supervise_target": "fixed_center",
        "pre_training": 0,
        "num_layer": 3,
        "h_dim": 64,
        "outact": "tanh",
        "dropout": 0.1,
        "lr": 1e-4,
        "lr_decay_step": 1000,
        "lr_decay": 0.9,
        "w_obj": 0.01,
        "w_ineq": 0.001,
        "resultsSaveFreq": 1000,
    }


def build_inn_runtime_args(
    *,
    runtime_device,
    runtime_dtype,
    base_inn_args,
    base_optimizer_config,
    model_config=None,
    model_config_overrides=None,
    optimizer_config=None,
    optimizer_config_overrides=None,
):
    model_args = merged(base_inn_args, model_config)
    model_args.update(merged({}, model_config_overrides))
    model_args["device"] = str(runtime_device)
    model_args["dtype"] = runtime_dtype

    optimizer_args = merged(base_optimizer_config, optimizer_config)
    optimizer_args.update(merged({}, optimizer_config_overrides))
    return model_args, optimizer_args


def normalize_qcqp_problem_config(
    *,
    seed,
    n_var,
    n_qua_cons,
    n_linear_cons,
    problem_config=None,
    problem_config_overrides=None,
):
    merged_problem = merged(problem_config, problem_config_overrides)
    return build_qcqp_problem_args(
        seed=seed,
        n_var=n_var,
        n_qua_cons=n_qua_cons,
        n_linear_cons=n_linear_cons,
        problem_config=merged_problem,
    )


def normalize_jcc_linear_problem_config(
    *,
    problem_family="jcc_linear",
    seed,
    n_var,
    n_input_dim,
    n_ineq,
    n_scenarios,
    epsilon,
    problem_config=None,
    problem_config_overrides=None,
):
    merged_problem = merged(problem_config, problem_config_overrides)
    resolved_family = str(
        merged_problem.pop("problem_family", merged_problem.pop("family", problem_family))
    ).strip().lower()
    if resolved_family not in {"jcc_linear", "jccim"}:
        raise ValueError(f"Unsupported JCC chance problem family: {resolved_family}")
    return {
        "problem_family": resolved_family,
        "n_var": int(n_var),
        "n_input_dim": int(n_input_dim),
        "n_ineq": int(n_ineq),
        "n_scenarios": int(n_scenarios),
        "epsilon": float(epsilon),
        "seed": int(seed),
        **dict(merged_problem or {}),
    }


def _normalize_jcc_linear_solver_group(default_solver_name, default_config, payload):
    translated_payload = merged({}, payload)
    if "name" in translated_payload and "solver_name" not in translated_payload:
        translated_payload["solver_name"] = translated_payload.pop("name")
    if "options" in translated_payload and "solver_options" not in translated_payload:
        translated_payload["solver_options"] = translated_payload.pop("options")
    cfg = merged(default_config, translated_payload)
    solver_name = cfg.pop("solver_name", cfg.get("solver", default_solver_name))
    cfg["solver"] = solver_name
    cfg["solver_options"] = dict(cfg.get("solver_options") or {})
    return cfg


def normalize_jcc_linear_solver_configs(
    *,
    solver_configs=None,
    solver_config_overrides=None,
):
    canonical = {
        "mixed_integer": _normalize_jcc_linear_solver_group(
            "GUROBI",
            {"solver_options": {}, "verbose": False, "M": 1000.0},
            None,
        ),
        "cvar": _normalize_jcc_linear_solver_group(
            "MOSEK",
            {"solver_options": {}, "verbose": False},
            None,
        ),
        "scenario": _normalize_jcc_linear_solver_group(
            "MOSEK",
            {"solver_options": {}, "verbose": False},
            None,
        ),
    }
    for payload in (solver_configs, solver_config_overrides):
        if not payload:
            continue
        for name, cfg in dict(payload).items():
            if name not in canonical:
                raise ValueError(f"Unsupported JCC linear solver config group: {name}")
            canonical[name] = _normalize_jcc_linear_solver_group(
                canonical[name].get("solver", "MOSEK"),
                canonical[name],
                cfg,
            )
    return canonical


def normalize_qcqp_train_config(
    *,
    n_samples=10000,
    batch_size=256,
    total_iteration=10000,
    train_config=None,
    train_config_overrides=None,
):
    payload = {
        "n_samples": n_samples,
        "batch_size": batch_size,
        "total_iteration": total_iteration,
    }
    payload.update(merged({}, train_config))
    payload.update(merged({}, train_config_overrides))
    return payload


def normalize_qcqp_inn_benchmark_config(
    *,
    seed,
    n_var,
    n_qua_cons,
    n_linear_cons,
    n_samples=10000,
    batch_size=256,
    total_iteration=10000,
    runtime_device,
    runtime_dtype,
    base_inn_args,
    base_optimizer_config,
    problem_config=None,
    problem_config_overrides=None,
    model_config=None,
    model_config_overrides=None,
    optimizer_config=None,
    optimizer_config_overrides=None,
    train_config=None,
    train_config_overrides=None,
):
    problem_args = normalize_qcqp_problem_config(
        seed=seed,
        n_var=n_var,
        n_qua_cons=n_qua_cons,
        n_linear_cons=n_linear_cons,
        problem_config=problem_config,
        problem_config_overrides=problem_config_overrides,
    )
    model_args, optimizer_args = build_inn_runtime_args(
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        base_inn_args=base_inn_args,
        base_optimizer_config=base_optimizer_config,
        model_config=model_config,
        model_config_overrides=model_config_overrides,
        optimizer_config=optimizer_config,
        optimizer_config_overrides=optimizer_config_overrides,
    )
    train_args = normalize_qcqp_train_config(
        n_samples=n_samples,
        batch_size=batch_size,
        total_iteration=total_iteration,
        train_config=train_config,
        train_config_overrides=train_config_overrides,
    )
    return {
        "problem": problem_args,
        "model": model_args,
        "optimizer": optimizer_args,
        "train": train_args,
    }


def build_qcqp_inn_training_payload(problem_args, model_args, train_args, *, ensure_results_save_freq=False):
    payload = {
        **model_args,
        "n_samples": int(train_args["n_samples"]),
        "batch_size": int(train_args["batch_size"]),
        "total_iteration": int(train_args["total_iteration"]),
    }
    if ensure_results_save_freq and "resultsSaveFreq" not in payload:
        payload["resultsSaveFreq"] = int(train_args["total_iteration"]) + 1
    return {
        "problem_config": problem_args,
        "model_config": payload,
    }


def build_qcqp_ipnn_training_payload(problem_args, model_args, train_args, *, ensure_results_save_freq=False):
    payload = {
        **model_args,
        "n_samples": int(train_args["n_samples"]),
        "batch_size": int(train_args["batch_size"]),
        "total_iteration": int(train_args["total_iteration"]),
    }
    if "pre_training" in train_args:
        payload["pre_training"] = int(train_args["pre_training"])
    if ensure_results_save_freq and "resultsSaveFreq" not in payload:
        payload["resultsSaveFreq"] = int(train_args["total_iteration"]) + 1
    return {
        "problem_config": problem_args,
        "ipnn_model_config": payload,
    }


def build_qcqp_predictor_training_payload(problem_args, model_args, train_args, *, ensure_results_save_freq=False):
    payload = {
        **model_args,
        "n_samples": int(train_args["n_samples"]),
        "batch_size": int(train_args["batch_size"]),
        "total_iteration": int(train_args["total_iteration"]),
    }
    if "pre_training" in train_args:
        payload["pre_training"] = int(train_args["pre_training"])
    if ensure_results_save_freq and "resultsSaveFreq" not in payload:
        payload["resultsSaveFreq"] = int(train_args["total_iteration"]) + 1
    return {
        "problem_config": problem_args,
        "predictor_model_config": payload,
    }


def _load_or_train_mapping(
    *,
    data,
    args,
    save_dir,
    retrain,
    model_filename,
    record_filename,
    load_fn,
    save_fn,
    train_fn,
):
    save_dir = Path(save_dir)
    model_checkpoint_path = save_dir / model_filename
    record_path = save_dir / record_filename

    if (not retrain) and model_checkpoint_path.exists():
        if not record_path.exists():
            raise FileNotFoundError(
                f"Missing training record for checkpoint {model_checkpoint_path}: expected {record_path}."
            )
        model = load_fn(model_checkpoint_path, map_location=data.device)
        model = model.to(device=data.device, dtype=data.dtype)
        training_record = np.load(record_path, allow_pickle=True).item()
        return model, training_record, model_checkpoint_path, record_path

    model, training_record = train_fn(data, args, save_dir)
    model_checkpoint_path = save_fn(model, save_dir, filename=model_filename)
    record_path = save_numpy(save_dir / record_filename, training_record)
    return model, training_record, model_checkpoint_path, record_path


def load_or_train_inn_mapping(data, args, save_dir, *, retrain=False):
    return _load_or_train_mapping(
        data=data,
        args=args,
        save_dir=save_dir,
        retrain=retrain,
        model_filename="inn_mapping.pt",
        record_filename="inn_training_record.npy",
        load_fn=load_inn_mapping,
        save_fn=save_inn_mapping,
        train_fn=train_mdh_mapping,
    )


def load_or_train_decision_predictor(data, args, save_dir, *, retrain=False):
    return _load_or_train_mapping(
        data=data,
        args=args,
        save_dir=save_dir,
        retrain=retrain,
        model_filename="decision_predictor.pt",
        record_filename="decision_predictor_training_record.npy",
        load_fn=load_decision_predictor,
        save_fn=save_decision_predictor,
        train_fn=train_decision_predictor,
    )


def load_or_train_ipnn_mapping(data, args, save_dir, *, retrain=False):
    return _load_or_train_mapping(
        data=data,
        args=args,
        save_dir=save_dir,
        retrain=retrain,
        model_filename="ipnn_mapping.pt",
        record_filename="ipnn_training_record.npy",
        load_fn=load_ipnn_mapping,
        save_fn=save_ipnn_mapping,
        train_fn=train_ipnn_mapping,
    )


def training_time_from_record(training_record):
    return float(np.sum(np.asarray(training_record.get("training_time_list", []), dtype=float)))


QCQP_INN_METHOD_NAME = "INN-PGD"


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


__all__ = [
    "build_inn_runtime_args",
    "build_qcqp_inn_training_payload",
    "build_qcqp_ipnn_training_payload",
    "build_qcqp_predictor_training_payload",
    "default_qcqp_predictor_args",
    "default_qcqp_ipnn_args",
    "default_qcqp_inn_args",
    "default_qcqp_optimizer_config",
    "default_qcqp_projection_args",
    "extract_qcqp_inn_instance_metrics",
    "extract_qcqp_inn_instance_metrics_from_record",
    "load_or_train_decision_predictor",
    "load_or_train_inn_mapping",
    "load_or_train_ipnn_mapping",
    "load_qcqp_inn_record",
    "normalize_jcc_linear_problem_config",
    "normalize_jcc_linear_solver_configs",
    "normalize_qcqp_inn_benchmark_config",
    "normalize_qcqp_problem_config",
    "normalize_qcqp_train_config",
    "normalize_solver_run_config",
    "QCQP_INN_METHOD_NAME",
    "summarize_qcqp_inn_record",
    "summarize_qcqp_inn_result",
    "training_time_from_record",
]
