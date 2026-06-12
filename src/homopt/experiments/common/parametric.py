"""Shared helpers for parametric benchmark entrypoints."""

from __future__ import annotations

import numpy as np

from .artifacts import load_incremental_comparison_artifacts
from .config import merged
from .inn_training import build_inn_runtime_args


def normalize_qcqp_problem_config(
    *,
    seed,
    n_var,
    n_qua_cons,
    n_linear_cons,
    problem_config=None,
):
    return {
        "n_var": int(n_var),
        "n_qua_cons": int(n_qua_cons),
        "n_linear_cons": int(n_linear_cons),
        "seed": int(seed),
        **merged(
            {
                "obj": "quad",
                "nonconvex_ratio": 0.5,
                "x_lower": -2.0,
                "x_upper": 2.0,
                "R": 5.0,
                "constraint_convexity": False,
                "objective_convexity": False,
            },
            problem_config,
        ),
    }


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
):
    merged_problem = merged(None, problem_config)
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
    invalid = sorted({"name", "options"}.intersection(translated_payload))
    if invalid:
        raise ValueError(
            f"Unsupported solver_configs keys: {invalid}. "
            "Use solver or solver_name with solver_options."
        )
    cfg = merged(default_config, translated_payload)
    solver_name = cfg.pop("solver_name", cfg.get("solver", default_solver_name))
    cfg["solver"] = solver_name
    cfg["solver_options"] = dict(cfg.get("solver_options") or {})
    return cfg


def normalize_jcc_linear_solver_configs(
    *,
    solver_configs=None,
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
    if solver_configs:
        for name, cfg in dict(solver_configs).items():
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
):
    payload = {
        "n_samples": n_samples,
        "batch_size": batch_size,
        "total_iteration": total_iteration,
    }
    payload.update(merged({}, train_config))
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
    model_config=None,
    optimizer_config=None,
    train_config=None,
):
    problem_args = normalize_qcqp_problem_config(
        seed=seed,
        n_var=n_var,
        n_qua_cons=n_qua_cons,
        n_linear_cons=n_linear_cons,
        problem_config=problem_config,
    )
    model_args, optimizer_args = build_inn_runtime_args(
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        base_inn_args=base_inn_args,
        base_optimizer_config=base_optimizer_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
    )
    train_args = normalize_qcqp_train_config(
        n_samples=n_samples,
        batch_size=batch_size,
        total_iteration=total_iteration,
        train_config=train_config,
    )
    return {
        "problem": problem_args,
        "model": model_args,
        "optimizer": optimizer_args,
        "train": train_args,
    }


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
    "extract_qcqp_inn_instance_metrics",
    "extract_qcqp_inn_instance_metrics_from_record",
    "load_qcqp_inn_record",
    "normalize_jcc_linear_problem_config",
    "normalize_jcc_linear_solver_configs",
    "normalize_qcqp_inn_benchmark_config",
    "normalize_qcqp_problem_config",
    "normalize_qcqp_train_config",
    "QCQP_INN_METHOD_NAME",
    "summarize_qcqp_inn_record",
    "summarize_qcqp_inn_result",
]
