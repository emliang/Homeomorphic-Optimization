"""QCQP problem and instance-context construction."""

from __future__ import annotations

from homopt.experiments.common.config import merged
from homopt.learning.training.inn import build_inn_runtime_args
from homopt.problems import NonConvexQCProblem
from homopt.utils import set_global_seed


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


def build_qcqp_problem_args(
    *,
    seed,
    n_var,
    n_qua_cons,
    n_linear_cons,
    problem_config=None,
):
    return normalize_qcqp_problem_config(
        seed=seed,
        n_var=n_var,
        n_qua_cons=n_qua_cons,
        n_linear_cons=n_linear_cons,
        problem_config=problem_config,
    )


def build_qcqp_problem_context(
    *,
    seed=7,
    n_var=10,
    n_qua_cons=10,
    n_linear_cons=0,
    n_instances=None,
    problem_args=None,
    problem_config=None,
    device=None,
    dtype=None,
    sample_obj=False,
):
    """Build a shared QCQP problem, optionally with one sampled instance batch."""

    set_global_seed(seed)
    if problem_args is None:
        problem_args = build_qcqp_problem_args(
            seed=seed,
            n_var=n_var,
            n_qua_cons=n_qua_cons,
            n_linear_cons=n_linear_cons,
            problem_config=problem_config,
        )
    qc_problem = NonConvexQCProblem(problem_args)
    if device is not None:
        qc_problem = qc_problem.to_device(device)
    if dtype is not None:
        qc_problem = qc_problem.to_dtype(dtype)

    context = {
        "qc_problem": qc_problem,
        "problem_args": problem_args,
    }
    if n_instances is not None:
        context["instance_batch"] = qc_problem.sample_instance_batch(
            n_instances=int(n_instances),
            seed=seed,
            sample_obj=sample_obj,
            device=qc_problem.device,
            dtype=qc_problem.dtype,
        )
    return context


def build_qcqp_learning_context(
    *,
    seed=7,
    n_var=10,
    n_qua_cons=10,
    n_linear_cons=0,
    n_samples=8,
    problem_args=None,
    problem_config=None,
    device=None,
    dtype=None,
    sample_obj=False,
):
    """Build shared QCQP learning-route inputs once for multiple baselines."""

    return build_qcqp_problem_context(
        seed=seed,
        n_var=n_var,
        n_qua_cons=n_qua_cons,
        n_linear_cons=n_linear_cons,
        n_instances=n_samples,
        problem_args=problem_args,
        problem_config=problem_config,
        device=device,
        dtype=dtype,
        sample_obj=sample_obj,
    )

__all__ = [
    "build_qcqp_learning_context",
    "build_qcqp_problem_args",
    "build_qcqp_problem_context",
    "normalize_qcqp_inn_benchmark_config",
    "normalize_qcqp_problem_config",
    "normalize_qcqp_train_config",
]
