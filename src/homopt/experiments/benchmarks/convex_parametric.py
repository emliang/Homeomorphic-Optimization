"""Convex-parametric predictor plus post-processing benchmarks."""

from __future__ import annotations

from homopt.experiments.common.parametric_learning import ParametricLearningSpec, run_parametric_learning_benchmark
from homopt.learning import DiffProjectionRefiner, IdentityRefiner
from homopt.problems import ParametricConvexQCQP, ParametricQP, ParametricSDP, ParametricSOCP


CONVEX_PARAMETRIC_LEARNING_BASELINES = (
    "predict_only",
    "predict_diff_projection",
)


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


def _convex_parametric_metadata(problem, *, seed, problem_type, n_eq, n_ineq, matrix_dim, **kwargs):
    del kwargs
    return {
        "problem_family": "convex_parametric",
        "problem_type": str(problem_type),
        "n_var": int(problem.nvar),
        "n_eq": int(n_eq),
        "n_ineq": int(n_ineq),
        "matrix_dim": int(matrix_dim),
        "seed": int(seed),
    }


def _make_convex_parametric_refiner(baseline, projection_config):
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


CONVEX_PARAMETRIC_LEARNING_SPEC = ParametricLearningSpec(
    family="convex-parametric",
    artifact_prefix="convex_parametric_learning",
    supported_baselines=CONVEX_PARAMETRIC_LEARNING_BASELINES,
    build_problem=_build_convex_parametric_problem,
    make_refiner=_make_convex_parametric_refiner,
    metadata=_convex_parametric_metadata,
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

    return run_parametric_learning_benchmark(
        spec=CONVEX_PARAMETRIC_LEARNING_SPEC,
        seed=seed,
        n_samples=n_samples,
        predictor_type=predictor_type,
        prediction_value=prediction_value,
        baselines=baselines,
        predictor_model_config=predictor_model_config,
        predictor_train_config=predictor_train_config,
        projection_config=projection_config,
        retrain=retrain,
        output_dir=output_dir,
        device=device,
        dtype=dtype,
        problem_kwargs={
            "problem_type": problem_type,
            "n_var": n_var,
            "n_eq": n_eq,
            "n_ineq": n_ineq,
            "matrix_dim": matrix_dim,
            "problem_config": problem_config,
        },
    )


__all__ = [
    "CONVEX_PARAMETRIC_LEARNING_BASELINES",
    "CONVEX_PARAMETRIC_LEARNING_SPEC",
    "convex_parametric_learning_benchmark",
]
