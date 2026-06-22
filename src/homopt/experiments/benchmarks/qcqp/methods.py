"""QCQP algorithm and route helpers."""

from __future__ import annotations

from homopt.learning import ConstantDecisionPredictor, IdentityRefiner, RayBisectionRefiner, summarize_prediction_route

from .reports import normalize_qcqp_learning_baseline_name
from .setup import build_qcqp_learning_context


def run_qcqp_learning_route(
    *,
    seed=7,
    n_var=None,
    n_qua_cons=10,
    n_linear_cons=0,
    n_samples=8,
    prediction_value=1.5,
    baseline="predict_ray_bisection",
    problem_args=None,
    problem_config=None,
    device=None,
    dtype=None,
    include_predictions=False,
    context=None,
):
    """Run the shared QCQP learning-route baseline setup and summarization."""

    if context is None:
        context = build_qcqp_learning_context(
            seed=seed,
            n_var=10 if n_var is None else n_var,
            n_qua_cons=n_qua_cons,
            n_linear_cons=n_linear_cons,
            n_samples=n_samples,
            problem_args=problem_args,
            problem_config=problem_config,
            device=device,
            dtype=dtype,
        )
    baseline = normalize_qcqp_learning_baseline_name(baseline)
    qc_problem = context["qc_problem"]
    input_samples = context["instance_batch"].inputs
    problem_args = context["problem_args"]
    if n_var is None:
        n_var = int(problem_args.get("n_var", getattr(qc_problem, "nvar")))

    problem = qc_problem
    predictor = ConstantDecisionPredictor(decision_dim=n_var, value=prediction_value)
    if baseline == "predict_only":
        refiner = IdentityRefiner()
    elif baseline == "predict_ray_bisection":
        refiner = RayBisectionRefiner(anchor=qc_problem.fixed_x0)
    else:
        raise ValueError(f"Unsupported QCQP learning baseline: {baseline}")
    summary = summarize_prediction_route(
        problem,
        input_samples,
        predictor,
        refiner,
        include_predictions=include_predictions,
    )
    return summary, problem_args

__all__ = ["run_qcqp_learning_route"]
