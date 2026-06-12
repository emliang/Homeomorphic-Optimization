"""Shared runtime helpers for parametric learning routes."""

from __future__ import annotations

from time import perf_counter

from homopt.learning import summarize_prediction_route


def sample_parametric_instances(problem, *, n_instances, seed=2025, sample_obj=False):
    return problem.sample_instance_batch(
        n_instances=int(n_instances),
        seed=int(seed),
        sample_obj=bool(sample_obj),
        device=problem.device,
        dtype=problem.dtype,
    )


def run_prediction_refinement_route(
    *,
    problem,
    input_samples,
    predictor,
    refiner,
    tol,
    objective_batch=None,
    reference_y=None,
    include_predictions=True,
):
    start = perf_counter()
    summary = summarize_prediction_route(
        problem,
        input_samples,
        predictor,
        refiner,
        tol=tol,
        include_predictions=include_predictions,
        objective_batch=objective_batch,
        reference_y=reference_y,
    )
    return summary, float(perf_counter() - start)


__all__ = ["run_prediction_refinement_route", "sample_parametric_instances"]
