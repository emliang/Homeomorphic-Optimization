"""Shared evaluation helpers for learning-based prediction routes."""

from __future__ import annotations

import inspect

import torch

from homopt.problems import normalize_constraint_violation


def _objective_xy(problem, input_params, y, objective_batch=None):
    objective_fn = problem.objective_xy
    signature = inspect.signature(objective_fn)
    if "objective_batch" in signature.parameters:
        return objective_fn(input_params, y, objective_batch=objective_batch)
    return objective_fn(input_params, y)


def _solution_error(prediction, reference):
    if reference is None:
        return None
    diff = prediction - reference.to(dtype=prediction.dtype, device=prediction.device)
    return {
        "mse": float(diff.pow(2).mean().item()),
        "mae": float(diff.abs().mean().item()),
        "max": float(diff.abs().max().item()),
    }


def summarize_prediction_route(
    problem,
    input_params,
    predictor,
    refiner,
    tol=1e-8,
    include_predictions=False,
    objective_batch=None,
    reference_y=None,
):
    """Evaluate a predictor and optional refiner on a shared `(input_params, y)` contract."""

    raw_y = predictor.predict(input_params)
    refined_y = refiner.refine(problem, input_params, raw_y)

    raw_obj = _objective_xy(problem, input_params, raw_y, objective_batch=objective_batch)
    refined_obj = _objective_xy(problem, input_params, refined_y, objective_batch=objective_batch)
    raw_cons = normalize_constraint_violation(problem.constraint_residual_xy(input_params, raw_y, clip=False))
    refined_cons = normalize_constraint_violation(problem.constraint_residual_xy(input_params, refined_y, clip=False))

    raw_violation = raw_cons.max(dim=1)[0]
    refined_violation = refined_cons.max(dim=1)[0]
    raw_feasibility_rate = float((raw_violation <= tol).float().mean().item())
    refined_feasibility_rate = float((refined_violation <= tol).float().mean().item())

    summary = {
        "predictor": predictor.name,
        "refiner": refiner.name,
        "raw_objective_mean": float(raw_obj.mean().item()),
        "refined_objective_mean": float(refined_obj.mean().item()),
        "raw_feasibility_rate": raw_feasibility_rate,
        "refined_feasibility_rate": refined_feasibility_rate,
        "raw_violation_mean": float(raw_violation.mean().item()),
        "refined_violation_mean": float(refined_violation.mean().item()),
        "objective_improvement_mean": float((raw_obj - refined_obj).mean().item()),
        "violation_reduction_mean": float((raw_violation - refined_violation).mean().item()),
    }
    if reference_y is not None:
        reference_y = reference_y.to(dtype=raw_y.dtype, device=raw_y.device)
        reference_obj = _objective_xy(problem, input_params, reference_y, objective_batch=objective_batch)
        denom = torch.clamp(reference_obj.abs(), min=1e-12)
        raw_error = _solution_error(raw_y, reference_y)
        refined_error = _solution_error(refined_y, reference_y)
        summary.update(
            {
                "reference_objective_mean": float(reference_obj.mean().item()),
                "raw_objective_gap_mean": float(((raw_obj - reference_obj).abs() / denom).mean().item()),
                "refined_objective_gap_mean": float(((refined_obj - reference_obj).abs() / denom).mean().item()),
                "raw_solution_mse": raw_error["mse"],
                "raw_solution_mae": raw_error["mae"],
                "raw_solution_max_error": raw_error["max"],
                "refined_solution_mse": refined_error["mse"],
                "refined_solution_mae": refined_error["mae"],
                "refined_solution_max_error": refined_error["max"],
            }
        )
    if include_predictions:
        summary["raw_predictions"] = raw_y.detach().cpu().numpy()
        summary["refined_predictions"] = refined_y.detach().cpu().numpy()
    return summary
