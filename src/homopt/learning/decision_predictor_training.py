"""Training routines for neural decision predictors."""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from homopt.models.nn import DecisionPredictorNet, save_decision_predictor
from homopt.problems import bind_problem_instance

from .training_utils import (
    build_adamw_with_step_scheduler,
    draw_batch_indices,
    parse_train_loop_config,
    resolve_training_input,
)


def _center_target_for_inputs(data, input_tensor, *, device, dtype):
    input_tensor = input_tensor.to(device=device, dtype=dtype)
    if hasattr(data, "partial_dim"):
        u_center = torch.zeros(
            input_tensor.shape[0],
            int(data.partial_dim),
            dtype=dtype,
            device=device,
        )
        y_partial = data.scale(input_tensor, u_center)
        return data.complete_partial(input_tensor, y_partial)
    center = torch.as_tensor(data.fixed_x0, dtype=dtype, device=device).view(1, -1)
    return center.expand(input_tensor.shape[0], -1)


def _is_expected_target_solver_failure(exc):
    return isinstance(exc, (ImportError, RuntimeError, ValueError))


def _solver_targets(problem_family, input_tensor, *, solver_name, solver_options, allow_fallback, fallback_target):
    from homopt.solvers import QCQPSolver, solve_exact_result

    targets = []
    success = 0
    for idx in range(int(input_tensor.shape[0])):
        fallback = fallback_target[idx]
        try:
            bound_problem = bind_problem_instance(problem_family, input_tensor[idx])
            solver = QCQPSolver(bound_problem.prob_para)
            result = solve_exact_result(
                solver,
                solve_config={
                    "solve_type": "opt",
                    "solver_name": solver_name,
                    "solver_options": solver_options,
                },
            )
            solution = result.get("solution")
            if solution is None:
                raise RuntimeError("exact solver returned no solution")
            target = torch.as_tensor(solution, dtype=input_tensor.dtype, device=input_tensor.device).view(1, -1)
            success += 1
        except Exception as exc:
            if not _is_expected_target_solver_failure(exc):
                raise
            if not allow_fallback:
                raise
            target = fallback.view(1, -1)
        targets.append(target)
    return torch.cat(targets, dim=0), success


def train_decision_predictor(data, args, save_dir, input_tensor=None):
    paras = dict(args["predictor_model_config"])
    loop = parse_train_loop_config(paras)
    runtime_device = loop["runtime_device"]
    runtime_dtype = loop["runtime_dtype"]
    training_sample = loop["training_sample"]
    batch_size = loop["batch_size"]
    total_iteration = loop["total_iteration"]
    pre_training = loop["pre_training"]
    results_save_freq = loop["results_save_freq"]
    approach = str(paras.get("approach", "supervise")).strip().lower()
    if approach not in {"supervise", "unsupervise"}:
        raise ValueError(f"Unsupported predictor training approach: {approach}")

    input_tensor, training_sample, batch_size = resolve_training_input(
        data,
        input_tensor=input_tensor,
        training_sample=training_sample,
        batch_size=batch_size,
        sample_obj=(approach == "unsupervise"),
        seed=int(paras.get("seed", 2025)),
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
    )
    model_input = input_tensor.reshape(input_tensor.shape[0], -1)
    in_dim = int(model_input.shape[-1])
    out_dim = int(getattr(data, "partial_dim", data.nvar))
    hidden_dim = max(out_dim + 1, int(paras.get("h_dim", 64)))
    num_layer = int(paras.get("num_layer", 3))
    model = DecisionPredictorNet(
        in_dim,
        out_dim,
        hidden_dim,
        num_layer,
        outact=paras.get("outact", "tanh"),
        dropout=float(paras.get("dropout", 0.1)),
    ).to(device=runtime_device, dtype=runtime_dtype)

    fallback_target = _center_target_for_inputs(
        data,
        input_tensor,
        device=runtime_device,
        dtype=runtime_dtype,
    )
    supervise_target = str(paras.get("supervise_target", "fixed_center")).strip().lower()
    solver_success_rate = None
    if supervise_target in {"fixed_center", "center"}:
        target_decision = fallback_target
    elif supervise_target in {"solver", "exact_solver"}:
        target_decision, solver_success = _solver_targets(
            data,
            input_tensor,
            solver_name=str(paras.get("solver_name", "ipopt")),
            solver_options=dict(paras.get("solver_options", {})),
            allow_fallback=bool(paras.get("allow_target_fallback", True)),
            fallback_target=fallback_target,
        )
        solver_success_rate = float(solver_success) / max(int(input_tensor.shape[0]), 1)
    else:
        raise ValueError(f"Unsupported predictor supervise_target: {supervise_target}")

    optimizer, scheduler = build_adamw_with_step_scheduler(
        paras,
        model,
        total_iteration=total_iteration,
        default_lr=1e-4,
        weight_decay=1e-5,
    )
    supervise_loss_fn = nn.HuberLoss()

    train_loss_list = []
    penalty_list = []
    objective_list = []
    supervision_loss_list = []
    training_time_list = []

    for iteration in range(pre_training):
        batch_idx = draw_batch_indices(training_sample, batch_size)
        input_batch = input_tensor[batch_idx]
        model_input_batch = model_input[batch_idx]
        pred_u = model(model_input_batch)
        pred_y_partial = data.scale(input_batch, pred_u)
        pred_y = data.complete_partial(input_batch, pred_y_partial)
        loss = supervise_loss_fn(pred_y, target_decision[batch_idx])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if iteration % 1000 == 0:
            print(f"Predictor pre-training {iteration}: loss={loss.detach().item():.6f}", end="\r")

    for iteration in range(total_iteration):
        start = time.perf_counter()
        model.train()
        batch_idx = draw_batch_indices(training_sample, batch_size)
        input_batch = input_tensor[batch_idx]
        model_input_batch = model_input[batch_idx]
        pred_u = model(model_input_batch)
        pred_y_partial = data.scale(input_batch, pred_u)
        pred_y = data.complete_partial(input_batch, pred_y_partial)

        violation = data.check_feasibility(input_batch, pred_y).abs()
        penalty_loss = violation.sum(dim=1).mean()
        objective_loss = data.objective_xy(input_batch, pred_y).view(-1).mean()
        supervision_loss = supervise_loss_fn(pred_y, target_decision[batch_idx])

        if approach == "supervise":
            train_loss = supervision_loss + float(paras.get("w_ineq", 1e-3)) * penalty_loss
        else:
            train_loss = (
                float(paras.get("w_ineq", 1e-3)) * penalty_loss
                + float(paras.get("w_obj", 1e-2)) * objective_loss
            )

        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss_list.append(float(train_loss.detach().cpu().item()))
        penalty_list.append(float(penalty_loss.detach().cpu().item()))
        objective_list.append(float(objective_loss.detach().cpu().item()))
        supervision_loss_list.append(float(supervision_loss.detach().cpu().item()))
        training_time_list.append(float(time.perf_counter() - start))

        if iteration % results_save_freq == 0:
            save_decision_predictor(model, save_dir)
            print(
                "Predictor iteration: {}/{}, loss: {:.4f}, penalty: {:.4f}, objective: {:.4f}".format(
                    iteration,
                    total_iteration,
                    train_loss_list[-1],
                    penalty_list[-1],
                    objective_list[-1],
                )
            )

    model.eval()
    training_record = {
        "approach": approach,
        "supervise_target": supervise_target,
        "train_loss_list": train_loss_list,
        "penalty_list": penalty_list,
        "objective_list": objective_list,
        "supervision_loss_list": supervision_loss_list,
        "training_time_list": training_time_list,
    }
    if solver_success_rate is not None:
        training_record["solver_target_success_rate"] = float(solver_success_rate)
    return model, training_record


__all__ = ["train_decision_predictor"]
