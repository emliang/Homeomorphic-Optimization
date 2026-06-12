"""Training routines for interior-point predictor networks."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from homopt.models.ipnn import IPNN, NoiseModule, save_ipnn_mapping

from .training_utils import (
    build_adamw_with_step_scheduler,
    draw_batch_indices,
    parse_train_loop_config,
    resolve_training_input,
)


def _normalized_target_from_fixed_center(data, input_tensor, *, device, dtype):
    del input_tensor
    center = torch.as_tensor(data.fixed_x0, dtype=dtype, device=device).view(1, -1)
    lower = torch.as_tensor(data.fixed_L, dtype=dtype, device=device).view(1, -1)
    upper = torch.as_tensor(data.fixed_U, dtype=dtype, device=device).view(1, -1)
    span = torch.clamp(upper - lower, min=1e-8)
    return 2 * (center - lower) / span - 1


def train_ipnn_mapping(data, args, save_dir, input_tensor=None):
    paras = dict(args["ipnn_model_config"])
    loop = parse_train_loop_config(paras)
    runtime_device = loop["runtime_device"]
    runtime_dtype = loop["runtime_dtype"]
    training_sample = loop["training_sample"]
    batch_size = loop["batch_size"]
    total_iteration = loop["total_iteration"]
    pre_training = loop["pre_training"]
    results_save_freq = loop["results_save_freq"]
    fixed_margin = bool(paras.get("fixed_margin", False))
    gamma = float(paras.get("gamma", 1e-3))

    input_tensor, training_sample, batch_size = resolve_training_input(
        data,
        input_tensor=input_tensor,
        training_sample=training_sample,
        batch_size=batch_size,
        sample_obj=False,
        seed=int(paras.get("seed", 2025)),
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
    )

    in_dim = int(input_tensor.shape[-1])
    out_dim = int(data.nvar)
    hidden_dim = max(out_dim + 1, int(paras.get("h_dim", 64)))
    num_layer = int(paras.get("num_layer", 3))
    model = IPNN(
        in_dim,
        out_dim,
        hidden_dim,
        num_layer,
        fixed_margin=fixed_margin,
        gamma=gamma,
        noise_type=str(paras.get("noise_type", "add")),
        outact=str(paras.get("outact", "tanh")),
    ).to(device=runtime_device, dtype=runtime_dtype)

    target_u = _normalized_target_from_fixed_center(data, input_tensor, device=runtime_device, dtype=runtime_dtype)
    optimizer, scheduler = build_adamw_with_step_scheduler(
        paras,
        model,
        total_iteration=total_iteration,
        default_lr=1e-4,
        weight_decay=1e-5,
    )
    mse = nn.MSELoss()

    for module in model.modules():
        if isinstance(module, NoiseModule) and hasattr(module, "log_gamma"):
            module.log_gamma.requires_grad = not fixed_margin

    penalty_list = []
    gamma_list = []
    valid_rate_list = []
    training_time_list = []

    for iteration in range(pre_training):
        batch_idx = draw_batch_indices(training_sample, batch_size)
        input_batch = input_tensor[batch_idx]
        pred_u = model(input_batch)
        loss = mse(pred_u, target_u.expand_as(pred_u))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if iteration % 1000 == 0:
            print(f"IPNN pre-training {iteration}: mse={loss.detach().item():.6f}", end="\r")

    for iteration in range(total_iteration):
        start = time.perf_counter()
        model.train()
        batch_idx = draw_batch_indices(training_sample, batch_size)
        input_batch = input_tensor[batch_idx]
        pred_u = model(input_batch)
        pred_y_partial = data.scale(input_batch, pred_u)
        pred_y = data.complete_partial(input_batch, pred_y_partial)
        ineq_vio = data.check_feasibility(input_batch, pred_y).abs()
        penalty_loss = ineq_vio.mean()
        reg_loss = torch.zeros((), device=runtime_device, dtype=runtime_dtype)
        for module in model.modules():
            if isinstance(module, NoiseModule) and hasattr(module, "log_gamma"):
                reg_loss = reg_loss + module.log_gamma.to(device=runtime_device, dtype=runtime_dtype) / np.log(10)
        loss = penalty_loss if fixed_margin else penalty_loss - 1e-2 * reg_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        penalty_list.append(float(penalty_loss.detach().cpu().item()))
        gamma_list.append(float(reg_loss.detach().cpu().item()))
        training_time_list.append(float(time.perf_counter() - start))

        if iteration % results_save_freq == 0:
            model.eval()
            with torch.inference_mode():
                eval_u = model(input_batch)
                eval_y_partial = data.scale(input_batch, eval_u)
                eval_y = data.complete_partial(input_batch, eval_y_partial)
                eval_vio = data.check_feasibility(input_batch, eval_y).abs()
                valid_rate = float((eval_vio.max(dim=1)[0] <= 1e-5).float().mean().cpu().item())
                valid_rate_list.append(valid_rate)
            save_ipnn_mapping(model, save_dir)
            print(
                "IPNN iteration: {}/{}, penalty: {:.4f}, feasibility: {:.4f}, gamma: {:.4f}".format(
                    iteration,
                    total_iteration,
                    penalty_list[-1],
                    valid_rate_list[-1],
                    gamma_list[-1],
                )
            )

    model.eval()
    training_record = {
        "penalty_list": penalty_list,
        "gamma_list": gamma_list,
        "valid_rate_list": valid_rate_list,
        "training_time_list": training_time_list,
    }
    return model, training_record


__all__ = ["train_ipnn_mapping"]
