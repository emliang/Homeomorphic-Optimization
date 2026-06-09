"""Interior-point predictor and bisection helpers for post-processing routes."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from ._nn_common import (
    ResidualMLP,
    apply_bounded_output,
    build_adamw_with_step_scheduler,
    draw_batch_indices,
    load_torch_model,
    parse_train_loop_config,
    resolve_training_input,
    save_torch_model,
)
from homopt.utils import resolve_torch_device, resolve_torch_dtype

IPNN_MODEL_FILENAME = "ipnn_mapping.pt"


class NoiseModule(nn.Module):
    def __init__(self, fixed_margin=False, gamma=1e-3, noise_type="add"):
        super().__init__()
        gamma = max(float(gamma), 1e-8)
        log_gamma = torch.log(torch.tensor(gamma, dtype=torch.float32))
        self.log_gamma = log_gamma if fixed_margin else nn.Parameter(log_gamma)
        self.noise_type = str(noise_type).strip().lower()

    def forward(self, x):
        if not self.training:
            return x
        noise = torch.randn_like(x)
        gamma = torch.exp(self.log_gamma).to(device=x.device, dtype=x.dtype)
        if self.noise_type == "mul":
            return x * (1 + noise.detach() * gamma)
        return x + noise.detach() * gamma


class IPNN(nn.Module):
    """Residual MLP that predicts a feasible interior point in normalized coordinates."""

    def __init__(
        self,
        nin,
        nout,
        nhid,
        nl,
        *,
        fixed_margin=False,
        gamma=1e-3,
        noise_type="add",
        outact="tanh",
    ):
        super().__init__()
        self.backbone = ResidualMLP(nin, nout, nhid, nl, dropout=0.1)
        self.noise = None
        if float(gamma) > 0:
            self.noise = NoiseModule(fixed_margin=fixed_margin, gamma=gamma, noise_type=noise_type)
        self.outact = outact

    def forward(self, x):
        out = self.backbone(x)
        if self.noise is not None:
            out = self.noise(out)
        return apply_bounded_output(out, self.outact, symmetric_sigmoid=False)


def save_ipnn_mapping(model, save_dir, filename=IPNN_MODEL_FILENAME):
    return save_torch_model(model, save_dir, filename)


def load_ipnn_mapping(path_or_dir, map_location=None):
    return load_torch_model(path_or_dir, IPNN_MODEL_FILENAME, map_location=map_location)


def _inverse_scale_fixed_box(data, y):
    lower = torch.as_tensor(data.fixed_L, dtype=y.dtype, device=y.device)
    upper = torch.as_tensor(data.fixed_U, dtype=y.dtype, device=y.device)
    span = torch.clamp(upper - lower, min=1e-8)
    return 2 * (y - lower) / span - 1


def _latent_target_from_fixed_center(data, input_tensor, *, device, dtype):
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

    target_latent = _latent_target_from_fixed_center(data, input_tensor, device=runtime_device, dtype=runtime_dtype)
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
        pred_latent = model(input_batch)
        loss = mse(pred_latent, target_latent.expand_as(pred_latent))
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
        pred_latent = model(input_batch)
        pred_scaled = data.scale(input_batch, pred_latent)
        pred_full = data.complete_partial(input_batch, pred_scaled)
        ineq_vio = data.check_feasibility(input_batch, pred_full).abs()
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
                eval_latent = model(input_batch)
                eval_scaled = data.scale(input_batch, eval_latent)
                eval_full = data.complete_partial(input_batch, eval_scaled)
                eval_vio = data.check_feasibility(input_batch, eval_full).abs()
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


def ip_bisection(feasible_ip, data, z_infeasible, c_params, args, eps_converge=1e-4):
    steps = int(args.get("proj_max_steps", 30))
    eps = float(args.get("proj_eps", 1e-5))
    bis_step = min(max(float(args.get("step_size", 0.5)), 1e-6), 1.0 - 1e-6)
    if feasible_ip.ndim == 2:
        feasible_ip = feasible_ip.unsqueeze(1)
    batch_size, n_ip, n_dim = feasible_ip.shape
    z_infeasible_expand = z_infeasible.view(batch_size, 1, n_dim).expand(-1, n_ip, -1).reshape(-1, n_dim)
    feasible_ip_flat = feasible_ip.reshape(-1, n_dim)
    c_extend = c_params.view(batch_size, 1, -1).expand(-1, n_ip, -1).reshape(-1, c_params.shape[-1])
    alpha_lower = torch.zeros((batch_size * n_ip, 1), device=z_infeasible.device, dtype=z_infeasible.dtype)
    alpha_upper = torch.ones_like(alpha_lower)

    with torch.inference_mode():
        for step in range(steps):
            alpha = (1 - bis_step) * alpha_lower + bis_step * alpha_upper
            z_candidate = alpha * (z_infeasible_expand - feasible_ip_flat) + feasible_ip_flat
            x_scaled = data.scale(c_extend, z_candidate)
            x_full = data.complete_partial(c_extend, x_scaled)
            violation = data.check_feasibility(c_extend, x_full).abs()
            penalty = torch.max(violation, dim=1, keepdim=True)[0]
            feasible_mask = penalty < eps
            alpha_lower = torch.where(feasible_mask, alpha, alpha_lower)
            alpha_upper = torch.where(feasible_mask, alpha_upper, alpha)
            if (alpha_upper - alpha_lower).max() < float(eps_converge):
                break

        z_feasible = alpha_lower * (z_infeasible_expand - feasible_ip_flat) + feasible_ip_flat
        z_feasible = z_feasible.view(batch_size, n_ip, n_dim)
        dist = torch.norm(z_feasible - z_infeasible.view(batch_size, 1, n_dim), dim=-1, p=2)
        min_idx = torch.argmin(dist, dim=1).view(batch_size, 1, 1).expand(-1, 1, n_dim)
        z_near = torch.gather(z_feasible, 1, min_idx).view(batch_size, n_dim)
        feasible_ip_near = torch.gather(feasible_ip, 1, min_idx).view(batch_size, n_dim)
        x_scaled = data.scale(c_params, z_near)
        x_full = data.complete_partial(c_params, x_scaled)
    return x_full, feasible_ip_near, step + 1


__all__ = [
    "IPNN",
    "NoiseModule",
    "ip_bisection",
    "load_ipnn_mapping",
    "save_ipnn_mapping",
    "train_ipnn_mapping",
]
