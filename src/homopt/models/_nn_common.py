"""Shared small NN/training helpers for predictor-style model families."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from homopt.utils import resolve_torch_device, resolve_torch_dtype


class ResidualMLPBlock(nn.Module):
    """Minimal residual block reused by small MLP predictor families."""

    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(width), int(width)),
            nn.ReLU(),
            nn.Linear(int(width), int(width)),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualMLP(nn.Module):
    """Residual MLP backbone with a shared width across blocks."""

    def __init__(self, nin, nout, nhid, nl, *, dropout=0.1):
        super().__init__()
        width = int(nhid)
        layers = [nn.Linear(int(nin), width)]
        for _ in range(int(nl)):
            layers += [ResidualMLPBlock(width), nn.Dropout(float(dropout))]
        layers.append(nn.Linear(width, int(nout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def apply_bounded_output(x, outact, *, symmetric_sigmoid=False):
    outact = None if outact is None else str(outact).strip().lower()
    if outact is None:
        return x
    if outact == "tanh":
        return torch.tanh(x)
    if outact == "sigmoid":
        x = torch.sigmoid(x)
        return 2 * x - 1 if symmetric_sigmoid else x
    raise ValueError(f"Unsupported output activation: {outact}")


def draw_batch_indices(training_sample, batch_size):
    return np.random.choice(int(training_sample), int(batch_size), replace=False)


def parse_train_loop_config(
    paras,
    *,
    default_n_samples=1024,
    default_batch_size=64,
    default_total_iteration=1000,
):
    runtime_device = resolve_torch_device(paras.get("device"), default="cpu")
    runtime_dtype = resolve_torch_dtype(paras.get("dtype"), default=torch.float32)
    training_sample = int(paras.get("n_samples", default_n_samples))
    batch_size = min(int(paras.get("batch_size", default_batch_size)), training_sample)
    total_iteration = int(paras.get("total_iteration", default_total_iteration))
    pre_training = int(paras.get("pre_training", 0))
    results_save_freq = max(1, int(paras.get("resultsSaveFreq", total_iteration)))
    return {
        "runtime_device": runtime_device,
        "runtime_dtype": runtime_dtype,
        "training_sample": training_sample,
        "batch_size": batch_size,
        "total_iteration": total_iteration,
        "pre_training": pre_training,
        "results_save_freq": results_save_freq,
    }


def resolve_training_input(
    data,
    *,
    input_tensor,
    training_sample,
    batch_size,
    sample_obj,
    seed,
    runtime_device,
    runtime_dtype,
):
    if input_tensor is None:
        sample_batch = data.sample_instance_batch(
            n_instances=int(training_sample),
            seed=int(seed),
            sample_obj=bool(sample_obj),
            device=runtime_device,
            dtype=runtime_dtype,
        )
        resolved_input = sample_batch.inputs
    else:
        resolved_input = input_tensor.to(device=runtime_device, dtype=runtime_dtype)
        training_sample = int(resolved_input.shape[0])
        batch_size = min(int(batch_size), training_sample)
    return resolved_input, int(training_sample), int(batch_size)


def build_adamw_with_step_scheduler(paras, model, *, total_iteration, default_lr=1e-4, weight_decay=1e-5):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(paras.get("lr", default_lr)),
        weight_decay=float(paras.get("weight_decay", weight_decay)),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(paras.get("lr_decay_step", total_iteration))),
        gamma=float(paras.get("lr_decay", 1.0)),
    )
    return optimizer, scheduler


def save_torch_model(model, save_dir, filename):
    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / filename
    torch.save(model, model_path)
    return model_path


def load_torch_model(path_or_dir, filename, map_location=None):
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / filename
    return torch.load(path, map_location=map_location)


__all__ = [
    "ResidualMLP",
    "ResidualMLPBlock",
    "apply_bounded_output",
    "build_adamw_with_step_scheduler",
    "draw_batch_indices",
    "load_torch_model",
    "parse_train_loop_config",
    "resolve_training_input",
    "save_torch_model",
]
