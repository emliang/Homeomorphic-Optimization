"""Shared training-loop utilities for learning modules."""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from homopt.utils import resolve_torch_device, resolve_torch_dtype


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


__all__ = [
    "build_adamw_with_step_scheduler",
    "draw_batch_indices",
    "parse_train_loop_config",
    "resolve_training_input",
]
