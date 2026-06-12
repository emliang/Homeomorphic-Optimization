"""Runtime and scalar conversion helpers for experiments."""

from __future__ import annotations

import numpy as np
import torch

from homopt.utils import resolve_torch_device, resolve_torch_dtype


def final_scalar(value):
    array = np.asarray(value).reshape(-1)
    return float(array[-1])


def final_or_nan(value):
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return float("nan")
    return float(array[-1])


def trajectory_dim(traj, fallback):
    if torch.is_tensor(traj) and traj.ndim == 2 and traj.shape[1] > 0:
        return int(traj.shape[1])
    return int(fallback)


def resolve_runtime(device=None, dtype=None, *, default_device="cpu", default_dtype=torch.float32):
    runtime_device = resolve_torch_device(device, default=default_device)
    runtime_dtype = resolve_torch_dtype(dtype, default=default_dtype)
    return runtime_device, runtime_dtype


def as_builtin(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return value.item() if hasattr(value, "item") else value


def ensure_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


__all__ = [
    "as_builtin",
    "ensure_int_list",
    "final_or_nan",
    "final_scalar",
    "resolve_runtime",
    "trajectory_dim",
]
