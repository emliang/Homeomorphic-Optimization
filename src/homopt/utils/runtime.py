"""Runtime helpers for device, dtype, seed, and GPU selection."""

from __future__ import annotations

import os
import random
import subprocess
from typing import Optional

import numpy as np
import torch


def move_tensors_to_device(obj, device):
    """Move all tensor attributes found on an object to the requested device."""

    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if isinstance(attr, torch.Tensor):
            setattr(obj, attr_name, attr.to(device))
    obj.device = device
    return obj


def cast_tensors_to_float32(obj):
    """Cast all tensor attributes found on an object to float32."""

    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if isinstance(attr, torch.Tensor):
            setattr(obj, attr_name, attr.float())
    return obj


def cast_tensors_to_dtype(obj, dtype):
    """Cast all floating-point tensor attributes found on an object to target dtype."""

    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)
        if isinstance(attr, torch.Tensor) and attr.is_floating_point():
            setattr(obj, attr_name, attr.to(dtype=dtype))
    return obj


def resolve_torch_device(device=None, default="cpu"):
    """Resolve runtime device from explicit value, env var, or default."""

    value = device if device is not None else os.getenv("HOMOPT_DEVICE", default)
    if isinstance(value, torch.device):
        return value
    value = str(value).strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_torch_dtype(dtype=None, default=torch.float32):
    """Resolve runtime dtype from explicit value, env var, or default."""

    value = dtype if dtype is not None else os.getenv("HOMOPT_DTYPE", None)
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    key = str(value).strip().lower()
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[key]


def get_default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_least_utilized_gpu():
    try:
        gpu_stats = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        gpu_memory = [tuple(map(int, x.split(","))) for x in gpu_stats.strip().splitlines() if x.strip()]
        gpu_memory = [(total - free, idx) for idx, (free, total) in enumerate(gpu_memory)]
        return sorted(gpu_memory, key=lambda x: x[0])[0][1]
    except Exception:
        return "cpu"


def set_global_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "cast_tensors_to_dtype",
    "cast_tensors_to_float32",
    "get_default_device",
    "get_least_utilized_gpu",
    "move_tensors_to_device",
    "resolve_torch_device",
    "resolve_torch_dtype",
    "set_global_seed",
]
