"""Small shared utilities for the package-owned code."""

from .imports import optional_import
from .io import ensure_dir
from .runtime import (
    cast_tensors_to_dtype,
    cast_tensors_to_float32,
    get_default_device,
    get_least_utilized_gpu,
    move_tensors_to_device,
    resolve_torch_device,
    resolve_torch_dtype,
    set_global_seed,
)

__all__ = [
    "cast_tensors_to_dtype",
    "cast_tensors_to_float32",
    "ensure_dir",
    "get_default_device",
    "get_least_utilized_gpu",
    "move_tensors_to_device",
    "optional_import",
    "resolve_torch_device",
    "resolve_torch_dtype",
    "set_global_seed",
]
