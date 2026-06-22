"""Shared tensor helpers for visualization-time problem evaluation."""

from __future__ import annotations

import torch


def problem_tensor_kwargs(problem, *, preferred_attrs=("fixed_Q", "Q", "p", "L", "A_obj")):
    """Infer torch device/dtype from common tensor attributes on a problem."""

    for attr_name in preferred_attrs:
        tensor = getattr(problem, attr_name, None)
        if torch.is_tensor(tensor):
            return {"device": tensor.device, "dtype": tensor.dtype}
    return {
        "device": getattr(problem, "device", torch.device("cpu")),
        "dtype": getattr(problem, "dtype", torch.float32),
    }


def expand_input_params(input_params, n_points, *, device, dtype):
    """Broadcast one parametric input row to a dense grid batch."""

    if input_params is None:
        return None
    if not torch.is_tensor(input_params):
        input_params = torch.as_tensor(input_params, device=device, dtype=dtype)
    else:
        input_params = input_params.to(device=device, dtype=dtype)
    if input_params.ndim == 1:
        input_params = input_params.view(1, -1)
    first = input_params[:1]
    return first.expand(n_points, *first.shape[1:])


__all__ = [
    "expand_input_params",
    "problem_tensor_kwargs",
]
