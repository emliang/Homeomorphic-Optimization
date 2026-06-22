"""Tensor conversion helpers for optimizer implementations."""

from __future__ import annotations

import numpy as np
import torch


def _project_to_ball(z, p_norm):
    if p_norm == 2:
        norms = torch.norm(z, dim=-1, p=2, keepdim=True)
        safe_norms = torch.clamp(norms, min=1.0)
        return torch.where(norms > 1, z / safe_norms, z)
    if p_norm == np.inf:
        return torch.clamp(z, min=-1, max=1)
    return z


def _as_numpy_vector(x):
    return x.detach().view(-1).cpu().numpy()


def _infer_problem_dtype(problem):
    for attr_name in ("Q", "p", "A_eq", "A", "L", "U", "weights"):
        value = getattr(problem, attr_name, None)
        if isinstance(value, torch.Tensor):
            return value.dtype
    return torch.float32


def _problem_tensor_kwargs(problem, *, device=None):
    return {
        "device": getattr(problem, "device", device),
        "dtype": _infer_problem_dtype(problem),
    }


def _randn_problem_row(problem, dim, *, device=None):
    return torch.randn(1, dim, **_problem_tensor_kwargs(problem, device=device))


def _as_problem_row(problem, value, *, device=None):
    return torch.as_tensor(value, **_problem_tensor_kwargs(problem, device=device)).view(1, -1)


def _as_torch_row(value, *, device, dtype):
    return torch.as_tensor(value, dtype=dtype, device=device).view(1, -1)


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
