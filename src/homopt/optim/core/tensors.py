"""Tensor conversion helpers for optimizer implementations."""

from __future__ import annotations

import numbers

import numpy as np
import torch


def _project_to_ball(z, p_norm):
    """Return a point in the unit p-norm ball by radial rescaling.

    The infinity-norm case retains coordinate clipping, which is its Euclidean
    projection.  For every finite valid p, radial normalization is sufficient to
    enforce the latent-ball invariant used by the homeomorphic optimizers.
    """

    if isinstance(p_norm, (bool, np.bool_)) or not isinstance(p_norm, numbers.Real):
        raise ValueError("p_norm must be a real number in [1, inf].")
    p_norm = float(p_norm)
    if np.isnan(p_norm) or p_norm < 1 or p_norm == -np.inf:
        raise ValueError("p_norm must be a real number in [1, inf].")
    if p_norm == np.inf:
        return torch.clamp(z, min=-1, max=1)
    norms = torch.norm(z, dim=-1, p=p_norm, keepdim=True)
    return z / torch.clamp(norms, min=1.0)


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
