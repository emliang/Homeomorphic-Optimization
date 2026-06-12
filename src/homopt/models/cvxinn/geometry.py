"""Compactification maps used by CvxINN wrappers."""

import torch

def radial_compactify(x, eps=1e-12):
    norm = torch.norm(x, dim=1, p=2, keepdim=True)
    safe_norm = norm.clamp_min(eps)
    scale = norm / (1.0 + norm)
    return torch.where(norm > eps, x / safe_norm * scale, torch.zeros_like(x))


def radial_decompactify(z, eps=1e-12):
    norm = torch.norm(z, dim=1, p=2, keepdim=True)
    safe_norm = norm.clamp_min(eps)
    clipped_norm = norm.clamp_max(1.0 - eps)
    scale = clipped_norm / (1.0 - clipped_norm)
    return torch.where(norm > eps, z / safe_norm * scale, torch.zeros_like(z))


def cube_compactify(x):
    return torch.tanh(x)


def cube_decompactify(z, eps=1e-12):
    clipped = torch.clamp(z, min=-1.0 + float(eps), max=1.0 - float(eps))
    return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))

__all__ = [
    "cube_compactify",
    "cube_decompactify",
    "radial_compactify",
    "radial_decompactify",
]
