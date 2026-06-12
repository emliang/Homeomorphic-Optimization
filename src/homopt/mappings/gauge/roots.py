"""Root solving utilities for radial gauge candidates."""

from __future__ import annotations

import torch

from .math import _pairwise_max, _safe_denominator


def _max_real_quadratic_root(A, B, C, eps=1e-12):
    """Return the largest nonnegative root of C * beta^2 + B * beta + A = 0."""
    discriminant = B * B - 4 * A * C
    valid_quad = discriminant >= 0
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0))
    root_linear = -A / _safe_denominator(B, eps=eps)
    root_1 = (-B + sqrt_discriminant) / _safe_denominator(2 * C, eps=eps)
    root_2 = (-B - sqrt_discriminant) / _safe_denominator(2 * C, eps=eps)
    root_1 = torch.where((root_1 > eps) & torch.isfinite(root_1), root_1, torch.zeros_like(root_1))
    root_2 = torch.where((root_2 > eps) & torch.isfinite(root_2), root_2, torch.zeros_like(root_2))
    root_nonlinear = _pairwise_max(root_1, root_2)
    root_linear = torch.where(
        (root_linear > eps) & torch.isfinite(root_linear),
        root_linear,
        torch.zeros_like(root_linear),
    )
    roots = torch.where(C.abs() < eps, root_linear, root_nonlinear)
    return torch.where(valid_quad, roots, torch.zeros_like(discriminant))


__all__ = ["_max_real_quadratic_root"]
