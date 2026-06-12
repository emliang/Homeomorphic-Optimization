"""Pure tensor math helpers for gauge maps."""

from __future__ import annotations

import torch


def soft_maximum(x, dim=None, temperature=1e-4, keepdim=False):
    return temperature * torch.logsumexp(x / temperature, dim=dim, keepdim=keepdim)


def _safe_denominator(denominator, eps=1e-12):
    sign = torch.sign(denominator)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return torch.where(denominator.abs() < eps, sign * eps, denominator)


def _safe_divide(numerator, denominator, eps=1e-12):
    return numerator / _safe_denominator(denominator, eps=eps)


def _pairwise_max(lhs, rhs):
    if hasattr(torch, "maximum"):
        return torch.maximum(lhs, rhs)
    return torch.max(torch.stack([lhs, rhs], dim=0), dim=0)[0]
