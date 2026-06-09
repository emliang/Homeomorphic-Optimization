"""Small reference implementations for learning-route smoke runs."""

from __future__ import annotations

import torch

from homopt.problems import normalize_constraint_violation

from .base import BasePredictor, BaseRefiner


class ConstantPredictor(BasePredictor):
    """Predict a constant candidate with the same shape as the input batch."""

    name = "constant"

    def __init__(self, value=0.0):
        self.value = float(value)

    def predict(self, x, **kwargs):
        del kwargs
        return torch.full_like(x, self.value)


class ConstantDecisionPredictor(BasePredictor):
    """Predict a constant decision vector with a fixed decision dimension."""

    name = "constant_decision"

    def __init__(self, decision_dim, value=0.0):
        self.decision_dim = int(decision_dim)
        self.value = float(value)

    def predict(self, x, **kwargs):
        del kwargs
        batch_size = int(x.shape[0])
        return torch.full((batch_size, self.decision_dim), self.value, dtype=x.dtype, device=x.device)


class ProjectionRefiner(BaseRefiner):
    """Refine a prediction by delegating to the problem projection hook."""

    name = "projection"

    def refine(self, problem, x, y, **kwargs):
        del kwargs
        return problem.project_xy(x, y)


class IdentityRefiner(BaseRefiner):
    """Leave predictions unchanged; useful as a predict-only baseline."""

    name = "identity"

    def refine(self, problem, x, y, **kwargs):
        del problem, x, kwargs
        return y


class RayBisectionRefiner(BaseRefiner):
    """Refine a prediction along a feasible-anchor ray using batched bisection."""

    name = "ray_bisection"

    def __init__(self, steps=24, tol=1e-6, anchor=None):
        self.steps = int(steps)
        self.tol = float(tol)
        self.anchor = anchor

    def _resolve_anchor(self, problem, x, y, anchor):
        base = self.anchor if anchor is None else anchor
        if base is None:
            base = getattr(problem, "fixed_x0", None)
        if base is None:
            return torch.zeros_like(y)
        base = torch.as_tensor(base, dtype=y.dtype, device=y.device)
        if base.ndim == 1:
            base = base.unsqueeze(0)
        if base.shape[0] == 1 and y.shape[0] > 1:
            base = base.expand(y.shape[0], -1)
        return base

    def refine(self, problem, x, y, **kwargs):
        anchor = self._resolve_anchor(problem, x, y, kwargs.pop("anchor", None))
        if kwargs:
            raise TypeError(f"Unsupported RayBisectionRefiner kwargs: {sorted(kwargs)}")

        raw_cons = normalize_constraint_violation(problem.constraint_residual_xy(x, y, clip=False))
        raw_max = raw_cons.max(dim=1, keepdim=True)[0]
        lo = torch.where(raw_max <= self.tol, torch.ones_like(raw_max), torch.zeros_like(raw_max))
        hi = torch.ones_like(lo)

        for _ in range(self.steps):
            mid = 0.5 * (lo + hi)
            cand = anchor + mid * (y - anchor)
            cand_cons = normalize_constraint_violation(problem.constraint_residual_xy(x, cand, clip=False))
            cand_max = cand_cons.max(dim=1, keepdim=True)[0]
            feasible = cand_max <= self.tol
            lo = torch.where(feasible, mid, lo)
            hi = torch.where(feasible, hi, mid)

        return anchor + lo * (y - anchor)
