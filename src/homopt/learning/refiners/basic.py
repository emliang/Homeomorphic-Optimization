"""Basic refinement policies for learning routes."""

from __future__ import annotations

import torch

from homopt.problems import normalize_constraint_violation

from ..base import BaseRefiner
from ..bisection import BisectionConfig, bisect_segment


class ProjectionRefiner(BaseRefiner):
    """Refine a prediction by delegating to the problem projection hook."""

    name = "projection"

    def refine(self, problem, input_params, y, **kwargs):
        del kwargs
        return problem.project_xy(input_params, y)


class IdentityRefiner(BaseRefiner):
    """Leave predictions unchanged; useful as a predict-only baseline."""

    name = "identity"

    def refine(self, problem, input_params, y, **kwargs):
        del problem, input_params, kwargs
        return y


class RayBisectionRefiner(BaseRefiner):
    """Refine a prediction along a feasible-anchor ray using batched bisection."""

    name = "ray_bisection"

    def __init__(self, steps=24, tol=1e-6, anchor=None):
        self.steps = int(steps)
        self.tol = float(tol)
        self.anchor = anchor

    def _resolve_anchor(self, problem, input_params, y, anchor):
        del input_params
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

    def refine(self, problem, input_params, y, **kwargs):
        anchor = self._resolve_anchor(problem, input_params, y, kwargs.pop("anchor", None))
        if kwargs:
            raise TypeError(f"Unsupported RayBisectionRefiner kwargs: {sorted(kwargs)}")

        def _violation(candidate_y):
            residual = normalize_constraint_violation(
                problem.constraint_residual_xy(input_params, candidate_y, clip=False)
            )
            return residual.max(dim=1, keepdim=True)[0]

        result = bisect_segment(
            anchor=anchor,
            target=y,
            decode_to_y=lambda candidate_y: candidate_y,
            violation=_violation,
            config=BisectionConfig(
                max_steps=self.steps,
                feasibility_tol=self.tol,
                convergence_tol=0.0,
                step_fraction=0.5,
                final_alpha="lower",
            ),
        )
        return result.point


__all__ = ["IdentityRefiner", "ProjectionRefiner", "RayBisectionRefiner"]
