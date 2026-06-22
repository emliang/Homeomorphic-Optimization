"""Homeomorphic and interior-point bisection refiners."""

from __future__ import annotations

import torch

from ..base import BaseRefiner
from ..bisection import homeomorphic_bisection, interior_point_bisection
from .common import infeasible_mask, inverse_scale_decision


class HomeomorphicProjectionRefiner(BaseRefiner):
    name = "homeomorphic_projection"

    def __init__(self, model, problem_family, *, projection_config=None, tol=1e-8):
        self.model = model
        self.problem_family = problem_family
        self.projection_config = dict(projection_config or {})
        self.tol = float(tol)

    def refine(self, problem, input_params, y_pred, **kwargs):
        del kwargs
        out = y_pred.detach().clone()
        infeasible = infeasible_mask(problem, input_params, y_pred, self.tol)
        if not bool(torch.any(infeasible)):
            return out
        input_sub = input_params[infeasible]
        y_sub = out[infeasible]
        z_infeasible = inverse_scale_decision(self.problem_family, input_sub, y_sub)
        z_feasible, _ = homeomorphic_bisection(
            self.model,
            self.problem_family,
            z_infeasible,
            input_sub,
            self.projection_config,
            eps_converge=self.tol,
        )
        with torch.inference_mode():
            u_mapped, *_ = self.model(z_feasible, input_sub)
            y_partial = self.problem_family.scale(input_sub, u_mapped)
            y_full = self.problem_family.complete_partial(input_sub, y_partial)
        out[infeasible] = y_full.to(dtype=y_pred.dtype, device=y_pred.device)
        return out


class IPNNBisectionRefiner(BaseRefiner):
    name = "ip_bisection"

    def __init__(self, model, problem_family, *, projection_config=None, tol=1e-8):
        self.model = model
        self.problem_family = problem_family
        self.projection_config = dict(projection_config or {})
        self.tol = float(tol)

    def refine(self, problem, input_params, y_pred, **kwargs):
        del kwargs
        out = y_pred.detach().clone()
        infeasible = infeasible_mask(problem, input_params, y_pred, self.tol)
        if not bool(torch.any(infeasible)):
            return out
        input_sub = input_params[infeasible]
        y_sub = out[infeasible]
        u_infeasible = inverse_scale_decision(self.problem_family, input_sub, y_sub)
        with torch.inference_mode():
            feasible_u = self.model(input_sub)
        y_proj, _, _ = interior_point_bisection(
            feasible_u,
            self.problem_family,
            u_infeasible,
            input_sub,
            self.projection_config,
            eps_converge=self.tol,
        )
        out[infeasible] = y_proj.to(dtype=y_pred.dtype, device=y_pred.device)
        return out


__all__ = ["HomeomorphicProjectionRefiner", "IPNNBisectionRefiner"]
