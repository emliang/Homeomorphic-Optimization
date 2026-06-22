"""Gradient-based refinement policies."""

from __future__ import annotations

import torch

from homopt.problems import normalize_constraint_violation

from ..base import BaseRefiner
from .common import infeasible_mask


class DiffProjectionRefiner(BaseRefiner):
    name = "diff_projection"

    def __init__(self, *, steps=30, lr=1e-3, momentum=0.5, tol=1e-8):
        self.steps = int(steps)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.tol = float(tol)

    def refine(self, problem, input_params, y_pred, **kwargs):
        del kwargs
        out = y_pred.detach().clone()
        infeasible = infeasible_mask(problem, input_params, y_pred, self.tol)
        if not bool(torch.any(infeasible)):
            return out
        input_sub = input_params[infeasible]
        y_sub = out[infeasible]
        velocity = torch.zeros_like(y_sub)
        for _ in range(self.steps):
            y_var = y_sub.detach().requires_grad_(True)
            residual = normalize_constraint_violation(problem.constraint_residual_xy(input_sub, y_var, clip=False))
            if bool(torch.all(residual.max(dim=1)[0] <= self.tol)):
                y_sub = y_var.detach()
                break
            penalty = residual.pow(2).sum(dim=1).mean()
            grad = torch.autograd.grad(penalty, y_var, create_graph=False)[0]
            velocity = (1 - self.momentum) * grad + self.momentum * velocity
            y_sub = (y_var - self.lr * velocity).detach()
        out[infeasible] = y_sub
        return out


__all__ = ["DiffProjectionRefiner"]
