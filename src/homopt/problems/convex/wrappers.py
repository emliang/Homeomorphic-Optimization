"""Projection and linear-oracle wrappers for convex problems."""

import torch

from homopt.problems.base import TensorRuntimeMixin

from .core import _square


class ProjProblem(TensorRuntimeMixin):
    def __init__(self, x_init, problem):
        self.x_init = torch.as_tensor(x_init).detach().clone().view(1, -1)
        self.problem = problem
        self.nvar = problem.nvar
        self.ncon = problem.ncon
        self.n_eq = 0
        self.ineq_cons = range(self.ncon)
        self.eq_cons = None

    def objective_x(self, x):
        return torch.sum((x - self.x_init)**2, dim=-1, keepdim=True)

    def gradient_objective_x(self, x):
        return 2*(x - self.x_init)

    def constraint_x(self, x, clip=True):
        return self.problem.constraint_x(x, clip)

    def gradient_constraint_x(self, x, clip=True, dual_var=None):
        return self.problem.gradient_constraint_x(x, clip = clip, dual_var=dual_var)

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None, method="explicit"):
        del method
        residual = self.constraint_x(x, clip=False)
        weights = dual_var
        if penalty_coef is not None:
            weights = weights + penalty_coef * torch.clamp(residual, min=0)
        grad = self.gradient_objective_x(x) + self.gradient_constraint_x(x, clip=False, dual_var=weights)
        if proximal_coef is not None and x_outer is not None:
            grad = grad + proximal_coef * (x - x_outer)
        return grad

    def gradient_penalty_x(self, x):
        x_detached = x.detach().requires_grad_(True)
        violation = 0.5 * _square(self.constraint_x(x_detached, clip=True)).sum(-1)
        grad = torch.autograd.grad(violation, x_detached, create_graph=False)[0]
        return grad

###################################################################
# Linear Optimization Oracle: Same constraint, Linear Objective
###################################################################
class LinearProblem(TensorRuntimeMixin):
    def __init__(self, x_init, problem):
        self.x_init = torch.as_tensor(x_init).detach().clone().view(1, -1)
        self.problem = problem
        self.nvar = problem.nvar
        self.ncon = problem.ncon
        self.n_eq = 0
        # Store the linear objective coefficients
        self.gradient =self.problem.gradient_objective_x(self.x_init)
        self.ineq_cons = range(self.ncon)
        self.eq_cons = None
    def objective_x(self, x):
        return torch.sum(self.gradient * x, dim=-1, keepdim=True)

    def gradient_objective_x(self, x):
        return self.gradient

    def constraint_x(self, x, clip=True):
        return self.problem.constraint_x(x, clip)

    def gradient_constraint_x(self, x, clip=True, dual_var=None):
        return self.problem.gradient_constraint_x(x, clip=clip, dual_var=dual_var)

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None, method="explicit"):
        del method
        residual = self.constraint_x(x, clip=False)
        weights = dual_var
        if penalty_coef is not None:
            weights = weights + penalty_coef * torch.clamp(residual, min=0)
        grad = self.gradient_objective_x(x) + self.gradient_constraint_x(x, clip=False, dual_var=weights)
        if proximal_coef is not None and x_outer is not None:
            grad = grad + proximal_coef * (x - x_outer)
        return grad

    def gradient_penalty_x(self, x):
        x_detached = x.detach().requires_grad_(True)
        violation = 0.5 * _square(self.constraint_x(x_detached, clip=True)).sum(-1)
        grad = torch.autograd.grad(violation, x_detached, create_graph=False)[0]
        return grad

__all__ = ["LinearProblem", "ProjProblem"]
