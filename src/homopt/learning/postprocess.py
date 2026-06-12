"""Post-processing refiners for learning-route baselines."""

from __future__ import annotations

import torch

from homopt.problems import bind_problem_instance, normalize_constraint_violation
from homopt.solvers import QCQPSolver, solve_exact_result

from .base import BaseRefiner
from .bisection_adapters import homeomorphic_bisection, interior_point_bisection


def _max_violation(problem, input_params, y):
    residual = normalize_constraint_violation(problem.constraint_residual_xy(input_params, y, clip=False))
    return residual.max(dim=1, keepdim=True)[0]


def _infeasible_mask(problem, input_params, y, tol):
    return (_max_violation(problem, input_params, y) > float(tol)).view(-1)


def _inverse_scale_fixed_box(problem_family, y):
    lower = torch.as_tensor(problem_family.fixed_L, dtype=y.dtype, device=y.device)
    upper = torch.as_tensor(problem_family.fixed_U, dtype=y.dtype, device=y.device)
    span = torch.clamp(upper - lower, min=1e-8)
    return 2 * (y - lower) / span - 1


class ExactSolverProjectionRefiner(BaseRefiner):
    name = "exact_projection"
    solve_type = "proj"

    def __init__(self, problem_family, *, solver_factory=QCQPSolver, solve_exact=solve_exact_result, solve_config=None, tol=1e-8):
        self.problem_family = problem_family
        self.solver_factory = solver_factory
        self.solve_exact = solve_exact
        self.solve_config = dict(solve_config or {})
        self.tol = float(tol)

    def refine(self, problem, input_params, y_pred, **kwargs):
        del kwargs
        out = y_pred.detach().clone()
        infeasible = _infeasible_mask(problem, input_params, y_pred, self.tol)
        if not bool(torch.any(infeasible)):
            return out
        for idx in torch.nonzero(infeasible, as_tuple=False).view(-1).tolist():
            bound_problem = bind_problem_instance(self.problem_family, input_params[idx])
            solver = self.solver_factory(bound_problem.prob_para)
            result = self.solve_exact(
                solver,
                solve_config={
                    "solve_type": self.solve_type,
                    "x_init": y_pred[idx].detach().cpu().numpy(),
                    **self.solve_config,
                },
            )
            solution = result.get("solution")
            if solution is not None:
                out[idx] = torch.as_tensor(solution, dtype=y_pred.dtype, device=y_pred.device)
        return out


class InitializedOptSolverRefiner(ExactSolverProjectionRefiner):
    name = "initialized_opt_solver"
    solve_type = "initialized_opt"


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
        infeasible = _infeasible_mask(problem, input_params, y_pred, self.tol)
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
        infeasible = _infeasible_mask(problem, input_params, y_pred, self.tol)
        if not bool(torch.any(infeasible)):
            return out
        input_sub = input_params[infeasible]
        y_sub = out[infeasible]
        z_infeasible = _inverse_scale_fixed_box(self.problem_family, y_sub)
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
        infeasible = _infeasible_mask(problem, input_params, y_pred, self.tol)
        if not bool(torch.any(infeasible)):
            return out
        input_sub = input_params[infeasible]
        y_sub = out[infeasible]
        u_infeasible = _inverse_scale_fixed_box(self.problem_family, y_sub)
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


__all__ = [
    "DiffProjectionRefiner",
    "ExactSolverProjectionRefiner",
    "HomeomorphicProjectionRefiner",
    "IPNNBisectionRefiner",
    "InitializedOptSolverRefiner",
]
