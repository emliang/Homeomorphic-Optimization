"""Solver-backed refinement policies."""

from __future__ import annotations

import torch

from homopt.problems import bind_problem_instance
from homopt.solvers import QCQPSolver, solve_exact_result

from ..base import BaseRefiner
from .common import infeasible_mask


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
        infeasible = infeasible_mask(problem, input_params, y_pred, self.tol)
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


__all__ = ["ExactSolverProjectionRefiner", "InitializedOptSolverRefiner"]
