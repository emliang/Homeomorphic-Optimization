"""Equality-constrained ALM wrapper around exact convex subproblem solves."""

from __future__ import annotations

import time

import torch

from homopt.optim.base import BaseOptimizer
from homopt.optim.core.constraints import _constraint_residual, _constraint_violation_split
from homopt.optim.core.penalty import (
    _dual_update_scale,
    _init_penalty_family_controls,
    _maybe_update_penalty,
)
from homopt.optim.core.result import OptimizerRunResult
from homopt.optim.core.solver_cache import _cached_exact_solver, _extract_exact_solution
from homopt.optim.core.tensors import (
    _as_numpy_vector,
    _as_problem_row,
    _problem_tensor_kwargs,
    _randn_problem_row,
)
from homopt.optim.core.verbose import write_iteration_row as _write_verbose_iteration_row
from homopt.solvers import solve_exact_result


class EqualityConstrainedALMOptimizer(BaseOptimizer):
    """Outer ALM loop for equality constraints with exact convex subproblem solves."""

    def __init__(self, problem, params):
        super().__init__(problem=problem, params=params)
        self.problem = problem
        self.device = self.problem.device
        self.max_running_time = params["max_running_time"]
        self.outer_iterations = params["outer_iterations"]
        self.convergence_threshold = params["convergence_threshold"]
        dual_dim = int(getattr(self.problem, "n_eq", 0))
        if dual_dim == 0:
            eq_cons = getattr(self.problem, "eq_cons", None)
            dual_dim = 0 if eq_cons is None else len(list(eq_cons))
        self.dual_var = torch.zeros(1, dual_dim, **_problem_tensor_kwargs(self.problem))
        _init_penalty_family_controls(self, params, proximal_coef_default=0.0)
        self.check_outer_objective_change = bool(params.get("check_outer_objective_change", True))
        self.verbose_interval = int(params.get("verbose_interval", 50))
        self.opt_type = params.get("opt_type", "ALM-EQ")

    def _eq_violation(self, x):
        residual = _constraint_residual(self.problem, x, equality_only=True)
        if residual.numel() == 0:
            return torch.zeros(1, 1, device=x.device, dtype=x.dtype)
        return residual.abs().max().view(1, -1)

    def _solve_subproblem(self, x, dual_state):
        solver = _cached_exact_solver(self, "convex")
        solve_config = {
            "solve_type": "eq_alm",
            "x_init": _as_numpy_vector(x),
            "dual_var": _as_numpy_vector(dual_state),
            "penalty_coef": self.penalty_coef,
            "use_lagrangian": self.use_lagrangian,
            "use_penalty": self.use_penalty,
            "use_proximal": self.use_proximal,
            "proximal_coef": self.proximal_coef,
        }
        result = solve_exact_result(solver, solve_config=solve_config)
        solution = _extract_exact_solution(result)
        return _as_problem_row(self.problem, solution), float(result.get("runtime_total", 0.0))

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        del seed
        if initial_point is None:
            x = _randn_problem_row(self.problem, self.problem.nvar)
        else:
            x = _as_problem_row(self.problem, initial_point)
        dual_state = self.dual_var
        objective = self.problem.objective_x(x)
        violation = self._eq_violation(x)
        decision_rows = [x.detach().clone()]
        objective_rows = [objective.detach().view(1, -1)]
        violation_rows = [violation.detach().view(1, -1)]
        per_iter_time = []
        solver_iter_time = []
        penalty_violation = violation.detach()
        previous_objective = objective.detach()
        printed_verbose_header = False
        start_time = time.perf_counter()

        for outer_iter in range(self.outer_iterations):
            if time.perf_counter() - start_time > self.max_running_time:
                break
            iter_start = time.perf_counter()
            x_next, solver_time = self._solve_subproblem(x, dual_state)
            solver_iter_time.append(solver_time)
            objective_next = self.problem.objective_x(x_next)
            violation_next = self._eq_violation(x_next)
            decision_rows.append(x_next.detach().clone())
            objective_rows.append(objective_next.detach().view(1, -1))
            violation_rows.append(violation_next.detach().view(1, -1))
            if self.use_lagrangian:
                residual = _constraint_residual(self.problem, x_next, equality_only=True)
                dual_state = dual_state + _dual_update_scale(self) * residual
            penalty_violation = _maybe_update_penalty(self, penalty_violation, violation_next)
            dual_state = torch.clamp(dual_state, min=-self.max_dual, max=self.max_dual)
            iter_time = time.perf_counter() - iter_start
            per_iter_time.append(iter_time)
            split_next = _constraint_violation_split(self.problem, x_next)
            printed_verbose_header = _write_verbose_iteration_row(
                print,
                enabled=verbose,
                printed_header=printed_verbose_header,
                iteration=outer_iter,
                interval=self.verbose_interval,
                eval_payload={
                    "objective": objective_next,
                    **split_next,
                },
                learning_rate=None,
                dual_state=dual_state,
                penalty=self.penalty_coef,
                iter_time=iter_time,
            )
            objective_change = torch.abs(objective_next - previous_objective).max()
            previous_objective = objective_next.detach()
            x = x_next
            objective_ready = (not self.check_outer_objective_change) or objective_change < self.convergence_threshold
            if violation_next < self.convergence_threshold and objective_ready:
                break

        self.last_solver_iter_time = solver_iter_time
        return OptimizerRunResult(
            final_decision=x,
            decision_trajectory=torch.cat(decision_rows, dim=0),
            objective_trajectory=torch.cat(objective_rows, dim=0).view(-1),
            violation_trajectory=torch.cat(violation_rows, dim=0).view(-1),
            per_iter_time=per_iter_time,
        )


__all__ = ["EqualityConstrainedALMOptimizer"]
