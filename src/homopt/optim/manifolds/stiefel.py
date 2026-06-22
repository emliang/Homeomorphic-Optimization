"""Stiefel-specific iterative optimization algorithms."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from homopt.optim.core import (
    _as_problem_row,
    _constraint_residual,
    _constraint_violation_summary,
    _is_better_candidate,
    _problem_tensor_kwargs,
    _randn_problem_row,
    _update_penalty_coefficient,
    _validate_stepsize_rule,
)
from homopt.optim.core.verbose import write_iteration_row as _write_verbose_iteration_row
from homopt.solvers.stiefel import _CvxpyConvexSideProjector


def _project_stiefel_tangent(problem, x, euclidean_grad):
    X = problem._matrix_view(x)
    G = problem._matrix_view(euclidean_grad)
    xtg = torch.matmul(X.transpose(1, 2), G)
    sym_xtg = 0.5 * (xtg + xtg.transpose(1, 2))
    return (G - torch.matmul(X, sym_xtg)).reshape_as(x)


def _stiefel_initial_point(problem, x_init=None, seed=2025, *, retract):
    if x_init is None:
        if seed is not None:
            torch.manual_seed(int(seed))
        x = _randn_problem_row(problem, problem.nvar)
    else:
        x = _as_problem_row(problem, x_init)
    return problem.retract(x) if retract else x


def _new_dual_row(problem, width):
    return torch.zeros(1, int(width or 0), **_problem_tensor_kwargs(problem))


def _stiefel_inequality_residual(problem, x):
    residual = _constraint_residual(problem, x, equality_only=False)
    return residual[:, list(getattr(problem, "ineq_cons", ()))]


def _append_stiefel_state(
    *,
    problem,
    x,
    objective,
    merit,
    objective_traj,
    merit_traj,
    eq_violation_traj,
    ineq_violation_traj,
):
    eq_violation, ineq_violation, violation = _constraint_violation_summary(problem, x)
    objective_traj.append(objective)
    merit_traj.append(merit)
    eq_violation_traj.append(eq_violation)
    ineq_violation_traj.append(ineq_violation)
    return eq_violation, ineq_violation, violation


def _update_best_stiefel_state(x, objective, violation, best_x, best_objective, best_violation):
    if _is_better_candidate(violation, objective, best_violation, best_objective):
        return x.detach().clone(), objective, violation
    return best_x, best_objective, best_violation


class StiefelALMEQPGDOptimizer:
    """Equality-ALM baseline with CVXPY projection onto convex side constraints."""

    def __init__(self, problem):
        self.problem = problem

    def _initial_point(self, x_init=None, seed=2025):
        return _stiefel_initial_point(self.problem, x_init=x_init, seed=seed, retract=False)

    def _as_problem_row(self, x):
        return _as_problem_row(self.problem, x)

    def _augmented_lagrangian_value(self, x, dual_var, penalty_coef, proximal_coef=0.0, x_outer=None):
        objective = self.problem.objective_x(x).view(-1)
        eq = self.problem.eq_constraint_x(x)
        value = objective + (dual_var * eq).sum(dim=-1) + 0.5 * float(penalty_coef) * (eq**2).sum(dim=-1)
        if proximal_coef and x_outer is not None:
            value = value + 0.5 * float(proximal_coef) * ((x - x_outer) ** 2).sum(dim=-1)
        return value

    def _augmented_lagrangian_gradient(self, x, dual_var, penalty_coef, proximal_coef=0.0, x_outer=None):
        eq = self.problem.eq_constraint_x(x)
        eq_weights = dual_var + float(penalty_coef) * eq
        grad = self.problem.gradient_objective_x(x)
        grad = grad + self.problem._explicit_equality_weighted_gradient_x(x, eq_weights)
        if proximal_coef and x_outer is not None:
            grad = grad + float(proximal_coef) * (x - x_outer)
        return grad

    def _project_tensor(self, projector, x):
        projected, projection_time = projector.project(x.detach().cpu().numpy())
        return self._as_problem_row(projected), projection_time

    def solve(self, x_init=None, **kwargs):
        return self.solve_result(x_init=x_init, **kwargs)["solution"]

    def solve_result(
        self,
        x_init=None,
        learning_rate=1e-2,
        outer_iterations=50,
        inner_iterations=10,
        max_running_time=600,
        convergence_threshold=1e-6,
        stepsize_rule="adaptive",
        lr_decay=0.8,
        min_lr=1e-6,
        dual_learning_rate=1.0,
        penalty_coef=1.0,
        penalty_growth=1.0,
        max_penalty=1e4,
        max_dual=1e4,
        proximal_coef=0.0,
        project_initial=True,
        projection_solver_options=None,
        projection_time_limit_sec=None,
        projection_verbose=False,
        projection_solver_priority=None,
        seed=2025,
        verbose=False,
        verbose_interval=50,
    ):
        stepsize_rule = _validate_stepsize_rule(stepsize_rule)
        start = perf_counter()
        projector = _CvxpyConvexSideProjector(
            self.problem,
            solver_options=projection_solver_options,
            time_limit_sec=projection_time_limit_sec,
            verbose=projection_verbose,
            solver_priority=projection_solver_priority,
        )
        x = self._initial_point(x_init=x_init, seed=seed)
        projection_time_total = 0.0
        if project_initial:
            x, projection_time = self._project_tensor(projector, x)
            projection_time_total += projection_time

        dual_var = _new_dual_row(self.problem, getattr(self.problem, "n_eq", 0))
        objective_traj = [float(self.problem.objective_x(x).item())]
        merit_traj = [float(self._augmented_lagrangian_value(x, dual_var, penalty_coef).item())]
        eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
        eq_violation_traj = [eq_violation]
        ineq_violation_traj = [ineq_violation]
        iter_time = []
        inner_iter_time = []
        projection_time_traj = []
        inner_projection_time_traj = []
        best_x = x.detach().clone()
        best_violation = violation
        best_objective = objective_traj[-1]
        lr = float(learning_rate)
        status = "max_iterations"
        printed_verbose_header = False

        for outer_iter in range(int(outer_iterations)):
            outer_start = perf_counter()
            if perf_counter() - start >= float(max_running_time):
                status = "time_limit"
                break
            x_outer = x.detach().clone()
            for inner_iter in range(int(inner_iterations)):
                if perf_counter() - start >= float(max_running_time):
                    status = "time_limit"
                    break
                iter_start = perf_counter()
                grad = self._augmented_lagrangian_gradient(
                    x,
                    dual_var,
                    penalty_coef,
                    proximal_coef=proximal_coef,
                    x_outer=x_outer,
                )
                grad_norm = float(torch.norm(grad).item())
                if grad_norm <= float(convergence_threshold):
                    status = "inner_stationary"
                    break
                step_lr = lr / (inner_iter + 1) ** 0.5 if stepsize_rule == "diminish" else lr
                candidate, projection_time = self._project_tensor(projector, x - step_lr * grad)
                projection_time_total += projection_time
                candidate_merit = float(
                    self._augmented_lagrangian_value(
                        candidate,
                        dual_var,
                        penalty_coef,
                        proximal_coef=proximal_coef,
                        x_outer=x_outer,
                    ).item()
                )
                if stepsize_rule == "adaptive" and candidate_merit > merit_traj[-1]:
                    lr = max(lr * float(lr_decay), float(min_lr))
                    inner_iter_time.append(float(perf_counter() - iter_start))
                    inner_projection_time_traj.append(float(projection_time))
                    continue

                x = candidate.detach()
                step_time = float(perf_counter() - iter_start)
                objective = float(self.problem.objective_x(x).item())
                eq_violation, ineq_violation, violation = _append_stiefel_state(
                    problem=self.problem,
                    x=x,
                    objective=objective,
                    merit=candidate_merit,
                    objective_traj=objective_traj,
                    merit_traj=merit_traj,
                    eq_violation_traj=eq_violation_traj,
                    ineq_violation_traj=ineq_violation_traj,
                )
                iter_time.append(step_time)
                inner_iter_time.append(step_time)
                projection_time_traj.append(float(projection_time))
                inner_projection_time_traj.append(float(projection_time))
                best_x, best_objective, best_violation = _update_best_stiefel_state(
                    x,
                    objective,
                    violation,
                    best_x,
                    best_objective,
                    best_violation,
                )
            if status == "time_limit":
                break

            eq_residual = self.problem.eq_constraint_x(x)
            if eq_residual.numel():
                dual_var = dual_var + float(dual_learning_rate) * eq_residual
                dual_var = torch.clamp(dual_var, min=-float(max_dual), max=float(max_dual))
            penalty_coef = _update_penalty_coefficient(float(penalty_coef), float(penalty_growth), float(max_penalty))
            eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
            printed_verbose_header = _write_verbose_iteration_row(
                print,
                enabled=verbose,
                printed_header=printed_verbose_header,
                iteration=outer_iter,
                interval=verbose_interval,
                objective=objective_traj[-1],
                eq_violation=eq_violation,
                ineq_violation=ineq_violation,
                learning_rate=lr,
                dual_norm=float(torch.norm(dual_var).item()),
                penalty=penalty_coef,
                lag_gap=None,
                iter_time=perf_counter() - outer_start,
            )
            if violation <= float(convergence_threshold):
                status = "converged"
                break
        else:
            status = "max_iterations"

        objective = float(self.problem.objective_x(best_x).item())
        eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, best_x)
        return {
            "solution": best_x.detach().cpu().numpy(),
            "status": status,
            "objective": objective,
            "runtime_total": float(perf_counter() - start),
            "violation": violation,
            "feasible": bool(violation <= 1e-6),
            "extras": {
                "eq_violation": eq_violation,
                "ineq_violation": ineq_violation,
                "objective_traj": np.asarray(objective_traj, dtype=float),
                "merit_traj": np.asarray(merit_traj, dtype=float),
                "eq_violation_traj": np.asarray(eq_violation_traj, dtype=float),
                "ineq_violation_traj": np.asarray(ineq_violation_traj, dtype=float),
                "iter_time": np.asarray(iter_time, dtype=float),
                "inner_iter_time": np.asarray(inner_iter_time, dtype=float),
                "projection_time_traj": np.asarray(projection_time_traj, dtype=float),
                "inner_projection_time_traj": np.asarray(inner_projection_time_traj, dtype=float),
                "projection_time_total": float(projection_time_total),
                "outer_iterations": int(outer_iterations),
                "inner_iterations": int(inner_iterations),
                "penalty_coef": float(penalty_coef),
            },
        }


class StiefelRetractionOptimizer:
    """Riemannian quadratic-penalty baseline using tangent projection and QR retraction."""

    def __init__(self, problem):
        self.problem = problem
        if getattr(problem, "B_eq", None) is not None:
            raise NotImplementedError("StiefelRetractionOptimizer currently supports only B_eq=None.")

    def _initial_point(self, x_init=None, seed=2025):
        return _stiefel_initial_point(self.problem, x_init=x_init, seed=seed, retract=True)

    def solve(self, x_init=None, **kwargs):
        return self.solve_result(x_init=x_init, **kwargs)["solution"]

    def _penalized_objective_x(self, x, inequality_penalty_coef):
        objective = self.problem.objective_x(x).view(-1)
        if inequality_penalty_coef <= 0:
            return objective
        ineq = self.problem.constraint_x(x, clip=True, eq_cons=False)
        if ineq.numel() == 0:
            return objective
        return objective + 0.5 * float(inequality_penalty_coef) * (ineq**2).sum(dim=-1)

    def _riemannian_penalized_gradient_x(self, x, inequality_penalty_coef):
        euclidean_grad = self.problem.gradient_objective_x(x)
        if inequality_penalty_coef > 0:
            residual = _stiefel_inequality_residual(self.problem, x)
            if residual.numel():
                ineq_weights = float(inequality_penalty_coef) * torch.clamp(residual, min=0.0)
                euclidean_grad = euclidean_grad + self.problem._explicit_inequality_weighted_gradient_x(
                    x, ineq_weights
                )
        return _project_stiefel_tangent(self.problem, x, euclidean_grad)

    def solve_result(
        self,
        x_init=None,
        learning_rate=1e-1,
        max_iterations=None,
        outer_iterations=50,
        inner_iterations=10,
        max_running_time=None,
        convergence_threshold=1e-8,
        stepsize_rule="adaptive",
        lr_decay=0.95,
        min_lr=1e-6,
        penalty_coef=1.0,
        penalty_growth=1.0,
        max_penalty=1e4,
        inequality_penalty_coef=None,
        seed=2025,
        verbose=False,
        verbose_interval=50,
    ):
        stepsize_rule = _validate_stepsize_rule(stepsize_rule)
        if max_iterations is not None:
            outer_iterations = int(max_iterations)
            inner_iterations = 1
        if inequality_penalty_coef is not None:
            penalty_coef = float(inequality_penalty_coef)

        start = perf_counter()
        x = self._initial_point(x_init=x_init, seed=seed)
        best_x = x.detach().clone()
        objective_traj = [float(self.problem.objective_x(x).item())]
        merit_traj = [float(self._penalized_objective_x(x, penalty_coef).item())]
        eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
        eq_violation_traj = [eq_violation]
        ineq_violation_traj = [ineq_violation]
        penalty_traj = [float(penalty_coef)]
        accepted_iter_time = []
        outer_iter_time = []
        inner_iter_time = []
        best_violation = violation
        best_objective = objective_traj[-1]

        lr = float(learning_rate)
        status = "max_iterations"
        printed_verbose_header = False
        total_inner_steps = 0
        last_grad_norm = None
        for outer_iter in range(int(outer_iterations)):
            outer_start = perf_counter()
            if max_running_time is not None and perf_counter() - start >= float(max_running_time):
                status = "time_limit"
                break
            current_merit = float(self._penalized_objective_x(x, penalty_coef).item())
            for inner_iter in range(int(inner_iterations)):
                iter_start = perf_counter()
                if max_running_time is not None and perf_counter() - start >= float(max_running_time):
                    status = "time_limit"
                    break
                grad = self._riemannian_penalized_gradient_x(x, penalty_coef)
                last_grad_norm = float(torch.norm(grad).item())
                if last_grad_norm <= convergence_threshold and violation <= convergence_threshold:
                    status = "converged"
                    break

                total_inner_steps += 1
                step_lr = lr / total_inner_steps**0.5 if stepsize_rule == "diminish" else lr
                candidate = self.problem.retract(x - step_lr * grad)
                candidate_objective = float(self.problem.objective_x(candidate).item())
                candidate_merit = float(self._penalized_objective_x(candidate, penalty_coef).item())
                if stepsize_rule == "adaptive" and candidate_merit > current_merit:
                    lr = max(lr * float(lr_decay), float(min_lr))
                    inner_iter_time.append(float(perf_counter() - iter_start))
                    continue

                x = candidate.detach()
                step_time = float(perf_counter() - iter_start)
                current_merit = candidate_merit
                eq_violation, ineq_violation, violation = _append_stiefel_state(
                    problem=self.problem,
                    x=x,
                    objective=candidate_objective,
                    merit=candidate_merit,
                    objective_traj=objective_traj,
                    merit_traj=merit_traj,
                    eq_violation_traj=eq_violation_traj,
                    ineq_violation_traj=ineq_violation_traj,
                )
                accepted_iter_time.append(step_time)
                inner_iter_time.append(step_time)
                best_x, best_objective, best_violation = _update_best_stiefel_state(
                    x,
                    candidate_objective,
                    violation,
                    best_x,
                    best_objective,
                    best_violation,
                )
            outer_time = perf_counter() - outer_start
            outer_iter_time.append(float(outer_time))
            if status in {"time_limit", "converged"}:
                break

            eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
            printed_verbose_header = _write_verbose_iteration_row(
                print,
                enabled=verbose,
                printed_header=printed_verbose_header,
                iteration=outer_iter,
                interval=verbose_interval,
                objective=objective_traj[-1],
                eq_violation=eq_violation,
                ineq_violation=ineq_violation,
                learning_rate=lr,
                dual_norm=None,
                penalty=penalty_coef,
                lag_gap=last_grad_norm,
                iter_time=outer_time,
            )
            if violation <= float(convergence_threshold) and (last_grad_norm is None or last_grad_norm <= convergence_threshold):
                status = "converged"
                break
            if outer_iter + 1 < int(outer_iterations):
                penalty_coef = _update_penalty_coefficient(
                    float(penalty_coef),
                    float(penalty_growth),
                    float(max_penalty),
                )
                penalty_traj.append(float(penalty_coef))

        objective = float(self.problem.objective_x(best_x).item())
        eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, best_x)
        return {
            "solution": best_x.detach().cpu().numpy(),
            "status": status,
            "objective": objective,
            "runtime_total": float(perf_counter() - start),
            "violation": violation,
            "feasible": bool(violation <= 1e-6),
            "extras": {
                "eq_violation": eq_violation,
                "ineq_violation": ineq_violation,
                "objective_traj": np.asarray(objective_traj, dtype=float),
                "merit_traj": np.asarray(merit_traj, dtype=float),
                "eq_violation_traj": np.asarray(eq_violation_traj, dtype=float),
                "ineq_violation_traj": np.asarray(ineq_violation_traj, dtype=float),
                "outer_iter_time": np.asarray(outer_iter_time, dtype=float),
                "inner_iter_time": np.asarray(inner_iter_time, dtype=float),
                "iter_time": np.asarray(accepted_iter_time, dtype=float),
                "penalty_traj": np.asarray(penalty_traj, dtype=float),
                "outer_iterations": int(outer_iterations),
                "inner_iterations": int(inner_iterations),
                "penalty_coef": float(penalty_coef),
                "inequality_penalty_coef": float(penalty_coef),
            },
        }


class StiefelRetractionALMOptimizer:
    """Riemannian ALM baseline for convex side inequalities on the Stiefel manifold."""

    def __init__(self, problem):
        self.problem = problem
        if getattr(problem, "B_eq", None) is not None:
            raise NotImplementedError("StiefelRetractionALMOptimizer currently supports only B_eq=None.")

    def _initial_point(self, x_init=None, seed=2025):
        return _stiefel_initial_point(self.problem, x_init=x_init, seed=seed, retract=True)

    def solve(self, x_init=None, **kwargs):
        return self.solve_result(x_init=x_init, **kwargs)["solution"]

    def _inequality_augmented_lagrangian_value(self, x, dual_var, penalty_coef):
        objective = self.problem.objective_x(x).view(-1)
        if int(getattr(self.problem, "ncon", 0)) == 0:
            return objective
        residual = _stiefel_inequality_residual(self.problem, x)
        shifted = torch.clamp(dual_var + float(penalty_coef) * residual, min=0.0)
        aug_terms = (shifted**2 - dual_var**2).sum(dim=-1) / (2.0 * float(penalty_coef))
        return objective + aug_terms

    def _riemannian_augmented_lagrangian_gradient_x(self, x, dual_var, penalty_coef):
        residual = _stiefel_inequality_residual(self.problem, x)
        ineq_weights = torch.clamp(dual_var + float(penalty_coef) * residual, min=0.0)
        euclidean_grad = self.problem.gradient_objective_x(x)
        if residual.numel():
            euclidean_grad = euclidean_grad + self.problem._explicit_inequality_weighted_gradient_x(x, ineq_weights)
        return _project_stiefel_tangent(self.problem, x, euclidean_grad)

    def solve_result(
        self,
        x_init=None,
        learning_rate=1e-1,
        outer_iterations=50,
        inner_iterations=10,
        max_running_time=600,
        convergence_threshold=1e-6,
        stepsize_rule="adaptive",
        lr_decay=0.95,
        min_lr=1e-6,
        dual_learning_rate=1.0,
        penalty_coef=10.0,
        penalty_growth=1.2,
        max_penalty=1e4,
        max_dual=1e4,
        seed=2025,
        verbose=False,
        verbose_interval=50,
    ):
        stepsize_rule = _validate_stepsize_rule(stepsize_rule)

        start = perf_counter()
        x = self._initial_point(x_init=x_init, seed=seed)
        dual_var = _new_dual_row(self.problem, getattr(self.problem, "ncon", 0))
        objective_traj = [float(self.problem.objective_x(x).item())]
        merit_traj = [float(self._inequality_augmented_lagrangian_value(x, dual_var, penalty_coef).item())]
        eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
        eq_violation_traj = [eq_violation]
        ineq_violation_traj = [ineq_violation]
        dual_norm_traj = [float(torch.norm(dual_var).item())]
        penalty_traj = [float(penalty_coef)]
        iter_time = []
        inner_iter_time = []
        outer_iter_time = []
        best_x = x.detach().clone()
        best_violation = violation
        best_objective = objective_traj[-1]
        lr = float(learning_rate)
        status = "max_iterations"
        printed_verbose_header = False
        last_grad_norm = float("inf")
        lag_gap_traj = []

        for outer_iter in range(int(outer_iterations)):
            outer_start = perf_counter()
            if perf_counter() - start >= float(max_running_time):
                status = "time_limit"
                break
            current_merit = float(self._inequality_augmented_lagrangian_value(x, dual_var, penalty_coef).item())
            for inner_iter in range(int(inner_iterations)):
                if perf_counter() - start >= float(max_running_time):
                    status = "time_limit"
                    break
                iter_start = perf_counter()
                grad = self._riemannian_augmented_lagrangian_gradient_x(x, dual_var, penalty_coef)
                last_grad_norm = float(torch.norm(grad).item())
                if last_grad_norm <= float(convergence_threshold) and violation <= float(convergence_threshold):
                    status = "converged"
                    break
                step_lr = lr / (inner_iter + 1) ** 0.5 if stepsize_rule == "diminish" else lr
                candidate = self.problem.retract(x - step_lr * grad)
                candidate_merit = float(
                    self._inequality_augmented_lagrangian_value(candidate, dual_var, penalty_coef).item()
                )
                if stepsize_rule == "adaptive" and candidate_merit > current_merit:
                    lr = max(lr * float(lr_decay), float(min_lr))
                    inner_iter_time.append(float(perf_counter() - iter_start))
                    continue

                x = candidate.detach()
                step_time = float(perf_counter() - iter_start)
                current_merit = candidate_merit
                objective = float(self.problem.objective_x(x).item())
                eq_violation, ineq_violation, violation = _append_stiefel_state(
                    problem=self.problem,
                    x=x,
                    objective=objective,
                    merit=candidate_merit,
                    objective_traj=objective_traj,
                    merit_traj=merit_traj,
                    eq_violation_traj=eq_violation_traj,
                    ineq_violation_traj=ineq_violation_traj,
                )
                iter_time.append(step_time)
                inner_iter_time.append(step_time)
                best_x, best_objective, best_violation = _update_best_stiefel_state(
                    x,
                    objective,
                    violation,
                    best_x,
                    best_objective,
                    best_violation,
                )
            outer_time = perf_counter() - outer_start
            outer_iter_time.append(float(outer_time))
            lag_gap_traj.append(float(last_grad_norm))
            if status in {"time_limit", "converged"}:
                break

            residual = _stiefel_inequality_residual(self.problem, x)
            if residual.numel():
                dual_var = torch.clamp(
                    dual_var + float(dual_learning_rate) * float(penalty_coef) * residual,
                    min=0.0,
                    max=float(max_dual),
                )
            penalty_coef = _update_penalty_coefficient(float(penalty_coef), float(penalty_growth), float(max_penalty))
            dual_norm_traj.append(float(torch.norm(dual_var).item()))
            penalty_traj.append(float(penalty_coef))
            eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
            printed_verbose_header = _write_verbose_iteration_row(
                print,
                enabled=verbose,
                printed_header=printed_verbose_header,
                iteration=outer_iter,
                interval=verbose_interval,
                objective=objective_traj[-1],
                eq_violation=eq_violation,
                ineq_violation=ineq_violation,
                learning_rate=lr,
                dual_norm=float(torch.norm(dual_var).item()),
                penalty=penalty_coef,
                lag_gap=last_grad_norm,
                iter_time=outer_time,
            )
            if violation <= float(convergence_threshold) and last_grad_norm <= float(convergence_threshold):
                status = "converged"
                break
        else:
            status = "max_iterations"

        objective = float(self.problem.objective_x(best_x).item())
        eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, best_x)
        return {
            "solution": best_x.detach().cpu().numpy(),
            "status": status,
            "objective": objective,
            "runtime_total": float(perf_counter() - start),
            "violation": violation,
            "feasible": bool(violation <= 1e-6),
            "extras": {
                "eq_violation": eq_violation,
                "ineq_violation": ineq_violation,
                "objective_traj": np.asarray(objective_traj, dtype=float),
                "merit_traj": np.asarray(merit_traj, dtype=float),
                "eq_violation_traj": np.asarray(eq_violation_traj, dtype=float),
                "ineq_violation_traj": np.asarray(ineq_violation_traj, dtype=float),
                "dual_norm_traj": np.asarray(dual_norm_traj, dtype=float),
                "penalty_traj": np.asarray(penalty_traj, dtype=float),
                "first_order_lagrangian_gap_traj": np.asarray(lag_gap_traj, dtype=float),
                "final_first_order_lagrangian_gap": float(lag_gap_traj[-1]) if lag_gap_traj else None,
                "iter_time": np.asarray(iter_time, dtype=float),
                "outer_iter_time": np.asarray(outer_iter_time, dtype=float),
                "inner_iter_time": np.asarray(inner_iter_time, dtype=float),
                "outer_iterations": int(outer_iterations),
                "inner_iterations": int(inner_iterations),
                "penalty_coef": float(penalty_coef),
                "dual_norm": float(torch.norm(dual_var).item()),
            },
        }
