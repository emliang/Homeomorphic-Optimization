"""Stiefel-specific solver baselines."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from homopt.optim._verbose import write_header as _write_verbose_header
from homopt.optim._verbose import write_row as _write_verbose_row
from homopt.solvers.core import _require_cvxpy, _solve_cvxpy_problem


def _torch_to_numpy(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()


def _constraint_violation_summary(problem, x):
    eq_violation = float(problem.eq_constraint_x(x).abs().max().item())
    ineq = problem.constraint_x(x, clip=True, eq_cons=False)
    ineq_violation = float(ineq.max().item()) if ineq.numel() else 0.0
    return eq_violation, ineq_violation, max(eq_violation, ineq_violation)


def _maybe_print_verbose_row(
    *,
    verbose,
    printed_header,
    iteration,
    verbose_interval,
    objective,
    eq_violation,
    ineq_violation,
    learning_rate,
    dual_norm=None,
    penalty=None,
    lag_gap=None,
    iter_time=0.0,
):
    if not verbose:
        return printed_header
    verbose_interval = max(1, int(verbose_interval))
    if iteration != 0 and (iteration + 1) % verbose_interval != 0:
        return printed_header
    if not printed_header:
        _write_verbose_header(print)
        printed_header = True
    _write_verbose_row(
        print,
        outer=iteration + 1,
        objective=objective,
        eq_violation=eq_violation,
        ineq_violation=ineq_violation,
        learning_rate=learning_rate,
        dual_norm=dual_norm,
        penalty=penalty,
        lag_gap=lag_gap,
        iter_time=iter_time,
    )
    return printed_header


def _project_stiefel_tangent(problem, x, euclidean_grad):
    X = problem._matrix_view(x)
    G = problem._matrix_view(euclidean_grad)
    xtg = torch.matmul(X.transpose(1, 2), G)
    sym_xtg = 0.5 * (xtg + xtg.transpose(1, 2))
    return (G - torch.matmul(X, sym_xtg)).reshape_as(x)


class _CvxpyConvexSideProjector:
    """Projection onto the convex side constraints of a StiefelProblem."""

    def __init__(
        self,
        problem,
        *,
        solver_options=None,
        time_limit_sec=None,
        verbose=False,
        solver_priority=None,
        warm_start=True,
    ):
        self.problem = problem
        self.solver_options = dict(solver_options or {})
        self.time_limit_sec = time_limit_sec
        self.verbose = bool(verbose)
        self.solver_priority = solver_priority
        self.warm_start = bool(warm_start)
        self._build_problem()

    def _build_problem(self):
        cp_mod = _require_cvxpy()
        nvar = int(self.problem.nvar)
        decision = cp_mod.Variable(nvar)
        target = cp_mod.Parameter(nvar)
        constraints = []

        A = _torch_to_numpy(getattr(self.problem, "A", None))
        b = _torch_to_numpy(getattr(self.problem, "b", None))
        if A is not None:
            constraints.append(A @ decision <= b)

        L = _torch_to_numpy(getattr(self.problem, "L", None))
        U = _torch_to_numpy(getattr(self.problem, "U", None))
        if L is not None:
            constraints.append(decision >= L)
            constraints.append(decision <= U)

        norm_radius = getattr(self.problem, "norm_radius", None)
        if norm_radius is not None:
            constraints.append(cp_mod.norm(decision, 2) <= float(norm_radius))

        for group in getattr(self.problem, "group_norm_constraints", ()):
            indices = np.asarray(group["flat_indices"], dtype=int)
            constraints.append(cp_mod.norm(decision[indices], 2) <= float(group["budget"]))

        self._cp_mod = cp_mod
        self._decision = decision
        self._target = target
        self._problem = cp_mod.Problem(
            cp_mod.Minimize(0.5 * cp_mod.sum_squares(decision - target)),
            constraints,
        )

    def project(self, x):
        target = np.asarray(x, dtype=float).reshape(-1)
        self._target.value = target
        start = perf_counter()
        kwargs = {}
        if self.solver_priority is not None:
            kwargs["solver_priority"] = tuple(self.solver_priority)
        _solve_cvxpy_problem(
            self._problem,
            self._cp_mod,
            warm_start=self.warm_start,
            verbose=self.verbose,
            solver_options=self.solver_options,
            time_limit_sec=self.time_limit_sec,
            **kwargs,
        )
        runtime = perf_counter() - start
        if self._decision.value is None:
            raise RuntimeError(f"CVXPY projection failed with status={self._problem.status}")
        return np.asarray(self._decision.value, dtype=float).reshape(1, -1), runtime


class StiefelALMEQPGDSolver:
    """Equality-ALM baseline with CVXPY projection onto convex side constraints."""

    def __init__(self, problem):
        self.problem = problem

    def _initial_point(self, x_init=None, seed=2025):
        if x_init is None:
            if seed is not None:
                torch.manual_seed(int(seed))
            return torch.randn(
                1,
                self.problem.nvar,
                device=self.problem.device,
                dtype=self.problem.A_obj.dtype,
            )
        return torch.as_tensor(
            x_init,
            device=self.problem.device,
            dtype=self.problem.A_obj.dtype,
        ).view(1, -1)

    def _as_problem_row(self, x):
        return torch.as_tensor(
            x,
            device=self.problem.device,
            dtype=self.problem.A_obj.dtype,
        ).view(1, -1)

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
        projection_warm_start=True,
        projection_solver_priority=None,
        seed=2025,
        verbose=False,
        verbose_interval=50,
    ):
        if stepsize_rule not in {"constant", "adaptive", "diminish"}:
            raise ValueError("stepsize_rule must be one of: constant, adaptive, diminish.")
        start = perf_counter()
        projector = _CvxpyConvexSideProjector(
            self.problem,
            solver_options=projection_solver_options,
            time_limit_sec=projection_time_limit_sec,
            verbose=projection_verbose,
            solver_priority=projection_solver_priority,
            warm_start=projection_warm_start,
        )
        x = self._initial_point(x_init=x_init, seed=seed)
        projection_time_total = 0.0
        if project_initial:
            x, projection_time = self._project_tensor(projector, x)
            projection_time_total += projection_time

        dual_var = torch.zeros(
            1,
            int(getattr(self.problem, "n_eq", 0) or 0),
            device=self.problem.device,
            dtype=self.problem.A_obj.dtype,
        )
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
                    lr = max(lr * float(lr_decay), 1e-12)
                    inner_iter_time.append(float(perf_counter() - iter_start))
                    inner_projection_time_traj.append(float(projection_time))
                    continue

                x = candidate.detach()
                step_time = float(perf_counter() - iter_start)
                objective = float(self.problem.objective_x(x).item())
                eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
                objective_traj.append(objective)
                merit_traj.append(candidate_merit)
                eq_violation_traj.append(eq_violation)
                ineq_violation_traj.append(ineq_violation)
                iter_time.append(step_time)
                inner_iter_time.append(step_time)
                projection_time_traj.append(float(projection_time))
                inner_projection_time_traj.append(float(projection_time))
                if violation < best_violation or (violation <= best_violation + 1e-12 and objective < best_objective):
                    best_x = x.detach().clone()
                    best_violation = violation
                    best_objective = objective
            if status == "time_limit":
                break

            eq_residual = self.problem.eq_constraint_x(x)
            if eq_residual.numel():
                dual_var = dual_var + float(dual_learning_rate) * eq_residual
                dual_var = torch.clamp(dual_var, min=-float(max_dual), max=float(max_dual))
            penalty_coef = min(float(max_penalty), float(penalty_coef) * float(penalty_growth))
            eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
            printed_verbose_header = _maybe_print_verbose_row(
                verbose=verbose,
                printed_header=printed_verbose_header,
                iteration=outer_iter,
                verbose_interval=verbose_interval,
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


class StiefelRetractionSolver:
    """Riemannian quadratic-penalty baseline using tangent projection and QR retraction."""

    def __init__(self, problem):
        self.problem = problem
        if getattr(problem, "B_eq", None) is not None:
            raise NotImplementedError("StiefelRetractionSolver currently supports only B_eq=None.")

    def _initial_point(self, x_init=None, seed=2025):
        if x_init is None:
            if seed is not None:
                torch.manual_seed(int(seed))
            raw = torch.randn(
                1,
                self.problem.nvar,
                device=self.problem.device,
                dtype=self.problem.A_obj.dtype,
            )
        else:
            raw = torch.as_tensor(
                x_init,
                device=self.problem.device,
                dtype=self.problem.A_obj.dtype,
            ).view(1, -1)
        return self.problem.retract(raw)

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
            residual = self.problem.constraint_x(x, clip=False, eq_cons=False)
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
        penalty_coef=1.0,
        penalty_growth=1.0,
        max_penalty=1e4,
        inequality_penalty_coef=None,
        seed=2025,
        verbose=False,
        verbose_interval=50,
    ):
        if stepsize_rule not in {"constant", "adaptive", "diminish"}:
            raise ValueError("stepsize_rule must be one of: constant, adaptive, diminish.")
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
                    lr = max(lr * float(lr_decay), float(convergence_threshold))
                    inner_iter_time.append(float(perf_counter() - iter_start))
                    continue

                x = candidate.detach()
                step_time = float(perf_counter() - iter_start)
                current_merit = candidate_merit
                eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
                objective_traj.append(candidate_objective)
                merit_traj.append(candidate_merit)
                eq_violation_traj.append(eq_violation)
                ineq_violation_traj.append(ineq_violation)
                accepted_iter_time.append(step_time)
                inner_iter_time.append(step_time)
                if violation < best_violation or (violation <= best_violation + 1e-12 and candidate_objective < best_objective):
                    best_x = x.detach().clone()
                    best_violation = violation
                    best_objective = candidate_objective
            outer_time = perf_counter() - outer_start
            outer_iter_time.append(float(outer_time))
            if status in {"time_limit", "converged"}:
                break

            eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
            printed_verbose_header = _maybe_print_verbose_row(
                verbose=verbose,
                printed_header=printed_verbose_header,
                iteration=outer_iter,
                verbose_interval=verbose_interval,
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
                penalty_coef = min(float(max_penalty), float(penalty_coef) * float(penalty_growth))
                penalty_traj.append(float(penalty_coef))

        objective = float(self.problem.objective_x(best_x).item())
        eq_violation = float(self.problem.eq_constraint_x(best_x).abs().max().item())
        ineq_violation = float(self.problem.constraint_x(best_x, clip=True, eq_cons=False).max().item())
        violation = max(eq_violation, ineq_violation)
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


StiefelRetractionPenaltySolver = StiefelRetractionSolver


class StiefelRetractionALMSolver:
    """Riemannian ALM baseline for convex side inequalities on the Stiefel manifold."""

    def __init__(self, problem):
        self.problem = problem
        if getattr(problem, "B_eq", None) is not None:
            raise NotImplementedError("StiefelRetractionALMSolver currently supports only B_eq=None.")

    def _initial_point(self, x_init=None, seed=2025):
        return StiefelRetractionSolver(self.problem)._initial_point(x_init=x_init, seed=seed)

    def solve(self, x_init=None, **kwargs):
        return self.solve_result(x_init=x_init, **kwargs)["solution"]

    def _inequality_augmented_lagrangian_value(self, x, dual_var, penalty_coef):
        objective = self.problem.objective_x(x).view(-1)
        if int(getattr(self.problem, "ncon", 0)) == 0:
            return objective
        residual = self.problem.constraint_x(x, clip=False, eq_cons=False)
        shifted = torch.clamp(dual_var + float(penalty_coef) * residual, min=0.0)
        aug_terms = (shifted**2 - dual_var**2).sum(dim=-1) / (2.0 * float(penalty_coef))
        return objective + aug_terms

    def _riemannian_augmented_lagrangian_gradient_x(self, x, dual_var, penalty_coef):
        residual = self.problem.constraint_x(x, clip=False, eq_cons=False)
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
        dual_learning_rate=1.0,
        penalty_coef=10.0,
        penalty_growth=1.2,
        max_penalty=1e4,
        max_dual=1e4,
        seed=2025,
        verbose=False,
        verbose_interval=50,
    ):
        if stepsize_rule not in {"constant", "adaptive", "diminish"}:
            raise ValueError("stepsize_rule must be one of: constant, adaptive, diminish.")

        start = perf_counter()
        x = self._initial_point(x_init=x_init, seed=seed)
        dual_var = torch.zeros(
            1,
            int(getattr(self.problem, "ncon", 0) or 0),
            device=self.problem.device,
            dtype=self.problem.A_obj.dtype,
        )
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
                    lr = max(lr * float(lr_decay), 1e-12)
                    inner_iter_time.append(float(perf_counter() - iter_start))
                    continue

                x = candidate.detach()
                step_time = float(perf_counter() - iter_start)
                current_merit = candidate_merit
                objective = float(self.problem.objective_x(x).item())
                eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
                objective_traj.append(objective)
                merit_traj.append(candidate_merit)
                eq_violation_traj.append(eq_violation)
                ineq_violation_traj.append(ineq_violation)
                iter_time.append(step_time)
                inner_iter_time.append(step_time)
                if violation < best_violation or (violation <= best_violation + 1e-12 and objective < best_objective):
                    best_x = x.detach().clone()
                    best_violation = violation
                    best_objective = objective
            outer_time = perf_counter() - outer_start
            outer_iter_time.append(float(outer_time))
            lag_gap_traj.append(float(last_grad_norm))
            if status in {"time_limit", "converged"}:
                break

            residual = self.problem.constraint_x(x, clip=False, eq_cons=False)
            if residual.numel():
                dual_var = torch.clamp(
                    dual_var + float(dual_learning_rate) * float(penalty_coef) * residual,
                    min=0.0,
                    max=float(max_dual),
                )
            penalty_coef = min(float(max_penalty), float(penalty_coef) * float(penalty_growth))
            dual_norm_traj.append(float(torch.norm(dual_var).item()))
            penalty_traj.append(float(penalty_coef))
            eq_violation, ineq_violation, violation = _constraint_violation_summary(self.problem, x)
            printed_verbose_header = _maybe_print_verbose_row(
                verbose=verbose,
                printed_header=printed_verbose_header,
                iteration=outer_iter,
                verbose_interval=verbose_interval,
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


class StiefelPyomoIPOPTSolver:
    """Ground-truth NLP baseline for small Stiefel instances using Pyomo + IPOPT."""

    def __init__(self, problem):
        self.problem = problem

    def _initial_point(self, x_init=None, seed=2025):
        return StiefelRetractionSolver(self.problem)._initial_point(x_init=x_init, seed=seed)

    def _build_model(self, x_init=None, seed=2025):
        try:
            import pyomo.environ as pyo
        except ImportError as exc:
            raise RuntimeError("Pyomo is required for StiefelPyomoIPOPTSolver.") from exc

        problem = self.problem
        n_rows = int(problem.n_rows)
        n_cols = int(problem.n_cols)
        A_obj = _torch_to_numpy(problem.A_obj)
        B_eq = _torch_to_numpy(problem.B_eq)
        A_ineq = _torch_to_numpy(problem.A)
        b_ineq = _torch_to_numpy(problem.b)
        G = _torch_to_numpy(problem.G)
        h = _torch_to_numpy(problem.h)
        C = _torch_to_numpy(problem.C)
        d = _torch_to_numpy(problem.d)
        L = _torch_to_numpy(problem.L)
        U = _torch_to_numpy(problem.U)
        init = self._initial_point(x_init=x_init, seed=seed).detach().cpu().numpy().reshape(n_rows, n_cols)

        model = pyo.ConcreteModel()
        model.rows = pyo.RangeSet(0, n_rows - 1)
        model.cols = pyo.RangeSet(0, n_cols - 1)

        def _bounds(_, i, j):
            flat = int(i) * n_cols + int(j)
            lower = None if L is None else float(L[flat])
            upper = None if U is None else float(U[flat])
            return lower, upper

        model.X = pyo.Var(model.rows, model.cols, bounds=_bounds)
        for i in range(n_rows):
            for j in range(n_cols):
                model.X[i, j].set_value(float(init[i, j]))

        def _objective(model):
            return -sum(
                float(A_obj[i, k]) * model.X[i, j] * model.X[k, j]
                for i in range(n_rows)
                for k in range(n_rows)
                for j in range(n_cols)
            )

        model.objective = pyo.Objective(rule=_objective, sense=pyo.minimize)

        def _flat(model, flat_index):
            i = int(flat_index) // n_cols
            j = int(flat_index) % n_cols
            return model.X[i, j]

        model.orthogonality = pyo.ConstraintList()
        for a in range(n_cols):
            for b in range(n_cols):
                if B_eq is None:
                    lhs = sum(model.X[i, a] * model.X[i, b] for i in range(n_rows))
                else:
                    lhs = sum(
                        float(B_eq[i, k]) * model.X[i, a] * model.X[k, b]
                        for i in range(n_rows)
                        for k in range(n_rows)
                    )
                model.orthogonality.add(lhs == (1.0 if a == b else 0.0))

        if A_ineq is not None:
            model.linear_ineq = pyo.ConstraintList()
            for row in range(A_ineq.shape[0]):
                model.linear_ineq.add(
                    sum(float(A_ineq[row, flat]) * _flat(model, flat) for flat in range(problem.nvar))
                    <= float(b_ineq[row])
                )

        if G is not None:
            model.soc_ineq = pyo.ConstraintList()
            for row in range(G.shape[0]):
                lhs_terms = []
                for k in range(problem.nvar):
                    lhs_terms.append(
                        sum(float(G[row, k, flat]) * _flat(model, flat) for flat in range(problem.nvar))
                        + float(h[row, k])
                    )
                rhs = sum(float(C[row, flat]) * _flat(model, flat) for flat in range(problem.nvar)) + float(d[row])
                if np.any(np.abs(C[row]) > 0):
                    model.soc_ineq.add(rhs >= 0.0)
                elif float(d[row]) < 0:
                    raise ValueError("SOC right-hand side is a negative constant.")
                model.soc_ineq.add(sum(term * term for term in lhs_terms) <= rhs * rhs)

        return model, pyo

    def solve_result(
        self,
        x_init=None,
        seed=2025,
        solver="ipopt",
        solver_options=None,
        tee=False,
        multi_start=1,
    ):
        multi_start = max(1, int(multi_start))
        start = perf_counter()
        try:
            import pyomo.environ as pyo
        except ImportError as exc:
            raise RuntimeError("Pyomo is required for StiefelPyomoIPOPTSolver.") from exc
        opt = pyo.SolverFactory(solver)
        if not opt.available(exception_flag=False):
            return {
                "solution": None,
                "status": f"{solver}_unavailable",
                "objective": None,
                "runtime_total": float(perf_counter() - start),
                "violation": None,
                "feasible": False,
                "extras": {"solver": solver},
            }
        for key, value in dict(solver_options or {}).items():
            opt.options[key] = value

        best = None
        attempts = []
        for start_index in range(multi_start):
            start_seed = None if seed is None else int(seed) + start_index
            start_x = x_init if start_index == 0 else None
            model, pyo = self._build_model(x_init=start_x, seed=start_seed)
            result = opt.solve(model, tee=bool(tee))
            solution = np.array(
                [
                    [pyo.value(model.X[i, j]) for j in range(self.problem.n_cols)]
                    for i in range(self.problem.n_rows)
                ],
                dtype=float,
            ).reshape(1, -1)
            x_tensor = torch.as_tensor(solution, device=self.problem.device, dtype=self.problem.A_obj.dtype)
            objective = float(self.problem.objective_x(x_tensor).item())
            eq_violation = float(self.problem.eq_constraint_x(x_tensor).abs().max().item())
            ineq_violation = float(self.problem.constraint_x(x_tensor, clip=True, eq_cons=False).max().item())
            violation = max(eq_violation, ineq_violation)
            status = f"{result.solver.status}:{result.solver.termination_condition}"
            attempt = {
                "solution": solution,
                "status": status,
                "objective": objective,
                "violation": violation,
                "eq_violation": eq_violation,
                "ineq_violation": ineq_violation,
                "pyomo_status": str(result.solver.status),
                "termination_condition": str(result.solver.termination_condition),
            }
            attempts.append(attempt)
            if best is None:
                best = attempt
                continue
            best_feasible = best["violation"] <= 1e-6
            attempt_feasible = attempt["violation"] <= 1e-6
            if attempt_feasible and (not best_feasible or attempt["objective"] < best["objective"]):
                best = attempt
            elif not attempt_feasible and not best_feasible and attempt["violation"] < best["violation"]:
                best = attempt

        return {
            "solution": best["solution"],
            "status": best["status"],
            "objective": best["objective"],
            "runtime_total": float(perf_counter() - start),
            "violation": best["violation"],
            "feasible": bool(best["violation"] <= 1e-6),
            "extras": {
                "solver": solver,
                "eq_violation": best["eq_violation"],
                "ineq_violation": best["ineq_violation"],
                "pyomo_status": best["pyomo_status"],
                "termination_condition": best["termination_condition"],
                "multi_start": multi_start,
                "attempt_objectives": np.asarray([attempt["objective"] for attempt in attempts], dtype=float),
                "attempt_violations": np.asarray([attempt["violation"] for attempt in attempts], dtype=float),
            },
        }
