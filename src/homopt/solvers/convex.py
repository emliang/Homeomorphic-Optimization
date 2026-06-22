"""CVXPY exact solver for deterministic convex problem instances."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .common import _normalized_exact_solver_result
from .cvxpy import _require_cvxpy, _soc_constraint, _solve_cvxpy_problem, _sum_multiply


class ConvexSolver:
    def __init__(self, problem_params):
        self.Q = problem_params["Q"]
        self.p = problem_params["p"]
        self.A = problem_params["A"]
        self.b = problem_params["b"]
        self.A_eq = problem_params["A_eq"]
        self.b_eq = problem_params["b_eq"]
        self.Qq = problem_params["Qq"]
        self.pq = problem_params["pq"]
        self.bq = problem_params["bq"]
        self.G = problem_params["G"]
        self.h = problem_params["h"]
        self.C = problem_params["C"]
        self.d = problem_params["d"]
        self.L = problem_params["L"]
        self.U = problem_params["U"]
        self.n_soc = self.C.shape[0] if self.C is not None else 0
        self.n_lin = self.A.shape[0] if self.A is not None else 0
        self.n_qua = self.Qq.shape[0] if self.Qq is not None else 0
        self.n_eq = self.A_eq.shape[0] if self.A_eq is not None else 0
        self.n_vars = len(self.Q)

    def _constraint_margin_scales(self):
        """Return static residual scales for a distance-like center margin."""

        box_abs = np.maximum(np.abs(self.L), np.abs(self.U))
        box_radius_bound = float(np.linalg.norm(box_abs, ord=2))
        scales = {
            "soc": None,
            "linear": None,
            "quadratic": None,
            "upper": np.ones(self.n_vars, dtype=float),
            "lower": np.ones(self.n_vars, dtype=float),
        }
        if self.n_soc > 0:
            scales["soc"] = np.asarray(
                [
                    np.linalg.norm(self.G[i], ord=2) + np.linalg.norm(self.C[i], ord=2)
                    for i in range(self.n_soc)
                ],
                dtype=float,
            )
        if self.n_lin > 0:
            scales["linear"] = np.linalg.norm(self.A, axis=1)
        if self.n_qua > 0:
            scales["quadratic"] = np.asarray(
                [
                    np.linalg.norm(self.Qq[i], ord=2) * box_radius_bound + np.linalg.norm(self.pq[i], ord=2)
                    for i in range(self.n_qua)
                ],
                dtype=float,
            )
        return scales

    @staticmethod
    def _margin(epsilon, margin_scales, key, index=None):
        if margin_scales is None:
            return epsilon
        epsilon_expr = epsilon if np.isscalar(epsilon) else epsilon[0]
        scale = margin_scales[key]
        if index is not None:
            scale = float(scale[index])
        return epsilon_expr * scale

    def _create_base_problem(self, epsilon=None, equality=False, margin_scales=None):
        cp_mod = _require_cvxpy()
        x = cp_mod.Variable(self.n_vars)
        constraints = []
        if epsilon is None:
            if self.n_soc > 0:
                constraints += [
                    _soc_constraint(cp_mod, self.C[i].T @ x + self.d[i], self.G[i] @ x + self.h[i])
                    for i in range(self.n_soc)
                ]
            if self.n_lin > 0:
                constraints += [self.A @ x <= self.b]
            constraints += [x <= self.U, x >= self.L]
            if self.n_qua > 0:
                constraints += [
                    0.5 * cp_mod.quad_form(x, self.Qq[i]) + self.pq[i].T @ x <= self.bq[i]
                    for i in range(self.n_qua)
                ]
            if equality and self.n_eq > 0:
                constraints += [self.A_eq @ x == self.b_eq]
        else:
            if self.n_soc > 0:
                constraints += [
                    _soc_constraint(
                        cp_mod,
                        self.C[i].T @ x + self.d[i] - self._margin(epsilon, margin_scales, "soc", i),
                        self.G[i] @ x + self.h[i],
                    )
                    for i in range(self.n_soc)
                ]
            if self.n_lin > 0:
                constraints += [self.A @ x <= self.b - self._margin(epsilon, margin_scales, "linear")]
            constraints += [
                x <= self.U - self._margin(epsilon, margin_scales, "upper"),
                x >= self.L + self._margin(epsilon, margin_scales, "lower"),
            ]
            if not np.isscalar(epsilon):
                constraints += [epsilon >= 0]
            if self.n_qua > 0:
                constraints += [
                    0.5 * cp_mod.quad_form(x, self.Qq[i]) + self.pq[i].T @ x
                    <= self.bq[i] - self._margin(epsilon, margin_scales, "quadratic", i)
                    for i in range(self.n_qua)
                ]
            if equality and self.n_eq > 0:
                constraints += [self.A_eq @ x == self.b_eq]
        return x, constraints

    def _quadratic_objective_expr(self, cp_mod, decision):
        return 0.5 * cp_mod.quad_form(decision, self.Q) + self.p.T @ decision

    def _projection_objective_expr(self, cp_mod, decision, target):
        return cp_mod.sum_squares(decision - np.asarray(target).reshape(-1))

    def _linear_oracle_objective_expr(self, cp_mod, decision, grad):
        return _sum_multiply(cp_mod, decision, np.asarray(grad).reshape(-1))

    def _eq_alm_objective_expr(
        self,
        cp_mod,
        decision,
        *,
        x_init=None,
        dual_var=None,
        penalty_coef=None,
        proximal_coef=None,
        x_outer=None,
    ):
        objective_expr = self._quadratic_objective_expr(cp_mod, decision)
        if self.n_eq > 0:
            if dual_var is None:
                dual_var = np.zeros(self.n_eq)
            dual_var = np.asarray(dual_var, dtype=float).reshape(-1)
            if dual_var.shape[0] != self.n_eq:
                raise ValueError(f"dual_var has size {dual_var.shape[0]}, expected {self.n_eq}.")
            eq_residual = self.A_eq @ decision - self.b_eq
            objective_expr += dual_var.T @ eq_residual
            if penalty_coef is not None:
                objective_expr += 0.5 * float(penalty_coef) * cp_mod.sum_squares(eq_residual)
        if proximal_coef is not None:
            if x_outer is None:
                if x_init is None:
                    raise ValueError("x_outer or x_init required when proximal_coef is set.")
                x_outer = x_init
            objective_expr += 0.5 * float(proximal_coef) * cp_mod.sum_squares(
                decision - np.asarray(x_outer).reshape(-1)
            )
        return objective_expr

    def solve(
        self,
        solve_type="opt",
        x_init=None,
        grad=None,
        eps=1e-7,
        equality=True,
        dual_var=None,
        penalty_coef=None,
        proximal_coef=None,
        x_outer=None,
        solver_name=None,
        solver_options=None,
        time_limit_sec=None,
        verbose=False,
    ):
        cp_mod = _require_cvxpy()
        if solve_type in ["opt", "initialized_opt"]:
            decision, constraints = self._create_base_problem(equality=True)
            objective = cp_mod.Minimize(self._quadratic_objective_expr(cp_mod, decision))
            if solve_type == "initialized_opt":
                if x_init is None:
                    raise ValueError("x_init required for initialized_opt")
                decision.value = x_init
        elif solve_type == "proj":
            decision, constraints = self._create_base_problem(equality=True)
            if x_init is None:
                raise ValueError("x_init required for projection")
            decision.value = x_init
            objective = cp_mod.Minimize(self._projection_objective_expr(cp_mod, decision, x_init))
        elif solve_type == "linear":
            decision, constraints = self._create_base_problem(equality=True)
            if x_init is None or grad is None:
                raise ValueError("x_init and grad required for LOO")
            decision.value = x_init
            objective = cp_mod.Minimize(self._linear_oracle_objective_expr(cp_mod, decision, grad))
        elif solve_type == "central_ip":
            epsilon = cp_mod.Variable(1)
            decision, constraints = self._create_base_problem(epsilon, equality=bool(equality))
            objective = cp_mod.Minimize(-epsilon)
        elif solve_type == "geometric_central_ip":
            epsilon = cp_mod.Variable(1)
            decision, constraints = self._create_base_problem(
                epsilon,
                equality=bool(equality),
                margin_scales=self._constraint_margin_scales(),
            )
            objective = cp_mod.Minimize(-epsilon)
        elif solve_type == "analytical_ip":
            epsilon = cp_mod.Variable(1)
            decision, constraints = self._create_base_problem(epsilon, equality=bool(equality))
            objective = cp_mod.Minimize(-cp_mod.log(epsilon))
        elif solve_type == "ip":
            epsilon_margin = None if eps is None else max(float(eps), 0.0)
            decision, constraints = self._create_base_problem(epsilon_margin, equality=bool(equality))
            objective = cp_mod.Minimize(0)
        elif solve_type == "eq_alm":
            decision, constraints = self._create_base_problem(equality=False)
            if x_init is not None:
                decision.value = x_init
            objective_expr = self._eq_alm_objective_expr(
                cp_mod,
                decision,
                x_init=x_init,
                dual_var=dual_var,
                penalty_coef=penalty_coef,
                proximal_coef=proximal_coef,
                x_outer=x_outer,
            )
            objective = cp_mod.Minimize(objective_expr)
        else:
            raise ValueError(f"Unknown solve_type: {solve_type}")

        problem = cp_mod.Problem(objective, constraints)
        _solve_cvxpy_problem(
            problem,
            cp_mod,
            warm_start=True,
            interior_point=solve_type in ["ip", "central_ip", "geometric_central_ip", "analytical_ip"],
            verbose=verbose,
            solver_options=solver_options,
            time_limit_sec=time_limit_sec,
            preferred_solver=solver_name,
        )
        return decision.value

    def _constraint_violation(self, x):
        if x is None:
            return None
        x = np.asarray(x).reshape(-1)
        violations = []
        if self.n_soc > 0:
            for i in range(self.n_soc):
                lhs = np.linalg.norm(np.dot(self.G[i], x) + self.h[i], 2)
                rhs = float(np.dot(self.C[i], x) + self.d[i])
                violations.append(max(lhs - rhs, 0.0))
        if self.n_lin > 0:
            violations.append(float(np.max(np.maximum(np.dot(self.A, x) - self.b, 0.0))))
        if self.n_qua > 0:
            quad = []
            for i in range(self.n_qua):
                value = 0.5 * np.dot(x, np.dot(self.Qq[i], x)) + np.dot(self.pq[i], x) - self.bq[i]
                quad.append(max(float(value), 0.0))
            violations.append(float(np.max(quad)))
        if self.n_eq > 0:
            violations.append(float(np.max(np.abs(np.dot(self.A_eq, x) - self.b_eq))))
        violations.append(float(np.max(np.maximum(x - self.U, 0.0))))
        violations.append(float(np.max(np.maximum(self.L - x, 0.0))))
        return float(max(violations)) if violations else 0.0

    def _objective_value(self, x):
        if x is None:
            return None
        x = np.asarray(x).reshape(-1)
        return float(0.5 * np.dot(x, np.dot(self.Q, x)) + np.dot(self.p, x))

    def solve_result(
        self,
        solve_type="opt",
        x_init=None,
        grad=None,
        eps=1e-7,
        equality=True,
        dual_var=None,
        penalty_coef=None,
        proximal_coef=None,
        x_outer=None,
        solver_name=None,
        solver_options=None,
        time_limit_sec=None,
        verbose=False,
    ):
        start = perf_counter()
        hard_equalities = bool(equality) and solve_type != "eq_alm"
        try:
            solution = self.solve(
                solve_type=solve_type,
                x_init=x_init,
                grad=grad,
                eps=eps,
                equality=equality,
                dual_var=dual_var,
                penalty_coef=penalty_coef,
                proximal_coef=proximal_coef,
                x_outer=x_outer,
                solver_name=solver_name,
                solver_options=solver_options,
                time_limit_sec=time_limit_sec,
                verbose=verbose,
            )
            runtime_total = perf_counter() - start
            violation = self._constraint_violation(solution)
            return _normalized_exact_solver_result(
                solution=solution,
                status="optimal" if solution is not None else "failed",
                objective=self._objective_value(solution) if solve_type in ["opt", "initialized_opt", "eq_alm"] else None,
                runtime_total=runtime_total,
                violation=violation,
                feasible=None if violation is None else violation <= 1e-6,
                extras={"solve_type": solve_type, "equality": hard_equalities, "hard_equalities": hard_equalities},
            )
        except Exception as exc:
            return _normalized_exact_solver_result(
                status="error",
                runtime_total=perf_counter() - start,
                extras={"solve_type": solve_type, "equality": hard_equalities, "hard_equalities": hard_equalities, "error": str(exc)},
            )

__all__ = ["ConvexSolver"]
