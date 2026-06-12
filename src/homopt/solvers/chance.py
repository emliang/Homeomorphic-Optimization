"""CVXPY exact solver for chance-constrained quadratic baselines."""

from __future__ import annotations

from time import perf_counter

from .common import _normalized_exact_solver_result
from .cvxpy import CVXPY_INTEGER_SOLVER_PRIORITY, CVXPY_SOLVER_PRIORITY, _require_cvxpy, _solve_cvxpy_problem


class ChanceConstraintSolver:
    def __init__(self, problem_params):
        (self.Q, self.p, self.A, self.b, self.L, self.U, self.b_noise_samples, self.A_noise_samples, self.delta) = problem_params
        self.n_vars = len(self.p)
        self.n_constraints = self.A.shape[0]
        self.n_samples = len(self.b_noise_samples)

    def _normalize_approach(self, approach):
        if approach == "robust":
            return "robust-scenario"
        if approach not in {"robust-scenario", "mixed-integer"}:
            raise ValueError(f"Unknown approach: {approach}")
        return approach

    def _create_base_problem(self, approach="robust-scenario"):
        cp_mod = _require_cvxpy()
        approach = self._normalize_approach(approach)
        x = cp_mod.Variable(self.n_vars)
        box_constraints = [x <= self.U, x >= self.L]
        chance_constraints = []
        if approach == "robust-scenario":
            for i in range(self.n_samples):
                b_noise = self.b_noise_samples[i]
                A_noise = self.A_noise_samples[i]
                chance_constraints.append((self.A + A_noise) @ x <= self.b + b_noise)
            prob_constraint = []
        elif approach == "mixed-integer":
            z = cp_mod.Variable((self.n_samples, 1), boolean=True)
            for i in range(self.n_samples):
                b_noise = self.b_noise_samples[i]
                A_noise = self.A_noise_samples[i]
                M = 1e8
                chance_constraints.append((self.A + A_noise) @ x - self.b - b_noise <= M * (1 - z[i]))
            prob_constraint = [cp_mod.sum(z) >= self.n_samples * (1 - self.delta)]
        else:
            raise ValueError(f"Unknown approach: {approach}")
        return x, box_constraints + chance_constraints + prob_constraint

    def solve(self, solve_type="opt", x_init=None, approach="robust-scenario"):
        cp_mod = _require_cvxpy()
        if solve_type not in ["opt", "initialized_opt"]:
            raise ValueError(f"Unknown solve_type: {solve_type}")
        approach = self._normalize_approach(approach)
        x, constraints = self._create_base_problem(approach)
        objective = cp_mod.Minimize(0.5 * cp_mod.quad_form(x, self.Q) + self.p.T @ x)
        if solve_type == "initialized_opt" and x_init is not None:
            x.value = x_init
        problem = cp_mod.Problem(objective, constraints)
        _solve_cvxpy_problem(
            problem,
            cp_mod,
            warm_start=(solve_type == "initialized_opt"),
            solver_priority=CVXPY_INTEGER_SOLVER_PRIORITY if approach == "mixed-integer" else CVXPY_SOLVER_PRIORITY,
        )
        status = problem.status
        if status == "optimal":
            return x.value, status
        try:
            return x.value, f"Solver status: {status}"
        except Exception:
            return None, f"Solver status: {status}, no solution found"

    def solve_result(self, solve_type="opt", x_init=None, approach="robust-scenario"):
        start = perf_counter()
        try:
            approach = self._normalize_approach(approach)
            solution, status = self.solve(solve_type=solve_type, x_init=x_init, approach=approach)
            runtime_total = perf_counter() - start
            return _normalized_exact_solver_result(
                solution=solution,
                status=status,
                objective=None,
                runtime_total=runtime_total,
                violation=None,
                feasible=bool(solution is not None and str(status) == "optimal"),
                extras={"solve_type": solve_type, "approach": approach},
            )
        except Exception as exc:
            return _normalized_exact_solver_result(
                status="error",
                runtime_total=perf_counter() - start,
                extras={"solve_type": solve_type, "approach": approach, "error": str(exc)},
            )

__all__ = ["ChanceConstraintSolver"]
