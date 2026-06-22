"""JCC linear exact solver implementations."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from homopt.problems.jcc.linear import _as_numpy, _evaluate_chance_metrics
from homopt.solvers.common import _normalized_exact_solver_result
from homopt.solvers.cvxpy import CVXPY_INTEGER_SOLVER_PRIORITY, CVXPY_SOLVER_PRIORITY, _solve_cvxpy_problem

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover - optional dependency
    cp = None
    _CVXPY_IMPORT_ERROR = exc
else:  # pragma: no cover - import success path
    _CVXPY_IMPORT_ERROR = None


def _require_cvxpy():
    if _CVXPY_IMPORT_ERROR is not None:
        raise ImportError(
            "JCC linear solvers require cvxpy. Install with: pip install -e .[solvers]"
        ) from _CVXPY_IMPORT_ERROR
    return cp


class _BaseJCCLinearSolver:
    default_solver = None

    def __init__(self, problem, solver_config=None):
        self.cp = _require_cvxpy()
        self.problem = problem
        self.config = {
            "solver": self.default_solver,
            "solver_options": {},
            "verbose": False,
            **(solver_config or {}),
        }

    def solve_result(self, solve_type="opt", x_init=None, solver=None, options=None):
        start = perf_counter()
        raw = self.solve(solve_type=solve_type, x_init=x_init, solver=solver, options=options)
        runtime_total = perf_counter() - start
        return _normalized_exact_solver_result(
            solution=raw.get("x_optimal"),
            status=raw.get("status", "unknown"),
            objective=raw.get("objective_value"),
            runtime_total=runtime_total,
            violation=raw.get("max_violation"),
            feasible=raw.get("chance_constraint_satisfied"),
            extras={
                key: value
                for key, value in raw.items()
                if key not in {"x_optimal", "status", "objective_value", "max_violation"}
            },
        )

    def _decision_variable(self):
        return self.cp.Variable(self.problem.nvar, name="y")

    def _deterministic_constraints(self, y):
        constraints = [y >= self.problem.L_np, y <= self.problem.U_np]
        if self.problem.n_ineq > 0:
            constraints.append(self.problem.G_np @ y <= self.problem.h_np)
        return constraints

    def _projection_or_objective(self, y, *, solve_type="opt", x_init=None):
        if solve_type == "proj":
            if x_init is None:
                raise ValueError("Projection solve requires x_init.")
            x_target = np.asarray(x_init, dtype=np.float32).reshape(-1)
            return self.cp.Minimize(self.cp.sum_squares(y - x_target))
        return self.cp.Minimize(self.problem.p_np @ y)

    def _solve_cvxpy_problem(self, problem, *, y_var, solve_type="opt", x_init=None, solver=None, options=None):
        base_options = dict(self.config.get("solver_options") or {})
        if options:
            base_options.update(dict(options))
        if x_init is not None and solve_type in {"opt", "initialized_opt"}:
            y_var.value = np.asarray(x_init, dtype=np.float32).reshape(-1)
        _solve_cvxpy_problem(
            problem,
            self.cp,
            warm_start=x_init is not None and solve_type in {"opt", "initialized_opt"},
            verbose=bool(self.config.get("verbose", False)),
            solver_options=base_options,
            solver_priority=self.solver_priority,
            preferred_solver=solver or self.config.get("solver"),
        )

    def _finalize_raw_result(self, *, x_input, y_var, problem):
        status = str(problem.status)
        if y_var.value is None:
            return {
                "x_optimal": None,
                "status": status,
                "objective_value": None,
                "feasibility_rate": 0.0,
                "chance_constraint_satisfied": False,
                "max_violation": float("inf"),
            }
        solution = np.asarray(y_var.value, dtype=np.float32).reshape(-1)
        metrics = _evaluate_chance_metrics(self.problem, x_input, solution)
        return {
            "x_optimal": solution,
            "status": status,
            "objective_value": float(problem.value) if problem.value is not None else None,
            **metrics,
        }

class JCCLinearSolver(_BaseJCCLinearSolver):
    """Mixed-integer chance-constrained baseline."""

    default_solver = "GUROBI"
    solver_priority = CVXPY_INTEGER_SOLVER_PRIORITY

    def solve(self, solve_type="opt", x_init=None, solver=None, options=None):
        x_input = _as_numpy(self.problem.current_input).reshape(-1)
        y = self._decision_variable()
        constraints = self._deterministic_constraints(y)
        z = self.cp.Variable(self.problem.n_scenarios, boolean=True, name="z")
        big_m = float(self.config.get("M", 1e3))
        for scenario_idx in range(self.problem.n_scenarios):
            residual = x_input + self.problem.W_np[scenario_idx] - self.problem.A_np @ y
            constraints.append(residual <= big_m * (1 - z[scenario_idx]))
        constraints.append(self.cp.sum(z) >= self.problem.n_scenarios * (1.0 - self.problem.epsilon))
        objective = self._projection_or_objective(y, solve_type=solve_type, x_init=x_init)
        problem = self.cp.Problem(objective, constraints)
        self._solve_cvxpy_problem(problem, y_var=y, solve_type=solve_type, x_init=x_init, solver=solver, options=options)
        return self._finalize_raw_result(x_input=x_input, y_var=y, problem=problem)


class JCCLinearCVaRSolver(_BaseJCCLinearSolver):
    """CVaR convex approximation baseline for JCC linear family."""

    default_solver = "MOSEK"
    solver_priority = CVXPY_SOLVER_PRIORITY

    def solve(self, solve_type="opt", x_init=None, solver=None, options=None):
        x_input = _as_numpy(self.problem.current_input).reshape(-1)
        y = self._decision_variable()
        constraints = self._deterministic_constraints(y)
        eta = self.cp.Variable(name="eta")
        xi = self.cp.Variable(self.problem.n_scenarios, name="xi")
        u = self.cp.Variable(self.problem.n_scenarios, nonneg=True, name="u")
        epsilon = max(float(self.problem.epsilon), 1e-6)
        for scenario_idx in range(self.problem.n_scenarios):
            residual = x_input + self.problem.W_np[scenario_idx] - self.problem.A_np @ y
            constraints.append(xi[scenario_idx] >= self.cp.max(residual))
            constraints.append(u[scenario_idx] >= xi[scenario_idx] - eta)
        constraints.append(eta + (1.0 / (epsilon * self.problem.n_scenarios)) * self.cp.sum(u) <= 0)
        objective = self._projection_or_objective(y, solve_type=solve_type, x_init=x_init)
        problem = self.cp.Problem(objective, constraints)
        self._solve_cvxpy_problem(problem, y_var=y, solve_type=solve_type, x_init=x_init, solver=solver, options=options)
        return self._finalize_raw_result(x_input=x_input, y_var=y, problem=problem)


class JCCLinearRobustScenarioSolver(_BaseJCCLinearSolver):
    """Robust-scenario baseline enforcing all sampled scenarios."""

    default_solver = "MOSEK"
    solver_priority = CVXPY_SOLVER_PRIORITY

    def solve(self, solve_type="opt", x_init=None, solver=None, options=None):
        x_input = _as_numpy(self.problem.current_input).reshape(-1)
        y = self._decision_variable()
        constraints = self._deterministic_constraints(y)
        for scenario_idx in range(self.problem.n_scenarios):
            constraints.append(self.problem.A_np @ y >= x_input + self.problem.W_np[scenario_idx])
        objective = self._projection_or_objective(y, solve_type=solve_type, x_init=x_init)
        problem = self.cp.Problem(objective, constraints)
        self._solve_cvxpy_problem(problem, y_var=y, solve_type=solve_type, x_init=x_init, solver=solver, options=options)
        return self._finalize_raw_result(x_input=x_input, y_var=y, problem=problem)





__all__ = [
    "JCCLinearCVaRSolver",
    "JCCLinearRobustScenarioSolver",
    "JCCLinearSolver",
]
