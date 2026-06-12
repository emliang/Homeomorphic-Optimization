"""CVXPY exact solver for Max-Cut SDP baselines."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .common import _normalized_exact_solver_result
from .cvxpy import _require_cvxpy, _solve_cvxpy_problem, _symmetric_matrix_variable


class MaxCutSolver:
    def __init__(self, problem_params):
        self.node = problem_params["node"]
        self.edge = problem_params["edge"]
        self.weights = problem_params["weights"]
        self.num_node = len(self.node)
        self.num_edge = len(self.edge)
        upper_triangle_index = [(i, j) for i in range(self.num_node) for j in range(self.num_node) if i < j]
        self.upper_triangle_index = np.array(upper_triangle_index)
        upper_index_by_edge = {edge: index for index, edge in enumerate(upper_triangle_index)}
        self.edge_index = np.array(
            [upper_index_by_edge[(min(i, j), max(i, j))] for i, j in self.edge],
            dtype=int,
        )

    def _create_base_problem(self, epsilon=None):
        cp_mod = _require_cvxpy()
        X, symmetry_constraints = _symmetric_matrix_variable(cp_mod, self.num_node)
        if epsilon is None:
            constraints = [X[i, i] == 1 for i in range(self.num_node)] + symmetry_constraints
            constraints += [X >> 0, X <= 1, X >= -1]
        else:
            constraints = [X[i, i] == 1 for i in range(self.num_node)] + symmetry_constraints
            constraints += [X >> epsilon, X <= 1 - epsilon, X >= -1 + epsilon, epsilon >= 0]
        return X, constraints

    def solve(self, solve_type="opt", x_init=None):
        cp_mod = _require_cvxpy()
        if solve_type == "opt":
            X, constraints = self._create_base_problem()
            edge_indices = np.array(self.edge)
            i_indices = edge_indices[:, 0]
            j_indices = edge_indices[:, 1]
            objective = cp_mod.Maximize(self.weights @ (1 - X[i_indices, j_indices]) / 2)
            problem = cp_mod.Problem(objective, constraints)
            _solve_cvxpy_problem(problem, cp_mod, warm_start=True, interior_point=True)
            return X.value
        if solve_type == "ip":
            X, constraints = self._create_base_problem()
            epsilon = cp_mod.Variable(1)
            objective = cp_mod.Maximize(epsilon)
            problem = cp_mod.Problem(objective, constraints)
            _solve_cvxpy_problem(problem, cp_mod, warm_start=True, interior_point=True)
            return X.value, epsilon.value
        if solve_type == "proj":
            X, constraints = self._create_base_problem()
            X_init = np.zeros([self.num_node, self.num_node])
            X_init[self.upper_triangle_index[:, 0], self.upper_triangle_index[:, 1]] = x_init
            X_init = X_init + X_init.T + np.eye(self.num_node)
            objective = cp_mod.Minimize(cp_mod.sum_squares(X - X_init))
            problem = cp_mod.Problem(objective, constraints)
            _solve_cvxpy_problem(problem, cp_mod, warm_start=True, interior_point=True)
            return X.value[self.upper_triangle_index[:, 0], self.upper_triangle_index[:, 1]]
        raise ValueError(f"Unknown solve_type: {solve_type}")

    def _edge_vector(self, solution):
        if solution is None:
            return None
        solution = np.asarray(solution)
        if solution.ndim == 2:
            return solution[self.upper_triangle_index[:, 0], self.upper_triangle_index[:, 1]]
        return solution.reshape(-1)

    def _objective_value(self, solution):
        edge_vector = self._edge_vector(solution)
        if edge_vector is None:
            return None
        return float(self.weights @ (1 - edge_vector[self.edge_index]) / 2)

    def solve_result(self, solve_type="opt", x_init=None):
        start = perf_counter()
        try:
            raw = self.solve(solve_type=solve_type, x_init=x_init)
            runtime_total = perf_counter() - start
            solution = raw[0] if isinstance(raw, tuple) else raw
            extras = {"solve_type": solve_type}
            if isinstance(raw, tuple) and len(raw) > 1:
                extras["auxiliary"] = raw[1]
            return _normalized_exact_solver_result(
                solution=solution,
                status="optimal" if solution is not None else "failed",
                objective=self._objective_value(solution) if solve_type == "opt" else None,
                runtime_total=runtime_total,
                violation=0.0 if solution is not None else None,
                feasible=None if solution is None else True,
                extras=extras,
            )
        except Exception as exc:
            return _normalized_exact_solver_result(
                status="error",
                runtime_total=perf_counter() - start,
                extras={"solve_type": solve_type, "error": str(exc)},
            )

__all__ = ["MaxCutSolver"]
