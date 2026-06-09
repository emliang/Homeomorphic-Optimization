"""Package-owned solver implementations."""

from __future__ import annotations

import inspect
import warnings
from time import perf_counter

import numpy as np

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover - depends on optional env state
    cp = None
    _CVXPY_IMPORT_ERROR = exc
else:
    _CVXPY_IMPORT_ERROR = None

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverStatus, TerminationCondition
    PYOMO_AVAILABLE = True
except ImportError:
    pyo = None
    SolverStatus = None
    TerminationCondition = None
    PYOMO_AVAILABLE = False


def _require_cvxpy():
    if cp is None:
        raise ImportError(
            "This solver requires cvxpy. Install with: pip install -e .[solvers]"
        ) from _CVXPY_IMPORT_ERROR
    return cp


def _soc_constraint(cp_mod, t_expr, x_expr):
    if hasattr(cp_mod, "SOC"):
        return cp_mod.SOC(t_expr, x_expr)
    return cp_mod.norm(x_expr, 2) <= t_expr


def _sum_multiply(cp_mod, x_expr, coeffs):
    if hasattr(cp_mod, "multiply") and hasattr(cp_mod, "sum"):
        return cp_mod.sum(cp_mod.multiply(x_expr, coeffs))
    return cp_mod.sum_entries(cp_mod.mul_elemwise(coeffs, x_expr))


CVXPY_SOLVER_PRIORITY = ("MOSEK", "ECOS", "SCS")
CVXPY_INTEGER_SOLVER_PRIORITY = ("GUROBI",)
PYOMO_NLP_SOLVER = "ipopt"
_CVXPY_SOLVER_CACHE = {}
_SOLVER_SIGNATURE_CACHE = {}


def _candidate_cvxpy_solvers(cp_mod, solver_priority=CVXPY_SOLVER_PRIORITY):
    cache_key = (cp_mod, tuple(solver_priority))
    if cache_key in _CVXPY_SOLVER_CACHE:
        return list(_CVXPY_SOLVER_CACHE[cache_key])
    try:
        installed = set(cp_mod.installed_solvers())
    except Exception:
        installed = set()
    candidates = [name for name in solver_priority if name in installed]
    _CVXPY_SOLVER_CACHE[cache_key] = tuple(candidates)
    return candidates


def _solver_time_limit_kwargs(solver_name, time_limit_sec):
    if time_limit_sec is None:
        return {}
    time_limit_sec = float(time_limit_sec)
    if solver_name == "GUROBI":
        return {"TimeLimit": time_limit_sec}
    if solver_name == "MOSEK":
        return {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": time_limit_sec}}
    if solver_name == "SCS":
        return {"time_limit_secs": time_limit_sec}
    return {}


def _merge_mosek_params(kwargs, extra_params):
    if not extra_params:
        return kwargs
    merged = dict(kwargs)
    merged["mosek_params"] = {
        **dict(merged.get("mosek_params") or {}),
        **dict(extra_params),
    }
    return merged


def _solve_cvxpy_problem(
    problem,
    cp_mod,
    *,
    warm_start=True,
    interior_point=False,
    verbose=False,
    solver_options=None,
    time_limit_sec=None,
    solver_priority=CVXPY_SOLVER_PRIORITY,
):
    candidates = _candidate_cvxpy_solvers(cp_mod, solver_priority=solver_priority)
    last_error = None
    for solver_name in candidates:
        solver = getattr(cp_mod, solver_name, None)
        if solver is None:
            continue
        kwargs = {
            "warm_start": warm_start,
            "solver": solver,
            "verbose": bool(verbose),
            **dict(solver_options or {}),
        }
        time_limit_kwargs = _solver_time_limit_kwargs(solver_name, time_limit_sec)
        if solver_name == "MOSEK":
            kwargs = _merge_mosek_params(kwargs, time_limit_kwargs.get("mosek_params"))
        else:
            kwargs.update(time_limit_kwargs)
        if interior_point and solver_name == "MOSEK":
            kwargs = _merge_mosek_params(
                kwargs,
                {
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1,
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-5,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-1,
                },
            )
        try:
            problem.solve(**kwargs)
            return
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(
        "No supported CVXPY solver is installed. Expected one of: "
        f"{', '.join(solver_priority)}"
    )


def _symmetric_matrix_variable(cp_mod, size):
    try:
        return cp_mod.Variable((size, size), symmetric=True), []
    except TypeError:
        X = cp_mod.Variable((size, size))
        constraints = []
        for i in range(size):
            for j in range(i + 1, size):
                constraints.append(X[i, j] == X[j, i])
        return X, constraints


def _as_numpy_or_none(value):
    if value is None:
        return None
    return np.asarray(value)


def _normalized_exact_solver_result(
    *,
    solution=None,
    status="unknown",
    objective=None,
    runtime_total=None,
    violation=None,
    feasible=None,
    extras=None,
):
    return {
        "solution": _as_numpy_or_none(solution),
        "status": str(status),
        "objective": None if objective is None else float(objective),
        "runtime_total": None if runtime_total is None else float(runtime_total),
        "violation": None if violation is None else float(violation),
        "feasible": None if feasible is None else bool(feasible),
        "extras": dict(extras or {}),
    }


def _solver_signature_or_none(fn):
    owner = getattr(fn, "__self__", None)
    cache_key = (type(owner) if owner is not None else None, getattr(fn, "__func__", fn))
    if cache_key in _SOLVER_SIGNATURE_CACHE:
        return _SOLVER_SIGNATURE_CACHE[cache_key]
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None
    _SOLVER_SIGNATURE_CACHE[cache_key] = signature
    return signature


def _filter_solver_kwargs(fn, kwargs):
    signature = _solver_signature_or_none(fn)
    if signature is None:
        return dict(kwargs)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(kwargs)
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }


def _solver_accepts_keyword(fn, keyword):
    signature = _solver_signature_or_none(fn)
    if signature is None:
        return True
    parameters = signature.parameters
    return keyword in parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )


def normalize_exact_solver_call_kwargs(exact_solver, *, solve_config=None, **kwargs):
    normalized = dict(solve_config or {})
    normalized.update(kwargs)
    target = exact_solver.solve_result if hasattr(exact_solver, "solve_result") else exact_solver.solve
    if (
        "solver" not in normalized
        and "solver_name" in normalized
        and not _solver_accepts_keyword(target, "solver_name")
        and _solver_accepts_keyword(target, "solver")
    ):
        normalized["solver"] = normalized.pop("solver_name")
    elif "solver_name" in normalized and not _solver_accepts_keyword(target, "solver_name"):
        normalized.pop("solver_name", None)
    if (
        "options" not in normalized
        and "solver_options" in normalized
        and not _solver_accepts_keyword(target, "solver_options")
        and _solver_accepts_keyword(target, "options")
    ):
        normalized["options"] = normalized.pop("solver_options")
    elif "solver_options" in normalized and not _solver_accepts_keyword(target, "solver_options"):
        normalized.pop("solver_options", None)
    return _filter_solver_kwargs(target, normalized)


def solve_exact_result(exact_solver, *args, **kwargs):
    call_kwargs = normalize_exact_solver_call_kwargs(exact_solver, **kwargs)
    if hasattr(exact_solver, "solve_result"):
        return exact_solver.solve_result(*args, **call_kwargs)
    start = perf_counter()
    raw = exact_solver.solve(*args, **call_kwargs)
    runtime_total = perf_counter() - start
    if isinstance(raw, dict):
        solution = raw.get("solution", raw.get("x_optimal"))
        status = raw.get("status", "unknown")
        objective = raw.get("objective", raw.get("objective_value"))
        violation = raw.get("violation", raw.get("max_violation"))
        feasible = raw.get("feasible")
        extras = {
            key: value
            for key, value in raw.items()
            if key not in {"solution", "x_optimal", "status", "objective", "objective_value", "violation", "max_violation", "feasible"}
        }
        return _normalized_exact_solver_result(
            solution=solution,
            status=status,
            objective=objective,
            runtime_total=runtime_total,
            violation=violation,
            feasible=feasible,
            extras=extras,
        )
    return _normalized_exact_solver_result(
        solution=raw,
        status="optimal" if raw is not None else "failed",
        objective=None,
        runtime_total=runtime_total,
        violation=None,
        feasible=None if raw is None else True,
        extras={},
    )


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
        solver_options=None,
        time_limit_sec=None,
        verbose=False,
    ):
        cp_mod = _require_cvxpy()
        if solve_type in ["opt", "warm_start"]:
            decision, constraints = self._create_base_problem(equality=True)
            objective = cp_mod.Minimize(self._quadratic_objective_expr(cp_mod, decision))
            if solve_type == "warm_start":
                if x_init is None:
                    raise ValueError("x_init required for warm start")
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
                solver_options=solver_options,
                time_limit_sec=time_limit_sec,
                verbose=verbose,
            )
            runtime_total = perf_counter() - start
            violation = self._constraint_violation(solution)
            return _normalized_exact_solver_result(
                solution=solution,
                status="optimal" if solution is not None else "failed",
                objective=self._objective_value(solution) if solve_type in ["opt", "warm_start", "eq_alm"] else None,
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
        if solve_type not in ["opt", "warm_start"]:
            raise ValueError(f"Unknown solve_type: {solve_type}")
        approach = self._normalize_approach(approach)
        x, constraints = self._create_base_problem(approach)
        objective = cp_mod.Minimize(0.5 * cp_mod.quad_form(x, self.Q) + self.p.T @ x)
        if solve_type == "warm_start" and x_init is not None:
            x.value = x_init
        problem = cp_mod.Problem(objective, constraints)
        _solve_cvxpy_problem(
            problem,
            cp_mod,
            warm_start=(solve_type == "warm_start"),
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


class QCQPSolver:
    def __init__(self, problem_params):
        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo is required for QCQPSolver. Install with: pip install -e .[solvers]")
        (self.Q, self.p, self.A, self.b, self.Qq, self.pq, self.bq, self.L, self.U, self.R) = problem_params
        self.n_vars = len(self.p)
        self.n_lin = self.A.shape[0] if self.A is not None else 0
        self.n_qua = self.Qq.shape[0] if self.Qq is not None else 0

    def _create_pyomo_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(range(self.n_vars), domain=pyo.Reals)
        for i in range(self.n_vars):
            model.x[i].setlb(self.L[i])
            model.x[i].setub(self.U[i])

        def obj_rule(model):
            quad_term = sum(0.5 * self.Q[i, j] * model.x[i] * model.x[j] for i in range(self.n_vars) for j in range(self.n_vars))
            linear_term = sum(self.p[i] * model.x[i] for i in range(self.n_vars))
            return quad_term + linear_term

        model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        if self.n_lin > 0:
            model.linear_cons = pyo.ConstraintList()
            for i in range(self.n_lin):
                lhs = sum(self.A[i, j] * model.x[j] for j in range(self.n_vars))
                model.linear_cons.add(lhs <= self.b[i])
        if self.n_qua > 0:
            model.quad_cons = pyo.ConstraintList()
            for i in range(self.n_qua):
                quad_term = sum(0.5 * self.Qq[i, j, k] * model.x[j] * model.x[k] for j in range(self.n_vars) for k in range(self.n_vars))
                linear_term = sum(self.pq[i, j] * model.x[j] for j in range(self.n_vars))
                model.quad_cons.add(quad_term + linear_term <= self.bq[i])
        if self.R > 0:
            model.r_ball_cons = pyo.ConstraintList()
            model.r_ball_cons.add(sum(model.x[i] ** 2 for i in range(self.n_vars)) <= self.R ** 2)
        return model

    def solve(self, solve_type="opt", x_init=None, solver=PYOMO_NLP_SOLVER, options=None):
        if solve_type not in {"opt", "warm_start", "proj"}:
            raise ValueError(f"Unknown solve_type: {solve_type}")
        model = self._create_pyomo_model()
        if solve_type == "proj":
            if x_init is None:
                raise ValueError("x_init required for projection")

            def proj_obj_rule(model):
                return sum((model.x[i] - x_init[i]) ** 2 for i in range(self.n_vars))

            model.del_component(model.objective)
            model.objective = pyo.Objective(rule=proj_obj_rule, sense=pyo.minimize)
        if x_init is not None and solve_type in ["opt", "warm_start"]:
            for i in range(self.n_vars):
                model.x[i].set_value(x_init[i])
        opt = pyo.SolverFactory(solver)
        if not opt.available():
            raise RuntimeError(f"Solver {solver} is not available")
        default_options = {
            "max_iter": 3000,
            "tol": 1e-8,
            "print_level": 0,
            "sb": "yes",
            "acceptable_tol": 1e-6,
            "acceptable_iter": 15,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
        }
        if options:
            default_options.update(options)
        for key, value in default_options.items():
            opt.options[key] = value
        try:
            results = opt.solve(model, tee=False)
            if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
                solution = np.array([pyo.value(model.x[i]) for i in range(self.n_vars)])
                return {
                    "solution": solution,
                    "status": "optimal",
                    "objective_value": pyo.value(model.objective),
                    "solver_status": results.solver.status,
                    "termination_condition": results.solver.termination_condition,
                }
            try:
                solution = np.array([pyo.value(model.x[i]) for i in range(self.n_vars)])
                obj_value = pyo.value(model.objective)
            except Exception:
                solution = None
                obj_value = None
            return {
                "solution": solution,
                "status": "non_optimal",
                "objective_value": obj_value,
                "solver_status": results.solver.status,
                "termination_condition": results.solver.termination_condition,
            }
        except Exception as e:
            return {"solution": None, "status": "error", "objective_value": None, "error": str(e)}

    def check_feasibility(self, x, tol=1e-6):
        violations = {}
        max_violation = 0
        box_violations = np.maximum(self.L - x, 0) + np.maximum(x - self.U, 0)
        violations["box"] = box_violations
        max_violation = max(max_violation, np.max(box_violations))
        if self.n_lin > 0:
            lin_violations = np.maximum(np.dot(self.A, x) - self.b, 0)
            violations["linear"] = lin_violations
            max_violation = max(max_violation, np.max(lin_violations))
        if self.n_qua > 0:
            quad_violations = []
            for i in range(self.n_qua):
                quad_val = 0.5 * np.dot(x, np.dot(self.Qq[i], x)) + np.dot(self.pq[i], x)
                violation = max(quad_val - self.bq[i], 0)
                quad_violations.append(violation)
                max_violation = max(max_violation, violation)
            violations["quadratic"] = np.array(quad_violations)
        if self.R > 0:
            r_ball_violation = max(float(np.sum(x ** 2) - self.R ** 2), 0.0)
            violations["r_ball"] = r_ball_violation
            max_violation = max(max_violation, r_ball_violation)
        return {"feasible": max_violation <= tol, "max_violation": max_violation, "violations": violations}

    def evaluate_objective(self, x):
        return 0.5 * np.dot(x, np.dot(self.Q, x)) + np.dot(self.p, x)

    def solve_result(self, solve_type="opt", x_init=None, solver=PYOMO_NLP_SOLVER, options=None):
        start = perf_counter()
        raw = self.solve(solve_type=solve_type, x_init=x_init, solver=solver, options=options)
        runtime_total = perf_counter() - start
        solution = raw.get("solution")
        feasibility = self.check_feasibility(solution) if solution is not None else None
        return _normalized_exact_solver_result(
            solution=solution,
            status=raw.get("status", "unknown"),
            objective=raw.get("objective_value"),
            runtime_total=runtime_total,
            violation=None if feasibility is None else feasibility.get("max_violation"),
            feasible=None if feasibility is None else feasibility.get("feasible"),
            extras={k: v for k, v in raw.items() if k not in {"solution", "status", "objective_value"}},
        )


__all__ = [
    "PYOMO_AVAILABLE",
    "CVXPY_SOLVER_PRIORITY",
    "CVXPY_INTEGER_SOLVER_PRIORITY",
    "PYOMO_NLP_SOLVER",
    "_candidate_cvxpy_solvers",
    "_normalized_exact_solver_result",
    "_solve_cvxpy_problem",
    "normalize_exact_solver_call_kwargs",
    "solve_exact_result",
    "ChanceConstraintSolver",
    "ConvexSolver",
    "MaxCutSolver",
    "QCQPSolver",
]
