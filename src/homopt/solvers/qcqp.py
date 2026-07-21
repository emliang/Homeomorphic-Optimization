"""Pyomo/IPOPT exact solver for QCQP baselines."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .common import _normalized_exact_solver_result
from .pyomo import PYOMO_AVAILABLE, PYOMO_NLP_SOLVER, SolverStatus, TerminationCondition, pyo


class QCQPSolver:
    def __init__(self, problem_params):
        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo is required for QCQPSolver. Install with: pip install -r requirements.txt")
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
        if solve_type not in {"opt", "initialized_opt", "proj"}:
            raise ValueError(f"Unknown solve_type: {solve_type}")
        model = self._create_pyomo_model()
        if solve_type == "proj":
            if x_init is None:
                raise ValueError("x_init required for projection")

            def proj_obj_rule(model):
                return sum((model.x[i] - x_init[i]) ** 2 for i in range(self.n_vars))

            model.del_component(model.objective)
            model.objective = pyo.Objective(rule=proj_obj_rule, sense=pyo.minimize)
        if x_init is not None and solve_type in ["opt", "initialized_opt"]:
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

__all__ = ["QCQPSolver"]
