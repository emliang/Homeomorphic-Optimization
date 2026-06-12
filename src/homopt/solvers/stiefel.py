"""Stiefel-specific external solver backends."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from homopt.solvers.cvxpy import _require_cvxpy, _solve_cvxpy_problem


def _torch_to_numpy(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()


def _stiefel_initial_point(problem, x_init=None, seed=2025):
    if x_init is None:
        if seed is not None:
            torch.manual_seed(int(seed))
        raw = torch.randn(
            1,
            problem.nvar,
            device=problem.device,
            dtype=problem.A_obj.dtype,
        )
    else:
        raw = torch.as_tensor(
            x_init,
            device=problem.device,
            dtype=problem.A_obj.dtype,
        ).view(1, -1)
    return problem.retract(raw)


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


class StiefelPyomoIPOPTSolver:
    """Ground-truth NLP baseline for small Stiefel instances using Pyomo + IPOPT."""

    def __init__(self, problem):
        self.problem = problem

    def _initial_point(self, x_init=None, seed=2025):
        return _stiefel_initial_point(self.problem, x_init=x_init, seed=seed)

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
