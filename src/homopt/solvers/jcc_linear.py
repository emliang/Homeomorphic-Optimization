"""JCC linear exact-solver wrappers exposed through the solver namespace."""

from __future__ import annotations

from time import perf_counter

from homopt.problems.jcc.linear import (
    JCCLinearCVaRSolver as _ProblemJCCLinearCVaRSolver,
    JCCLinearRobustScenarioSolver as _ProblemJCCLinearRobustScenarioSolver,
    JCCLinearSolver as _ProblemJCCLinearSolver,
)
from homopt.solvers.common import _normalized_exact_solver_result


class _BaseJCCLinearSolverMixin:
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


class JCCLinearSolver(_BaseJCCLinearSolverMixin, _ProblemJCCLinearSolver):
    """Mixed-integer exact solver facade for the JCC linear family."""


class JCCLinearCVaRSolver(_BaseJCCLinearSolverMixin, _ProblemJCCLinearCVaRSolver):
    """CVaR approximation facade for the JCC linear family."""


class JCCLinearRobustScenarioSolver(_BaseJCCLinearSolverMixin, _ProblemJCCLinearRobustScenarioSolver):
    """Robust scenario facade for the JCC linear family."""


__all__ = [
    "JCCLinearCVaRSolver",
    "JCCLinearRobustScenarioSolver",
    "JCCLinearSolver",
]
