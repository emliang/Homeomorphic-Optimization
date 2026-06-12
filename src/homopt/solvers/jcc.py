"""JCC exact-solver wrappers exposed through the solver namespace."""

from __future__ import annotations

from time import perf_counter

from homopt.problems.jcc.opf import (
    JCCDCOPFCVaRSolver as _ProblemJCCDCOPFCVaRSolver,
    JCCDCOPFRobustScenarioSolver as _ProblemJCCDCOPFRobustScenarioSolver,
    JCCDCOPFSolver as _ProblemJCCDCOPFSolver,
)
from homopt.solvers.common import _normalized_exact_solver_result


class _BaseJCCSolverMixin:
    def solve_result(self, solve_type="opt", x_init=None, solver=None, options=None):
        if solve_type not in {"opt", "initialized_opt"}:
            raise ValueError(f"Unsupported JCC solve_type: {solve_type}")
        warm_start_payload = {}
        if x_init is not None:
            warm_start_payload.setdefault("x", x_init)
        if not warm_start_payload:
            warm_start_payload = None
        original_config = dict(self.config)
        if solver is not None:
            resolved_solver = getattr(self.cp, solver) if isinstance(solver, str) and hasattr(self.cp, solver) else solver
            self.config["solver"] = resolved_solver
        if options:
            self.config.update(dict(options))
        start = perf_counter()
        try:
            raw = self.solve(warm_start=warm_start_payload)
            runtime_total = perf_counter() - start
        finally:
            self.config = original_config
        return _normalized_exact_solver_result(
            solution=raw.get("x_optimal"),
            status=raw.get("status", "unknown"),
            objective=raw.get("objective_value"),
            runtime_total=runtime_total,
            violation=None
            if raw.get("feasibility_rate") is None
            else max(0.0, (1.0 - float(raw["feasibility_rate"])) - float(self.problem.config["epsilon"])),
            feasible=raw.get("chance_constraint_satisfied"),
            extras={k: v for k, v in raw.items() if k not in {"x_optimal", "status", "objective_value"}},
        )


class JCCDCOPFSolver(_BaseJCCSolverMixin, _ProblemJCCDCOPFSolver):
    pass


class JCCDCOPFCVaRSolver(_BaseJCCSolverMixin, _ProblemJCCDCOPFCVaRSolver):
    pass


class JCCDCOPFRobustScenarioSolver(_BaseJCCSolverMixin, _ProblemJCCDCOPFRobustScenarioSolver):
    pass


__all__ = [
    "JCCDCOPFSolver",
    "JCCDCOPFCVaRSolver",
    "JCCDCOPFRobustScenarioSolver",
]
