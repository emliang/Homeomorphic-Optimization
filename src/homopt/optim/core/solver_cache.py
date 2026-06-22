"""Exact-solver caching helpers used by optimizers."""

from __future__ import annotations


def _load_solver_classes():
    from homopt.solvers import ConvexSolver, MaxCutSolver

    return ConvexSolver, MaxCutSolver


def _extract_exact_solution(result):
    solution = result.get("solution")
    if solution is None:
        extras = result.get("extras") or {}
        error = result.get("error") or extras.get("error")
        detail = "" if error is None else f": {error}"
        raise RuntimeError(f"Exact solver did not return a solution: status={result.get('status')}{detail}")
    return solution


def _cached_exact_solver(owner, solver_kind):
    cache = getattr(owner, "_exact_solver_cache", None)
    if cache is None:
        cache = {}
        owner._exact_solver_cache = cache
    if solver_kind not in cache:
        ConvexSolver, MaxCutSolver = _load_solver_classes()
        if solver_kind == "convex":
            cache[solver_kind] = ConvexSolver(owner.problem.prob_para)
        elif solver_kind == "maxcut":
            cache[solver_kind] = MaxCutSolver(owner.problem.prob_para)
        else:
            raise ValueError(f"Unsupported exact solver kind: {solver_kind}")
    return cache[solver_kind]


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
