"""Solver namespace."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_TARGET = {
    "ConvexSolver": ("homopt.solvers.core", "ConvexSolver"),
    "MaxCutSolver": ("homopt.solvers.core", "MaxCutSolver"),
    "PYOMO_AVAILABLE": ("homopt.solvers.core", "PYOMO_AVAILABLE"),
    "normalize_exact_solver_call_kwargs": ("homopt.solvers.core", "normalize_exact_solver_call_kwargs"),
    "solve_exact_result": ("homopt.solvers.core", "solve_exact_result"),
}


def __getattr__(name):
    if name not in _SYMBOL_TO_TARGET:
        raise AttributeError(name)
    module_name, attr_name = _SYMBOL_TO_TARGET[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "ConvexSolver",
    "MaxCutSolver",
    "PYOMO_AVAILABLE",
    "normalize_exact_solver_call_kwargs",
    "solve_exact_result",
]
