"""Solver namespace."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_TARGET = {
    "ChanceConstraintSolver": ("homopt.solvers.chance", "ChanceConstraintSolver"),
    "ConvexSolver": ("homopt.solvers.convex", "ConvexSolver"),
    "JCCDCOPFCVaRSolver": ("homopt.solvers.jcc", "JCCDCOPFCVaRSolver"),
    "JCCDCOPFRobustScenarioSolver": ("homopt.solvers.jcc", "JCCDCOPFRobustScenarioSolver"),
    "JCCDCOPFSolver": ("homopt.solvers.jcc", "JCCDCOPFSolver"),
    "JCCLinearCVaRSolver": ("homopt.solvers.jcc_linear", "JCCLinearCVaRSolver"),
    "JCCLinearRobustScenarioSolver": ("homopt.solvers.jcc_linear", "JCCLinearRobustScenarioSolver"),
    "JCCLinearSolver": ("homopt.solvers.jcc_linear", "JCCLinearSolver"),
    "MaxCutSolver": ("homopt.solvers.maxcut", "MaxCutSolver"),
    "PYOMO_AVAILABLE": ("homopt.solvers.pyomo", "PYOMO_AVAILABLE"),
    "QCQPSolver": ("homopt.solvers.qcqp", "QCQPSolver"),
    "StiefelPyomoIPOPTSolver": ("homopt.solvers.stiefel", "StiefelPyomoIPOPTSolver"),
    "normalize_exact_solver_call_kwargs": ("homopt.solvers.common", "normalize_exact_solver_call_kwargs"),
    "solve_exact_result": ("homopt.solvers.common", "solve_exact_result"),
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
    "ChanceConstraintSolver",
    "ConvexSolver",
    "JCCDCOPFCVaRSolver",
    "JCCDCOPFRobustScenarioSolver",
    "JCCDCOPFSolver",
    "JCCLinearCVaRSolver",
    "JCCLinearRobustScenarioSolver",
    "JCCLinearSolver",
    "MaxCutSolver",
    "PYOMO_AVAILABLE",
    "QCQPSolver",
    "StiefelPyomoIPOPTSolver",
    "normalize_exact_solver_call_kwargs",
    "solve_exact_result",
]
