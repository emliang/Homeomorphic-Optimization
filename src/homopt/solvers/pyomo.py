"""Pyomo backend utilities shared by NLP exact solvers."""

from __future__ import annotations

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverStatus, TerminationCondition
    PYOMO_AVAILABLE = True
except ImportError:
    pyo = None
    SolverStatus = None
    TerminationCondition = None
    PYOMO_AVAILABLE = False

PYOMO_NLP_SOLVER = "ipopt"

__all__ = [
    "PYOMO_AVAILABLE",
    "PYOMO_NLP_SOLVER",
    "SolverStatus",
    "TerminationCondition",
    "pyo",
]
