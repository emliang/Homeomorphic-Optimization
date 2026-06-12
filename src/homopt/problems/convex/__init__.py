"""Deterministic convex problem families."""

from .generators import create_test_problem
from .maxcut import BMSDP, MaxCutSDP, create_maxcut_problem
from .standard import ConvexOpt, ConvexOptEq, ParametricConvexProblem
from .star import PolyStarOpt, ToyStarOpt
from .wrappers import LinearProblem, ProjProblem

__all__ = [
    "BMSDP",
    "ConvexOpt",
    "ConvexOptEq",
    "LinearProblem",
    "MaxCutSDP",
    "ParametricConvexProblem",
    "PolyStarOpt",
    "ProjProblem",
    "ToyStarOpt",
    "create_maxcut_problem",
    "create_test_problem",
]
