"""Gauge-based Hom-PGD for inequality-constrained problems."""

from .mappings import GaugeMap, GaugeMapMaxCut, StarMap
from .optim import FrankWolfeOptimizer, HomPGDOptimizer, PGDOptimizer, RadialDualOptimizer, run_algorithm
from .problems import ConvexOpt, MaxCutSDP, ToyStarOpt, create_maxcut_problem, create_test_problem

__all__ = [
    'GaugeMap',
    'GaugeMapMaxCut',
    'StarMap',
    'FrankWolfeOptimizer',
    'HomPGDOptimizer',
    'PGDOptimizer',
    'RadialDualOptimizer',
    'run_algorithm',
    'ConvexOpt',
    'MaxCutSDP',
    'ToyStarOpt',
    'create_maxcut_problem',
    'create_test_problem',
]
