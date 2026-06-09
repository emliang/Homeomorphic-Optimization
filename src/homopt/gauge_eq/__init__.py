"""Gauge-based Hom-PGD with ALM-style methods for equality-constrained problems."""

from .mappings import GaugeMap
from .optim import HomALMOptimizer, LagrangianOptimizer, run_algorithm
from .problems import ConvexOptEq, create_test_problem

__all__ = [
    'GaugeMap',
    'HomALMOptimizer',
    'LagrangianOptimizer',
    'run_algorithm',
    'ConvexOptEq',
    'create_test_problem',
]
