"""Inequality-focused optimizer layer."""

from .registry import FrankWolfeOptimizer, HomPGDOptimizer, PGDOptimizer, RadialDualOptimizer
from .dispatch import run_algorithm as _run_algorithm
from .shared import AdamOptimizer, GDOptimizer, NormalizedGDOptimizer

_ALLOWED = {'PGD', 'Hom-PGD', 'RD', 'FW'}


def run_gauge_ineq_algorithm(name, problem, params, hom_map=None, init_point=None):
    if name not in _ALLOWED:
        raise ValueError(f'Unsupported gauge_ineq algorithm: {name}')
    return _run_algorithm(name, problem, params, hom_map=hom_map, init_point=init_point)


__all__ = [
    'AdamOptimizer',
    'FrankWolfeOptimizer',
    'GDOptimizer',
    'HomPGDOptimizer',
    'NormalizedGDOptimizer',
    'PGDOptimizer',
    'RadialDualOptimizer',
    'run_gauge_ineq_algorithm',
]
