"""Equality-focused optimizer layer."""

from ._core_impl import (
    EqualityConstrainedALMOptimizer,
    HomALMOptimizer,
    LagrangianOptimizer,
)
from .dispatch import run_algorithm as _run_algorithm
from .shared import AdamOptimizer, GDOptimizer, NormalizedGDOptimizer

_ALLOWED = {
    'Penalty',
    'Prox-Penalty',
    'ALM',
    'Prox-ALM',
    'Penalty-EQ',
    'Prox-Penalty-EQ',
    'ALM-EQ',
    'Prox-ALM-EQ',
    'Hom-ALM',
    'Prox-Hom-ALM',
}


def run_gauge_eq_algorithm(name, problem, params, hom_map=None, init_point=None):
    if name not in _ALLOWED:
        raise ValueError(f'Unsupported gauge_eq algorithm: {name}')
    return _run_algorithm(name, problem, params, hom_map=hom_map, init_point=init_point)


__all__ = [
    'AdamOptimizer',
    'EqualityConstrainedALMOptimizer',
    'GDOptimizer',
    'HomALMOptimizer',
    'LagrangianOptimizer',
    'NormalizedGDOptimizer',
    'run_gauge_eq_algorithm',
]
