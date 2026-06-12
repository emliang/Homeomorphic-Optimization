"""Optimization algorithms namespace."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_TARGET = {
    "AdamOptimizer": ("homopt.optim.shared", "AdamOptimizer"),
    "BaseOptimizer": ("homopt.optim.base", "BaseOptimizer"),
    "FrankWolfeOptimizer": ("homopt.optim.first_order", "FrankWolfeOptimizer"),
    "GDOptimizer": ("homopt.optim.shared", "GDOptimizer"),
    "EqualityConstrainedALMOptimizer": ("homopt.optim.lagrangian", "EqualityConstrainedALMOptimizer"),
    "HomALMOptimizer": ("homopt.optim.hom_alm", "HomALMOptimizer"),
    "HomPGDOptimizer": ("homopt.optim.first_order", "HomPGDOptimizer"),
    "INNPGDOptimizer": ("homopt.optim.inn_pgd", "INNPGDOptimizer"),
    "LagrangianOptimizer": ("homopt.optim.lagrangian", "LagrangianOptimizer"),
    "NormalizedGDOptimizer": ("homopt.optim.shared", "NormalizedGDOptimizer"),
    "PGDOptimizer": ("homopt.optim.first_order", "PGDOptimizer"),
    "RadialDualOptimizer": ("homopt.optim.first_order", "RadialDualOptimizer"),
    "StiefelALMEQPGDOptimizer": ("homopt.optim.stiefel", "StiefelALMEQPGDOptimizer"),
    "StiefelRetractionALMOptimizer": ("homopt.optim.stiefel", "StiefelRetractionALMOptimizer"),
    "StiefelRetractionOptimizer": ("homopt.optim.stiefel", "StiefelRetractionOptimizer"),
    "pgd_transformed_space": ("homopt.optim.inn_pgd", "pgd_transformed_space"),
    "run_algorithm": ("homopt.optim.dispatch", "run_algorithm"),
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
    "AdamOptimizer",
    "BaseOptimizer",
    "FrankWolfeOptimizer",
    "GDOptimizer",
    "EqualityConstrainedALMOptimizer",
    "HomALMOptimizer",
    "HomPGDOptimizer",
    "INNPGDOptimizer",
    "LagrangianOptimizer",
    "NormalizedGDOptimizer",
    "PGDOptimizer",
    "RadialDualOptimizer",
    "StiefelALMEQPGDOptimizer",
    "StiefelRetractionALMOptimizer",
    "StiefelRetractionOptimizer",
    "pgd_transformed_space",
    "run_algorithm",
]
