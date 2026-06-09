"""Optimization algorithms namespace."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_TARGET = {
    "AdamOptimizer": ("homopt.optim.shared", "AdamOptimizer"),
    "BaseOptimizer": ("homopt.optim.base", "BaseOptimizer"),
    "FrankWolfeOptimizer": ("homopt.optim._core_impl", "FrankWolfeOptimizer"),
    "GDOptimizer": ("homopt.optim.shared", "GDOptimizer"),
    "HomPGDOptimizer": ("homopt.optim._core_impl", "HomPGDOptimizer"),
    "LagrangianOptimizer": ("homopt.optim._core_impl", "LagrangianOptimizer"),
    "NormalizedGDOptimizer": ("homopt.optim.shared", "NormalizedGDOptimizer"),
    "OptimizerAdapter": ("homopt.optim.base", "OptimizerAdapter"),
    "PGDOptimizer": ("homopt.optim._core_impl", "PGDOptimizer"),
    "RadialAlgorithm": ("homopt.optim._core_impl", "RadialAlgorithm"),
    "RadialDualOptimizer": ("homopt.optim._core_impl", "RadialDualOptimizer"),
    "as_optimizer": ("homopt.optim.base", "as_optimizer"),
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
    "HomPGDOptimizer",
    "LagrangianOptimizer",
    "NormalizedGDOptimizer",
    "OptimizerAdapter",
    "PGDOptimizer",
    "RadialAlgorithm",
    "RadialDualOptimizer",
    "as_optimizer",
    "run_algorithm",
]
