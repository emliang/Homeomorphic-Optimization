"""Base interfaces for optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Unified interface for optimization algorithms."""

    name: str = "optimizer"
    supports_mapping: bool = False

    def __init__(self, problem=None, params=None, hom_map=None):
        self.problem = problem
        self.params = params or {}
        self.hom_map = hom_map

    @abstractmethod
    def optimize(self, initial_point=None, verbose=False, seed=2025):
        """Run optimization and return implementation-defined results."""

    def set_problem(self, problem):
        self.problem = problem
        return self


class OptimizerAdapter(BaseOptimizer):
    """Wrap an optimizer-like object into the BaseOptimizer contract."""

    def __init__(self, optimizer, problem=None, params=None, hom_map=None):
        super().__init__(problem=problem, params=params, hom_map=hom_map)
        self.optimizer = optimizer
        self.name = type(optimizer).__name__

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        if hasattr(self.optimizer, "optimize"):
            return self.optimizer.optimize(initial_point=initial_point, verbose=verbose, seed=seed)
        if hasattr(self.optimizer, "solve"):
            return self.optimizer.solve(initial_point=initial_point, verbose=verbose, seed=seed)
        raise AttributeError(f"{type(self.optimizer).__name__} has neither optimize nor solve.")

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


def as_optimizer(optimizer):
    """Return an optimizer as a BaseOptimizer without breaking legacy classes."""
    if isinstance(optimizer, BaseOptimizer):
        return optimizer
    return OptimizerAdapter(optimizer)
