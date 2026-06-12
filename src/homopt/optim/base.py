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

