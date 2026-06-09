"""Base interfaces for homeomorphic mappings."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMap(ABC):
    """Minimal interface for a map between latent and feasible spaces."""

    @abstractmethod
    def forward(self, z):
        """Map latent points to problem-space points."""

    @abstractmethod
    def inverse(self, x):
        """Map problem-space points back to latent points."""
