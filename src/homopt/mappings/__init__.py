"""Homeomorphic mappings namespace."""

from .gauge import GaugeMap, GaugeMapMaxCut, max_eigenvalue_with_gradient, soft_maximum
from .star import PolyStarMap, StarMap

__all__ = ["GaugeMap", "GaugeMapMaxCut", "PolyStarMap", "StarMap", "max_eigenvalue_with_gradient", "soft_maximum"]
