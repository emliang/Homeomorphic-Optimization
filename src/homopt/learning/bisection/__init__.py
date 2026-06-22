"""Bisection primitives and model-specific adapters."""

from .adapters import homeomorphic_bisection, interior_point_bisection
from .core import BisectionConfig, BisectionResult, bisect_segment

__all__ = [
    "BisectionConfig",
    "BisectionResult",
    "bisect_segment",
    "homeomorphic_bisection",
    "interior_point_bisection",
]
