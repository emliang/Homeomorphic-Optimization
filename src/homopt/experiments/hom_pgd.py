"""Hom-PGD experiment entrypoints.

This module is the public home for inequality-only Hom-PGD experiments.
Scripts should import benchmark entrypoints from here.
"""

from __future__ import annotations

from .adversarial.experiment import adversarial_attack_experiment
from .benchmarks.convex import convex_algorithm_comparison, socp_hompgd_benchmark
from .benchmarks.maxcut import maxcut_algorithm_comparison
from .benchmarks.star import poly_star_benchmark

__all__ = [
    "adversarial_attack_experiment",
    "convex_algorithm_comparison",
    "maxcut_algorithm_comparison",
    "poly_star_benchmark",
    "socp_hompgd_benchmark",
]
