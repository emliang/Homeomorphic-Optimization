"""Hom-PGD experiment entrypoints.

This module is the public home for inequality-only Hom-PGD experiments.
Scripts should import benchmark entrypoints from here.
"""

from __future__ import annotations

from .adversarial.experiment import adversarial_attack_experiment
from .benchmarks.convex import convex_algorithm_comparison, socp_hompgd_benchmark
from .benchmarks.maxcut import maxcut_algorithm_comparison
from .benchmarks.star import build_poly_star_toy_context, poly_star_benchmark, solve_poly_star_toy_reference

__all__ = [
    "adversarial_attack_experiment",
    "build_poly_star_toy_context",
    "convex_algorithm_comparison",
    "maxcut_algorithm_comparison",
    "poly_star_benchmark",
    "solve_poly_star_toy_reference",
    "socp_hompgd_benchmark",
]
