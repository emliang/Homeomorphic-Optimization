"""Hom-PGD experiment entrypoints.

This module is the public home for inequality-only Hom-PGD experiments.  The
large implementation still lives in ``benchmarks_single`` while we split the
package by experiment family; scripts should import from here.
"""

from __future__ import annotations

from .adversarial import adversarial_attack_experiment, adversarial_attack_workflow
from .benchmarks_single import (
    convex_algorithm_comparison,
    maxcut_algorithm_comparison,
    poly_star_benchmark,
    socp_hompgd_benchmark,
)

__all__ = [
    "adversarial_attack_experiment",
    "adversarial_attack_workflow",
    "convex_algorithm_comparison",
    "maxcut_algorithm_comparison",
    "poly_star_benchmark",
    "socp_hompgd_benchmark",
]
