"""Backward-compatible import path for legacy visualization helpers."""

from __future__ import annotations

from .legacy import plot_both_spaces, plot_optimization_trajectory, vis

__all__ = [
    "plot_both_spaces",
    "plot_optimization_trajectory",
    "vis",
]
