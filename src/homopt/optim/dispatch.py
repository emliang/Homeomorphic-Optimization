"""Algorithm dispatch entrypoint.

This module is the public routing layer for named optimization algorithms.
Implementation classes can move between internal modules without changing
script and benchmark imports.
"""

from __future__ import annotations

from ._core_impl import run_algorithm

__all__ = ["run_algorithm"]
