"""Experiment helpers for script-only runs."""

from __future__ import annotations

from .catalog import (
    EXPERIMENT_FAMILIES,
    SCRIPT_EXPERIMENTS,
)
from .context import ExperimentContext
from .entrypoints import default_script_output_dir, record_script_run
from .results import ExperimentResult, load_result, save_result
from .runner import run_and_record


_EAGER_EXPORTS = (
    "ExperimentResult",
    "ExperimentContext",
    "EXPERIMENT_FAMILIES",
    "SCRIPT_EXPERIMENTS",
    "default_script_output_dir",
    "load_result",
    "record_script_run",
    "run_and_record",
    "save_result",
)

__all__ = list(_EAGER_EXPORTS)
