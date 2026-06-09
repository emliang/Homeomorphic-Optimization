"""Hom-PGD experiment helpers for script-only runs."""

from __future__ import annotations

from importlib import import_module

from .context import ExperimentContext
from .entrypoints import default_script_output_dir, record_script_run
from .results import ExperimentResult, load_result, save_result
from .runner import run_and_record


_EAGER_EXPORTS = (
    "ExperimentResult",
    "ExperimentContext",
    "default_script_output_dir",
    "load_result",
    "record_script_run",
    "run_and_record",
    "save_result",
)

_LAZY_EXPORTS = {
    "adversarial_attack_experiment": ("homopt.experiments.hom_pgd", "adversarial_attack_experiment"),
    "adversarial_attack_workflow": ("homopt.experiments.hom_pgd", "adversarial_attack_workflow"),
    "convex_algorithm_comparison": ("homopt.experiments.hom_pgd", "convex_algorithm_comparison"),
    "maxcut_algorithm_comparison": ("homopt.experiments.hom_pgd", "maxcut_algorithm_comparison"),
    "poly_star_benchmark": ("homopt.experiments.hom_pgd", "poly_star_benchmark"),
    "socp_hompgd_benchmark": ("homopt.experiments.hom_pgd", "socp_hompgd_benchmark"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [*_EAGER_EXPORTS, *sorted(_LAZY_EXPORTS)]
