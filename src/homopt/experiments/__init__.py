"""Experiment helpers for script-only runs."""

from __future__ import annotations

from importlib import import_module

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

_LAZY_EXPORTS = {
    "adversarial_attack_experiment": ("homopt.experiments.hom_pgd", "adversarial_attack_experiment"),
    "adversarial_attack_workflow": ("homopt.experiments.hom_pgd", "adversarial_attack_workflow"),
    "convex_algorithm_comparison": ("homopt.experiments.hom_alm", "convex_algorithm_comparison"),
    "convex_problem_summary": ("homopt.experiments.examples", "convex_problem_summary"),
    "inn_training_benchmark": ("homopt.experiments.inn_pgd", "inn_training_benchmark"),
    "jcc_algorithm_comparison": ("homopt.experiments.hom_alm", "jcc_algorithm_comparison"),
    "jcc_baseline_solver_sweep": ("homopt.experiments.hom_alm", "jcc_baseline_solver_sweep"),
    "jcc_linear_solver_benchmark": ("homopt.experiments.inn_pgd", "jcc_linear_solver_benchmark"),
    "jcc_problem_benchmark": ("homopt.experiments.inn_pgd", "jcc_problem_benchmark"),
    "learning_route_summary": ("homopt.experiments.examples", "learning_route_summary"),
    "maxcut_algorithm_comparison": ("homopt.experiments.hom_pgd", "maxcut_algorithm_comparison"),
    "poly_star_benchmark": ("homopt.experiments.hom_pgd", "poly_star_benchmark"),
    "qcqp_inn_experiment": ("homopt.experiments.inn_pgd", "qcqp_inn_experiment"),
    "qcqp_inn_sensitivity_sweep": ("homopt.experiments.inn_pgd", "qcqp_inn_sensitivity_sweep"),
    "qcqp_learning_benchmark": ("homopt.experiments.learning_postprocess", "qcqp_learning_benchmark"),
    "qcqp_learning_route_summary": ("homopt.experiments.examples", "qcqp_learning_route_summary"),
    "qcqp_route_comparison": ("homopt.experiments.learning_postprocess", "qcqp_route_comparison"),
    "socp_hompgd_benchmark": ("homopt.experiments.hom_pgd", "socp_hompgd_benchmark"),
    "toy_star_summary": ("homopt.experiments.examples", "toy_star_summary"),
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
