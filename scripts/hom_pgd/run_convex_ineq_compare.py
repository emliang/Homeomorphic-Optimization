"""Editable Hom-PGD comparison on SOCP inequality instances."""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import (
    convex_problem_scale_label,
    make_benchmark_entrypoint,
    merge_params,
    script_family,
)
from homopt.experiments._labels import CONVEX_METHOD_LABELS
from homopt.experiments.hom_pgd import convex_algorithm_comparison


EXPERIMENT_NAME = "convex_ineq_compare"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = convex_algorithm_comparison
RUNTIME_SETTINGS = {
    "device": "auto",      # "cuda:0" | "cpu" | "auto"
    "dtype": "float32",    # "float32" | "float64"
}
BASE_PARAMS = {
    **RUNTIME_SETTINGS,
    # Problem family and algorithms.
    "problem_type": "socp",
    "algorithms": ["ALM", "RD", "Hom-PGD"],
    # Problem setup.
    "seed": 2025,
    "n_var": 100,
    "n_linear_cons": 0,
    "n_soc_cons": 10,
    "n_qua_cons": 0,
    "n_lin_eq": 0,
    "obj": "quad",
    "x_lower": -5.0,
    "x_upper": 5.0,
    # Shared optimization schedule.
    "max_iterations": 10000,
    "max_running_time": 600,
    "learning_rate": 1e-2,
    "stepsize_rule": "adaptive",
    "lr_decay": 0.999,
    "warm_start": False,
    "acceleration_method": "none",
    "acceleration_space": "z",
    "smooth": True,
    "momentum": 0.0,
    "visualize": True,
    "visualize_only": False,
    "visualization_prefix": None,
    "plot_algorithm_order": None,
    "method_labels": CONVEX_METHOD_LABELS,
    "reference_label": "MOSEK",
    "show_convergence_legend": True,
    # Optional explicit override blocks.
    "problem_config_overrides": {},
    "common_config": {
        "convergence_threshold": 1e-6,
        "max_iterations": 10000,
        "max_running_time": 600,
        "opt": "gd",
        "warm_start": False,
        "acceleration_method": "none",
        "acceleration_space": "z",
        "lr_decay": 0.999,
        "stepsize_rule": "adaptive",
        "smooth": True,
        "momentum": 0.0,
    },
    # Per-algorithm overrides.
    "algorithm_config": {
        "PGD": {
            "learning_rate": 1e-2,
            "projection_outer_iterations": 10,
            "projection_inner_iterations": 100,
        },
        "FW": {
            "learning_rate": 1e-2,
            "linearization_outer_iterations": 10,
            "linearization_inner_iterations": 100,
        },
        "Hom-PGD": {"learning_rate": 1e-3, "hom_p_norm": 2, "momentum": 0.99},
    },
}
QUICK_OVERRIDES = {
    # Edit here first for routine experiments.
    # "device": "cpu",
    # "dtype": "float64",
    # "n_var": 20,
    # "n_soc_cons": 20,
    # "max_iterations": 300,
    # "algorithm_config": {"Hom-PGD": {"learning_rate": 5e-4}},
}

# Edit this list to run several instance sizes in one process. Each entry is
# (scale_label, overrides). Use label=None to keep the saved run name tied to
# the actual problem dimensions, e.g. socp_n100_soc100.
INSTANCE_OVERRIDES = [
    (
        None,
        {
            "n_var": 10,
            "n_soc_cons": 5,
        },
    ),
    (
        None,
        {
            "n_var": 100,
            "n_soc_cons": 50,
        },
    ),
    (
        None,
        {
            "n_var": 1000,
            "n_soc_cons": 500,
        },
    ),
]
PARAMS = merge_params(BASE_PARAMS, QUICK_OVERRIDES)
run, main, OUTPUT_DIR = make_benchmark_entrypoint(
    EXPERIMENT_NAME,
    BENCHMARK,
    PARAMS,
    family=EXPERIMENT_FAMILY,
    instances=INSTANCE_OVERRIDES,
    label_builder=convex_problem_scale_label,
    default_params=PARAMS,
)


if __name__ == "__main__":
    main()
