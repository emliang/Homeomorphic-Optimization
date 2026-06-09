"""Editable Hom-PGD comparison on the MaxCut SDP formulation."""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, merge_params, script_family
from hom_pgd._presets import maxcut_result_label
from homopt.experiments.hom_pgd import maxcut_algorithm_comparison


EXPERIMENT_NAME = "maxcut_sdp_compare"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = maxcut_algorithm_comparison
RUNTIME_SETTINGS = {
    "device": "auto",      # "cuda:0" | "cpu" | "auto"
    "dtype": "float32",    # "float32" | "float64"
}
BASE_PARAMS = {
    **RUNTIME_SETTINGS,
    # Problem family and algorithms.
    "algorithms": ["PGD", "ALM-log", "ALM-bp", "RD", "Hom-PGD"],
    # Problem setup.
    "seed": 2025,
    "n": 10,
    "alpha": 0.5,
    # Shared optimization schedule.
    "max_iterations": 1000,
    "max_running_time": 600,
    "visualize": True,
    "visualize_only": False,
    "visualization_prefix": "maxcut_sdp",
    "reference_label": "SDP",
    "show_convergence_legend": True,
    "common_config": {
        "learning_rate": 1e-3,
        "convergence_threshold": 1e-6,
        "opt": "gd",
        "stepsize_rule": "adaptive",
        "warm_start": False,
        "acceleration_method": "none",
        "acceleration_space": "z",
        "smooth": True,
        "momentum": 0.0,
        "lr_decay": 0.999,
    },
    # Per-algorithm overrides.
    "algorithm_config": {
        "PGD": {
            "opt_type": "PGD",
            "projection_outer_iterations": 10,
            "projection_inner_iterations": 100,
        },
        "ALM": {"opt_type": "ALM", "outer_iterations": 200, "inner_iterations": 100, "dual_learning_rate": 0.1},
        "RD": {"opt_type": "RD"},
        "Hom-PGD": {"opt_type": "Hom-PGD", "learning_rate": 1e-3, "hom_p_norm": 2, "momentum": 0.99},
    },
}
QUICK_OVERRIDES = {
    # Edit here first for routine experiments.
    # "device": "cpu",
    # "dtype": "float64",
    # "n": 20,
    # "alpha": 0.7,
    # "max_iterations": 300,
    # "common_config": {"learning_rate": 5e-4},
}


def build_params():
    params = merge_params(BASE_PARAMS, QUICK_OVERRIDES)
    params["result_label"] = maxcut_result_label(params)
    return params


PARAMS = build_params()
run, main, OUTPUT_DIR = make_benchmark_entrypoint(
    EXPERIMENT_NAME,
    BENCHMARK,
    PARAMS,
    family=EXPERIMENT_FAMILY,
)


if __name__ == "__main__":
    main()
