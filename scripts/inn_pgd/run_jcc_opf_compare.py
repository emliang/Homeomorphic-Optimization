"""Editable JCC/OPF INN-PGD comparison."""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, script_family
from inn_pgd._configs import build_inn_script_params, jcc_inn_pgd_config, jcc_result_label
from homopt.experiments.inn_pgd import jcc_algorithm_comparison


EXPERIMENT_NAME = "jcc_opf_compare"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = jcc_algorithm_comparison
BASE_PARAMS = jcc_inn_pgd_config({
    "device": "cuda:0",      # "cuda:0" | "cpu" | "auto"
    "num_bus": 57,
    "n_scenarios": 10,
    "epsilon": 0.1,
    "demand_std": 0.05,
    "seed": 2025,
    # The 30-bus case is a solver/baseline smoke run after active-variable
    # reduction. Use 57-bus or larger when enabling INN-PGD.
    "run_inn_pgd": True,
    "max_iterations": 300,
    "num_test_instance": 1,
    "model_config": {
        "Con_type": "PI",
    },
    "solver_configs": {
        "mixed_integer": {
            "solver": "GUROBI",
            "M": 1000.0,
            "verbose": False,
            "time_limit_sec": 600.0,
        },
        "cvar": {
            "solver": "GUROBI",
            "verbose": False,
            "time_limit_sec": 600.0,
        },
        "scenario": {
            "solver": "GUROBI",
            "verbose": False,
            "time_limit_sec": 600.0,
        },
    },
    "lagrangian_baselines": [], # "ALM", "Penalty", "Prox-Penalty"
})
QUICK_OVERRIDES = {
    # Edit here first for routine experiments.
    # "device": "cpu",
    # "dtype": "float64",
    # "n_scenarios": 10,
    # "seed": 2026,
    # "num_bus": 57,
    # "run_inn_pgd": True,
    # "retrain": True,
    # "num_test_instance": 3,
    # "train_config": {"total_iteration": 2000, "batch_size": 128},
    # "max_iterations": 100,
    # "optimizer_config": {"learning_rate": 5e-2},
    # "optimizer_config": {"initial_latent_mode": "sphere_boundary_random"},
    # "lagrangian_baseline_config": {"learning_rate": 5e-4},
    # "problem_config": {"epsilon": 0.15},
}


def build_params():
    """Load experiment config in order: base JCC setup -> local overrides."""

    return build_inn_script_params(BASE_PARAMS, QUICK_OVERRIDES, result_label=jcc_result_label)


PARAMS = build_params()
run, main, OUTPUT_DIR = make_benchmark_entrypoint(
    EXPERIMENT_NAME,
    BENCHMARK,
    PARAMS,
    family=EXPERIMENT_FAMILY,
)


if __name__ == "__main__":
    main()
