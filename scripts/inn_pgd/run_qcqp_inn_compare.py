"""High-dimensional INN-PGD comparison for nonconvex QCQP inequalities."""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, script_family
from inn_pgd._configs import build_inn_script_params, qcqp_inn_pgd_config, qcqp_ipopt_reference_config, qcqp_result_label
from homopt.experiments.inn_pgd import qcqp_inn_experiment


EXPERIMENT_NAME = "qcqp_inn_compare"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = qcqp_inn_experiment
BASE_PARAMS = qcqp_inn_pgd_config(qcqp_ipopt_reference_config(), {
    "device": "cuda:0",      # "cuda:0" | "cpu" | "auto"
    "dtype": "float32",      # "float32" | "float64"
    "seed": 2025,
    "retrain": True,  # False -> reuse existing inn_mapping.pt if present
    "n_var": 50,
    "n_qua_cons": 100,
    "num_test_instance": 3,  # number of sampled test instances for comparison
    "max_iterations": 500,
    # High-dimensional sweep mode.
    "run_high_dim_sweep": True,
    "high_dim_sweep_cases": [
        [10, 10],
        [10, 100],
        [10, 1000],
        # [30, 10],
        # [30, 100],
        # [30, 1000],
        # [50, 10],
        # [50, 100],
        # [50, 1000],
        # [100, 10],
        # [100, 100],
        # [100, 1000],
    ],
    "high_dim_max_iter_factor": 10,  # max_iterations = n_var * factor
    # Fixed-instance ALM/Penalty baselines live in the 2D diagnostic script;
    # high-dimensional compare focuses on INN-PGD vs IPOPT.
    "lagrangian_baselines": [],
})
QUICK_OVERRIDES = {
    # Edit here first for routine experiments.
    # "device": "cpu",
    # "dtype": "float64",
    # "seed": 2026,
    # "retrain": True,
    # Disable sweep and run a single high-dimensional case:
    # "run_high_dim_sweep": False,
    # High-dimensional sweep:
    # "high_dim_sweep_cases": [[10, 100], [30, 100], [50, 100]],
    # "high_dim_max_iter_factor": 10,
    # Single-case manual override:
    # "n_var": 50, "n_qua_cons": 100, "max_iterations": 300,
    # "num_test_instance": 8,
    # "train_config": {"n_samples": 64, "total_iteration": 40},
    # "max_iterations": 50,
    # "model_config": {"lr": 5e-4},
    # "optimizer_config": {"initial_latent_mode": "sphere_boundary_random", "learning_rate": 5e-3},
}


def build_params():
    """Load experiment config in order: base high-dimensional setup -> local overrides."""

    return build_inn_script_params(BASE_PARAMS, QUICK_OVERRIDES, result_label=qcqp_result_label)


PARAMS = build_params()
run, main, OUTPUT_DIR = make_benchmark_entrypoint(
    EXPERIMENT_NAME,
    BENCHMARK,
    PARAMS,
    family=EXPERIMENT_FAMILY,
)


if __name__ == "__main__":
    main()
