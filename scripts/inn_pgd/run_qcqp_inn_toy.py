"""2D INN-PGD training and visualization artifacts for QCQP."""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, merge_params, script_family
from inn_pgd._configs import build_inn_script_params, qcqp_inn_pgd_config, qcqp_ipopt_reference_config, qcqp_result_label
from homopt.experiments.inn_pgd import qcqp_inn_experiment


EXPERIMENT_NAME = "qcqp_inn_toy"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = qcqp_inn_experiment
INN_ONLY = False
BASE_PARAMS = qcqp_inn_pgd_config({
    "device": "cuda:0",      # "cuda:0" | "cpu" | "auto"
    "n_var": 2,
    "n_qua_cons": 3,
    "max_iterations": 500,
    "model_config": {
        "latent_init_shape": "sphere",  # "auto" | "sphere" | "cube"
    },
    "train_config": {
        "batch_size": 256,
    },
    "optimizer_config": {
        "initial_latent_mode": "sphere_boundary_random",  # "center" | "sphere_boundary_random"
        "proj_max_steps": 20,
    },
    "visualize": True,
    "visualize_instance_idx": [0],  # int, list[int], or "all"; selected test instances for 2D plots
    "violation_y_min": 1e-5,
    "plot_objective_gap": False,
    "plot_gap_plus_violation": False,
    # Notebook-style 2D mapping visualization (from notebooks/qcqp_ablation_analysis).
    "visualize_mdh_mapping": True,
})
BASELINE_COMPARE_PARAMS = {
    **qcqp_ipopt_reference_config(),
    "lagrangian_baselines": ["ALM", "Penalty", "Prox-Penalty"],
}
QUICK_OVERRIDES = {
    # Edit here first.
    # "device": "cpu",
    # "dtype": "float64",
    # "seed": 2026,
    "retrain": True,
    "num_test_instance": 3,
    # "train_config": {"n_samples": 64, "total_iteration": 40},
    # "max_iterations": 50,
    # "model_config": {"lr": 5e-4, "num_layer": 3},
    # Boundary-start INN-PGD:
    # "optimizer_config": {"initial_latent_mode": "sphere_boundary_random"},
    # "optimizer_config": {"learning_rate": 5e-3, "max_running_time": 30},
}


def build_params():
    """Load experiment config in order: base 2D setup -> optional baselines -> local overrides."""

    params = BASE_PARAMS if INN_ONLY else merge_params(BASE_PARAMS, BASELINE_COMPARE_PARAMS)
    suffix = "inn_only" if INN_ONLY else "baseline_compare"
    return build_inn_script_params(
        params,
        QUICK_OVERRIDES,
        result_label=lambda current: qcqp_result_label(current, suffix=suffix),
    )


PARAMS = build_params()
run, main, OUTPUT_DIR = make_benchmark_entrypoint(
    EXPERIMENT_NAME,
    BENCHMARK,
    PARAMS,
    family=EXPERIMENT_FAMILY,
)


if __name__ == "__main__":
    main()
