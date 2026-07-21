"""Network/optimizer sensitivity sweep for INN-PGD on nonconvex QCQP."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, script_family
from inn_pgd._configs import build_inn_script_params, qcqp_inn_pgd_config, qcqp_result_label
from homopt.experiments.inn_pgd import qcqp_inn_sensitivity_sweep


EXPERIMENT_NAME = "qcqp_inn_ablation"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = qcqp_inn_sensitivity_sweep
BASE_PARAMS = qcqp_inn_pgd_config({
    "device": "auto",      # "cuda:0" | "cpu" | "auto"
    "n_var": 2,
    "n_qua_cons": 3,
    "max_iterations": 500,
    "visualize": True,
    "visualize_instance_idx": [0, 1],
    # Per-case plots keep training/constraint/trajectory diagnostics; convergence
    # comparisons are drawn once per sweep by the sensitivity overlay.
    "plot_case_convergence": False,
    "visualize_mdh_mapping": True,
})
QUICK_OVERRIDES = {
    # Fast smoke profile:
    # "num_test_instance": 3,
    # "train_config": {"n_samples": 64, "total_iteration": 40},
    # "max_iterations": 50,
    # "retrain": True,
}
SENSITIVITY_SWEEPS = {
    # Model architecture sweeps: generate training diagnostics and selected
    # test-instance constraint/trajectory visualizations for 1, 3, and 5 layers.
    "num_layer": [1, 3, 5],
    # Optimizer tolerance sweeps under the default INN architecture.
    "feasibility_eps": [1e-5, 1e-3, 1e-1],
    # Optional broader sweeps:
    # "h_dim": [32, 64, 128],
    # "w_distortion": [0.01, 0.1, 1.0],
    # "opt": ["gd", "adam"],
}


def build_params():
    """Load experiment config in order: base sensitivity setup -> sweep grid -> local overrides."""

    return build_inn_script_params(
        BASE_PARAMS,
        QUICK_OVERRIDES,
        extra_params={"sensitivity_sweeps": SENSITIVITY_SWEEPS},
        result_label=lambda params: qcqp_result_label(params, suffix="ablation"),
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
