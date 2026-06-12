"""Editable Hom-PGD comparison on the MaxCut SDP formulation."""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, merge_params, script_family
from hom_pgd._configs import MAXCUT_SDP_COMPARE_CONFIG, maxcut_result_label
from homopt.experiments.hom_pgd import maxcut_algorithm_comparison


EXPERIMENT_NAME = "maxcut_sdp_compare"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = maxcut_algorithm_comparison
BASE_PARAMS = MAXCUT_SDP_COMPARE_CONFIG
QUICK_OVERRIDES = {
    # Edit here first for routine experiments.
    # "device": "cpu",
    # "dtype": "float64",
    "n": 10,
    "alpha": 0.5,
    # "common_config": {"max_iterations": 300},
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
