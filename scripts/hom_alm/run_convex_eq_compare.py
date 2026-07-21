"""Editable Hom-ALM comparison on SOCP-EQ instances."""

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
from hom_alm._configs import CONVEX_EQ_COMPARE_CONFIG
from homopt.experiments.hom_alm import convex_algorithm_comparison


EXPERIMENT_NAME = "convex_eq_compare"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = convex_algorithm_comparison
BASE_PARAMS = CONVEX_EQ_COMPARE_CONFIG
QUICK_OVERRIDES = {
    # "device": "cpu",
    # "dtype": "float64",
    # "common_config": {"max_iterations": 200},
    # "inner_solver_common": {"inner_iterations": 50},
    # "algorithms": ["Hom-ALM"],
}

# Edit this list to run several instance sizes in one process. Each entry is
# (scale_label, overrides). Use label=None to keep the saved run name tied to
# the actual problem dimensions, e.g. socp_eq_n1000_q100_eq900.
INSTANCE_OVERRIDES = []
# INSTANCE_OVERRIDES = [
#     (None, {"n_var": 500, "n_qua_cons": 100, "n_lin_eq": 100}),
#     (None, {"n_var": 500, "n_qua_cons": 100, "n_lin_eq": 400}),
#     (None, {"n_var": 1000, "n_qua_cons": 100, "n_lin_eq": 100}),
#     (None, {"n_var": 1000, "n_qua_cons": 100, "n_lin_eq": 500}),
#     (None, {"n_var": 1000, "n_qua_cons": 100, "n_lin_eq": 900}),
#     (None, {"n_var": 1000, "n_qua_cons": 500, "n_lin_eq": 100}),
#     (None, {"n_var": 1000, "n_qua_cons": 500, "n_lin_eq": 500}),
#     (None, {"n_var": 1000, "n_qua_cons": 500, "n_lin_eq": 900}),
# ]
PARAMS = merge_params(BASE_PARAMS, QUICK_OVERRIDES)
run, main, OUTPUT_DIR = make_benchmark_entrypoint(
    EXPERIMENT_NAME,
    BENCHMARK,
    PARAMS,
    family=EXPERIMENT_FAMILY,
    instances=INSTANCE_OVERRIDES,
    label_builder=convex_problem_scale_label,
)


if __name__ == "__main__":
    main()
