"""Editable Hom-PGD comparison on SOCP inequality instances."""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import convex_problem_scale_label, make_benchmark_entrypoint, merge_params, script_family
from hom_pgd._configs import CONVEX_INEQ_COMPARE_CONFIG
from homopt.experiments.common.labels import CONVEX_METHOD_LABELS
from homopt.experiments.hom_pgd import convex_algorithm_comparison


EXPERIMENT_NAME = "convex_ineq_compare"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = convex_algorithm_comparison
BASE_PARAMS = merge_params(CONVEX_INEQ_COMPARE_CONFIG, {"method_labels": CONVEX_METHOD_LABELS})
QUICK_OVERRIDES = {
    # Edit here first for routine experiments.
    # "device": "cpu",
    # "dtype": "float64",
    # "n_var": 20,
    # "n_soc_cons": 20,
    # "common_config": {"max_iterations": 300},
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
            "n_soc_cons": 10,
        },
    ),
    # (
    #     None,
    #     {
    #         "n_var": 100,
    #         "n_soc_cons": 50,
    #     },
    # ),
    # (
    #     None,
    #     {
    #         "n_var": 1000,
    #         "n_soc_cons": 50,
    #     },
    # ),
]
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
