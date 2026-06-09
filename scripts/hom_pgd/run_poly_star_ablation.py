"""Editable Hom-PGD ablation on the poly-star inequality toy problem."""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, merge_params, script_family
from hom_pgd._presets import poly_star_base_params, poly_star_result_label
from homopt.experiments.hom_pgd import poly_star_benchmark


EXPERIMENT_NAME = "poly_star_ablation"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = poly_star_benchmark
BASE_PARAMS = poly_star_base_params(
    problem_type="star",
    algorithms=["Hom-PGD"],
    alpha=0.4,
    max_iterations=10000,
    convergence_threshold=1e-6,
    momentum=0.9,
)
QUICK_OVERRIDES = {
    # Edit here first for routine experiments.
    # "device": "cpu",
    # "dtype": "float64",
    # "problem_type": "poly_star",
    # "alpha": 0.3,
    # "num_star": 6,
    # "max_iterations": 300,
    # "algorithm_config": {"Hom-PGD": {"momentum": 0.95}},
}
PARAMS = merge_params(BASE_PARAMS, QUICK_OVERRIDES)
PARAMS["result_label"] = poly_star_result_label(PARAMS)
run, main, OUTPUT_DIR = make_benchmark_entrypoint(
    EXPERIMENT_NAME,
    BENCHMARK,
    PARAMS,
    family=EXPERIMENT_FAMILY,
)


if __name__ == "__main__":
    main()
