"""Editable Hom-PGD comparison on the poly-star inequality toy problem."""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, merge_params, script_family
from hom_pgd._presets import poly_star_base_params, poly_star_result_label
from homopt.experiments.hom_pgd import poly_star_benchmark


EXPERIMENT_NAME = "poly_star_compare"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = poly_star_benchmark
BASE_PARAMS = poly_star_base_params(
    problem_type="star",
    algorithms=["PGD", "Hom-PGD"],
    alpha=0.6,
    max_iterations=1000,
    convergence_threshold=1e-5,
    momentum=0.0,
)
QUICK_OVERRIDES = {
    # Edit here first for routine experiments.
    "device": "cpu",
    # "dtype": "float64",
    "seed": 2026,
    "problem_type": "star",
    "max_iterations": 1000,
    "algorithms": ["Hom-PGD", "PGD", "ALM"],
    "algorithm_config": {"Hom-PGD": {"learning_rate": 5e-3}, 
                        "PGD": {"learning_rate": 5e-3}, 
                        "ALM": {"learning_rate": 5e-3}},
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
