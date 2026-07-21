"""2D Hom-ALM visualization on SOCP-EQ instances."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, merge_params, script_family
from hom_alm._configs import convex_eq_2d_config
from homopt.experiments.hom_alm import TOY_PROBLEM_OVERRIDES, convex_eq_2d_benchmark


EXPERIMENT_NAME = "convex_eq_2d"
EXPERIMENT_FAMILY = script_family(__file__)

BASE_PARAMS = convex_eq_2d_config(TOY_PROBLEM_OVERRIDES)

QUICK_OVERRIDES = {
    # Routine edit point:
    # "common_config": {"max_iterations": 120},
    # "algorithm_config": {"Hom-ALM": {"inner_solver": "prox_gd", "inner_proximal_coef": 1e-3}},
}

PARAMS = merge_params(BASE_PARAMS, QUICK_OVERRIDES)

BENCHMARK = convex_eq_2d_benchmark
run, main, OUTPUT_DIR = make_benchmark_entrypoint(
    EXPERIMENT_NAME,
    BENCHMARK,
    PARAMS,
    family=EXPERIMENT_FAMILY,
)


if __name__ == "__main__":
    main()

