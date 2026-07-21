"""Editable script entrypoint for mixed Stiefel equality/convex-side tests."""

from __future__ import annotations

from pathlib import Path
import sys
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import (
    make_benchmark_entrypoint,
    merge_params,
    scale_label,
    script_family,
)
from hom_alm._configs import (
    INTERMEDIATE_LFW_OVERRIDES,
    LARGE_MNIST_OVERRIDES,
    MEDIUM_LFW_OVERRIDES,
    MEDIUM_OLIVETTI_OVERRIDES,
    SMALL_DIGITS_OVERRIDES,
    STIEFEL_EQ_COMPARE_CONFIG,
)
from homopt.experiments.hom_alm import stiefel_algorithm_comparison

EXPERIMENT_NAME = "stiefel_eq_compare"
EXPERIMENT_FAMILY = script_family(__file__)
BASE_PARAMS = STIEFEL_EQ_COMPARE_CONFIG

# Select the active scale. These overrides only change BASE_PARAMS["problem_config"].
# Choices: SMALL_DIGITS_OVERRIDES, INTERMEDIATE_LFW_OVERRIDES,
# MEDIUM_LFW_OVERRIDES, MEDIUM_OLIVETTI_OVERRIDES, LARGE_MNIST_OVERRIDES.
QUICK_OVERRIDES = SMALL_DIGITS_OVERRIDES
INSTANCE_OVERRIDES = [
    ("small_digits", SMALL_DIGITS_OVERRIDES),
    ("intermediate_lfw", INTERMEDIATE_LFW_OVERRIDES),
    # ("medium_olivetti", MEDIUM_OLIVETTI_OVERRIDES),
    ("medium_lfw", MEDIUM_LFW_OVERRIDES),
    # ("large_mnist", LARGE_MNIST_OVERRIDES),
]

PARAMS = merge_params(BASE_PARAMS, QUICK_OVERRIDES)
BENCHMARK = stiefel_algorithm_comparison
run, main, OUTPUT_DIR = make_benchmark_entrypoint(
    EXPERIMENT_NAME,
    BENCHMARK,
    PARAMS,
    family=EXPERIMENT_FAMILY,
    instances=INSTANCE_OVERRIDES,
    label_builder=scale_label,
)


if __name__ == "__main__":
    main()
