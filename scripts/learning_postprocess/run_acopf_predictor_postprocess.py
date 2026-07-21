"""Editable AC-OPF predictor + post-processing comparison."""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, merge_params, script_family
from learning_postprocess._configs import (
    acopf_parametric_learning_postprocess_config,
    acopf_parametric_result_label,
)
from homopt.experiments.learning_postprocess import acopf_parametric_learning_benchmark


EXPERIMENT_NAME = "acopf_predictor_postprocess"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = acopf_parametric_learning_benchmark
BASE_PARAMS = acopf_parametric_learning_postprocess_config({
    "seed": 2026,
    "case": 57,
    "dataset_samples": 10000,
    "dataset_path": None,
    "n_samples": 32,
    "predictor_type": "nn",
    "prediction_value": 0.0,
    "baselines": [
        "predict_only",
        "predict_diff_projection",
    ],
    "retrain": False,
})
QUICK_OVERRIDES = {
    # Edit here first.
    # "device": "cuda:0",
    # "predictor_type": "constant",
    # "baselines": ["predict_only"],
    # "retrain": True,
    # "case": 30,
    # "dataset_samples": 10000,
    # "n_samples": 128,
}


def build_params():
    params = merge_params(BASE_PARAMS, QUICK_OVERRIDES)
    params["result_label"] = acopf_parametric_result_label(params)
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
