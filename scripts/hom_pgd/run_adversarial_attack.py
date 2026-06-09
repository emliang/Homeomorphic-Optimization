"""Editable Hom-PGD adversarial attack experiment.

Simplified runtime chain:
1) Ensure one shared base model checkpoint exists (train once if missing).
2) Run either a single attack config or a norm sweep on a shared eval subset.
"""

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import make_benchmark_entrypoint, merge_params, script_family
from hom_pgd._presets import adversarial_result_label
from homopt.experiments.hom_pgd import adversarial_attack_experiment


EXPERIMENT_NAME = "adversarial_attack"
EXPERIMENT_FAMILY = script_family(__file__)
BENCHMARK = adversarial_attack_experiment


# ---- Base config ----
RUNTIME = {
    "device": "auto",  # "cuda:0" | "cpu" | "auto"
    "seed": 42,
    "download": True,
}

WORKFLOW = {
    "dataset_name": "cifar10",
    "train": False,
    "attack": True,
    "visualize": True,
    "train_if_missing": True,
}

PATHS = {
    "data_root": "data",
    "checkpoint_dir": "data",
}

TRAINING = {
    "training_epochs": 100,
    "training_lr": 1e-3,
    "batch_size": 256,
    "verbose": True,
    "train_log_every": 10,
}

EVAL = {
    "attack_selection_size": 500,
    "visualize_examples": 2,
}

# Weighted-norm geometry config (clear and centrally editable).
WEIGHTED_NORM = {
    "enabled": True,
    # "identity" | "random_rank1" | "diag" | "matrix" | "channel_corr" | "spatial_smooth" | "channel_spatial"
    "mode": "channel_spatial",
    # Channel correlation (used by "channel_corr" and "channel_spatial")
    "channel_corr": [
        [1.0, 0.4, 0.3],
        [0.4, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ],
    "corr_jitter": 1e-4,
    # Spatial smoothing (used by "spatial_smooth" and "channel_spatial")
    "spatial_sigma": 1.2,
    "spatial_kernel_size": 7,  # 0 -> auto from sigma
    # Legacy/random modes
    "weight_scale": 1.0,
}

ATTACK = {
    "eps": 1 / 255,
    "alpha": 1e-3,
    "iters": 500,
    "optimizer": "adam",  # "gd" | "adam"
    "norm": "inf",  # "inf" | 2 | 4 | ...
    "weighted_norm": bool(WEIGHTED_NORM["enabled"]),
    "weight_mode": WEIGHTED_NORM["mode"],
    "channel_corr": WEIGHTED_NORM["channel_corr"],
    "corr_jitter": WEIGHTED_NORM["corr_jitter"],
    "spatial_sigma": WEIGHTED_NORM["spatial_sigma"],
    "spatial_kernel_size": WEIGHTED_NORM["spatial_kernel_size"],
    "weight_scale": WEIGHTED_NORM["weight_scale"],
}

NORM_EPS_SWEEP = [
    (0.5, 100000/ 255),
    (1, 50/ 255),
    (2, 5/ 255),
    (4, 1/ 255),
    ("inf", 1/ 255),
]

AUTO_EPS = {
    "enabled": False,
    "force_override": False,  # True -> always overwrite attack_overrides["eps"] when enabled
    "reference_eps_linf": 100 / 255,
    "min_eps": 1e-6,
    "max_eps": None,  # None -> dimension-aware cap
}

BASE_PARAMS = {
    **RUNTIME,
    **WORKFLOW,
    **PATHS,
    **TRAINING,
    **EVAL,
    "attack_overrides": ATTACK,
}

QUICK_OVERRIDES = {
    # "device": "cpu",
    # "train": False,
    # "verbose": False,
    # "attack_overrides": {"norm": 2, "weighted_norm": True, "weight_mode": "random_rank1", "weight_scale": 0.02},
}


def build_params():
    params = merge_params(BASE_PARAMS, QUICK_OVERRIDES)
    params["norm_eps_sweep"] = NORM_EPS_SWEEP
    params["auto_eps"] = AUTO_EPS
    params["result_label"] = adversarial_result_label(params)
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
