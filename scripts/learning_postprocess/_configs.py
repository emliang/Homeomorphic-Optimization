"""Shared configs for predictor/post-processing scripts."""

from pathlib import Path

from _common import merge_param_layers, sanitize_label


RUNTIME_CONFIG = {
    "device": "cpu",
    "dtype": "float32",
}

ACOPF_PARAMETRIC_PROBLEM_CONFIG = {
    "case": 57,
    "dataset_samples": 10000,
    "dataset_path": None,
    "load_std": 0.05,
    "pglib_data_dir": "data/pglib-opf",
    "pglib_case_name": None,
    "download_if_missing": True,
    "branch_constraints": True,
    "enforce_slack_angle": True,
    "problem_config": {
        "prediction_space": "pf_partial",
        "completion_config": {
            "gradient_mode": "implicit",
            "max_iters": 8,
            "tol": 1e-6,
            "damping": 1e-5,
            "allocation": "soft_cost",
            "soft_tau": 1.0,
        },
    },
}


ACOPF_PARAMETRIC_PREDICTOR_CONFIG = {
    "predictor_model_config": {
        "h_dim": 128,
        "num_layer": 3,
        "outact": "tanh",
        "dropout": 0.05,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "lr_decay": 0.9,
        "lr_decay_step": 1000,
        "w_ineq": 1e-3,
        "w_obj": 1e-2,
        "approach": "unsupervise",
        "resultsSaveFreq": 1000,
    },
    "predictor_train_config": {
        "n_samples": 10000,
        "batch_size": 64,
        "total_iteration": 10000,
        "pre_training": 0,
    },
}


ACOPF_PARAMETRIC_POSTPROCESS_CONFIG = {
    "projection_config": {
        "proj_max_steps": 50,
        "corr_lr": 1e-4,
        "corr_momentum": 0.5,
        "proj_eps": 1e-6,
    },
}


CONVEX_PARAMETRIC_PROBLEM_CONFIG = {
    "problem_config": {
        "y_lower": -1.0,
        "y_upper": 1.0,
        "input_lower": -1.0,
        "input_upper": 1.0,
    },
}


CONVEX_PARAMETRIC_PREDICTOR_CONFIG = {
    "predictor_model_config": {
        "h_dim": 64,
        "num_layer": 3,
        "outact": "tanh",
        "dropout": 0.1,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "lr_decay": 0.9,
        "lr_decay_step": 1000,
        "w_ineq": 1e-3,
        "w_obj": 1e-2,
        "approach": "unsupervise",
        "resultsSaveFreq": 1000,
    },
    "predictor_train_config": {
        "n_samples": 10000,
        "batch_size": 64,
        "total_iteration": 10000,
        "pre_training": 0,
    },
}


CONVEX_PARAMETRIC_POSTPROCESS_CONFIG = {
    "projection_config": {
        "proj_max_steps": 50,
        "corr_lr": 1e-3,
        "corr_momentum": 0.5,
        "proj_eps": 1e-6,
    },
}


def acopf_parametric_result_label(params):
    dataset_path = params.get("dataset_path")
    if dataset_path:
        stem = Path(dataset_path).name
    else:
        stem = f"case{int(params['case'])}_samples{int(params['dataset_samples'])}"
    return sanitize_label(
        f"acopf_{stem}_eval{int(params['n_samples'])}",
        fallback="acopf_parametric",
    )


def acopf_parametric_learning_postprocess_config(*overrides):
    return merge_param_layers(
        RUNTIME_CONFIG,
        ACOPF_PARAMETRIC_PROBLEM_CONFIG,
        ACOPF_PARAMETRIC_PREDICTOR_CONFIG,
        ACOPF_PARAMETRIC_POSTPROCESS_CONFIG,
        *overrides,
    )


def convex_parametric_result_label(params):
    problem_type = str(params["problem_type"])
    if problem_type == "sdp":
        size = f"m{int(params['matrix_dim'])}"
    else:
        size = f"n{int(params['n_var'])}"
    return sanitize_label(
        f"{problem_type}_{size}_eq{int(params['n_eq'])}_ineq{int(params['n_ineq'])}_samples{int(params['n_samples'])}",
        fallback="convex_parametric",
    )


def convex_parametric_learning_postprocess_config(*overrides):
    return merge_param_layers(
        RUNTIME_CONFIG,
        CONVEX_PARAMETRIC_PROBLEM_CONFIG,
        CONVEX_PARAMETRIC_PREDICTOR_CONFIG,
        CONVEX_PARAMETRIC_POSTPROCESS_CONFIG,
        *overrides,
    )


__all__ = [
    "ACOPF_PARAMETRIC_POSTPROCESS_CONFIG",
    "ACOPF_PARAMETRIC_PREDICTOR_CONFIG",
    "ACOPF_PARAMETRIC_PROBLEM_CONFIG",
    "CONVEX_PARAMETRIC_POSTPROCESS_CONFIG",
    "CONVEX_PARAMETRIC_PREDICTOR_CONFIG",
    "CONVEX_PARAMETRIC_PROBLEM_CONFIG",
    "RUNTIME_CONFIG",
    "acopf_parametric_learning_postprocess_config",
    "acopf_parametric_result_label",
    "convex_parametric_learning_postprocess_config",
    "convex_parametric_result_label",
]
