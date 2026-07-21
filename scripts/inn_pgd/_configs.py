"""Shared INN-PGD script configs and merge helpers."""

from _common import merge_param_layers, merge_params


def _label_number(value):
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace(".", "p").replace("-", "m")


def inn_pgd_common_config():
    return {
        "device": "auto",
        "dtype": "float32",
        "seed": 2030,
        "retrain": False,
        "num_test_instance": 10,
        "model_config": {
            "num_layer": 3,
            "inv_type": "made",
            "bilip": False,
            "L": 2,
            "h_dim": 64,
            "w_penalty": 10.0,
            "w_distortion": 0.01,
            "w_lipschitz": 0.1,
            "center": 0.0,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "lr_decay_step": 100,
            "lr_decay": 0.99,
            "ema_decay": 0.99,
            "resultsSaveFreq": 1000,
            "use_amp": False,
            "latent_init_shape": "auto",
            "cond_embed_mode": "shared",
        },
        "train_config": {
            "n_samples": 10000,
            "batch_size": 64,
            "total_iteration": 10000,
        },
        "optimizer_config": {
            "initial_latent_mode": "center",
            "initial_latent_radius": 1.0,
            "learning_rate": 0.1,
            "max_running_time": 600,
            "convergence_threshold": 1e-5,
            "lr_decay": 0.999,
            "min_lr": 1e-6,
            "verbose_interval": 50,
            "step_size": 0.5,
            "proj_max_steps": 10,
            "feasibility_eps": 1e-5,
            "opt": "gd",
            "momentum": 0.0,
            "stepsize_rule": "adaptive",
        },
        "lagrangian_baselines": [],
        "lagrangian_baseline_config": {
            "inner_iterations": 100,
            "learning_rate": 0.1,
            "inner_learning_rate": 0.001,
            "dual_learning_rate": 1.0,
            "lr_decay": 0.999,
            "inner_lr_decay": 0.99,
            "min_lr": 1e-6,
            "inner_min_lr": 1e-6,
            "penalty_coef": 1.0,
            "penalty_growth": 1.1,
            "proximal_coef": 5.0,
            "max_penalty": 1000.0,
            "max_dual": 100.0,
            "max_running_time": 600,
            "convergence_threshold": 1e-5,
            "stepsize_rule": "adaptive",
            "inner_stepsize_rule": "constant",
            "inner_solver": "gd",
            "lagrangian_gradient": "explicit",
            "proximal_space": "x",
            "opt": "gd",
            "momentum": 0.0,
            "verbose": False,
            "verbose_interval": 50,
        },
    }


def inn_pgd_config(*overrides):
    return merge_param_layers(inn_pgd_common_config(), *overrides)


def build_inn_script_params(base_params, quick_overrides=None, *, extra_params=None, result_label=None):
    params = merge_param_layers(base_params, extra_params or {}, quick_overrides or {})
    if result_label is not None:
        params["result_label"] = result_label(params) if callable(result_label) else result_label
    return params


def qcqp_result_label(params, *, suffix=None):
    if params.get("run_high_dim_sweep"):
        cases = params.get("high_dim_sweep_cases") or []
        case_labels = []
        for case in cases:
            if isinstance(case, dict):
                n_var = case.get("n_var")
                n_qua_cons = case.get("n_qua_cons")
            else:
                n_var, n_qua_cons = case[:2]
            case_labels.append(f"n{int(n_var)}q{int(n_qua_cons)}")
        label = "qcqp_sweep"
        if case_labels:
            label = "_".join([label, *case_labels])
        iter_factor = params.get("high_dim_max_iter_factor")
        if iter_factor is not None:
            label = f"{label}_iter{_label_number(float(iter_factor))}"
    else:
        label = f"qcqp_n{int(params['n_var'])}_q{int(params['n_qua_cons'])}"
    return f"{label}_{suffix}" if suffix else label


def jcc_result_label(params):
    return (
        f"jcc_bus{int(params['num_bus'])}"
        f"_scen{int(params['n_scenarios'])}"
        f"_eps{_label_number(float(params['epsilon']))}"
    )


QCQP_PROBLEM_CONFIG = {
    "n_var": 2,
    "n_qua_cons": 3,
    "n_linear_cons": 0,
    "problem_config": {
        "obj": "quad",
        "nonconvex_ratio": 1.0,
        "x_lower": -5.0,
        "x_upper": 5.0,
        "R": 5.0,
        "constraint_convexity": False,
        "objective_convexity": False,
    },
    "model_config": {
        "Con_type": "Mix",
    },
}


QCQP_INN_VISUALIZATION_CONFIG = {
    "config": None,
    "run_high_dim_sweep": False,
    "high_dim_sweep_cases": None,
    "high_dim_max_iter_factor": 10,
    "visualize": False,
    "visualize_instance_idx": 0,
    "visualize_mdh_mapping": False,
    "show_convergence_legend": True,
    "violation_y_min": 1e-8,
    "plot_case_convergence": True,
    "plot_objective_gap": True,
    "plot_gap_plus_violation": True,
    "plot_runtime_summary": True,
    "convergence_reference_objective": None,
    "compare_ipopt_baseline": False,
}


QCQP_INN_PGD_CONFIG = merge_params(QCQP_PROBLEM_CONFIG, QCQP_INN_VISUALIZATION_CONFIG)


QCQP_IPOPT_REFERENCE_CONFIG = {
    "compare_ipopt_baseline": True,
    "ipopt_solver_name": "ipopt",
    "ipopt_options": {
        "max_iter": 3000,
        "tol": 1e-5,
        "print_level": 0,
    },
}


def qcqp_inn_pgd_config(*overrides):
    return inn_pgd_config(QCQP_INN_PGD_CONFIG, *overrides)


def qcqp_ipopt_reference_config():
    return merge_params({}, QCQP_IPOPT_REFERENCE_CONFIG)


JCC_INN_CONFIG = {
    "problem_config": {
        # ``None`` selects the repository-resolved PGLib cache, avoiding a
        # dependence on the caller's current working directory.  Scripts can
        # still override this with a specific local case directory.
        "pglib_data_dir": None,
        "pglib_case_name": None,
        "download_if_missing": True,
    },
    "model_config": {
        "inv_type": "made",
        "bilip": False,
        "L": 2,
        "h_dim": 128,
        "w_penalty": 10.0,
        "w_distortion": 0.01,
        "c_samples": 10000,
        "lr_decay_step": 1000,
        "ema_decay": 0.9,
        "resultsSaveFreq": 2000,
    },
    "train_config": {
        "batch_size": 256,
    },
    "lagrangian_baseline_config": {
        "lagrangian_gradient": "autograd",
    },
}


def jcc_inn_pgd_config(*overrides):
    return inn_pgd_config(JCC_INN_CONFIG, *overrides)
