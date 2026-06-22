"""Algorithm method-parameter factories for experiment benchmarks."""

from __future__ import annotations

import copy

from homopt.optim.defaults import build_first_order_subproblem_defaults

from homopt.experiments.common.config import merged, reject_iteration_budget_keys


def penalty_family_shared_defaults(
    *,
    dual_learning_rate=1e-1,
    penalty_coef=10.0,
    penalty_growth=1.1,
    proximal_coef=0.1,
    max_penalty=1e2,
    max_dual=1e2,
    use_lagrangian=True,
    use_penalty=True,
    use_proximal=False,
):
    return {
        "dual_learning_rate": dual_learning_rate,
        "penalty_coef": penalty_coef,
        "penalty_growth": penalty_growth,
        "proximal_coef": proximal_coef,
        "max_penalty": max_penalty,
        "max_dual": max_dual,
        "use_lagrangian": use_lagrangian,
        "use_penalty": use_penalty,
        "use_proximal": use_proximal,
    }


def default_penalty_method_config():
    return {
        "learning_rate": 1e-3,
        "outer_iterations": 50,
        "inner_iterations": 10,
        "max_running_time": 300,
        "convergence_threshold": 1e-6,
        "outer_stepsize_rule": "adaptive",
        "inner_stepsize_rule": "constant",
        "inner_solver": "gd",
        "inner_stopping_rule": "fixed",
        "inner_proximal_update_iterations": 1,
        "inner_proximal_coef": 0.0,
        "inner_proximal_space": "x",
        "acceleration_method": "none",
        "acceleration_space": "x",
        "outer_lr_decay": 0.999,
        "inner_lr_decay": 0.999,
        "min_lr": 1e-6,
        "inner_min_lr": 1e-6,
        "momentum": 0.0,
        "proximal_space": "x",
        **penalty_family_shared_defaults(),
        "opt": "gd",
    }


def normalize_penalty_method_config(
    *,
    penalty_method_args=None,
    penalty_method_config=None,
):
    penalty_method_args = copy.deepcopy(penalty_method_args) if penalty_method_args else penalty_method_args
    penalty_method_config = copy.deepcopy(penalty_method_config) if penalty_method_config else penalty_method_config
    cfg = merged(default_penalty_method_config(), penalty_method_args)
    cfg = merged(cfg, penalty_method_config)
    return cfg


CONVEX_ALM_INNER_ITERATIONS = 50

EQ_CVXPY_BASELINE_SPECS = {
    "Penalty-EQ": {
        "use_lagrangian": False,
        "use_penalty": True,
        "use_proximal": False,
    },
    "Prox-Penalty-EQ": {
        "use_lagrangian": False,
        "use_penalty": True,
        "use_proximal": True,
    },
    "ALM-EQ": {
        "use_lagrangian": True,
        "use_penalty": True,
        "use_proximal": False,
    },
    "Prox-ALM-EQ": {
        "use_lagrangian": True,
        "use_penalty": True,
        "use_proximal": True,
    },
}

EQ_CVXPY_BASELINE_KEYS = (
    "max_running_time",
    "outer_iterations",
    "convergence_threshold",
    "dual_learning_rate",
    "penalty_coef",
    "penalty_growth",
    "proximal_coef",
    "max_penalty",
    "max_dual",
    "use_lagrangian",
    "use_penalty",
    "use_proximal",
)


def build_first_order_subproblem_params(
    common,
    *,
    outer_iterations,
    inner_iterations,
    config=None,
):
    """Canonical Lagrangian subproblem config for PGD projection and FW LOO."""

    return normalize_penalty_method_config(
        penalty_method_args=build_first_order_subproblem_defaults(
            common,
            outer_iterations=outer_iterations,
            inner_iterations=inner_iterations,
        ),
        penalty_method_config=config,
    )


def base_common_params(seed, max_iterations, max_running_time, learning_rate, stepsize_rule, lr_decay, min_lr):
    return {
        "seed": seed,
        "convergence_threshold": 1e-6,
        "max_iterations": max_iterations,
        "max_running_time": max_running_time,
        "opt": "gd",
        "learning_rate": learning_rate,
        "stepsize_rule": stepsize_rule,
        "hom_p_norm": 2,
        "hom_map_explicit_gradient_rule": "polynomial",
        "hom_map_smooth_tie_tol": 1e-7,
        "hom_map_smooth_temperature": 1e-4,
        "momentum": 0.0,
        "initial_point_mode": "gauge_center",
        "acceleration_method": "none",
        "acceleration_space": "z",
        "smooth": False,
        "lr_decay": lr_decay,
        "outer_lr_decay": lr_decay,
        "inner_lr_decay": lr_decay,
        "min_lr": min_lr,
        "inner_min_lr": min_lr,
        "outer_stepsize_rule": stepsize_rule,
    }


def build_first_order_method_params(
    common,
    *,
    projection_outer_iterations,
    projection_inner_iterations,
    linearization_outer_iterations=None,
    linearization_inner_iterations=None,
    projection_subproblem=None,
    linearization_subproblem=None,
    include_fw=True,
    include_rd=True,
    hom_p_norm=2,
    hom_map_gradient="explicit",
    hom_momentum=None,
):
    pgd_projection_subproblem = build_first_order_subproblem_params(
        common,
        outer_iterations=projection_outer_iterations,
        inner_iterations=projection_inner_iterations,
        config=projection_subproblem,
    )
    effective_linearization_outer_iterations = (
        linearization_outer_iterations
        if linearization_outer_iterations is not None
        else projection_outer_iterations
    )
    effective_linearization_inner_iterations = (
        linearization_inner_iterations
        if linearization_inner_iterations is not None
        else projection_inner_iterations
    )
    fw_linearization_subproblem = build_first_order_subproblem_params(
        common,
        outer_iterations=effective_linearization_outer_iterations,
        inner_iterations=effective_linearization_inner_iterations,
        config=linearization_subproblem,
    )
    x_space_common = {**common, "acceleration_space": "x"}
    params = {
        "PGD": {
            **x_space_common,
            "opt_type": "PGD",
            "projection_outer_iterations": projection_outer_iterations,
            "projection_inner_iterations": projection_inner_iterations,
            "projection_subproblem": pgd_projection_subproblem,
        },
        "Hom-PGD": {
            **common,
            "opt_type": "Hom-PGD",
            "hom_p_norm": hom_p_norm,
            "hom_map_gradient": hom_map_gradient,
            "use_proximal": False,
            "proximal_update_iterations": 1,
            "proximal_coef": 0.0,
            "proximal_space": "z",
        },
    }
    if hom_momentum is not None:
        params["Hom-PGD"]["momentum"] = hom_momentum
    if include_fw:
        params["FW"] = {
            **x_space_common,
            "opt_type": "FW",
            "linearization_outer_iterations": effective_linearization_outer_iterations,
            "linearization_inner_iterations": effective_linearization_inner_iterations,
            "linearization_subproblem": fw_linearization_subproblem,
        }
    if include_rd:
        params["RD"] = {
            **x_space_common,
            "opt_type": "RD",
        }
    return params


def filter_params(common, keys):
    return {key: common[key] for key in keys if key in common}


def optional_params(**kwargs):
    return {key: value for key, value in kwargs.items() if value is not None}


def build_convex_penalty_method_kwargs(
    common,
    *,
    outer_iterations,
    inner_iterations=CONVEX_ALM_INNER_ITERATIONS,
    dual_learning_rate=1e-1,
    proximal_space="x",
    use_lagrangian=True,
    use_penalty=True,
    use_proximal=False,
):
    """Canonical penalty/ALM defaults for fixed convex comparison experiments."""

    return {
        "outer_iterations": outer_iterations,
        "inner_iterations": inner_iterations,
        "proximal_space": proximal_space,
        "outer_stepsize_rule": common["outer_stepsize_rule"],
        "min_lr": common["min_lr"],
        "inner_min_lr": common["inner_min_lr"],
        "inner_stepsize_rule": "constant",
        "inner_solver": "gd",
        "inner_proximal_update_iterations": 1,
        "inner_proximal_coef": 0.0,
        "inner_proximal_space": proximal_space,
        "inner_restart": False,
        "acceleration_space": proximal_space,
        **penalty_family_shared_defaults(
            dual_learning_rate=dual_learning_rate,
            use_lagrangian=use_lagrangian,
            use_penalty=use_penalty,
            use_proximal=use_proximal,
        ),
    }


def build_penalty_method_params(
    common,
    *,
    outer_iterations=None,
    inner_iterations=None,
    dual_learning_rate=None,
    penalty_coef=None,
    penalty_growth=None,
    proximal_coef=None,
    proximal_space=None,
    inner_learning_rate=None,
    outer_lr_decay=None,
    inner_lr_decay=None,
    min_lr=None,
    inner_min_lr=None,
    outer_stepsize_rule=None,
    inner_stepsize_rule=None,
    inner_solver=None,
    inner_proximal_update_iterations=None,
    inner_proximal_coef=None,
    inner_proximal_space=None,
    inner_restart=None,
    acceleration_method=None,
    acceleration_space=None,
    lagrangian_gradient="explicit",
    max_penalty=None,
    max_dual=None,
    use_lagrangian=True,
    use_penalty=True,
    use_proximal=False,
):
    params = normalize_penalty_method_config(
        penalty_method_args={
            **filter_params(
                common,
                (
                    "learning_rate",
                    "inner_learning_rate",
                    "max_running_time",
                    "outer_lr_decay",
                    "inner_lr_decay",
                    "min_lr",
                    "inner_min_lr",
                    "convergence_threshold",
                    "outer_stepsize_rule",
                    "opt",
                    "momentum",
                    "verbose_interval",
                    "inner_restart",
                    "acceleration_method",
                    "acceleration_space",
                ),
            ),
            "outer_lr_decay": common["outer_lr_decay"],
            "inner_lr_decay": common["inner_lr_decay"],
            "min_lr": common["min_lr"],
            "inner_min_lr": common["inner_min_lr"],
            "use_lagrangian": use_lagrangian,
            "use_penalty": use_penalty,
            "use_proximal": use_proximal,
        },
        penalty_method_config={
            **optional_params(
                outer_iterations=outer_iterations,
                inner_iterations=inner_iterations,
                dual_learning_rate=dual_learning_rate,
                penalty_coef=penalty_coef,
                penalty_growth=penalty_growth,
                proximal_coef=proximal_coef,
                proximal_space=proximal_space,
                inner_learning_rate=inner_learning_rate,
                outer_lr_decay=outer_lr_decay,
                inner_lr_decay=inner_lr_decay,
                min_lr=min_lr,
                inner_min_lr=inner_min_lr,
                outer_stepsize_rule=outer_stepsize_rule,
                inner_stepsize_rule=inner_stepsize_rule,
                inner_solver=inner_solver,
                inner_proximal_update_iterations=inner_proximal_update_iterations,
                inner_proximal_coef=inner_proximal_coef,
                inner_restart=inner_restart,
                acceleration_method=acceleration_method,
                max_penalty=max_penalty,
                max_dual=max_dual,
            ),
            **{"inner_proximal_space": inner_proximal_space if inner_proximal_space is not None else "x"},
            **{"acceleration_space": acceleration_space if acceleration_space is not None else "x"},
            "lagrangian_gradient": lagrangian_gradient,
        },
    )
    params["opt_type"] = "ALM"
    return params


def build_eq_alm_method_params(
    common,
    *,
    opt_type="ALM-EQ",
    outer_iterations=None,
    dual_learning_rate=None,
    penalty_coef=None,
    penalty_growth=None,
    proximal_coef=None,
    max_penalty=None,
    max_dual=None,
    use_lagrangian=True,
    use_penalty=True,
    use_proximal=False,
):
    params = normalize_penalty_method_config(
        penalty_method_args={
            **filter_params(
                common,
                (
                    "max_running_time",
                    "convergence_threshold",
                ),
            ),
            "inner_iterations": 1,
            "proximal_space": "x",
            "use_lagrangian": use_lagrangian,
            "use_penalty": use_penalty,
            "use_proximal": use_proximal,
        },
        penalty_method_config={
            **optional_params(
                outer_iterations=outer_iterations,
                dual_learning_rate=dual_learning_rate,
                penalty_coef=penalty_coef,
                penalty_growth=penalty_growth,
                proximal_coef=proximal_coef,
                max_penalty=max_penalty,
                max_dual=max_dual,
            ),
        },
    )
    return {
        **{key: params[key] for key in EQ_CVXPY_BASELINE_KEYS},
        "opt_type": opt_type,
    }


def build_eq_cvxpy_baseline_params(common, *, outer_iterations):
    return {
        name: build_eq_alm_method_params(
            common,
            opt_type=name,
            outer_iterations=outer_iterations,
            **penalty_family_shared_defaults(**flags),
        )
        for name, flags in EQ_CVXPY_BASELINE_SPECS.items()
    }


def build_hom_penalty_method_params(
    common,
    *,
    outer_iterations=None,
    inner_iterations=None,
    dual_learning_rate=None,
    penalty_coef=None,
    penalty_growth=None,
    proximal_coef=None,
    proximal_space=None,
    inner_learning_rate=None,
    outer_lr_decay=None,
    inner_lr_decay=None,
    min_lr=None,
    inner_min_lr=None,
    outer_stepsize_rule=None,
    inner_stepsize_rule=None,
    inner_solver=None,
    inner_proximal_update_iterations=None,
    inner_proximal_coef=None,
    inner_proximal_space=None,
    inner_restart=None,
    acceleration_method=None,
    acceleration_space=None,
    hom_map_gradient="explicit",
    max_penalty=None,
    max_dual=None,
    use_lagrangian=True,
    use_penalty=True,
    use_proximal=False,
):
    params = build_penalty_method_params(
        common,
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
        dual_learning_rate=dual_learning_rate,
        penalty_coef=penalty_coef,
        penalty_growth=penalty_growth,
        proximal_coef=proximal_coef,
        proximal_space=proximal_space if proximal_space is not None else "z",
        inner_learning_rate=inner_learning_rate,
        outer_lr_decay=outer_lr_decay,
        inner_lr_decay=inner_lr_decay,
        min_lr=min_lr,
        inner_min_lr=inner_min_lr,
        outer_stepsize_rule=outer_stepsize_rule,
        inner_stepsize_rule=inner_stepsize_rule,
        inner_solver=inner_solver,
        inner_proximal_update_iterations=inner_proximal_update_iterations,
        inner_proximal_coef=inner_proximal_coef,
        inner_proximal_space=inner_proximal_space if inner_proximal_space is not None else "z",
        inner_restart=inner_restart,
        acceleration_method=acceleration_method,
        acceleration_space=acceleration_space if acceleration_space is not None else "z",
        max_penalty=max_penalty,
        max_dual=max_dual,
        use_lagrangian=use_lagrangian,
        use_penalty=use_penalty,
        use_proximal=use_proximal,
    )
    params["hom_map_gradient"] = hom_map_gradient
    params["opt_type"] = "Hom-ALM"
    return params


def normalize_jcc_lagrangian_config(
    *,
    max_iterations=None,
    lagrangian_baseline_config=None,
):
    if lagrangian_baseline_config is None:
        raise ValueError("JCC lagrangian_baseline_config is required when Lagrangian baselines are enabled.")
    reject_iteration_budget_keys(lagrangian_baseline_config, context="JCC lagrangian_baseline_config")
    invalid = sorted({"outer_lr_decay", "outer_stepsize_rule", "outer_learning_rate"}.intersection(lagrangian_baseline_config))
    if invalid:
        raise ValueError(
            f"Unsupported JCC lagrangian_baseline_config keys: {invalid}. "
            "Use learning_rate, lr_decay, and stepsize_rule."
        )
    required = {
        "learning_rate",
        "inner_learning_rate",
        "inner_iterations",
        "max_running_time",
        "convergence_threshold",
        "stepsize_rule",
        "inner_stepsize_rule",
        "inner_solver",
        "lr_decay",
        "inner_lr_decay",
        "min_lr",
        "inner_min_lr",
        "momentum",
        "proximal_space",
        "lagrangian_gradient",
        "dual_learning_rate",
        "penalty_coef",
        "penalty_growth",
        "proximal_coef",
        "max_penalty",
        "max_dual",
        "opt",
        "verbose",
        "verbose_interval",
    }
    missing = sorted(required - set(lagrangian_baseline_config))
    if missing:
        raise ValueError(f"Missing JCC lagrangian_baseline_config keys: {missing}")
    public_config = copy.deepcopy(lagrangian_baseline_config)
    public_config["outer_lr_decay"] = public_config["lr_decay"]
    public_config["outer_stepsize_rule"] = public_config["stepsize_rule"]
    shared_controls = penalty_family_shared_defaults()
    lag_args = normalize_penalty_method_config(
        penalty_method_args={
            **shared_controls,
        },
        penalty_method_config=public_config,
    )
    if max_iterations is not None:
        lag_args["outer_iterations"] = int(max_iterations)
    return lag_args


__all__ = [
    "EQ_CVXPY_BASELINE_SPECS",
    "base_common_params",
    "build_convex_penalty_method_kwargs",
    "build_eq_alm_method_params",
    "build_eq_cvxpy_baseline_params",
    "build_first_order_method_params",
    "build_first_order_subproblem_params",
    "build_hom_penalty_method_params",
    "build_penalty_method_params",
    "default_penalty_method_config",
    "normalize_jcc_lagrangian_config",
    "normalize_penalty_method_config",
    "penalty_family_shared_defaults",
]
