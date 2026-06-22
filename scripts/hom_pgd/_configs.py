"""Shared Hom-PGD script configs."""


CONVEX_INEQ_COMPARE_CONFIG = {
    "device": "auto",
    "dtype": "float32",
    "problem_type": "socp",
    "algorithms": ["PGD", "FW", "ALM", "RD", "Hom-PGD"],
    "seed": 2026,
    "n_var": 10,
    "n_linear_cons": 0,
    "n_soc_cons": 10,
    "n_qua_cons": 0,
    "n_lin_eq": 0,
    "obj": "quad",
    "x_lower": -5.0,
    "x_upper": 5.0,
    "margin_scale": 0.1,
    "anchor_interior_ratio": 0.8,
    "objective_quadratic_type": "diagonal",
    "constraint_quadratic_type": "diagonal",
    "quad_diag_lower": 1e-2,
    "quad_diag_upper": 1.0,
    "low_rank_quad_ridge": 1e-2,
    "outer_common": None,
    "inner_solver_common": None,
    "problem_config": None,
    "visualize": True,
    "visualize_only": False,
    "visualization_prefix": None,
    "plot_algorithm_order": None,
    "reference_label": "MOSEK",
    "show_convergence_legend": True,
    "verbose": True,
    "verbose_interval": 1000,
    "include_reference_solver": None,
    "reference_need_opt": True,
    "reference_cache": True,
    "hom_origin_constraints": "full",
    "hom_origin_ip_mode": "ip",
    "hom_origin_ip_eps": 1e-3,
    "hom_origin": None,
    "common_config": {
        "convergence_threshold": 1e-6,
        "max_iterations": 10000,
        "max_running_time": 600,
        "learning_rate": 1e-2,
        "opt": "gd",
        "initial_point_mode": "gauge_center",
        "acceleration_method": "none",
        "acceleration_space": "z",
        "lr_decay": 0.999,
        "min_lr": 1e-6,
        "stepsize_rule": "adaptive",
        "smooth": True,
        "momentum": 0.0,
    },
    "algorithm_config": {
        "PGD": {
            "learning_rate": 1e-3,
            "projection_max_iterations": 10,
            "projection_inner_iterations": 100,
        },
        "FW": {
            "learning_rate": 1e-3,
            "linearization_max_iterations": 10,
            "linearization_inner_iterations": 100,
        },
        "ALM": {
            "learning_rate": 1e-3,
            "inner_iterations": 100,
            "dual_learning_rate": 1e-2,
        },
        "RD": {
            "learning_rate": 1e-3,
        },
        "Hom-PGD": {
            "learning_rate": 1e-3,
            "hom_p_norm": 2,
            "momentum": 0.99,
        },
    },
}


MAXCUT_SDP_COMPARE_CONFIG = {
    "device": "auto",
    "dtype": "float32",
    "algorithms": ["PGD", "ALM-log", "ALM-bp", "RD", "Hom-PGD"],
    "seed": 2025,
    "n": 10,
    "alpha": 0.5,
    "visualize": True,
    "visualize_only": False,
    "visualization_prefix": "maxcut_sdp",
    "method_labels": None,
    "reference_label": "SDP",
    "show_convergence_legend": True,
    "common_config": {
        "learning_rate": 1e-3,
        "convergence_threshold": 1e-6,
        "max_iterations": 10000,
        "max_running_time": 600,
        "opt": "gd",
        "stepsize_rule": "adaptive",
        "initial_point_mode": "gauge_center",
        "acceleration_method": "none",
        "acceleration_space": "z",
        "smooth": True,
        "momentum": 0.0,
        "lr_decay": 0.999,
        "min_lr": 1e-6,
    },
}


def _label_number(value):
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace(".", "p").replace("-", "m")


def poly_star_base_params(
    *,
    problem_type,
    algorithms,
    alpha,
    max_iterations,
    convergence_threshold,
    momentum,
    seed=2025,
):
    """Return the common editable toy poly/star benchmark parameters."""

    return {
        "device": "auto",
        "dtype": "float32",
        # Explicit toy constrained set: "poly" uses a 2D convex polytope;
        # "star" uses the nonconvex star-shaped set; "poly_star" uses their intersection.
        "problem_type": problem_type,
        "algorithms": list(algorithms),
        "alpha": float(alpha),
        "num_star": 4,
        "poly_config": {
            "n_linear_cons": 4,
            "x_lower": -2.0,
            "x_upper": 2.0,
            "margin_scale": 0.2,
            "poly_radius": 1.0,
        },
        "seed": int(seed),
        "visualize": True,
        "visualize_only": False,
        "show_constraint_notation": False,
        # Reference policy: auto uses MOSEK/ConvexSolver for "poly" and IPOPT
        # for the nonconvex "star" and "poly_star" toy problems.
        "reference_solver": "auto",
        "reference_label": None,
        "reference_ipopt_solver": "ipopt",
        "reference_ipopt_options": {
            "max_iter": 3000,
            "tol": 1e-8,
            "print_level": 0,
            "sb": "yes",
            "acceptable_tol": 1e-6,
            "acceptable_iter": 15,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
        },
        "reference_ipopt_num_starts": 16,
        "common_config": {
            "convergence_threshold": float(convergence_threshold),
            "max_iterations": int(max_iterations),
            "max_running_time": 600,
            "learning_rate": 1e-2,
            "opt": "gd",
            "initial_point_mode": "gauge_center",
            "acceleration_method": "none",
            "acceleration_space": "z",
            "stepsize_rule": "adaptive",
            "lr_decay": 0.999,
            "min_lr": 1e-6,
            "smooth": True,
            "hom_p_norm": 2,
            "momentum": float(momentum),
        },
        "algorithm_config": {
            "PGD": {
                "projection_max_iterations": 10,
                "projection_inner_iterations": 100,
            },
        },
    }


def poly_star_result_label(params):
    """Label editable poly/star toy runs by the active problem identity."""

    problem_type = str(params.get("problem_type", "star"))
    pieces = [
        problem_type,
        f"a{_label_number(float(params.get('alpha', 0.0)))}",
        f"k{int(params.get('num_star', 0) or 0)}",
    ]
    poly_config = params.get("poly_config") or {}
    if problem_type in {"poly", "poly_star"}:
        pieces.append(f"lin{int(poly_config.get('n_linear_cons', 0) or 0)}")
    if params.get("seed") is not None:
        pieces.append(f"seed{int(params['seed'])}")
    return "_".join(pieces)


def maxcut_result_label(params):
    return (
        f"maxcut_n{int(params['n'])}"
        f"_alpha{_label_number(float(params['alpha']))}"
        f"_seed{int(params['seed'])}"
    )


def adversarial_result_label(params):
    attack = params.get("attack_overrides") or {}
    norm_eps_sweep = params.get("norm_eps_sweep") or []
    if norm_eps_sweep:
        sweep_label = f"sweep{len(norm_eps_sweep)}"
    else:
        sweep_label = f"norm{_label_number(attack.get('norm', 'unknown'))}_eps{_label_number(float(attack.get('eps', 0.0)))}"
    weight_label = attack.get("weight_mode", "plain") if attack.get("weighted_norm") else "plain"
    return "_".join(
        [
            str(params.get("dataset_name", "dataset")),
            sweep_label,
            str(weight_label),
            str(attack.get("optimizer", "opt")),
            f"iter{int(attack.get('iters', 0) or 0)}",
            f"n{int(params.get('attack_selection_size', 0) or 0)}",
            f"seed{int(params.get('seed', 0) or 0)}",
        ]
    )
