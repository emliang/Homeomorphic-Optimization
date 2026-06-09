"""Shared Hom-PGD script presets."""


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
        "max_iterations": int(max_iterations),
        "max_running_time": 600,
        "seed": int(seed),
        "learning_rate": 1e-2,
        "stepsize_rule": "adaptive",
        "lr_decay": 0.999,
        "visualize": True,
        "visualize_only": False,
        "show_constraint_notation": False,
        "common_config": {
            "convergence_threshold": float(convergence_threshold),
            "opt": "gd",
            "warm_start": False,
            "acceleration_method": "none",
            "acceleration_space": "z",
            "stepsize_rule": "adaptive",
            "smooth": False,
            "hom_p_norm": 2,
            "momentum": float(momentum),
        },
        "algorithm_config": {
            "PGD": {
                "opt_type": "PGD",
                "learning_rate": 1e-2,
                "projection_outer_iterations": 10,
                "projection_inner_iterations": 100,
            },
            "Hom-PGD": {
                "opt_type": "Hom-PGD",
                "learning_rate": 1e-2,
                "hom_p_norm": 2,
                "momentum": float(momentum),
                "max_iterations": int(max_iterations),
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
