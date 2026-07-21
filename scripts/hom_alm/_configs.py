"""Shared Hom-ALM script configs."""

from homopt.experiments.common.labels import CONVEX_METHOD_LABELS
from homopt.experiments.hom_alm import (
    CONVEX_EQ_2D_ALGORITHMS,
    STIEFEL_PLOT_ALGORITHM_ORDER,
    STIEFEL_PLOT_LABELS,
)


HOM_ALM_RUNTIME_CONFIG = {
    "device": "cuda:0",
    "dtype": "float32",
}


HOM_ALM_REFERENCE_CONFIG = {
    "include_reference_solver": True,
    "reference_need_opt": True,
    "reference_cache": True,
    "visualize": True,
    "visualize_only": False,
    "visualization_prefix": None,
    "plot_algorithm_order": None,
    "reference_label": "MOSEK",
    "show_convergence_legend": False,
    "verbose": True,
    "verbose_interval": 50,
}


HOM_ALM_MAPPING_CONFIG = {
    "hom_p_norm": 2,
    "hom_map_explicit_gradient_rule": "polynomial",
    "hom_map_smooth_tie_tol": 1e-7,
    "hom_map_smooth_temperature": 1e-4,
    "smooth": True,
}


HOM_ALM_OUTER_CONFIG = {
    "convergence_threshold": 1e-6,
    "first_order_lagrangian_gap_threshold": 1e-6,
    "dual_learning_rate": 1e-1,
    "penalty_coef": 1.0,
    "penalty_growth": 1.01,
    "proximal_coef": 1e-1,
    "max_penalty": 1e1,
    "max_dual": 1e2,
    "return_best_violation": False,
}


HOM_ALM_INNER_SOLVER_CONFIG = {
    "inner_iterations": 100,
    "inner_stopping_rule": "adaptive",
    "inner_iterations_min": 100,
    "inner_iterations_max": 100,
    "inner_tol_factor": 0.1,
    "inner_tol_min": 1e-6,
    "inner_stepsize_rule": "adaptive",
    "inner_solver": "gd",
    "inner_proximal_update_iterations": 10,
    "inner_proximal_coef": 1e-3,
    "inner_restart": False,
    "opt": "gd",
    "inner_learning_rate": 1e-3,
    "acceleration_method": "none",
    "acceleration_space": "z",
    "momentum": 0.9,
    "inner_lr_decay": 0.999,
    "inner_min_lr": 1e-6,
}


CONVEX_EQ_COMPARE_CONFIG = {
    **HOM_ALM_RUNTIME_CONFIG,
    **HOM_ALM_REFERENCE_CONFIG,
    "problem_type": "socp_eq",
    "algorithms": ["PGD", "Prox-Penalty", "Prox-ALM", "Prox-ALM-EQ", "Prox-Hom-ALM"],
    "seed": 2026,
    "n_var": 10,
    "n_linear_cons": 0,
    "n_soc_cons": 0,
    "n_qua_cons": 5,
    "n_lin_eq": 5,
    "obj": "quad",
    "objective_quadratic_type": "diagonal",
    "constraint_quadratic_type": "diagonal",
    "quad_diag_lower": 1e-5,
    "quad_diag_upper": 1e-1,
    "low_rank_quad_ridge": 1e-5,
    "x_lower": -5.0,
    "x_upper": 5.0,
    "margin_scale": 0.1,
    "anchor_interior_ratio": 0.1,
    "problem_config": None,
    "method_labels": CONVEX_METHOD_LABELS,
    "hom_origin_constraints": "inequality",
    "hom_origin_ip_mode": "central_ip",
    "hom_origin_ip_eps": 1e-3,
    "hom_origin": None,
    "scale_label": None,
    "common_config": {
        "max_iterations": 100,
        "max_running_time": 600,
        "learning_rate": 1e-2,
        "stepsize_rule": "adaptive",
        "lr_decay": 0.999,
        "min_lr": 1e-6,
        "acceleration_method": "none",
        "acceleration_space": "z",
        "momentum": 0.0,
        "initial_point_mode": "gauge_center",
        **HOM_ALM_MAPPING_CONFIG,
    },
    "outer_common": HOM_ALM_OUTER_CONFIG,
    "inner_solver_common": HOM_ALM_INNER_SOLVER_CONFIG,
    "algorithm_config": {
        "PGD": {
            "convergence_metric": "objective_change",
            "objective_change_threshold": 1e-6,
            "min_iterations": 2,
        },
        "Prox-Hom-ALM": {
            "hom_map_gradient": "explicit",
            "use_proximal": True,
        },
    },
}


def convex_eq_2d_config(problem_overrides):
    return {
        "device": "cpu",
        "dtype": "float32",
        "problem_type": "socp_eq",
        "scale_label": "socp_eq_2d_toy",
        "algorithms": CONVEX_EQ_2D_ALGORITHMS,
        "seed": 2026,
        "n_var": 2,
        "n_linear_cons": 10,
        "n_qua_cons": 1,
        "n_soc_cons": 0,
        "n_lin_eq": 1,
        "obj": "quad",
        "quad_diag_lower": 1e-2,
        "quad_diag_upper": 1.0,
        "low_rank_quad_ridge": 1e-2,
        "margin_scale": 0.1,
        "anchor_interior_ratio": 0.8,
        "x_lower": -2.0,
        "x_upper": 2.0,
        "problem_config": problem_overrides,
        "reference_need_opt": True,
        "include_reference_solver": True,
        "method_labels": CONVEX_METHOD_LABELS,
        "reference_label": "MOSEK",
        "show_constraint_notation": True,
        "hom_origin_constraints": "ineq",
        "common_config": {
            "convergence_threshold": 1e-6,
            "max_iterations": 80,
            "max_running_time": 180,
            "learning_rate": 1e-3,
            "stepsize_rule": "adaptive",
            "lr_decay": 0.999,
            "min_lr": 1e-6,
            "initial_point_mode": "gauge_center",
            "acceleration_method": "none",
            "acceleration_space": "z",
            "momentum": 0.0,
            **HOM_ALM_MAPPING_CONFIG,
        },
        "algorithm_config": {
            "ALM": {
                "inner_iterations": 50,
            },
            "Hom-ALM": {
                "inner_iterations": 50,
            },
        },
    }


STIEFEL_EQ_COMPARE_CONFIG = {
    "device": "auto",
    "dtype": "float32",
    "seed": 2026,
    "scale_label": "small_digits_default",
    "max_iterations": 2000,
    "algorithms": ["Prox-Penalty", "StiefelRetractionPenalty", "StiefelRetractionALM", "Prox-ALM", "Prox-Hom-ALM"],
    "verbose": True,
    "verbose_interval": 50,
    "visualize": True,
    "visualize_only": False,
    "visualization_prefix": None,
    "show_convergence_legend": False,
    "include_reference_solver": True,
    "reference_need_opt": True,
    "reference_cache": True,
    "plot_algorithm_order": STIEFEL_PLOT_ALGORITHM_ORDER,
    "method_labels": STIEFEL_PLOT_LABELS,
    "problem_config": {
        "data_source": "sklearn_digits",
        "n_cols": 3,
        "download_if_missing": False,
        "restricted_rows": list(range(16)),
        "group_budget": 1.2,
        "box_lower": -1.5,
        "box_upper": 1.5,
        "standardize": True,
        "max_samples": None,
    },
    "mapping": {
        **HOM_ALM_MAPPING_CONFIG,
        "x_origin": "zero",
    },
    "initialization": {
        "mode": "random_ball",
        "scale": 0.1,
    },
    "common_config": {
        "learning_rate": 1e-2,
        "stepsize_rule": "adaptive",
        "lr_decay": 0.995,
        "min_lr": 1e-6,
    },
    "outer_common": {
        **HOM_ALM_OUTER_CONFIG,
        "max_running_time": 600,
        "max_penalty": 1e3,
        "max_dual": 1e3,
    },
    "inner_solver_common": HOM_ALM_INNER_SOLVER_CONFIG,
    "algorithm_config": {
        "Prox-ALM": {
            "lagrangian_gradient": "explicit",
            "proximal_space": "x",
            "inner_proximal_space": "x",
            "acceleration_space": "x",
            "use_proximal": True,
        },
        "Prox-Hom-ALM": {
            "hom_map_gradient": "explicit",
            "proximal_space": "z",
            "inner_proximal_space": "z",
            "acceleration_space": "z",
            "use_proximal": True,
        },
        "StiefelIPOPT": {
            "solver": "ipopt",
            "multi_start": 1,
            "solver_options": {
                "tol": 1e-6,
                "max_iter": 100000,
                "print_level": 0,
            },
            "tee": False,
        },
    },
}


SMALL_DIGITS_OVERRIDES = {
    "scale_label": "small_digits",
    "problem_config": {
        "data_source": "sklearn_digits",
        "n_cols": 3,
        "download_if_missing": False,
        "restricted_rows": "top_loading",
        "restricted_row_count": 16,
        "group_budget": 0.75,
        "box_lower": -0.5,
        "box_upper": 0.5,
        "standardize": True,
        "objective_normalization": "top_sum",
        "max_samples": None,
    },
}


INTERMEDIATE_LFW_OVERRIDES = {
    "scale_label": "intermediate_lfw",
    "problem_config": {
        "data_source": "sklearn_lfw_people",
        "data_home": "data/sklearn_data",
        "n_cols": 3,
        "restricted_rows": "top_loading",
        "restricted_row_count": 96,
        "group_budget": 0.55,
        "box_lower": -0.5,
        "box_upper": 0.5,
        "standardize": True,
        "objective_normalization": "top_sum",
        "download_if_missing": False,
        "lfw_resize": 0.25,
        "max_samples": None,
    },
}


MEDIUM_LFW_OVERRIDES = {
    "scale_label": "medium_lfw",
    "problem_config": {
        "data_source": "sklearn_lfw_people",
        "data_home": "data/sklearn_data",
        "n_cols": 5,
        "restricted_rows": "top_loading",
        "restricted_row_count": 128,
        "group_budget": 0.85,
        "box_lower": -0.5,
        "box_upper": 0.5,
        "standardize": True,
        "objective_normalization": "top_sum",
        "download_if_missing": False,
        "lfw_resize": 0.25,
        "max_samples": None,
    },
}


MEDIUM_OLIVETTI_OVERRIDES = {
    "scale_label": "medium_olivetti",
    "problem_config": {
        "data_source": "sklearn_olivetti_faces",
        "data_home": "data/sklearn_data",
        "n_cols": 2,
        "restricted_rows": "top_loading",
        "restricted_row_count": 128,
        "group_budget": 0.2,
        "box_lower": -0.5,
        "box_upper": 0.5,
        "standardize": True,
        "objective_normalization": "top_sum",
        "download_if_missing": False,
        "max_samples": None,
    },
}


LARGE_MNIST_OVERRIDES = {
    "scale_label": "large_mnist",
    "problem_config": {
        "data_source": "openml_mnist",
        "data_home": "data/sklearn_data",
        "n_cols": 13,
        "restricted_rows": "top_loading",
        "restricted_row_count": 196,
        "group_budget": 0.9,
        "box_lower": -0.5,
        "box_upper": 0.5,
        "standardize": True,
        "objective_normalization": "top_sum",
        "download_if_missing": False,
        "max_samples": 20000,
    },
}
