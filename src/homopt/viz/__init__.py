"""Visualization namespace."""

_EXPORT_MODULES = {
    "ALGORITHM_COLORS": "style",
    "ALGORITHM_LINE_STYLES": "style",
    "PAPER_STYLE": "style",
    "apply_paper_axis_style": "style",
    "apply_paper_style_rcparams": "style",
    "build_convex_2d_traces": "traces",
    "configure_matplotlib_cache": "style",
    "evaluate_coninn_exactness": "coninn",
    "save_convex_2d_visualizations": "convex2d",
    "save_comparison_visualizations": "comparison",
    "save_inn_training_visualizations": "inn_training",
    "split_problem_metrics": "traces",
    "visualize_coninn_exactness": "coninn",
    "visualize_adversarial_attacks": "adversarial",
    "visualize_mdh_mapping_transformation": "mdh",
}

__all__ = [
    "ALGORITHM_COLORS",
    "ALGORITHM_LINE_STYLES",
    "PAPER_STYLE",
    "apply_paper_axis_style",
    "apply_paper_style_rcparams",
    "build_convex_2d_traces",
    "configure_matplotlib_cache",
    "evaluate_coninn_exactness",
    "save_convex_2d_visualizations",
    "save_comparison_visualizations",
    "save_inn_training_visualizations",
    "split_problem_metrics",
    "visualize_adversarial_attacks",
    "visualize_coninn_exactness",
    "visualize_mdh_mapping_transformation",
]


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'homopt.viz' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
