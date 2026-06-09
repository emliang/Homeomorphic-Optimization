"""Visualization namespace."""

from ._coninn_impl import evaluate_coninn_exactness, visualize_coninn_exactness
from ._convex2d_impl import save_convex_2d_visualizations
from .comparison import save_comparison_visualizations
from .inn_training import save_inn_training_visualizations
from .style import (
    ALGORITHM_COLORS,
    ALGORITHM_LINE_STYLES,
    PAPER_STYLE,
    apply_paper_axis_style,
    apply_paper_style_rcparams,
    configure_matplotlib_cache,
)
from .traces import (
    build_convex_2d_traces,
    split_convex_metrics,
)
from ._mdh_impl import visualize_mdh_mapping_transformation

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
    "split_convex_metrics",
    "visualize_coninn_exactness",
    "visualize_mdh_mapping_transformation",
]
