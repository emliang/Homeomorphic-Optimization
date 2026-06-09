"""Hom-PGD visualization namespace."""

from ._convex2d_impl import save_convex_2d_visualizations
from .comparison import save_comparison_visualizations
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

__all__ = [
    "ALGORITHM_COLORS",
    "ALGORITHM_LINE_STYLES",
    "PAPER_STYLE",
    "apply_paper_axis_style",
    "apply_paper_style_rcparams",
    "build_convex_2d_traces",
    "configure_matplotlib_cache",
    "save_convex_2d_visualizations",
    "save_comparison_visualizations",
    "split_convex_metrics",
]
