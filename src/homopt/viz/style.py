"""Shared plotting style and matplotlib helpers."""

from __future__ import annotations

import os


ALGORITHM_COLORS = {
    "PGD": "#0072B2",
    "FW": "#56B4E9",
    "RD": "#7E57C2",
    "Penalty": "#4C78A8",
    "Prox-Penalty": "#72B7B2",
    "ALM": "#E69F00",
    "Prox-ALM": "#F28E2B",
    "Penalty-EQ": "#4C78A8",
    "Prox-Penalty-EQ": "#72B7B2",
    "ALM-EQ": "#009E73",
    "Prox-ALM-EQ": "#CC79A7",
    "Hom-PGD": "#D62728",
    "Hom-ALM": "#E83947",
    "Prox-Hom-ALM": "#B2182B",
    "StiefelRetraction": "#0072B2",
    "StiefelRetractionPenalty": "#0072B2",
    "StiefelRetractionALM": "#7E57C2",
    "StiefelIPOPT": "#222222",
    "ConvexSolver": "#222222",
    "ALM-log": "#F28E2B",
    "ALM-bp": "#B55A0A",
    "INN-PGD": "#D62728",
    "Lagrangian": "#4C78A8",
    "predict_only": "#7F7F7F",
    "predict_ray_bisection": "#0072B2",
    "predict_diff_projection": "#56B4E9",
    "predict_exact_projection": "#009E73",
    "predict_homeomorphic_projection": "#D62728",
    "predict_ip_bisection": "#9467BD",
    "predict_initialized_opt": "#E69F00",
}

ALGORITHM_LINE_STYLES = {
    "PGD": ":",
    "FW": "-.",
    "RD": (0, (3, 1, 1, 1, 1, 1)),
    "Penalty": ":",
    "Prox-Penalty": "-.",
    "ALM": "--",
    "Prox-ALM": "-",
    "Penalty-EQ": ":",
    "Prox-Penalty-EQ": "-.",
    "ALM-EQ": "-",
    "Prox-ALM-EQ": (0, (5, 1, 1, 1)),
    "Hom-PGD": "-",
    "Hom-ALM": "--",
    "Prox-Hom-ALM": "-",
    "StiefelRetraction": "--",
    "StiefelRetractionPenalty": "--",
    "StiefelRetractionALM": (0, (1.2, 1.4)),
    "StiefelIPOPT": ":",
    "ConvexSolver": ":",
    "ALM-log": "--",
    "ALM-bp": "-.",
    "INN-PGD": "-",
    "Lagrangian": "--",
}

INN_PGD_METHOD_LABELS = {
    "INN-PGD": r"Hom-PGD$^+$",
    "Penalty": "EPM",
    "ALM": "ALM",
    "Prox-Penalty": "PPP",
}

PAPER_STYLE = {
    "convergence_figsize": (5.6, 5.0),
    "trajectory_figsize": (6.2, 5.6),
    "runtime_figsize": (5.6, 5.0),
    "two_panel_figsize": (11.2, 4.8),
    "three_panel_figsize": (14.0, 3.8),
    "font_family": "DejaVu Sans",
    "base_fontsize": 14,
    "label_fontsize": 18,
    "tick_fontsize": 16,
    "legend_fontsize": 14,
    "compact_legend_fontsize": 13,
    "title_fontsize": 16,
    "axis_linewidth": 1.35,
    "tick_width": 1.25,
    "tick_length": 5.0,
    "show_grid": False,
    "grid_linewidth": 0.45,
    "grid_alpha": 0.18,
    "curve_linewidth": 2.6,
    "legend_linewidth": 3.0,
    "reference_linewidth": 1.7,
    "objective_contour_linewidth": 1.05,
    "objective_contour_alpha": 0.72,
    "objective_contour_color": "#666666",
    "contour_label_fontsize": 10,
    "constraint_linewidth": 2.2,
    "equality_linewidth": 3.8,
    "inn_unit_boundary_color": "#555555",
    "inn_unit_point_color": "#BDBDBD",
    "inn_constraint_boundary_color": "#C44E52",
    "inn_feasible_fill_color": "#C44E52",
    "inn_instance_colors": ("#2F6FB0", "#8A63D2", "#009E73"),
    "inn_unit_point_alpha": 0.14,
    "inn_mapped_point_alpha": 0.26,
    "inn_feasible_fill_alpha": 0.035,
    "inn_scatter_size": 8.0,
    "inn_point_zorder": 6,
    "x_space_color": "#C44E52",
    "z_space_color": "#4F79E8",
    "feasible_fill_alpha": 0.12,
    "boundary_alpha": 0.52,
    "equality_alpha": 0.86,
    "trajectory_linewidth": 2.2,
    "trajectory_marker_size": 2.2,
    "trajectory_color": "#2CA02C",
    "start_marker_size": 9.5,
    "final_marker_size": 14.5,
    "marker_edge_width": 1.25,
    "start_marker_color": "#0047FF",
    "final_marker_color": "#FF1F1F",
    "notation_fontsize": 16,
    "notation_box_alpha": 0.8,
    "dpi": 300,
}


_PAPER_STYLE_RCPARAMS_APPLIED = False


def apply_paper_style_rcparams():
    """Apply global rcParams needed for consistent paper PDF output."""

    global _PAPER_STYLE_RCPARAMS_APPLIED
    if _PAPER_STYLE_RCPARAMS_APPLIED:
        return
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": PAPER_STYLE["font_family"],
            "font.size": PAPER_STYLE["base_fontsize"],
            "axes.labelsize": PAPER_STYLE["label_fontsize"],
            "axes.titlesize": PAPER_STYLE["title_fontsize"],
            "xtick.labelsize": PAPER_STYLE["tick_fontsize"],
            "ytick.labelsize": PAPER_STYLE["tick_fontsize"],
            "legend.fontsize": PAPER_STYLE["legend_fontsize"],
            "figure.dpi": PAPER_STYLE["dpi"],
            "savefig.dpi": PAPER_STYLE["dpi"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    _PAPER_STYLE_RCPARAMS_APPLIED = True


def configure_matplotlib_cache(
    *,
    mplconfig_dir="/tmp/homopt_mpl_config",
    xdg_cache_home="/tmp/homopt_xdg_cache",
):
    """Point matplotlib/font caches to writable locations for script runs."""

    os.makedirs(mplconfig_dir, exist_ok=True)
    os.makedirs(xdg_cache_home, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_home))


def require_matplotlib():
    configure_matplotlib_cache()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib. Install with: pip install -r requirements.txt"
        ) from exc
    apply_paper_style_rcparams()
    return plt


def apply_paper_axis_style(ax, *, grid=None):
    """Apply the paper-style frame, ticks, and grid shared by 2D plots."""

    if grid is None:
        grid = bool(PAPER_STYLE["show_grid"])
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(PAPER_STYLE["axis_linewidth"])
        spine.set_color("black")
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=PAPER_STYLE["tick_fontsize"],
        width=PAPER_STYLE["tick_width"],
        length=PAPER_STYLE["tick_length"],
        direction="out",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=PAPER_STYLE["tick_width"] * 0.8,
        length=PAPER_STYLE["tick_length"] * 0.65,
        direction="out",
    )
    if grid:
        ax.grid(
            alpha=PAPER_STYLE["grid_alpha"],
            color="gray",
            linewidth=PAPER_STYLE["grid_linewidth"],
        )
    else:
        ax.grid(False)


def set_axis_labels(ax, xlabel, ylabel, *, fontweight="normal", color=None):
    label_kwargs = {"fontsize": PAPER_STYLE["label_fontsize"], "fontweight": fontweight}
    if color is not None:
        label_kwargs["color"] = color
    ax.set_xlabel(xlabel, **label_kwargs)
    ax.set_ylabel(ylabel, **label_kwargs)


def style_legend(legend):
    if legend is None:
        return
    legend.get_frame().set_linewidth(PAPER_STYLE["axis_linewidth"] * 0.8)
    legend.get_frame().set_alpha(0.92)
    handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
    for handle in handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(PAPER_STYLE["legend_linewidth"])


__all__ = [
    "ALGORITHM_COLORS",
    "INN_PGD_METHOD_LABELS",
    "ALGORITHM_LINE_STYLES",
    "PAPER_STYLE",
    "apply_paper_axis_style",
    "apply_paper_style_rcparams",
    "configure_matplotlib_cache",
    "require_matplotlib",
    "set_axis_labels",
    "style_legend",
]
