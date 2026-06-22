"""Reusable drawing primitives for paper-style 2D visualizations."""

from __future__ import annotations

import numpy as np
import warnings

from .style import (
    PAPER_STYLE,
    apply_paper_axis_style,
    set_axis_labels,
)


def _closed_contour_vertices(ax, X, Y, values, level):
    contour = ax.contour(X, Y, values, levels=[level], colors="black", alpha=0.0, linewidths=0.0)
    vertices = []
    if hasattr(contour, "collections"):
        contour_paths = [
            path
            for collection in contour.collections
            for path in collection.get_paths()
        ]
    else:
        contour_paths = [
            segment
            for level_segments in getattr(contour, "allsegs", [])
            for segment in level_segments
        ]
    for path_or_segment in contour_paths:
        if hasattr(path_or_segment, "vertices"):
            points = np.asarray(path_or_segment.vertices, dtype=float)
        else:
            points = np.asarray(path_or_segment, dtype=float)
        if points.shape[0] < 4:
            continue
        span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
        if np.linalg.norm(points[0] - points[-1]) <= 1e-3 * span:
            points[-1] = points[0]
            vertices.append(points)
    contour.remove()
    return vertices


def _polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])))


def _smooth_closed_vertices(vertices):
    points = np.asarray(vertices, dtype=float)
    if points.shape[0] < 8:
        return points
    closed_points = points[:-1]
    keep = np.ones(closed_points.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(closed_points, axis=0), axis=1) > 1e-10
    closed_points = closed_points[keep]
    endpoint_tol = 1e-6 * max(
        float(np.ptp(closed_points[:, 0])) if closed_points.size else 1.0,
        float(np.ptp(closed_points[:, 1])) if closed_points.size else 1.0,
        1.0,
    )
    if closed_points.shape[0] >= 2 and np.linalg.norm(closed_points[-1] - closed_points[0]) <= endpoint_tol:
        closed_points = closed_points[:-1]
    if closed_points.shape[0] < 4:
        return points
    try:
        from scipy.interpolate import splprep, splev
    except ImportError:
        return points

    k = min(3, closed_points.shape[0] - 1)
    span = max(float(np.ptp(closed_points[:, 0])), float(np.ptp(closed_points[:, 1])), 1.0)
    smoothness = 1e-5 * closed_points.shape[0] * span * span
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Setting x.*", category=RuntimeWarning)
        tck, _ = splprep([closed_points[:, 0], closed_points[:, 1]], s=smoothness, per=True, k=k)
    sample_count = max(360, closed_points.shape[0] * 4)
    xs, ys = splev(np.linspace(0.0, 1.0, sample_count), tck)
    smoothed = np.column_stack([xs, ys])
    return np.vstack([smoothed, smoothed[0]])


def _draw_feasible_region(
    ax,
    X,
    Y,
    inequality,
    color,
    feasibility_tol,
    *,
    boundary_color=None,
    fill_alpha=None,
    boundary_alpha=None,
):
    """Draw a unified feasible-set boundary without exposing grid stair-steps."""

    from matplotlib.patches import Polygon

    boundary_color = boundary_color or color
    fill_alpha = PAPER_STYLE["feasible_fill_alpha"] if fill_alpha is None else fill_alpha
    boundary_alpha = PAPER_STYLE["boundary_alpha"] if boundary_alpha is None else boundary_alpha
    raw_boundaries = _closed_contour_vertices(ax, X, Y, inequality, feasibility_tol)
    raw_boundaries = [boundary for boundary in raw_boundaries if _polygon_area(boundary) > 1e-8]
    if not raw_boundaries:
        ax.contourf(
            X,
            Y,
            inequality <= feasibility_tol,
            levels=[0.5, 1.5],
            colors=[color],
            alpha=fill_alpha,
            zorder=0,
        )
        ax.contour(
            X,
            Y,
            inequality,
            levels=[feasibility_tol],
            colors=boundary_color,
            linewidths=PAPER_STYLE["constraint_linewidth"],
            alpha=boundary_alpha,
            zorder=3,
        )
        return

    for boundary in sorted(raw_boundaries, key=_polygon_area, reverse=True):
        smooth_boundary = _smooth_closed_vertices(boundary)
        ax.add_patch(
            Polygon(
                smooth_boundary,
                closed=True,
                facecolor=color,
                edgecolor="none",
                alpha=fill_alpha,
                zorder=0,
            )
        )
        ax.plot(
            smooth_boundary[:, 0],
            smooth_boundary[:, 1],
            color=boundary_color,
            linewidth=PAPER_STYLE["constraint_linewidth"],
            alpha=boundary_alpha,
            zorder=3,
        )


def draw_objective_residual_landscape(
    ax,
    X,
    Y,
    objective,
    residual,
    *,
    objective_levels=10,
    feasibility_level=0.0,
    feasible_color=None,
    boundary_color=None,
    fill_alpha=None,
    boundary_alpha=0.95,
    label_contours=True,
    image_fill=False,
):
    """Draw objective contours and the feasible set defined by residual <= level."""

    feasible_color = feasible_color or PAPER_STYLE["inn_feasible_fill_color"]
    boundary_color = boundary_color or PAPER_STYLE["inn_constraint_boundary_color"]
    fill_alpha = PAPER_STYLE["inn_feasible_fill_alpha"] if fill_alpha is None else fill_alpha
    objective = None if objective is None else np.asarray(objective, dtype=float)
    residual = np.asarray(residual, dtype=float)

    if image_fill:
        import matplotlib.colors as mcolors

        feasible = (residual <= float(feasibility_level)).astype(float)
        fill_color = np.array(mcolors.to_rgba(feasible_color))
        image = np.ones((*feasible.shape, 4), dtype=float)
        image[..., :3] = fill_color[:3]
        image[..., 3] = feasible * fill_alpha
        ax.imshow(
            image,
            origin="lower",
            extent=[float(X.min()), float(X.max()), float(Y.min()), float(Y.max())],
            interpolation="nearest",
            zorder=0,
        )
    else:
        _draw_feasible_region(
            ax,
            X,
            Y,
            residual,
            feasible_color,
            float(feasibility_level),
            boundary_color=boundary_color,
            fill_alpha=fill_alpha,
            boundary_alpha=boundary_alpha,
        )

    finite_objective = np.array([]) if objective is None else objective[np.isfinite(objective)]
    if objective is not None and finite_objective.size and float(np.ptp(finite_objective)) > 0.0:
        contour = ax.contour(
            X,
            Y,
            objective,
            levels=objective_levels,
            colors=PAPER_STYLE["objective_contour_color"],
            alpha=PAPER_STYLE["objective_contour_alpha"],
            linewidths=PAPER_STYLE["objective_contour_linewidth"],
            zorder=1,
        )
        if label_contours:
            ax.clabel(contour, inline=True, fontsize=PAPER_STYLE["contour_label_fontsize"], fmt="%.2g")

    finite_residual = residual[np.isfinite(residual)]
    if image_fill and finite_residual.size and float(np.nanmin(finite_residual)) <= feasibility_level <= float(np.nanmax(finite_residual)):
        ax.contour(
            X,
            Y,
            residual,
            levels=[feasibility_level],
            colors=[boundary_color],
            linewidths=PAPER_STYLE["constraint_linewidth"],
            alpha=boundary_alpha,
            zorder=3,
        )


def draw_convex_landscape(
    ax,
    X,
    Y,
    objective,
    inequality,
    equality,
    eq_resid,
    *,
    objective_levels=12,
    feasibility_tol=1e-5,
):
    """Draw x-space objective contours, inequality region, and equality boundary."""

    x_color = PAPER_STYLE["x_space_color"]
    contour = ax.contour(
        X,
        Y,
        objective,
        levels=objective_levels,
        colors=PAPER_STYLE["objective_contour_color"],
        alpha=PAPER_STYLE["objective_contour_alpha"],
        linewidths=PAPER_STYLE["objective_contour_linewidth"],
    )
    ax.clabel(contour, inline=True, fontsize=PAPER_STYLE["contour_label_fontsize"], fmt="%.2g")
    _draw_feasible_region(ax, X, Y, inequality, x_color, feasibility_tol)
    if eq_resid is not None:
        ax.contour(
            X,
            Y,
            eq_resid,
            levels=[0.0],
            colors=x_color,
            linewidths=PAPER_STYLE["equality_linewidth"],
            alpha=PAPER_STYLE["equality_alpha"],
            zorder=4,
        )
    elif np.max(equality) > 0:
        ax.contour(
            X,
            Y,
            equality,
            levels=[1e-3],
            colors=x_color,
            linewidths=PAPER_STYLE["equality_linewidth"],
            alpha=PAPER_STYLE["equality_alpha"],
            zorder=4,
        )
    set_axis_labels(ax, r"$x_1$", r"$x_2$", fontweight="bold", color=x_color)
    ax.set_aspect("equal", adjustable="box")
    apply_paper_axis_style(ax)


def draw_unit_ball(ax, p_norm, *, color="#E83947", zorder=3):
    """Draw the z-space unit ball boundary for common Lp norms."""

    if p_norm == np.inf:
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
        ax.plot(
            square[:, 0],
            square[:, 1],
            color=color,
            linewidth=PAPER_STYLE["constraint_linewidth"],
            alpha=PAPER_STYLE["boundary_alpha"],
            zorder=zorder,
        )
        return

    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    if p_norm == 2:
        boundary_x = np.cos(theta)
        boundary_y = np.sin(theta)
    else:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        denom = (np.abs(cos_t) ** p_norm + np.abs(sin_t) ** p_norm) ** (1.0 / p_norm)
        boundary_x = cos_t / denom
        boundary_y = sin_t / denom
    ax.plot(
        boundary_x,
        boundary_y,
        color=color,
        linewidth=PAPER_STYLE["constraint_linewidth"],
        alpha=PAPER_STYLE["boundary_alpha"],
        zorder=zorder,
    )


def draw_convex_z_landscape(
    ax,
    Z1,
    Z2,
    objective,
    eq_resid,
    inside,
    p_norm,
    *,
    objective_levels=12,
):
    """Draw z-space objective contours, latent feasible set, and mapped equality boundary."""

    z_color = PAPER_STYLE["z_space_color"]
    contour = ax.contour(
        Z1,
        Z2,
        objective,
        levels=objective_levels,
        colors=PAPER_STYLE["objective_contour_color"],
        alpha=PAPER_STYLE["objective_contour_alpha"],
        linewidths=PAPER_STYLE["objective_contour_linewidth"],
    )
    ax.clabel(contour, inline=True, fontsize=PAPER_STYLE["contour_label_fontsize"], fmt="%.2g")
    ax.contourf(
        Z1,
        Z2,
        inside,
        levels=[0.5, 1.5],
        colors=[z_color],
        alpha=PAPER_STYLE["feasible_fill_alpha"],
        zorder=0,
    )
    draw_unit_ball(ax, p_norm, color=z_color)
    if eq_resid is not None:
        eq_values = np.asarray(eq_resid).reshape(-1)
        if eq_values.size and eq_values.min() <= 0.0 <= eq_values.max():
            ax.contour(
                Z1,
                Z2,
                eq_resid,
                levels=[0.0],
                colors=z_color,
                linewidths=PAPER_STYLE["equality_linewidth"],
                alpha=PAPER_STYLE["equality_alpha"],
                zorder=4,
            )
    set_axis_labels(ax, r"$z_1$", r"$z_2$", fontweight="bold", color=z_color)
    ax.set_aspect("equal", adjustable="box")
    apply_paper_axis_style(ax)


def draw_constraint_notation(ax, *, space):
    """Draw optional feasible-set/equality notation boxes for 2D trajectory plots."""

    if space == "z":
        color = PAPER_STYLE["z_space_color"]
        annotations = (
            (0.58, 0.12, r"$\mathbf{z} \in \mathcal{B}$", "bottom", "right"),
            (0.03, 0.97, r"$\phi(\mathbf{z}) = 0$", "top", "left"),
        )
    elif space == "x":
        color = PAPER_STYLE["x_space_color"]
        annotations = (
            (0.64, 0.14, r"$\mathbf{x} \in \mathcal{C}$", "bottom", "right"),
            (0.03, 0.97, r"$\varphi(\mathbf{x}) = 0$", "top", "left"),
        )
    else:
        raise ValueError(f"Unsupported notation space: {space}")

    box = {
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": PAPER_STYLE["notation_box_alpha"],
        "edgecolor": color,
    }
    for x_pos, y_pos, text, va, ha in annotations:
        ax.text(
            x_pos,
            y_pos,
            text,
            transform=ax.transAxes,
            fontsize=PAPER_STYLE["notation_fontsize"],
            verticalalignment=va,
            horizontalalignment=ha,
            color=color,
            fontweight="bold",
            bbox=box,
            zorder=8,
        )


def draw_trajectory(
    ax,
    traj,
    *,
    color=None,
    linestyle="-",
    marker=None,
    label=None,
    start_label=None,
    final_label=None,
    zorder=5,
):
    """Draw a trajectory with consistent start/final markers."""

    if traj.size == 0:
        return
    line_kwargs = {
        "linestyle": linestyle,
        "color": color or PAPER_STYLE["trajectory_color"],
        "linewidth": PAPER_STYLE["trajectory_linewidth"],
        "alpha": 0.92,
        "label": label,
        "zorder": zorder,
    }
    if marker is not None:
        line_kwargs["marker"] = marker
        line_kwargs["markersize"] = PAPER_STYLE["trajectory_marker_size"]
    ax.plot(traj[:, 0], traj[:, 1], **line_kwargs)
    ax.plot(
        traj[0, 0],
        traj[0, 1],
        marker="o",
        color=PAPER_STYLE["start_marker_color"],
        markeredgecolor="black",
        markeredgewidth=PAPER_STYLE["marker_edge_width"],
        markersize=PAPER_STYLE["start_marker_size"],
        label=start_label,
        zorder=zorder + 1,
    )
    ax.plot(
        traj[-1, 0],
        traj[-1, 1],
        marker="*",
        color=PAPER_STYLE["final_marker_color"],
        markeredgecolor="black",
        markeredgewidth=PAPER_STYLE["marker_edge_width"],
        markersize=PAPER_STYLE["final_marker_size"],
        label=final_label,
        zorder=zorder + 1,
    )


from .convergence import draw_metric_convergence, plot_metric_convergence


__all__ = [
    "draw_convex_landscape",
    "draw_convex_z_landscape",
    "draw_objective_residual_landscape",
    "draw_metric_convergence",
    "draw_constraint_notation",
    "draw_trajectory",
    "draw_unit_ball",
    "plot_metric_convergence",
]
