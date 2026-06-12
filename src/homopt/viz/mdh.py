"""MDH visualization helpers."""

from __future__ import annotations

import numpy as np
import torch

from .artifacts import save_figure_with_suffix
from .primitives import draw_objective_residual_landscape, draw_unit_ball
from .style import (
    PAPER_STYLE,
    apply_paper_axis_style,
    require_matplotlib,
    set_axis_labels,
    style_legend,
)


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_model_runtime(model):
    param = next(model.parameters(), None)
    if param is None:
        return _default_device(), torch.float32
    return param.device, param.dtype


def _style_axis(ax, *, xlabel=None, ylabel=None, legend=False, aspect_equal=False):
    resolved_xlabel = ax.get_xlabel() if xlabel is None else xlabel
    resolved_ylabel = ax.get_ylabel() if ylabel is None else ylabel
    if resolved_xlabel or resolved_ylabel:
        set_axis_labels(ax, resolved_xlabel, resolved_ylabel)
    if aspect_equal:
        ax.set_aspect("equal", adjustable="box")
    legend_obj = ax.legend(fontsize=PAPER_STYLE["legend_fontsize"]) if legend else ax.get_legend()
    style_legend(legend_obj)
    apply_paper_axis_style(ax)


def _unit_circle_boundary_points(n_points):
    """Return evenly-spaced points on the 2D latent unit-circle boundary."""

    n_points = max(int(n_points), 64)
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True, dtype=np.float32)
    return np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)


def visualize_mdh_mapping_transformation(
    model,
    data,
    save_path=None,
    n_samples=360,
    seed=2030,
    grid_resolution=140,
):
    plt = require_matplotlib()
    device, dtype = _infer_model_runtime(model)
    if data.nvar != 2:
        return None

    test_input_samples, _ = data.generate_problem_samples(n_samples=3, seed=seed)
    test_input_samples = test_input_samples.to(device=device, dtype=dtype)
    unit_boundary_points = torch.tensor(_unit_circle_boundary_points(n_samples), dtype=dtype, device=device)
    n_boundary_points = int(unit_boundary_points.shape[0])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = list(PAPER_STYLE["inn_instance_colors"])
    instance_labels = ["Instance 1", "Instance 2", "Instance 3"]
    was_training = bool(getattr(model, "training", False))
    model.eval()

    ax_ball = axes[0]
    draw_unit_ball(ax_ball, 2, color=PAPER_STYLE["inn_unit_boundary_color"], zorder=PAPER_STYLE["inn_point_zorder"] + 1)
    ax_ball.set_xlim(-1.5, 1.5)
    ax_ball.set_ylim(-1.5, 1.5)
    ax_ball.set_aspect("equal")
    ax_ball.set_title("Homeomorphic Space")
    ax_ball.set_xlabel("z1")
    ax_ball.set_ylabel("z2")
    _style_axis(ax_ball, aspect_equal=True)

    for idx in range(3):
        boundary_input = test_input_samples[idx : idx + 1]
        boundary_input = boundary_input.expand(n_boundary_points, *[-1] * (boundary_input.dim() - 1))
        with torch.no_grad():
            transformed_boundary = model(unit_boundary_points, boundary_input)
            scaled_boundary = data.scale(boundary_input, transformed_boundary)
            final_boundary = data.complete_partial(boundary_input, scaled_boundary)
            final_boundary_np = final_boundary.cpu().numpy()

        ax_transformed = axes[idx + 1]
        x_range = np.linspace(-4.2, 4.2, int(grid_resolution))
        y_range = np.linspace(-4.2, 4.2, int(grid_resolution))
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        with torch.no_grad():
            grid_points = torch.tensor(np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1), dtype=dtype, device=device)
            grid_batch_size = len(grid_points)
            grid_input_batch = test_input_samples[idx : idx + 1]
            grid_input_batch = grid_input_batch.expand(grid_batch_size, *[-1] * (grid_input_batch.dim() - 1))
            violations = data.check_feasibility(grid_input_batch, grid_points)
            residual = violations.max(dim=1)[0].detach().cpu().numpy().reshape(X_grid.shape)

        draw_objective_residual_landscape(
            ax_transformed,
            X_grid,
            Y_grid,
            None,
            residual,
            feasible_color=PAPER_STYLE["inn_feasible_fill_color"],
            boundary_color=PAPER_STYLE["inn_constraint_boundary_color"],
            label_contours=False,
        )
        ax_transformed.plot(
            final_boundary_np[:, 0],
            final_boundary_np[:, 1],
            color=colors[idx % len(colors)],
            linewidth=PAPER_STYLE["trajectory_linewidth"],
            alpha=0.9,
            zorder=PAPER_STYLE["inn_point_zorder"],
        )
        ax_transformed.set_title(instance_labels[idx])
        ax_transformed.set_xlabel("x1")
        ax_transformed.set_ylabel("x2")
        ax_transformed.set_xlim(-4.2, 4.2)
        ax_transformed.set_ylim(-4.2, 4.2)
        _style_axis(ax_transformed, aspect_equal=True)

    plt.tight_layout()
    save_figure_with_suffix(fig, save_path, "_mdh_mapping_visualization.pdf")
    if was_training:
        model.train()
    return fig


__all__ = [
    "visualize_mdh_mapping_transformation",
]
