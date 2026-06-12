"""QCQP trajectory landscape helpers for INN visualizations."""

from __future__ import annotations

import numpy as np
import torch

from .inn_records import _as_numpy_1d
from .primitives import draw_objective_residual_landscape, draw_unit_ball
from .style import PAPER_STYLE, apply_paper_axis_style, set_axis_labels

QCQP_LATENT_TRAJECTORY_VIEW_LIM = 1.5


def _problem_tensor_kwargs(problem):
    tensor = getattr(problem, "fixed_Q", None)
    if torch.is_tensor(tensor):
        return {"device": tensor.device, "dtype": tensor.dtype}
    return {
        "device": getattr(problem, "device", torch.device("cpu")),
        "dtype": torch.float32,
    }


def _expand_input_params(input_params, n_points, *, device, dtype):
    if input_params is None:
        return None
    if not torch.is_tensor(input_params):
        input_params = torch.as_tensor(input_params, device=device, dtype=dtype)
    else:
        input_params = input_params.to(device=device, dtype=dtype)
    if input_params.ndim == 1:
        input_params = input_params.view(1, -1)
    first = input_params[:1]
    return first.expand(n_points, *first.shape[1:])


def _evaluate_objective(problem, input_batch, points, objective_params=None):
    if objective_params is not None and hasattr(problem, "objective_xy"):
        tensor_kwargs = {"device": points.device, "dtype": points.dtype}
        objective_batch = _expand_input_params(objective_params, points.shape[0], **tensor_kwargs)
        return problem.objective_xy(input_batch, points, objective_batch).reshape(-1)
    if hasattr(problem, "objective"):
        return problem.objective(points).reshape(-1)
    return problem.objective_xy(input_batch, points).reshape(-1)


def _trajectory_bounds(problem, traj, *, padding=0.25):
    if hasattr(problem, "fixed_L") and hasattr(problem, "fixed_U"):
        lower = _as_numpy_1d(problem.fixed_L)[:2]
        upper = _as_numpy_1d(problem.fixed_U)[:2]
    else:
        lower = np.nanmin(traj[:, :2], axis=0) - 1.0
        upper = np.nanmax(traj[:, :2], axis=0) + 1.0
    low = np.minimum(lower, np.nanmin(traj[:, :2], axis=0) - padding)
    high = np.maximum(upper, np.nanmax(traj[:, :2], axis=0) + padding)
    center = 0.5 * (low + high)
    half_span = 0.5 * max(float(np.max(high - low)), 1.0)
    return center - half_span, center + half_span


def _iter_point_batches(n_points, batch_size):
    batch_size = int(batch_size) if batch_size is not None else int(n_points)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, int(n_points), batch_size):
        yield start, min(start + batch_size, int(n_points))


def _constraint_residual(problem, input_batch, x_points):
    if input_batch is not None and hasattr(problem, "constraint_residual_xy"):
        return problem.constraint_residual_xy(input_batch, x_points, clip=False)
    if input_batch is not None and hasattr(problem, "ineq_resid"):
        return problem.ineq_resid(input_batch, x_points)
    if input_batch is not None and hasattr(problem, "check_feasibility"):
        return problem.check_feasibility(input_batch, x_points)
    if hasattr(problem, "constraint_x"):
        return problem.constraint_x(x_points, clip=False)
    raise ValueError("QCQP trajectory visualization requires a raw inequality residual API.")


def _evaluate_qcqp_points(problem, input_batch, x_points, *, objective_params=None):
    objective = _evaluate_objective(problem, input_batch, x_points, objective_params=objective_params)
    residual = _constraint_residual(problem, input_batch, x_points)
    max_residual = residual.reshape(x_points.shape[0], -1).max(dim=1).values
    return objective.reshape(-1), max_residual.reshape(-1)


def _evaluate_qcqp_grid(problem, input_params, points_np, *, objective_params=None, batch_size=None):
    tensor_kwargs = _problem_tensor_kwargs(problem)
    objective_np = np.empty(points_np.shape[0], dtype=float)
    residual_np = np.empty(points_np.shape[0], dtype=float)
    with torch.no_grad():
        for start, end in _iter_point_batches(points_np.shape[0], batch_size):
            points = torch.as_tensor(points_np[start:end], **tensor_kwargs)
            input_batch = _expand_input_params(input_params, points.shape[0], **tensor_kwargs)
            objective, max_residual = _evaluate_qcqp_points(
                problem,
                input_batch,
                points,
                objective_params=objective_params,
            )
            objective_np[start:end] = objective.detach().cpu().numpy()
            residual_np[start:end] = max_residual.detach().cpu().numpy()
    return objective_np, residual_np


def _grid_points_from_bounds(low, high, grid_size):
    x_grid = np.linspace(low[0], high[0], int(grid_size))
    y_grid = np.linspace(low[1], high[1], int(grid_size))
    X, Y = np.meshgrid(x_grid, y_grid)
    points_np = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    return X, Y, points_np


def _compute_qcqp_landscape(
    problem,
    input_params,
    traj,
    *,
    objective_params=None,
    grid_size=720,
    eval_batch_size=None,
):
    low, high = _trajectory_bounds(problem, traj)
    X, Y, points_np = _grid_points_from_bounds(low, high, grid_size)
    objective_np, residual_np = _evaluate_qcqp_grid(
        problem,
        input_params,
        points_np,
        objective_params=objective_params,
        batch_size=eval_batch_size,
    )
    return {
        "X": X,
        "Y": Y,
        "objective": objective_np.reshape(X.shape),
        "residual": residual_np.reshape(X.shape),
    }

def _evaluate_latent_qcqp_grid(
    problem,
    model,
    input_params,
    z_points,
    mesh_shape,
    *,
    objective_params=None,
    batch_size=None,
):
    objective_np = np.empty(z_points.shape[0], dtype=float)
    residual_np = np.empty(z_points.shape[0], dtype=float)
    tensor_kwargs = _problem_tensor_kwargs(problem)
    with torch.no_grad():
        for start, end in _iter_point_batches(z_points.shape[0], batch_size):
            z_tensor = torch.as_tensor(z_points[start:end], **tensor_kwargs)
            input_batch = _expand_input_params(input_params, z_tensor.shape[0], **tensor_kwargs)
            if input_batch is None:
                return None, None
            x_points = model(z_tensor, input_batch)
            if hasattr(problem, "scale"):
                x_points = problem.scale(input_batch, x_points)
            if hasattr(problem, "complete_partial"):
                x_points = problem.complete_partial(input_batch, x_points)
            objective, max_residual = _evaluate_qcqp_points(
                problem,
                input_batch,
                x_points,
                objective_params=objective_params,
            )
            objective_np[start:end] = objective.detach().cpu().numpy()
            residual_np[start:end] = max_residual.detach().cpu().numpy()
    return objective_np.reshape(mesh_shape), residual_np.reshape(mesh_shape)


def _compute_latent_landscape(
    problem,
    model,
    input_params,
    traj,
    *,
    objective_params=None,
    grid_size=720,
    eval_batch_size=None,
    view_lim=QCQP_LATENT_TRAJECTORY_VIEW_LIM,
):
    del traj
    lim = float(view_lim)
    if not np.isfinite(lim) or lim <= 0.0:
        raise ValueError("view_lim must be a positive finite value.")
    z_grid = np.linspace(-lim, lim, int(grid_size))
    Z1, Z2 = np.meshgrid(z_grid, z_grid)
    z_points = np.column_stack([Z1.reshape(-1), Z2.reshape(-1)])
    landscape = {"X": Z1, "Y": Z2, "objective": None, "residual": None, "lim": lim}
    if model is None or input_params is None:
        return landscape

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        objective_np, residual_np = _evaluate_latent_qcqp_grid(
            problem,
            model,
            input_params,
            z_points,
            Z1.shape,
            objective_params=objective_params,
            batch_size=eval_batch_size,
        )
        landscape["objective"] = objective_np
        landscape["residual"] = residual_np
    finally:
        if was_training:
            model.train()
    return landscape


def _draw_latent_landscape_from_data(ax, landscape):
    objective_np = landscape.get("objective")
    residual_np = landscape.get("residual")
    if objective_np is not None and residual_np is not None:
        draw_objective_residual_landscape(
            ax,
            landscape["X"],
            landscape["Y"],
            objective_np,
            residual_np,
            label_contours=False,
            image_fill=True,
        )

    lim = float(landscape.get("lim", 1.5))
    draw_unit_ball(ax, 2, color=PAPER_STYLE["inn_unit_boundary_color"], zorder=PAPER_STYLE["inn_point_zorder"] + 1)
    ax.set_title("Homeomorphic Space", fontsize=PAPER_STYLE["title_fontsize"])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    set_axis_labels(ax, r"$z_1$", r"$z_2$", fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    apply_paper_axis_style(ax)
