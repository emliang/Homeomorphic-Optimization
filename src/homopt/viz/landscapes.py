"""2D landscape construction helpers for convex visualizations."""

from __future__ import annotations

import numpy as np
import torch

from .traces import as_numpy, split_convex_metrics


def plot_bounds(problem, traces, *, padding=0.35, fallback=(-2.5, 2.5)):
    """Choose an x-space plotting window that includes trajectories and box bounds."""

    has_box = hasattr(problem, "L") and hasattr(problem, "U")
    if has_box:
        lower = as_numpy(problem.L).reshape(-1)[:2]
        upper = as_numpy(problem.U).reshape(-1)[:2]
    else:
        lower = np.array([fallback[0], fallback[0]], dtype=float)
        upper = np.array([fallback[1], fallback[1]], dtype=float)
    low = np.array([fallback[0], fallback[0]], dtype=float)
    high = np.array([fallback[1], fallback[1]], dtype=float)

    try:
        x_grid = np.linspace(lower[0], upper[0], 240)
        y_grid = np.linspace(lower[1], upper[1], 240)
        X, Y = np.meshgrid(x_grid, y_grid)
        points = np.column_stack([X.reshape(-1), Y.reshape(-1)])
        metrics = split_convex_metrics(problem, points)
        feasible = metrics["inequality_violation"] <= 1e-6
        if feasible.any():
            feasible_points = points[feasible]
            low = feasible_points.min(axis=0) - padding
            high = feasible_points.max(axis=0) + padding
    except Exception:
        pass

    points = [trace["trajectory"] for trace in traces.values() if trace["trajectory"].size]
    if points:
        stacked = np.vstack(points)
        low = np.minimum(low, stacked.min(axis=0) - padding)
        high = np.maximum(high, stacked.max(axis=0) + padding)

    min_span = 1.8
    span = high - low
    for idx in range(2):
        if span[idx] < min_span:
            center = 0.5 * (low[idx] + high[idx])
            low[idx] = center - 0.5 * min_span
            high[idx] = center + 0.5 * min_span
    center = 0.5 * (low + high)
    half_span = 0.5 * float(np.max(high - low))
    low = center - half_span
    high = center + half_span
    if not has_box:
        return low, high
    clipped_low = np.maximum(low, lower)
    clipped_high = np.minimum(high, upper)
    span = clipped_high - clipped_low
    if abs(float(span[0] - span[1])) > 1e-12:
        target = min(float(np.max(span)), float(np.min(upper - lower)))
        center = 0.5 * (clipped_low + clipped_high)
        clipped_low = np.maximum(center - 0.5 * target, lower)
        clipped_high = np.minimum(center + 0.5 * target, upper)
    return clipped_low, clipped_high


def convex_landscape_grid(problem, low, high, *, grid_size=640):
    """Evaluate objective and feasibility metrics on an x-space 2D grid."""

    x_grid = np.linspace(low[0], high[0], grid_size)
    y_grid = np.linspace(low[1], high[1], grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    points = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    metrics = split_convex_metrics(problem, points)
    objective = metrics["objective"].reshape(X.shape)
    inequality = metrics["inequality_violation"].reshape(X.shape)
    equality = metrics["equality_violation"].reshape(X.shape)
    eq_resid = None
    points_tensor = None
    if getattr(problem, "n_eq", 0) == 1 and hasattr(problem, "eq_constraint_x"):
        if points_tensor is None:
            points_tensor = torch.as_tensor(points, dtype=torch.float32, device=problem.device)
        with torch.no_grad():
            eq_resid = problem.eq_constraint_x(points_tensor).detach().cpu().numpy().reshape(X.shape)
    return X, Y, objective, inequality, equality, eq_resid


def _hom_forward(hom_map, z_tensor):
    return hom_map.forward(z_tensor, method="explicit")


def convex_z_landscape_grid(problem, hom_map, *, grid_size=260, bounds=(-1.5, 1.5)):
    """Evaluate objective/equality contours after mapping a z-space grid to x-space."""

    z1_grid = np.linspace(bounds[0], bounds[1], grid_size)
    z2_grid = np.linspace(bounds[0], bounds[1], grid_size)
    Z1, Z2 = np.meshgrid(z1_grid, z2_grid)
    z_points = np.column_stack([Z1.reshape(-1), Z2.reshape(-1)])
    z_tensor = torch.as_tensor(z_points, dtype=torch.float32, device=problem.device)
    with torch.no_grad():
        x_tensor = _hom_forward(hom_map, z_tensor)
        objective = problem.objective_x(x_tensor).detach().reshape(-1).cpu().numpy().reshape(Z1.shape)
        eq_resid = (
            problem.eq_constraint_x(x_tensor).detach().reshape(-1).cpu().numpy().reshape(Z1.shape)
            if getattr(problem, "n_eq", 0) == 1 and hasattr(problem, "eq_constraint_x")
            else None
        )
    p_norm = getattr(hom_map, "p_norm", 2)
    if p_norm == np.inf:
        inside = np.maximum(np.abs(Z1), np.abs(Z2)) <= 1.0
    else:
        inside = np.linalg.norm(z_points, ord=p_norm, axis=1).reshape(Z1.shape) <= 1.0
    return Z1, Z2, objective, eq_resid, inside, p_norm


__all__ = [
    "convex_landscape_grid",
    "convex_z_landscape_grid",
    "plot_bounds",
]
