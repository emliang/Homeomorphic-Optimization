"""Shared 2D grid and batching helpers for visualization evaluators."""

from __future__ import annotations

import numpy as np


def iter_point_batches(n_points, batch_size):
    """Yield ``(start, end)`` slices for evaluating dense visualization grids."""

    batch_size = int(batch_size) if batch_size is not None else int(n_points)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, int(n_points), batch_size):
        yield start, min(start + batch_size, int(n_points))


def grid_points_from_bounds(low, high, grid_size):
    """Build a 2D mesh and its flattened point matrix from coordinate bounds."""

    x_grid = np.linspace(low[0], high[0], int(grid_size))
    y_grid = np.linspace(low[1], high[1], int(grid_size))
    X, Y = np.meshgrid(x_grid, y_grid)
    points = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    return X, Y, points


__all__ = [
    "grid_points_from_bounds",
    "iter_point_batches",
]
