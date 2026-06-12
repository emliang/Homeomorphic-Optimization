"""CvxINN evaluation and visualization helpers."""

from __future__ import annotations

import inspect
import math

import numpy as np
import torch

from homopt.models import radial_compactify, radial_decompactify
from .artifacts import save_figure
from .style import PAPER_STYLE, apply_paper_axis_style, require_matplotlib, set_axis_labels, style_legend


def _style_panel(ax, *, xlabel=None, ylabel=None, legend=False, aspect_equal=False):
    resolved_xlabel = ax.get_xlabel() if xlabel is None else xlabel
    resolved_ylabel = ax.get_ylabel() if ylabel is None else ylabel
    if resolved_xlabel or resolved_ylabel:
        set_axis_labels(ax, resolved_xlabel, resolved_ylabel)
    if aspect_equal:
        ax.set_aspect("equal", adjustable="box")
    legend_obj = ax.legend(fontsize=PAPER_STYLE["legend_fontsize"]) if legend else ax.get_legend()
    style_legend(legend_obj)
    apply_paper_axis_style(ax)


def _infer_model_runtime(model):
    param = next(model.parameters(), None)
    if param is None:
        return torch.device("cpu"), torch.float32
    return param.device, param.dtype


def _call_constraint_x(constraint_set, x, clip=False):
    fn = constraint_set.constraint_x
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "clip" in signature.parameters:
        return fn(x, clip=clip)
    return fn(x)


def _resolve_condition_batch(condition, batch_size, *, device, dtype):
    if condition is None:
        return None
    if callable(condition):
        condition = condition(batch_size)
    if torch.is_tensor(condition):
        condition = condition.to(device=device, dtype=dtype)
        if condition.shape[0] == batch_size:
            return condition
        if condition.shape[0] == 1:
            expand_shape = [batch_size] + list(condition.shape[1:])
            return condition.expand(*expand_shape)
    return condition


def _sample_unit_ball_2d(n_samples, radius_max, seed, device, dtype):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n_samples)
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=n_samples)) * radius_max
    z = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    return torch.tensor(z, device=device, dtype=dtype)


def _sample_unit_circle_2d(n_samples, radius, device, dtype):
    theta = torch.linspace(0.0, 2.0 * math.pi, steps=n_samples + 1, device=device, dtype=dtype)[:-1]
    return torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)


def evaluate_coninn_exactness(
    model,
    constraint_set,
    *,
    condition=None,
    n_interior=2048,
    n_boundary=361,
    seed=2030,
    boundary_radius=1.0 - 1e-4,
):
    """Evaluate numerical exactness of a 2D/ND CvxINN map.

    The CvxINN construction is an exact homeomorphism from the open unit ball
    to the interior of the target convex set, assuming the inner model is
    invertible. Because the open ball excludes the boundary, this evaluator
    samples:

    - interior points with radius strictly less than 1,
    - a near-boundary ring with radius `boundary_radius < 1`.

    The returned metrics therefore certify exact feasibility and inverse
    roundtrip on the open-ball domain, and report how tightly the near-boundary
    ring approaches the target constraint boundary.
    """

    device, dtype = _infer_model_runtime(model)
    interior = _sample_unit_ball_2d(
        n_samples=n_interior,
        radius_max=min(float(boundary_radius), 1.0 - 1e-5),
        seed=seed,
        device=device,
        dtype=dtype,
    )
    boundary = _sample_unit_circle_2d(n_samples=n_boundary, radius=float(boundary_radius), device=device, dtype=dtype)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        condition_interior = _resolve_condition_batch(condition, interior.shape[0], device=device, dtype=dtype)
        condition_boundary = _resolve_condition_batch(condition, boundary.shape[0], device=device, dtype=dtype)

        x_interior = model(interior, condition_interior)
        z_back = model.inverse(x_interior, condition_interior)

        unconstrained = model.forward_unconstrained(interior, condition_interior)
        ball_after_inner = radial_compactify(unconstrained, eps=model.compactify_eps)

        x_boundary = model(boundary, condition_boundary)

        cons_interior = _call_constraint_x(constraint_set, x_interior, clip=False)
        cons_boundary = _call_constraint_x(constraint_set, x_boundary, clip=False)

    if was_training:
        model.train()

    positive_violation_interior = torch.clamp(cons_interior, min=0)
    positive_violation_boundary = torch.clamp(cons_boundary, min=0)
    max_violation_per_boundary_point = cons_boundary.max(dim=1)[0]
    boundary_slack = torch.abs(max_violation_per_boundary_point)
    roundtrip_error = torch.norm(z_back - interior, dim=1, p=2)
    inner_ball_radius = torch.norm(ball_after_inner, dim=1, p=2)

    return {
        "interior_z": interior.detach().cpu(),
        "boundary_z": boundary.detach().cpu(),
        "ball_after_inner": ball_after_inner.detach().cpu(),
        "x_interior": x_interior.detach().cpu(),
        "x_boundary": x_boundary.detach().cpu(),
        "max_constraint_violation": float(positive_violation_interior.max().item()),
        "mean_constraint_violation": float(positive_violation_interior.mean().item()),
        "max_boundary_violation": float(positive_violation_boundary.max().item()),
        "max_boundary_slack": float(boundary_slack.max().item()),
        "mean_boundary_slack": float(boundary_slack.mean().item()),
        "max_roundtrip_error": float(roundtrip_error.max().item()),
        "mean_roundtrip_error": float(roundtrip_error.mean().item()),
        "max_inner_ball_radius": float(inner_ball_radius.max().item()),
        "mean_inner_ball_radius": float(inner_ball_radius.mean().item()),
        "boundary_radius": float(boundary_radius),
        "n_interior": int(n_interior),
        "n_boundary": int(n_boundary),
    }


def visualize_coninn_exactness(
    model,
    constraint_set,
    *,
    save_path=None,
    condition=None,
    n_interior=2048,
    n_boundary=361,
    seed=2030,
    boundary_radius=1.0 - 1e-4,
    grid_resolution=300,
):
    """Visualize the CvxINN route on a 2D convex set."""

    if int(getattr(constraint_set, "nvar", -1)) != 2:
        raise ValueError("CvxINN visualization currently requires a 2D convex set.")

    plt = require_matplotlib()
    stats = evaluate_coninn_exactness(
        model,
        constraint_set,
        condition=condition,
        n_interior=n_interior,
        n_boundary=n_boundary,
        seed=seed,
        boundary_radius=boundary_radius,
    )

    z_interior = stats["interior_z"].numpy()
    z_boundary = stats["boundary_z"].numpy()
    ball_after_inner = stats["ball_after_inner"].numpy()
    x_interior = stats["x_interior"].numpy()
    x_boundary = stats["x_boundary"].numpy()

    x_min = min(float(np.min(x_interior[:, 0])), float(np.min(x_boundary[:, 0])))
    x_max = max(float(np.max(x_interior[:, 0])), float(np.max(x_boundary[:, 0])))
    y_min = min(float(np.min(x_interior[:, 1])), float(np.min(x_boundary[:, 1])))
    y_max = max(float(np.max(x_interior[:, 1])), float(np.max(x_boundary[:, 1])))
    x_pad = 0.15 * max(x_max - x_min, 1.0)
    y_pad = 0.15 * max(y_max - y_min, 1.0)

    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, grid_resolution)
    y_grid = np.linspace(y_min - y_pad, y_max + y_pad, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        cons_grid = _call_constraint_x(constraint_set, grid_points, clip=False)
        feas_grid = (cons_grid.max(dim=1)[0] <= 1e-6).cpu().numpy().reshape(X.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    theta = np.linspace(0.0, 2.0 * math.pi, 512)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    axes[0].scatter(z_interior[:, 0], z_interior[:, 1], s=5, alpha=0.20, color="#4C78A8", linewidths=0)
    axes[0].plot(circle[:, 0], circle[:, 1], "--", color="black", linewidth=1.0)
    axes[0].plot(z_boundary[:, 0], z_boundary[:, 1], color="#E45756", linewidth=1.5)
    axes[0].set_title("Input Unit Ball")
    axes[0].set_xlabel("z1")
    axes[0].set_ylabel("z2")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    _style_panel(axes[0], aspect_equal=True)

    axes[1].scatter(ball_after_inner[:, 0], ball_after_inner[:, 1], s=5, alpha=0.20, color="#72B7B2", linewidths=0)
    axes[1].plot(circle[:, 0], circle[:, 1], "--", color="black", linewidth=1.0)
    axes[1].set_title("After INN, Back in Ball")
    axes[1].set_xlabel("u1")
    axes[1].set_ylabel("u2")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-1.1, 1.1)
    axes[1].set_ylim(-1.1, 1.1)
    _style_panel(axes[1], aspect_equal=True)

    axes[2].contourf(X, Y, feas_grid.astype(float), levels=[-0.1, 0.5, 1.1], colors=["white", "#D5E8D4"], alpha=0.5)
    axes[2].contour(X, Y, feas_grid.astype(float), levels=[0.5], colors=["#2E7D32"], linewidths=1.2)
    axes[2].scatter(x_interior[:, 0], x_interior[:, 1], s=5, alpha=0.18, color="#4C78A8", linewidths=0)
    axes[2].plot(x_boundary[:, 0], x_boundary[:, 1], color="#E45756", linewidth=1.5, label="near-boundary image")
    axes[2].set_title("Mapped Convex Set")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    axes[2].set_aspect("equal")
    _style_panel(axes[2], legend=True, aspect_equal=True)

    fig.suptitle(
        (
            "CvxINN exactness on the open ball: "
            f"max feas viol={stats['max_constraint_violation']:.2e}, "
            f"max roundtrip err={stats['max_roundtrip_error']:.2e}, "
            f"max boundary slack={stats['max_boundary_slack']:.2e}"
        ),
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        output_path = save_figure(fig, save_path)
        return fig, stats, str(output_path)
    return fig, stats, None
