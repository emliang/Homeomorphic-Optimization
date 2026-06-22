"""General convex gauge boundary search helpers."""

from __future__ import annotations

import torch

from .math import _safe_denominator


def _as_constraint_matrix(values, *, batch_size):
    if values.ndim == 1:
        values = values.view(batch_size, 1)
    if values.ndim != 2 or values.shape[0] != batch_size:
        raise ValueError("General convex constraint values must have shape (batch, n_constraints).")
    return values


def _as_constraint_gradient_tensor(values, *, batch_size, n_constraints, nvar):
    if values.ndim == 2:
        values = values.view(batch_size, 1, nvar)
    if values.shape != (batch_size, n_constraints, nvar):
        raise ValueError("General convex constraint gradients must have shape (batch, n_constraints, nvar).")
    return values


def general_boundary_candidates(
    u,
    *,
    center,
    constraint_fn,
    gradient_fn=None,
    eps,
    initial_radius=1.0,
    max_radius=1e6,
    expand_steps=40,
    iterations=60,
    need_gradient=True,
):
    """Return beta and optional d beta / d u for general convex constraints.

    Each constraint is searched independently along `center + rho * u`.
    Invalid or unbounded candidates are returned as `-inf` beta values.
    """

    batch_size, nvar = u.shape
    device, dtype = u.device, u.dtype
    center = center.to(device=device, dtype=dtype)
    center_batch = center.expand(batch_size, -1)
    center_values = _as_constraint_matrix(constraint_fn(center_batch), batch_size=batch_size).to(
        device=device, dtype=dtype
    )
    if (center_values >= -eps).any():
        raise ValueError("GaugeMap general convex constraints require a strictly feasible center.")

    n_constraints = center_values.shape[1]
    constraint_index = torch.arange(n_constraints, device=device).view(1, -1).expand(batch_size, -1).reshape(-1)

    def _selected_values(radius):
        points = center.view(1, 1, nvar) + radius.unsqueeze(-1) * u.view(batch_size, 1, nvar)
        flat_points = points.reshape(batch_size * n_constraints, nvar)
        flat_values = _as_constraint_matrix(
            constraint_fn(flat_points),
            batch_size=batch_size * n_constraints,
        ).to(device=device, dtype=dtype)
        return flat_values.gather(1, constraint_index.view(-1, 1)).view(batch_size, n_constraints)

    lower = torch.zeros(batch_size, n_constraints, device=device, dtype=dtype)
    upper = torch.full_like(lower, float(initial_radius))
    upper_values = _selected_values(upper)
    hit = upper_values >= 0

    for _ in range(int(expand_steps)):
        unresolved = (~hit) & (upper < float(max_radius))
        if not unresolved.any():
            break
        upper = torch.where(unresolved, (2.0 * upper).clamp_max(float(max_radius)), upper)
        upper_values = torch.where(unresolved, _selected_values(upper), upper_values)
        hit = hit | (upper_values >= 0)

    for _ in range(int(iterations)):
        mid = 0.5 * (lower + upper)
        mid_values = _selected_values(mid)
        above = mid_values >= 0
        lower = torch.where(hit & (~above), mid, lower)
        upper = torch.where(hit & above, mid, upper)

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    beta = torch.where(hit, torch.ones_like(upper) / upper.clamp_min(eps), neg_inf)
    if not need_gradient:
        return beta, None
    if gradient_fn is None:
        raise ValueError("General convex explicit gauge gradients require a gradient function.")

    boundary_points = center.view(1, 1, nvar) + upper.unsqueeze(-1) * u.view(batch_size, 1, nvar)
    flat_boundary = boundary_points.reshape(batch_size * n_constraints, nvar)
    flat_gradients = _as_constraint_gradient_tensor(
        gradient_fn(flat_boundary),
        batch_size=batch_size * n_constraints,
        n_constraints=n_constraints,
        nvar=nvar,
    ).to(device=device, dtype=dtype)
    gather_shape = (batch_size * n_constraints, 1, 1)
    gather_index = constraint_index.view(*gather_shape).expand(-1, 1, nvar)
    grad_x = flat_gradients.gather(1, gather_index).view(batch_size, n_constraints, nvar)
    radial_dot = (grad_x * u.view(batch_size, 1, nvar)).sum(dim=-1, keepdim=True)
    grad_u = beta.unsqueeze(-1) * grad_x / _safe_denominator(radial_dot, eps=eps)
    grad_u = torch.where(hit.unsqueeze(-1), grad_u, torch.zeros_like(grad_u))
    return beta, grad_u
