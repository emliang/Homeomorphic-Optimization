"""Coordinate-specific bisection adapters built on the shared bisection kernel."""

from __future__ import annotations

import torch

from homopt.models.inn import _embed_condition_if_available, _forward_with_condition

from .core import BisectionConfig, bisect_segment


def homeomorphic_bisection(model, data, z_infeasible, input_params, args, eps_converge=1e-4, condition_emb=None):
    """Bisect from the INN latent origin toward infeasible latent coordinates."""

    model.eval()
    batch_size = z_infeasible.shape[0]
    input_emb = condition_emb if condition_emb is not None else _embed_condition_if_available(model, input_params, batch_size)

    def _decode_to_y(z_candidate):
        u_mapped = _forward_with_condition(model, z_candidate, input_params, input_emb)
        y_partial = data.scale(input_params, u_mapped)
        return data.complete_partial(input_params, y_partial)

    def _violation(y_candidate):
        residual = data.check_feasibility(input_params, y_candidate)
        return torch.amax(residual, dim=1, keepdim=True)

    config = BisectionConfig.from_projection_args(args, convergence_tol=eps_converge, default_max_steps=20)
    config = BisectionConfig(
        max_steps=config.max_steps,
        feasibility_tol=config.feasibility_tol,
        convergence_tol=config.convergence_tol,
        step_fraction=config.step_fraction,
        final_alpha="midpoint",
    )
    result = bisect_segment(
        anchor=torch.zeros_like(z_infeasible),
        target=z_infeasible,
        decode_to_y=_decode_to_y,
        violation=_violation,
        config=config,
    )
    return result.point, result.steps


def interior_point_bisection(feasible_u, data, u_infeasible, input_params, args, eps_converge=1e-4):
    """Bisect in normalized decision coordinates from interior anchors to infeasible points."""

    if feasible_u.ndim == 2:
        feasible_u = feasible_u.unsqueeze(1)
    batch_size, n_ip, _ = feasible_u.shape
    input_expand = input_params.view(batch_size, 1, -1).expand(-1, n_ip, -1).reshape(-1, input_params.shape[-1])

    def _decode_to_y(u_candidate):
        y_partial = data.scale(input_expand, u_candidate)
        return data.complete_partial(input_expand, y_partial)

    def _violation(y_candidate):
        residual = data.check_feasibility(input_expand, y_candidate).abs()
        return residual.max(dim=1, keepdim=True)[0]

    config = BisectionConfig.from_projection_args(args, convergence_tol=eps_converge, default_max_steps=30)
    result = bisect_segment(
        anchor=feasible_u,
        target=u_infeasible,
        decode_to_y=_decode_to_y,
        violation=_violation,
        config=config,
    )
    y_partial = data.scale(input_params, result.point)
    y_full = data.complete_partial(input_params, y_partial)
    return y_full, result.anchor, result.steps


__all__ = ["homeomorphic_bisection", "interior_point_bisection"]
