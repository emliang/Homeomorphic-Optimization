"""Coordinate-specific bisection adapters built on the shared bisection kernel."""

from __future__ import annotations

import torch

from homopt.models.inn import _embed_condition_if_available, _forward_with_condition

from .core import BisectionConfig, bisect_segment


def homeomorphic_bisection(
    model,
    data,
    z_infeasible,
    input_params,
    args,
    eps_converge=1e-4,
    condition_emb=None,
    feasible_anchor=None,
):
    """Project toward a verified feasible latent anchor.

    By default the latent origin is the anchor.  Callers whose learned map
    does not make the origin feasible must provide a per-instance feasible
    anchor instead.  The shared kernel may also produce a midpoint estimate,
    but this adapter always returns the retained lower endpoint so callers
    never receive the infeasible side of the bracket.
    """

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

    def _is_feasible(z_candidate):
        candidate_violation = _violation(_decode_to_y(z_candidate))
        return torch.isfinite(candidate_violation) & (candidate_violation <= config.feasibility_tol)

    config = BisectionConfig.from_projection_args(args, convergence_tol=eps_converge, default_max_steps=20)
    config = BisectionConfig(
        max_steps=config.max_steps,
        feasibility_tol=config.feasibility_tol,
        convergence_tol=config.convergence_tol,
        step_fraction=config.step_fraction,
        final_alpha="lower",
    )
    if feasible_anchor is None:
        anchor = torch.zeros_like(z_infeasible)
        anchor_name = "latent origin"
    else:
        anchor = torch.as_tensor(
            feasible_anchor,
            dtype=z_infeasible.dtype,
            device=z_infeasible.device,
        )
        if anchor.ndim == 1:
            anchor = anchor.unsqueeze(0)
        if anchor.shape[0] == 1 and z_infeasible.shape[0] > 1:
            anchor = anchor.expand_as(z_infeasible)
        if anchor.shape != z_infeasible.shape:
            raise ValueError(
                "feasible_anchor must have shape (batch, latent_dim) matching "
                f"z_infeasible; got {tuple(anchor.shape)} and {tuple(z_infeasible.shape)}."
            )
        anchor_name = "provided feasible anchor"
    with torch.inference_mode():
        anchor_feasible = _is_feasible(anchor)
    if not bool(torch.all(anchor_feasible)):
        invalid_count = int((~anchor_feasible).sum().item())
        raise ValueError(
            f"homeomorphic_bisection requires the {anchor_name} to be feasible "
            f"for every instance; found {invalid_count} infeasible anchor(s)."
        )

    result = bisect_segment(
        anchor=anchor,
        target=z_infeasible,
        decode_to_y=_decode_to_y,
        violation=_violation,
        config=config,
    )
    with torch.inference_mode():
        point_feasible = _is_feasible(result.point)
        # The lower endpoint was checked by the kernel.  Retain the verified
        # origin if a fresh decode exposes numerical drift or non-determinism.
        z_feasible = torch.where(point_feasible, result.point, anchor)
    return z_feasible, result.steps


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
