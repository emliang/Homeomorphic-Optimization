"""Shared segment-bisection utilities for learning refiners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class BisectionConfig:
    """Configuration for batched segment bisection in an arbitrary coordinate space.

    ``final_alpha="lower"`` returns the retained feasible endpoint.  A
    ``"midpoint"`` is only a bracket estimate and can be infeasible; consult
    :attr:`BisectionResult.feasible_mask` before using it as a projection.
    """

    max_steps: int = 30
    feasibility_tol: float = 1e-5
    convergence_tol: float = 1e-4
    step_fraction: float = 0.5
    final_alpha: str = "lower"

    @classmethod
    def from_projection_args(cls, args, *, convergence_tol=None, default_max_steps=30):
        args = dict(args or {})
        return cls(
            max_steps=int(args.get("proj_max_steps", default_max_steps)),
            feasibility_tol=float(args.get("proj_eps", 1e-5)),
            convergence_tol=float(
                args.get("eps_converge", 1e-4) if convergence_tol is None else convergence_tol
            ),
            step_fraction=float(args.get("step_size", 0.5)),
        )

    def normalized(self):
        max_steps = int(self.max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive for bisection.")
        step_fraction = min(max(float(self.step_fraction), 1e-6), 1.0 - 1e-6)
        convergence_tol = max(float(self.convergence_tol), 0.0)
        final_alpha = str(self.final_alpha).strip().lower()
        if final_alpha not in {"lower", "midpoint"}:
            raise ValueError("final_alpha must be 'lower' or 'midpoint'.")
        return BisectionConfig(
            max_steps=max_steps,
            feasibility_tol=float(self.feasibility_tol),
            convergence_tol=convergence_tol,
            step_fraction=step_fraction,
            final_alpha=final_alpha,
        )


@dataclass
class BisectionResult:
    """Result of batched segment bisection."""

    point: torch.Tensor
    anchor: torch.Tensor
    steps: int
    feasible_mask: torch.Tensor


def bisect_segment(
    *,
    anchor: torch.Tensor,
    target: torch.Tensor,
    decode_to_y: Callable[[torch.Tensor], torch.Tensor],
    violation: Callable[[torch.Tensor], torch.Tensor],
    config: BisectionConfig,
) -> BisectionResult:
    """Bisect segments from feasible anchors to target coordinates.

    ``anchor`` may contain one or more anchors per instance with shape
    ``(batch, dim)`` or ``(batch, n_anchor, dim)``. ``target`` has shape
    ``(batch, dim)``. The bisection is performed in the coordinate space of
    ``anchor`` and ``target``; feasibility is evaluated after ``decode_to_y``.
    """

    config = config.normalized()
    target = _as_batched_points(target)
    anchor = _as_anchor_points(anchor, target)

    batch_size, n_anchor, coord_dim = anchor.shape
    target_expand = target.view(batch_size, 1, coord_dim).expand(-1, n_anchor, -1)
    anchor_flat = anchor.reshape(batch_size * n_anchor, coord_dim)
    target_flat = target_expand.reshape(batch_size * n_anchor, coord_dim)

    with torch.inference_mode():
        target_violation = _as_column(violation(decode_to_y(target_flat)))
        target_feasible = target_violation <= config.feasibility_tol
        alpha_lower = torch.where(
            target_feasible,
            torch.ones_like(target_violation),
            torch.zeros_like(target_violation),
        )
        alpha_upper = torch.ones_like(alpha_lower)

        steps_done = 0
        for step in range(config.max_steps):
            alpha = (1.0 - config.step_fraction) * alpha_lower + config.step_fraction * alpha_upper
            candidate = anchor_flat + alpha * (target_flat - anchor_flat)
            candidate_violation = _as_column(violation(decode_to_y(candidate)))
            feasible = candidate_violation <= config.feasibility_tol
            alpha_lower = torch.where(feasible, alpha, alpha_lower)
            alpha_upper = torch.where(feasible, alpha_upper, alpha)
            steps_done = step + 1
            if config.convergence_tol > 0 and (alpha_upper - alpha_lower).max() < config.convergence_tol:
                break

        if config.final_alpha == "midpoint":
            alpha_final = 0.5 * (alpha_lower + alpha_upper)
        else:
            alpha_final = alpha_lower
        coordinate = anchor_flat + alpha_final * (target_flat - anchor_flat)
        coordinate = coordinate.view(batch_size, n_anchor, coord_dim)
        alpha_lower = alpha_lower.view(batch_size, n_anchor)
        distance = torch.linalg.norm(coordinate - target_expand, dim=-1)
        selected_anchor_idx = torch.argmin(distance, dim=1).view(batch_size, 1)
        selected = selected_anchor_idx.view(batch_size, 1, 1).expand(-1, 1, coord_dim)
        point = torch.gather(coordinate, 1, selected).view(batch_size, coord_dim)
        selected_anchor = torch.gather(anchor, 1, selected).view(batch_size, coord_dim)
        selected_violation = _as_column(violation(decode_to_y(point)))
        selected_mask = torch.isfinite(selected_violation) & (selected_violation <= config.feasibility_tol)

    return BisectionResult(point=point, anchor=selected_anchor, steps=steps_done, feasible_mask=selected_mask)


def _as_batched_points(points):
    if points.ndim == 1:
        return points.view(1, -1)
    if points.ndim != 2:
        raise ValueError(f"Expected target shape (batch, dim), got {tuple(points.shape)}.")
    return points


def _as_anchor_points(anchor, target):
    if anchor.ndim == 1:
        anchor = anchor.view(1, 1, -1)
    elif anchor.ndim == 2:
        anchor = anchor.unsqueeze(1)
    elif anchor.ndim != 3:
        raise ValueError(f"Expected anchor shape (batch, dim) or (batch, n_anchor, dim), got {tuple(anchor.shape)}.")
    if anchor.shape[0] == 1 and target.shape[0] > 1:
        anchor = anchor.expand(target.shape[0], -1, -1)
    if anchor.shape[0] != target.shape[0]:
        raise ValueError(f"Anchor batch size {anchor.shape[0]} does not match target batch size {target.shape[0]}.")
    if anchor.shape[-1] != target.shape[-1]:
        raise ValueError(f"Anchor dimension {anchor.shape[-1]} does not match target dimension {target.shape[-1]}.")
    return anchor


def _as_column(value):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.ndim == 1:
        value = value.view(-1, 1)
    if value.ndim != 2 or value.shape[1] != 1:
        value = value.reshape(value.shape[0], -1).amax(dim=1, keepdim=True)
    return value


__all__ = ["BisectionConfig", "BisectionResult", "bisect_segment"]
