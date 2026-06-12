"""Shared local-stretch utilities for learned homeomorphisms."""

import torch


def _extract_forward_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _project_to_latent_domain(z_tensor, domain_shape="ball", boundary_eps=1e-6):
    shape = str(domain_shape).strip().lower()
    if shape in {"ball", "sphere", "l2_ball"}:
        norm = torch.norm(z_tensor, dim=-1, p=2, keepdim=True).clamp_min(1e-12)
        max_norm = 1.0 - float(boundary_eps)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return z_tensor * scale
    if shape in {"cube", "square", "linf_ball"}:
        limit = 1.0 - float(boundary_eps)
        return torch.clamp(z_tensor, min=-limit, max=limit)
    raise ValueError(f"Unsupported latent domain shape: {domain_shape}")


def local_log_stretch_loss(
    forward_fn,
    z_tensor,
    *,
    num_directions=8,
    fd_eps=1e-3,
    domain_shape="ball",
    boundary_eps=1e-6,
    log_eps=1e-8,
    aggregate="max",
):
    """Estimate end-to-end local stretch with a zeroth-order log-stabilized loss.

    For each sample ``z`` and random unit direction ``v``, this uses the
    symmetric finite-difference quotient

        ||T(z + eps v) - T(z - eps v)|| / ||(z + eps v) - (z - eps v)||

    as a directional local stretch estimate. The returned loss applies
    ``log(stretch + log_eps)`` for numerical stability, then aggregates across
    directions by ``max``, ``mean``, or ``spread`` (max-min).
    """

    if int(num_directions) <= 0:
        raise ValueError("num_directions must be positive.")
    batch_size, n_dim = z_tensor.shape
    dirs = torch.randn(batch_size, int(num_directions), n_dim, device=z_tensor.device, dtype=z_tensor.dtype)
    dirs = dirs / torch.norm(dirs, dim=-1, p=2, keepdim=True).clamp_min(1e-12)

    z_base = z_tensor.unsqueeze(1)
    z_plus = _project_to_latent_domain(z_base + float(fd_eps) * dirs, domain_shape=domain_shape, boundary_eps=boundary_eps)
    z_minus = _project_to_latent_domain(z_base - float(fd_eps) * dirs, domain_shape=domain_shape, boundary_eps=boundary_eps)

    flat_plus = z_plus.reshape(-1, n_dim)
    flat_minus = z_minus.reshape(-1, n_dim)
    out_plus = forward_fn(flat_plus)
    out_minus = forward_fn(flat_minus)
    out_plus = _extract_forward_tensor(out_plus).reshape(batch_size, int(num_directions), -1)
    out_minus = _extract_forward_tensor(out_minus).reshape(batch_size, int(num_directions), -1)

    output_diff = torch.norm(out_plus - out_minus, dim=-1, p=2)
    input_diff = torch.norm(z_plus - z_minus, dim=-1, p=2).clamp_min(float(log_eps))
    stretch = output_diff / input_diff
    log_stretch = torch.log(stretch.clamp_min(float(log_eps)))

    aggregate_mode = str(aggregate).strip().lower()
    if aggregate_mode == "max":
        per_sample = log_stretch.max(dim=1)[0]
    elif aggregate_mode == "mean":
        per_sample = log_stretch.mean(dim=1)
    elif aggregate_mode == "spread":
        per_sample = log_stretch.max(dim=1)[0] - log_stretch.min(dim=1)[0]
    else:
        raise ValueError(f"Unsupported local stretch aggregate mode: {aggregate}")

    loss = per_sample.mean()
    stats = {
        "loss": loss,
        "per_sample": per_sample,
        "stretch": stretch,
        "log_stretch": log_stretch,
    }
    return loss, stats


def _robust_log_stretch_spread(log_stretch, trim_k=2):
    k = max(1, min(int(trim_k), max(log_stretch.shape[1] // 2, 1)))
    top = torch.topk(log_stretch, k=k, dim=1, largest=True, sorted=False)[0].mean(dim=1)
    bottom = torch.topk(log_stretch, k=k, dim=1, largest=False, sorted=False)[0].mean(dim=1)
    return top - bottom


def _soft_log_stretch_spread(log_stretch, temperature=0.1):
    tau = max(float(temperature), 1e-6)
    soft_max = tau * torch.logsumexp(log_stretch / tau, dim=1)
    soft_min = -tau * torch.logsumexp(-log_stretch / tau, dim=1)
    return soft_max - soft_min


def combine_local_stretch_objective(
    loss_spread,
    stats,
    objective="spread",
    soft_temperature=0.1,
    spread_weight=0.25,
):
    """Build a training objective from local-stretch summary statistics."""

    mode = str(objective).strip().lower()
    if mode == "spread":
        return loss_spread
    log_stretch = stats["log_stretch"]
    centered = log_stretch - log_stretch.mean(dim=1, keepdim=True)
    variance_loss = centered.pow(2).mean()
    weight = float(spread_weight)
    if mode == "variance":
        return variance_loss
    if mode == "variance_plus_spread":
        return variance_loss + weight * loss_spread
    if mode == "variance_plus_robust_spread":
        robust_spread = _robust_log_stretch_spread(log_stretch, trim_k=2).mean()
        return variance_loss + weight * robust_spread
    if mode == "soft_spread":
        return _soft_log_stretch_spread(log_stretch, temperature=soft_temperature).mean()
    if mode == "variance_plus_soft_spread":
        soft_spread = _soft_log_stretch_spread(log_stretch, temperature=soft_temperature).mean()
        return variance_loss + weight * soft_spread
    raise ValueError(f"Unsupported local stretch objective: {objective}")


__all__ = [
    "combine_local_stretch_objective",
    "local_log_stretch_loss",
]
