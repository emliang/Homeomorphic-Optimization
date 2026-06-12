"""Explicit gauge VJP and radial candidate gradient rules."""

from __future__ import annotations

import torch

from .constants import FAMILY_BOX_LOWER, FAMILY_BOX_UPPER, FAMILY_LINEAR, FAMILY_QUAD, FAMILY_SOC
from .math import _safe_denominator


def explicit_vjp_from_state(*, z, u, r, beta, scaling, best_grad_u, grad_x, eps):
    grad_beta = (best_grad_u - (best_grad_u * u).sum(dim=1, keepdim=True) * u) / r.clamp_min(eps)
    beta_safe = beta.clamp_min(eps)
    grad_scaling = -grad_beta / (beta_safe * beta_safe)
    dot_term = (grad_x * z).sum(dim=1, keepdim=True)
    grad_z = scaling * grad_x + dot_term * grad_scaling
    center_mask = torch.norm(z, dim=1, p=2, keepdim=True) <= eps
    if center_mask.any():
        grad_z = torch.where(center_mask, grad_x, grad_z)
    return grad_z


def best_beta_gradient_u(
    u,
    *,
    gradient_rule,
    beta,
    best_family,
    best_index,
    candidate_cache,
    batch_index,
    center,
    nvar,
    eps,
    A,
    inv_slack,
    inv_lower_denom,
    inv_upper_denom,
    G,
    C,
    Gx0,
    soc_gradB_const,
    soc_Ccoef,
    Qsym,
    pq,
    quad_gradB_const,
    quad_Ccoef,
):
    if gradient_rule == "implicit":
        return implicit_best_beta_gradient_u(
            u,
            beta=beta,
            best_family=best_family,
            best_index=best_index,
            candidate_cache=candidate_cache,
            batch_index=batch_index,
            center=center,
            eps=eps,
            A=A,
            G=G,
            C=C,
            Gx0=Gx0,
            Qsym=Qsym,
            pq=pq,
            quad_gradB_const=quad_gradB_const,
        )
    if gradient_rule == "hybrid":
        return hybrid_best_beta_gradient_u(
            u,
            beta=beta,
            best_family=best_family,
            best_index=best_index,
            candidate_cache=candidate_cache,
            batch_index=batch_index,
            nvar=nvar,
            eps=eps,
            A=A,
            inv_slack=inv_slack,
            inv_lower_denom=inv_lower_denom,
            inv_upper_denom=inv_upper_denom,
            G=G,
            C=C,
            Gx0=Gx0,
            quad_gradB_const=quad_gradB_const,
        )
    if gradient_rule == "polynomial":
        return polynomial_best_beta_gradient_u(
            u,
            beta=beta,
            best_family=best_family,
            best_index=best_index,
            candidate_cache=candidate_cache,
            batch_index=batch_index,
            nvar=nvar,
            eps=eps,
            A=A,
            inv_slack=inv_slack,
            inv_lower_denom=inv_lower_denom,
            inv_upper_denom=inv_upper_denom,
            G=G,
            C=C,
            soc_gradB_const=soc_gradB_const,
            soc_Ccoef=soc_Ccoef,
            quad_gradB_const=quad_gradB_const,
            quad_Ccoef=quad_Ccoef,
        )
    raise ValueError(f"Unsupported explicit gauge gradient_rule: {gradient_rule}")


def implicit_best_beta_gradient_u(
    u,
    *,
    beta,
    best_family,
    best_index,
    candidate_cache,
    batch_index,
    center,
    eps,
    A,
    G,
    C,
    Gx0,
    Qsym,
    pq,
    quad_gradB_const,
):
    batch_size, nvar = u.shape
    device, dtype = u.device, u.dtype
    grad_x = torch.zeros(batch_size, nvar, device=device, dtype=dtype)
    rho = torch.ones_like(beta) / beta.clamp_min(eps)

    if A is not None:
        linear_mask = best_family == FAMILY_LINEAR
        linear_idx = best_index[linear_mask]
        grad_x[linear_mask] = A[linear_idx]

    lower_mask = best_family == FAMILY_BOX_LOWER
    lower_batch = batch_index[lower_mask]
    lower_idx = best_index[lower_mask]
    grad_x[lower_batch, lower_idx] = -1.0

    upper_mask = best_family == FAMILY_BOX_UPPER
    upper_batch = batch_index[upper_mask]
    upper_idx = best_index[upper_mask]
    grad_x[upper_batch, upper_idx] = 1.0

    if G is not None:
        soc_mask = best_family == FAMILY_SOC
        soc_batch = batch_index[soc_mask]
        soc_idx = best_index[soc_mask]
        rho_soc = rho[soc_mask]
        Gu_local = candidate_cache.get("soc_Gu")
        if Gu_local is not None:
            Gu_sel = Gu_local[soc_mask]
        else:
            Gu_cache = candidate_cache.get("Gu")
            if Gu_cache is not None:
                Gu_sel = Gu_cache[soc_batch, soc_idx]
            else:
                Gu_sel = torch.einsum("bn,bkn->bk", u[soc_mask], G[soc_idx])
        if Gu_sel.numel() == 0:
            Gu_sel = torch.einsum("bn,bkn->bk", u[soc_mask], G[soc_idx])
        gx_boundary = Gx0.squeeze(0)[soc_idx] + rho_soc * Gu_sel
        gx_norm = torch.norm(gx_boundary, dim=-1, keepdim=True).clamp_min(eps)
        grad_x[soc_mask] = torch.einsum("bk,bkn->bn", gx_boundary / gx_norm, G[soc_idx]) - C[soc_idx]

    if Qsym is not None:
        quad_mask = best_family == FAMILY_QUAD
        quad_batch = batch_index[quad_mask]
        quad_idx = best_index[quad_mask]
        rho_quad = rho[quad_mask]
        Qu_local = candidate_cache.get("quad_Qu")
        if Qu_local is not None:
            grad_x[quad_mask] = quad_gradB_const[quad_idx] + rho_quad * Qu_local[quad_mask]
        else:
            Qu_cache = candidate_cache.get("Qu")
            if Qu_cache is not None:
                grad_x[quad_mask] = quad_gradB_const[quad_idx] + rho_quad * Qu_cache[quad_batch, quad_idx]
            else:
                x_boundary = center + rho_quad * u[quad_mask]
                grad_x[quad_mask] = torch.einsum("bij,bj->bi", Qsym[quad_idx], x_boundary) + pq[quad_idx]
        if quad_idx.numel() == 0:
            x_boundary = center + rho_quad * u[quad_mask]
            grad_x[quad_mask] = torch.einsum("bij,bj->bi", Qsym[quad_idx], x_boundary) + pq[quad_idx]

    radial_dot = (grad_x * u).sum(dim=1, keepdim=True)
    return beta * grad_x / _safe_denominator(radial_dot, eps=eps)


def polynomial_best_beta_gradient_u(
    u,
    *,
    beta,
    best_family,
    best_index,
    candidate_cache,
    batch_index,
    nvar,
    eps,
    A,
    inv_slack,
    inv_lower_denom,
    inv_upper_denom,
    G,
    C,
    soc_gradB_const,
    soc_Ccoef,
    quad_gradB_const,
    quad_Ccoef,
):
    del beta
    batch_size = best_family.shape[0]
    device = best_family.device
    dtype = u.dtype
    best_grad_u = torch.zeros(batch_size, nvar, device=device, dtype=dtype)

    if A is not None:
        linear_mask = best_family == FAMILY_LINEAR
        linear_idx = best_index[linear_mask]
        best_grad_u[linear_mask] = A[linear_idx] * inv_slack.view(-1)[linear_idx].unsqueeze(-1)

    if inv_lower_denom is not None:
        lower_mask = best_family == FAMILY_BOX_LOWER
        lower_batch = batch_index[lower_mask]
        lower_idx = best_index[lower_mask]
        best_grad_u[lower_batch, lower_idx] = inv_lower_denom.view(-1)[lower_idx]

        upper_mask = best_family == FAMILY_BOX_UPPER
        upper_batch = batch_index[upper_mask]
        upper_idx = best_index[upper_mask]
        best_grad_u[upper_batch, upper_idx] = inv_upper_denom.view(-1)[upper_idx]

    if G is not None:
        soc_mask = best_family == FAMILY_SOC
        soc_batch = batch_index[soc_mask]
        soc_idx = best_index[soc_mask]
        alpha_soc_sel = candidate_cache["soc_beta"][soc_mask]
        B_soc_local = candidate_cache.get("soc_B")
        B_soc_sel = (
            B_soc_local[soc_mask].unsqueeze(-1)
            if B_soc_local is not None
            else candidate_cache["B_soc"][soc_batch, soc_idx].unsqueeze(-1)
        )
        C_soc_sel = soc_Ccoef.view(-1)[soc_idx].unsqueeze(-1)
        Gu_local = candidate_cache.get("soc_Gu")
        Cu_local = candidate_cache.get("soc_Cu")
        Gu_sel = Gu_local[soc_mask] if Gu_local is not None else candidate_cache["Gu"][soc_batch, soc_idx]
        Cu_sel = (
            Cu_local[soc_mask].unsqueeze(-1)
            if Cu_local is not None
            else candidate_cache["Cu"][soc_batch, soc_idx].unsqueeze(-1)
        )
        G_sel = G[soc_idx]
        C_sel = C[soc_idx]
        gradB_soc_sel = soc_gradB_const[soc_idx]
        gradA_soc_sel = 2 * (torch.einsum("bk,bkn->bn", Gu_sel, G_sel) - Cu_sel * C_sel)
        denom_soc_sel = 2 * C_soc_sel * alpha_soc_sel + B_soc_sel
        best_grad_u[soc_mask] = -(gradA_soc_sel + alpha_soc_sel * gradB_soc_sel) / _safe_denominator(
            denom_soc_sel,
            eps=eps,
        )

    if quad_gradB_const is not None:
        quad_mask = best_family == FAMILY_QUAD
        quad_batch = batch_index[quad_mask]
        quad_idx = best_index[quad_mask]
        alpha_quad_sel = candidate_cache["quad_beta"][quad_mask]
        B_quad_local = candidate_cache.get("quad_B")
        B_quad_sel = (
            B_quad_local[quad_mask].unsqueeze(-1)
            if B_quad_local is not None
            else candidate_cache["B_quad"][quad_batch, quad_idx].unsqueeze(-1)
        )
        C_quad_sel = quad_Ccoef.view(-1)[quad_idx].unsqueeze(-1)
        Qu_local = candidate_cache.get("quad_Qu")
        gradA_quad_sel = Qu_local[quad_mask] if Qu_local is not None else candidate_cache["Qu"][quad_batch, quad_idx]
        gradB_quad_sel = quad_gradB_const[quad_idx]
        denom_quad_sel = 2 * C_quad_sel * alpha_quad_sel + B_quad_sel
        best_grad_u[quad_mask] = -(gradA_quad_sel + alpha_quad_sel * gradB_quad_sel) / _safe_denominator(
            denom_quad_sel,
            eps=eps,
        )

    return best_grad_u


def smooth_polynomial_beta_gradient_u(
    u,
    *,
    candidate_beta,
    candidate_family,
    candidate_index,
    active_candidate,
    weights,
    candidate_cache,
    nvar,
    eps,
    A,
    inv_slack,
    inv_lower_denom,
    inv_upper_denom,
    G,
    C,
    soc_gradB_const,
    soc_Ccoef,
    quad_gradB_const,
    quad_Ccoef,
):
    batch_size, candidate_count = candidate_family.shape
    device = candidate_family.device
    dtype = u.dtype
    flat_family = candidate_family.reshape(-1)
    flat_index = candidate_index.reshape(-1)
    flat_active = active_candidate.reshape(-1)
    flat_weight = weights.reshape(-1, 1)
    flat_beta = torch.where(
        flat_active.view(-1, 1),
        candidate_beta.reshape(-1, 1),
        torch.ones(batch_size * candidate_count, 1, device=device, dtype=dtype),
    )
    flat_batch = None

    def _flat_batch(mask):
        nonlocal flat_batch
        if flat_batch is None:
            flat_batch = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, candidate_count).reshape(-1)
        return flat_batch[mask]

    grad = torch.zeros(batch_size * candidate_count, nvar, device=device, dtype=dtype)

    if A is not None:
        linear_mask = flat_family == FAMILY_LINEAR
        linear_idx = flat_index[linear_mask]
        grad[linear_mask] = A[linear_idx] * inv_slack.view(-1)[linear_idx].unsqueeze(-1)

    if inv_lower_denom is not None:
        row_index = torch.arange(batch_size * candidate_count, device=device)
        lower_mask = flat_family == FAMILY_BOX_LOWER
        lower_idx = flat_index[lower_mask]
        grad[row_index[lower_mask], lower_idx] = inv_lower_denom.view(-1)[lower_idx]

        upper_mask = flat_family == FAMILY_BOX_UPPER
        upper_idx = flat_index[upper_mask]
        grad[row_index[upper_mask], upper_idx] = inv_upper_denom.view(-1)[upper_idx]

    if G is not None:
        soc_mask = flat_family == FAMILY_SOC
        soc_idx = flat_index[soc_mask]
        alpha_soc_sel = flat_beta[soc_mask]
        if batch_size == 1:
            B_soc_sel = candidate_cache["B_soc"][0, soc_idx].unsqueeze(-1)
            Gu_sel = candidate_cache["Gu"][0, soc_idx]
            Cu_sel = candidate_cache["Cu"][0, soc_idx].unsqueeze(-1)
        else:
            soc_batch = _flat_batch(soc_mask)
            B_soc_sel = candidate_cache["B_soc"][soc_batch, soc_idx].unsqueeze(-1)
            Gu_sel = candidate_cache["Gu"][soc_batch, soc_idx]
            Cu_sel = candidate_cache["Cu"][soc_batch, soc_idx].unsqueeze(-1)
        C_soc_sel = soc_Ccoef.view(-1)[soc_idx].unsqueeze(-1)
        G_sel = G[soc_idx]
        C_sel = C[soc_idx]
        gradB_soc_sel = soc_gradB_const[soc_idx]
        gradA_soc_sel = 2 * (torch.einsum("bk,bkn->bn", Gu_sel, G_sel) - Cu_sel * C_sel)
        denom_soc_sel = 2 * C_soc_sel * alpha_soc_sel + B_soc_sel
        grad[soc_mask] = -(gradA_soc_sel + alpha_soc_sel * gradB_soc_sel) / _safe_denominator(
            denom_soc_sel,
            eps=eps,
        )

    if quad_gradB_const is not None:
        quad_mask = flat_family == FAMILY_QUAD
        quad_idx = flat_index[quad_mask]
        alpha_quad_sel = flat_beta[quad_mask]
        if batch_size == 1:
            B_quad_sel = candidate_cache["B_quad"][0, quad_idx].unsqueeze(-1)
            gradA_quad_sel = candidate_cache["Qu"][0, quad_idx]
        else:
            quad_batch = _flat_batch(quad_mask)
            B_quad_sel = candidate_cache["B_quad"][quad_batch, quad_idx].unsqueeze(-1)
            gradA_quad_sel = candidate_cache["Qu"][quad_batch, quad_idx]
        C_quad_sel = quad_Ccoef.view(-1)[quad_idx].unsqueeze(-1)
        gradB_quad_sel = quad_gradB_const[quad_idx]
        denom_quad_sel = 2 * C_quad_sel * alpha_quad_sel + B_quad_sel
        grad[quad_mask] = -(gradA_quad_sel + alpha_quad_sel * gradB_quad_sel) / _safe_denominator(
            denom_quad_sel,
            eps=eps,
        )

    weighted = flat_weight * torch.where(flat_active.view(-1, 1), grad, torch.zeros_like(grad))
    return weighted.view(batch_size, candidate_count, nvar).sum(dim=1)


def hybrid_best_beta_gradient_u(
    u,
    *,
    beta,
    best_family,
    best_index,
    candidate_cache,
    batch_index,
    nvar,
    eps,
    A,
    inv_slack,
    inv_lower_denom,
    inv_upper_denom,
    G,
    C,
    Gx0,
    quad_gradB_const,
):
    batch_size = best_family.shape[0]
    device = best_family.device
    dtype = u.dtype
    best_grad_u = torch.zeros(batch_size, nvar, device=device, dtype=dtype)
    rho = torch.ones_like(beta) / beta.clamp_min(eps)

    if A is not None:
        linear_mask = best_family == FAMILY_LINEAR
        linear_idx = best_index[linear_mask]
        best_grad_u[linear_mask] = A[linear_idx] * inv_slack.view(-1)[linear_idx].unsqueeze(-1)

    if inv_lower_denom is not None:
        lower_mask = best_family == FAMILY_BOX_LOWER
        lower_batch = batch_index[lower_mask]
        lower_idx = best_index[lower_mask]
        best_grad_u[lower_batch, lower_idx] = inv_lower_denom.view(-1)[lower_idx]

        upper_mask = best_family == FAMILY_BOX_UPPER
        upper_batch = batch_index[upper_mask]
        upper_idx = best_index[upper_mask]
        best_grad_u[upper_batch, upper_idx] = inv_upper_denom.view(-1)[upper_idx]

    if G is not None:
        soc_mask = best_family == FAMILY_SOC
        soc_batch = batch_index[soc_mask]
        soc_idx = best_index[soc_mask]
        rho_soc = rho[soc_mask]
        Gu_local = candidate_cache.get("soc_Gu")
        Gu_sel = Gu_local[soc_mask] if Gu_local is not None else candidate_cache["Gu"][soc_batch, soc_idx]
        gx_boundary = Gx0.squeeze(0)[soc_idx] + rho_soc * Gu_sel
        gx_norm = torch.norm(gx_boundary, dim=-1, keepdim=True).clamp_min(eps)
        grad_x = torch.einsum("bk,bkn->bn", gx_boundary / gx_norm, G[soc_idx]) - C[soc_idx]
        radial_dot = (grad_x * u[soc_mask]).sum(dim=1, keepdim=True)
        best_grad_u[soc_mask] = beta[soc_mask] * grad_x / _safe_denominator(radial_dot, eps=eps)

    if quad_gradB_const is not None:
        quad_mask = best_family == FAMILY_QUAD
        quad_batch = batch_index[quad_mask]
        quad_idx = best_index[quad_mask]
        rho_quad = rho[quad_mask]
        Qu_local = candidate_cache.get("quad_Qu")
        Qu_sel = Qu_local[quad_mask] if Qu_local is not None else candidate_cache["Qu"][quad_batch, quad_idx]
        grad_x = quad_gradB_const[quad_idx] + rho_quad * Qu_sel
        radial_dot = (grad_x * u[quad_mask]).sum(dim=1, keepdim=True)
        best_grad_u[quad_mask] = beta[quad_mask] * grad_x / _safe_denominator(radial_dot, eps=eps)

    return best_grad_u

