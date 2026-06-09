"""Gauge mapping implementations."""

from __future__ import annotations

import numpy as np
import torch

from homopt.utils import move_tensors_to_device


def soft_maximum(x, dim=None, temperature=1e-4, keepdim=False):
    return temperature * torch.logsumexp(x / temperature, dim=dim, keepdim=keepdim)


def _safe_denominator(denominator, eps=1e-12):
    sign = torch.sign(denominator)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return torch.where(denominator.abs() < eps, sign * eps, denominator)


def _safe_divide(numerator, denominator, eps=1e-12):
    return numerator / _safe_denominator(denominator, eps=eps)


def _pairwise_max(lhs, rhs):
    if hasattr(torch, "maximum"):
        return torch.maximum(lhs, rhs)
    return torch.max(torch.stack([lhs, rhs], dim=0), dim=0)[0]


_FAMILY_LINEAR = 0
_FAMILY_BOX_LOWER = 1
_FAMILY_BOX_UPPER = 2
_FAMILY_SOC = 3
_FAMILY_QUAD = 4


class _GaugeForwardExplicitFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z,
        x,
        u,
        r,
        beta,
        scaling,
        best_grad_u,
        eps,
    ):
        ctx.save_for_backward(z, u, r, beta, scaling, best_grad_u)
        ctx.eps = float(eps)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        z, u, r, beta, scaling, best_grad_u = ctx.saved_tensors
        grad_z = _explicit_vjp_from_state(
            z=z,
            u=u,
            r=r,
            beta=beta,
            scaling=scaling,
            best_grad_u=best_grad_u,
            grad_x=grad_output,
            eps=ctx.eps,
        )
        return (
            grad_z,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _explicit_vjp_from_state(*, z, u, r, beta, scaling, best_grad_u, grad_x, eps):
    grad_beta = (best_grad_u - (best_grad_u * u).sum(dim=1, keepdim=True) * u) / r.clamp_min(eps)
    beta_safe = beta.clamp_min(eps)
    grad_scaling = -grad_beta / (beta_safe * beta_safe)
    dot_term = (grad_x * z).sum(dim=1, keepdim=True)
    return scaling * grad_x + dot_term * grad_scaling


def _explicit_candidate_cache(
    u,
    *,
    eps,
    inv_slack,
    inv_lower_denom,
    inv_upper_denom,
    G,
    C,
    Gx0,
    Cx0,
    soc_Ccoef,
    Qsym,
    quad_gradB_const,
    quad_Ccoef,
    A=None,
    need_grad_aux=True,
    reduce="full",
):
    device, dtype = u.device, u.dtype
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)

    def _positive(candidate):
        return torch.where(candidate > eps, candidate, neg_inf)

    def _winner(candidate):
        beta, idx = torch.max(_positive(candidate), dim=1)
        return beta.view(-1, 1), idx

    def _winner_gather(values, idx):
        if values.dim() == 2:
            return values.gather(1, idx.view(-1, 1)).view(-1)
        view_shape = [idx.shape[0], 1] + [1] * (values.dim() - 2)
        expand_shape = [idx.shape[0], 1] + list(values.shape[2:])
        return values.gather(1, idx.view(*view_shape).expand(*expand_shape)).squeeze(1)

    cache = {}

    if A is not None:
        alpha_lin = torch.matmul(u, A.T) * inv_slack
        if reduce == "winner":
            cache["linear_beta"], cache["linear_idx"] = _winner(alpha_lin)
        else:
            cache["alpha_lin"] = _positive(alpha_lin)

    if inv_lower_denom is not None:
        alpha_box_lower = u * inv_lower_denom.view(1, -1)
        alpha_box_upper = u * inv_upper_denom.view(1, -1)
        if reduce == "winner":
            cache["box_lower_beta"], cache["box_lower_idx"] = _winner(alpha_box_lower)
            cache["box_upper_beta"], cache["box_upper_idx"] = _winner(alpha_box_upper)
        else:
            cache["alpha_box_lower"] = _positive(alpha_box_lower)
            cache["alpha_box_upper"] = _positive(alpha_box_upper)

    if G is not None:
        if u.shape[0] == 1:
            Gu = torch.matmul(G, u[0]).unsqueeze(0)
            Cu = torch.matmul(C, u[0]).view(1, -1)
        else:
            Gu = torch.einsum("bn,mkn->bmk", u, G)
            Cu = torch.matmul(u, C.T)
        g0 = Gx0.squeeze(0)
        c0 = Cx0.squeeze(0)
        A_soc = torch.sum(Gu * Gu, dim=-1) - Cu * Cu
        B_soc = 2 * (torch.sum(Gu * g0.unsqueeze(0), dim=-1) - Cu * c0.unsqueeze(0))
        alpha_soc = _max_real_quadratic_root(A_soc, B_soc, soc_Ccoef, eps=eps)
        if reduce == "winner":
            soc_beta, soc_idx = _winner(alpha_soc)
            cache["soc_beta"], cache["soc_idx"] = soc_beta, soc_idx
            if need_grad_aux:
                cache["soc_Gu"] = _winner_gather(Gu, soc_idx)
                cache["soc_Cu"] = _winner_gather(Cu, soc_idx)
                cache["soc_B"] = _winner_gather(B_soc, soc_idx)
        elif need_grad_aux:
            cache["Gu"] = Gu
            cache["Cu"] = Cu
            cache["B_soc"] = B_soc
            cache["alpha_soc"] = _positive(alpha_soc)
        else:
            cache["alpha_soc"] = _positive(alpha_soc)

    if Qsym is not None:
        Qu = torch.matmul(Qsym, u[0]).unsqueeze(0) if u.shape[0] == 1 else torch.einsum("bn,mnk->bmk", u, Qsym)
        B_quad = torch.matmul(u, quad_gradB_const.T)
        A_quad = 0.5 * torch.sum(Qu * u.unsqueeze(1), dim=-1)
        alpha_quad = _max_real_quadratic_root(A_quad, B_quad, quad_Ccoef, eps=eps)
        if reduce == "winner":
            quad_beta, quad_idx = _winner(alpha_quad)
            cache["quad_beta"], cache["quad_idx"] = quad_beta, quad_idx
            if need_grad_aux:
                cache["quad_Qu"] = _winner_gather(Qu, quad_idx)
                cache["quad_B"] = _winner_gather(B_quad, quad_idx)
        elif need_grad_aux:
            cache["Qu"] = Qu
            cache["B_quad"] = B_quad
            cache["alpha_quad"] = _positive(alpha_quad)
        else:
            cache["alpha_quad"] = _positive(alpha_quad)

    return cache


def _best_beta_gradient_u(
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
        return _implicit_best_beta_gradient_u(
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
        return _hybrid_best_beta_gradient_u(
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
        return _polynomial_best_beta_gradient_u(
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


def _explicit_forward_state(
    z,
    *,
    center,
    eps,
    tie_tol,
    smooth_tie_tol=None,
    smooth_temperature=1e-4,
    A,
    inv_slack,
    inv_lower_denom,
    inv_upper_denom,
    G,
    C,
    Gx0,
    Cx0,
    soc_gradB_const,
    soc_Ccoef,
    Qsym,
    pq,
    quad_gradB_const,
    quad_Ccoef,
    smooth_family_codes=None,
    smooth_candidate_index=None,
    gradient_rule="polynomial",
):
    batch_size, nvar = z.shape
    device, dtype = z.device, z.dtype
    batch_index = torch.arange(batch_size, device=device)
    r = torch.norm(z, dim=1, p=2, keepdim=True).clamp_min(eps)
    u = z / r

    best_beta = None
    best_family = None
    best_index = None
    smooth_enabled = smooth_tie_tol is not None

    reduce_mode = "full" if smooth_enabled else "winner"
    candidate_cache = _explicit_candidate_cache(
        u,
        eps=eps,
        A=A,
        inv_slack=inv_slack,
        inv_lower_denom=inv_lower_denom,
        inv_upper_denom=inv_upper_denom,
        G=G,
        C=C,
        Gx0=Gx0,
        Cx0=Cx0,
        soc_Ccoef=soc_Ccoef,
        Qsym=Qsym,
        quad_gradB_const=quad_gradB_const,
        quad_Ccoef=quad_Ccoef,
        need_grad_aux=(gradient_rule in {"implicit", "polynomial", "hybrid"}),
        reduce=reduce_mode,
    )

    beta_columns = []
    family_codes = []
    index_columns = []

    if smooth_enabled:
        def _append_full_candidate(beta, family_code):
            beta = beta.view(batch_size, -1)
            beta_columns.append(beta)
            if smooth_family_codes is None:
                family_codes.extend([family_code] * beta.shape[1])
            if smooth_candidate_index is None:
                index_columns.append(torch.arange(beta.shape[1], device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1))

        local_beta = candidate_cache.get("alpha_lin")
        if local_beta is not None:
            _append_full_candidate(local_beta, _FAMILY_LINEAR)
        local_beta = candidate_cache.get("alpha_box_lower")
        if local_beta is not None:
            _append_full_candidate(local_beta, _FAMILY_BOX_LOWER)
            _append_full_candidate(candidate_cache["alpha_box_upper"], _FAMILY_BOX_UPPER)
        local_beta = candidate_cache.get("alpha_soc")
        if local_beta is not None:
            _append_full_candidate(local_beta, _FAMILY_SOC)
        local_beta = candidate_cache.get("alpha_quad")
        if local_beta is not None:
            _append_full_candidate(local_beta, _FAMILY_QUAD)

        if not beta_columns:
            raise ValueError("No supported constraints available for explicit gauge backward.")

        candidate_beta = torch.cat(beta_columns, dim=1)
        valid_candidate = torch.isfinite(candidate_beta)
        if not valid_candidate.any(dim=1).all():
            raise ValueError("No valid positive boundary candidate found.")
        finite_candidate_beta = torch.where(valid_candidate, candidate_beta, torch.full_like(candidate_beta, float("-inf")))
        row_max, hard_pos = torch.max(finite_candidate_beta, dim=1, keepdim=True)
        active_candidate = valid_candidate & (candidate_beta >= row_max - float(smooth_tie_tol))
        shifted = torch.where(
            active_candidate,
            (candidate_beta - row_max) / float(smooth_temperature),
            torch.full_like(candidate_beta, float("-inf")),
        )
        weights = torch.softmax(shifted, dim=1)
        best_beta = row_max + float(smooth_temperature) * torch.logsumexp(shifted, dim=1, keepdim=True)
        if smooth_family_codes is not None and smooth_candidate_index is not None:
            candidate_family = smooth_family_codes.view(1, -1).expand(batch_size, -1)
            candidate_index = smooth_candidate_index.view(1, -1).expand(batch_size, -1)
        else:
            candidate_index = torch.cat(index_columns, dim=1)
            family_code_tensor = torch.tensor(family_codes, device=device, dtype=torch.long)
            candidate_family = family_code_tensor.view(1, -1).expand(batch_size, -1)
        best_family = candidate_family.gather(1, hard_pos).view(-1)
        best_index = candidate_index.gather(1, hard_pos).view(-1)
        if gradient_rule == "polynomial":
            best_grad_u = _smooth_polynomial_beta_gradient_u(
                u,
                candidate_beta=candidate_beta,
                candidate_family=candidate_family,
                candidate_index=candidate_index,
                active_candidate=active_candidate,
                weights=weights,
                candidate_cache=candidate_cache,
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
        else:
            best_grad_u = torch.zeros(batch_size, nvar, device=device, dtype=dtype)
            smooth_cache = dict(candidate_cache)
            for active_pos in range(candidate_beta.shape[1]):
                beta_i = torch.where(
                    active_candidate[:, active_pos : active_pos + 1],
                    candidate_beta[:, active_pos : active_pos + 1],
                    torch.ones_like(candidate_beta[:, active_pos : active_pos + 1]),
                )
                smooth_cache["soc_beta"] = beta_i
                smooth_cache["quad_beta"] = beta_i
                family_i = candidate_family[:, active_pos]
                index_i = candidate_index[:, active_pos]
                grad_i = _best_beta_gradient_u(
                    u,
                    gradient_rule=gradient_rule,
                    beta=beta_i,
                    best_family=family_i,
                    best_index=index_i,
                    candidate_cache=smooth_cache,
                    batch_index=batch_index,
                    center=center,
                    nvar=nvar,
                    eps=eps,
                    A=A,
                    inv_slack=inv_slack,
                    inv_lower_denom=inv_lower_denom,
                    inv_upper_denom=inv_upper_denom,
                    G=G,
                    C=C,
                    Gx0=Gx0,
                    soc_gradB_const=soc_gradB_const,
                    soc_Ccoef=soc_Ccoef,
                    Qsym=Qsym,
                    pq=pq,
                    quad_gradB_const=quad_gradB_const,
                    quad_Ccoef=quad_Ccoef,
                )
                best_grad_u = best_grad_u + weights[:, active_pos : active_pos + 1] * grad_i

        scaling = torch.ones_like(best_beta) / best_beta.clamp_min(eps)
        x = scaling * z + center
        return {
            "x": x,
            "z": z,
            "u": u,
            "r": r,
            "beta": best_beta,
            "scaling": scaling,
            "best_grad_u": best_grad_u,
            "best_family": best_family,
            "best_index": best_index,
        }

    def _append_candidate(beta, family_code, index):
        beta_columns.append(beta.view(batch_size, 1))
        family_codes.append(family_code)
        index_columns.append(index.view(batch_size, 1))

    local_beta = candidate_cache.get("linear_beta")
    if local_beta is not None:
        _append_candidate(local_beta, _FAMILY_LINEAR, candidate_cache["linear_idx"])

    local_beta = candidate_cache.get("box_lower_beta")
    if local_beta is not None:
        _append_candidate(local_beta, _FAMILY_BOX_LOWER, candidate_cache["box_lower_idx"])
        _append_candidate(candidate_cache["box_upper_beta"], _FAMILY_BOX_UPPER, candidate_cache["box_upper_idx"])

    local_beta = candidate_cache.get("soc_beta")
    if local_beta is not None:
        _append_candidate(local_beta, _FAMILY_SOC, candidate_cache["soc_idx"])

    local_beta = candidate_cache.get("quad_beta")
    if local_beta is not None:
        _append_candidate(local_beta, _FAMILY_QUAD, candidate_cache["quad_idx"])

    if beta_columns:
        candidate_beta = torch.cat(beta_columns, dim=1)
        candidate_index = torch.cat(index_columns, dim=1)
        best_beta_flat, best_pos = torch.max(candidate_beta, dim=1)
        gather_pos = best_pos.view(batch_size, 1)
        best_beta = best_beta_flat.view(batch_size, 1)
        family_code_tensor = torch.tensor(family_codes, device=device, dtype=torch.long)
        best_family = family_code_tensor[best_pos]
        best_index = candidate_index.gather(1, gather_pos).view(-1)

    if best_beta is None or best_family is None or best_index is None:
        raise ValueError("No supported constraints available for explicit gauge backward.")
    if (not torch.isfinite(best_beta).all()) or (best_beta <= eps).any():
        raise ValueError("Invalid explicit gauge scaling candidate encountered.")

    best_grad_u = _best_beta_gradient_u(
        u,
        gradient_rule=gradient_rule,
        beta=best_beta,
        best_family=best_family,
        best_index=best_index,
        candidate_cache=candidate_cache,
        batch_index=batch_index,
        center=center,
        nvar=nvar,
        eps=eps,
        A=A,
        inv_slack=inv_slack,
        inv_lower_denom=inv_lower_denom,
        inv_upper_denom=inv_upper_denom,
        G=G,
        C=C,
        Gx0=Gx0,
        soc_gradB_const=soc_gradB_const,
        soc_Ccoef=soc_Ccoef,
        Qsym=Qsym,
        pq=pq,
        quad_gradB_const=quad_gradB_const,
        quad_Ccoef=quad_Ccoef,
    )

    beta = best_beta
    scaling = torch.ones_like(beta) / beta.clamp_min(eps)
    x = scaling * z + center
    return {
        "x": x,
        "z": z,
        "u": u,
        "r": r,
        "beta": beta,
        "scaling": scaling,
        "best_grad_u": best_grad_u,
        "best_family": best_family,
        "best_index": best_index,
    }


def _implicit_best_beta_gradient_u(
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
        linear_mask = best_family == _FAMILY_LINEAR
        linear_idx = best_index[linear_mask]
        grad_x[linear_mask] = A[linear_idx]

    lower_mask = best_family == _FAMILY_BOX_LOWER
    lower_batch = batch_index[lower_mask]
    lower_idx = best_index[lower_mask]
    grad_x[lower_batch, lower_idx] = -1.0

    upper_mask = best_family == _FAMILY_BOX_UPPER
    upper_batch = batch_index[upper_mask]
    upper_idx = best_index[upper_mask]
    grad_x[upper_batch, upper_idx] = 1.0

    if G is not None:
        soc_mask = best_family == _FAMILY_SOC
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
        quad_mask = best_family == _FAMILY_QUAD
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


def _polynomial_best_beta_gradient_u(
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
        linear_mask = best_family == _FAMILY_LINEAR
        linear_idx = best_index[linear_mask]
        best_grad_u[linear_mask] = A[linear_idx] * inv_slack.view(-1)[linear_idx].unsqueeze(-1)

    if inv_lower_denom is not None:
        lower_mask = best_family == _FAMILY_BOX_LOWER
        lower_batch = batch_index[lower_mask]
        lower_idx = best_index[lower_mask]
        best_grad_u[lower_batch, lower_idx] = inv_lower_denom.view(-1)[lower_idx]

        upper_mask = best_family == _FAMILY_BOX_UPPER
        upper_batch = batch_index[upper_mask]
        upper_idx = best_index[upper_mask]
        best_grad_u[upper_batch, upper_idx] = inv_upper_denom.view(-1)[upper_idx]

    if G is not None:
        soc_mask = best_family == _FAMILY_SOC
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
        quad_mask = best_family == _FAMILY_QUAD
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


def _smooth_polynomial_beta_gradient_u(
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
        linear_mask = flat_family == _FAMILY_LINEAR
        linear_idx = flat_index[linear_mask]
        grad[linear_mask] = A[linear_idx] * inv_slack.view(-1)[linear_idx].unsqueeze(-1)

    if inv_lower_denom is not None:
        row_index = torch.arange(batch_size * candidate_count, device=device)
        lower_mask = flat_family == _FAMILY_BOX_LOWER
        lower_idx = flat_index[lower_mask]
        grad[row_index[lower_mask], lower_idx] = inv_lower_denom.view(-1)[lower_idx]

        upper_mask = flat_family == _FAMILY_BOX_UPPER
        upper_idx = flat_index[upper_mask]
        grad[row_index[upper_mask], upper_idx] = inv_upper_denom.view(-1)[upper_idx]

    if G is not None:
        soc_mask = flat_family == _FAMILY_SOC
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
        quad_mask = flat_family == _FAMILY_QUAD
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


def _hybrid_best_beta_gradient_u(
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
        linear_mask = best_family == _FAMILY_LINEAR
        linear_idx = best_index[linear_mask]
        best_grad_u[linear_mask] = A[linear_idx] * inv_slack.view(-1)[linear_idx].unsqueeze(-1)

    if inv_lower_denom is not None:
        lower_mask = best_family == _FAMILY_BOX_LOWER
        lower_batch = batch_index[lower_mask]
        lower_idx = best_index[lower_mask]
        best_grad_u[lower_batch, lower_idx] = inv_lower_denom.view(-1)[lower_idx]

        upper_mask = best_family == _FAMILY_BOX_UPPER
        upper_batch = batch_index[upper_mask]
        upper_idx = best_index[upper_mask]
        best_grad_u[upper_batch, upper_idx] = inv_upper_denom.view(-1)[upper_idx]

    if G is not None:
        soc_mask = best_family == _FAMILY_SOC
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
        quad_mask = best_family == _FAMILY_QUAD
        quad_batch = batch_index[quad_mask]
        quad_idx = best_index[quad_mask]
        rho_quad = rho[quad_mask]
        Qu_local = candidate_cache.get("quad_Qu")
        Qu_sel = Qu_local[quad_mask] if Qu_local is not None else candidate_cache["Qu"][quad_batch, quad_idx]
        grad_x = quad_gradB_const[quad_idx] + rho_quad * Qu_sel
        radial_dot = (grad_x * u[quad_mask]).sum(dim=1, keepdim=True)
        best_grad_u[quad_mask] = beta[quad_mask] * grad_x / _safe_denominator(radial_dot, eps=eps)

    return best_grad_u


def _max_real_quadratic_root(A, B, C, eps=1e-12):
    discriminant = B * B - 4 * A * C
    valid_quad = discriminant >= 0
    linear_like = A.abs() < eps
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0))
    root_linear = -C / _safe_denominator(B, eps=eps)
    root_1 = (2 * A) / _safe_denominator(-B + sqrt_discriminant, eps=eps)
    root_2 = (2 * A) / _safe_denominator(-B - sqrt_discriminant, eps=eps)
    root_nonlinear = _pairwise_max(root_1, root_2)
    roots = torch.where(linear_like, root_linear, root_nonlinear)
    return torch.where(valid_quad, roots, torch.zeros_like(discriminant))


class GaugeMap:
    def __init__(
        self,
        convex_set,
        p_norm=2,
        x_origin=None,
        smooth=False,
        explicit_gradient_rule="polynomial",
        smooth_tie_tol=1e-7,
        smooth_temperature=1e-4,
    ):
        self.constraint_set = convex_set
        self.set = convex_set  # compatibility alias for legacy callers
        self.p_norm = p_norm
        self.smooth = smooth
        if explicit_gradient_rule not in {"implicit", "polynomial", "hybrid"}:
            raise ValueError("explicit_gradient_rule must be 'implicit', 'polynomial', or 'hybrid'.")
        self.explicit_gradient_rule = explicit_gradient_rule
        self.smooth_tie_tol = float(smooth_tie_tol)
        if self.smooth_tie_tol < 0:
            raise ValueError("smooth_tie_tol must be nonnegative.")
        self.smooth_temperature = float(smooth_temperature)
        if self.smooth_temperature <= 0:
            raise ValueError("smooth_temperature must be positive.")
        self._eps = 1e-12
        self._static_cache = {}
        self.last_forward_mode = "uninitialized"
        self.forward_mode_counts = {"autograd": 0, "explicit": 0}
        if x_origin is None:
            self.center = torch.zeros(1, convex_set.nvar)
        else:
            self.center = torch.as_tensor(x_origin, dtype=torch.float32).view(1, -1)

    def to_device(self, device):
        self._static_cache.clear()
        return move_tensors_to_device(self, device)

    def reset_forward_stats(self):
        self.last_forward_mode = "uninitialized"
        self.forward_mode_counts = {"autograd": 0, "explicit": 0}

    def forward(self, z, method="autograd", tie_tol=1e-7, return_state=False):
        if method == "explicit":
            if not self._supports_explicit():
                raise NotImplementedError("GaugeMap explicit forward requires p_norm=2 and supported constraints.")
            terms = self._get_static_terms(z.device, z.dtype)
            state = _explicit_forward_state(
                z,
                center=terms["x0"],
                eps=float(self._eps),
                tie_tol=float(tie_tol),
                smooth_tie_tol=self.smooth_tie_tol if self.smooth else None,
                smooth_temperature=self.smooth_temperature,
                A=terms.get("A"),
                inv_slack=terms.get("inv_slack"),
                inv_lower_denom=terms.get("inv_lower_denom"),
                inv_upper_denom=terms.get("inv_upper_denom"),
                G=terms.get("G"),
                C=terms.get("C"),
                Gx0=terms.get("Gx0"),
                Cx0=terms.get("Cx0"),
                soc_gradB_const=terms.get("soc_gradB_const"),
                soc_Ccoef=terms.get("soc_Ccoef"),
                Qsym=terms.get("Qsym"),
                pq=terms.get("pq"),
                quad_gradB_const=terms.get("quad_gradB_const"),
                quad_Ccoef=terms.get("Cq0"),
                smooth_family_codes=terms.get("smooth_family_codes"),
                smooth_candidate_index=terms.get("smooth_candidate_index"),
                gradient_rule=self.explicit_gradient_rule,
            )
            self.last_forward_mode = "explicit"
            self.forward_mode_counts["explicit"] += 1
            if return_state:
                return state["x"], state
            return _GaugeForwardExplicitFunction.apply(
                z,
                state["x"],
                state["u"],
                state["r"],
                state["beta"],
                state["scaling"],
                state["best_grad_u"],
                float(self._eps),
            )
        if method != "autograd":
            raise ValueError(f"Unsupported GaugeMap forward method: {method}.")
        scaling = self._compute_scaling(z, forward=True)
        self.last_forward_mode = "autograd"
        self.forward_mode_counts["autograd"] += 1
        x = scaling * z + self.center
        if return_state:
            return x, None
        return x

    def inverse(self, x, method="autograd"):
        if method not in {"autograd", "explicit"}:
            raise ValueError(f"Unsupported GaugeMap inverse method: {method}.")
        x_shifted = x - self.center
        scaling = self._compute_scaling(x_shifted, forward=False)
        return scaling * x_shifted

    def vjp(self, z, grad_x, method="autograd", state=None):
        if method == "explicit":
            explicit_state = state
            if explicit_state is None:
                _, explicit_state = self.forward(z, method="explicit", return_state=True)
            return _explicit_vjp_from_state(
                z=z,
                u=explicit_state["u"],
                r=explicit_state["r"],
                beta=explicit_state["beta"],
                scaling=explicit_state["scaling"],
                best_grad_u=explicit_state["best_grad_u"],
                grad_x=grad_x,
                eps=self._eps,
            )
        if method != "autograd":
            raise ValueError(f"Unsupported GaugeMap vjp method: {method}.")
        with torch.enable_grad():
            z_detached = z.detach().requires_grad_(True)
            x = self.forward(z_detached, method=method)
            grad_z = torch.autograd.grad(x, z_detached, grad_outputs=grad_x, create_graph=False)[0]
        return grad_z

    def _direction_and_norm(self, x):
        x_norm = torch.norm(x, dim=1, p=2, keepdim=True)
        safe_x_norm = x_norm.clamp_min(self._eps)
        u_dir = x / safe_x_norm
        return x_norm, safe_x_norm, u_dir

    def _compute_scaling(self, x, forward=True):
        x_norm, safe_x_norm, u_dir = self._direction_and_norm(x)
        nonzero_mask = (x_norm > self._eps).view(-1)
        if not torch.any(nonzero_mask):
            return torch.ones_like(x_norm)
        p_norm = x_norm if self.p_norm == 2 else torch.norm(x, dim=1, p=self.p_norm, keepdim=True)
        safe_p_norm = p_norm.clamp_min(self._eps)
        boundary_distance = torch.ones_like(x_norm)
        boundary_distance[nonzero_mask] = self._closed_form_point_to_boundary_distance(
            u_dir[nonzero_mask]
        ).clamp_min(self._eps)
        if forward:
            scaling = p_norm / safe_x_norm / boundary_distance
        else:
            scaling = x_norm / safe_p_norm * boundary_distance
        return torch.where(x_norm > self._eps, scaling, torch.ones_like(scaling))

    def gauge(self, x, method="autograd"):
        if method not in {"autograd", "explicit"}:
            raise ValueError(f"Unsupported GaugeMap gauge method: {method}.")
        x_norm, _, u_dir = self._direction_and_norm(x)
        nonzero_mask = (x_norm > self._eps).view(-1)
        if not torch.any(nonzero_mask):
            return torch.zeros_like(x_norm)
        boundary_distance = torch.ones_like(x_norm)
        boundary_distance[nonzero_mask] = self._closed_form_point_to_boundary_distance(u_dir[nonzero_mask])
        scaling = x_norm * boundary_distance
        return torch.where(x_norm > self._eps, scaling, torch.zeros_like(scaling))

    def _get_static_terms(self, device, dtype):
        key = (device.type, device.index, dtype)
        cached = self._static_cache.get(key)
        if cached is not None:
            return cached

        terms = {}
        x0 = self.center.to(device=device, dtype=dtype)
        terms["x0"] = x0
        constraint_set = self.constraint_set

        if constraint_set.G is not None:
            G = constraint_set.G.to(device=device, dtype=dtype)
            h = constraint_set.h.to(device=device, dtype=dtype)
            C = constraint_set.C.to(device=device, dtype=dtype)
            d = constraint_set.d.to(device=device, dtype=dtype)
            Gx0 = torch.matmul(G, x0.T).permute(2, 0, 1) + h.unsqueeze(0)
            Cx0 = torch.matmul(x0, C.T) + d
            terms.update(
                {
                    "G": G,
                    "C": C,
                    "Gx0": Gx0,
                    "Cx0": Cx0,
                    "soc_gradB_const": 2
                    * (torch.einsum("mkn,mk->mn", G, Gx0.squeeze(0)) - Cx0.squeeze(0).unsqueeze(-1) * C),
                    "soc_Ccoef": torch.sum(Gx0.squeeze(0) * Gx0.squeeze(0), dim=-1, keepdim=True).T
                    - Cx0 * Cx0,
                }
            )

        if constraint_set.Qq is not None:
            Qq = constraint_set.Qq.to(device=device, dtype=dtype)
            pq = constraint_set.pq.to(device=device, dtype=dtype)
            bq = constraint_set.bq.to(device=device, dtype=dtype)
            Qsym = 0.5 * (Qq + torch.transpose(Qq, 1, 2))
            Qq_x0 = torch.matmul(Qq, x0.T).permute(2, 0, 1)
            Qsym_x0 = torch.matmul(Qsym, x0.T).permute(2, 0, 1)
            Cq0 = (
                0.5 * torch.sum(x0.unsqueeze(1) * Qq_x0, dim=-1)
                + torch.matmul(x0, pq.T)
                - bq.view(1, -1)
            )
            terms.update(
                {
                    "Qsym": Qsym,
                    "pq": pq,
                    "Cq0": Cq0,
                    "quad_gradB_const": (Qsym_x0.squeeze(0) + pq),
                }
            )

        if constraint_set.A is not None:
            A = constraint_set.A.to(device=device, dtype=dtype)
            b = constraint_set.b.to(device=device, dtype=dtype)
            Ax0 = torch.matmul(x0, A.T)
            slack = b - Ax0
            terms.update(
                {
                    "A": A,
                    "inv_slack": _safe_divide(torch.ones_like(slack), slack, eps=self._eps),
                }
            )

        if constraint_set.L is not None:
            L = constraint_set.L.to(device=device, dtype=dtype)
            U = constraint_set.U.to(device=device, dtype=dtype)
            lower_denom = L.view(1, -1) - x0
            upper_denom = U.view(1, -1) - x0
            terms.update(
                {
                    "inv_lower_denom": _safe_divide(torch.ones_like(lower_denom), lower_denom, eps=self._eps),
                    "inv_upper_denom": _safe_divide(torch.ones_like(upper_denom), upper_denom, eps=self._eps),
                }
            )

        family_parts = []
        index_parts = []
        if terms.get("A") is not None:
            count = int(terms["A"].shape[0])
            family_parts.append(torch.full((count,), _FAMILY_LINEAR, device=device, dtype=torch.long))
            index_parts.append(torch.arange(count, device=device, dtype=torch.long))
        if terms.get("inv_lower_denom") is not None:
            count = int(self.constraint_set.nvar)
            arange = torch.arange(count, device=device, dtype=torch.long)
            family_parts.append(torch.full((count,), _FAMILY_BOX_LOWER, device=device, dtype=torch.long))
            index_parts.append(arange)
            family_parts.append(torch.full((count,), _FAMILY_BOX_UPPER, device=device, dtype=torch.long))
            index_parts.append(arange)
        if terms.get("G") is not None:
            count = int(terms["G"].shape[0])
            family_parts.append(torch.full((count,), _FAMILY_SOC, device=device, dtype=torch.long))
            index_parts.append(torch.arange(count, device=device, dtype=torch.long))
        if terms.get("Qsym") is not None:
            count = int(terms["Qsym"].shape[0])
            family_parts.append(torch.full((count,), _FAMILY_QUAD, device=device, dtype=torch.long))
            index_parts.append(torch.arange(count, device=device, dtype=torch.long))
        if family_parts:
            terms["smooth_family_codes"] = torch.cat(family_parts)
            terms["smooth_candidate_index"] = torch.cat(index_parts)

        self._static_cache[key] = terms
        return terms

    def _supports_explicit(self):
        has_supported_constraints = any(
            (
                getattr(self.constraint_set, "A", None) is not None,
                getattr(self.constraint_set, "L", None) is not None,
                getattr(self.constraint_set, "G", None) is not None,
                getattr(self.constraint_set, "Qq", None) is not None,
            )
        )
        return (self.p_norm == 2) and has_supported_constraints

    def _bisection_point_to_boundary_distance(self, u):
        batch = u.shape[0]
        au = torch.ones(batch, 1, device=u.device) * 5
        al = torch.zeros(batch, 1, device=u.device)
        while (au - al).max() > 1e-6:
            am = (au + al) / 2
            points = self.center + am * u
            residual = self.constraint_set.constraint_x(points)
            feasible_mask = residual.max(1, keepdim=True)[0] <= 1e-5
            al[feasible_mask] = am[feasible_mask]
            au[~feasible_mask] = am[~feasible_mask]
        return al

    def _closed_form_point_to_boundary_distance(self, u):
        terms = self._get_static_terms(u.device, u.dtype)
        batch_size = u.shape[0]
        smooth_candidates = []
        max_alpha = None
        neg_inf = torch.tensor(float("-inf"), device=u.device, dtype=u.dtype)
        reduce_mode = "full" if self.smooth else "winner"
        candidate_cache = _explicit_candidate_cache(
            u,
            eps=self._eps,
            A=terms.get("A"),
            inv_slack=terms.get("inv_slack"),
            inv_lower_denom=terms.get("inv_lower_denom"),
            inv_upper_denom=terms.get("inv_upper_denom"),
            G=terms.get("G"),
            C=terms.get("C"),
            Gx0=terms.get("Gx0"),
            Cx0=terms.get("Cx0"),
            soc_Ccoef=terms.get("soc_Ccoef"),
            Qsym=terms.get("Qsym"),
            quad_gradB_const=terms.get("quad_gradB_const"),
            quad_Ccoef=terms.get("Cq0"),
            need_grad_aux=False,
            reduce=reduce_mode,
        )

        def _consume(candidate):
            nonlocal max_alpha
            candidate = candidate.reshape(batch_size, -1)
            candidate = torch.where(candidate > self._eps, candidate, neg_inf)
            if self.smooth:
                smooth_candidates.append(candidate)
                return
            candidate_max = torch.max(candidate, dim=-1, keepdim=True)[0]
            max_alpha = candidate_max if max_alpha is None else _pairwise_max(max_alpha, candidate_max)

        candidate_keys = (
            ("alpha_soc", "alpha_quad", "alpha_lin", "alpha_box_lower", "alpha_box_upper")
            if self.smooth
            else ("soc_beta", "quad_beta", "linear_beta", "box_lower_beta", "box_upper_beta")
        )
        for key in candidate_keys:
            candidate = candidate_cache.get(key)
            if candidate is not None:
                _consume(candidate)
        if self.smooth:
            if not smooth_candidates:
                raise ValueError("GaugeMap requires at least one supported constraint family.")
            dist_list = torch.cat(smooth_candidates, dim=-1)
            valid_candidate = torch.isfinite(dist_list)
            if not valid_candidate.any(dim=-1, keepdim=True).all():
                raise ValueError("No valid boundary candidate found for at least one sample.")
            finite_dist = torch.where(valid_candidate, dist_list, torch.full_like(dist_list, float("-inf")))
            hard_max = torch.max(finite_dist, dim=-1, keepdim=True)[0]
            active_candidate = valid_candidate & (dist_list >= hard_max - self.smooth_tie_tol)
            active_shifted = torch.where(
                active_candidate,
                (dist_list - hard_max) / self.smooth_temperature,
                torch.full_like(dist_list, float("-inf")),
            )
            max_alpha = hard_max + self.smooth_temperature * torch.logsumexp(active_shifted, dim=-1, keepdim=True)
            return max_alpha.view(-1, 1)
        if max_alpha is None:
            raise ValueError("GaugeMap requires at least one supported constraint family.")
        if (not torch.isfinite(max_alpha).all()) or (max_alpha <= self._eps).any():
            raise ValueError("No valid positive boundary candidate found.")
        return max_alpha


class GaugeMapMaxCut(GaugeMap):
    def __init__(self, convex_set, p_norm=np.inf, x_origin=None, smooth=False):
        super().__init__(convex_set, p_norm, x_origin, smooth)
        self.node = convex_set.node
        self.edge = convex_set.edge
        self.I = torch.eye(len(self.node))
        self.smooth = smooth

    def _supports_explicit(self):
        return False

    def _closed_form_point_to_boundary_distance(self, u):
        batch_size = u.shape[0]
        S = torch.zeros(batch_size, len(self.node), len(self.node), device=u.device)
        S[:, self.constraint_set.upper_triangle_index[:, 0], self.constraint_set.upper_triangle_index[:, 1]] = u
        S = S + S.transpose(1, 2)
        eigenvals = torch.linalg.eigvalsh(-S)
        if self.smooth:
            hard_max = torch.max(eigenvals, dim=-1, keepdim=True)[0]
            active = eigenvals >= hard_max - self.smooth_tie_tol
            shifted = torch.where(
                active,
                (eigenvals - hard_max) / self.smooth_temperature,
                torch.full_like(eigenvals, float("-inf")),
            )
            max_eigenval = hard_max + self.smooth_temperature * torch.logsumexp(shifted, dim=-1, keepdim=True)
        else:
            max_eigenval = torch.max(eigenvals, dim=-1)[0]
        return max_eigenval.view(batch_size, 1)

    def power_method(self, A, max_iter=50, tol=1e-3):
        batch_size, n = A.shape[0], A.shape[1]
        x = torch.randn(batch_size, n, 1, device=A.device)
        x = x / torch.norm(x, dim=1, keepdim=True)
        for _ in range(max_iter):
            x_new = torch.matmul(A, x)
            x_new = x_new / torch.norm(x_new, dim=1, keepdim=True)
            error = torch.abs(x_new - x).max(1)[0]
            if error.max() < tol:
                break
            x = x_new
        max_eigenval = torch.sum(x * torch.matmul(A, x), dim=1, keepdim=True)
        max_eigenval[error > tol] = 0
        return max_eigenval


def max_eigenvalue_with_gradient(A):
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    max_eigenvalue = eigenvalues[-1]
    max_eigenvector = eigenvectors[:, -1]

    class MaxEigenvalueFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_matrix, eigenvalue, eigenvector):
            ctx.save_for_backward(eigenvector)
            return eigenvalue

        @staticmethod
        def backward(ctx, grad_output):
            (eigenvector,) = ctx.saved_tensors
            grad_matrix = grad_output * torch.outer(eigenvector, eigenvector)
            return grad_matrix, None, None

    return MaxEigenvalueFunction.apply(A, max_eigenvalue, max_eigenvector)
