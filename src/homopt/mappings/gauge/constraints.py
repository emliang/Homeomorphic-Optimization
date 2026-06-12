"""Constraint-family candidate builders for gauge maps."""

from __future__ import annotations

import torch

from .roots import _max_real_quadratic_root


def explicit_candidate_cache(
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
