"""Explicit gauge forward-state construction."""

from __future__ import annotations

import torch

from .constants import FAMILY_BOX_LOWER, FAMILY_BOX_UPPER, FAMILY_LINEAR, FAMILY_QUAD, FAMILY_SOC
from .constraints import explicit_candidate_cache
from .gradients import best_beta_gradient_u, smooth_polynomial_beta_gradient_u


def explicit_forward_state(
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
    del tie_tol
    batch_size, nvar = z.shape
    device, dtype = z.device, z.dtype
    batch_index = torch.arange(batch_size, device=device)
    raw_r = torch.norm(z, dim=1, p=2, keepdim=True)
    center_mask = raw_r <= eps
    r = raw_r.clamp_min(eps)
    u = z / r
    if center_mask.any():
        fallback_u = torch.zeros_like(u)
        fallback_u[:, 0] = 1
        u = torch.where(center_mask, fallback_u, u)

    best_beta = None
    best_family = None
    best_index = None
    smooth_enabled = smooth_tie_tol is not None

    reduce_mode = "full" if smooth_enabled else "winner"
    candidate_cache = explicit_candidate_cache(
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
                index_columns.append(
                    torch.arange(beta.shape[1], device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
                )

        local_beta = candidate_cache.get("alpha_lin")
        if local_beta is not None:
            _append_full_candidate(local_beta, FAMILY_LINEAR)
        local_beta = candidate_cache.get("alpha_box_lower")
        if local_beta is not None:
            _append_full_candidate(local_beta, FAMILY_BOX_LOWER)
            _append_full_candidate(candidate_cache["alpha_box_upper"], FAMILY_BOX_UPPER)
        local_beta = candidate_cache.get("alpha_soc")
        if local_beta is not None:
            _append_full_candidate(local_beta, FAMILY_SOC)
        local_beta = candidate_cache.get("alpha_quad")
        if local_beta is not None:
            _append_full_candidate(local_beta, FAMILY_QUAD)

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
            best_grad_u = smooth_polynomial_beta_gradient_u(
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
                grad_i = best_beta_gradient_u(
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
        _append_candidate(local_beta, FAMILY_LINEAR, candidate_cache["linear_idx"])

    local_beta = candidate_cache.get("box_lower_beta")
    if local_beta is not None:
        _append_candidate(local_beta, FAMILY_BOX_LOWER, candidate_cache["box_lower_idx"])
        _append_candidate(candidate_cache["box_upper_beta"], FAMILY_BOX_UPPER, candidate_cache["box_upper_idx"])

    local_beta = candidate_cache.get("soc_beta")
    if local_beta is not None:
        _append_candidate(local_beta, FAMILY_SOC, candidate_cache["soc_idx"])

    local_beta = candidate_cache.get("quad_beta")
    if local_beta is not None:
        _append_candidate(local_beta, FAMILY_QUAD, candidate_cache["quad_idx"])

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

    best_grad_u = best_beta_gradient_u(
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

