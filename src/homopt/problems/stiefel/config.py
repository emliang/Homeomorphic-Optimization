"""Configuration helpers for Stiefel-manifold problem families."""

from __future__ import annotations

import torch


def _square(value):
    return value * value


def _as_float_tensor(value, *, name):
    if value is None:
        raise ValueError(f"Missing required StiefelProblem config field: {name}")
    return torch.as_tensor(value, dtype=torch.float32)


def _optional_tensor(value):
    return torch.as_tensor(value, dtype=torch.float32) if value is not None else None


def _expand_bound(value, nvar, *, name):
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.repeat(nvar)
    tensor = tensor.reshape(-1)
    if tensor.numel() != nvar:
        raise ValueError(f"{name} must be scalar or have length {nvar}; got {tensor.numel()}.")
    return tensor


def _selection_matrix(nvar, indices):
    selector = torch.zeros(nvar, nvar, dtype=torch.float32)
    if indices:
        idx = torch.as_tensor(indices, dtype=torch.long)
        selector[idx, idx] = 1.0
    return selector


def _flatten_group_indices(group, n_rows, n_cols):
    if "flat_indices" in group:
        flat_indices = [int(index) for index in group["flat_indices"]]
    elif "rows" in group:
        flat_indices = [
            int(row) * n_cols + col
            for row in group["rows"]
            for col in range(n_cols)
        ]
    else:
        raise ValueError("Each group norm constraint needs either 'rows' or 'flat_indices'.")
    nvar = n_rows * n_cols
    if not flat_indices:
        raise ValueError("Group norm constraints cannot be empty.")
    if min(flat_indices) < 0 or max(flat_indices) >= nvar:
        raise ValueError(f"Group norm flat indices must be in [0, {nvar}).")
    return sorted(set(flat_indices))


def _normalize_group_norm_constraints(config, n_rows, n_cols):
    raw_constraints = config.get("group_norm_constraints", config.get("feature_group_norms", []))
    normalized = []
    for index, group in enumerate(raw_constraints):
        budget = float(group["budget"])
        if budget <= 0:
            raise ValueError("Group norm constraint budget must be positive.")
        flat_indices = _flatten_group_indices(group, n_rows, n_cols)
        normalized.append(
            {
                "name": str(group.get("name", f"group_{index}")),
                "rows": None if "rows" not in group else [int(row) for row in group["rows"]],
                "flat_indices": flat_indices,
                "budget": budget,
            }
        )
    return normalized


def _build_soc_constraint_tensors(nvar, *, norm_radius=None, group_norm_constraints=()):
    matrices = []
    radii = []
    if norm_radius is not None:
        norm_radius = float(norm_radius)
        if norm_radius <= 0:
            raise ValueError("norm_radius must be positive.")
        matrices.append(torch.eye(nvar, dtype=torch.float32))
        radii.append(norm_radius)
    for group in group_norm_constraints:
        matrices.append(_selection_matrix(nvar, group["flat_indices"]))
        radii.append(float(group["budget"]))
    if not matrices:
        return None, None, None, None
    G = torch.stack(matrices, dim=0)
    h = torch.zeros(len(matrices), nvar, dtype=torch.float32)
    C = torch.zeros(len(matrices), nvar, dtype=torch.float32)
    d = torch.tensor(radii, dtype=torch.float32)
    return G, h, C, d


def _normalize_pca_covariance(covariance_tensor, *, n_cols, objective_normalization=None, objective_scale=None):
    mode = "none" if objective_normalization is None else str(objective_normalization).lower()
    aliases = {
        "": "none",
        "none": "none",
        "false": "none",
        "no": "none",
        "off": "none",
        "top-k-sum": "top_sum",
        "top_k_sum": "top_sum",
        "top_sum": "top_sum",
        "trace": "trace",
        "spectral": "spectral",
        "top": "spectral",
        "fro": "frobenius",
        "frobenius": "frobenius",
    }
    if mode not in aliases:
        raise ValueError(
            "objective_normalization must be one of: None, 'top_sum', "
            "'trace', 'spectral', or 'frobenius'."
        )
    mode = aliases[mode]

    if objective_scale is not None:
        scale = float(objective_scale)
        if scale <= 0:
            raise ValueError("objective_scale must be positive.")
        return covariance_tensor / scale, scale, "custom"

    if mode == "none":
        return covariance_tensor, 1.0, mode
    if mode == "top_sum":
        eigvals = torch.linalg.eigvalsh(covariance_tensor)
        top_count = min(max(int(n_cols), 1), int(eigvals.numel()))
        scale = float(torch.topk(eigvals, top_count).values.sum().detach().cpu().item())
    elif mode == "trace":
        scale = float(torch.trace(covariance_tensor).detach().cpu().item())
    elif mode == "spectral":
        scale = float(torch.linalg.eigvalsh(covariance_tensor).abs().max().detach().cpu().item())
    else:
        scale = float(torch.linalg.norm(covariance_tensor, ord="fro").detach().cpu().item())

    if scale <= 1e-12:
        raise ValueError(f"Cannot normalize Stiefel PCA objective with nonpositive scale {scale}.")
    return covariance_tensor / scale, scale, mode


def create_constrained_pca_stiefel_config(
    *,
    data=None,
    covariance=None,
    n_cols=1,
    restricted_rows=None,
    group_budget=None,
    group_norm_constraints=None,
    L=-1.5,
    U=1.5,
    norm_radius=None,
    objective_normalization=None,
    objective_scale=None,
):
    """Build a Stiefel config for constrained PCA/subspace learning."""

    if covariance is None:
        if data is None:
            raise ValueError("Either covariance or data must be provided.")
        data_tensor = torch.as_tensor(data, dtype=torch.float32)
        if data_tensor.ndim != 2:
            raise ValueError("data must have shape (n_samples, n_features).")
        centered = data_tensor - data_tensor.mean(dim=0, keepdim=True)
        covariance_tensor = centered.T @ centered / max(int(data_tensor.shape[0]) - 1, 1)
    else:
        covariance_tensor = torch.as_tensor(covariance, dtype=torch.float32)
    if covariance_tensor.ndim != 2 or covariance_tensor.shape[0] != covariance_tensor.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    covariance_tensor, normalization_scale, normalization_mode = _normalize_pca_covariance(
        covariance_tensor,
        n_cols=n_cols,
        objective_normalization=objective_normalization,
        objective_scale=objective_scale,
    )

    constraints = list(group_norm_constraints or [])
    if restricted_rows is not None or group_budget is not None:
        if restricted_rows is None or group_budget is None:
            raise ValueError("restricted_rows and group_budget must be provided together.")
        constraints.append(
            {
                "name": "restricted_rows",
                "rows": [int(row) for row in restricted_rows],
                "budget": float(group_budget),
            }
        )

    config = {
        "A_obj": covariance_tensor,
        "n_cols": int(n_cols),
        "L": L,
        "U": U,
        "group_norm_constraints": constraints,
        "objective_normalization": normalization_mode,
        "objective_scale": normalization_scale,
    }
    if norm_radius is not None:
        config["norm_radius"] = float(norm_radius)
    return config


__all__ = ["create_constrained_pca_stiefel_config"]
