"""Attack configuration and weight-matrix helpers."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .common import _merged
from .data import get_dataset_config


def _dataset_feature_dim(dataset_name):
    cfg = get_dataset_config(dataset_name)
    return int(cfg["input_channels"]) * int(cfg["image_size"]) * int(cfg["image_size"])


def _norm_to_p(norm_value):
    key = str(norm_value).strip().lower()
    if key in {"inf", "linf", "l_inf", "infinity"}:
        return math.inf
    return float(norm_value)


def _estimate_weight_gain(attack_overrides):
    if not attack_overrides.get("weighted_norm", False):
        return 1.0
    mode = str(attack_overrides.get("weight_mode", "random_rank1")).strip().lower()
    if mode in {"channel_corr", "spatial_smooth", "channel_spatial", "identity"}:
        return 1.0
    if mode == "random_rank1":
        return max(1.0, 1.0 + abs(float(attack_overrides.get("weight_scale", 0.01))))
    if mode == "diag":
        diag = attack_overrides.get("weight_diag")
        if diag is None:
            return 1.0
        vals = [abs(float(v)) for v in diag]
        return max(sum(vals) / max(len(vals), 1), 1e-12)
    if mode == "matrix":
        matrix = attack_overrides.get("weight_matrix")
        if matrix is None or not matrix:
            return 1.0
        flat = [abs(float(v)) for row in matrix for v in row]
        return max(sum(flat) / max(len(flat), 1), 1e-12)
    return 1.0


def _auto_eps_value(dataset_name, attack_overrides, auto_eps):
    d = _dataset_feature_dim(dataset_name)
    p = _norm_to_p(attack_overrides.get("norm", "inf"))
    eps_ref = float(dict(auto_eps).get("reference_eps_linf", 8 / 255))
    eps = eps_ref if p == math.inf else eps_ref * (d ** (1.0 / p))
    gain = _estimate_weight_gain(attack_overrides)
    eps = eps / max(gain, 1e-12)

    min_eps = float(dict(auto_eps).get("min_eps", 1e-6))
    max_eps_cfg = dict(auto_eps).get("max_eps", None)
    if max_eps_cfg is None:
        max_eps = (1.0 if p == math.inf else d ** (1.0 / p)) / max(gain, 1e-12)
    else:
        max_eps = float(max_eps_cfg)
    return float(min(max(max(eps, min_eps), min_eps), max_eps))


def _apply_auto_eps(params, auto_eps):
    if not auto_eps or not dict(auto_eps).get("enabled", False):
        return params, None
    merged = _merged(params, {})
    attack = dict(merged.get("attack_overrides", {}))
    if (not dict(auto_eps).get("force_override", False)) and ("eps" in attack) and (attack["eps"] is not None):
        return merged, None
    eps = _auto_eps_value(merged.get("dataset_name", "cifar10"), attack, auto_eps)
    attack["eps"] = eps
    merged["attack_overrides"] = attack
    return merged, eps


def create_weight_matrix(dim=784, scale=0.01, device=None):
    w = torch.rand(dim, device=device).view(-1, 1)
    return torch.eye(dim, device=device) + scale * (w @ w.T)


def _normalize_attack_norm(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"inf", "linf", "l_inf", "infinity"}:
            return np.inf
        try:
            return float(key)
        except ValueError as exc:
            raise ValueError(f"Unsupported attack norm value: {value}") from exc
    if value is None:
        return np.inf
    if value == np.inf:
        return np.inf
    return float(value)


def _normalized_attack_config(dataset_config, attack_overrides=None):
    attack_cfg = dict(dataset_config.get("attack_defaults", {}))
    if attack_overrides:
        attack_cfg.update(dict(attack_overrides))
    attack_cfg["eps"] = float(attack_cfg["eps"])
    attack_cfg["alpha"] = float(attack_cfg["alpha"])
    attack_cfg["iters"] = int(attack_cfg["iters"])
    attack_cfg["norm"] = _normalize_attack_norm(attack_cfg.get("norm", np.inf))
    attack_cfg["optimizer"] = str(attack_cfg.get("optimizer", "gd")).strip().lower()
    if attack_cfg["optimizer"] not in {"gd", "adam"}:
        raise ValueError(f"Unsupported attack optimizer: {attack_cfg['optimizer']}. Use 'gd' or 'adam'.")
    # Weighted norm controls:
    # - weighted_norm=False -> no weighting (W=None)
    # - weighted_norm=True with:
    #   weight_mode in {"random_rank1", "identity", "diag", "matrix", "channel_corr", "spatial_smooth", "channel_spatial"}
    #   weight_scale / weight_diag / weight_matrix as needed.
    attack_cfg["weighted_norm"] = bool(attack_cfg.get("weighted_norm", False))
    attack_cfg["weight_mode"] = str(attack_cfg.get("weight_mode", "random_rank1")).strip().lower()
    attack_cfg["weight_scale"] = float(attack_cfg.get("weight_scale", 0.01))
    attack_cfg["weight_diag"] = attack_cfg.get("weight_diag", None)
    attack_cfg["weight_matrix"] = attack_cfg.get("weight_matrix", None)
    attack_cfg["channel_corr"] = attack_cfg.get("channel_corr", None)
    attack_cfg["corr_jitter"] = float(attack_cfg.get("corr_jitter", 1e-4))
    attack_cfg["spatial_sigma"] = float(attack_cfg.get("spatial_sigma", 1.0))
    attack_cfg["spatial_kernel_size"] = int(attack_cfg.get("spatial_kernel_size", 0))
    return attack_cfg


def _build_attack_weight_matrix(
    feature_dim: int,
    attack_cfg: dict,
    device: torch.device,
    input_channels: Optional[int] = None,
    image_size: Optional[int] = None,
):
    if not attack_cfg.get("weighted_norm", False):
        return None

    mode = attack_cfg.get("weight_mode", "random_rank1")
    if mode == "identity":
        return torch.eye(feature_dim, device=device)
    if mode == "random_rank1":
        return create_weight_matrix(
            dim=feature_dim,
            scale=float(attack_cfg.get("weight_scale", 0.01)),
            device=device,
        )
    if mode == "channel_corr":
        channels = int(input_channels or 1)
        if channels <= 1 or feature_dim % channels != 0:
            raise ValueError(
                f"weight_mode='channel_corr' requires valid input_channels; got channels={channels}, feature_dim={feature_dim}."
            )
        hw = feature_dim // channels
        corr = attack_cfg.get("channel_corr", None)
        if corr is None:
            # Default: mild positive RGB correlations.
            corr = [
                [1.0, 0.35, 0.20],
                [0.35, 1.0, 0.30],
                [0.20, 0.30, 1.0],
            ]
        corr_tensor = torch.as_tensor(corr, dtype=torch.float32, device=device)
        if corr_tensor.shape != (channels, channels):
            raise ValueError(
                f"channel_corr shape mismatch: expected ({channels}, {channels}), got {tuple(corr_tensor.shape)}."
            )
        corr_tensor = 0.5 * (corr_tensor + corr_tensor.t())
        jitter = float(attack_cfg.get("corr_jitter", 1e-4))
        corr_tensor = corr_tensor + jitter * torch.eye(channels, dtype=torch.float32, device=device)
        try:
            chol = torch.linalg.cholesky(corr_tensor)
        except Exception as exc:  # pragma: no cover - backend-dependent error types
            raise ValueError("channel_corr must be symmetric positive definite (SPD).") from exc
        chol = chol.contiguous()
        eye_hw = torch.eye(hw, dtype=torch.float32, device=device).contiguous()
        try:
            return torch.kron(chol, eye_hw)
        except RuntimeError:
            # Compatibility fallback for older torch builds with stricter view/stride assumptions.
            return (chol[:, None, :, None] * eye_hw[None, :, None, :]).reshape(
                channels * hw, channels * hw
            )
    if mode == "channel_spatial":
        channels = int(input_channels or 1)
        side = int(image_size or 0)
        if channels <= 1 or feature_dim != channels * side * side:
            raise ValueError(
                f"weight_mode='channel_spatial' requires image dims/channels; got channels={channels}, image_size={image_size}, feature_dim={feature_dim}."
            )
        corr = attack_cfg.get("channel_corr", None)
        if corr is None:
            corr = [
                [1.0, 0.35, 0.20],
                [0.35, 1.0, 0.30],
                [0.20, 0.30, 1.0],
            ]
        corr_tensor = torch.as_tensor(corr, dtype=torch.float32, device=device)
        if corr_tensor.shape != (channels, channels):
            raise ValueError(
                f"channel_corr shape mismatch: expected ({channels}, {channels}), got {tuple(corr_tensor.shape)}."
            )
        corr_tensor = 0.5 * (corr_tensor + corr_tensor.t())
        jitter = float(attack_cfg.get("corr_jitter", 1e-4))
        corr_tensor = corr_tensor + jitter * torch.eye(channels, dtype=torch.float32, device=device)
        try:
            channel_chol = torch.linalg.cholesky(corr_tensor).contiguous()
        except Exception as exc:  # pragma: no cover
            raise ValueError("channel_corr must be symmetric positive definite (SPD).") from exc

        sigma = max(float(attack_cfg.get("spatial_sigma", 1.0)), 1e-4)
        k_cfg = int(attack_cfg.get("spatial_kernel_size", 0))
        if k_cfg > 0:
            k = int(k_cfg)
        else:
            k = int(2 * np.ceil(3.0 * sigma) + 1)
        if k % 2 == 0:
            k += 1
        radius = (k - 1) // 2
        grid = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        g = torch.exp(-(grid ** 2) / (2.0 * sigma * sigma))
        g = g / (g.sum() + 1e-12)
        kernel_2d = torch.outer(g, g)
        kernel_2d = kernel_2d / (kernel_2d.sum() + 1e-12)
        return {
            "type": "channel_spatial",
            "channels": channels,
            "image_size": side,
            "channel_chol": channel_chol,
            "kernel": kernel_2d.contiguous(),
            "sigma": sigma,
            "kernel_size": k,
        }
    if mode == "spatial_smooth":
        channels = int(input_channels or 1)
        side = int(image_size or 0)
        if side <= 0 or feature_dim != channels * side * side:
            raise ValueError(
                f"weight_mode='spatial_smooth' requires square image dims; got channels={channels}, image_size={image_size}, feature_dim={feature_dim}."
            )
        sigma = max(float(attack_cfg.get("spatial_sigma", 1.0)), 1e-4)
        k_cfg = int(attack_cfg.get("spatial_kernel_size", 0))
        if k_cfg > 0:
            k = int(k_cfg)
        else:
            k = int(2 * np.ceil(3.0 * sigma) + 1)
        if k % 2 == 0:
            k += 1
        radius = (k - 1) // 2
        grid = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        g = torch.exp(-(grid ** 2) / (2.0 * sigma * sigma))
        g = g / (g.sum() + 1e-12)
        kernel_2d = torch.outer(g, g)
        kernel_2d = kernel_2d / (kernel_2d.sum() + 1e-12)
        return {
            "type": "spatial_smooth",
            "channels": channels,
            "image_size": side,
            "kernel": kernel_2d.contiguous(),
            "sigma": sigma,
            "kernel_size": k,
        }
    if mode == "diag":
        weight_diag = attack_cfg.get("weight_diag", None)
        if weight_diag is None:
            scale = float(attack_cfg.get("weight_scale", 0.0))
            diag_tensor = torch.ones(feature_dim, dtype=torch.float32, device=device) * (1.0 + scale)
            return torch.diag(diag_tensor)
        diag_tensor = torch.as_tensor(weight_diag, dtype=torch.float32, device=device).view(-1)
        if diag_tensor.numel() != feature_dim:
            raise ValueError(
                f"weight_diag size mismatch: expected {feature_dim}, got {diag_tensor.numel()}."
            )
        return torch.diag(diag_tensor)
    if mode == "matrix":
        weight_matrix = attack_cfg.get("weight_matrix", None)
        if weight_matrix is None:
            raise ValueError("weighted_norm with weight_mode='matrix' requires 'weight_matrix'.")
        matrix_tensor = torch.as_tensor(weight_matrix, dtype=torch.float32, device=device)
        if matrix_tensor.shape != (feature_dim, feature_dim):
            raise ValueError(
                f"weight_matrix shape mismatch: expected ({feature_dim}, {feature_dim}), got {tuple(matrix_tensor.shape)}."
            )
        return matrix_tensor
    raise ValueError(f"Unsupported weight_mode: {mode}")


__all__ = [
    "_apply_auto_eps",
    "_auto_eps_value",
    "_build_attack_weight_matrix",
    "_dataset_feature_dim",
    "_normalize_attack_norm",
    "_normalized_attack_config",
    "create_weight_matrix",
]
