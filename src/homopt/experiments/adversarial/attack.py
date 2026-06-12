"""Adversarial attack core algorithms."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PGDAttack:
    def __init__(self, model: nn.Module, device: torch.device, random_start: bool = True, hom: bool = True):
        self.model = model
        self.device = device
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()
        self.hom = hom

    def _apply_weight_operator(self, delta: torch.Tensor, W):
        if W is None:
            return delta
        if torch.is_tensor(W):
            flat = delta.reshape(delta.shape[0], -1) @ W
            return flat.reshape_as(delta)
        if isinstance(W, dict) and W.get("type") == "spatial_smooth":
            kernel = W["kernel"].to(device=delta.device, dtype=delta.dtype)
            channels = int(delta.shape[1])
            k = kernel.shape[-1]
            pad = k // 2
            weight = kernel.view(1, 1, k, k).repeat(channels, 1, 1, 1)
            return F.conv2d(delta, weight, padding=pad, groups=channels)
        if isinstance(W, dict) and W.get("type") == "channel_spatial":
            channel_chol = W["channel_chol"].to(device=delta.device, dtype=delta.dtype)
            kernel = W["kernel"].to(device=delta.device, dtype=delta.dtype)
            channels = int(delta.shape[1])
            if channel_chol.shape != (channels, channels):
                raise ValueError(
                    f"channel_spatial operator channel mismatch: expected ({channels}, {channels}), got {tuple(channel_chol.shape)}."
                )
            mixed = torch.einsum("bcij,dc->bdij", delta, channel_chol)
            k = kernel.shape[-1]
            pad = k // 2
            weight = kernel.view(1, 1, k, k).repeat(channels, 1, 1, 1)
            return F.conv2d(mixed, weight, padding=pad, groups=channels)
        raise ValueError(f"Unsupported weight operator type: {type(W)}")

    def _weighted_lp_norm(self, delta: torch.Tensor, p, W: Optional[torch.Tensor] = None):
        delta_wp_flat = self._apply_weight_operator(delta, W).reshape(delta.shape[0], -1)
        return torch.norm(delta_wp_flat, dim=1, p=p, keepdim=True)

    def _box_radial_limit(self, delta: torch.Tensor, images: torch.Tensor):
        delta_flat = delta.reshape(delta.shape[0], -1)
        x_flat = images.reshape(images.shape[0], -1)
        upper = 1.0 - x_flat
        lower = -x_flat
        inf = torch.full_like(delta_flat, float("inf"))
        pos_ratio = torch.where(delta_flat > 0, upper / (delta_flat + 1e-12), inf)
        neg_ratio = torch.where(delta_flat < 0, lower / (delta_flat - 1e-12), inf)
        scale_max = torch.minimum(pos_ratio, neg_ratio).amin(dim=1, keepdim=True)
        return torch.clamp(scale_max, min=0.0)

    def _attack_radius(self, delta: torch.Tensor, images: torch.Tensor, eps: float, p, W: Optional[torch.Tensor] = None, clamp_max_one: bool = False):
        radius_ball = eps / (self._weighted_lp_norm(delta, p, W) + 1e-8)
        radius_box = self._box_radial_limit(delta, images)
        if clamp_max_one:
            radius_ball = torch.clamp(radius_ball, max=1.0)
            radius_box = torch.clamp(radius_box, max=1.0)
        return torch.minimum(radius_ball, radius_box)

    def _project_lp_ball(self, delta: torch.Tensor, eps: float, p, W: Optional[torch.Tensor] = None):
        delta_norm = self._weighted_lp_norm(delta, p, W)
        factor = torch.min(eps / (delta_norm + 1e-8), torch.ones_like(delta_norm))
        return delta * factor.view(-1, 1, 1, 1)

    def _shrink_to_image_box(self, delta: torch.Tensor, images: torch.Tensor):
        """Project to {-x <= delta <= 1-x} by radial shrink from the origin."""
        scale = torch.clamp(self._box_radial_limit(delta, images), max=1.0)
        return delta * scale.view(-1, 1, 1, 1)

    def _project_to_attack_set(self, delta: torch.Tensor, images: torch.Tensor, eps: float, p, W: Optional[torch.Tensor]):
        """Radial scaling projection onto Delta(x) = {||delta W||_p <= eps} cap {-x <= delta <= 1-x}."""
        scale = self._attack_radius(delta, images, eps, p, W, clamp_max_one=True)
        return delta * scale.view(-1, 1, 1, 1)

    def gauge_map(self, z: torch.Tensor, images: torch.Tensor, eps: float, p, W: Optional[torch.Tensor] = None):
        """Gauge map from unit L2 ball to Delta(x) using delta = ||z||_2 * rho(u) * u, u=z/||z||_2."""
        z_flat = z.reshape(z.shape[0], -1)
        z_norm = torch.norm(z_flat, dim=1, p=2, keepdim=True)
        u_flat = z_flat / (z_norm + 1e-12)
        rho = self._attack_radius(u_flat.reshape_as(z), images, eps, p, W, clamp_max_one=False)
        x_flat = z_norm * rho * u_flat
        return x_flat.reshape(z.shape)

    def _project_attack_state(self, delta: torch.Tensor, images: torch.Tensor, eps: float, p, W: Optional[torch.Tensor] = None):
        if self.hom:
            return self._project_lp_ball(delta, 1.0, 2, None)
        return self._project_to_attack_set(delta, images, eps, p, W)

    def _feasible_delta(self, delta: torch.Tensor, images: torch.Tensor, eps: float, p, W: Optional[torch.Tensor] = None):
        if self.hom:
            return self.gauge_map(delta, images, eps, p, W)
        return delta

    def _normalize_optimizer_name(self, optimizer_name: str) -> str:
        opt_name = str(optimizer_name).strip().lower()
        if opt_name not in {"gd", "adam"}:
            raise ValueError(f"Unsupported attack optimizer: {optimizer_name}. Use 'gd' or 'adam'.")
        return opt_name

    def _initialize_attack_backend(self, delta: torch.Tensor, *, optimizer_name: str, alpha: float):
        opt_name = self._normalize_optimizer_name(optimizer_name)
        if opt_name == "adam":
            delta = delta.detach().clone().requires_grad_(True)
            return delta, opt_name, optim.Adam([delta], lr=float(alpha))
        return delta.detach(), opt_name, None

    def _prepare_attack_iterate(self, delta: torch.Tensor, *, optimizer_name: str):
        if optimizer_name == "gd":
            return delta.detach().requires_grad_(True)
        return delta

    def _apply_attack_update(self, delta: torch.Tensor, *, optimizer_name: str, attack_opt, alpha: float):
        if optimizer_name == "adam":
            attack_opt.zero_grad()
            return delta
        return delta

    def _finish_attack_update(self, delta: torch.Tensor, *, optimizer_name: str, attack_opt, alpha: float):
        if optimizer_name == "adam":
            with torch.no_grad():
                delta.grad.mul_(-1.0)
            attack_opt.step()
            return delta
        grad = delta.grad.detach().sign()
        return delta + alpha * grad

    def _project_attack_iterate(self, delta: torch.Tensor, images: torch.Tensor, eps: float, p, W, *, optimizer_name: str):
        projected = self._project_attack_state(delta, images, eps, p, W)
        if optimizer_name == "adam":
            delta.copy_(projected)
            return delta
        return projected

    def generate(
        self,
        images,
        labels,
        eps=0.3,
        alpha=0.01,
        iters=40,
        norm=np.inf,
        W: Optional[torch.Tensor] = None,
        optimizer_name: str = "gd",
    ):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        delta = (
            torch.randn_like(images).to(self.device) * 0.01
            if self.random_start
            else torch.zeros_like(images).to(self.device)
        )
        # Enforce feasibility before the first forward pass.
        with torch.no_grad():
            delta = self._project_attack_state(delta, images, eps, norm, W)
        best_loss = float("inf")
        delta, opt_name, attack_opt = self._initialize_attack_backend(delta, optimizer_name=optimizer_name, alpha=alpha)

        for _ in range(iters):
            delta = self._prepare_attack_iterate(delta, optimizer_name=opt_name)
            feasible_delta = self._feasible_delta(delta, images, eps, norm, W)
            outputs = self.model(images + feasible_delta)
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            self._apply_attack_update(delta, optimizer_name=opt_name, attack_opt=attack_opt, alpha=alpha)
            loss.backward()
            delta = self._finish_attack_update(delta, optimizer_name=opt_name, attack_opt=attack_opt, alpha=alpha)
            alpha = alpha if loss.item() < best_loss else alpha * 0.99
            best_loss = min(best_loss, float(loss.item()))
            with torch.no_grad():
                delta = self._project_attack_iterate(delta, images, eps, norm, W, optimizer_name=opt_name)

        with torch.no_grad():
            final_delta = self._feasible_delta(delta, images, eps, norm, W)
        return (images + final_delta).detach()

    def evaluate_attack_success(
        self,
        model,
        test_loader,
        eps=0.3,
        alpha=0.01,
        iters=40,
        norm_type=np.inf,
        W=None,
        optimizer_name: str = "gd",
        return_details: bool = False,
    ):
        model.eval()
        correct_clean = 0
        correct_adv = 0
        total = 0
        linf_sum = 0.0
        l2_sum = 0.0
        lp_sum = 0.0
        linf_max = 0.0
        l2_max = 0.0
        lp_max = 0.0

        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs_clean = model(images)
                predicted_clean = torch.argmax(outputs_clean, dim=1)
                correct_clean += int((predicted_clean == labels).sum().item())
            adv_images = self.generate(
                images,
                labels,
                eps,
                alpha,
                iters,
                norm_type,
                W,
                optimizer_name=optimizer_name,
            )
            delta_flat = (adv_images - images).reshape(images.shape[0], -1)
            linf_vals = torch.norm(delta_flat, dim=1, p=np.inf)
            l2_vals = torch.norm(delta_flat, dim=1, p=2)
            lp_vals = torch.norm(
                self._apply_weight_operator((adv_images - images), W).reshape(images.shape[0], -1),
                dim=1,
                p=norm_type,
            )
            linf_sum += float(linf_vals.sum().item())
            l2_sum += float(l2_vals.sum().item())
            lp_sum += float(lp_vals.sum().item())
            linf_max = max(linf_max, float(linf_vals.max().item()))
            l2_max = max(l2_max, float(l2_vals.max().item()))
            lp_max = max(lp_max, float(lp_vals.max().item()))
            with torch.no_grad():
                outputs_adv = model(adv_images)
                predicted_adv = torch.argmax(outputs_adv, dim=1)
                correct_adv += int((predicted_adv == labels).sum().item())
            total += int(labels.size(0))

        clean_accuracy = 100.0 * correct_clean / max(total, 1)
        adv_accuracy = 100.0 * correct_adv / max(total, 1)
        success_rate = 100.0 * (correct_clean - correct_adv) / max(correct_clean, 1)
        if not return_details:
            return clean_accuracy, adv_accuracy, success_rate
        detail = {
            "delta_linf_mean": linf_sum / max(total, 1),
            "delta_linf_max": linf_max,
            "delta_l2_mean": l2_sum / max(total, 1),
            "delta_l2_max": l2_max,
            "delta_lp_mean": lp_sum / max(total, 1),
            "delta_lp_max": lp_max,
        }
        return clean_accuracy, adv_accuracy, success_rate, detail


__all__ = ["PGDAttack"]
