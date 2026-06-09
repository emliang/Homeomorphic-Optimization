"""Package-native adversarial attack workflow for Hom-PGD experiments."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from homopt.utils import ensure_dir, resolve_torch_device, set_global_seed


def _require_torchvision():
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "Adversarial workflow requires torchvision. Install with: pip install -e .[research]"
        ) from exc
    return datasets, transforms


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib. Install with: pip install -e .[viz]"
        ) from exc
    return plt


def _dataset_registry():
    datasets, transforms = _require_torchvision()
    return {
        "mnist": {
            "dataset_cls": datasets.MNIST,
            "input_channels": 1,
            "image_size": 28,
            "num_classes": 10,
            "batch_size": 256,
            "class_names": [str(i) for i in range(10)],
            "train_transform": transforms.Compose([transforms.ToTensor()]),
            "test_transform": transforms.Compose([transforms.ToTensor()]),
            "attack_defaults": {"eps": 0.1, "alpha": 0.1, "iters": 1000, "norm": 4},
            "training_defaults": {"epochs": 10, "learning_rate": 1e-3, "batch_size": 256},
            "model_defaults": {
                "base_channels": 32,
                "num_stages": 3,
                "blocks_per_stage": 2,
                "hidden_dim": 128,
                "feature_dropout": 0.0,
                "classifier_dropout": 0.25,
            },
        },
        "cifar10": {
            "dataset_cls": datasets.CIFAR10,
            "input_channels": 3,
            "image_size": 32,
            "num_classes": 10,
            "batch_size": 256,
            "class_names": [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
            "train_transform": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            "test_transform": transforms.Compose([transforms.ToTensor()]),
            "attack_defaults": {"eps": 8 / 255, "alpha": 2 / 255, "iters": 50, "norm": np.inf},
            "training_defaults": {"epochs": 30, "learning_rate": 3e-4, "batch_size": 128},
            "model_defaults": {
                "base_channels": 64,
                "num_stages": 4,
                "blocks_per_stage": 3,
                "hidden_dim": 512,
                "feature_dropout": 0.1,
                "classifier_dropout": 0.5,
            },
        },
    }


def get_dataset_config(dataset_name: str):
    key = dataset_name.lower()
    registry = _dataset_registry()
    if key not in registry:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Available options: {sorted(registry)}")
    return registry[key]


def get_model_checkpoint(dataset_name: str, checkpoint_dir="data"):
    checkpoint_root = ensure_dir(Path(checkpoint_dir))
    return checkpoint_root / f"{dataset_name.lower()}_model.pth"


def _merged(base, overrides=None):
    result = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merged(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _norm_label(value):
    return str(value).lower().replace(".", "_")


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


def _rebase_artifacts(artifacts, *, root_output_dir, case_output_dir):
    if root_output_dir is None or case_output_dir is None:
        return dict(artifacts or {})
    prefix = Path(case_output_dir).relative_to(Path(root_output_dir))
    rebased = {}
    for key, value in dict(artifacts or {}).items():
        if value is None:
            rebased[key] = None
        else:
            rebased[key] = str(prefix / value)
    return rebased


def _print_attack_config(attack_cfg, auto_eps_applied):
    print(
        "[adv-run] effective attack config:",
        {
            "eps": attack_cfg.get("eps"),
            "alpha": attack_cfg.get("alpha"),
            "iters": attack_cfg.get("iters"),
            "norm": attack_cfg.get("norm"),
            "optimizer": attack_cfg.get("optimizer"),
            "weighted_norm": attack_cfg.get("weighted_norm"),
            "auto_eps_applied": auto_eps_applied,
        },
    )


class ResidualBlock(nn.Module):
    """Two-layer residual block with optional channel projection."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return self.act(out)


class AdaptiveCNN(nn.Module):
    """Dataset-aware CNN used by the package-native adversarial workflow."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        image_size: int,
        base_channels: int = None,
        num_stages: int = None,
        blocks_per_stage: int = None,
        hidden_dim: int = None,
        feature_dropout: float = None,
        classifier_dropout: float = None,
    ):
        super().__init__()
        is_small = image_size <= 28
        base_channels = int(base_channels if base_channels is not None else (32 if is_small else 64))
        num_stages = int(num_stages if num_stages is not None else (3 if is_small else 4))
        blocks_per_stage = int(blocks_per_stage if blocks_per_stage is not None else (2 if is_small else 3))

        layers = []
        in_channels = input_channels
        for stage_idx in range(num_stages):
            out_channels = base_channels * (2 ** stage_idx)
            if feature_dropout is not None:
                dropout = float(feature_dropout)
            else:
                dropout = 0.0 if is_small else 0.05 * (stage_idx + 1)
            for block_idx in range(blocks_per_stage):
                block_in_channels = in_channels if block_idx == 0 else out_channels
                layers.append(ResidualBlock(block_in_channels, out_channels, dropout=dropout))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2))

        self.features = nn.Sequential(*layers)
        self.flatten_dim = self._infer_feature_dim(input_channels, image_size)
        hidden_dim = int(hidden_dim if hidden_dim is not None else (128 if is_small else 512))
        dropout = float(classifier_dropout if classifier_dropout is not None else (0.25 if is_small else 0.5))
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _infer_feature_dim(self, input_channels: int, image_size: int):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, image_size, image_size)
            return int(self.features(dummy).view(1, -1).size(1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(dataset_config):
    model_cfg = dict(dataset_config.get("model_defaults", {}))
    return AdaptiveCNN(
        input_channels=dataset_config["input_channels"],
        num_classes=dataset_config["num_classes"],
        image_size=dataset_config["image_size"],
        **model_cfg,
    )


class ModelTrainer:
    def __init__(self, model: nn.Module, device: torch.device, save_path: Path, learning_rate=1e-3):
        self.model = model
        self.device = device
        self.save_path = Path(save_path)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    def train(self, train_loader: DataLoader, epochs: int, verbose: bool = False, log_every: int = 1):
        self.model.train()
        log_every = max(int(log_every), 1)
        for epoch in range(epochs):
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                batch_size = int(labels.size(0))
                running_loss += float(loss.item()) * batch_size
                running_total += batch_size
                running_correct += int((torch.argmax(outputs, dim=1) == labels).sum().item())
            if verbose and ((epoch + 1) % log_every == 0):
                mean_loss = running_loss / max(running_total, 1)
                train_acc = 100.0 * running_correct / max(running_total, 1)
                print(
                    f"[train] epoch {epoch + 1}/{epochs} "
                    f"loss={mean_loss:.4f} acc={train_acc:.2f}%"
                )
        ensure_dir(self.save_path.parent)
        torch.save(self.model.state_dict(), self.save_path)

    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += int(labels.size(0))
                correct += int((predicted == labels).sum().item())
        return 100.0 * correct / max(total, 1)


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


def find_lowest_confidence_accurate_samples(model, test_dataset, num_samples=100):
    device = next(model.parameters()).device
    model.eval()
    accurate_samples = []
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, true_label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            if predicted_label == true_label:
                accurate_samples.append((idx, float(torch.max(probabilities).item()), true_label))
    accurate_samples.sort(key=lambda item: item[1])
    return [idx for idx, _, _ in accurate_samples[:num_samples]]


def find_accurately_classified_samples(model, test_dataset):
    device = next(model.parameters()).device
    model.eval()
    accurate_indices = []
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, true_label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            predicted_label = torch.argmax(output, dim=1).item()
            if predicted_label == true_label:
                accurate_indices.append(int(idx))
    return accurate_indices


def _filter_correctly_classified_indices(
    model: nn.Module,
    dataset,
    indices,
    device: torch.device,
    batch_size: int = 256,
):
    selected = [int(i) for i in list(indices)]
    selected = [i for i in selected if 0 <= i < len(dataset)]
    if len(selected) == 0:
        return []
    subset = Subset(dataset, selected)
    loader = DataLoader(subset, batch_size=max(int(batch_size), 1), shuffle=False)
    model.eval()
    correct = []
    cursor = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predicted = torch.argmax(model(inputs), dim=1)
            mask = predicted.eq(labels).detach().cpu().tolist()
            batch_indices = selected[cursor: cursor + len(mask)]
            correct.extend(idx for idx, keep in zip(batch_indices, mask) if bool(keep))
            cursor += len(mask)
    return correct


def load_data(dataset_name: str, data_root="data", download=True):
    config = get_dataset_config(dataset_name)
    dataset_cls = config["dataset_cls"]
    train_dataset = dataset_cls(root=str(data_root), train=True, download=download, transform=config["train_transform"])
    test_dataset = dataset_cls(root=str(data_root), train=False, download=download, transform=config["test_transform"])
    return train_dataset, test_dataset, config


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


def _select_device(device=None):
    return resolve_torch_device(device=device, default="auto")


def _save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _save_checkpoint_artifact(checkpoint_path: Path, output_dir, dataset_name: str):
    if output_dir is None or not Path(checkpoint_path).exists():
        return {}
    artifact_root = ensure_dir(Path(output_dir) / "artifacts")
    artifact_path = artifact_root / f"{dataset_name}_model.pth"
    artifact_path.write_bytes(Path(checkpoint_path).read_bytes())
    return {"checkpoint": str(artifact_path.relative_to(Path(output_dir)))}


def _load_model_for_dataset(dataset_name, dataset_config, checkpoint_path, device):
    model = build_model(dataset_config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def _render_attack_figure(
    images,
    labels,
    clean_pred,
    adv_images,
    adv_pred,
    title_prefix: str,
    rows: int = 3,
    class_names=None,
):
    plt = _require_matplotlib()
    # Paper-friendly typography (larger labels for readability after scaling in manuscripts).
    title_fs = 15
    label_fs = 14
    row_fs = 13
    cbar_label_fs = 12
    cbar_tick_fs = 11
    rows = max(int(rows), 1)
    sample_count = int(images.shape[0])
    cmap = "gray" if images.shape[1] == 1 else None
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(9.2, 2.8 * rows),
        gridspec_kw={"wspace": 0.03, "hspace": 0.12},
    )
    if rows == 1:
        axes = axes.reshape(1, -1)
    col_titles = ["Original", "Attack", "Perturbation"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=title_fs, fontweight="bold")
    def _name(v):
        idx = int(v)
        if class_names is None:
            return str(idx)
        if 0 <= idx < len(class_names):
            return str(class_names[idx])
        return str(idx)

    for idx in range(rows):
        if idx >= sample_count:
            for col in range(3):
                axes[idx, col].axis("off")
            continue
        image = images[idx].detach().cpu()
        adv_image = adv_images[idx].detach().cpu()
        image = image[0] if images.shape[1] == 1 else image.permute(1, 2, 0)
        adv_image = adv_image[0] if images.shape[1] == 1 else adv_image.permute(1, 2, 0)
        perturbation = (adv_images[idx].detach().cpu() - images[idx].detach().cpu()).abs()
        perturbation = perturbation[0] if images.shape[1] == 1 else perturbation.mean(dim=0)
        axes[idx, 0].imshow(image, cmap=cmap)
        axes[idx, 1].imshow(adv_image, cmap=cmap)
        pert_img = axes[idx, 2].imshow(perturbation, cmap="hot")
        true_name = _name(labels[idx])
        clean_name = _name(clean_pred[idx])
        adv_name = _name(adv_pred[idx])
        axes[idx, 0].set_xlabel(f"true: {true_name}", fontsize=label_fs)
        axes[idx, 1].set_xlabel(f"pred: {adv_name}", fontsize=label_fs)
        axes[idx, 2].set_xlabel("|delta|", fontsize=label_fs)
        axes[idx, 0].set_ylabel(f"#{idx + 1}", rotation=0, labelpad=10, fontsize=row_fs)
        cbar = fig.colorbar(pert_img, ax=axes[idx, 2], fraction=0.046, pad=0.01)
        cbar.ax.tick_params(labelsize=cbar_tick_fs, length=2)
        cbar.set_label("intensity", fontsize=cbar_label_fs)
        for col in range(3):
            axes[idx, col].set_xticks([])
            axes[idx, col].set_yticks([])
            for spine in axes[idx, col].spines.values():
                spine.set_visible(False)
    fig.subplots_adjust(top=0.98, bottom=0.06)
    return fig, plt


def train_adversarial_model(
    dataset_name="mnist",
    data_root="data",
    checkpoint_dir="data",
    training_epochs=10,
    training_lr=None,
    batch_size=None,
    seed=42,
    device=None,
    download=True,
    output_dir=None,
    verbose=False,
    log_every=1,
):
    set_global_seed(seed)
    device = _select_device(device)
    train_dataset, test_dataset, dataset_config = load_data(dataset_name, data_root=data_root, download=download)
    train_defaults = dict(dataset_config.get("training_defaults", {}))
    batch_size = int(batch_size or train_defaults.get("batch_size", dataset_config.get("batch_size", 256)))
    epochs = int(training_epochs or train_defaults.get("epochs", 10))
    learning_rate = float(training_lr or train_defaults.get("learning_rate", 1e-3))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    checkpoint_path = get_model_checkpoint(dataset_name, checkpoint_dir=checkpoint_dir)
    model = build_model(dataset_config).to(device)
    trainer = ModelTrainer(model, device, checkpoint_path, learning_rate=learning_rate)

    start = perf_counter()
    trainer.train(train_loader, epochs=epochs, verbose=bool(verbose), log_every=log_every)
    test_accuracy = trainer.evaluate(test_loader)
    elapsed = perf_counter() - start

    summary = {
        "dataset": dataset_name,
        "test_accuracy": test_accuracy,
        "training_epochs": epochs,
        "training_lr": learning_rate,
        "batch_size": batch_size,
        "checkpoint": str(checkpoint_path),
        "elapsed_sec": elapsed,
        "verbose": bool(verbose),
        "log_every": int(max(int(log_every), 1)),
    }
    artifact_root = ensure_dir(Path(output_dir) / "artifacts") if output_dir is not None else None
    artifacts = {}
    if artifact_root is not None:
        summary_path = artifact_root / f"{dataset_name}_training_summary.json"
        _save_json(summary_path, summary)
        artifacts["training_summary"] = str(summary_path.relative_to(Path(output_dir)))
    artifacts.update(_save_checkpoint_artifact(checkpoint_path, output_dir, dataset_name))
    return summary, artifacts


def evaluate_adversarial_attacks(
    dataset_name="mnist",
    data_root="data",
    checkpoint_dir="data",
    attack_overrides=None,
    selection_size=1000,
    eval_indices=None,
    batch_size=None,
    seed=42,
    device=None,
    download=True,
    output_dir=None,
):
    set_global_seed(seed)
    device = _select_device(device)
    _, test_dataset, dataset_config = load_data(dataset_name, data_root=data_root, download=download)
    checkpoint_path = get_model_checkpoint(dataset_name, checkpoint_dir=checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")

    model = _load_model_for_dataset(dataset_name, dataset_config, checkpoint_path, device)
    if eval_indices is not None:
        selected = _filter_correctly_classified_indices(
            model,
            test_dataset,
            eval_indices,
            device=device,
            batch_size=int(batch_size or dataset_config.get("batch_size", 256)),
        )
        if len(selected) == 0:
            raise RuntimeError(
                "Provided eval_indices has no correctly classified samples in dataset range."
            )
        chosen_indices = np.asarray(selected, dtype=int)
        chosen_count = int(len(chosen_indices))
        selection_mode = "fixed_indices_correct_only"
    else:
        accurate_indices = find_accurately_classified_samples(model, test_dataset)
        chosen_count = min(int(selection_size), len(accurate_indices))
        if chosen_count == 0:
            raise RuntimeError("No accurately classified samples found for adversarial evaluation.")
        chosen_indices = np.random.choice(accurate_indices, chosen_count, replace=False)
        selection_mode = "random_correct_subset"
    subset = Subset(test_dataset, indices=chosen_indices.tolist())
    eval_batch_size = min(chosen_count, int(batch_size or dataset_config.get("batch_size", 256)))
    test_loader = DataLoader(subset, batch_size=eval_batch_size, shuffle=False)

    attack_cfg = _normalized_attack_config(dataset_config, attack_overrides=attack_overrides)
    feature_dim = dataset_config["input_channels"] * (dataset_config["image_size"] ** 2)
    weight_matrix = _build_attack_weight_matrix(
        feature_dim,
        attack_cfg,
        device,
        input_channels=dataset_config["input_channels"],
        image_size=dataset_config["image_size"],
    )

    results = {}
    for name, hom in (("pgd", False), ("hom_pgd", True)):
        attacker = PGDAttack(model, device, hom=hom)
        start = perf_counter()
        clean_accuracy, adv_accuracy, success_rate, perturb_detail = attacker.evaluate_attack_success(
            model,
            test_loader,
            eps=attack_cfg["eps"],
            alpha=attack_cfg["alpha"],
            iters=int(attack_cfg["iters"]),
            norm_type=attack_cfg["norm"],
            W=weight_matrix,
            optimizer_name=attack_cfg["optimizer"],
            return_details=True,
        )
        elapsed = perf_counter() - start
        results[name] = {
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "success_rate": success_rate,
            "per_iter_sec": elapsed / max(int(attack_cfg["iters"]), 1),
            "perturbation": perturb_detail,
        }

    summary = {
        "dataset": dataset_name,
        "selection_size": chosen_count,
        "selection_mode": selection_mode,
        "attack_config": {
            "eps": float(attack_cfg["eps"]),
            "alpha": float(attack_cfg["alpha"]),
            "iters": attack_cfg["iters"],
            "norm": "inf" if attack_cfg["norm"] == np.inf else float(attack_cfg["norm"]),
            "weighted_norm": bool(attack_cfg.get("weighted_norm", False)),
            "weight_mode": attack_cfg.get("weight_mode", "random_rank1"),
            "weight_scale": float(attack_cfg.get("weight_scale", 0.01)),
            "optimizer": attack_cfg.get("optimizer", "gd"),
        },
        "results": results,
    }
    artifacts = {}
    if output_dir is not None:
        artifact_root = ensure_dir(Path(output_dir) / "artifacts")
        eval_indices_path = artifact_root / f"{dataset_name}_attack_eval_indices.npy"
        np.save(eval_indices_path, chosen_indices.astype(np.int64))
        artifacts["attack_eval_indices"] = str(eval_indices_path.relative_to(Path(output_dir)))
        summary["eval_indices_artifact"] = artifacts["attack_eval_indices"]
        summary_path = artifact_root / f"{dataset_name}_attack_summary.json"
        _save_json(summary_path, summary)
        artifacts["attack_summary"] = str(summary_path.relative_to(Path(output_dir)))
    artifacts.update(_save_checkpoint_artifact(checkpoint_path, output_dir, dataset_name))
    return summary, artifacts


def visualize_adversarial_attacks(
    dataset_name="mnist",
    data_root="data",
    checkpoint_dir="data",
    attack_overrides=None,
    eval_indices=None,
    num_examples=5,
    seed=42,
    device=None,
    download=True,
    output_dir=None,
):
    set_global_seed(seed)
    device = _select_device(device)
    _, test_dataset, dataset_config = load_data(dataset_name, data_root=data_root, download=download)
    checkpoint_path = get_model_checkpoint(dataset_name, checkpoint_dir=checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")
    model = _load_model_for_dataset(dataset_name, dataset_config, checkpoint_path, device)
    display_rows = max(int(num_examples), 1)
    target_examples = display_rows
    candidate_count = min(max(target_examples * 20, 64), len(test_dataset))
    if candidate_count == 0:
        raise RuntimeError("No samples available for visualization.")

    if eval_indices is not None:
        selected = _filter_correctly_classified_indices(
            model,
            test_dataset,
            eval_indices,
            device=device,
            batch_size=dataset_config.get("batch_size", 256),
        )
        if len(selected) == 0:
            raise RuntimeError(
                "Provided eval_indices has no correctly classified samples in dataset range."
            )
        indices = np.asarray(selected[:candidate_count], dtype=int)
    else:
        candidate_indices = np.random.choice(len(test_dataset), candidate_count, replace=False)
        correct_indices = _filter_correctly_classified_indices(
            model,
            test_dataset,
            candidate_indices,
            device=device,
            batch_size=dataset_config.get("batch_size", 256),
        )
        if len(correct_indices) == 0:
            raise RuntimeError("No correctly classified samples available for visualization.")
        indices = np.asarray(correct_indices[:candidate_count], dtype=int)
    images = torch.stack([test_dataset[i][0] for i in indices])
    labels = torch.tensor([test_dataset[i][1] for i in indices])

    attack_cfg = _normalized_attack_config(dataset_config, attack_overrides=attack_overrides)
    feature_dim = dataset_config["input_channels"] * (dataset_config["image_size"] ** 2)
    weight_matrix = _build_attack_weight_matrix(
        feature_dim,
        attack_cfg,
        device,
        input_channels=dataset_config["input_channels"],
        image_size=dataset_config["image_size"],
    )

    iterations = min(attack_cfg["iters"], 200)
    with torch.no_grad():
        clean_pred = torch.argmax(model(images.to(device)), dim=1).cpu()
    visualizations = {}
    artifacts = {}
    for attack_name, hom in (("pgd", False), ("hom_pgd", True)):
        attacker = PGDAttack(model, device, hom=hom)
        adv_images = attacker.generate(
            images.to(device),
            labels.to(device),
            eps=attack_cfg["eps"],
            alpha=attack_cfg["alpha"],
            iters=iterations,
            norm=attack_cfg["norm"],
            W=weight_matrix,
            optimizer_name=attack_cfg["optimizer"],
        ).cpu()
        with torch.no_grad():
            adv_pred = torch.argmax(model(adv_images.to(device)), dim=1).cpu()
        success_mask = (clean_pred == labels) & (adv_pred != labels)
        success_indices = torch.where(success_mask)[0][:display_rows]

        success_images = images[success_indices]
        success_labels = labels[success_indices]
        success_clean_pred = clean_pred[success_indices]
        success_adv_images = adv_images[success_indices]
        success_adv_pred = adv_pred[success_indices]

        fig, plt = _render_attack_figure(
            success_images,
            success_labels,
            success_clean_pred,
            success_adv_images,
            success_adv_pred,
            attack_name.upper(),
            rows=display_rows,
            class_names=dataset_config.get("class_names"),
        )
        visualizations[attack_name] = {
            "successful_examples": int(success_indices.numel()),
            "clean_predictions": [int(value) for value in success_clean_pred.tolist()],
            "adversarial_predictions": [int(value) for value in success_adv_pred.tolist()],
        }
        if output_dir is not None:
            artifact_root = ensure_dir(Path(output_dir) / "artifacts")
            fig_path = artifact_root / f"{dataset_name}_{attack_name}_examples.pdf"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            artifacts[f"{attack_name}_visualization"] = str(fig_path.relative_to(Path(output_dir)))
        plt.close(fig)

    artifacts.update(_save_checkpoint_artifact(checkpoint_path, output_dir, dataset_name))
    return {
        "dataset": dataset_name,
        "num_examples": display_rows,
        "candidate_pool": candidate_count,
        "visualizations": visualizations,
    }, artifacts


def adversarial_attack_workflow(
    dataset_name="mnist",
    train=False,
    attack=False,
    visualize=False,
    run_all=False,
    train_if_missing=False,
    data_root="data",
    checkpoint_dir="data",
    training_epochs=10,
    training_lr=None,
    batch_size=None,
    attack_selection_size=1000,
    attack_eval_indices=None,
    visualize_examples=5,
    visualize_eval_indices=None,
    attack_overrides=None,
    seed=42,
    device=None,
    dtype=None,
    download=True,
    output_dir=None,
    verbose=False,
    train_log_every=1,
):
    """Run the package-native adversarial workflow for the old 1.3 family."""
    del dtype

    if run_all:
        train = attack = visualize = True
    if not any([train, attack, visualize]):
        raise ValueError("At least one of train, attack, visualize, or run_all must be enabled.")

    dataset_name = dataset_name.lower()
    checkpoint_path = get_model_checkpoint(dataset_name, checkpoint_dir=checkpoint_dir)
    artifacts = {}
    metrics = {
        "dataset": dataset_name,
        "train": bool(train),
        "attack": bool(attack),
        "visualize": bool(visualize),
        "checkpoint": str(checkpoint_path),
    }

    def _run_training():
        return train_adversarial_model(
            dataset_name=dataset_name,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            training_epochs=training_epochs,
            training_lr=training_lr,
            batch_size=batch_size,
            seed=seed,
            device=device,
            download=download,
            output_dir=output_dir,
            verbose=verbose,
            log_every=train_log_every,
        )

    if train:
        train_summary, train_artifacts = _run_training()
        metrics["training"] = train_summary
        artifacts.update(train_artifacts)

    checkpoint_exists = checkpoint_path.exists()
    if (attack or visualize) and not checkpoint_exists:
        if train_if_missing:
            train_summary, train_artifacts = _run_training()
            metrics["training"] = train_summary
            artifacts.update(train_artifacts)
        else:
            raise FileNotFoundError(f"Checkpoint not found for attack workflow: {checkpoint_path}")

    if attack:
        attack_summary, attack_artifacts = evaluate_adversarial_attacks(
            dataset_name=dataset_name,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            attack_overrides=attack_overrides,
            selection_size=attack_selection_size,
            eval_indices=attack_eval_indices,
            batch_size=batch_size,
            seed=seed,
            device=device,
            download=download,
            output_dir=output_dir,
        )
        metrics["attack_summary"] = attack_summary
        artifacts.update(attack_artifacts)

    if visualize:
        viz_summary, viz_artifacts = visualize_adversarial_attacks(
            dataset_name=dataset_name,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            attack_overrides=attack_overrides,
            eval_indices=visualize_eval_indices,
            num_examples=visualize_examples,
            seed=seed,
            device=device,
            download=download,
            output_dir=output_dir,
        )
        metrics["visualization"] = viz_summary
        artifacts.update(viz_artifacts)

    objective = None
    feasible = None
    if "attack_summary" in metrics:
        objective = metrics["attack_summary"]["results"]["hom_pgd"]["success_rate"]

    return {
        "objective": objective,
        "feasible": feasible,
        "artifacts": artifacts,
        **metrics,
    }


def adversarial_attack_experiment(
    dataset_name="mnist",
    train=False,
    attack=False,
    visualize=False,
    run_all=False,
    train_if_missing=False,
    data_root="data",
    checkpoint_dir="data",
    training_epochs=10,
    training_lr=None,
    batch_size=None,
    attack_selection_size=1000,
    attack_eval_indices=None,
    visualize_examples=5,
    visualize_eval_indices=None,
    attack_overrides=None,
    norm_eps_sweep=None,
    auto_eps=None,
    single_case_subdir_by_norm=True,
    seed=42,
    device=None,
    dtype=None,
    download=True,
    output_dir=None,
    verbose=False,
    train_log_every=1,
):
    """High-level adversarial experiment wrapper for single runs and norm sweeps."""

    base_params = {
        "dataset_name": dataset_name,
        "train": train,
        "attack": attack,
        "visualize": visualize,
        "run_all": run_all,
        "train_if_missing": train_if_missing,
        "data_root": data_root,
        "checkpoint_dir": checkpoint_dir,
        "training_epochs": training_epochs,
        "training_lr": training_lr,
        "batch_size": batch_size,
        "attack_selection_size": attack_selection_size,
        "attack_eval_indices": attack_eval_indices,
        "visualize_examples": visualize_examples,
        "visualize_eval_indices": visualize_eval_indices,
        "attack_overrides": attack_overrides,
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "download": download,
        "verbose": verbose,
        "train_log_every": train_log_every,
    }

    def _run_case(case_params, case_output_dir, *, root_output_dir=None):
        run_params, auto_eps_value = _apply_auto_eps(case_params, auto_eps)
        attack_cfg = dict(run_params.get("attack_overrides", {}))
        _print_attack_config(attack_cfg, auto_eps_applied=auto_eps_value is not None)
        payload = adversarial_attack_workflow(output_dir=case_output_dir, **run_params)
        payload["artifacts"] = _rebase_artifacts(
            payload.get("artifacts", {}),
            root_output_dir=root_output_dir,
            case_output_dir=case_output_dir,
        )
        return payload, run_params, auto_eps_value

    def _ensure_base_model_once():
        checkpoint = get_model_checkpoint(dataset_name, checkpoint_dir=checkpoint_dir)
        if checkpoint.exists():
            return
        warmup = _merged(
            base_params,
            {
                "train": True,
                "attack": False,
                "visualize": False,
                "train_if_missing": False,
            },
        )
        warmup_out = None if output_dir is None else Path(output_dir) / "base_model_warmup"
        adversarial_attack_workflow(output_dir=warmup_out, **warmup)

    if norm_eps_sweep:
        _ensure_base_model_once()
        metrics = {"norm_results": {}, "auto_eps": dict(auto_eps or {})}
        shared_indices = None
        for norm, eps in list(norm_eps_sweep):
            case_params = _merged(
                base_params,
                {
                    "attack_overrides": {"norm": norm, "eps": eps},
                    "attack_eval_indices": shared_indices,
                    "train": False,
                    "train_if_missing": False,
                },
            )
            case_out = None if output_dir is None else Path(output_dir) / f"norm_{_norm_label(norm)}"
            payload, run_params, auto_eps_value = _run_case(case_params, case_out, root_output_dir=output_dir)
            summary = payload.get("attack_summary", {})
            if shared_indices is None:
                eval_idx_art = payload.get("artifacts", {}).get("attack_eval_indices")
                if output_dir is not None and eval_idx_art is not None:
                    eval_idx_path = Path(output_dir) / eval_idx_art
                    if eval_idx_path.exists():
                        shared_indices = np.load(eval_idx_path).astype(int).tolist()
            pgd = summary.get("results", {}).get("pgd", {})
            metrics["norm_results"][_norm_label(norm)] = {
                "norm": norm,
                "eps": auto_eps_value if auto_eps_value is not None else run_params.get("attack_overrides", {}).get("eps"),
                "hom_pgd_success_rate": payload.get("objective"),
                "pgd_success_rate": pgd.get("success_rate"),
                "selection_mode": summary.get("selection_mode"),
                "selection_size": summary.get("selection_size"),
                "output_dir": None if case_out is None else str(case_out),
            }

        artifacts = {}
        if output_dir is not None:
            artifact_root = ensure_dir(Path(output_dir) / "artifacts")
            summary_path = artifact_root / "adversarial_norm_sweep_summary.json"
            _save_json(summary_path, metrics)
            artifacts["norm_sweep_summary"] = str(summary_path.relative_to(Path(output_dir)))
        hom_vals = [
            value["hom_pgd_success_rate"]
            for value in metrics["norm_results"].values()
            if value["hom_pgd_success_rate"] is not None
        ]
        return {
            "objective": (sum(hom_vals) / len(hom_vals)) if hom_vals else None,
            "feasible": None,
            "metrics": metrics,
            "artifacts": artifacts,
        }

    single_out = Path(output_dir) / f"norm_{_norm_label(dict(attack_overrides or {}).get('norm', 'unknown'))}" if (output_dir is not None and single_case_subdir_by_norm) else output_dir
    payload, run_params, auto_eps_value = _run_case(base_params, single_out, root_output_dir=output_dir)
    payload.setdefault("metrics", {})
    payload["metrics"]["effective_attack_overrides"] = dict(run_params.get("attack_overrides", {}))
    payload["metrics"]["effective_auto_eps"] = {
        "enabled": bool(dict(auto_eps or {}).get("enabled", False)),
        "force_override": bool(dict(auto_eps or {}).get("force_override", False)),
        "applied": auto_eps_value is not None,
        "eps": auto_eps_value,
    }
    payload["metrics"]["single_run_output_dir"] = None if single_out is None else str(single_out)
    if auto_eps_value is not None:
        payload["metrics"]["auto_eps"] = {
            "enabled": True,
            "eps": auto_eps_value,
            "norm": run_params.get("attack_overrides", {}).get("norm"),
        }
    return payload


__all__ = [
    "AdaptiveCNN",
    "ModelTrainer",
    "PGDAttack",
    "ResidualBlock",
    "adversarial_attack_experiment",
    "adversarial_attack_workflow",
    "build_model",
    "create_weight_matrix",
    "evaluate_adversarial_attacks",
    "find_accurately_classified_samples",
    "find_lowest_confidence_accurate_samples",
    "get_dataset_config",
    "get_model_checkpoint",
    "load_data",
    "train_adversarial_model",
    "visualize_adversarial_attacks",
]
