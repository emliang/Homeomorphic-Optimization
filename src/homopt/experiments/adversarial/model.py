"""Model and trainer components for adversarial experiments."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homopt.utils import ensure_dir


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


__all__ = ["AdaptiveCNN", "ModelTrainer", "ResidualBlock", "build_model"]
