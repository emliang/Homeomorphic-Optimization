"""Dataset utilities for adversarial attack experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from homopt.utils import ensure_dir


def _require_torchvision():
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "Adversarial workflow requires torchvision. Install with: pip install -e .[research]"
        ) from exc
    return datasets, transforms


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


def load_data(dataset_name: str, data_root="data", download=True):
    config = get_dataset_config(dataset_name)
    dataset_cls = config["dataset_cls"]
    train_dataset = dataset_cls(root=str(data_root), train=True, download=download, transform=config["train_transform"])
    test_dataset = dataset_cls(root=str(data_root), train=False, download=download, transform=config["test_transform"])
    return train_dataset, test_dataset, config


__all__ = ["get_dataset_config", "get_model_checkpoint", "load_data"]
