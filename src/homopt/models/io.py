"""Model serialization helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch


def save_torch_model(model, save_dir, filename):
    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / filename
    torch.save(model, model_path)
    return model_path


def load_torch_model(path_or_dir, filename, map_location=None):
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / filename
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


__all__ = ["load_torch_model", "save_torch_model"]
