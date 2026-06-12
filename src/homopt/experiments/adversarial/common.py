"""Shared utilities for adversarial attack experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from .model import build_model
from homopt.utils import ensure_dir, resolve_torch_device


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


__all__ = [
    "_load_model_for_dataset",
    "_merged",
    "_norm_label",
    "_rebase_artifacts",
    "_save_checkpoint_artifact",
    "_save_json",
    "_select_device",
]
