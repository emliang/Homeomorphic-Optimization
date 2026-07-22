"""Shared model-training checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


TRAINING_IDENTITY_VERSION = 1


def _identity_json_value(value):
    """Return a deterministic representation for a model-training payload."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return {"__float__": "nan"}
        if np.isposinf(value):
            return {"__float__": "inf"}
        if np.isneginf(value):
            return {"__float__": "-inf"}
        return value
    if isinstance(value, np.generic):
        return _identity_json_value(value.item())
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, (torch.dtype, torch.device)):
        return {"__torch_type__": str(value)}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        }
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            }
        }
    if isinstance(value, dict):
        return {
            str(key): _identity_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, range)):
        return [_identity_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_identity_json_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    raise TypeError(
        "Training identities must contain deterministic configuration values; "
        f"got unsupported value of type {type(value).__name__}."
    )


def build_training_identity(*, data, args, model_filename, record_filename):
    """Fingerprint the payload and runtime needed to safely reuse a checkpoint."""

    payload = {
        "args": dict(args or {}),
        "model_filename": str(model_filename),
        "record_filename": str(record_filename),
        "runtime": {
            "device": str(data.device),
            "dtype": str(data.dtype),
        },
    }
    canonical_payload = _identity_json_value(payload)
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return {
        "version": TRAINING_IDENTITY_VERSION,
        "fingerprint": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "payload": canonical_payload,
    }


def training_identity_path(save_dir, model_filename):
    checkpoint_path = Path(save_dir) / model_filename
    return checkpoint_path.with_name(f"{checkpoint_path.name}.identity.json")


def save_training_identity(path, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _require_matching_training_identity(path, identity, checkpoint_path):
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"Missing training identity for checkpoint {checkpoint_path}: expected {path}. "
            "Set retrain=True to regenerate a verified checkpoint."
        )
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unreadable training identity for checkpoint {checkpoint_path}: {path}. "
            "Set retrain=True to regenerate a verified checkpoint."
        ) from exc
    if not isinstance(stored, dict) or stored.get("version") != TRAINING_IDENTITY_VERSION:
        raise ValueError(
            f"Unsupported training identity for checkpoint {checkpoint_path}: {path}. "
            "Set retrain=True to regenerate a verified checkpoint."
        )
    if stored.get("fingerprint") != identity["fingerprint"]:
        raise ValueError(
            f"Training checkpoint identity mismatch for {checkpoint_path}. "
            "The checkpoint was trained for a different configuration; set retrain=True."
        )


def save_training_record(path, training_record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, training_record, allow_pickle=True)
    return path


def build_training_payload(
    problem_args,
    model_args,
    train_args,
    *,
    problem_key="problem_config",
    model_key="model_config",
    ensure_results_save_freq=False,
):
    payload = {
        **dict(model_args or {}),
        "n_samples": int(train_args["n_samples"]),
        "batch_size": int(train_args["batch_size"]),
        "total_iteration": int(train_args["total_iteration"]),
    }
    if "pre_training" in train_args:
        payload["pre_training"] = int(train_args["pre_training"])
    if "resultsSaveFreq" in train_args:
        payload["resultsSaveFreq"] = int(train_args["resultsSaveFreq"])
    if ensure_results_save_freq and "resultsSaveFreq" not in payload:
        payload["resultsSaveFreq"] = int(train_args["total_iteration"]) + 1
    return {
        problem_key: dict(problem_args or {}),
        model_key: payload,
    }


def load_or_train_mapping(
    *,
    data,
    args,
    save_dir,
    retrain,
    model_filename,
    record_filename,
    load_fn,
    save_fn,
    train_fn,
):
    save_dir = Path(save_dir)
    model_checkpoint_path = save_dir / model_filename
    record_path = save_dir / record_filename
    identity = build_training_identity(
        data=data,
        args=args,
        model_filename=model_filename,
        record_filename=record_filename,
    )
    identity_path = training_identity_path(save_dir, model_filename)

    if (not retrain) and model_checkpoint_path.exists():
        if not record_path.exists():
            raise FileNotFoundError(
                f"Missing training record for checkpoint {model_checkpoint_path}: expected {record_path}."
            )
        _require_matching_training_identity(identity_path, identity, model_checkpoint_path)
        model = load_fn(model_checkpoint_path, map_location=data.device)
        model = model.to(device=data.device, dtype=data.dtype)
        training_record = np.load(record_path, allow_pickle=True).item()
        return model, training_record, model_checkpoint_path, record_path

    model, training_record = train_fn(data, args, save_dir)
    model_checkpoint_path = save_fn(model, save_dir, filename=model_filename)
    record_path = save_training_record(save_dir / record_filename, training_record)
    save_training_identity(identity_path, identity)
    return model, training_record, model_checkpoint_path, record_path


def training_time_from_record(training_record):
    return float(np.sum(np.asarray(training_record.get("training_time_list", []), dtype=float)))


__all__ = [
    "build_training_identity",
    "build_training_payload",
    "load_or_train_mapping",
    "save_training_record",
    "save_training_identity",
    "training_identity_path",
    "training_time_from_record",
]
