"""Shared model-training checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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

    if (not retrain) and model_checkpoint_path.exists():
        if not record_path.exists():
            raise FileNotFoundError(
                f"Missing training record for checkpoint {model_checkpoint_path}: expected {record_path}."
            )
        model = load_fn(model_checkpoint_path, map_location=data.device)
        model = model.to(device=data.device, dtype=data.dtype)
        training_record = np.load(record_path, allow_pickle=True).item()
        return model, training_record, model_checkpoint_path, record_path

    model, training_record = train_fn(data, args, save_dir)
    model_checkpoint_path = save_fn(model, save_dir, filename=model_filename)
    record_path = save_training_record(save_dir / record_filename, training_record)
    return model, training_record, model_checkpoint_path, record_path


def training_time_from_record(training_record):
    return float(np.sum(np.asarray(training_record.get("training_time_list", []), dtype=float)))


__all__ = [
    "build_training_payload",
    "load_or_train_mapping",
    "save_training_record",
    "training_time_from_record",
]
