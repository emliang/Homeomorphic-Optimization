"""Shared INN-family training cache helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import save_numpy
from .config import merged
from homopt.learning.inn_training import train_mdh_mapping
from homopt.learning.ipnn_training import train_ipnn_mapping
from homopt.models import (
    load_inn_mapping,
    load_ipnn_mapping,
    save_inn_mapping,
    save_ipnn_mapping,
)


_RETIRED_MODEL_MODULES = {
    "homopt.models._inn_impl",
    "homopt.models._ipnn_impl",
    "homopt.models._predictor_impl",
    "homopt.models._nn_common",
}


def _is_retired_model_checkpoint_error(exc):
    return isinstance(exc, ModuleNotFoundError) and str(getattr(exc, "name", "")) in _RETIRED_MODEL_MODULES


def build_inn_runtime_args(
    *,
    runtime_device,
    runtime_dtype,
    base_inn_args,
    base_optimizer_config,
    model_config=None,
    optimizer_config=None,
):
    model_args = merged(base_inn_args, model_config)
    model_args["device"] = str(runtime_device)
    model_args["dtype"] = runtime_dtype

    optimizer_args = merged(base_optimizer_config, optimizer_config)
    return model_args, optimizer_args


def build_qcqp_inn_training_payload(problem_args, model_args, train_args, *, ensure_results_save_freq=False):
    payload = {
        **model_args,
        "n_samples": int(train_args["n_samples"]),
        "batch_size": int(train_args["batch_size"]),
        "total_iteration": int(train_args["total_iteration"]),
    }
    if ensure_results_save_freq and "resultsSaveFreq" not in payload:
        payload["resultsSaveFreq"] = int(train_args["total_iteration"]) + 1
    return {
        "problem_config": problem_args,
        "model_config": payload,
    }


def build_qcqp_ipnn_training_payload(problem_args, model_args, train_args, *, ensure_results_save_freq=False):
    payload = {
        **model_args,
        "n_samples": int(train_args["n_samples"]),
        "batch_size": int(train_args["batch_size"]),
        "total_iteration": int(train_args["total_iteration"]),
    }
    if "pre_training" in train_args:
        payload["pre_training"] = int(train_args["pre_training"])
    if ensure_results_save_freq and "resultsSaveFreq" not in payload:
        payload["resultsSaveFreq"] = int(train_args["total_iteration"]) + 1
    return {
        "problem_config": problem_args,
        "ipnn_model_config": payload,
    }


def _load_or_train_mapping(
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
        try:
            model = load_fn(model_checkpoint_path, map_location=data.device)
            model = model.to(device=data.device, dtype=data.dtype)
            training_record = np.load(record_path, allow_pickle=True).item()
            return model, training_record, model_checkpoint_path, record_path
        except ModuleNotFoundError as exc:
            if not _is_retired_model_checkpoint_error(exc):
                raise
            print(
                f"[HomOPT] Ignoring stale model checkpoint {model_checkpoint_path}: "
                f"references retired module {exc.name!r}; retraining."
            )

    model, training_record = train_fn(data, args, save_dir)
    model_checkpoint_path = save_fn(model, save_dir, filename=model_filename)
    record_path = save_numpy(save_dir / record_filename, training_record)
    return model, training_record, model_checkpoint_path, record_path


def load_or_train_inn_mapping(data, args, save_dir, *, retrain=False):
    return _load_or_train_mapping(
        data=data,
        args=args,
        save_dir=save_dir,
        retrain=retrain,
        model_filename="inn_mapping.pt",
        record_filename="inn_training_record.npy",
        load_fn=load_inn_mapping,
        save_fn=save_inn_mapping,
        train_fn=train_mdh_mapping,
    )


def load_or_train_ipnn_mapping(data, args, save_dir, *, retrain=False):
    return _load_or_train_mapping(
        data=data,
        args=args,
        save_dir=save_dir,
        retrain=retrain,
        model_filename="ipnn_mapping.pt",
        record_filename="ipnn_training_record.npy",
        load_fn=load_ipnn_mapping,
        save_fn=save_ipnn_mapping,
        train_fn=train_ipnn_mapping,
    )


def training_time_from_record(training_record):
    return float(np.sum(np.asarray(training_record.get("training_time_list", []), dtype=float)))


__all__ = [
    "build_inn_runtime_args",
    "build_qcqp_inn_training_payload",
    "build_qcqp_ipnn_training_payload",
    "load_or_train_inn_mapping",
    "load_or_train_ipnn_mapping",
    "training_time_from_record",
]
