"""Shared INN/IPNN mapping training entrypoints."""

from __future__ import annotations

from .cache import build_training_payload


MAPPING_MODEL_KEYS = {
    "inn": "model_config",
    "ipnn": "ipnn_model_config",
}


def normalize_mapping_type(mapping_type):
    key = str(mapping_type).strip().lower()
    if key not in MAPPING_MODEL_KEYS:
        raise ValueError(f"Unsupported mapping_type: {mapping_type}. Use 'inn' or 'ipnn'.")
    return key


def build_learning_mapping_payload(
    mapping_type,
    problem_args,
    model_args,
    train_args,
    *,
    runtime_device=None,
    runtime_dtype=None,
    ensure_results_save_freq=False,
):
    """Build the canonical training payload for INN/IPNN mapping models."""

    key = normalize_mapping_type(mapping_type)
    model_payload = dict(model_args or {})
    if runtime_device is not None:
        model_payload["device"] = str(runtime_device)
    if runtime_dtype is not None:
        model_payload["dtype"] = runtime_dtype
    payload = build_training_payload(
        problem_args,
        model_payload,
        train_args,
        problem_key="problem_config",
        model_key=MAPPING_MODEL_KEYS[key],
        ensure_results_save_freq=ensure_results_save_freq,
    )
    return payload


def load_or_train_learning_mapping(mapping_type, data, training_payload, save_dir, *, retrain=False):
    """Load or train an INN/IPNN mapping through one public call shape."""

    key = normalize_mapping_type(mapping_type)
    if key == "inn":
        from .inn import load_or_train_inn_mapping

        return load_or_train_inn_mapping(data, training_payload, save_dir, retrain=retrain)
    if key == "ipnn":
        from .ipnn import load_or_train_ipnn_mapping

        return load_or_train_ipnn_mapping(data, training_payload, save_dir, retrain=retrain)
    raise AssertionError(f"Unhandled mapping_type: {mapping_type}")


def prepare_learning_mapping(
    mapping_type,
    data,
    *,
    problem_args,
    model_args,
    train_args,
    save_dir,
    retrain=False,
    runtime_device=None,
    runtime_dtype=None,
    ensure_results_save_freq=False,
):
    """Build payload and load/train a mapping model for a problem family."""

    payload = build_learning_mapping_payload(
        mapping_type,
        problem_args,
        model_args,
        train_args,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        ensure_results_save_freq=ensure_results_save_freq,
    )
    model, training_record, model_path, record_path = load_or_train_learning_mapping(
        mapping_type,
        data,
        payload,
        save_dir,
        retrain=retrain,
    )
    return {
        "payload": payload,
        "model": model,
        "training_record": training_record,
        "model_path": model_path,
        "record_path": record_path,
    }


__all__ = [
    "MAPPING_MODEL_KEYS",
    "build_learning_mapping_payload",
    "load_or_train_learning_mapping",
    "normalize_mapping_type",
    "prepare_learning_mapping",
]
