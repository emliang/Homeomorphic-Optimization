"""Shared naming helpers for experiment runs and artifacts."""

from __future__ import annotations


def sanitize_label(value, *, fallback="unlabeled"):
    label = str(value or "").strip().lower()
    label = "".join(char if char.isalnum() else "_" for char in label)
    label = "_".join(part for part in label.split("_") if part)
    return label or fallback


def scale_label(params, *, fallback=""):
    return sanitize_label((params or {}).get("scale_label"), fallback=fallback)


def labeled_run_name(base_name, params):
    label = scale_label(params, fallback="")
    return f"{base_name}_{label}" if label else str(base_name)


def labeled_artifact_prefix(base_prefix, params, *, explicit_prefix=None):
    if explicit_prefix:
        return str(explicit_prefix)
    label = scale_label(params, fallback="")
    if not label:
        return str(base_prefix)
    base_prefix = str(base_prefix)
    return label if label.startswith(base_prefix) else f"{base_prefix}_{label}"


__all__ = [
    "labeled_artifact_prefix",
    "labeled_run_name",
    "sanitize_label",
    "scale_label",
]
