"""Filesystem helpers shared across package layers."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path):
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


__all__ = ["ensure_dir"]
