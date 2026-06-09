"""Helpers for resilient optional imports."""

from __future__ import annotations

from importlib import import_module


def optional_import(module_name, symbol_name):
    """Return a symbol from a module or ``None`` if import/loading fails."""
    try:
        module = import_module(module_name)
        return getattr(module, symbol_name)
    except Exception:
        return None
