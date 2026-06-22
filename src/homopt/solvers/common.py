"""Shared exact-solver result and call-normalization helpers."""

from __future__ import annotations

import inspect
from time import perf_counter

import numpy as np

_SOLVER_SIGNATURE_CACHE = {}


def _as_numpy_or_none(value):
    if value is None:
        return None
    return np.asarray(value)


def _normalized_exact_solver_result(
    *,
    solution=None,
    status="unknown",
    objective=None,
    runtime_total=None,
    violation=None,
    feasible=None,
    extras=None,
):
    return {
        "solution": _as_numpy_or_none(solution),
        "status": str(status),
        "objective": None if objective is None else float(objective),
        "runtime_total": None if runtime_total is None else float(runtime_total),
        "violation": None if violation is None else float(violation),
        "feasible": None if feasible is None else bool(feasible),
        "extras": dict(extras or {}),
    }


def _solver_signature_or_none(fn):
    owner = getattr(fn, "__self__", None)
    cache_key = (type(owner) if owner is not None else None, getattr(fn, "__func__", fn))
    if cache_key in _SOLVER_SIGNATURE_CACHE:
        return _SOLVER_SIGNATURE_CACHE[cache_key]
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None
    _SOLVER_SIGNATURE_CACHE[cache_key] = signature
    return signature


def _filter_solver_kwargs(fn, kwargs):
    signature = _solver_signature_or_none(fn)
    if signature is None:
        return dict(kwargs)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(kwargs)
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }


def _solver_accepts_keyword(fn, keyword):
    signature = _solver_signature_or_none(fn)
    if signature is None:
        return True
    parameters = signature.parameters
    return keyword in parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )


def normalize_exact_solver_call_kwargs(exact_solver, *, solve_config=None, **kwargs):
    normalized = dict(solve_config or {})
    normalized.update(kwargs)
    target = exact_solver.solve_result if hasattr(exact_solver, "solve_result") else exact_solver.solve
    if (
        "solver" not in normalized
        and "solver_name" in normalized
        and not _solver_accepts_keyword(target, "solver_name")
        and _solver_accepts_keyword(target, "solver")
    ):
        normalized["solver"] = normalized.pop("solver_name")
    elif "solver_name" in normalized and not _solver_accepts_keyword(target, "solver_name"):
        raise TypeError(f"{type(exact_solver).__name__} does not accept explicit solver_name.")
    if (
        "options" not in normalized
        and "solver_options" in normalized
        and not _solver_accepts_keyword(target, "solver_options")
        and _solver_accepts_keyword(target, "options")
    ):
        normalized["options"] = normalized.pop("solver_options")
    elif "solver_options" in normalized and not _solver_accepts_keyword(target, "solver_options"):
        raise TypeError(f"{type(exact_solver).__name__} does not accept explicit solver_options.")
    return _filter_solver_kwargs(target, normalized)


def solve_exact_result(exact_solver, *args, **kwargs):
    call_kwargs = normalize_exact_solver_call_kwargs(exact_solver, **kwargs)
    if hasattr(exact_solver, "solve_result"):
        return exact_solver.solve_result(*args, **call_kwargs)
    start = perf_counter()
    raw = exact_solver.solve(*args, **call_kwargs)
    runtime_total = perf_counter() - start
    if isinstance(raw, dict):
        solution = raw.get("solution", raw.get("x_optimal"))
        status = raw.get("status", "unknown")
        objective = raw.get("objective", raw.get("objective_value"))
        violation = raw.get("violation", raw.get("max_violation"))
        feasible = raw.get("feasible")
        extras = {
            key: value
            for key, value in raw.items()
            if key not in {"solution", "x_optimal", "status", "objective", "objective_value", "violation", "max_violation", "feasible"}
        }
        return _normalized_exact_solver_result(
            solution=solution,
            status=status,
            objective=objective,
            runtime_total=runtime_total,
            violation=violation,
            feasible=feasible,
            extras=extras,
        )
    return _normalized_exact_solver_result(
        solution=raw,
        status="optimal" if raw is not None else "failed",
        objective=None,
        runtime_total=runtime_total,
        violation=None,
        feasible=None if raw is None else True,
        extras={},
    )

__all__ = [
    "_normalized_exact_solver_result",
    "normalize_exact_solver_call_kwargs",
    "solve_exact_result",
]
