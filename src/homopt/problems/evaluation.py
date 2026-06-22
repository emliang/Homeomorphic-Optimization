"""Problem-evaluation helpers shared by solvers and reports."""

from __future__ import annotations

import inspect


def _supports_keyword(callable_obj, keyword):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def call_problem_constraint(problem, x, *, include_equalities, clip):
    constraint_x = problem.constraint_x
    kwargs = {}
    if _supports_keyword(constraint_x, "clip"):
        kwargs["clip"] = clip
    if _supports_keyword(constraint_x, "eq_cons"):
        kwargs["eq_cons"] = bool(include_equalities)
    return constraint_x(x, **kwargs)


__all__ = ["call_problem_constraint"]
