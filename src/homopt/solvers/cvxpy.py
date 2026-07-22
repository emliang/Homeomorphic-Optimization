"""CVXPY backend utilities shared by exact solvers."""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover - depends on optional env state
    cp = None
    _CVXPY_IMPORT_ERROR = exc
else:
    _CVXPY_IMPORT_ERROR = None

CVXPY_SOLVER_PRIORITY = ("MOSEK", "ECOS", "SCS")
CVXPY_INTEGER_SOLVER_PRIORITY = ("GUROBI",)
_CVXPY_SOLVER_CACHE = {}


def _require_cvxpy():
    if cp is None:
        raise ImportError(
            "This solver requires cvxpy. Install with: pip install -r requirements.txt"
        ) from _CVXPY_IMPORT_ERROR
    return cp


def _soc_constraint(cp_mod, t_expr, x_expr):
    if hasattr(cp_mod, "SOC"):
        return cp_mod.SOC(t_expr, x_expr)
    return cp_mod.norm(x_expr, 2) <= t_expr


def _sum_multiply(cp_mod, x_expr, coeffs):
    if hasattr(cp_mod, "multiply") and hasattr(cp_mod, "sum"):
        return cp_mod.sum(cp_mod.multiply(x_expr, coeffs))
    return cp_mod.sum_entries(cp_mod.mul_elemwise(coeffs, x_expr))


def _candidate_cvxpy_solvers(cp_mod, solver_priority=CVXPY_SOLVER_PRIORITY):
    cache_key = (cp_mod, tuple(solver_priority))
    if cache_key in _CVXPY_SOLVER_CACHE:
        return list(_CVXPY_SOLVER_CACHE[cache_key])
    try:
        installed = set(cp_mod.installed_solvers())
    except Exception:
        installed = set()
    candidates = [name for name in solver_priority if name in installed]
    _CVXPY_SOLVER_CACHE[cache_key] = tuple(candidates)
    return candidates


def _solver_time_limit_kwargs(solver_name, time_limit_sec):
    if time_limit_sec is None:
        return {}
    time_limit_sec = float(time_limit_sec)
    if solver_name == "GUROBI":
        return {"TimeLimit": time_limit_sec}
    if solver_name == "MOSEK":
        return {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": time_limit_sec}}
    if solver_name == "SCS":
        return {"time_limit_secs": time_limit_sec}
    return {}


def _merge_mosek_params(kwargs, extra_params):
    if not extra_params:
        return kwargs
    merged = dict(kwargs)
    merged["mosek_params"] = {
        **dict(merged.get("mosek_params") or {}),
        **dict(extra_params),
    }
    return merged


def _solver_name_text(solver_name):
    if solver_name is None:
        return None
    return str(solver_name)


def _normalize_preferred_solver(preferred_solver):
    """Return a canonical explicit solver request, or ``None`` for auto.

    ``auto`` is the only request that permits the project priority/fallback
    chain.  Any other value is an explicit user choice and is consequently
    tried alone.
    """

    if preferred_solver is None:
        return None
    solver_name = _solver_name_text(preferred_solver).strip()
    if not solver_name or solver_name.lower() == "auto":
        return None
    return solver_name.upper()


def _candidate_cvxpy_solvers_for_request(cp_mod, solver_priority, preferred_solver=None):
    preferred_solver = _normalize_preferred_solver(preferred_solver)
    if preferred_solver is None:
        return _candidate_cvxpy_solvers(cp_mod, solver_priority=solver_priority)
    try:
        installed = set(cp_mod.installed_solvers())
    except Exception:
        installed = set()
    return [preferred_solver] if preferred_solver in installed else []


def _start_gurobi_env(*, verbose, solver_options):
    import gurobipy as gp

    env_params = dict(solver_options.pop("gurobi_env_params", {}) or {})
    env = gp.Env(empty=True)
    try:
        env.setParam("OutputFlag", int(bool(verbose)))
        for key, value in env_params.items():
            env.setParam(key, value)
        env.start()
        return env
    except Exception:
        try:
            env.close()
        except Exception:
            pass
        raise


def _solve_cvxpy_problem(
    problem,
    cp_mod,
    *,
    warm_start=True,
    interior_point=False,
    verbose=False,
    solver_options=None,
    time_limit_sec=None,
    solver_priority=CVXPY_SOLVER_PRIORITY,
    preferred_solver=None,
):
    requested_solver = _normalize_preferred_solver(preferred_solver)
    candidates = _candidate_cvxpy_solvers_for_request(
        cp_mod,
        solver_priority=solver_priority,
        preferred_solver=requested_solver,
    )
    last_error = None
    attempted_solvers = []
    for solver_name in candidates:
        solver = getattr(cp_mod, solver_name, None)
        if solver is None:
            continue
        attempted_solvers.append(_solver_name_text(solver_name))
        call_solver_options = dict(solver_options or {})
        gurobi_env = None
        if _solver_name_text(solver_name).upper() == "GUROBI":
            if "env" in call_solver_options:
                call_solver_options.pop("gurobi_env_params", None)
            else:
                gurobi_env = _start_gurobi_env(verbose=verbose, solver_options=call_solver_options)
                call_solver_options["env"] = gurobi_env
        kwargs = {
            "warm_start": warm_start,
            "solver": solver,
            "verbose": bool(verbose),
            **call_solver_options,
        }
        time_limit_kwargs = _solver_time_limit_kwargs(solver_name, time_limit_sec)
        if solver_name == "MOSEK":
            kwargs = _merge_mosek_params(kwargs, time_limit_kwargs.get("mosek_params"))
        else:
            kwargs.update(time_limit_kwargs)
        if interior_point and solver_name == "MOSEK":
            kwargs = _merge_mosek_params(
                kwargs,
                {
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1,
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-5,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-1,
                },
            )
        try:
            problem.solve(**kwargs)
            return {
                "solver_requested": requested_solver or "auto",
                "solver_used": _solver_name_text(solver_name),
                "solver_attempts": attempted_solvers,
                "solver_fallback_used": requested_solver is None and len(attempted_solvers) > 1,
                "solver_candidates": [_solver_name_text(candidate) for candidate in candidates],
                "cvxpy_status": getattr(problem, "status", None),
            }
        except Exception as exc:
            last_error = exc
        finally:
            if gurobi_env is not None:
                gurobi_env.close()
    if last_error is not None:
        raise last_error
    raise RuntimeError(
        "No supported CVXPY solver is installed. Expected one of: "
        f"{requested_solver if requested_solver is not None else ', '.join(solver_priority)}"
    )


def _symmetric_matrix_variable(cp_mod, size):
    try:
        return cp_mod.Variable((size, size), symmetric=True), []
    except TypeError:
        X = cp_mod.Variable((size, size))
        constraints = []
        for i in range(size):
            for j in range(i + 1, size):
                constraints.append(X[i, j] == X[j, i])
        return X, constraints

__all__ = [
    "CVXPY_INTEGER_SOLVER_PRIORITY",
    "CVXPY_SOLVER_PRIORITY",
    "_candidate_cvxpy_solvers",
    "_require_cvxpy",
    "_solve_cvxpy_problem",
    "_soc_constraint",
    "_sum_multiply",
    "_symmetric_matrix_variable",
]
