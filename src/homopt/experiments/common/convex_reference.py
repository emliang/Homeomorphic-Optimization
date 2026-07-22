"""Convex experiment construction, reference solving, and cache ownership."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from homopt.mappings import GaugeMap
from homopt.problems import ConvexOpt, ConvexOptEq, create_test_problem
from homopt.problems.evaluation import call_problem_constraint
from homopt.records.artifacts import artifact_root
from homopt.solvers import ConvexSolver, solve_exact_result
from homopt.utils import cast_tensors_to_dtype


def build_convex_problem_config(
    *,
    seed,
    n_var,
    n_linear_cons,
    n_soc_cons,
    n_qua_cons,
    n_lin_eq=0,
    obj="quad",
    x_lower=-5.0,
    x_upper=5.0,
    margin_scale=0.1,
    anchor_interior_ratio=0.8,
    objective_quadratic_type="diagonal",
    constraint_quadratic_type="diagonal",
    quad_diag_lower=1e-2,
    quad_diag_upper=1.0,
    low_rank_quad_ridge=1e-2,
    explicit_config=None,
):
    config = create_test_problem(
        {
            "obj": obj,
            "seed": seed,
            "n_var": n_var,
            "n_linear_cons": n_linear_cons,
            "n_qua_cons": n_qua_cons,
            "n_soc_cons": n_soc_cons,
            "n_lin_eq": n_lin_eq,
            "x_lower": x_lower,
            "x_upper": x_upper,
            "margin_scale": margin_scale,
            "anchor_interior_ratio": anchor_interior_ratio,
            "objective_quadratic_type": objective_quadratic_type,
            "constraint_quadratic_type": constraint_quadratic_type,
            "quad_diag_lower": quad_diag_lower,
            "quad_diag_upper": quad_diag_upper,
            "low_rank_quad_ridge": low_rank_quad_ridge,
        }
    )
    if explicit_config:
        config.update(explicit_config)
    return config


def normalize_convex_problem_type(problem_type):
    normalized = str(problem_type).lower()
    if normalized not in {"socp", "socp_eq"}:
        raise ValueError("problem_type must be one of: 'socp', 'socp_eq'.")
    return normalized


def build_convex_problem(*, config, runtime_device, runtime_dtype, problem_type="socp"):
    problem_type = normalize_convex_problem_type(problem_type)
    problem = ConvexOptEq(config) if problem_type == "socp_eq" else ConvexOpt(config)
    return cast_tensors_to_dtype(problem.to_device(runtime_device), runtime_dtype)


def build_convex_hom_map(
    problem,
    *,
    runtime_device,
    runtime_dtype,
    x_origin,
    smooth=False,
    hom_p_norm=2,
    hom_map_explicit_gradient_rule="polynomial",
    hom_map_smooth_tie_tol=1e-7,
    hom_map_smooth_temperature=1e-4,
):
    return cast_tensors_to_dtype(
        GaugeMap(
            problem,
            p_norm=hom_p_norm,
            x_origin=x_origin,
            smooth=smooth,
            explicit_gradient_rule=hom_map_explicit_gradient_rule,
            smooth_tie_tol=hom_map_smooth_tie_tol,
            smooth_temperature=hom_map_smooth_temperature,
        ).to_device(runtime_device),
        runtime_dtype,
    )


def normalize_hom_origin_constraints(value):
    aliases = {
        "full": "full",
        "all": "full",
        "eq": "full",
        "equality": "full",
        "ineq": "ineq",
        "inequality": "ineq",
        "inequalities": "ineq",
    }
    key = str(value).lower()
    if key not in aliases:
        raise ValueError("hom_origin_constraints must be one of: 'full', 'ineq'.")
    return aliases[key]


def _origin_equality_violation(problem, x_origin):
    if x_origin is None or getattr(problem, "n_eq", 0) <= 0:
        return None
    x_tensor = torch.as_tensor(x_origin, dtype=problem.Q.dtype, device=problem.Q.device).view(1, -1)
    return float(problem.eq_constraint_x(x_tensor).abs().max().detach().cpu().item())


def _project_to_equalities(problem, x):
    if getattr(problem, "n_eq", 0) <= 0 or getattr(problem, "A_eq", None) is None:
        return x
    residual = x @ problem.A_eq.T - problem.b_eq
    gram_inv = torch.linalg.pinv(problem.A_eq @ problem.A_eq.T)
    return x - residual @ gram_inv @ problem.A_eq


def _approximate_chebyshev_origin(
    problem,
    *,
    include_equality=False,
    iterations=200,
    learning_rate=5e-2,
    temperature=1e-2,
):
    """Approximate an interior origin when a central-point solver is unavailable."""

    device = getattr(problem, "device", torch.device("cpu"))
    dtype = getattr(problem.Q, "dtype", torch.float32)
    lower = problem.L.to(device=device, dtype=dtype).view(1, -1)
    upper = problem.U.to(device=device, dtype=dtype).view(1, -1)
    x = (0.5 * (lower + upper)).detach()
    if include_equality:
        x = _project_to_equalities(problem, x).clamp(lower, upper)
    x_var = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_var], lr=float(learning_rate))
    best_x = x_var.detach().clone()
    best_score = float("inf")
    temp = float(max(temperature, 1e-8))

    for _ in range(int(max(iterations, 1))):
        optimizer.zero_grad()
        ineq_residual = call_problem_constraint(problem, x_var, include_equalities=False, clip=False)
        loss = temp * torch.logsumexp(ineq_residual / temp, dim=1).mean()
        if include_equality and getattr(problem, "n_eq", 0) > 0:
            eq_residual = problem.eq_constraint_x(x_var)
            loss = loss + 100.0 * torch.mean(eq_residual * eq_residual)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x_var.clamp_(lower, upper)
            if include_equality:
                x_var.copy_(_project_to_equalities(problem, x_var).clamp(lower, upper))
            score = float(
                call_problem_constraint(problem, x_var, include_equalities=False, clip=False)
                .max()
                .detach()
                .cpu()
                .item()
            )
            if include_equality and getattr(problem, "n_eq", 0) > 0:
                score += float(problem.eq_constraint_x(x_var).abs().max().detach().cpu().item())
            if score < best_score:
                best_score = score
                best_x = x_var.detach().clone()

    return best_x.view(-1).detach().cpu().numpy()


# Solver request controls are part of the cache key below, so older entries
# naturally miss rather than being reused under the new provenance contract.
_CONVEX_REFERENCE_CACHE_VERSION = 1
_CONVEX_REFERENCE_CACHE_FILENAME = "convex_reference_context_cache.npy"
_CVXPY_BACKEND_LABELS = {
    "CLARABEL",
    "CVXOPT",
    "ECOS",
    "GUROBI",
    "MOSEK",
    "OSQP",
    "SCS",
}


def normalize_convex_reference_solver(value):
    """Normalize the public convex reference solver request.

    ``auto`` is deliberately distinct from a named backend: auto may follow
    the CVXPY priority/fallback chain, while a named backend is passed through
    as an explicit no-fallback request.
    """

    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized.upper()


def _reference_solver_request_label(reference_solver):
    return "auto" if reference_solver is None else str(reference_solver)


def _reference_solve_kwargs(
    *,
    reference_solver,
    reference_solver_options,
    reference_solver_time_limit_sec,
    reference_solver_verbose,
):
    kwargs = {}
    if reference_solver is not None:
        kwargs["solver_name"] = reference_solver
    if reference_solver_options is not None:
        kwargs["solver_options"] = dict(reference_solver_options)
    if reference_solver_time_limit_sec is not None:
        kwargs["time_limit_sec"] = float(reference_solver_time_limit_sec)
    if reference_solver_verbose:
        kwargs["verbose"] = True
    return kwargs


def _reference_solver_provenance(result, *, reference_solver):
    extras = result.get("extras") if isinstance(result, dict) else None
    extras = extras if isinstance(extras, dict) else {}
    solver_used = extras.get("solver_used")
    if solver_used is not None:
        solver_used = str(solver_used)
    attempts = extras.get("solver_attempts")
    if attempts is not None:
        attempts = [str(item) for item in attempts]
    fallback_used = extras.get("solver_fallback_used")
    if fallback_used is not None:
        fallback_used = bool(fallback_used)
    return {
        "solver_requested": str(extras.get("solver_requested") or _reference_solver_request_label(reference_solver)),
        "solver_used": solver_used,
        "solver_attempts": attempts,
        "solver_fallback_used": fallback_used,
        "solver_candidates": extras.get("solver_candidates"),
        "cvxpy_status": extras.get("cvxpy_status"),
        "solve_status": result.get("status") if isinstance(result, dict) else None,
    }


def _require_reference_solution(result, *, solve_type, reference_solver):
    """Fail at the reference boundary instead of continuing with ``None``."""

    status = result.get("status") if isinstance(result, dict) else None
    solution = result.get("solution") if isinstance(result, dict) else None
    if solution is not None and status not in {"error", "failed"}:
        return result
    extras = result.get("extras") if isinstance(result, dict) else None
    extras = extras if isinstance(extras, dict) else {}
    requested = extras.get("solver_requested") or _reference_solver_request_label(reference_solver)
    detail = extras.get("error")
    message = f"Convex reference solve {solve_type!r} failed for solver request {requested!r}."
    if detail:
        message = f"{message} Backend error: {detail}"
    raise RuntimeError(message)


def resolve_convex_reference_label(reference_label, solver_provenance):
    """Choose a display label without attributing a fallback solve to MOSEK."""

    solver_used = (solver_provenance or {}).get("solver_used")
    if reference_label is None or not str(reference_label).strip():
        return solver_used or "ConvexSolver"
    label = str(reference_label)
    if solver_used is not None and label.upper() in _CVXPY_BACKEND_LABELS:
        if label.upper() != str(solver_used).upper():
            return str(solver_used)
    return label


def _hash_reference_value(hasher, value):
    if value is None:
        hasher.update(b"<none>")
    elif torch.is_tensor(value):
        _hash_reference_value(hasher, value.detach().cpu().numpy())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(f"array:{array.shape}:{array.dtype}".encode("utf-8"))
        hasher.update(array.tobytes())
    elif isinstance(value, dict):
        hasher.update(b"dict")
        for key in sorted(value):
            hasher.update(str(key).encode("utf-8"))
            _hash_reference_value(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(f"seq:{len(value)}".encode("utf-8"))
        for item in value:
            _hash_reference_value(hasher, item)
    else:
        hasher.update(repr(value).encode("utf-8"))


def _reference_cache_key(
    problem,
    *,
    ip_mode,
    ip_eps,
    hom_origin_constraints,
    hom_origin,
    include_equality,
    reference_solver,
    reference_solver_options,
    reference_solver_time_limit_sec,
    reference_solver_verbose,
):
    hasher = hashlib.sha256()
    hasher.update(f"version:{_CONVEX_REFERENCE_CACHE_VERSION}".encode("utf-8"))
    _hash_reference_value(hasher, getattr(problem, "prob_para", {}))
    _hash_reference_value(
        hasher,
        {
            "ip_mode": str(ip_mode),
            "ip_eps": float(ip_eps),
            "hom_origin_constraints": str(hom_origin_constraints),
            "hom_origin": None if hom_origin is None else np.asarray(hom_origin, dtype=float).reshape(-1),
            "include_equality": bool(include_equality),
            "reference_solver": _reference_solver_request_label(reference_solver),
            "reference_solver_options": reference_solver_options,
            "reference_solver_time_limit_sec": reference_solver_time_limit_sec,
            "reference_solver_verbose": bool(reference_solver_verbose),
        },
    )
    return hasher.hexdigest()


def _reference_cache_path(output_dir):
    root = artifact_root(output_dir)
    return None if root is None else root / _CONVEX_REFERENCE_CACHE_FILENAME


def _load_reference_cache(cache_path, cache_key):
    if cache_path is None or not Path(cache_path).exists():
        return None, 0.0
    start = perf_counter()
    try:
        cache = np.load(cache_path, allow_pickle=True).item()
    except (OSError, ValueError, EOFError, AttributeError) as exc:
        raise RuntimeError(f"Unreadable convex reference cache: {cache_path}") from exc
    if not isinstance(cache, dict) or cache.get("version") != _CONVEX_REFERENCE_CACHE_VERSION:
        raise ValueError(f"Unsupported convex reference cache schema: {cache_path}")
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Invalid convex reference cache entries: {cache_path}")
    entry = entries.get(cache_key)
    return (copy.deepcopy(entry) if entry is not None else None), perf_counter() - start


def _save_reference_cache(cache_path, cache_key, payload):
    if cache_path is None:
        return
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {"version": _CONVEX_REFERENCE_CACHE_VERSION, "entries": {}}
    if cache_path.exists():
        try:
            cache = np.load(cache_path, allow_pickle=True).item()
        except (OSError, ValueError, EOFError, AttributeError) as exc:
            raise RuntimeError(f"Unreadable convex reference cache: {cache_path}") from exc
        if not isinstance(cache, dict) or cache.get("version") != _CONVEX_REFERENCE_CACHE_VERSION:
            raise ValueError(f"Unsupported convex reference cache schema: {cache_path}")
        if not isinstance(cache.get("entries"), dict):
            raise ValueError(f"Invalid convex reference cache entries: {cache_path}")
    cache["entries"][cache_key] = copy.deepcopy(payload)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, cache, allow_pickle=True)
    temporary.replace(cache_path)


def _cache_payload(
    *,
    x_opt,
    objective_opt,
    solver_violation,
    solver_time,
    opt_solver_provenance,
    x_origin,
    ip_solver_time,
    origin_method,
    origin_solver_provenance,
):
    payload = {"origin_method": origin_method}
    if x_opt is not None:
        payload.update(
            {
                "x_opt": np.asarray(x_opt, dtype=float).reshape(-1),
                "objective_opt": objective_opt,
                "solver_violation": solver_violation,
                "solver_time": solver_time,
                "opt_solver_provenance": copy.deepcopy(opt_solver_provenance),
            }
        )
    if x_origin is not None:
        payload.update(
            {
                "x_origin": np.asarray(x_origin, dtype=float).reshape(-1),
                "ip_solver_time": ip_solver_time,
                "origin_solver_provenance": copy.deepcopy(origin_solver_provenance),
            }
        )
    return payload


def prepare_convex_reference_context(
    problem,
    *,
    runtime_device,
    runtime_dtype,
    hom_p_norm=2,
    smooth=False,
    hom_map_explicit_gradient_rule="polynomial",
    hom_map_smooth_tie_tol=1e-7,
    hom_map_smooth_temperature=1e-4,
    need_opt=True,
    ip_mode="ip",
    ip_eps=1e-3,
    hom_origin_constraints="full",
    hom_origin=None,
    reference_solver="auto",
    reference_solver_options=None,
    reference_solver_time_limit_sec=None,
    reference_solver_verbose=False,
    reference_label=None,
    output_dir=None,
    cache_reference=True,
):
    hom_origin_constraints = normalize_hom_origin_constraints(hom_origin_constraints)
    include_equality = hom_origin_constraints == "full"
    reference_solver = normalize_convex_reference_solver(reference_solver)
    reference_solver_options = (
        None if reference_solver_options is None else dict(reference_solver_options)
    )
    reference_solve_kwargs = _reference_solve_kwargs(
        reference_solver=reference_solver,
        reference_solver_options=reference_solver_options,
        reference_solver_time_limit_sec=reference_solver_time_limit_sec,
        reference_solver_verbose=reference_solver_verbose,
    )
    solver = ConvexSolver(problem.prob_para)
    cache_path = _reference_cache_path(output_dir) if cache_reference else None
    cache_key = _reference_cache_key(
        problem,
        ip_mode=ip_mode,
        ip_eps=ip_eps,
        hom_origin_constraints=hom_origin_constraints,
        hom_origin=hom_origin,
        include_equality=include_equality,
        reference_solver=reference_solver,
        reference_solver_options=reference_solver_options,
        reference_solver_time_limit_sec=reference_solver_time_limit_sec,
        reference_solver_verbose=reference_solver_verbose,
    )
    cached, cache_load_time = _load_reference_cache(cache_path, cache_key)
    cached = cached or {}
    cache_dirty = False
    cache_hit = False
    cached_solver_time = cached_ip_solver_time = None

    x_opt = objective_opt = solver_violation = solver_time = None
    opt_solver_provenance = None
    origin_solver_provenance = None
    if need_opt:
        if "x_opt" in cached:
            x_opt = np.asarray(cached["x_opt"], dtype=float).reshape(-1)
            objective_opt = cached.get("objective_opt")
            solver_violation = cached.get("solver_violation")
            opt_solver_provenance = copy.deepcopy(cached.get("opt_solver_provenance"))
            cached_solver_time = cached.get("solver_time")
            solver_time = 0.0
            cache_hit = True
        else:
            result = solve_exact_result(solver, "opt", **reference_solve_kwargs)
            result = _require_reference_solution(
                result,
                solve_type="opt",
                reference_solver=reference_solver,
            )
            x_opt = result["solution"]
            solver_time = result["runtime_total"]
            objective_opt = None if result["objective"] is None else float(result["objective"])
            solver_violation = None if result["violation"] is None else float(result["violation"])
            opt_solver_provenance = _reference_solver_provenance(
                result,
                reference_solver=reference_solver,
            )
            cached_solver_time = solver_time
            cache_dirty = True

    origin_method = str(ip_mode)
    if hom_origin is not None:
        x_origin = np.asarray(hom_origin, dtype=float).reshape(-1)
        ip_solver_time = 0.0
        origin_method = "provided"
    elif "x_origin" in cached:
        x_origin = np.asarray(cached["x_origin"], dtype=float).reshape(-1)
        origin_method = cached.get("origin_method", origin_method)
        origin_solver_provenance = copy.deepcopy(cached.get("origin_solver_provenance"))
        cached_ip_solver_time = cached.get("ip_solver_time")
        ip_solver_time = 0.0
        cache_hit = True
    elif ip_mode == "central_ip":
        result = solve_exact_result(
            solver,
            "central_ip",
            equality=include_equality,
            **reference_solve_kwargs,
        )
        result = _require_reference_solution(
            result,
            solve_type="central_ip",
            reference_solver=reference_solver,
        )
        x_origin, ip_solver_time = result["solution"], result["runtime_total"]
        origin_solver_provenance = _reference_solver_provenance(
            result,
            reference_solver=reference_solver,
        )
        cached_ip_solver_time, cache_dirty = ip_solver_time, True
    elif ip_mode == "geometric_central_ip":
        result = solve_exact_result(
            solver,
            "geometric_central_ip",
            equality=include_equality,
            **reference_solve_kwargs,
        )
        result = _require_reference_solution(
            result,
            solve_type="geometric_central_ip",
            reference_solver=reference_solver,
        )
        x_origin, ip_solver_time = result["solution"], result["runtime_total"]
        origin_solver_provenance = _reference_solver_provenance(
            result,
            reference_solver=reference_solver,
        )
        cached_ip_solver_time, cache_dirty = ip_solver_time, True
    elif str(ip_mode).lower() in {"chebyshev", "chebyshev_approx", "approx_chebyshev"}:
        start = perf_counter()
        x_origin = _approximate_chebyshev_origin(problem, include_equality=include_equality)
        ip_solver_time = perf_counter() - start
        origin_method = "chebyshev_approx"
        cached_ip_solver_time, cache_dirty = ip_solver_time, True
    else:
        result = solve_exact_result(
            solver,
            "ip",
            eps=ip_eps,
            equality=include_equality,
            **reference_solve_kwargs,
        )
        result = _require_reference_solution(
            result,
            solve_type="ip",
            reference_solver=reference_solver,
        )
        x_origin, ip_solver_time = result["solution"], result["runtime_total"]
        origin_solver_provenance = _reference_solver_provenance(
            result,
            reference_solver=reference_solver,
        )
        cached_ip_solver_time, cache_dirty = ip_solver_time, True

    if cache_dirty and cache_path is not None:
        next_payload = dict(cached)
        next_payload.update(
            _cache_payload(
                x_opt=x_opt,
                objective_opt=objective_opt,
                solver_violation=solver_violation,
                solver_time=cached_solver_time if need_opt else cached.get("solver_time"),
                opt_solver_provenance=(
                    opt_solver_provenance if need_opt else cached.get("opt_solver_provenance")
                ),
                x_origin=x_origin if hom_origin is None else cached.get("x_origin"),
                ip_solver_time=cached_ip_solver_time if hom_origin is None else cached.get("ip_solver_time"),
                origin_method=origin_method if hom_origin is None else cached.get("origin_method", origin_method),
                origin_solver_provenance=(
                    origin_solver_provenance
                    if hom_origin is None
                    else cached.get("origin_solver_provenance")
                ),
            )
        )
        _save_reference_cache(cache_path, cache_key, next_payload)

    hom_map = build_convex_hom_map(
        problem,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        x_origin=x_origin,
        smooth=smooth,
        hom_p_norm=hom_p_norm,
        hom_map_explicit_gradient_rule=hom_map_explicit_gradient_rule,
        hom_map_smooth_tie_tol=hom_map_smooth_tie_tol,
        hom_map_smooth_temperature=hom_map_smooth_temperature,
    )
    reference_label_effective = resolve_convex_reference_label(
        reference_label,
        opt_solver_provenance,
    )
    return {
        "x_opt": x_opt,
        "x_origin": x_origin,
        "objective_opt": objective_opt,
        "solver_violation": solver_violation,
        "solver_time": solver_time,
        "ip_solver_time": ip_solver_time,
        "origin_method": origin_method,
        "hom_origin_constraints": hom_origin_constraints,
        "x_origin_eq_violation": _origin_equality_violation(problem, x_origin),
        "hom_map": hom_map,
        "reference_cache_hit": cache_hit,
        "reference_cache_path": str(cache_path) if cache_path is not None and cache_path.exists() else None,
        "reference_cache_key": cache_key,
        "reference_cache_load_time": cache_load_time,
        "cached_solver_time": cached_solver_time,
        "cached_ip_solver_time": cached_ip_solver_time,
        "reference_label": reference_label_effective,
        "reference_label_requested": reference_label,
        "reference_solver_requested": _reference_solver_request_label(reference_solver),
        "reference_solver_used": (opt_solver_provenance or {}).get("solver_used"),
        "reference_solver_attempts": (opt_solver_provenance or {}).get("solver_attempts"),
        "reference_solver_fallback_used": (opt_solver_provenance or {}).get("solver_fallback_used"),
        "reference_solver_provenance": opt_solver_provenance,
        "reference_origin_solver_provenance": origin_solver_provenance,
    }


__all__ = [
    "build_convex_hom_map",
    "build_convex_problem",
    "build_convex_problem_config",
    "normalize_convex_problem_type",
    "normalize_convex_reference_solver",
    "normalize_hom_origin_constraints",
    "prepare_convex_reference_context",
    "resolve_convex_reference_label",
]
