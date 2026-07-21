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


_CONVEX_REFERENCE_CACHE_VERSION = 1
_CONVEX_REFERENCE_CACHE_FILENAME = "convex_reference_context_cache.npy"


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


def _reference_cache_key(problem, *, ip_mode, ip_eps, hom_origin_constraints, hom_origin, include_equality):
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


def _cache_payload(*, x_opt, objective_opt, solver_violation, solver_time, x_origin, ip_solver_time, origin_method):
    payload = {"origin_method": origin_method}
    if x_opt is not None:
        payload.update(
            {
                "x_opt": np.asarray(x_opt, dtype=float).reshape(-1),
                "objective_opt": objective_opt,
                "solver_violation": solver_violation,
                "solver_time": solver_time,
            }
        )
    if x_origin is not None:
        payload.update(
            {"x_origin": np.asarray(x_origin, dtype=float).reshape(-1), "ip_solver_time": ip_solver_time}
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
    output_dir=None,
    cache_reference=True,
):
    hom_origin_constraints = normalize_hom_origin_constraints(hom_origin_constraints)
    include_equality = hom_origin_constraints == "full"
    solver = ConvexSolver(problem.prob_para)
    cache_path = _reference_cache_path(output_dir) if cache_reference else None
    cache_key = _reference_cache_key(
        problem,
        ip_mode=ip_mode,
        ip_eps=ip_eps,
        hom_origin_constraints=hom_origin_constraints,
        hom_origin=hom_origin,
        include_equality=include_equality,
    )
    cached, cache_load_time = _load_reference_cache(cache_path, cache_key)
    cached = cached or {}
    cache_dirty = False
    cache_hit = False
    cached_solver_time = cached_ip_solver_time = None

    x_opt = objective_opt = solver_violation = solver_time = None
    if need_opt:
        if "x_opt" in cached:
            x_opt = np.asarray(cached["x_opt"], dtype=float).reshape(-1)
            objective_opt = cached.get("objective_opt")
            solver_violation = cached.get("solver_violation")
            cached_solver_time = cached.get("solver_time")
            solver_time = 0.0
            cache_hit = True
        else:
            result = solve_exact_result(solver, "opt")
            x_opt = result["solution"]
            solver_time = result["runtime_total"]
            objective_opt = None if result["objective"] is None else float(result["objective"])
            solver_violation = None if result["violation"] is None else float(result["violation"])
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
        cached_ip_solver_time = cached.get("ip_solver_time")
        ip_solver_time = 0.0
        cache_hit = True
    elif ip_mode == "central_ip":
        result = solve_exact_result(solver, "central_ip", equality=include_equality)
        x_origin, ip_solver_time = result["solution"], result["runtime_total"]
        cached_ip_solver_time, cache_dirty = ip_solver_time, True
    elif ip_mode == "geometric_central_ip":
        result = solve_exact_result(solver, "geometric_central_ip", equality=include_equality)
        x_origin, ip_solver_time = result["solution"], result["runtime_total"]
        cached_ip_solver_time, cache_dirty = ip_solver_time, True
    elif str(ip_mode).lower() in {"chebyshev", "chebyshev_approx", "approx_chebyshev"}:
        start = perf_counter()
        x_origin = _approximate_chebyshev_origin(problem, include_equality=include_equality)
        ip_solver_time = perf_counter() - start
        origin_method = "chebyshev_approx"
        cached_ip_solver_time, cache_dirty = ip_solver_time, True
    else:
        result = solve_exact_result(solver, "ip", eps=ip_eps, equality=include_equality)
        x_origin, ip_solver_time = result["solution"], result["runtime_total"]
        cached_ip_solver_time, cache_dirty = ip_solver_time, True

    if cache_dirty and cache_path is not None:
        next_payload = dict(cached)
        next_payload.update(
            _cache_payload(
                x_opt=x_opt,
                objective_opt=objective_opt,
                solver_violation=solver_violation,
                solver_time=cached_solver_time if need_opt else cached.get("solver_time"),
                x_origin=x_origin if hom_origin is None else cached.get("x_origin"),
                ip_solver_time=cached_ip_solver_time if hom_origin is None else cached.get("ip_solver_time"),
                origin_method=origin_method if hom_origin is None else cached.get("origin_method", origin_method),
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
    }


__all__ = [
    "build_convex_hom_map",
    "build_convex_problem",
    "build_convex_problem_config",
    "normalize_convex_problem_type",
    "normalize_hom_origin_constraints",
    "prepare_convex_reference_context",
]
