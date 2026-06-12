"""Shared helpers for optimizer implementations."""

from __future__ import annotations

import inspect
import numpy as np
import torch

from homopt.optim.updates import AdamOptimizer, GDOptimizer, NormalizedGDOptimizer


def _load_solver_classes():
    from homopt.solvers import ConvexSolver, MaxCutSolver

    return ConvexSolver, MaxCutSolver


def _constraint_layout(problem):
    ncon = getattr(problem, "ncon", 0)
    n_eq = getattr(problem, "n_eq", 0)
    ineq_cons = getattr(problem, "ineq_cons", None)
    eq_cons = getattr(problem, "eq_cons", None)
    if ineq_cons is None and eq_cons is not None:
        dual_dim = n_eq if n_eq else ncon
    elif eq_cons is None:
        dual_dim = ncon
    else:
        dual_dim = ncon + n_eq
    return dual_dim, ineq_cons, eq_cons


def _validate_stepsize_rule(value):
    allowed = {"constant", "adaptive", "diminish"}
    if value not in allowed:
        raise ValueError(f"Unsupported stepsize_rule: {value}. Supported: {sorted(allowed)}")
    return value


def _validate_proximal_space(value):
    allowed = {"x", "z"}
    if value not in allowed:
        raise ValueError(f"Unsupported proximal_space: {value}. Supported: {sorted(allowed)}")
    return value


def _validate_inner_solver(value):
    allowed = {"gd", "prox_gd"}
    if value not in allowed:
        raise ValueError(f"Unsupported inner_solver: {value}. Supported: {sorted(allowed)}")
    return value


def _uses_inner_proximal_stages(inner_solver):
    return inner_solver == "prox_gd"


def _quadratic_proximal_point(step_origin, descent_direction, proximal_center, step_size, proximal_coef):
    gamma = float(proximal_coef)
    return (
        step_origin
        - step_size * descent_direction
        + step_size * gamma * proximal_center
    ) / (1.0 + step_size * gamma)


def _validate_acceleration_method(value):
    allowed = {"none", "nag"}
    if value not in allowed:
        raise ValueError(f"Unsupported acceleration_method: {value}. Supported: {sorted(allowed)}")
    return value


def _validate_acceleration_space(value):
    allowed = {"x", "z"}
    if value not in allowed:
        raise ValueError(f"Unsupported acceleration_space: {value}. Supported: {sorted(allowed)}")
    return value


def _resolve_acceleration_config(params, *, default_space):
    method = _validate_acceleration_method(params.get("acceleration_method", "none"))
    space = _validate_acceleration_space(params.get("acceleration_space", default_space))
    return method, space


def _build_algorithm_update_backend(params, acceleration_method):
    if acceleration_method == "nag":
        if params["opt"] != "gd":
            raise ValueError("Use opt='gd' with acceleration_method='nag'.")
        return GDOptimizer(beta1=0.0)
    return _build_update_backend(params["opt"], momentum=params.get("momentum", 0.0))


def _nag_lookahead(point, velocity, beta):
    return point + beta * velocity


def _hom_nag_lookahead(
    z,
    *,
    acceleration_space,
    z_velocity,
    x_velocity,
    beta,
    hom_forward,
    hom_inverse,
    project_z,
):
    if acceleration_space == "z":
        return project_z(_nag_lookahead(z, z_velocity, beta))
    x_mid = _nag_lookahead(hom_forward(z), x_velocity, beta)
    return project_z(hom_inverse(x_mid))


def _validate_positive_int(value, *, name):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _resolve_inner_restart(params):
    return bool(params.get("inner_restart", False))


def _resolve_inner_stopping_config(params, *, inner_iterations):
    rule = str(params.get("inner_stopping_rule", "fixed")).strip().lower()
    if rule not in {"fixed", "adaptive"}:
        raise ValueError("inner_stopping_rule must be 'fixed' or 'adaptive'.")
    min_iterations = int(params.get("inner_iterations_min", 1))
    max_iterations = int(params.get("inner_iterations_max", inner_iterations))
    if min_iterations < 1:
        raise ValueError("inner_iterations_min must be positive.")
    if max_iterations < min_iterations:
        raise ValueError("inner_iterations_max must be >= inner_iterations_min.")
    tol_factor = float(params.get("inner_tol_factor", 0.1))
    tol_min = float(params.get("inner_tol_min", 1e-6))
    if tol_factor < 0:
        raise ValueError("inner_tol_factor must be nonnegative.")
    if tol_min < 0:
        raise ValueError("inner_tol_min must be nonnegative.")
    effective_max_iterations = max_iterations if rule == "adaptive" else inner_iterations
    return {
        "inner_stopping_rule": rule,
        "inner_iterations_min": min(min_iterations, effective_max_iterations),
        "inner_iterations_max": effective_max_iterations,
        "inner_tol_factor": tol_factor,
        "inner_tol_min": tol_min,
    }


def _adaptive_inner_tolerance(current_violation, *, tol_factor, tol_min):
    violation = float(current_violation.detach().max().item())
    return max(float(tol_min), float(tol_factor) * violation)


def _should_stop_inner_loop(error, inner_iter, *, stopping_rule, min_iterations, max_iterations, tolerance):
    if stopping_rule != "adaptive":
        return False
    if inner_iter + 1 >= int(max_iterations):
        return True
    if inner_iter + 1 < int(min_iterations):
        return False
    return float(error.detach().max().item()) <= float(tolerance)


def _velocity_needs_reset(velocity, reference):
    return velocity is None or velocity.shape != reference.shape or velocity.device != reference.device or velocity.dtype != reference.dtype


def _reset_update_backend(update_backend):
    if hasattr(update_backend, "reset"):
        update_backend.reset()


def _resolve_penalty_inner_solver_config(
    params,
    *,
    inner_iterations,
    default_proximal_space,
    inner_iteration_budget=None,
):
    inner_solver = _validate_inner_solver(params.get('inner_solver', 'gd'))
    inner_proximal_update_iterations = _validate_positive_int(
        params.get('inner_proximal_update_iterations', 1),
        name='inner_proximal_update_iterations',
    )
    budget = int(inner_iterations if inner_iteration_budget is None else inner_iteration_budget)
    if _uses_inner_proximal_stages(inner_solver) and inner_proximal_update_iterations > budget:
        raise ValueError(
            f"inner_proximal_update_iterations must be <= active inner iteration budget for {inner_solver}."
        )
    return {
        "inner_solver": inner_solver,
        "uses_inner_proximal_stages": _uses_inner_proximal_stages(inner_solver),
        "inner_proximal_update_iterations": inner_proximal_update_iterations,
        "inner_proximal_steps": max(1, budget // inner_proximal_update_iterations),
        "inner_proximal_coef": params.get('inner_proximal_coef', 0.0),
        "inner_proximal_space": _validate_proximal_space(
            params.get('inner_proximal_space', default_proximal_space)
        ),
    }


def _validate_gradient_method(value, *, name):
    allowed = {"explicit", "autograd"}
    if value not in allowed:
        raise ValueError(f"Unsupported {name}: {value}. Supported: {sorted(allowed)}")
    return value


def _validate_hom_map_gradient(value):
    return _validate_gradient_method(value, name="hom_map_gradient")


def _supports_keyword(callable_obj, keyword):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _build_update_backend(name, *, momentum=0.0):
    if name == "gd":
        return GDOptimizer(beta1=momentum)
    if name == "normalized_gd":
        return NormalizedGDOptimizer()
    if name == "adam":
        return AdamOptimizer()
    raise ValueError(f"Unsupported opt: {name}. Supported: ['adam', 'gd', 'normalized_gd']")


def _resolve_penalty_stepsize_rules(params):
    outer_rule = _validate_stepsize_rule(params.get("outer_stepsize_rule", params.get("stepsize_rule", "adaptive")))
    inner_rule = _validate_stepsize_rule(params.get("inner_stepsize_rule", "constant"))
    return outer_rule, inner_rule


def _resolve_penalty_lr_decays(params):
    fallback = params.get("lr_decay", 0.999)
    return params.get("outer_lr_decay", fallback), params.get("inner_lr_decay", fallback)


def _resolve_penalty_min_lrs(params):
    min_lr = params["min_lr"]
    return min_lr, params.get("inner_min_lr", min_lr)


def _get_max_penalty(params):
    return params.get("max_penalty", 1e2)


def _update_penalty_coefficient(current_penalty, penalty_growth, max_penalty):
    return min(current_penalty * penalty_growth, max_penalty)


def _should_increase_penalty(previous_violation, current_violation, tolerance=0.999):
    previous = float(previous_violation.detach().max().item())
    current = float(current_violation.detach().max().item())
    return current >= previous * float(tolerance)


def _init_penalty_family_controls(owner, params, *, proximal_coef_default):
    owner.use_lagrangian = params.get('use_lagrangian', True)
    owner.use_penalty = params.get('use_penalty', True)
    owner.use_proximal = params.get('use_proximal', False)
    owner.dual_learning_rate = params['dual_learning_rate']
    owner.penalty_coef = params.get('penalty_coef', 1.0) if owner.use_penalty else None
    owner.penalty_growth = params.get('penalty_growth', 1.1)
    owner.max_penalty = _get_max_penalty(params)
    owner.max_dual = params.get('max_dual', 1e2)
    owner.proximal_coef = (
        params.get('proximal_coef', proximal_coef_default)
        if owner.use_proximal
        else None
    )
    owner.return_best_violation = bool(params.get('return_best_violation', False))


def _maybe_update_penalty(owner, previous_violation, current_violation):
    if (
        owner.use_penalty
        and owner.penalty_coef is not None
        and _should_increase_penalty(previous_violation, current_violation)
    ):
        owner.penalty_coef = _update_penalty_coefficient(
            owner.penalty_coef,
            owner.penalty_growth,
            owner.max_penalty,
        )
    return current_violation.detach()


def _clamp_dual_state(dual_state, *, max_dual, eq_cons=None, ineq_cons=None):
    if eq_cons is None and ineq_cons is None:
        return torch.clamp(dual_state, min=-max_dual, max=max_dual)
    if eq_cons is not None:
        dual_state[:, eq_cons] = torch.clamp(dual_state[:, eq_cons], min=-max_dual, max=max_dual)
    if ineq_cons is not None:
        dual_state[:, ineq_cons] = torch.clamp(dual_state[:, ineq_cons], min=0, max=max_dual)
    return dual_state


def _dual_update_scale(owner):
    penalty_coef = owner.penalty_coef if owner.penalty_coef is not None else 1.0
    return owner.dual_learning_rate * penalty_coef


def _update_penalty_inner_learning_rate(
    base_lr,
    current_lr,
    *,
    stepsize_rule,
    lr_decay,
    min_lr,
    current_error,
    best_error,
    iteration,
):
    if stepsize_rule == "adaptive" and current_error > best_error:
        return max(current_lr * lr_decay, min_lr)
    if stepsize_rule == "diminish":
        return max(base_lr / (iteration + 1) ** 0.5, min_lr)
    return current_lr


def _update_penalty_outer_learning_rate(
    current_lr,
    recorder,
    *,
    stepsize_rule,
    lr_decay,
    min_lr,
):
    if stepsize_rule == "adaptive" and recorder.violation_traj[-1].mean() > recorder.violation_traj[-2].mean():
        return max(current_lr * lr_decay, min_lr)
    return current_lr


def _constraint_residual(problem, x, *, equality_only=False):
    if equality_only:
        eq_constraint = getattr(problem, "eq_constraint_x", None)
        if eq_constraint is not None:
            return eq_constraint(x)
    constraint_x = problem.constraint_x
    if _supports_keyword(constraint_x, "eq_cons"):
        return constraint_x(x, clip=False, eq_cons=True)
    return constraint_x(x, clip=False)


def _constraint_violation(problem, x, *, equality_only=False):
    residual = _constraint_residual(problem, x, equality_only=equality_only)
    if residual.numel() == 0:
        return torch.zeros(1, 1, device=x.device, dtype=x.dtype)
    if equality_only:
        return residual.abs().max().view(1, -1)

    _, ineq_cons, eq_cons = _constraint_layout(problem)
    violation_terms = []
    if ineq_cons is not None:
        ineq_idx = list(ineq_cons)
        if ineq_idx:
            violation_terms.append(torch.clamp(residual[:, ineq_idx], min=0))
    elif eq_cons is None:
        violation_terms.append(torch.clamp(residual, min=0))

    if eq_cons is not None:
        eq_idx = list(eq_cons)
        if eq_idx:
            violation_terms.append(residual[:, eq_idx].abs())

    if not violation_terms:
        return torch.zeros(1, 1, device=x.device, dtype=x.dtype)
    return torch.cat(violation_terms, dim=1).max().view(1, -1)


def _constraint_violation_split(problem, x):
    residual = _constraint_residual(problem, x, equality_only=False)
    batch_size = int(x.shape[0]) if x.ndim > 1 else 1
    if residual.ndim == 0:
        residual = residual.view(1, 1)
    elif residual.ndim == 1:
        residual = residual.view(1, -1)
    zero = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
    if residual.numel() == 0:
        return {
            "eq_violation": zero,
            "ineq_violation": zero,
            "full_violation": zero,
        }

    _, ineq_cons, eq_cons = _constraint_layout(problem)
    if ineq_cons is not None:
        ineq_idx = list(ineq_cons)
        if ineq_idx:
            if max(ineq_idx) >= residual.shape[1]:
                raise IndexError(
                    f"Inequality constraint indices {ineq_idx} exceed residual width {residual.shape[1]} "
                    f"for {type(problem).__name__}."
                )
            ineq_res = residual[:, ineq_idx]
            ineq_violation = torch.clamp(ineq_res, min=0).max(dim=1, keepdim=True).values
        else:
            ineq_violation = zero
    elif eq_cons is None:
        ineq_violation = torch.clamp(residual, min=0).max(dim=1, keepdim=True).values
    else:
        ineq_violation = zero

    if eq_cons is not None:
        eq_idx = list(eq_cons)
        if eq_idx:
            if max(eq_idx) < residual.shape[1]:
                eq_res = residual[:, eq_idx]
            else:
                eq_constraint = getattr(problem, "eq_constraint_x", None)
                if eq_constraint is None:
                    raise IndexError(
                        f"Equality constraint indices {eq_idx} exceed residual width {residual.shape[1]} "
                        f"for {type(problem).__name__}, and no eq_constraint_x is available."
                    )
                eq_res = eq_constraint(x)
                if eq_res.ndim == 0:
                    eq_res = eq_res.view(1, 1)
                elif eq_res.ndim == 1:
                    eq_res = eq_res.view(1, -1)
            eq_violation = eq_res.abs().max(dim=1, keepdim=True).values
        else:
            eq_violation = zero
    else:
        eq_violation = zero

    return {
        "eq_violation": eq_violation,
        "ineq_violation": ineq_violation,
        "full_violation": torch.maximum(eq_violation, ineq_violation),
    }


def _constraint_violation_summary(problem, x):
    split = _constraint_violation_split(problem, x)
    eq_violation = float(split["eq_violation"].detach().max().item())
    ineq_violation = float(split["ineq_violation"].detach().max().item())
    return eq_violation, ineq_violation, max(eq_violation, ineq_violation)


def _is_better_candidate(violation, objective, best_violation, best_objective, *, violation_tolerance=1e-12):
    return violation < best_violation or (
        violation <= best_violation + float(violation_tolerance)
        and objective < best_objective
    )


def _project_to_ball(z, p_norm):
    if p_norm == 2:
        norms = torch.norm(z, dim=-1, p=2, keepdim=True)
        safe_norms = torch.clamp(norms, min=1.0)
        return torch.where(norms > 1, z / safe_norms, z)
    if p_norm == np.inf:
        return torch.clamp(z, min=-1, max=1)
    return z


def _pack_run_result(
    final_decision,
    decision_trajectory,
    objective_trajectory,
    violation_trajectory,
    per_iter_time,
    *,
    last_trans_time=0.0,
    latent_trajectory=None,
    extra_metrics=None,
):
    result = {
        'x_traj': decision_trajectory,
        'obj_traj': objective_trajectory.detach().cpu().numpy(),
        'cons_traj': violation_trajectory.detach().cpu().numpy(),
        'iter_time': per_iter_time,
        'last_trans_time': last_trans_time,
        'x_solved': final_decision.detach().cpu().numpy(),
    }
    if latent_trajectory is not None:
        result['z_traj'] = latent_trajectory
    if extra_metrics:
        result.update(extra_metrics)
    return result


def _extract_exact_solution(result):
    solution = result.get("solution")
    if solution is None:
        raise RuntimeError(f"Exact solver did not return a solution: status={result.get('status')}")
    return solution


def _cached_exact_solver(owner, solver_kind):
    cache = getattr(owner, "_exact_solver_cache", None)
    if cache is None:
        cache = {}
        owner._exact_solver_cache = cache
    if solver_kind not in cache:
        ConvexSolver, MaxCutSolver = _load_solver_classes()
        if solver_kind == "convex":
            cache[solver_kind] = ConvexSolver(owner.problem.prob_para)
        elif solver_kind == "maxcut":
            cache[solver_kind] = MaxCutSolver(owner.problem.prob_para)
        else:
            raise ValueError(f"Unsupported exact solver kind: {solver_kind}")
    return cache[solver_kind]


def _as_numpy_vector(x):
    return x.detach().view(-1).cpu().numpy()


def _infer_problem_dtype(problem):
    for attr_name in ("Q", "p", "A_eq", "A", "L", "U", "weights"):
        value = getattr(problem, attr_name, None)
        if isinstance(value, torch.Tensor):
            return value.dtype
    return torch.float32


def _problem_tensor_kwargs(problem, *, device=None):
    return {
        "device": getattr(problem, "device", device),
        "dtype": _infer_problem_dtype(problem),
    }


def _randn_problem_row(problem, dim, *, device=None):
    return torch.randn(1, dim, **_problem_tensor_kwargs(problem, device=device))


def _as_problem_row(problem, value, *, device=None):
    return torch.as_tensor(value, **_problem_tensor_kwargs(problem, device=device)).view(1, -1)


def _as_torch_row(value, *, device, dtype):
    return torch.as_tensor(value, dtype=dtype, device=device).view(1, -1)


def _resolve_first_order_lagrangian_gap_config(params, *, default_enabled):
    enabled = bool(params.get("check_first_order_lagrangian_gap", default_enabled))
    threshold = float(
        params.get(
            "first_order_lagrangian_gap_threshold",
            params.get("outer_first_order_gap_threshold", params["convergence_threshold"]),
        )
    )
    return enabled, threshold


def _sequence_as_metric(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return list(value)


def _last_sequence_scalar(value):
    if value is None or len(value) == 0:
        return None
    if torch.is_tensor(value):
        return float(value.reshape(-1)[-1].detach().cpu().item())
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return None
    return float(array[-1])


def _store_first_order_lagrangian_gap_metrics(owner, payload):
    trajectory = payload.get("first_order_lagrangian_gap")
    owner.last_first_order_lagrangian_gap_traj = trajectory
    owner.last_final_first_order_lagrangian_gap = _last_sequence_scalar(trajectory)


def _store_constraint_violation_split_metrics(owner, payload):
    owner.last_eq_violation_traj = payload.get("eq_violation")
    owner.last_ineq_violation_traj = payload.get("ineq_violation")
    owner.last_full_violation_traj = payload.get("full_violation")


_CONSTRAINT_SPLIT_TRACK_NAMES = ("eq_violation", "ineq_violation", "full_violation")
_EQ_CVXPY_ALGORITHMS = ('Penalty-EQ', 'Prox-Penalty-EQ', 'ALM-EQ', 'Prox-ALM-EQ')
_OPTIMIZER_SEQUENCE_METRICS = (
    ("last_outer_iter_time", "outer_iter_time"),
    ("last_inner_iter_time", "inner_iter_time"),
    ("last_solver_iter_time", "solver_iter_time"),
    ("last_first_order_lagrangian_gap_traj", "first_order_lagrangian_gap_traj"),
    ("last_eq_violation_traj", "eq_violation_traj"),
    ("last_ineq_violation_traj", "ineq_violation_traj"),
    ("last_full_violation_traj", "full_violation_traj"),
)
_OPTIMIZER_SCALAR_METRICS = (
    ("last_initial_transform_time", "initial_transform_time"),
    ("last_final_transform_time", "final_transform_time"),
    ("last_final_first_order_lagrangian_gap", "final_first_order_lagrangian_gap"),
)
_OPTIMIZER_ROUTE_ATTRS = (
    "lagrangian_gradient",
    "hom_map_gradient",
)


def _collect_optimizer_timing_metrics(optimizer):
    metrics = {}
    for attr_name, metric_name in _OPTIMIZER_SEQUENCE_METRICS:
        value = getattr(optimizer, attr_name, None)
        if value is not None:
            metrics[metric_name] = _sequence_as_metric(value)
    for attr_name, metric_name in _OPTIMIZER_SCALAR_METRICS:
        value = getattr(optimizer, attr_name, None)
        if value is not None:
            metrics[metric_name] = float(value)
    for attr_name in _OPTIMIZER_ROUTE_ATTRS:
        value = getattr(optimizer, attr_name, None)
        if value is not None:
            metrics[attr_name] = value
    return metrics


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
