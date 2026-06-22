"""Configuration parsing and validation helpers for optimizers."""

from __future__ import annotations

import inspect

from .updates import AdamOptimizer, GDOptimizer, NormalizedGDOptimizer


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


def _build_update_backend(name, *, momentum=0.0):
    if name == "gd":
        return GDOptimizer(beta1=momentum)
    if name == "normalized_gd":
        return NormalizedGDOptimizer()
    if name == "adam":
        return AdamOptimizer()
    raise ValueError(f"Unsupported opt: {name}. Supported: ['adam', 'gd', 'normalized_gd']")


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


def _resolve_first_order_lagrangian_gap_config(params, *, default_enabled):
    enabled = bool(params.get("check_first_order_lagrangian_gap", default_enabled))
    threshold = float(
        params.get(
            "first_order_lagrangian_gap_threshold",
            params.get("outer_first_order_gap_threshold", params["convergence_threshold"]),
        )
    )
    return enabled, threshold


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
