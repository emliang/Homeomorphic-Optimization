"""Core optimizer implementations."""

import copy
import inspect
import numpy as np
import torch
import time
from tqdm import tqdm
from homopt.optim.base import BaseOptimizer
from homopt.optim._defaults import build_first_order_subproblem_defaults
from homopt.optim._first_order import run_first_order_loop
from homopt.optim._penalty import run_penalty_outer_loop
from homopt.optim._recording import IterationRecorder
from homopt.optim._updates import AdamOptimizer, GDOptimizer, NormalizedGDOptimizer
from homopt.problems import ConvexOpt, LinearProblem, MaxCutSDP, PolyStarOpt, ProjProblem, ToyStarOpt
from homopt.solvers import solve_exact_result


def _load_solver_classes():
    from homopt.solvers.core import ConvexSolver, MaxCutSolver

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
    if params["opt"] == "nesterov":
        raise ValueError("Use acceleration_method='nag' with opt='gd' instead of opt='nesterov'.")
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


def _default_first_order_subproblem_params(params, *, outer_iterations_key, inner_iterations_key):
    return build_first_order_subproblem_defaults(
        params,
        outer_iterations=params[outer_iterations_key],
        inner_iterations=params[inner_iterations_key],
    )


def _resolve_penalty_stepsize_rules(params):
    outer_rule = _validate_stepsize_rule(params.get("outer_stepsize_rule", params.get("stepsize_rule", "adaptive")))
    inner_rule = _validate_stepsize_rule(params.get("inner_stepsize_rule", "constant"))
    return outer_rule, inner_rule


def _resolve_penalty_lr_decays(params):
    fallback = params.get("lr_decay", 0.999)
    return params.get("outer_lr_decay", fallback), params.get("inner_lr_decay", fallback)


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
    convergence_threshold,
    current_error,
    best_error,
    iteration,
):
    if stepsize_rule == "adaptive" and current_error > best_error:
        return max(current_lr * lr_decay, convergence_threshold)
    if stepsize_rule == "diminish":
        return max(base_lr / (iteration + 1) ** 0.5, convergence_threshold)
    return current_lr


def _update_penalty_outer_learning_rate(
    current_lr,
    recorder,
    *,
    stepsize_rule,
    lr_decay,
    convergence_threshold,
):
    if stepsize_rule == "adaptive" and recorder.violation_traj[-1].mean() > recorder.violation_traj[-2].mean():
        return max(current_lr * lr_decay, convergence_threshold)
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


def _algorithm_specs():
    def _params_for_algorithm(params, name, *, force_proximal=False):
        if name not in params:
            raise KeyError(f"Missing optimizer parameters for algorithm: {name}")
        algorithm_params = dict(params[name])
        algorithm_params["opt_type"] = name
        if force_proximal:
            algorithm_params["use_proximal"] = True
            algorithm_params.setdefault("proximal_coef", params.get("common", {}).get("proximal_coef", 0.1))
        return algorithm_params

    specs = {
        'PGD': (lambda problem, params, hom_map: PGDOptimizer(problem, params['PGD']), False),
        'Penalty': (
            lambda problem, params, hom_map: LagrangianOptimizer(
                problem,
                _params_for_algorithm(params, "Penalty"),
            ),
            False,
        ),
        'Prox-Penalty': (
            lambda problem, params, hom_map: LagrangianOptimizer(
                problem,
                _params_for_algorithm(params, "Prox-Penalty", force_proximal=True),
            ),
            False,
        ),
        'ALM': (lambda problem, params, hom_map: LagrangianOptimizer(problem, params['ALM']), False),
        'Prox-ALM': (
            lambda problem, params, hom_map: LagrangianOptimizer(
                problem,
                _params_for_algorithm(params, "Prox-ALM", force_proximal=True),
            ),
            False,
        ),
        'Hom-PGD': (lambda problem, params, hom_map: HomPGDOptimizer(problem, params['Hom-PGD'], hom_map=hom_map), True),
        'Hom-ALM': (lambda problem, params, hom_map: HomALMOptimizer(problem, params['Hom-ALM'], hom_map=hom_map), True),
        'Prox-Hom-ALM': (
            lambda problem, params, hom_map: HomALMOptimizer(
                problem,
                _params_for_algorithm(params, "Prox-Hom-ALM", force_proximal=True),
                hom_map=hom_map,
            ),
            True,
        ),
        'RD': (lambda problem, params, hom_map: RadialDualOptimizer(problem, params['RD'], hom_map=hom_map), True),
        'FW': (lambda problem, params, hom_map: FrankWolfeOptimizer(problem, params['FW']), False),
    }
    specs.update({
        name: (
            lambda problem, params, hom_map, algorithm=name: EqualityConstrainedALMOptimizer(
                problem,
                params[algorithm],
            ),
            False,
        )
        for name in _EQ_CVXPY_ALGORITHMS
    })
    return specs


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


def run_algorithm(name, problem, params, hom_map=None, init_point=None):
    """Helper function to run a specific algorithm and collect results"""
    print(f'Running {name}')
    specs = _algorithm_specs()
    if name not in specs:
        raise ValueError(f"Unknown algorithm: {name}")
    seed = params['common']['seed']
    builder, returns_transform_time = specs[name]
    optimizer = builder(problem, params, hom_map)
    if hom_map is not None and hasattr(hom_map, "reset_forward_stats"):
        hom_map.reset_forward_stats()
    run_start_time = time.perf_counter()
    result = optimizer.optimize(init_point, verbose=bool(params["common"].get("verbose", False)), seed=seed)
    total_wall_time = time.perf_counter() - run_start_time
    extra_metrics = {"total_wall_time": total_wall_time}
    if name in {"Hom-ALM", "Prox-Hom-ALM"}:
        extra_metrics["violation_scope"] = "equality"
    extra_metrics.update(_collect_optimizer_timing_metrics(optimizer))
    if hom_map is not None and hasattr(hom_map, "last_forward_mode"):
        extra_metrics.update(
            {
                "hom_map_forward_mode": hom_map.last_forward_mode,
                "hom_map_forward_mode_counts": dict(getattr(hom_map, "forward_mode_counts", {})),
                "hom_map_smooth": bool(getattr(hom_map, "smooth", False)),
                "hom_map_smooth_tie_tol": float(getattr(hom_map, "smooth_tie_tol", 0.0)),
                "hom_map_smooth_temperature": float(getattr(hom_map, "smooth_temperature", 0.0)),
            }
        )
    if returns_transform_time:
        latent_trajectory = None
        if len(result) == 7:
            final_decision, decision_trajectory, latent_trajectory, objective_trajectory, violation_trajectory, per_iter_time, last_trans_time = result
        else:
            final_decision, decision_trajectory, objective_trajectory, violation_trajectory, per_iter_time, last_trans_time = result
            latent_trajectory = getattr(optimizer, "last_z_trajectory", None)
        return _pack_run_result(
            final_decision,
            decision_trajectory,
            objective_trajectory,
            violation_trajectory,
            per_iter_time,
            last_trans_time=last_trans_time,
            latent_trajectory=latent_trajectory,
            extra_metrics=extra_metrics,
        )
    final_decision, decision_trajectory, objective_trajectory, violation_trajectory, per_iter_time = result
    return _pack_run_result(
        final_decision,
        decision_trajectory,
        objective_trajectory,
        violation_trajectory,
        per_iter_time,
        extra_metrics=extra_metrics,
    )

class PGDOptimizer(BaseOptimizer):
    """
    Implementation of Projected Gradient Descent optimizer.
    """
    def __init__(self, problem, params, hom_map=None):
        super().__init__(problem=problem, params=params, hom_map=hom_map)
        self.problem = problem
        self._init_optimization_params(params)
        self.hom_map = hom_map
        self.device = self.problem.device

    def _init_optimization_params(self, params):
        """Initialize all optimization-related parameters"""
        # Base parameters
        self.learning_rate = params['learning_rate']
        self.max_iterations = params['max_iterations']
        self.max_running_time = params['max_running_time']
        self.convergence_threshold = params['convergence_threshold']
        self.acceleration_method, self.acceleration_space = _resolve_acceleration_config(params, default_space="x")
        self.acceleration_beta = float(params.get('momentum', 0.0))
        self.stepsize_rule = _validate_stepsize_rule(params['stepsize_rule'])
        self.opt = _build_algorithm_update_backend(params, self.acceleration_method)
        self.lr_decay = params['lr_decay']
        self.opt_type = params['opt_type']
        self.convergence_metric = params.get('convergence_metric')
        self.objective_change_threshold = params.get('objective_change_threshold', self.convergence_threshold)
        self.min_iterations = int(params.get('min_iterations', 1))
        if self.opt_type == 'PGD' and self.convergence_metric is None:
            self.convergence_metric = 'objective_change'
        if params['opt_type'] == 'PGD':
            if 'projection_subproblem' in params:
                self.projection_subproblem_params = copy.deepcopy(params['projection_subproblem'])
            else:
                self.projection_subproblem_params = _default_first_order_subproblem_params(
                    params,
                    outer_iterations_key='projection_outer_iterations',
                    inner_iterations_key='projection_inner_iterations',
                )

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        should_track_traj = self.problem.nvar == 2
        prob_dim = self.problem.nvar

        if initial_point is None:
            if seed is not None:
                torch.manual_seed(seed)
            x = _randn_problem_row(self.problem, prob_dim, device=self.device)
        else:
            x = _as_problem_row(self.problem, initial_point, device=self.device)
        initial_split = _constraint_violation_split(self.problem, x)
        initial_eval = {
            "objective": self.problem.objective_x(x),
            "violation": initial_split["full_violation"],
            "decision": x,
            **initial_split,
        }

        self._nag_velocity = torch.zeros_like(x)

        def _compute_gradient(point, prev_point, iteration):
            del prev_point, iteration
            if self.acceleration_method == "nag":
                step_point = _nag_lookahead(point, self._nag_velocity, self.acceleration_beta)
            else:
                step_point = point
            self._last_step_origin = step_point
            return self.problem.gradient_objective_x(step_point)

        def _apply_step(point, update, lr, prev_point, iteration):
            del prev_point
            step_origin = getattr(self, "_last_step_origin", point)
            point_update = step_origin - lr * update
            if self.opt_type in ['PGD']:
                next_point = self.project_to_set(point_update)
            elif self.opt_type in ['HP-GD']:
                next_point = self.hom_projection_to_set(point_update)
            else:
                next_point = point_update
            if self.acceleration_method == "nag":
                self._nag_velocity = (next_point - point).detach()
            return next_point

        def _evaluate_state(point):
            split = _constraint_violation_split(self.problem, point)
            return {
                "objective": self.problem.objective_x(point),
                "violation": split["full_violation"],
                "decision": point,
                **split,
            }

        point_proj, payload, self.learning_rate = run_first_order_loop(
            initial_state=x,
            initial_eval=initial_eval,
            update_backend=self.opt,
            compute_gradient=_compute_gradient,
            apply_step=_apply_step,
            evaluate_state=_evaluate_state,
            max_iterations=self.max_iterations,
            max_running_time=self.max_running_time,
            initial_learning_rate=self.learning_rate,
            stepsize_rule=self.stepsize_rule,
            lr_decay=self.lr_decay,
            convergence_threshold=self.convergence_threshold,
            verbose=verbose,
            progress_disable=False,
            track_decisions=should_track_traj,
            extra_track_names=_CONSTRAINT_SPLIT_TRACK_NAMES,
            convergence_metric=self.convergence_metric,
            objective_change_threshold=self.objective_change_threshold,
            min_iterations=self.min_iterations,
            require_feasible_for_objective_change=True,
        )
        _store_constraint_violation_split_metrics(self, payload)
        return (
            point_proj,
            payload["decision_trajectory"],
            payload["objective_traj"],
            payload["violation_traj"],
            payload["per_iter_time"],
        )

    def project_to_ball(self, z):
        """Project points onto the unit ball."""
        return _project_to_ball(z, self.hom_map.p_norm)

    def project_to_set(self, x_init):
        """Project point onto general set using augmented Lagrangian method."""
        if _constraint_violation(self.problem, x_init, equality_only=False).max() > self.convergence_threshold:
            if isinstance(self.problem, ConvexOpt):
                solver = _cached_exact_solver(self, "convex")
                x_proj = _extract_exact_solution(
                    solve_exact_result(solver, 'proj', _as_numpy_vector(x_init))
                )
            elif isinstance(self.problem, MaxCutSDP):
                solver = _cached_exact_solver(self, "maxcut")
                x_proj = _extract_exact_solution(
                    solve_exact_result(solver, 'proj', _as_numpy_vector(x_init))
                )
            elif isinstance(self.problem, (ToyStarOpt, PolyStarOpt)):
                proj_problem = ProjProblem(x_init, self.problem).to_device(x_init.device)
                lag_optimizer = LagrangianOptimizer(proj_problem, self.projection_subproblem_params)
                # Use Lagrangian optimizer to solve projection
                x_proj, _, _, _, _ = lag_optimizer.optimize(initial_point=x_init, verbose=False)
            else:
                raise ValueError(f"Projection is not implemented for {type(self.problem).__name__}")
            return _as_torch_row(x_proj, device=self.device, dtype=x_init.dtype)
        else:
            return x_init

    def hom_projection_to_set(self, x):
        """Homomorphic projection onto star-shaped set."""
        if self.problem.constraint_x(x).max() > 0:
            x = self.hom_map.forward(self.project_to_ball(self.hom_map.inverse(x)))
        return x

class HomPGDOptimizer(BaseOptimizer):
    """
    Implementation of Projected Gradient Descent optimizer.
    """
    def __init__(self, problem, params, hom_map=None):
        super().__init__(problem=problem, params=params, hom_map=hom_map)
        self.problem = problem
        self._init_optimization_params(params)
        self.hom_map = hom_map
        self.device = self.problem.device

    def _init_optimization_params(self, params):
        """Initialize all optimization-related parameters"""
        # Base parameters
        self.learning_rate = params['learning_rate']
        self.max_iterations = params['max_iterations']
        self.max_running_time = params['max_running_time']
        self.convergence_threshold = params['convergence_threshold']
        self.acceleration_method, self.acceleration_space = _resolve_acceleration_config(params, default_space="z")
        self.acceleration_beta = float(params.get('momentum', 0.0))
        self.stepsize_rule = _validate_stepsize_rule(params['stepsize_rule'])
        self.lr_decay = params['lr_decay']
        self.opt = _build_algorithm_update_backend(params, self.acceleration_method)
        self.hom_map_gradient = _validate_hom_map_gradient(params.get('hom_map_gradient', 'explicit'))
        self.use_proximal = bool(params.get('use_proximal', False))
        self.proximal_update_iterations = _validate_positive_int(
            params.get('proximal_update_iterations', 1),
            name='proximal_update_iterations',
        )
        if self.use_proximal and self.proximal_update_iterations > self.max_iterations:
            raise ValueError("proximal_update_iterations must be <= max_iterations for Hom-PGD proximal mode.")
        if self.use_proximal and params.get("opt", "gd") != "gd":
            raise ValueError("Hom-PGD proximal mode requires opt='gd'.")
        self.proximal_steps = max(1, self.max_iterations // self.proximal_update_iterations)
        self.proximal_coef = float(params.get('proximal_coef', 0.0)) if self.use_proximal else 0.0
        self.proximal_space = _validate_proximal_space(params.get('proximal_space', 'z'))
        self.convergence_metric = params.get('convergence_metric', 'objective_change')
        self.objective_change_threshold = params.get('objective_change_threshold', self.convergence_threshold)
        self.min_iterations = int(params.get('min_iterations', 1))
        self.opt_type = params['opt_type']

    def _hom_forward(self, z):
        if _supports_keyword(self.hom_map.forward, "method"):
            return self.hom_map.forward(z, method=self.hom_map_gradient)
        return self.hom_map.forward(z)

    def _proximal_gradient_z(self, z, z_center, x_center):
        if self.proximal_coef == 0:
            return torch.zeros_like(z)
        if self.proximal_space == "z":
            return self.proximal_coef * (z - z_center)
        if not hasattr(self.hom_map, "vjp"):
            raise NotImplementedError("Hom-PGD x-space proximal gradient requires hom_map.vjp.")
        x_current = self._hom_forward(z)
        grad_x = self.proximal_coef * (x_current - x_center)
        return self.hom_map.vjp(z, grad_x, method=self.hom_map_gradient).detach()

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        should_track_traj = self.problem.nvar == 2
        objective_x = self.problem.objective_x
        constraint_x = self.problem.constraint_x
        hom_forward = self._hom_forward
        hom_inverse = self.hom_map.inverse

        prob_dim = self.problem.nvar

        if initial_point is None:
            if seed is not None:
                torch.manual_seed(seed)
            x = _randn_problem_row(self.problem, prob_dim, device=self.device)
        else:
            x = _as_problem_row(self.problem, initial_point, device=self.device)
        init_transform_start = time.perf_counter()
        point = hom_inverse(x)
        initial_split = _constraint_violation_split(self.problem, x)

        initial_eval = {
            "objective": objective_x(x),
            "violation": initial_split["full_violation"],
            "decision": x,
            "z_trajectory": point,
            **initial_split,
        }
        self._nag_velocity_z = torch.zeros_like(point)
        self._nag_velocity_x = torch.zeros_like(x)
        self._hom_pgd_prox_z_center = point.clone().detach()
        self._hom_pgd_prox_x_center = (
            hom_forward(self._hom_pgd_prox_z_center).detach()
            if self.proximal_space == "x"
            else None
        )
        self.last_initial_transform_time = time.perf_counter() - init_transform_start

        def _accelerated_point(point, prev_point, iteration):
            del prev_point, iteration
            if self.acceleration_method != "nag":
                return point
            return _hom_nag_lookahead(
                point,
                acceleration_space=self.acceleration_space,
                z_velocity=self._nag_velocity_z,
                x_velocity=self._nag_velocity_x,
                beta=self.acceleration_beta,
                hom_forward=hom_forward,
                hom_inverse=hom_inverse,
                project_z=self.project_to_ball,
            )

        def _compute_gradient(point, prev_point, iteration):
            mid_point = _accelerated_point(point, prev_point, iteration)
            self._last_step_origin = mid_point
            kwargs = {"hom_map_method": self.hom_map_gradient}
            if _supports_keyword(self.problem.gradient_objective_z, "method"):
                kwargs["method"] = self.hom_map_gradient
            grad = self.problem.gradient_objective_z(mid_point, self.hom_map, **kwargs)
            if self.use_proximal and self.proximal_space == "x":
                grad = grad + self._proximal_gradient_z(
                    mid_point,
                    self._hom_pgd_prox_z_center,
                    self._hom_pgd_prox_x_center,
                )
            return grad

        def _apply_step(point, update, lr, prev_point, iteration):
            del prev_point
            step_origin = getattr(self, "_last_step_origin", point)
            if self.use_proximal and self.proximal_space == "z" and self.proximal_coef != 0:
                step_candidate = _quadratic_proximal_point(
                    step_origin,
                    update,
                    self._hom_pgd_prox_z_center,
                    lr,
                    self.proximal_coef,
                )
            else:
                step_candidate = step_origin - lr * update
            next_point = self.project_to_ball(step_candidate)
            if self.acceleration_method == "nag":
                if self.acceleration_space == "z":
                    self._nag_velocity_z = (next_point - point).detach()
                else:
                    self._nag_velocity_x = (hom_forward(next_point) - hom_forward(point)).detach()
            if self.use_proximal and (iteration + 1) % self.proximal_steps == 0:
                self._hom_pgd_prox_z_center = next_point.clone().detach()
                self._hom_pgd_prox_x_center = (
                    hom_forward(self._hom_pgd_prox_z_center).detach()
                    if self.proximal_space == "x"
                    else None
                )
                self._nag_velocity_z = torch.zeros_like(next_point)
                self._nag_velocity_x = torch.zeros_like(hom_forward(next_point))
            return next_point

        def _evaluate_state(point):
            with torch.no_grad():
                x_eval = hom_forward(point)
                split = _constraint_violation_split(self.problem, x_eval)
                return {
                    "objective": objective_x(x_eval),
                    "violation": split["full_violation"],
                    "decision": x_eval,
                    "z_trajectory": point,
                    **split,
                }

        point_proj, payload, self.learning_rate = run_first_order_loop(
            initial_state=point,
            initial_eval=initial_eval,
            update_backend=self.opt,
            compute_gradient=_compute_gradient,
            apply_step=_apply_step,
            evaluate_state=_evaluate_state,
            max_iterations=self.max_iterations,
            max_running_time=self.max_running_time,
            initial_learning_rate=self.learning_rate,
            stepsize_rule=self.stepsize_rule,
            lr_decay=self.lr_decay,
            convergence_threshold=self.convergence_threshold,
            verbose=verbose,
            progress_disable=not verbose,
            track_decisions=should_track_traj,
            extra_track_names=_CONSTRAINT_SPLIT_TRACK_NAMES
            + (("z_trajectory",) if should_track_traj else ()),
            convergence_metric=self.convergence_metric,
            objective_change_threshold=self.objective_change_threshold,
            min_iterations=self.min_iterations,
            require_feasible_for_objective_change=True,
        )

        st = time.perf_counter()
        x_opt = hom_forward(point_proj)
        et = time.perf_counter()
        last_trans_time = et - st
        self.last_final_transform_time = last_trans_time
        self.last_z_trajectory = payload.get("z_trajectory")
        _store_constraint_violation_split_metrics(self, payload)
        return (
            x_opt,
            payload["decision_trajectory"],
            payload["objective_traj"],
            payload["violation_traj"],
            payload["per_iter_time"],
            last_trans_time,
        )

    def project_to_ball(self, z):
        """Project points onto the unit ball."""
        return _project_to_ball(z, self.hom_map.p_norm)

class HomALMOptimizer(BaseOptimizer):
    """Implements Homomorphic Augmented Lagrangian Optimization."""
    def __init__(self, problem, params, hom_map=None):
        super().__init__(problem=problem, params=params, hom_map=hom_map)
        self.problem = problem
        self._init_optimization_params(params)
        self.hom_map = hom_map
        self.device = self.problem.device

    def _init_optimization_params(self, params):
        """Initialize all optimization-related parameters"""
        # Base parameters
        self.max_running_time = params['max_running_time']
        self.learning_rate = params['learning_rate']
        self.inner_learning_rate = params.get('inner_learning_rate', None)
        self.outer_iterations = params['outer_iterations']
        self.inner_iterations = params['inner_iterations']
        self.outer_lr_decay, self.inner_lr_decay = _resolve_penalty_lr_decays(params)
        self.convergence_threshold = params['convergence_threshold']
        (
            self.check_first_order_lagrangian_gap,
            self.first_order_lagrangian_gap_threshold,
        ) = _resolve_first_order_lagrangian_gap_config(
            params,
            default_enabled=True,
        )
        self.outer_stepsize_rule, self.inner_stepsize_rule = _resolve_penalty_stepsize_rules(params)
        if getattr(self.problem, "eq_cons", None) is not None:
            dual_dim = int(getattr(self.problem, "n_eq", 0))
            self.dual_ineq_cons = None
            self.dual_eq_cons = range(dual_dim)
        else:
            dual_dim, self.dual_ineq_cons, self.dual_eq_cons = _constraint_layout(self.problem)
        self.dual_var = torch.zeros(1, dual_dim, **_problem_tensor_kwargs(self.problem))
        _init_penalty_family_controls(self, params, proximal_coef_default=1.0)
        self.proximal_space = _validate_proximal_space(params.get('proximal_space', 'z'))
        inner_stopping_config = _resolve_inner_stopping_config(params, inner_iterations=self.inner_iterations)
        self.inner_stopping_rule = inner_stopping_config["inner_stopping_rule"]
        self.inner_iterations_min = inner_stopping_config["inner_iterations_min"]
        self.inner_iterations_max = inner_stopping_config["inner_iterations_max"]
        self.inner_tol_factor = inner_stopping_config["inner_tol_factor"]
        self.inner_tol_min = inner_stopping_config["inner_tol_min"]
        active_inner_budget = self.inner_iterations_max if self.inner_stopping_rule == "adaptive" else self.inner_iterations
        inner_solver_config = _resolve_penalty_inner_solver_config(
            params,
            inner_iterations=self.inner_iterations,
            inner_iteration_budget=active_inner_budget,
            default_proximal_space="z",
        )
        self.inner_solver = inner_solver_config["inner_solver"]
        self.uses_inner_proximal_stages = inner_solver_config["uses_inner_proximal_stages"]
        self.inner_proximal_update_iterations = inner_solver_config["inner_proximal_update_iterations"]
        self.inner_proximal_steps = inner_solver_config["inner_proximal_steps"]
        self.inner_proximal_coef = inner_solver_config["inner_proximal_coef"]
        self.inner_proximal_space = inner_solver_config["inner_proximal_space"]
        self.inner_restart = _resolve_inner_restart(params)
        self._nag_velocity_z = None
        self._nag_velocity_x = None
        self.acceleration_method, self.acceleration_space = _resolve_acceleration_config(params, default_space="z")
        self.acceleration_beta = float(params.get('momentum', 0.0))
        self.hom_map_gradient = _validate_hom_map_gradient(params.get('hom_map_gradient', 'explicit'))
        self.opt = _build_algorithm_update_backend(params, self.acceleration_method)
        self._problem_grad_z = self.problem.gradient_lagrangian_z
        self._problem_grad_z_supports_method = _supports_keyword(self._problem_grad_z, "method")
        self._problem_grad_z_supports_x = _supports_keyword(self._problem_grad_z, "x")
        self._problem_grad_z_supports_hom_state = _supports_keyword(self._problem_grad_z, "hom_state")
        self._problem_grad_z_supports_outer_cache = _supports_keyword(self._problem_grad_z, "outer_cache")
        self._hom_forward_fn = self.hom_map.forward
        self._hom_forward_supports_method = _supports_keyword(self._hom_forward_fn, "method")
        self._hom_forward_supports_return_state = _supports_keyword(self._hom_forward_fn, "return_state")
        self._eq_constraint_x = getattr(self.problem, "eq_constraint_x", None)
        self._build_explicit_grad_cache = getattr(self.problem, "build_explicit_lagrangian_z_outer_cache", None)

    def gradient_lagrangian(self, z, dual_var, z_outer=None, x_outer=None, x=None, hom_state=None, outer_cache=None):
        """Gradient of augmented objective function."""
        kwargs = {
            "penalty_coef": self.penalty_coef,
            "proximal_coef": self.proximal_coef,
            "hom_map": self.hom_map,
            "z_outer": z_outer,
            "x_outer": x_outer,
            "proximal_space": self.proximal_space,
            "hom_map_method": self.hom_map_gradient,
        }
        if self._problem_grad_z_supports_method:
            kwargs["method"] = self.hom_map_gradient
        if self._problem_grad_z_supports_x:
            kwargs["x"] = x
        if self._problem_grad_z_supports_hom_state:
            kwargs["hom_state"] = hom_state
        if self._problem_grad_z_supports_outer_cache:
            kwargs["outer_cache"] = outer_cache
        return self._problem_grad_z(z, dual_var, **kwargs)

    def _original_lagrangian_gradient_z(self, z, dual_var, *, x=None, hom_state=None):
        outer_cache = None
        if self.hom_map_gradient == "explicit" and self._build_explicit_grad_cache is not None:
            outer_cache = self._build_explicit_grad_cache(
                dual_var,
                penalty_coef=None,
                proximal_coef=None,
                x_outer=None,
                proximal_space=self.proximal_space,
            )
        kwargs = {
            "penalty_coef": None,
            "proximal_coef": None,
            "hom_map": self.hom_map,
            "z_outer": None,
            "x_outer": None,
            "proximal_space": self.proximal_space,
            "hom_map_method": self.hom_map_gradient,
        }
        if self._problem_grad_z_supports_method:
            kwargs["method"] = self.hom_map_gradient
        if self._problem_grad_z_supports_x:
            kwargs["x"] = x
        if self._problem_grad_z_supports_hom_state:
            kwargs["hom_state"] = hom_state
        if self._problem_grad_z_supports_outer_cache:
            kwargs["outer_cache"] = outer_cache
        return self._problem_grad_z(z, dual_var, **kwargs)

    def _latent_ball_dual_norm(self, grad):
        p_norm = getattr(self.hom_map, "p_norm", 2)
        if p_norm == 2:
            return torch.norm(grad, dim=-1, p=2, keepdim=True)
        if p_norm == np.inf:
            return torch.norm(grad, dim=-1, p=1, keepdim=True)
        if p_norm == 1:
            return torch.norm(grad, dim=-1, p=np.inf, keepdim=True)
        dual_p = float(p_norm) / (float(p_norm) - 1.0)
        return torch.norm(grad, dim=-1, p=dual_p, keepdim=True)

    def first_order_lagrangian_gap(self, current_state, dual_state):
        z = current_state["z"]
        x = current_state.get("x")
        hom_state = current_state.get("hom_state")
        if x is None or hom_state is None:
            x, hom_state = self._hom_forward_with_state(z)
        grad = self._original_lagrangian_gradient_z(z, dual_state, x=x, hom_state=hom_state)
        support_term = self._latent_ball_dual_norm(grad)
        linear_term = torch.sum(grad * z, dim=-1, keepdim=True)
        return torch.clamp(linear_term + support_term, min=0.0).max().view(1, -1)

    def project_to_ball(self, z):
        """Project points onto the unit ball."""
        return _project_to_ball(z, self.hom_map.p_norm)

    def _hom_forward(self, z):
        if self._hom_forward_supports_method:
            return self._hom_forward_fn(z, method=self.hom_map_gradient)
        return self._hom_forward_fn(z)

    def _hom_forward_with_state(self, z):
        if self._hom_forward_supports_return_state:
            return self._hom_forward_fn(z, method=self.hom_map_gradient, return_state=True)
        return self._hom_forward(z), None

    def _eq_residual(self, x):
        if self._eq_constraint_x is not None:
            return self._eq_constraint_x(x)
        return _constraint_residual(self.problem, x, equality_only=True)

    def _eq_violation(self, x):
        residual = self._eq_residual(x)
        if residual.numel() == 0:
            return torch.zeros(1, 1, device=x.device, dtype=x.dtype)
        return residual.abs().max().view(1, -1)

    def _inner_proximal_gradient(self, z, z_center, x_center, x_current=None, hom_state_current=None):
        if self.inner_proximal_coef == 0:
            return torch.zeros_like(z)
        if self.inner_proximal_space == "z":
            return self.inner_proximal_coef * (z - z_center)
        if x_center is None:
            raise ValueError("x_center is required when inner_proximal_space='x'.")
        if x_current is None or hom_state_current is None:
            x_current, hom_state_current = self._hom_forward_with_state(z)
        if hasattr(self.hom_map, "vjp"):
            grad_x = self.inner_proximal_coef * (x_current - x_center)
            return self.hom_map.vjp(z, grad_x, method=self.hom_map_gradient, state=hom_state_current).detach()
        raise NotImplementedError("Hom-ALM x-space inner proximal gradient requires hom_map.vjp.")

    def _accelerated_inner_point(self, z):
        if self.acceleration_method != "nag":
            return z
        return _hom_nag_lookahead(
            z,
            acceleration_space=self.acceleration_space,
            z_velocity=self._nag_velocity_z,
            x_velocity=self._nag_velocity_x,
            beta=self.acceleration_beta,
            hom_forward=self._hom_forward,
            hom_inverse=self.hom_map.inverse,
            project_z=self.project_to_ball,
        )

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        """
        Optimize using homomorphic augmented Lagrangian method.
        Dual over equality constraints
        Homeomorphism for inequality constraints
        """
        prob_dim = self.problem.nvar
        # Initialize starting point
        if initial_point is None:
            if seed is not None:
                torch.manual_seed(seed)
            x = _randn_problem_row(self.problem, prob_dim)
        else:
            x = _as_problem_row(self.problem, initial_point)
        init_transform_start = time.perf_counter()
        z = self.hom_map.inverse(x)
        x, hom_state = self._hom_forward_with_state(z)
        self.last_initial_transform_time = time.perf_counter() - init_transform_start

        dual_state = self.dual_var
        last_trans_time = 0.0
        penalty_violation = self._eq_violation(x)
        state = {
            "z": z,
            "x": x,
            "hom_state": hom_state,
        }
        initial_gap = (
            self.first_order_lagrangian_gap(state, dual_state)
            if self.check_first_order_lagrangian_gap
            else torch.zeros(1, 1, **_problem_tensor_kwargs(self.problem))
        )
        initial_split = _constraint_violation_split(self.problem, x)
        initial_eval = {
            "objective": self.problem.objective_x(x),
            "violation": self._eq_violation(x),
            "decision": x,
            "z_trajectory": z,
            "first_order_lagrangian_gap": initial_gap,
            **initial_split,
        }

        def _ensure_state_cache(current_state, *, measure_time=False):
            nonlocal last_trans_time
            x_state = current_state["x"]
            hom_state_state = current_state.get("hom_state")
            if x_state is None or hom_state_state is None:
                start_time = time.perf_counter() if measure_time else None
                x_state, hom_state_state = self._hom_forward_with_state(current_state["z"])
                current_state["x"] = x_state
                current_state["hom_state"] = hom_state_state
                if measure_time:
                    last_trans_time = time.perf_counter() - start_time
            elif measure_time:
                last_trans_time = 0.0
            return x_state, hom_state_state

        def _run_inner_loop(current_state, dual_state, outer_lr, outer_iter):
            del outer_iter
            inner_lr = float(self.inner_learning_rate if self.inner_learning_rate is not None else outer_lr)
            base_inner_lr = inner_lr
            z_outer = current_state["z"].clone()
            x_iter, hom_state_iter = _ensure_state_cache(current_state)
            x_outer = x_iter.detach() if self.use_proximal and self.proximal_space == "x" else None
            outer_grad_cache = None
            if self.hom_map_gradient == "explicit" and self._build_explicit_grad_cache is not None:
                outer_grad_cache = self._build_explicit_grad_cache(
                    dual_state,
                    penalty_coef=self.penalty_coef,
                    proximal_coef=self.proximal_coef,
                    x_outer=x_outer,
                    proximal_space=self.proximal_space,
                )
            error = torch.tensor(float("inf"), device=self.problem.device)
            best_violation = torch.tensor(float("inf"), device=self.problem.device)
            z_iter = current_state["z"]
            inner_tolerance = _adaptive_inner_tolerance(
                self._eq_violation(x_iter),
                tol_factor=self.inner_tol_factor,
                tol_min=self.inner_tol_min,
            )
            if self.inner_restart:
                _reset_update_backend(self.opt)
            if self.acceleration_method == "nag":
                if self.inner_restart or _velocity_needs_reset(self._nag_velocity_z, z_iter):
                    self._nag_velocity_z = torch.zeros_like(z_iter)
                if self.acceleration_space == "x":
                    if self.inner_restart or _velocity_needs_reset(self._nag_velocity_x, x_iter):
                        self._nag_velocity_x = torch.zeros_like(x_iter)
                else:
                    self._nag_velocity_x = None

            def _step_once(z_current, x_current, hom_state_current, inner_iter, inner_z_center=None, inner_x_center=None):
                step_origin = self._accelerated_inner_point(z_current)
                step_x = x_current
                step_hom_state = hom_state_current
                if self.acceleration_method == "nag":
                    step_x, step_hom_state = self._hom_forward_with_state(step_origin)
                grad = self.gradient_lagrangian(
                    step_origin,
                    dual_state,
                    z_outer=z_outer,
                    x_outer=x_outer,
                    x=step_x,
                    hom_state=step_hom_state,
                    outer_cache=outer_grad_cache,
                )
                if inner_z_center is not None:
                    grad = grad + self._inner_proximal_gradient(
                        step_origin,
                        inner_z_center,
                        inner_x_center,
                        x_current=step_x,
                        hom_state_current=step_hom_state,
                    )
                update = self.opt.step(grad)
                z_inner = step_origin - inner_lr * update
                z_inner = z_inner.detach()
                z_proj = self.project_to_ball(z_inner)
                x_proj, hom_state_proj = self._hom_forward_with_state(z_proj)
                if self.acceleration_method == "nag":
                    if self.acceleration_space == "z":
                        self._nag_velocity_z = (z_proj - z_current).detach()
                    else:
                        self._nag_velocity_x = (x_proj - x_current).detach()
                current_error = torch.norm(z_proj - z_current, p=2, dim=-1)
                if self.inner_stepsize_rule == "adaptive":
                    current_violation = self._eq_violation(x_proj)
                    previous_best_violation = best_violation
                    next_best_violation = torch.minimum(best_violation, current_violation)
                    next_lr = _update_penalty_inner_learning_rate(
                        base_inner_lr,
                        inner_lr,
                        stepsize_rule=self.inner_stepsize_rule,
                        lr_decay=self.inner_lr_decay,
                        convergence_threshold=self.convergence_threshold,
                        current_error=float(current_violation.detach().max().item()),
                        best_error=float(previous_best_violation.detach().max().item()),
                        iteration=inner_iter,
                    )
                else:
                    next_best_violation = best_violation
                    next_lr = inner_lr
                return z_proj, x_proj.detach(), hom_state_proj, current_error, next_best_violation, next_lr

            if self.uses_inner_proximal_stages:
                for prox_iter in range(self.inner_proximal_update_iterations):
                    if self.inner_restart:
                        _reset_update_backend(self.opt)
                        if self.acceleration_method == "nag":
                            self._nag_velocity_z = torch.zeros_like(z_iter)
                            self._nag_velocity_x = torch.zeros_like(x_iter) if self.acceleration_space == "x" else None
                    inner_z_center = z_iter.clone().detach()
                    inner_x_center = (
                        x_iter.detach()
                        if self.inner_proximal_space == "x"
                        else None
                    )
                    for local_iter in range(self.inner_proximal_steps):
                        inner_iter = prox_iter * self.inner_proximal_steps + local_iter
                        z_iter, x_iter, hom_state_iter, error, best_violation, inner_lr = _step_once(
                            z_iter,
                            x_iter,
                            hom_state_iter,
                            inner_iter,
                            inner_z_center=inner_z_center,
                            inner_x_center=inner_x_center,
                        )
                        if _should_stop_inner_loop(
                            error,
                            inner_iter,
                            stopping_rule=self.inner_stopping_rule,
                            min_iterations=self.inner_iterations_min,
                            max_iterations=self.inner_iterations_max,
                            tolerance=inner_tolerance,
                        ):
                            break
                    if _should_stop_inner_loop(
                        error,
                        inner_iter,
                        stopping_rule=self.inner_stopping_rule,
                        min_iterations=self.inner_iterations_min,
                        max_iterations=self.inner_iterations_max,
                        tolerance=inner_tolerance,
                    ):
                        break
            else:
                inner_budget = self.inner_iterations_max if self.inner_stopping_rule == "adaptive" else self.inner_iterations
                for inner_iter in range(inner_budget):
                    z_iter, x_iter, hom_state_iter, error, best_violation, inner_lr = _step_once(
                        z_iter,
                        x_iter,
                        hom_state_iter,
                        inner_iter,
                    )
                    if _should_stop_inner_loop(
                        error,
                        inner_iter,
                        stopping_rule=self.inner_stopping_rule,
                        min_iterations=self.inner_iterations_min,
                        max_iterations=self.inner_iterations_max,
                        tolerance=inner_tolerance,
                    ):
                        break
            return {
                "z": z_iter,
                "x": x_iter,
                "hom_state": hom_state_iter,
            }, error

        def _evaluate_state(current_state):
            x_state, _ = _ensure_state_cache(current_state, measure_time=True)
            split = _constraint_violation_split(self.problem, x_state)
            return {
                "objective": self.problem.objective_x(x_state),
                "violation": self._eq_violation(x_state),
                "decision": x_state,
                "z_trajectory": current_state["z"],
                **split,
            }

        def _augment_eval_payload(current_state, dual_state, eval_payload):
            del eval_payload
            if not self.check_first_order_lagrangian_gap:
                return None
            return {
                "first_order_lagrangian_gap": self.first_order_lagrangian_gap(current_state, dual_state),
            }

        def _should_stop(eval_payload, error, outer_iter):
            del outer_iter
            first_order_gap = eval_payload.get("first_order_lagrangian_gap")
            first_order_ready = (
                True
                if not self.check_first_order_lagrangian_gap
                else first_order_gap is not None
                and float(first_order_gap.detach().max().item()) < self.first_order_lagrangian_gap_threshold
            )
            if (
                eval_payload["violation"] < self.convergence_threshold
                and error < self.convergence_threshold
                and first_order_ready
            ):
                if verbose:
                    print("Converged outer loop for Hom-ALM")
                return True
            return False

        def _update_dual_state(dual_state, current_state, outer_iter):
            nonlocal penalty_violation
            del outer_iter
            x_state, _ = _ensure_state_cache(current_state)
            if self.use_lagrangian:
                eq_constraint_residual = self._eq_residual(x_state)
                dual_state = dual_state + _dual_update_scale(self) * eq_constraint_residual
            current_penalty_violation = self._eq_violation(x_state)
            penalty_violation = _maybe_update_penalty(self, penalty_violation, current_penalty_violation)
            return _clamp_dual_state(dual_state, max_dual=self.max_dual)

        def _update_learning_rate(current_lr, recorder, outer_iter):
            del outer_iter
            return _update_penalty_outer_learning_rate(
                current_lr,
                recorder,
                stepsize_rule=self.outer_stepsize_rule,
                lr_decay=self.outer_lr_decay,
                convergence_threshold=self.convergence_threshold,
            )

        state, dual_state, payload, self.learning_rate, _ = run_penalty_outer_loop(
            initial_state=state,
            initial_eval=initial_eval,
            initial_dual_state=dual_state,
            run_inner_loop=_run_inner_loop,
            evaluate_state=_evaluate_state,
            update_dual_state=_update_dual_state,
            update_learning_rate=_update_learning_rate,
            should_stop=_should_stop,
            augment_eval_payload=_augment_eval_payload,
            outer_iterations=self.outer_iterations,
            max_running_time=self.max_running_time,
            initial_learning_rate=self.learning_rate,
            outer_stepsize_rule=self.outer_stepsize_rule,
            verbose=verbose,
            progress_disable=isinstance(self.problem, (ProjProblem, LinearProblem)),
            track_decisions=self.problem.nvar == 2,
            record_best_decision=self.return_best_violation,
            extra_track_names=_CONSTRAINT_SPLIT_TRACK_NAMES
            + (("z_trajectory",) if self.problem.nvar == 2 else ())
            + (("first_order_lagrangian_gap",) if self.check_first_order_lagrangian_gap else ()),
            verbose_metrics=lambda current_state, current_dual: {
                **(
                    {"lag_gap": self.first_order_lagrangian_gap(current_state, current_dual)}
                    if self.check_first_order_lagrangian_gap
                    else {}
                ),
                "penalty": self.penalty_coef,
                "dual_lr": self.dual_learning_rate,
            },
            verbose_interval=self.params.get("verbose_interval", 50),
        )
        if self.return_best_violation and payload.get("best_decision") is not None:
            x = payload["best_decision"].to(self.problem.device)
        else:
            x = state["x"].to(self.problem.device)
        self.last_final_transform_time = float(last_trans_time)
        self.last_z_trajectory = payload.get("z_trajectory")
        if "outer_iter_time" not in payload:
            raise KeyError("Penalty/Hom-ALM payload must include explicit outer_iter_time.")
        self.last_outer_iter_time = payload["outer_iter_time"]
        self.last_inner_iter_time = payload.get("inner_iter_time", [])
        _store_first_order_lagrangian_gap_metrics(self, payload)
        _store_constraint_violation_split_metrics(self, payload)
        return (
            x,
            payload["decision_trajectory"],
            payload["objective_traj"],
            payload["violation_traj"],
            payload["per_iter_time"],
            last_trans_time,
        )

class LagrangianOptimizer(BaseOptimizer):
    """Implements Lagrangian optimization with dual update."""
    def __init__(self, problem, params):
        super().__init__(problem=problem, params=params, hom_map=None)
        self.problem = problem
        self._init_optimization_params(params)

    def _init_optimization_params(self, params):
        """Initialize all optimization-related parameters"""
        # Base parameters
        self.max_running_time = params['max_running_time']
        self.learning_rate = params['learning_rate']
        self.inner_learning_rate = params.get('inner_learning_rate', None)
        self.outer_iterations = params['outer_iterations']
        self.inner_iterations = params['inner_iterations']
        self.outer_lr_decay, self.inner_lr_decay = _resolve_penalty_lr_decays(params)
        self.convergence_threshold = params['convergence_threshold']
        self.outer_stepsize_rule, self.inner_stepsize_rule = _resolve_penalty_stepsize_rules(params)
        self.acceleration_method, self.acceleration_space = _resolve_acceleration_config(params, default_space="x")
        self.acceleration_beta = float(params.get('momentum', 0.0))
        self.opt = _build_algorithm_update_backend(params, self.acceleration_method)
        dual_dim, self.dual_ineq_cons, self.dual_eq_cons = _constraint_layout(self.problem)
        self.dual_var = torch.zeros(1, dual_dim, **_problem_tensor_kwargs(self.problem))
        _init_penalty_family_controls(self, params, proximal_coef_default=1.0)
        (
            self.check_first_order_lagrangian_gap,
            self.first_order_lagrangian_gap_threshold,
        ) = _resolve_first_order_lagrangian_gap_config(
            params,
            default_enabled=self.use_lagrangian,
        )
        self.proximal_space = _validate_proximal_space(params.get('proximal_space', 'x'))
        inner_stopping_config = _resolve_inner_stopping_config(params, inner_iterations=self.inner_iterations)
        self.inner_stopping_rule = inner_stopping_config["inner_stopping_rule"]
        self.inner_iterations_min = inner_stopping_config["inner_iterations_min"]
        self.inner_iterations_max = inner_stopping_config["inner_iterations_max"]
        self.inner_tol_factor = inner_stopping_config["inner_tol_factor"]
        self.inner_tol_min = inner_stopping_config["inner_tol_min"]
        active_inner_budget = self.inner_iterations_max if self.inner_stopping_rule == "adaptive" else self.inner_iterations
        inner_solver_config = _resolve_penalty_inner_solver_config(
            params,
            inner_iterations=self.inner_iterations,
            inner_iteration_budget=active_inner_budget,
            default_proximal_space="x",
        )
        self.inner_solver = inner_solver_config["inner_solver"]
        self.uses_inner_proximal_stages = inner_solver_config["uses_inner_proximal_stages"]
        self.inner_proximal_update_iterations = inner_solver_config["inner_proximal_update_iterations"]
        self.inner_proximal_steps = inner_solver_config["inner_proximal_steps"]
        self.inner_proximal_coef = inner_solver_config["inner_proximal_coef"]
        self.inner_proximal_space = inner_solver_config["inner_proximal_space"]
        self.inner_restart = _resolve_inner_restart(params)
        self._nag_velocity = None
        self.lagrangian_gradient = _validate_gradient_method(
            params.get('lagrangian_gradient', 'explicit'),
            name="lagrangian_gradient",
        )
        self._gradient_lagrangian_x = getattr(self.problem, "gradient_lagrangian_x", None)
        self._gradient_lagrangian_x_accepts_method = (
            self._gradient_lagrangian_x is not None
            and _supports_keyword(self._gradient_lagrangian_x, "method")
        )

    def gradient_lagrangian(self, x, dual_var, x_outer=None):
        """Gradient of augmented objective function."""
        if self._gradient_lagrangian_x is not None:
            kwargs = {
                "penalty_coef": self.penalty_coef,
                "proximal_coef": self.proximal_coef,
                "x_outer": x_outer,
            }
            if self._gradient_lagrangian_x_accepts_method:
                kwargs["method"] = self.lagrangian_gradient
            grad = self._gradient_lagrangian_x(x, dual_var, **kwargs)
        else:
            raise NotImplementedError(
                f"{type(self.problem).__name__} must implement gradient_lagrangian_x for ALM/PALM."
            )
        if torch.isnan(grad).any():
            raise FloatingPointError("Lagrangian gradient contains NaN values.")
        return grad

    def _original_lagrangian_gradient_x(self, x, dual_var):
        if self._gradient_lagrangian_x is not None:
            kwargs = {
                "penalty_coef": None,
                "proximal_coef": None,
                "x_outer": None,
            }
            if self._gradient_lagrangian_x_accepts_method:
                kwargs["method"] = self.lagrangian_gradient
            grad = self._gradient_lagrangian_x(x, dual_var, **kwargs)
        else:
            raise NotImplementedError(
                f"{type(self.problem).__name__} must implement gradient_lagrangian_x for ALM/PALM."
            )
        if torch.isnan(grad).any():
            raise FloatingPointError("Original Lagrangian gradient contains NaN values.")
        return grad

    def first_order_lagrangian_gap(self, x, dual_state):
        grad = self._original_lagrangian_gradient_x(x, dual_state)
        stationarity = torch.norm(grad, p=np.inf).view(1, -1)
        residual = _constraint_residual(self.problem, x, equality_only=False)
        complementarity = torch.zeros_like(stationarity)
        if residual.numel() > 0:
            if self.dual_ineq_cons is not None:
                ineq_idx = list(self.dual_ineq_cons)
                if ineq_idx:
                    complementarity = torch.abs(dual_state[:, ineq_idx] * residual[:, ineq_idx]).max().view(1, -1)
            elif self.dual_eq_cons is None:
                complementarity = torch.abs(dual_state * residual).max().view(1, -1)
        return torch.maximum(stationarity, complementarity)

    def _inner_proximal_gradient(self, x, x_center):
        if self.inner_proximal_coef == 0:
            return torch.zeros_like(x)
        return self.inner_proximal_coef * (x - x_center)

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        """
        Optimize using augmented Lagrangian method.
        """
        prob_dim = self.problem.nvar
        # Initialize starting point
        if initial_point is None:
            if seed is not None:
                torch.manual_seed(seed)
            x = _randn_problem_row(self.problem, prob_dim)
        else:
            x = _as_problem_row(self.problem, initial_point)

        dual_state = self.dual_var
        penalty_violation = _constraint_violation(
            self.problem,
            x,
            equality_only=self.dual_eq_cons is not None,
        )
        initial_split = _constraint_violation_split(self.problem, x)
        initial_eval = {
            "objective": self.problem.objective_x(x),
            "violation": initial_split["full_violation"],
            "decision": x,
            **initial_split,
        }
        if self.check_first_order_lagrangian_gap:
            initial_eval["first_order_lagrangian_gap"] = self.first_order_lagrangian_gap(x, dual_state)

        def _run_inner_loop(x_state, dual_state, outer_lr, outer_iter):
            del outer_iter
            inner_lr = float(self.inner_learning_rate if self.inner_learning_rate is not None else outer_lr)
            base_inner_lr = inner_lr
            x_outer = x_state.clone()
            error = torch.tensor(float("inf"), device=self.problem.device)
            best_violation = torch.tensor(float("inf"), device=self.problem.device)
            equality_only = self.dual_eq_cons is not None
            x_iter = x_state
            inner_tolerance = _adaptive_inner_tolerance(
                _constraint_violation(self.problem, x_state, equality_only=equality_only),
                tol_factor=self.inner_tol_factor,
                tol_min=self.inner_tol_min,
            )
            if self.inner_restart:
                _reset_update_backend(self.opt)
            if self.acceleration_method == "nag":
                if self.inner_restart or _velocity_needs_reset(self._nag_velocity, x_state):
                    self._nag_velocity = torch.zeros_like(x_state)
                nag_velocity = self._nag_velocity
            else:
                nag_velocity = None

            def _step_once(x_current, inner_iter, inner_x_center=None):
                nonlocal inner_lr, best_violation, nag_velocity
                step_origin = (
                    _nag_lookahead(x_current, nag_velocity, self.acceleration_beta)
                    if self.acceleration_method == "nag"
                    else x_current
                )
                grad = self.gradient_lagrangian(step_origin, dual_state, x_outer=x_outer)
                if inner_x_center is not None:
                    grad = grad + self._inner_proximal_gradient(step_origin, inner_x_center)
                    update = self.opt.step(grad)
                    x_inner = step_origin - inner_lr * update
                else:
                    update = self.opt.step(grad)
                    x_inner = step_origin - inner_lr * update
                x_inner = x_inner.detach()
                if self.acceleration_method == "nag":
                    nag_velocity = (x_inner - x_current).detach()
                current_error = torch.norm(grad, p=np.inf)
                if self.inner_stepsize_rule == "adaptive":
                    current_violation = _constraint_violation(
                        self.problem,
                        x_inner,
                        equality_only=equality_only,
                    )
                    previous_best_violation = best_violation
                    best_violation = torch.minimum(best_violation, current_violation)
                    inner_lr = _update_penalty_inner_learning_rate(
                        base_inner_lr,
                        inner_lr,
                        stepsize_rule=self.inner_stepsize_rule,
                        lr_decay=self.inner_lr_decay,
                        convergence_threshold=self.convergence_threshold,
                        current_error=float(current_violation.detach().max().item()),
                        best_error=float(previous_best_violation.detach().max().item()),
                        iteration=inner_iter,
                    )
                return x_inner.clone(), current_error

            if self.uses_inner_proximal_stages:
                for prox_iter in range(self.inner_proximal_update_iterations):
                    if self.inner_restart:
                        _reset_update_backend(self.opt)
                        if self.acceleration_method == "nag":
                            nag_velocity = torch.zeros_like(x_iter)
                    inner_x_center = x_iter.clone().detach()
                    for local_iter in range(self.inner_proximal_steps):
                        inner_iter = prox_iter * self.inner_proximal_steps + local_iter
                        x_iter, error = _step_once(x_iter, inner_iter, inner_x_center=inner_x_center)
                        if _should_stop_inner_loop(
                            error,
                            inner_iter,
                            stopping_rule=self.inner_stopping_rule,
                            min_iterations=self.inner_iterations_min,
                            max_iterations=self.inner_iterations_max,
                            tolerance=inner_tolerance,
                        ):
                            break
                    if _should_stop_inner_loop(
                        error,
                        inner_iter,
                        stopping_rule=self.inner_stopping_rule,
                        min_iterations=self.inner_iterations_min,
                        max_iterations=self.inner_iterations_max,
                        tolerance=inner_tolerance,
                    ):
                        break
            else:
                inner_budget = self.inner_iterations_max if self.inner_stopping_rule == "adaptive" else self.inner_iterations
                for inner_iter in range(inner_budget):
                    x_iter, error = _step_once(x_iter, inner_iter)
                    if _should_stop_inner_loop(
                        error,
                        inner_iter,
                        stopping_rule=self.inner_stopping_rule,
                        min_iterations=self.inner_iterations_min,
                        max_iterations=self.inner_iterations_max,
                        tolerance=inner_tolerance,
                    ):
                        break
                    if self.inner_stopping_rule == "fixed" and error < self.convergence_threshold:
                        if verbose:
                            print(f"Inner loop converged after {inner_iter} iterations")
                        break
            if self.acceleration_method == "nag":
                self._nag_velocity = nag_velocity.detach()
            return x_iter, error

        def _evaluate_state(x_state):
            split = _constraint_violation_split(self.problem, x_state)
            return {
                "objective": self.problem.objective_x(x_state),
                "violation": split["full_violation"],
                "decision": x_state,
                **split,
            }

        def _augment_eval_payload(x_state, dual_state, eval_payload):
            del eval_payload
            if not self.check_first_order_lagrangian_gap:
                return None
            return {
                "first_order_lagrangian_gap": self.first_order_lagrangian_gap(x_state, dual_state),
            }

        def _should_stop(eval_payload, error, outer_iter):
            first_order_gap = eval_payload.get("first_order_lagrangian_gap")
            first_order_ready = (
                True
                if not self.check_first_order_lagrangian_gap
                else first_order_gap is not None
                and float(first_order_gap.detach().max().item()) < self.first_order_lagrangian_gap_threshold
            )
            if (
                eval_payload["violation"] < self.convergence_threshold
                and error < self.convergence_threshold
                and first_order_ready
            ):
                if verbose:
                    print(f"Converged after {outer_iter + 1} outer iterations")
                return True
            return False

        def _update_dual_state(dual_state, x_state, outer_iter):
            nonlocal penalty_violation
            del outer_iter
            if self.use_lagrangian:
                constraint_residual = _constraint_residual(self.problem, x_state, equality_only=False)
                dual_state = dual_state + _dual_update_scale(self) * constraint_residual
            current_penalty_violation = _constraint_violation(
                self.problem,
                x_state,
                equality_only=self.dual_eq_cons is not None,
            )
            penalty_violation = _maybe_update_penalty(self, penalty_violation, current_penalty_violation)
            return _clamp_dual_state(
                dual_state,
                max_dual=self.max_dual,
                eq_cons=self.dual_eq_cons,
                ineq_cons=self.dual_ineq_cons,
            )

        def _update_learning_rate(current_lr, recorder, outer_iter):
            del outer_iter
            return _update_penalty_outer_learning_rate(
                current_lr,
                recorder,
                stepsize_rule=self.outer_stepsize_rule,
                lr_decay=self.outer_lr_decay,
                convergence_threshold=self.convergence_threshold,
            )

        x, dual_state, payload, self.learning_rate, _ = run_penalty_outer_loop(
            initial_state=x,
            initial_eval=initial_eval,
            initial_dual_state=dual_state,
            run_inner_loop=_run_inner_loop,
            evaluate_state=_evaluate_state,
            update_dual_state=_update_dual_state,
            update_learning_rate=_update_learning_rate,
            should_stop=_should_stop,
            augment_eval_payload=_augment_eval_payload,
            outer_iterations=self.outer_iterations,
            max_running_time=self.max_running_time,
            initial_learning_rate=self.learning_rate,
            outer_stepsize_rule=self.outer_stepsize_rule,
            verbose=verbose,
            progress_disable=isinstance(self.problem, (ProjProblem, LinearProblem)),
            track_decisions=self.problem.nvar == 2,
            record_best_decision=self.return_best_violation,
            extra_track_names=_CONSTRAINT_SPLIT_TRACK_NAMES
            + (("first_order_lagrangian_gap",) if self.check_first_order_lagrangian_gap else ()),
            verbose_metrics=lambda current_state, current_dual: {
                **(
                    {"lag_gap": self.first_order_lagrangian_gap(current_state, current_dual)}
                    if self.check_first_order_lagrangian_gap
                    else {}
                ),
                "penalty": self.penalty_coef,
                "dual_lr": self.dual_learning_rate,
            },
            verbose_interval=self.params.get("verbose_interval", 50),
        )
        if self.return_best_violation and payload.get("best_decision") is not None:
            x = payload["best_decision"].to(self.problem.device)
        if "outer_iter_time" not in payload:
            raise KeyError("ALM/PALM payload must include explicit outer_iter_time.")
        self.last_outer_iter_time = payload["outer_iter_time"]
        self.last_inner_iter_time = payload.get("inner_iter_time", [])
        _store_first_order_lagrangian_gap_metrics(self, payload)
        _store_constraint_violation_split_metrics(self, payload)
        return x, payload["decision_trajectory"], payload["objective_traj"], payload["violation_traj"], payload["per_iter_time"]

class EqualityConstrainedALMOptimizer(BaseOptimizer):
    """ALM over equalities with convex inequalities kept as CVXPY constraints."""
    def __init__(self, problem, params):
        super().__init__(problem=problem, params=params, hom_map=None)
        self.problem = problem
        self._init_optimization_params(params)

    def _init_optimization_params(self, params):
        self.max_running_time = params['max_running_time']
        self.outer_iterations = params['outer_iterations']
        self.convergence_threshold = params['convergence_threshold']
        self.check_outer_objective_change = bool(params.get('check_outer_objective_change', True))
        self.outer_objective_change_threshold = float(
            params.get(
                'outer_objective_change_threshold',
                params.get('objective_change_threshold', self.convergence_threshold),
            )
        )
        self.min_outer_iterations = int(params.get('min_outer_iterations', 2))
        _init_penalty_family_controls(self, params, proximal_coef_default=0.0)
        self.solver_options = params.get('solver_options')
        self.subproblem_time_limit_sec = params.get('subproblem_time_limit_sec')
        self.solver_verbose = bool(params.get('solver_verbose', False))
        self.opt_type = params.get('opt_type', 'ALM-EQ')
        self.n_eq = int(getattr(self.problem, "n_eq", 0) or 0)
        self.dual_var = torch.zeros(1, self.n_eq, **_problem_tensor_kwargs(self.problem))

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        prob_dim = self.problem.nvar
        if initial_point is None:
            if seed is not None:
                torch.manual_seed(seed)
            x = _randn_problem_row(self.problem, prob_dim)
        else:
            x = _as_problem_row(self.problem, initial_point)

        recorder = IterationRecorder(
            track_decisions=self.problem.nvar == 2,
            extra_track_names=_CONSTRAINT_SPLIT_TRACK_NAMES,
        )

        def _evaluate_state(x_state):
            split = _constraint_violation_split(self.problem, x_state)
            return {
                "objective": self.problem.objective_x(x_state),
                "violation": split["full_violation"],
                "decision": x_state,
                **split,
            }

        initial_eval = _evaluate_state(x)
        penalty_violation = self.problem.eq_constraint_x(x).abs().max() if self.n_eq > 0 else initial_eval["violation"]
        recorder.record_state(
            initial_eval["objective"],
            initial_eval["violation"],
            decision=initial_eval["decision"],
            **{name: initial_eval.get(name) for name in _CONSTRAINT_SPLIT_TRACK_NAMES},
        )

        solver = _cached_exact_solver(self, "convex")
        dual_state = self.dual_var
        start_time = time.perf_counter()
        solver_iter_time = []
        outer_range = tqdm(range(self.outer_iterations), disable=not verbose)
        for outer_iter in outer_range:
            if time.perf_counter() - start_time >= self.max_running_time:
                break
            outer_start_time = time.perf_counter()
            x_outer = _as_numpy_vector(x) if self.use_proximal else None
            solver_start_time = time.perf_counter()
            solve_result = solve_exact_result(
                solver,
                "eq_alm",
                x_init=_as_numpy_vector(x),
                dual_var=dual_state.detach().view(-1).cpu().numpy(),
                penalty_coef=self.penalty_coef,
                proximal_coef=self.proximal_coef,
                x_outer=x_outer,
                solver_options=self.solver_options,
                time_limit_sec=self.subproblem_time_limit_sec,
                verbose=self.solver_verbose,
            )
            reported_solver_time = solve_result.get("runtime_total")
            solver_time = (
                float(reported_solver_time)
                if reported_solver_time is not None
                else time.perf_counter() - solver_start_time
            )
            solver_iter_time.append(solver_time)
            if solve_result.get("solution") is None:
                status = solve_result.get("status")
                error = solve_result.get("extras", {}).get("error")
                raise RuntimeError(f"{self.opt_type} subproblem failed at outer iteration {outer_iter}: {status}; {error}")

            x = _as_torch_row(
                solve_result["solution"],
                device=self.problem.device,
                dtype=_infer_problem_dtype(self.problem),
            )
            eval_payload = _evaluate_state(x)
            recorder.append_iter_time(time.perf_counter() - outer_start_time)
            recorder.record_state(
                eval_payload["objective"],
                eval_payload["violation"],
                decision=eval_payload["decision"],
                **{name: eval_payload.get(name) for name in _CONSTRAINT_SPLIT_TRACK_NAMES},
            )
            eq_residual = self.problem.eq_constraint_x(x)
            eq_violation = eq_residual.abs().max() if eq_residual.numel() else torch.zeros((), device=x.device, dtype=x.dtype)
            if self.use_lagrangian and self.n_eq > 0:
                dual_state = dual_state + _dual_update_scale(self) * eq_residual
                dual_state = _clamp_dual_state(dual_state, max_dual=self.max_dual)
            penalty_violation = _maybe_update_penalty(self, penalty_violation, eq_violation)
            objective_ready = True
            if self.check_outer_objective_change:
                objective_ready = False
                if outer_iter + 1 >= self.min_outer_iterations and len(recorder.objective_traj) >= 2:
                    objective_change = abs(
                        float(recorder.objective_traj[-1].detach().mean().item())
                        - float(recorder.objective_traj[-2].detach().mean().item())
                    )
                    objective_ready = objective_change <= self.outer_objective_change_threshold
            if (
                eval_payload["violation"] < self.convergence_threshold
                and eq_violation < self.convergence_threshold
                and objective_ready
            ):
                if verbose:
                    print(f"Converged {self.opt_type} after {outer_iter + 1} outer iterations")
                break

        self.dual_var = dual_state
        payload = recorder.finalize()
        self.last_outer_iter_time = payload["per_iter_time"]
        self.last_solver_iter_time = solver_iter_time
        _store_constraint_violation_split_metrics(self, payload)
        if self.return_best_violation and payload.get("best_decision") is not None:
            x = payload["best_decision"].to(self.problem.device)
        return x, payload["decision_trajectory"], payload["objective_traj"], payload["violation_traj"], payload["per_iter_time"]

class FrankWolfeOptimizer(BaseOptimizer):
    """Implements Frank-Wolfe optimization."""
    def __init__(self, problem, params):
        super().__init__(problem=problem, params=params, hom_map=None)
        self.problem = problem
        self._init_optimization_params(params)

    def _init_optimization_params(self, params):
        """Initialize all optimization-related parameters"""
        # Base parameters
        self.learning_rate = params['learning_rate']
        self.max_iterations = params['max_iterations']
        self.max_running_time = params['max_running_time']
        self.convergence_threshold = params['convergence_threshold']
        self.stepsize_rule = _validate_stepsize_rule(params['stepsize_rule'])
        self.lr_decay = params['lr_decay']
        self.fw_gap_threshold = params.get('fw_gap_threshold', self.convergence_threshold)
        if 'linearization_subproblem' in params:
            self.linearization_subproblem_params = copy.deepcopy(params['linearization_subproblem'])
        else:
            self.linearization_subproblem_params = _default_first_order_subproblem_params(
                params,
                outer_iterations_key='linearization_outer_iterations',
                inner_iterations_key='linearization_inner_iterations',
            )

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        recorder = IterationRecorder(track_decisions=self.problem.nvar == 2)
        prob_dim = self.problem.nvar

        if initial_point is None:
            if seed is not None:
                torch.manual_seed(seed)
            x = _randn_problem_row(self.problem, prob_dim)
        else:
            x = _as_problem_row(self.problem, initial_point)
        recorder.record_state(
            self.problem.objective_x(x),
            _constraint_violation(self.problem, x, equality_only=False),
            decision=x,
        )

        point = x
        elapsed_time = 0.0
        for iter in tqdm(range(self.max_iterations)):
            st = time.time()
            grad = self.problem.gradient_objective_x(x)
            if isinstance(self.problem, ConvexOpt):
                solver = _cached_exact_solver(self, "convex")
                s = _extract_exact_solution(
                    solve_exact_result(
                        solver,
                        'linear',
                        _as_numpy_vector(x),
                        _as_numpy_vector(grad),
                    )
                )
                s = _as_torch_row(s, device=x.device, dtype=x.dtype)
            elif isinstance(self.problem, ToyStarOpt):
                linearization_problem = LinearProblem(x, self.problem).to_device(x.device)
                lag_optimizer = LagrangianOptimizer(linearization_problem, self.linearization_subproblem_params)
                s,  _, _, _, _ = lag_optimizer.optimize(initial_point=x, verbose=False)
            else:
                raise ValueError(f"Linear oracle is not implemented for {type(self.problem).__name__}")
            update = x - s
            fw_gap = torch.sum(grad * update, dim=-1).abs().max()
            if fw_gap <= float(self.fw_gap_threshold):
                break
            lr = self.learning_rate
            if self.stepsize_rule == 'diminish':
                lr /= (iter+1)**0.5
            point_proj = (x - lr * update).detach()

            et = time.time()
            iter_time = et - st
            recorder.append_iter_time(iter_time)
            elapsed_time += iter_time

            obj_value = self.problem.objective_x(point_proj)
            recorder.record_state(
                obj_value,
                _constraint_violation(self.problem, point_proj, equality_only=False),
                decision=point_proj,
            )
            x = point_proj

            if verbose and iter % 1000 == 0:
                print(f"Iteration {iter}, Objective: {obj_value:.6f}")
            point = point_proj.detach()

            if self.stepsize_rule == 'adaptive':
                if recorder.objective_traj[-1] > recorder.objective_traj[-2]:
                    self.learning_rate = max(self.learning_rate * self.lr_decay, self.convergence_threshold)

            if elapsed_time > self.max_running_time:
                break

        payload = recorder.finalize()
        return point, payload["decision_trajectory"], payload["objective_traj"], payload["violation_traj"], payload["per_iter_time"]

class RadialDualOptimizer(BaseOptimizer):
    def __init__(self, problem, params, hom_map=None):
        """
        Initialize the radial algorithm for QP problems
        min_x 1 - 1/2 * Q^TxQ - p^T x
        s.t.  Ax <= b
        Requirement:
            objective is always positive
            0 is in the interior of the feasible set, (or b>0)
        """
        super().__init__(problem=problem, params=params, hom_map=hom_map)
        self._init_optimization_params(params)
        self.problem = problem
        self.hom_map = hom_map
        self.x_origin = hom_map.center
        self.obj_origin = problem.objective_x(self.x_origin)

        self.Q = problem.Q
        self.p = problem.p

        self.dim = problem.nvar
        self.ncon = problem.ncon

    def _init_optimization_params(self, params):
        """Initialize all optimization-related parameters"""
        # Base parameters
        self.learning_rate = params['learning_rate']
        self.max_running_time = params['max_running_time']
        self.max_iterations = params['max_iterations']
        self.convergence_threshold = params['convergence_threshold']
        self.stepsize_rule = _validate_stepsize_rule(params['stepsize_rule'])
        self.acceleration_method, self.acceleration_space = _resolve_acceleration_config(params, default_space="x")
        self.acceleration_beta = float(params.get('momentum', 0.0))
        self.lr_decay = params['lr_decay']
        self.smooth = params['smooth']
        self.convergence_metric = params.get('convergence_metric', 'objective_change')
        self.objective_change_threshold = params.get('objective_change_threshold', self.convergence_threshold)
        self.min_iterations = int(params.get('min_iterations', 1))
        self.eta = 0.0001

    def radial_point(self, x, u):
        """Compute radial point (y, v) from (x, u)"""
        y = (x-self.x_origin) / u + self.x_origin
        v = 1 / u
        return y, v

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        """
        Optimize using radial dual algorithm
        """
        if initial_point is None:
            if seed is not None:
                torch.manual_seed(seed)
            x = _randn_problem_row(self.problem, self.dim)
        else:
            x = _as_problem_row(self.problem, initial_point)

        recorder = IterationRecorder(track_decisions=self.problem.nvar == 2)
        recorder.record_state(
            self.problem.objective_x(x),
            self.problem.constraint_x(x).max().view(1, -1),
            decision=x,
        )

        y, _ = self.radial_point(x, self.problem.radial_primal_obj(x, self.hom_map))
        y_velocity = torch.zeros_like(y)
        elapsed_time = 0.0
        for iter in tqdm(range(self.max_iterations)):
            st = time.time()
            if self.acceleration_method == "nag":
                y_mid = _nag_lookahead(y, y_velocity, self.acceleration_beta)
            else:
                y_mid = y
            grad = self.problem.radial_dual_objective_gradient(y_mid, self.hom_map, method="autograd")

            lr = self.learning_rate
            if self.stepsize_rule == 'diminish':
                lr /= (iter+1)**0.5  # Default step size if dual_opt not provided
            y_next = y_mid - lr * grad
            if self.acceleration_method == "nag":
                y_velocity = (y_next - y).detach()
            y = y_next
            et = time.time()
            iter_time = et - st
            recorder.append_iter_time(iter_time)
            elapsed_time += iter_time

            x, _ = self.radial_point(y, self.problem.radial_dual_objective(y, self.hom_map))
            recorder.record_state(
                self.problem.objective_x(x),
                self.problem.constraint_x(x).max().view(1, -1),
                decision=x,
            )

            if self.stepsize_rule == 'adaptive':
                if recorder.objective_traj[-1] > recorder.objective_traj[-2]:
                    self.learning_rate = max(self.learning_rate * self.lr_decay, self.convergence_threshold)

            if elapsed_time > self.max_running_time:
                break
            if str(self.convergence_metric).lower() == "objective_change":
                if iter + 1 >= self.min_iterations:
                    objective_change = abs(
                        float(recorder.objective_traj[-1].detach().mean().item())
                        - float(recorder.objective_traj[-2].detach().mean().item())
                    )
                    if objective_change <= float(self.objective_change_threshold):
                        break
            elif self.convergence_metric is not None:
                raise ValueError(f"Unsupported convergence_metric: {self.convergence_metric}")
        st = time.time()
        x_opt,_ = self.radial_point(y, self.problem.radial_dual_objective(y, self.hom_map))
        et = time.time()
        last_trans_time = et - st
        payload = recorder.finalize()
        return x_opt, payload["decision_trajectory"], payload["objective_traj"], payload["violation_traj"], payload["per_iter_time"], last_trans_time
    
