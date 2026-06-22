"""Homeomorphic augmented Lagrangian optimizer implementation."""

from __future__ import annotations

import torch
import time
import numpy as np

from homopt.optim.base import BaseOptimizer
from homopt.optim.alm.penalty_loop import run_penalty_outer_loop
from homopt.optim.core.config import (
    _adaptive_inner_tolerance,
    _build_algorithm_update_backend,
    _hom_nag_lookahead,
    _resolve_acceleration_config,
    _resolve_first_order_lagrangian_gap_config,
    _resolve_inner_restart,
    _resolve_inner_stopping_config,
    _resolve_penalty_lr_decays,
    _resolve_penalty_min_lrs,
    _resolve_penalty_stepsize_rules,
    _reset_update_backend,
    _should_stop_inner_loop,
    _supports_keyword,
    _validate_hom_map_gradient,
    _validate_proximal_space,
    _velocity_needs_reset,
)
from homopt.optim.core.constraints import (
    _CONSTRAINT_SPLIT_TRACK_NAMES,
    _constraint_layout,
    _constraint_residual,
    _constraint_violation_split,
)
from homopt.optim.core.penalty import (
    _clamp_dual_state,
    _dual_update_scale,
    _init_penalty_family_controls,
    _maybe_update_penalty,
    _resolve_penalty_inner_solver_config,
    _update_penalty_inner_learning_rate,
    _update_penalty_outer_learning_rate,
)
from homopt.optim.core.result import (
    OptimizerRunResult,
    _store_constraint_violation_split_metrics,
    _store_first_order_lagrangian_gap_metrics,
)
from homopt.optim.core.tensors import (
    _as_problem_row,
    _problem_tensor_kwargs,
    _project_to_ball,
    _randn_problem_row,
)
from homopt.problems import LinearProblem, ProjProblem


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
        self.min_lr, self.inner_min_lr = _resolve_penalty_min_lrs(params)
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
                        min_lr=self.inner_min_lr,
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
                min_lr=self.min_lr,
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
        return OptimizerRunResult(
            final_decision=x,
            decision_trajectory=payload["decision_trajectory"],
            objective_trajectory=payload["objective_traj"],
            violation_trajectory=payload["violation_traj"],
            per_iter_time=payload["per_iter_time"],
            last_trans_time=last_trans_time,
            latent_trajectory=payload.get("z_trajectory"),
            include_transform_time=True,
        )



__all__ = ["HomALMOptimizer"]
