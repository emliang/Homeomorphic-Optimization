"""Lagrangian and penalty optimizer for x-space constrained problems."""

from __future__ import annotations

import time

import torch

from homopt.optim.alm.penalty_loop import run_penalty_outer_loop
from homopt.optim.base import BaseOptimizer
from homopt.optim.core.config import (
    _adaptive_inner_tolerance,
    _build_algorithm_update_backend,
    _nag_lookahead,
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
    _validate_proximal_space,
    _velocity_needs_reset,
)
from homopt.optim.core.constraints import (
    _CONSTRAINT_SPLIT_TRACK_NAMES,
    _constraint_layout,
    _constraint_residual,
    _constraint_violation,
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
    _randn_problem_row,
)
from homopt.problems import LinearProblem, ProjProblem


class LagrangianOptimizer(BaseOptimizer):
    """Augmented Lagrangian / penalty optimizer in the original x-space."""

    def __init__(self, problem, params):
        super().__init__(problem=problem, params=params)
        self.problem = problem
        self.device = self.problem.device
        self._init_optimization_params(params)

    def _init_optimization_params(self, params):
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
        ) = _resolve_first_order_lagrangian_gap_config(params, default_enabled=True)
        self.outer_stepsize_rule, self.inner_stepsize_rule = _resolve_penalty_stepsize_rules(params)
        dual_dim, self.dual_ineq_cons, self.dual_eq_cons = _constraint_layout(self.problem)
        self.dual_var = torch.zeros(1, dual_dim, **_problem_tensor_kwargs(self.problem))
        _init_penalty_family_controls(self, params, proximal_coef_default=1.0)
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
        self.acceleration_method, self.acceleration_space = _resolve_acceleration_config(params, default_space="x")
        self.acceleration_beta = float(params.get('momentum', 0.0))
        self._nag_velocity_x = None
        self.opt = _build_algorithm_update_backend(params, self.acceleration_method)
        self._gradient_lagrangian_x = self.problem.gradient_lagrangian_x
        self._grad_supports_penalty = _supports_keyword(self._gradient_lagrangian_x, "penalty_coef")
        self._grad_supports_proximal = _supports_keyword(self._gradient_lagrangian_x, "proximal_coef")
        self._grad_supports_x_outer = _supports_keyword(self._gradient_lagrangian_x, "x_outer")

    def gradient_lagrangian(self, x, dual_var, x_outer=None):
        kwargs = {}
        if self._grad_supports_penalty:
            kwargs["penalty_coef"] = self.penalty_coef
        if self._grad_supports_proximal:
            kwargs["proximal_coef"] = self.proximal_coef
        if self._grad_supports_x_outer:
            kwargs["x_outer"] = x_outer
        return self._gradient_lagrangian_x(x, dual_var, **kwargs)

    def _original_lagrangian_gradient_x(self, x, dual_var):
        kwargs = {}
        if self._grad_supports_penalty:
            kwargs["penalty_coef"] = None
        if self._grad_supports_proximal:
            kwargs["proximal_coef"] = None
        if self._grad_supports_x_outer:
            kwargs["x_outer"] = None
        return self._gradient_lagrangian_x(x, dual_var, **kwargs)

    def first_order_lagrangian_gap(self, x, dual_state):
        grad = self._original_lagrangian_gradient_x(x, dual_state)
        return torch.amax(torch.abs(grad), dim=-1, keepdim=True).max().view(1, -1)

    def _accelerated_inner_point(self, x):
        if self.acceleration_method != "nag":
            return x
        return _nag_lookahead(x, self._nag_velocity_x, self.acceleration_beta)

    def _inner_proximal_gradient(self, x, x_center):
        if self.inner_proximal_coef == 0:
            return torch.zeros_like(x)
        return self.inner_proximal_coef * (x - x_center)

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        if initial_point is None:
            if seed is not None:
                torch.manual_seed(seed)
            x = _randn_problem_row(self.problem, self.problem.nvar)
        else:
            x = _as_problem_row(self.problem, initial_point)

        dual_state = self.dual_var
        penalty_violation = _constraint_violation(self.problem, x)
        initial_gap = (
            self.first_order_lagrangian_gap(x, dual_state)
            if self.check_first_order_lagrangian_gap
            else torch.zeros(1, 1, **_problem_tensor_kwargs(self.problem))
        )
        initial_split = _constraint_violation_split(self.problem, x)
        initial_eval = {
            "objective": self.problem.objective_x(x),
            "violation": _constraint_violation(self.problem, x),
            "decision": x,
            "first_order_lagrangian_gap": initial_gap,
            **initial_split,
        }

        def _run_inner_loop(x_current, dual_state, outer_lr, outer_iter):
            del outer_iter
            inner_lr = float(self.inner_learning_rate if self.inner_learning_rate is not None else outer_lr)
            base_inner_lr = inner_lr
            x_outer = x_current.clone().detach()
            error = torch.tensor(float("inf"), device=self.problem.device)
            best_violation = torch.tensor(float("inf"), device=self.problem.device)
            x_iter = x_current
            inner_tolerance = _adaptive_inner_tolerance(
                _constraint_violation(self.problem, x_iter),
                tol_factor=self.inner_tol_factor,
                tol_min=self.inner_tol_min,
            )
            if self.inner_restart:
                _reset_update_backend(self.opt)
            if self.acceleration_method == "nag" and (self.inner_restart or _velocity_needs_reset(self._nag_velocity_x, x_iter)):
                self._nag_velocity_x = torch.zeros_like(x_iter)

            def _step_once(x_base, inner_iter, inner_center=None):
                nonlocal inner_lr, best_violation
                step_origin = self._accelerated_inner_point(x_base)
                grad = self.gradient_lagrangian(
                    step_origin,
                    dual_state,
                    x_outer=x_outer if self.use_proximal and self.proximal_space == "x" else None,
                )
                if inner_center is not None:
                    grad = grad + self._inner_proximal_gradient(step_origin, inner_center)
                update = self.opt.step(grad)
                x_next = (step_origin - inner_lr * update).detach()
                if self.acceleration_method == "nag":
                    self._nag_velocity_x = (x_next - x_base).detach()
                current_error = torch.norm(x_next - x_base, p=2, dim=-1)
                if self.inner_stepsize_rule == "adaptive":
                    current_violation = _constraint_violation(self.problem, x_next)
                    previous_best_violation = best_violation
                    best_violation = torch.minimum(best_violation, current_violation)
                    inner_lr = _update_penalty_inner_learning_rate(
                        base_inner_lr,
                        inner_lr,
                        stepsize_rule=self.inner_stepsize_rule,
                        lr_decay=self.inner_lr_decay,
                        min_lr=self.inner_min_lr,
                        current_error=float(current_violation.detach().max().item()),
                        best_error=float(previous_best_violation.detach().max().item()),
                        iteration=inner_iter,
                    )
                return x_next, current_error

            if self.uses_inner_proximal_stages:
                inner_iter = 0
                for prox_iter in range(self.inner_proximal_update_iterations):
                    if self.inner_restart:
                        _reset_update_backend(self.opt)
                        if self.acceleration_method == "nag":
                            self._nag_velocity_x = torch.zeros_like(x_iter)
                    inner_center = x_iter.clone().detach()
                    for local_iter in range(self.inner_proximal_steps):
                        inner_iter = prox_iter * self.inner_proximal_steps + local_iter
                        x_iter, error = _step_once(x_iter, inner_iter, inner_center=inner_center)
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
            return x_iter, error

        def _evaluate_state(x_state):
            split = _constraint_violation_split(self.problem, x_state)
            return {
                "objective": self.problem.objective_x(x_state),
                "violation": _constraint_violation(self.problem, x_state),
                "decision": x_state,
                **split,
            }

        def _augment_eval_payload(x_state, dual_state, eval_payload):
            del eval_payload
            if not self.check_first_order_lagrangian_gap:
                return None
            return {"first_order_lagrangian_gap": self.first_order_lagrangian_gap(x_state, dual_state)}

        def _should_stop(eval_payload, error, outer_iter):
            del outer_iter
            first_order_gap = eval_payload.get("first_order_lagrangian_gap")
            first_order_ready = (
                True
                if not self.check_first_order_lagrangian_gap
                else first_order_gap is not None
                and float(first_order_gap.detach().max().item()) < self.first_order_lagrangian_gap_threshold
            )
            return (
                eval_payload["violation"] < self.convergence_threshold
                and error < self.convergence_threshold
                and first_order_ready
            )

        def _update_dual_state(dual_state, x_state, outer_iter):
            nonlocal penalty_violation
            del outer_iter
            residual = _constraint_residual(self.problem, x_state)
            if self.use_lagrangian:
                dual_state = dual_state + _dual_update_scale(self) * residual
            current_penalty_violation = _constraint_violation(self.problem, x_state)
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
                min_lr=self.min_lr,
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
            track_decisions=bool(self.params.get("track_decisions", self.problem.nvar == 2)),
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
        else:
            x = x.to(self.problem.device)
        if "outer_iter_time" not in payload:
            raise KeyError("Penalty/ALM payload must include explicit outer_iter_time.")
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
        )


__all__ = ["LagrangianOptimizer"]
