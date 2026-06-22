"""First-order optimizer implementations."""

from __future__ import annotations

import copy
import numpy as np
import torch
import time
from tqdm import tqdm

from homopt.optim.base import BaseOptimizer
from homopt.optim.first_order.loop import run_first_order_loop
from homopt.optim.core import *

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
        self.min_lr = params['min_lr']
        self.verbose_interval = int(params.get("verbose_interval", 50))
        self.opt = _build_algorithm_update_backend(params, self.acceleration_method)
        self.hom_map_gradient = _validate_hom_map_gradient(params.get('hom_map_gradient', 'explicit'))
        self._problem_grad_z_supports_method = _supports_keyword(self.problem.gradient_objective_z, "method")
        self._problem_grad_z_supports_x = _supports_keyword(self.problem.gradient_objective_z, "x")
        self._problem_grad_z_supports_hom_state = _supports_keyword(self.problem.gradient_objective_z, "hom_state")
        self._hom_forward_fn = self.hom_map.forward
        self._hom_forward_supports_method = _supports_keyword(self._hom_forward_fn, "method")
        self._hom_forward_supports_return_state = _supports_keyword(self._hom_forward_fn, "return_state")
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

    def _hom_forward(self, z):
        if self._hom_forward_supports_method:
            return self._hom_forward_fn(z, method=self.hom_map_gradient)
        return self._hom_forward_fn(z)

    def _hom_forward_with_state(self, z):
        if self._hom_forward_supports_return_state:
            return self._hom_forward_fn(z, method=self.hom_map_gradient, return_state=True)
        return self._hom_forward(z), None

    def _proximal_gradient_z(self, z, z_center, x_center, x_current=None, hom_state_current=None):
        if self.proximal_coef == 0:
            return torch.zeros_like(z)
        if self.proximal_space == "z":
            return self.proximal_coef * (z - z_center)
        if not hasattr(self.hom_map, "vjp"):
            raise NotImplementedError("Hom-PGD x-space proximal gradient requires hom_map.vjp.")
        if x_current is None or hom_state_current is None:
            x_current, hom_state_current = self._hom_forward_with_state(z)
        grad_x = self.proximal_coef * (x_current - x_center)
        return self.hom_map.vjp(z, grad_x, method=self.hom_map_gradient, state=hom_state_current).detach()

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
            x_mid, hom_state_mid = self._hom_forward_with_state(mid_point)
            kwargs = {"hom_map_method": self.hom_map_gradient}
            if self._problem_grad_z_supports_method:
                kwargs["method"] = self.hom_map_gradient
            if self._problem_grad_z_supports_x:
                kwargs["x"] = x_mid
            if self._problem_grad_z_supports_hom_state:
                kwargs["hom_state"] = hom_state_mid
            grad = self.problem.gradient_objective_z(mid_point, self.hom_map, **kwargs)
            if self.use_proximal and self.proximal_space == "x":
                grad = grad + self._proximal_gradient_z(
                    mid_point,
                    self._hom_pgd_prox_z_center,
                    self._hom_pgd_prox_x_center,
                    x_current=x_mid,
                    hom_state_current=hom_state_mid,
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
            min_lr=self.min_lr,
            convergence_threshold=self.convergence_threshold,
            verbose=verbose,
            verbose_interval=self.verbose_interval,
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
        return OptimizerRunResult(
            final_decision=x_opt,
            decision_trajectory=payload["decision_trajectory"],
            objective_trajectory=payload["objective_traj"],
            violation_trajectory=payload["violation_traj"],
            per_iter_time=payload["per_iter_time"],
            last_trans_time=last_trans_time,
            latent_trajectory=payload.get("z_trajectory"),
            include_transform_time=True,
        )

    def project_to_ball(self, z):
        """Project points onto the unit ball."""
        return _project_to_ball(z, self.hom_map.p_norm)




__all__ = ["HomPGDOptimizer"]
