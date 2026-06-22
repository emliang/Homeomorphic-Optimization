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
        self.min_lr = params['min_lr']
        self.verbose_interval = int(params.get("verbose_interval", 50))
        self.convergence_metric = params.get('convergence_metric', 'objective_change')
        self.objective_change_threshold = params.get('objective_change_threshold', self.convergence_threshold)
        self.min_iterations = int(params.get('min_iterations', 1))
        if 'projection_subproblem' not in params:
            raise KeyError("PGD requires explicit projection_subproblem config.")
        self.projection_subproblem_params = copy.deepcopy(params['projection_subproblem'])

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
            next_point = self.project_to_set(point_update)
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
            min_lr=self.min_lr,
            convergence_threshold=self.convergence_threshold,
            verbose=verbose,
            verbose_interval=self.verbose_interval,
            progress_disable=not verbose,
            track_decisions=should_track_traj,
            extra_track_names=_CONSTRAINT_SPLIT_TRACK_NAMES,
            convergence_metric=self.convergence_metric,
            objective_change_threshold=self.objective_change_threshold,
            min_iterations=self.min_iterations,
            require_feasible_for_objective_change=True,
        )
        _store_constraint_violation_split_metrics(self, payload)
        return OptimizerRunResult(
            final_decision=point_proj,
            decision_trajectory=payload["decision_trajectory"],
            objective_trajectory=payload["objective_traj"],
            violation_trajectory=payload["violation_traj"],
            per_iter_time=payload["per_iter_time"],
        )

    def project_to_ball(self, z):
        """Project points onto the unit ball."""
        return _project_to_ball(z, self.hom_map.p_norm)

    def project_to_set(self, x_init):
        """Project point onto general set using augmented Lagrangian method."""
        if _constraint_violation(self.problem, x_init, equality_only=False).max() > self.convergence_threshold:
            return _project_problem_point(
                self,
                x_init,
                subproblem_params=self.projection_subproblem_params,
                device=self.device,
                dtype=x_init.dtype,
            )
        else:
            return x_init

    def hom_projection_to_set(self, x):
        """Homomorphic projection onto star-shaped set."""
        if self.problem.constraint_x(x).max() > 0:
            x = self.hom_map.forward(self.project_to_ball(self.hom_map.inverse(x)))
        return x



__all__ = ["PGDOptimizer"]
