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
from homopt.optim.core.verbose import write_iteration_row as _write_verbose_iteration_row

class RadialDualOptimizer(BaseOptimizer):
    def __init__(self, problem, params, hom_map=None):
        """
        Radial-dual descent on the generalized gauge radial objective.

        The linear-constraint radial-dual method uses a linear gauge term for
        the feasible set. This implementation uses the active hom_map gauge so
        the same radial-dual update can be applied to the convex set represented
        by the current gauge map.
        """
        super().__init__(problem=problem, params=params, hom_map=hom_map)
        if hom_map is None:
            raise ValueError("RadialDualOptimizer requires hom_map.")
        self._init_optimization_params(params)
        self.problem = problem
        self.hom_map = hom_map
        self.x_origin = hom_map.center

    def _init_optimization_params(self, params):
        """Initialize all optimization-related parameters"""
        self.learning_rate = params['learning_rate']
        self.max_running_time = params['max_running_time']
        self.max_iterations = params['max_iterations']
        self.convergence_threshold = params['convergence_threshold']
        self.stepsize_rule = _validate_stepsize_rule(params['stepsize_rule'])
        self.acceleration_method, self.acceleration_space = _resolve_acceleration_config(params, default_space="x")
        self.acceleration_beta = float(params.get('momentum', 0.0))
        self.lr_decay = params['lr_decay']
        self.min_lr = params['min_lr']
        self.verbose_interval = int(params.get("verbose_interval", 50))
        self.smooth = params['smooth']
        self.convergence_metric = params.get('convergence_metric', 'objective_change')
        self.objective_change_threshold = params.get('objective_change_threshold', self.convergence_threshold)
        self.min_iterations = int(params.get('min_iterations', 1))
        self.eta = 0.0001

    def radial_point(self, x, u):
        """Map radial-dual variable x back to the primal point x / u around the center."""
        y = (x - self.x_origin) / u + self.x_origin
        v = 1 / u
        return y, v

    def optimize(self, initial_point=None, verbose=False, seed=2025):
        """Optimize using the generalized-gauge radial dual algorithm."""
        if initial_point is None:
            del seed
            x = self.x_origin.detach().clone()
        else:
            x = _as_problem_row(self.problem, initial_point)

        recorder = IterationRecorder(track_decisions=self.problem.nvar == 2)
        recorder.record_state(
            self.problem.objective_x(x),
            self.problem.constraint_x(x).max().view(1, -1),
            decision=x,
        )

        initial_radial_value = self.problem.radial_primal_obj(x, self.hom_map)
        if torch.any(initial_radial_value <= 0):
            raise ValueError("RD requires an initial point with positive radial primal payoff.")
        y, _ = self.radial_point(x, initial_radial_value)
        y_velocity = torch.zeros_like(y)
        elapsed_time = 0.0
        printed_verbose_header = False
        for iter in tqdm(range(self.max_iterations), disable=not verbose):
            st = time.time()
            if self.acceleration_method == "nag":
                y_mid = _nag_lookahead(y, y_velocity, self.acceleration_beta)
            else:
                y_mid = y
            grad = self.problem.radial_dual_objective_gradient(y_mid, self.hom_map, method="autograd")

            lr = self.learning_rate
            if self.stepsize_rule == 'diminish':
                lr = max(lr / (iter + 1) ** 0.5, self.min_lr)
            y_next = y_mid - lr * grad
            if self.acceleration_method == "nag":
                y_velocity = (y_next - y).detach()
            y = y_next
            et = time.time()
            iter_time = et - st
            recorder.append_iter_time(iter_time)
            elapsed_time += iter_time

            x, _ = self.radial_point(y, self.problem.radial_dual_objective(y, self.hom_map))
            split = _constraint_violation_split(self.problem, x)
            objective = self.problem.objective_x(x)
            recorder.record_state(
                objective,
                split["full_violation"],
                decision=x,
            )
            printed_verbose_header = _write_verbose_iteration_row(
                tqdm.write,
                enabled=verbose,
                printed_header=printed_verbose_header,
                iteration=iter,
                interval=self.verbose_interval,
                eval_payload={
                    "objective": objective,
                    **split,
                },
                learning_rate=lr,
                iter_time=iter_time,
            )

            if self.stepsize_rule == 'adaptive':
                if recorder.objective_traj[-1] > recorder.objective_traj[-2]:
                    self.learning_rate = max(self.learning_rate * self.lr_decay, self.min_lr)

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
        x_opt, _ = self.radial_point(y, self.problem.radial_dual_objective(y, self.hom_map))
        et = time.time()
        last_trans_time = et - st
        payload = recorder.finalize()
        return OptimizerRunResult(
            final_decision=x_opt,
            decision_trajectory=payload["decision_trajectory"],
            objective_trajectory=payload["objective_traj"],
            violation_trajectory=payload["violation_traj"],
            per_iter_time=payload["per_iter_time"],
            last_trans_time=last_trans_time,
            include_transform_time=True,
        )




__all__ = ["RadialDualOptimizer"]
