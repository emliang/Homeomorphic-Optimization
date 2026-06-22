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
        self.min_lr = params['min_lr']
        self.verbose_interval = int(params.get("verbose_interval", 50))
        self.fw_gap_threshold = params.get('fw_gap_threshold', self.convergence_threshold)
        if 'linearization_subproblem' not in params:
            raise KeyError("FW requires explicit linearization_subproblem config.")
        self.linearization_subproblem_params = copy.deepcopy(params['linearization_subproblem'])

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
        printed_verbose_header = False
        for iter in tqdm(range(self.max_iterations), disable=not verbose):
            st = time.time()
            grad = self.problem.gradient_objective_x(x)
            s = _linear_oracle_point(
                self,
                x,
                grad,
                subproblem_params=self.linearization_subproblem_params,
            )
            update = x - s
            fw_gap = torch.sum(grad * update, dim=-1).abs().max()
            if fw_gap <= float(self.fw_gap_threshold):
                break
            lr = self.learning_rate
            if self.stepsize_rule == 'diminish':
                lr = max(lr / (iter + 1) ** 0.5, self.min_lr)
            point_proj = (x - lr * update).detach()

            et = time.time()
            iter_time = et - st
            recorder.append_iter_time(iter_time)
            elapsed_time += iter_time

            obj_value = self.problem.objective_x(point_proj)
            split = _constraint_violation_split(self.problem, point_proj)
            recorder.record_state(
                obj_value,
                split["full_violation"],
                decision=point_proj,
            )
            x = point_proj

            printed_verbose_header = _write_verbose_iteration_row(
                tqdm.write,
                enabled=verbose,
                printed_header=printed_verbose_header,
                iteration=iter,
                interval=self.verbose_interval,
                eval_payload={
                    "objective": obj_value,
                    **split,
                },
                learning_rate=lr,
                iter_time=iter_time,
            )
            point = point_proj.detach()

            if self.stepsize_rule == 'adaptive':
                if recorder.objective_traj[-1] > recorder.objective_traj[-2]:
                    self.learning_rate = max(self.learning_rate * self.lr_decay, self.min_lr)

            if elapsed_time > self.max_running_time:
                break

        payload = recorder.finalize()
        return OptimizerRunResult(
            final_decision=point,
            decision_trajectory=payload["decision_trajectory"],
            objective_trajectory=payload["objective_traj"],
            violation_trajectory=payload["violation_traj"],
            per_iter_time=payload["per_iter_time"],
        )

    




__all__ = ["FrankWolfeOptimizer"]
