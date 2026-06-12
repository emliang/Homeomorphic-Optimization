"""First-order optimizer implementations."""

from __future__ import annotations

import copy
import numpy as np
import torch
import time
from tqdm import tqdm

from homopt.optim.base import BaseOptimizer
from homopt.optim.first_order_loop import run_first_order_loop
from homopt.optim.recording import IterationRecorder
from homopt.optim.common import *
from homopt.problems import ConvexOpt, LinearProblem, MaxCutSDP, PolyStarOpt, ProjProblem, ToyStarOpt
from homopt.solvers import solve_exact_result
from homopt.optim.lagrangian import LagrangianOptimizer


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
            progress_disable=not verbose,
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
        self.min_lr = params['min_lr']
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
            min_lr=self.min_lr,
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
        for iter in tqdm(range(self.max_iterations), disable=not verbose):
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
                obj_scalar = float(obj_value.detach().reshape(-1).mean().cpu().item())
                print(f"Iteration {iter}, Objective: {obj_scalar:.6f}")
            point = point_proj.detach()

            if self.stepsize_rule == 'adaptive':
                if recorder.objective_traj[-1] > recorder.objective_traj[-2]:
                    self.learning_rate = max(self.learning_rate * self.lr_decay, self.min_lr)

            if elapsed_time > self.max_running_time:
                break

        payload = recorder.finalize()
        return point, payload["decision_trajectory"], payload["objective_traj"], payload["violation_traj"], payload["per_iter_time"]

    


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
        for iter in tqdm(range(self.max_iterations), disable=not verbose):
            st = time.time()
            if self.acceleration_method == "nag":
                y_mid = _nag_lookahead(y, y_velocity, self.acceleration_beta)
            else:
                y_mid = y
            grad = self.problem.radial_dual_objective_gradient(y_mid, self.hom_map, method="autograd")

            lr = self.learning_rate
            if self.stepsize_rule == 'diminish':
                lr /= (iter + 1) ** 0.5
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
        return (
            x_opt,
            payload["decision_trajectory"],
            payload["objective_traj"],
            payload["violation_traj"],
            payload["per_iter_time"],
            last_trans_time,
        )


__all__ = ["FrankWolfeOptimizer", "HomPGDOptimizer", "PGDOptimizer", "RadialDualOptimizer"]
