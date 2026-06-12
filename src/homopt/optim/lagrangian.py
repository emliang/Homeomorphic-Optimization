"""Lagrangian and equality-constrained ALM optimizer implementations."""

from __future__ import annotations

import numpy as np
import torch
import time
from tqdm import tqdm

from homopt.optim.base import BaseOptimizer
from homopt.optim.penalty import run_penalty_outer_loop
from homopt.optim.recording import IterationRecorder
from homopt.optim.common import *
from homopt.problems import LinearProblem, ProjProblem
from homopt.solvers import solve_exact_result


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
        self.min_lr, self.inner_min_lr = _resolve_penalty_min_lrs(params)
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
                        min_lr=self.inner_min_lr,
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



__all__ = ["EqualityConstrainedALMOptimizer", "LagrangianOptimizer"]
