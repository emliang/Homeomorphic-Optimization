"""INN latent-space optimization routines."""

from __future__ import annotations

import torch

from homopt.learning.bisection import homeomorphic_bisection
from homopt.models.inn import (
    _build_quadratic_terms,
    _embed_condition_if_available,
    _forward_with_condition,
    _quadratic_objective,
)
from homopt.optim.core.config import _validate_stepsize_rule
from homopt.optim.core.result import ParametricOptimizerRunResult
from homopt.optim.core.updates import AdamOptimizer, GDOptimizer
from homopt.optim.first_order.loop import run_first_order_loop


def pgd_transformed_space(model, data, input_params, args, initial_z=None):
    model.eval()
    batch_size = input_params.shape[0]
    device = input_params.device
    dtype = input_params.dtype
    max_iter = args.get('pgd_max_iter', 200)
    lr = args.get('pgd_lr', 0.001)
    z = (
        torch.zeros(batch_size, data.nvar, device=device, dtype=dtype)
        if initial_z is None
        else initial_z.clone().detach().to(device=device, dtype=dtype)
    )
    z.requires_grad_(True)
    optimizer = torch.optim.SGD([z], lr=lr, momentum=0.0)
    lr_decay = float(args["lr_decay"])
    min_lr = float(args["min_lr"])
    current_lr = lr
    proj_eps = float(args.get('proj_eps', 1e-5))
    best_objective = float('inf')
    patience = 0
    lr_patience = 1
    trace_every = max(int(args.get('trace_every', 1)), 1)
    objective_history, violation_history, trajectory_data = [], [], []
    fixed_Q, fixed_p = _build_quadratic_terms(data, device, dtype)
    condition_emb = _embed_condition_if_available(model, input_params, batch_size, detach=True)
    for iteration in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        u_mapped = _forward_with_condition(model, z, input_params, condition_emb)
        y_partial = data.scale(input_params, u_mapped)
        y_full = data.complete_partial(input_params, y_partial)
        objective = _quadratic_objective(y_full, fixed_Q, fixed_p)
        with torch.inference_mode():
            violations = data.check_feasibility(input_params, y_full)
            max_violations = torch.amax(violations, dim=1)
        if (iteration % trace_every) == 0:
            trajectory_data.append({
                'iteration': iteration,
                'z': z.detach().cpu().numpy().copy(),
                'y': y_full.detach().cpu().numpy().copy(),
                'x': y_full.detach().cpu().numpy().copy(),
                'objective': objective.detach().cpu().numpy().copy(),
                'violation': max_violations.detach().cpu().numpy().copy(),
            })
        loss = objective.sum()
        grad_z = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
        z.grad = grad_z
        optimizer.step()
        with torch.inference_mode():
            u_update = _forward_with_condition(model, z, input_params, condition_emb)
            y_update = data.complete_partial(input_params, data.scale(input_params, u_update))
            violations_update = data.check_feasibility(input_params, y_update)
            max_violations_update = torch.amax(violations_update, dim=1)
            if torch.any(max_violations_update > proj_eps):
                z_proj, _ = homeomorphic_bisection(
                    model,
                    data,
                    z.detach(),
                    input_params,
                    args,
                    condition_emb=condition_emb,
                )
                z.copy_(z_proj)
        objective_history.append(objective.detach().cpu().numpy())
        violation_history.append(max_violations.detach().cpu().numpy())
        current_obj = objective.mean().item()
        if current_obj < best_objective:
            best_objective = current_obj
            patience = 0
        else:
            patience += 1
        if patience >= lr_patience and current_lr > min_lr:
            current_lr = max(current_lr * lr_decay, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            patience = 0
    with torch.inference_mode():
        u_final = _forward_with_condition(model, z, input_params, condition_emb)
        y_partial = data.scale(input_params, u_final)
        y_full = data.complete_partial(input_params, y_partial)
        final_violations = data.check_feasibility(input_params, y_full)
        final_max_violations = torch.amax(final_violations, dim=1)
        if hasattr(data, 'fixed_Q') and hasattr(data, 'fixed_p'):
            final_objective = _quadratic_objective(y_full, fixed_Q, fixed_p)
        else:
            final_objective = torch.sum(y_full ** 2, dim=1)
    return {
        'z_final': z.detach(),
        'y_final': y_full,
        'x_final': y_full,
        'final_objective': final_objective,
        'final_violations': final_max_violations,
        'objective_history': objective_history,
        'violation_history': violation_history,
        'iterations': max_iter,
        'feasible_solutions': (final_max_violations < 1e-5).sum().item(),
        'trajectory': trajectory_data,
    }


class INNPGDOptimizer:
    def __init__(self, problem, paras, model=None):
        self.problem = problem
        self.model = model
        self._init_optimization_params(paras)
        model_param = next(self.model.parameters(), None) if self.model is not None else None
        self.device = self.problem.device if hasattr(self.problem, 'device') else (
            model_param.device if model_param is not None else torch.device('cpu')
        )
        self.dtype = getattr(self.problem, 'dtype', None)
        if self.dtype is None:
            self.dtype = model_param.dtype if model_param is not None else torch.get_default_dtype()

    def _init_optimization_params(self, paras):
        self.learning_rate = paras.get('learning_rate', 0.001)
        self.max_iterations = paras.get('max_iterations', 200)
        self.max_running_time = paras.get('max_running_time', 300)
        self.convergence_threshold = paras.get('convergence_threshold', 1e-6)
        self.lr_decay = paras.get('lr_decay', 0.9)
        self.min_lr = paras['min_lr']
        self.verbose_interval = int(paras.get('verbose_interval', 50))
        self.eps_converge = paras.get('feasibility_eps', 1e-5)
        self.proj_max_steps = paras.get('proj_max_steps', 20)
        self.proj_step_size = paras.get('step_size', 0.5)
        self.proj_eps = paras.get('proj_eps', 1e-5)
        self.opt_type = paras.get('opt', 'gd')
        self.momentum = paras.get('momentum', 0.0)
        self.stepsize_rule = _validate_stepsize_rule(paras.get('stepsize_rule', 'constant'))

    def optimize(self, initial_point=None, verbose=False, seed=2025, input_params=None, objective_params=None):
        del seed
        if self.model is None:
            raise ValueError('INN model must be provided for INNPGDOptimizer')
        if input_params is None:
            raise ValueError('input_params must be provided for INN optimization')
        self.model.eval()
        input_params = input_params.to(device=self.device, dtype=self.dtype)
        if objective_params is not None:
            objective_params = objective_params.to(device=self.device, dtype=self.dtype)
        if self.opt_type == 'gd':
            opt = GDOptimizer(beta1=self.momentum)
        else:
            opt = AdamOptimizer()
        learning_rate = self.learning_rate
        batch_size = input_params.shape[0]
        should_track_traj = self.problem.nvar == 2
        model_forward = self.model
        scale = self.problem.scale
        complete_partial = self.problem.complete_partial
        violations_fn = self.problem.violations
        condition_emb = _embed_condition_if_available(model_forward, input_params, batch_size, detach=True)

        def _forward_to_full(z_tensor):
            u_tensor = _forward_with_condition(model_forward, z_tensor, input_params, condition_emb)
            y_partial_tensor = scale(input_params, u_tensor)
            return complete_partial(input_params, y_partial_tensor)

        def _objective_value(y_full):
            if objective_params is not None and hasattr(self.problem, "objective_xy"):
                return self.problem.objective_xy(input_params, y_full, objective_params).reshape(-1)
            return self.problem.objective(y_full).reshape(-1)

        if initial_point is None:
            z = torch.zeros(batch_size, self.problem.nvar, device=self.device, dtype=self.dtype)
        else:
            z = torch.as_tensor(initial_point, dtype=self.dtype, device=self.device).view(batch_size, -1)
        z.requires_grad_(True)
        with torch.inference_mode():
            y_current = _forward_to_full(z)
            initial_violation = violations_fn(input_params, y_current).reshape(-1)
            initial_eval = {
                "objective": _objective_value(y_current),
                "violation": initial_violation,
                "eq_violation": torch.zeros(batch_size, device=self.device, dtype=self.dtype),
                "ineq_violation": initial_violation,
                "decision": y_current,
                "latent_trajectory": z,
            }

        def _compute_gradient(z_state, prev_state, iteration):
            del prev_state, iteration
            z_state = z_state.detach().requires_grad_(True)
            u_mapped = _forward_with_condition(model_forward, z_state, input_params, condition_emb)
            y_partial = scale(input_params, u_mapped)
            y_full = complete_partial(input_params, y_partial)
            objective = _objective_value(y_full)
            grad = torch.autograd.grad(objective.sum(), z_state)[0]
            return grad

        def _apply_step(z_state, update, lr, prev_state, iteration):
            del prev_state, iteration
            z_update = z_state - lr * update
            with torch.inference_mode():
                y_update = _forward_to_full(z_update)
                violation_update = violations_fn(input_params, y_update)
                if torch.any(violation_update > self.proj_eps):
                    z_proj, _ = self._homeomorphic_bisection(z_update, input_params, condition_emb=condition_emb)
                else:
                    z_proj = z_update
            return z_proj.detach().clone()

        def _evaluate_state(z_state):
            with torch.inference_mode():
                y_eval = _forward_to_full(z_state)
                violation = violations_fn(input_params, y_eval).reshape(-1)
                return {
                    "objective": _objective_value(y_eval),
                    "violation": violation,
                    "eq_violation": torch.zeros_like(violation),
                    "ineq_violation": violation,
                    "decision": y_eval,
                    "latent_trajectory": z_state,
                }

        z, payload, learning_rate = run_first_order_loop(
            initial_state=z,
            initial_eval=initial_eval,
            update_backend=opt,
            compute_gradient=_compute_gradient,
            apply_step=_apply_step,
            evaluate_state=_evaluate_state,
            max_iterations=self.max_iterations,
            max_running_time=self.max_running_time,
            initial_learning_rate=learning_rate,
            stepsize_rule=self.stepsize_rule,
            lr_decay=self.lr_decay,
            min_lr=self.min_lr,
            convergence_threshold=self.convergence_threshold,
            verbose=verbose,
            verbose_interval=self.verbose_interval,
            progress_disable=not verbose,
            track_decisions=should_track_traj,
            extra_track_names=("latent_trajectory",),
        )
        with torch.inference_mode():
            y_opt = _forward_to_full(z)
        return ParametricOptimizerRunResult(
            final_decision=y_opt,
            decision_trajectory=payload["decision_trajectory"],
            latent_trajectory=payload["latent_trajectory"],
            objective_trajectory=payload["objective_traj"],
            violation_trajectory=payload["violation_traj"],
            per_iter_time=payload["per_iter_time"],
        )

    def _homeomorphic_bisection(self, z_infeasible, input_params, condition_emb=None):
        args = {
            'proj_max_steps': self.proj_max_steps,
            'proj_eps': self.proj_eps,
            'step_size': self.proj_step_size,
        }
        return homeomorphic_bisection(
            self.model,
            self.problem,
            z_infeasible,
            input_params,
            args,
            self.eps_converge,
            condition_emb=condition_emb,
        )


__all__ = ["INNPGDOptimizer", "pgd_transformed_space"]
