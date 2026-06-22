"""Differentiable power-flow completion for AC-OPF partial predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ACOPFPowerFlowCompletionConfig:
    max_iters: int = 8
    tol: float = 1e-6
    damping: float = 1e-5
    allocation: str = "soft_cost"
    soft_tau: float = 1.0
    gradient_mode: str = "unrolled"


@dataclass(frozen=True)
class ACOPFPowerFlowPartialLayout:
    pg_gen_idx: torch.Tensor
    vm_bus_idx: torch.Tensor
    vm_newton_bus_idx: torch.Tensor
    va_newton_bus_idx: torch.Tensor
    newton_eq_idx: torch.Tensor
    newton_voltage_col_idx: torch.Tensor
    newton_y_idx: torch.Tensor
    slack_bus_idx: torch.Tensor
    generator_bus_idx: torch.Tensor

    @property
    def partial_dim(self):
        return int(self.pg_gen_idx.numel() + self.vm_bus_idx.numel())

    @property
    def n_pg_partial(self):
        return int(self.pg_gen_idx.numel())


def build_power_flow_partial_layout(problem):
    """Build the partial-prediction layout used by differentiable PF completion."""

    device = problem.device
    pv_gen = problem.pv_gen_idx.to(device=device)
    spv_bus = problem.spv_idx.to(device=device)
    pq_bus = problem.pq_idx.to(device=device)
    non_slack_bus = problem.non_slack_bus_idx.to(device=device)
    newton_eq_idx = torch.cat([non_slack_bus, problem.nb + pq_bus])
    newton_voltage_col_idx = torch.cat([pq_bus, problem.nb + non_slack_bus])
    newton_y_idx = torch.cat([problem.vm_start_yidx + pq_bus, problem.va_start_yidx + non_slack_bus])
    return ACOPFPowerFlowPartialLayout(
        pg_gen_idx=pv_gen,
        vm_bus_idx=spv_bus,
        vm_newton_bus_idx=pq_bus,
        va_newton_bus_idx=non_slack_bus,
        newton_eq_idx=newton_eq_idx,
        newton_voltage_col_idx=newton_voltage_col_idx,
        newton_y_idx=newton_y_idx,
        slack_bus_idx=problem.slack_idx.to(device=device),
        generator_bus_idx=spv_bus,
    )


class DifferentiablePowerFlowSolver:
    """Unrolled Newton solver for AC-OPF partial predictions.

    Partial decisions contain active power for all generators on PV buses and
    voltage magnitudes for slack/PV buses. Newton solves non-slack voltage
    angles and PQ-bus voltage magnitudes. Slack active generation and generator
    bus reactive generation are then computed from power-balance residuals and
    distributed to generators by a differentiable allocation policy.
    """

    def __init__(self, problem, config: ACOPFPowerFlowCompletionConfig | dict | None = None):
        self.problem = problem
        if config is None:
            config = ACOPFPowerFlowCompletionConfig()
        elif isinstance(config, dict):
            config = ACOPFPowerFlowCompletionConfig(**config)
        self.config = config
        if self.config.allocation != "soft_cost":
            raise ValueError(f"Unsupported differentiable AC-OPF allocation: {self.config.allocation}")
        if self.config.gradient_mode not in {"unrolled", "last_step", "implicit"}:
            raise ValueError(f"Unsupported AC-OPF PF gradient mode: {self.config.gradient_mode}")
        self.layout = build_power_flow_partial_layout(problem)

    @property
    def partial_dim(self):
        return self.layout.partial_dim

    def partial_bounds(self):
        lower = torch.cat(
            [
                self.problem.pmin[self.layout.pg_gen_idx],
                self.problem.vmin[self.layout.vm_bus_idx],
            ]
        )
        upper = torch.cat(
            [
                self.problem.pmax[self.layout.pg_gen_idx],
                self.problem.vmax[self.layout.vm_bus_idx],
            ]
        )
        return lower, upper

    def scale_partial(self, normalized_partial):
        normalized_partial = self.problem._as_batch_tensor(normalized_partial)
        lower, upper = self.partial_bounds()
        return 0.5 * (normalized_partial + 1.0) * (upper - lower) + lower

    def inverse_scale_partial(self, partial):
        partial = self.problem._as_batch_tensor(partial)
        lower, upper = self.partial_bounds()
        denom = torch.clamp(upper - lower, min=torch.finfo(partial.dtype).eps)
        return 2.0 * (partial - lower) / denom - 1.0

    def extract_partial(self, y):
        pg, _, vm, _ = self.problem.get_yvars(y)
        return torch.cat([pg[:, self.layout.pg_gen_idx], vm[:, self.layout.vm_bus_idx]], dim=1)

    def complete(self, input_params, partial):
        input_params = self.problem._as_batch_tensor(input_params)
        partial = self.problem._as_batch_tensor(partial)
        if partial.shape[-1] != self.partial_dim:
            raise ValueError(f"Expected AC-OPF partial width {self.partial_dim}, got {partial.shape[-1]}.")

        y = self._initial_y(input_params, partial)
        y = self._newton_solve(input_params, partial, y)
        return self._complete_slack_pg_and_qg(input_params, y)

    def partial_y_idx(self):
        return torch.cat(
            [
                self.layout.pg_gen_idx,
                self.problem.vm_start_yidx + self.layout.vm_bus_idx,
            ]
        )

    def _initial_y(self, input_params, partial):
        y = self.problem.output_init.to(device=input_params.device, dtype=input_params.dtype).expand(
            input_params.shape[0],
            -1,
        ).clone()
        n_pg = self.layout.n_pg_partial
        y[:, self.layout.pg_gen_idx] = partial[:, :n_pg]
        y[:, self.problem.vm_start_yidx + self.layout.vm_bus_idx] = partial[:, n_pg:]
        if self.problem.enforce_slack_angle and self.problem.slack_idx.numel():
            y[:, self.problem.va_start_yidx + self.problem.slack_idx] = self.problem.slack_angle
        return y

    def _newton_solve(self, input_params, partial, y):
        if self.config.gradient_mode == "implicit":
            return _ImplicitNewtonSolve.apply(partial, input_params, self)
        if self.config.gradient_mode == "last_step":
            return self._newton_solve_last_step(input_params, partial, y)
        return self._newton_solve_unrolled(input_params, y)

    def _newton_solve_unrolled(self, input_params, y):
        eye = torch.eye(
            int(self.layout.newton_eq_idx.numel()),
            device=y.device,
            dtype=y.dtype,
        ).unsqueeze(0)
        for _ in range(int(self.config.max_iters)):
            residual = self.problem.power_balance_residual(input_params, y)[:, self.layout.newton_eq_idx]
            if float(residual.detach().abs().max().cpu()) < float(self.config.tol):
                break
            jac_voltage = self.problem.power_balance_jacobian_voltage(y)
            jac = jac_voltage[:, self.layout.newton_eq_idx, :][:, :, self.layout.newton_voltage_col_idx]
            delta = torch.linalg.solve(jac + float(self.config.damping) * eye, residual.unsqueeze(-1)).squeeze(-1)
            y = y.clone()
            y[:, self.layout.newton_y_idx] = y[:, self.layout.newton_y_idx] - delta
        return y

    def _newton_solve_last_step(self, input_params, partial, y):
        max_iters = int(self.config.max_iters)
        if max_iters <= 0:
            return y
        with torch.no_grad():
            y_state = y.detach()
            for _ in range(max_iters - 1):
                residual = self.problem.power_balance_residual(input_params, y_state)[:, self.layout.newton_eq_idx]
                if float(residual.detach().abs().max().cpu()) < float(self.config.tol):
                    break
                jac_voltage = self.problem.power_balance_jacobian_voltage(y_state)
                jac = jac_voltage[:, self.layout.newton_eq_idx, :][:, :, self.layout.newton_voltage_col_idx]
                eye = torch.eye(jac.shape[-1], device=y.device, dtype=y.dtype).unsqueeze(0)
                delta = torch.linalg.solve(jac + float(self.config.damping) * eye, residual.unsqueeze(-1)).squeeze(-1)
                y_state = y_state.clone()
                y_state[:, self.layout.newton_y_idx] = y_state[:, self.layout.newton_y_idx] - delta

        y_live = y_state.detach().clone()
        n_pg = self.layout.n_pg_partial
        y_live[:, self.layout.pg_gen_idx] = partial[:, :n_pg]
        y_live[:, self.problem.vm_start_yidx + self.layout.vm_bus_idx] = partial[:, n_pg:]
        if self.problem.enforce_slack_angle and self.problem.slack_idx.numel():
            y_live[:, self.problem.va_start_yidx + self.problem.slack_idx] = self.problem.slack_angle
        return self._newton_solve_unrolled(input_params, y_live) if max_iters == 1 else self._newton_step(input_params, y_live)

    def _newton_step(self, input_params, y):
        residual = self.problem.power_balance_residual(input_params, y)[:, self.layout.newton_eq_idx]
        jac_voltage = self.problem.power_balance_jacobian_voltage(y)
        jac = jac_voltage[:, self.layout.newton_eq_idx, :][:, :, self.layout.newton_voltage_col_idx]
        eye = torch.eye(jac.shape[-1], device=y.device, dtype=y.dtype).unsqueeze(0)
        delta = torch.linalg.solve(jac + float(self.config.damping) * eye, residual.unsqueeze(-1)).squeeze(-1)
        y = y.clone()
        y[:, self.layout.newton_y_idx] = y[:, self.layout.newton_y_idx] - delta
        return y

    def _complete_slack_pg_and_qg(self, input_params, y):
        residual = self.problem.power_balance_residual(input_params, y)
        pg, qg, _, _ = self.problem.get_yvars(y)
        p_aggregate = _aggregate_by_bus(pg, self.problem.gen_bus_idx, self.problem.nb)
        q_aggregate = _aggregate_by_bus(qg, self.problem.gen_bus_idx, self.problem.nb)

        y = y.clone()
        if self.layout.slack_bus_idx.numel():
            target_p = p_aggregate[:, self.layout.slack_bus_idx] - residual[:, self.layout.slack_bus_idx]
            updated_pg = _allocate_generator_bus_totals(
                problem=self.problem,
                current=pg,
                bus_idx=self.layout.slack_bus_idx,
                totals=target_p,
                lower=self.problem.pmin,
                upper=self.problem.pmax,
                costs=_active_power_cost_key(self.problem, pg),
                tau=float(self.config.soft_tau),
            )
            y[:, : self.problem.ng] = updated_pg

        q_eq_idx = self.problem.nb + self.layout.generator_bus_idx
        target_q = q_aggregate[:, self.layout.generator_bus_idx] - residual[:, q_eq_idx]
        updated_qg = _allocate_generator_bus_totals(
            problem=self.problem,
            current=y[:, self.problem.qg_start_yidx : self.problem.vm_start_yidx],
            bus_idx=self.layout.generator_bus_idx,
            totals=target_q,
            lower=self.problem.qmin,
            upper=self.problem.qmax,
            costs=_active_power_cost_key(self.problem, y[:, : self.problem.ng]),
            tau=float(self.config.soft_tau),
        )
        y[:, self.problem.qg_start_yidx : self.problem.vm_start_yidx] = updated_qg
        return y


class _ImplicitNewtonSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial, input_params, solver):
        problem = solver.problem
        with torch.no_grad():
            y = solver._initial_y(input_params, partial)
            y = solver._newton_solve_unrolled(input_params, y)
            jac_full = problem.power_balance_jacobian_y(y)[:, solver.layout.newton_eq_idx, :]
            jac_newton = jac_full[:, :, solver.layout.newton_y_idx]
            jac_partial = jac_full[:, :, solver.partial_y_idx()]
            eye = torch.eye(jac_newton.shape[-1], device=y.device, dtype=y.dtype).unsqueeze(0)
            jac_newton = jac_newton + float(solver.config.damping) * eye
        ctx.save_for_backward(jac_newton, jac_partial, solver.partial_y_idx(), solver.layout.newton_y_idx)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        jac_newton, jac_partial, partial_y_idx, newton_y_idx = ctx.saved_tensors
        direct = grad_y[:, partial_y_idx]
        newton_grad = grad_y[:, newton_y_idx]
        implicit_adj = torch.linalg.solve(jac_newton.transpose(1, 2), newton_grad.unsqueeze(-1)).squeeze(-1)
        indirect = -torch.bmm(implicit_adj.unsqueeze(1), jac_partial).squeeze(1)
        return direct + indirect, None, None


def _aggregate_by_bus(values, bus_idx, nb):
    out = values.new_zeros(values.shape[0], int(nb))
    out.index_add_(1, bus_idx, values)
    return out


def _active_power_cost_key(problem, pg):
    pg_mw = pg * float(problem.baseMVA)
    return 2.0 * problem.quad_costs.to(device=pg.device, dtype=pg.dtype) * pg_mw + problem.lin_costs.to(
        device=pg.device,
        dtype=pg.dtype,
    )


def _allocate_generator_bus_totals(*, problem, current, bus_idx, totals, lower, upper, costs, tau):
    del upper
    allocation = current.clone()
    tau = max(float(tau), 1e-8)
    for col, bus in enumerate(bus_idx.tolist()):
        gen_mask = problem.gen_bus_idx == int(bus)
        gen_idx = torch.nonzero(gen_mask, as_tuple=False).view(-1).to(device=current.device)
        if gen_idx.numel() == 0:
            continue
        if gen_idx.numel() == 1:
            allocation[:, gen_idx[0]] = totals[:, col]
            continue
        lower_sub = lower[gen_idx].to(device=current.device, dtype=current.dtype)
        remaining = totals[:, col] - lower_sub.sum()
        weights = torch.softmax(-costs[:, gen_idx] / tau, dim=1)
        allocation[:, gen_idx] = lower_sub.unsqueeze(0) + weights * remaining.unsqueeze(1)
    return allocation


__all__ = [
    "ACOPFPowerFlowCompletionConfig",
    "ACOPFPowerFlowPartialLayout",
    "DifferentiablePowerFlowSolver",
    "build_power_flow_partial_layout",
]
