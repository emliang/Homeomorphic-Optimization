"""Parametric AC-OPF problem instances."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from homopt.problems.base import ParametricProblemBase, ProblemInstanceBatch, TensorRuntimeMixin

from .data import (
    ACOPFDataset,
    build_acopf_input_tensor,
    build_acopf_reference_tensor,
    load_acopf_dataset,
    load_pglib_acopf_dataset,
    normalize_acopf_dataset,
)
from .completion import DifferentiablePowerFlowSolver
from .power_flow import (
    ANGMAX,
    BUS_TYPE,
    COST,
    PD,
    PG,
    PMAX,
    PMIN,
    QD,
    QG,
    QMAX,
    QMIN,
    RATE_A,
    VA,
    VM,
    VMAX,
    VMIN,
    branch_bus_indices,
    generator_bus_indices,
    make_admittance_matrices,
)


class ParametricACOPFProblem(TensorRuntimeMixin, ParametricProblemBase):
    """Dataset-backed deterministic parametric AC-OPF family.

    The parametric input is ``[Pd, Qd] / baseMVA`` and the decision vector is
    ``[Pg, Qg, Vm, Va]`` with ``Pg`` and ``Qg`` in per-unit.
    """

    name = "parametric_acopf"
    has_eq = True

    def __init__(
        self,
        dataset_path=None,
        dataset: ACOPFDataset | dict | None = None,
        *,
        case: int | None = None,
        pglib_case_name: str | None = None,
        pglib_data_dir=None,
        download_if_missing: bool = True,
        dataset_samples: int = 10000,
        load_std: float = 0.05,
        seed: int = 2025,
        branch_constraints: bool = True,
        enforce_slack_angle: bool = True,
        prediction_space: str = "full",
        completion_config: dict | None = None,
        reference_test_size: int = 1024,
        device=None,
        dtype=None,
    ):
        if dataset is None:
            if dataset_path is None:
                if case is None:
                    raise ValueError("ParametricACOPFProblem requires dataset_path, dataset, or case.")
                dataset = load_pglib_acopf_dataset(
                    int(case),
                    samples=int(dataset_samples),
                    seed=int(seed),
                    load_std=float(load_std),
                    case_name=pglib_case_name,
                    data_dir=pglib_data_dir,
                    download_if_missing=download_if_missing,
                )
            else:
                dataset = load_acopf_dataset(dataset_path)
        elif isinstance(dataset, dict):
            dataset = normalize_acopf_dataset(dataset)
        if not isinstance(dataset, ACOPFDataset):
            raise TypeError(f"Unsupported AC-OPF dataset type: {type(dataset).__name__}.")

        self.seed = int(seed)
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = _resolve_dtype(dtype)
        self.dataset_path = str(dataset_path or dataset.source_path or "")
        self.branch_constraints = bool(branch_constraints)
        self.enforce_slack_angle = bool(enforce_slack_angle)
        self.prediction_space = str(prediction_space).strip().lower()
        if self.prediction_space not in {"full", "pf_partial"}:
            raise ValueError(f"Unsupported AC-OPF prediction_space: {prediction_space}")
        self.completion_config = dict(completion_config or {})
        self.reference_test_size = int(reference_test_size)
        self.ppc = dataset.ppc
        self.baseMVA = float(self.ppc["baseMVA"])

        self._load_case_tensors(self.ppc)
        self.input_samples = build_acopf_input_tensor(
            dataset,
            base_mva=self.baseMVA,
            device=self.device,
            dtype=self.dtype,
        )
        self.reference_y = build_acopf_reference_tensor(
            dataset,
            base_mva=self.baseMVA,
            device=self.device,
            dtype=self.dtype,
        )
        if self.reference_y is not None:
            mask = ~torch.isnan(self.reference_y).any(dim=1)
            self.input_samples = self.input_samples[mask]
            self.reference_y = self.reference_y[mask]
        self.fixed_x0 = self.output_init.squeeze(0)
        self.n_constraints = self.neq + self.nineq
        self.power_flow_completion = None
        if self.prediction_space == "pf_partial":
            self.power_flow_completion = DifferentiablePowerFlowSolver(self, self.completion_config)
            self.partial_dim = self.power_flow_completion.partial_dim

    def to_dtype(self, dtype):
        dtype = _resolve_dtype(dtype)
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor) and attr.is_floating_point():
                setattr(self, attr_name, attr.to(dtype=dtype))
        self.dtype = dtype
        return self

    def _load_case_tensors(self, ppc):
        bus = np.asarray(ppc["bus"], dtype=float)
        gen = np.asarray(ppc["gen"], dtype=float)
        branch = np.asarray(ppc["branch"], dtype=float)
        gencost = np.asarray(ppc["gencost"], dtype=float)

        self.nb = int(bus.shape[0])
        self.ng = int(gen.shape[0])
        self.nl = int(branch.shape[0])
        self.xdim = 2 * self.nb
        self.nvar = 2 * self.ng + 2 * self.nb
        self.ydim = self.nvar

        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2 * self.ng
        self.va_start_yidx = 2 * self.ng + self.nb

        self.gen_bus_idx = torch.as_tensor(generator_bus_indices(ppc), device=self.device, dtype=torch.long)
        self.branch_idx = torch.as_tensor(branch_bus_indices(ppc), device=self.device, dtype=torch.long)
        gen_bus_idx_np = generator_bus_indices(ppc)
        pq_idx_np = np.where(bus[:, BUS_TYPE] == 1)[0]
        pv_idx_np = np.where(bus[:, BUS_TYPE] == 2)[0]
        slack_idx_np = np.where(bus[:, BUS_TYPE] == 3)[0]
        non_slack_idx_np = np.setdiff1d(np.arange(self.nb), slack_idx_np)
        self.pq_idx = torch.as_tensor(pq_idx_np, device=self.device, dtype=torch.long)
        self.pv_idx = torch.as_tensor(pv_idx_np, device=self.device, dtype=torch.long)
        self.slack_idx = torch.as_tensor(slack_idx_np, device=self.device, dtype=torch.long)
        self.non_slack_bus_idx = torch.as_tensor(non_slack_idx_np, device=self.device, dtype=torch.long)
        self.non_slack_gen_idx = torch.as_tensor(
            np.where(~np.isin(gen_bus_idx_np, slack_idx_np))[0],
            device=self.device,
            dtype=torch.long,
        )
        self.pv_gen_idx = torch.as_tensor(
            np.where(np.isin(gen_bus_idx_np, pv_idx_np))[0],
            device=self.device,
            dtype=torch.long,
        )
        self.slack_gen_idx = torch.as_tensor(
            np.where(np.isin(gen_bus_idx_np, slack_idx_np))[0],
            device=self.device,
            dtype=torch.long,
        )
        self.spv_idx = torch.as_tensor(
            np.sort(np.concatenate([slack_idx_np, pv_idx_np])),
            device=self.device,
            dtype=torch.long,
        )
        self.gen_bus_unique_idx = torch.as_tensor(np.unique(gen_bus_idx_np), device=self.device, dtype=torch.long)

        self.quad_costs = _tensor(gencost[:, COST], self.device, self.dtype)
        self.lin_costs = _tensor(gencost[:, COST + 1], self.device, self.dtype)
        self.const_cost = float(gencost[:, COST + 2].sum())

        ybus, yf, yt = make_admittance_matrices(ppc)
        self.Ybusr = _tensor(np.real(ybus), self.device, self.dtype)
        self.Ybusi = _tensor(np.imag(ybus), self.device, self.dtype)
        self.Yfr = _tensor(np.real(yf), self.device, self.dtype)
        self.Yfi = _tensor(np.imag(yf), self.device, self.dtype)
        self.Ytr = _tensor(np.real(yt), self.device, self.dtype)
        self.Yti = _tensor(np.imag(yt), self.device, self.dtype)

        self.pg_init = _tensor(gen[:, PG] / self.baseMVA, self.device, self.dtype)
        self.qg_init = _tensor(gen[:, QG] / self.baseMVA, self.device, self.dtype)
        self.vm_init = _tensor(bus[:, VM], self.device, self.dtype)
        self.va_init = _tensor(np.deg2rad(bus[:, VA]), self.device, self.dtype)
        self.pd_init = _tensor(bus[:, PD] / self.baseMVA, self.device, self.dtype)
        self.qd_init = _tensor(bus[:, QD] / self.baseMVA, self.device, self.dtype)
        self.output_init = torch.cat([self.pg_init, self.qg_init, self.vm_init, self.va_init]).view(1, -1)
        self.input_init = torch.cat([self.pd_init, self.qd_init]).view(1, -1)

        self.pmax = _tensor(gen[:, PMAX] / self.baseMVA, self.device, self.dtype)
        self.pmin = _tensor(gen[:, PMIN] / self.baseMVA, self.device, self.dtype)
        self.qmax = _tensor(gen[:, QMAX] / self.baseMVA, self.device, self.dtype)
        self.qmin = _tensor(gen[:, QMIN] / self.baseMVA, self.device, self.dtype)
        self.vmax = _tensor(bus[:, VMAX], self.device, self.dtype)
        self.vmin = _tensor(bus[:, VMIN], self.device, self.dtype)
        self.smax = _tensor(branch[:, RATE_A] / self.baseMVA, self.device, self.dtype)
        self.amax = _tensor(np.deg2rad(np.abs(branch[:, ANGMAX])), self.device, self.dtype)
        self.amax = torch.where(self.amax > 0, self.amax, torch.full_like(self.amax, torch.pi))

        self.slack_angle = self.va_init[self.slack_idx] if self.slack_idx.numel() else self.va_init.new_empty(0)
        self.L = torch.cat(
            [
                self.pmin,
                self.qmin,
                self.vmin,
                torch.full((self.nb,), -torch.pi / 2, device=self.device, dtype=self.dtype),
            ]
        )
        self.U = torch.cat(
            [
                self.pmax,
                self.qmax,
                self.vmax,
                torch.full((self.nb,), torch.pi / 2, device=self.device, dtype=self.dtype),
            ]
        )
        self.neq = 2 * self.nb + (int(self.slack_idx.numel()) if self.enforce_slack_angle else 0)
        branch_ineq = 2 * self.nl if self.branch_constraints else 0
        self.nineq = 4 * self.ng + 2 * self.nb + branch_ineq
        self.generator_bus_incidence = torch.zeros(self.nb, self.ng, device=self.device, dtype=self.dtype)
        self.generator_bus_incidence.index_put_(
            (self.gen_bus_idx, torch.arange(self.ng, device=self.device)),
            torch.ones(self.ng, device=self.device, dtype=self.dtype),
        )

    def _sample_indices(self, n_samples, seed):
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        generator = torch.Generator(device=self.device if self.device.type != "mps" else torch.device("cpu"))
        generator.manual_seed(int(seed))
        count = int(self.input_samples.shape[0])
        if count == 0:
            raise ValueError("AC-OPF dataset contains no load samples.")
        if n_samples <= count:
            indices = torch.randperm(count, generator=generator, device=self.device)[:n_samples]
        else:
            indices = torch.randint(count, (n_samples,), generator=generator, device=self.device)
        return indices

    def sample_instances(self, n_samples, seed=2025, **kwargs):
        del kwargs
        indices = self._sample_indices(n_samples, seed)
        return self.input_samples[indices]

    def sample_supervised_instances(self, n_samples, seed=2025, **kwargs):
        del kwargs
        if self.reference_y is None:
            raise ValueError("AC-OPF dataset does not contain reference decisions for supervised training.")
        indices = self._sample_indices(n_samples, seed)
        return self.input_samples[indices], self.reference_y[indices]

    def sample_instance_batch(self, n_instances=1, seed=2025, **kwargs):
        inputs = self.sample_instances(n_instances, seed=seed, **kwargs)
        return ProblemInstanceBatch(
            family=self.name,
            inputs=inputs,
            objectives=None,
            n_instances=int(inputs.shape[0]),
            metadata={
                "seed": int(seed),
                "requested_n_instances": int(n_instances),
                "dataset_path": self.dataset_path,
                "nb": self.nb,
                "ng": self.ng,
                "nl": self.nl,
            },
        )

    def get_yvars(self, y):
        y = self._as_batch_tensor(y)
        pg = y[:, self.pg_start_yidx : self.qg_start_yidx]
        qg = y[:, self.qg_start_yidx : self.vm_start_yidx]
        vm = y[:, self.vm_start_yidx : self.va_start_yidx]
        va = y[:, self.va_start_yidx :]
        return pg, qg, vm, va

    def objective_xy(self, input_params, y, objective_batch=None):
        del input_params, objective_batch
        pg, _, _, _ = self.get_yvars(y)
        pg_mw = pg * self.baseMVA
        cost = (self.quad_costs * pg_mw.pow(2)).sum(dim=1) + (self.lin_costs * pg_mw).sum(dim=1)
        return ((cost + self.const_cost) / self.baseMVA / max(float(self.nb), 1.0)).view(-1, 1)

    def power_balance_residual(self, input_params, y):
        """Return raw AC power-balance residuals ``[P_res, Q_res]``.

        The residual convention is generation minus load minus network
        injection. Generator output is accumulated at buses, so multiple
        generators on the same bus are supported.
        """

        input_params = self._as_batch_tensor(input_params)
        pg, qg, vm, va = self.get_yvars(y)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        ir = vr @ self.Ybusr.T - vi @ self.Ybusi.T
        ii = vr @ self.Ybusi.T + vi @ self.Ybusr.T

        pg_expand = torch.zeros(input_params.shape[0], self.nb, device=input_params.device, dtype=input_params.dtype)
        qg_expand = torch.zeros_like(pg_expand)
        pg_expand.index_add_(1, self.gen_bus_idx, pg)
        qg_expand.index_add_(1, self.gen_bus_idx, qg)

        real_resid = (pg_expand - input_params[:, : self.nb]) - (vr * ir + vi * ii)
        react_resid = (qg_expand - input_params[:, self.nb :]) - (vi * ir - vr * ii)
        return torch.cat([real_resid, react_resid], dim=1)

    def eq_resid(self, input_params, y):
        residual = self.power_balance_residual(input_params, y)
        _, _, _, va = self.get_yvars(y)
        if self.enforce_slack_angle and self.slack_idx.numel():
            residual = torch.cat([residual, va[:, self.slack_idx] - self.slack_angle], dim=1)
        return residual

    def power_balance_jacobian_voltage(self, y):
        """Jacobian of power-balance residuals with respect to ``[Vm, Va]``.

        Returns ``batch x (2*nb) x (2*nb)`` ordered as
        ``[dP/dVm, dP/dVa; dQ/dVm, dQ/dVa]``. The signs match
        :meth:`power_balance_residual`.
        """

        _, _, vm, va = self.get_yvars(y)
        cos_va = torch.cos(va)
        sin_va = torch.sin(va)
        vr = vm * cos_va
        vi = vm * sin_va
        ir = vr @ self.Ybusr.T - vi @ self.Ybusi.T
        ii = vr @ self.Ybusi.T + vi @ self.Ybusr.T

        ydiagv_yi_cos_yr_sin = _compute_ydiagv(self.Ybusi, cos_va) + _compute_ydiagv(self.Ybusr, sin_va)
        ydiagv_yr_cos_yi_sin = _compute_ydiagv(self.Ybusr, cos_va) - _compute_ydiagv(self.Ybusi, sin_va)
        ydiagv_yi_vi_yr_vr = _compute_ydiagv(self.Ybusi, -vi) + _compute_ydiagv(self.Ybusr, vr)
        ydiagv_yr_vi_yi_vr = _compute_ydiagv(self.Ybusr, -vi) - _compute_ydiagv(self.Ybusi, vr)

        dp_dvm = (
            -_diagonal_batch(cos_va, ir)
            - _scale_matrix_rows(vr, ydiagv_yr_cos_yi_sin)
            - _diagonal_batch(sin_va, ii)
            - _scale_matrix_rows(vi, ydiagv_yi_cos_yr_sin)
        )
        dp_dva = (
            -_diagonal_batch(-vi, ir)
            - _scale_matrix_rows(vr, ydiagv_yr_vi_yi_vr)
            - _diagonal_batch(vr, ii)
            - _scale_matrix_rows(vi, ydiagv_yi_vi_yr_vr)
        )
        dq_dvm = (
            _diagonal_batch(cos_va, ii)
            + _scale_matrix_rows(vr, ydiagv_yi_cos_yr_sin)
            - _diagonal_batch(sin_va, ir)
            - _scale_matrix_rows(vi, ydiagv_yr_cos_yi_sin)
        )
        dq_dva = (
            _diagonal_batch(-vi, ii)
            + _scale_matrix_rows(vr, ydiagv_yi_vi_yr_vr)
            - _diagonal_batch(vr, ir)
            - _scale_matrix_rows(vi, ydiagv_yr_vi_yi_vr)
        )
        return torch.cat(
            [
                torch.cat([dp_dvm, dp_dva], dim=2),
                torch.cat([dq_dvm, dq_dva], dim=2),
            ],
            dim=1,
        )

    def power_balance_jacobian_y(self, y):
        """Jacobian of power-balance residuals with respect to full decision ``y``."""

        y = self._as_batch_tensor(y)
        batch_size = int(y.shape[0])
        incidence = self.generator_bus_incidence.to(device=y.device, dtype=y.dtype)
        zeros_bg = incidence.new_zeros(self.nb, self.ng)
        dp_dpg_qg = torch.cat([incidence, zeros_bg], dim=1)
        dq_dpg_qg = torch.cat([zeros_bg, incidence], dim=1)
        gen_jac = torch.cat([dp_dpg_qg, dq_dpg_qg], dim=0).unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([gen_jac, self.power_balance_jacobian_voltage(y)], dim=2)

    def equality_jacobian_y(self, y):
        """Jacobian of :meth:`eq_resid` with respect to full decision ``y``."""

        jacobian = self.power_balance_jacobian_y(y)
        if self.enforce_slack_angle and self.slack_idx.numel():
            y = self._as_batch_tensor(y)
            slack_jac = y.new_zeros(y.shape[0], int(self.slack_idx.numel()), self.nvar)
            row_idx = torch.arange(self.slack_idx.numel(), device=y.device)
            slack_jac[:, row_idx, self.va_start_yidx + self.slack_idx] = 1.0
            jacobian = torch.cat([jacobian, slack_jac], dim=1)
        return jacobian

    def equality_penalty_gradient_y(self, input_params, y):
        """Gradient of ``sum(eq_resid(input_params, y)^2)`` with respect to ``y``."""

        residual = self.eq_resid(input_params, y)
        jacobian = self.equality_jacobian_y(y)
        return 2.0 * jacobian.transpose(1, 2).bmm(residual.unsqueeze(-1)).squeeze(-1)

    def eq_jac_v(self, y):
        """Compatibility alias for the voltage power-balance Jacobian."""

        return self.power_balance_jacobian_voltage(y)

    def eq_jac(self, y):
        """Compatibility alias for the full equality Jacobian."""

        return self.equality_jacobian_y(y)

    def eq_grad(self, input_params, y):
        """Compatibility alias for the equality-penalty gradient."""

        return self.equality_penalty_gradient_y(input_params, y)

    def ineq_resid(self, input_params, y, clip=True):
        del input_params
        pg, qg, vm, _ = self.get_yvars(y)
        residual = torch.cat(
            [
                pg - self.pmax,
                self.pmin - pg,
                qg - self.qmax,
                self.qmin - qg,
                vm - self.vmax,
                self.vmin - vm,
            ],
            dim=1,
        )
        if self.branch_constraints:
            residual = torch.cat([residual, self.branch_ineq_resid(y)], dim=1)
        return torch.clamp(residual, min=0.0) if clip else residual

    def branch_ineq_resid(self, y):
        _, _, vm, va = self.get_yvars(y)
        f_index = self.branch_idx[:, 0]
        t_index = self.branch_idx[:, 1]
        angle_resid = (va[:, f_index] - va[:, t_index]).abs() - self.amax

        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        if_real = vr @ self.Yfr.T - vi @ self.Yfi.T
        if_imag = vi @ self.Yfr.T + vr @ self.Yfi.T
        it_real = vr @ self.Ytr.T - vi @ self.Yti.T
        it_imag = vi @ self.Ytr.T + vr @ self.Yti.T
        sf_real = vr[:, f_index] * if_real + vi[:, f_index] * if_imag
        sf_imag = vr[:, f_index] * if_imag - vi[:, f_index] * if_real
        st_real = vr[:, t_index] * it_real + vi[:, t_index] * it_imag
        st_imag = vr[:, t_index] * it_imag - vi[:, t_index] * it_real
        flow_resid = torch.maximum(sf_real.pow(2) + sf_imag.pow(2), st_real.pow(2) + st_imag.pow(2)) - self.smax.pow(2)
        return torch.cat([angle_resid, flow_resid], dim=1)

    def constraint_residual_xy(self, input_params, y, clip=True):
        eq = self.eq_resid(input_params, y)
        ineq = self.ineq_resid(input_params, y, clip=clip)
        return torch.cat([eq.abs(), ineq], dim=1) if clip else torch.cat([eq.abs(), torch.clamp(ineq, min=0.0)], dim=1)

    def check_feasibility(self, input_params, y):
        return self.constraint_residual_xy(input_params, y, clip=True)

    def scale(self, input_params, u):
        del input_params
        u = self._as_batch_tensor(u)
        if self.prediction_space == "pf_partial":
            return self.power_flow_completion.scale_partial(u)
        if u.shape[-1] != self.nvar:
            raise ValueError(f"Expected normalized AC-OPF decision width {self.nvar}, got {u.shape[-1]}.")
        return 0.5 * (u + 1.0) * (self.U - self.L) + self.L

    def inverse_scale(self, input_params, y):
        del input_params
        y = self._as_batch_tensor(y)
        if self.prediction_space == "pf_partial" and y.shape[-1] == self.partial_dim:
            return self.power_flow_completion.inverse_scale_partial(y)
        denom = torch.clamp(self.U - self.L, min=torch.finfo(y.dtype).eps)
        return 2.0 * (y - self.L) / denom - 1.0

    def complete_partial(self, input_params, y_partial):
        y_partial = self._as_batch_tensor(y_partial)
        if self.prediction_space == "pf_partial":
            return self.power_flow_completion.complete(input_params, y_partial)
        if y_partial.shape[-1] != self.nvar:
            raise ValueError("ParametricACOPFProblem currently supports full decision predictions only.")
        return y_partial

    def _as_batch_tensor(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        else:
            value = value.to(device=self.device)
            if value.is_floating_point():
                value = value.to(dtype=self.dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        return value


def default_acopf_dataset_path(case: int = 57, samples: int = 10000):
    root = Path(__file__).resolve().parents[4]
    return root / "archive" / "Homeomorphic_Projection" / "datasets" / "acopf" / f"acopf_2023_{case}_{samples}_dataset"


def _tensor(value, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype)


def _diagonal_batch(v1, v2):
    return torch.diag_embed(v1 * v2)


def _compute_ydiagv(matrix, vector):
    return matrix.unsqueeze(0) * vector.unsqueeze(1)


def _scale_matrix_rows(vector, matrix):
    return vector.unsqueeze(2) * matrix


def _resolve_dtype(dtype):
    if dtype is None:
        return torch.get_default_dtype()
    if isinstance(dtype, str):
        if not hasattr(torch, dtype):
            raise ValueError(f"Unsupported torch dtype string: {dtype}")
        dtype = getattr(torch, dtype)
    return torch.empty((), dtype=dtype).dtype


__all__ = [
    "ParametricACOPFProblem",
    "default_acopf_dataset_path",
]
