"""Parametric convex quadratic problem families."""

from __future__ import annotations

import torch

from .common import ConvexParametricProblemBase, _make_generator, _randn


class ParametricQP(ConvexParametricProblemBase):
    """Convex quadratic programs with parametric equality right-hand sides."""

    name = "parametric_qp"

    def __init__(
        self,
        *,
        n_var: int,
        n_eq: int,
        n_ineq: int,
        seed: int = 2025,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(n_var=n_var, n_eq=n_eq, seed=seed, device=device, dtype=dtype, **kwargs)
        self.n_ineq = int(n_ineq)
        generator = _make_generator(seed + 101, self.device)

        raw_q = _randn((self.nvar, self.nvar), generator, self.device, self.dtype)
        self.Q = raw_q.T @ raw_q / max(float(self.nvar), 1.0) + 1e-3 * torch.eye(
            self.nvar, device=self.device, dtype=self.dtype
        )
        self.p = _randn((self.nvar,), generator, self.device, self.dtype) / max(float(self.nvar), 1.0)
        self.G = _randn((self.n_ineq, self.nvar), generator, self.device, self.dtype) / max(
            float(self.nvar) ** 0.5,
            1.0,
        )
        self.h = _linear_ineq_offset(self.G, self.A)
        self.n_constraints = self.n_eq + self.n_ineq + 2 * self.nvar

    def objective_xy(self, input_params, y, objective_batch=None):
        del input_params, objective_batch
        y = self._as_batch_tensor(y)
        quad = 0.5 * torch.sum((y @ self.Q) * y, dim=1, keepdim=True)
        lin = y @ self.p.view(-1, 1)
        return (quad + lin) / max(float(self.nvar), 1.0)

    def ineq_resid(self, input_params, y, clip=True):
        del input_params
        y = self._as_batch_tensor(y)
        if self.n_ineq == 0:
            residual = y.new_zeros((y.shape[0], 0))
        else:
            residual = y @ self.G.T - self.h
        residual = torch.cat([residual, self.box_resid(y, clip=False)], dim=1)
        return torch.clamp(residual, min=0.0) if clip else residual


class ParametricConvexQCQP(ParametricQP):
    """Convex QCQPs with positive-semidefinite quadratic constraints."""

    name = "parametric_convex_qcqp"

    def __init__(self, *, seed: int = 2025, **kwargs):
        super().__init__(seed=seed, **kwargs)
        generator = _make_generator(seed + 211, self.device)
        if self.n_ineq == 0:
            self.H = torch.empty(0, self.nvar, self.nvar, device=self.device, dtype=self.dtype)
        else:
            diagonal = torch.rand(
                self.n_ineq,
                self.nvar,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            ) / max(float(self.nvar), 1.0)
            self.H = torch.diag_embed(diagonal)

    def ineq_resid(self, input_params, y, clip=True):
        del input_params
        y = self._as_batch_tensor(y)
        if self.n_ineq == 0:
            residual = y.new_zeros((y.shape[0], 0))
        else:
            hy = torch.einsum("mij,bj->bmi", self.H, y)
            quad = 0.5 * torch.sum(hy * y.unsqueeze(1), dim=2)
            residual = quad + y @ self.G.T - self.h
        residual = torch.cat([residual, self.box_resid(y, clip=False)], dim=1)
        return torch.clamp(residual, min=0.0) if clip else residual


def _linear_ineq_offset(G, A):
    if G.numel() == 0:
        return G.new_empty((0,))
    if A.shape[0] == 0:
        return torch.ones(G.shape[0], device=G.device, dtype=G.dtype)
    pinv = torch.linalg.pinv(A)
    return torch.sum(torch.abs(G @ pinv), dim=1) + 1.0
