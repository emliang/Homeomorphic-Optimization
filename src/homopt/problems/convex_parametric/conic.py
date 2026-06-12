"""Parametric conic convex problem families."""

from __future__ import annotations

import torch

from .common import ConvexParametricProblemBase, _make_generator, _randn
from .quadratic import ParametricQP


class ParametricSOCP(ParametricQP):
    """Second-order cone programs with parametric equality right-hand sides."""

    name = "parametric_socp"

    def __init__(
        self,
        *,
        n_var: int,
        n_eq: int,
        n_ineq: int,
        cone_dim: int | None = None,
        seed: int = 2025,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            n_var=n_var,
            n_eq=n_eq,
            n_ineq=n_ineq,
            seed=seed,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.cone_dim = int(cone_dim or max(2, min(5, self.nvar)))
        generator = _make_generator(seed + 307, self.device)
        self.G_soc = _randn((self.n_ineq, self.cone_dim, self.nvar), generator, self.device, self.dtype) / max(
            float(self.nvar) ** 0.5,
            1.0,
        )
        self.h_soc = _randn((self.n_ineq, self.cone_dim), generator, self.device, self.dtype) / max(
            float(self.cone_dim) ** 0.5,
            1.0,
        )
        self.C_soc = _randn((self.n_ineq, self.nvar), generator, self.device, self.dtype) / max(
            float(self.nvar) ** 0.5,
            1.0,
        )
        self.d_soc = torch.linalg.norm(self.h_soc, dim=1) + 1.0
        self.n_constraints = self.n_eq + self.n_ineq + 2 * self.nvar

    def ineq_resid(self, input_params, y, clip=True):
        del input_params
        y = self._as_batch_tensor(y)
        if self.n_ineq == 0:
            residual = y.new_zeros((y.shape[0], 0))
        else:
            cone_left = torch.einsum("mkn,bn->bmk", self.G_soc, y) + self.h_soc.unsqueeze(0)
            residual = torch.linalg.norm(cone_left, dim=2) - (y @ self.C_soc.T + self.d_soc)
        residual = torch.cat([residual, self.box_resid(y, clip=False)], dim=1)
        return torch.clamp(residual, min=0.0) if clip else residual


class ParametricSDP(ConvexParametricProblemBase):
    """Semidefinite programs over symmetric matrix decisions.

    The decision variable is the lower-triangular vector of a symmetric matrix.
    """

    name = "parametric_sdp"

    def __init__(
        self,
        *,
        matrix_dim: int,
        n_eq: int,
        seed: int = 2025,
        device=None,
        dtype=None,
        y_lower: float = -1.0,
        y_upper: float = 1.0,
        **kwargs,
    ):
        self.matrix_dim = int(matrix_dim)
        if self.matrix_dim <= 0:
            raise ValueError("matrix_dim must be positive.")
        n_var = self.matrix_dim * (self.matrix_dim + 1) // 2
        super().__init__(
            n_var=n_var,
            n_eq=n_eq,
            y_lower=y_lower,
            y_upper=y_upper,
            seed=seed,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        generator = _make_generator(seed + 401, self.device)
        self.tril_idx = torch.tril_indices(self.matrix_dim, self.matrix_dim, device=self.device)
        raw_q = _randn((self.matrix_dim, self.matrix_dim), generator, self.device, self.dtype)
        self.Q = 0.5 * (raw_q + raw_q.T)
        self.matrix_lower = torch.full(
            (self.matrix_dim, self.matrix_dim),
            float(y_lower),
            device=self.device,
            dtype=self.dtype,
        )
        self.matrix_upper = torch.full(
            (self.matrix_dim, self.matrix_dim),
            float(y_upper),
            device=self.device,
            dtype=self.dtype,
        )
        self.n_constraints = self.n_eq + 2 * self.matrix_dim * self.matrix_dim + self.matrix_dim

    def to_device(self, device):
        super().to_device(device)
        self.tril_idx = torch.tril_indices(self.matrix_dim, self.matrix_dim, device=self.device)
        return self

    def lower_vector_to_matrix(self, y):
        y = self._as_batch_tensor(y)
        matrix = y.new_zeros((y.shape[0], self.matrix_dim, self.matrix_dim))
        matrix[:, self.tril_idx[0], self.tril_idx[1]] = y
        return matrix + matrix.transpose(1, 2) - torch.diag_embed(torch.diagonal(matrix, dim1=1, dim2=2))

    def matrix_to_lower_vector(self, matrix):
        if not torch.is_tensor(matrix):
            matrix = torch.as_tensor(matrix, device=self.device, dtype=self.dtype)
        matrix = matrix.to(device=self.device, dtype=self.dtype)
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        return matrix[:, self.tril_idx[0], self.tril_idx[1]]

    def objective_xy(self, input_params, y, objective_batch=None):
        del input_params, objective_batch
        matrix = self.lower_vector_to_matrix(y)
        trace = torch.sum(matrix * self.Q.unsqueeze(0), dim=(1, 2), keepdim=False)
        return trace.view(-1, 1) / max(float(self.nvar), 1.0)

    def ineq_resid(self, input_params, y, clip=True):
        del input_params
        matrix = self.lower_vector_to_matrix(y)
        box = torch.cat(
            [
                (self.matrix_lower.unsqueeze(0) - matrix).flatten(start_dim=1),
                (matrix - self.matrix_upper.unsqueeze(0)).flatten(start_dim=1),
            ],
            dim=1,
        )
        psd = -torch.linalg.eigvalsh(matrix)
        residual = torch.cat([box, psd], dim=1)
        return torch.clamp(residual, min=0.0) if clip else residual
