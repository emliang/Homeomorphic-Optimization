"""Shared machinery for parametric convex learning problems."""

from __future__ import annotations

from abc import abstractmethod

import torch

from homopt.problems.base import ParametricProblemBase, TensorRuntimeMixin


class ConvexParametricProblemBase(TensorRuntimeMixin, ParametricProblemBase):
    """Base class for convex parametric families with equality completion.

    ``input_params`` are the right-hand sides of linear equalities
    ``A y = input_params``. Learning models may predict only the free decision
    variables ``u``; the remaining decision variables are completed by solving
    the equality system.
    """

    name = "convex_parametric"
    has_eq = True

    def __init__(
        self,
        *,
        n_var: int,
        n_eq: int,
        y_lower: float | torch.Tensor = -1.0,
        y_upper: float | torch.Tensor = 1.0,
        input_lower: float = -1.0,
        input_upper: float = 1.0,
        seed: int = 2025,
        device=None,
        dtype=None,
    ):
        if n_var <= 0:
            raise ValueError("n_var must be positive.")
        if n_eq < 0 or n_eq >= n_var:
            raise ValueError("n_eq must satisfy 0 <= n_eq < n_var for partial completion.")

        self.nvar = int(n_var)
        self.n_eq = int(n_eq)
        self.xdim = int(n_eq)
        self.partial_dim = self.nvar - self.n_eq
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self.seed = int(seed)

        self.L = _as_bound_tensor(y_lower, self.nvar, device=self.device, dtype=self.dtype)
        self.U = _as_bound_tensor(y_upper, self.nvar, device=self.device, dtype=self.dtype)
        if torch.any(self.U <= self.L):
            raise ValueError("All upper decision bounds must be greater than lower bounds.")

        self.input_lower = float(input_lower)
        self.input_upper = float(input_upper)
        if self.input_upper <= self.input_lower:
            raise ValueError("input_upper must be greater than input_lower.")

        generator = _make_generator(self.seed, self.device)
        self.A = _sample_full_row_rank_matrix(self.n_eq, self.nvar, generator, self.device, self.dtype)
        self.partial_vars_idx, self.other_vars = _select_partial_variables(self.A, generator)
        self.A_partial = self.A[:, self.partial_vars_idx] if self.n_eq else self.A.new_empty((0, self.partial_dim))
        self.A_other = self.A[:, self.other_vars] if self.n_eq else self.A.new_empty((0, 0))
        self.A_other_inv = torch.linalg.inv(self.A_other) if self.n_eq else self.A.new_empty((0, 0))
        self.n_constraints = self.n_eq + 2 * self.nvar

    def to_dtype(self, dtype):
        dtype = torch.empty((), dtype=dtype).dtype
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor) and attr.is_floating_point():
                setattr(self, attr_name, attr.to(dtype=dtype))
        self.dtype = dtype
        return self

    def sample_instances(self, n_samples, seed=2025, **kwargs):
        del kwargs
        generator = _make_generator(int(seed), self.device)
        n_samples = int(n_samples)
        if self.xdim == 0:
            return torch.empty(n_samples, 0, device=self.device, dtype=self.dtype)
        values = torch.rand(n_samples, self.xdim, generator=generator, device=self.device, dtype=self.dtype)
        return values * (self.input_upper - self.input_lower) + self.input_lower

    def scale(self, input_params, u):
        """Map normalized free variables from ``[-1, 1]`` into decision bounds."""
        del input_params
        u = self._as_batch_tensor(u)
        lower, upper = self._bounds_for_width(u.shape[-1])
        return 0.5 * (u + 1.0) * (upper - lower) + lower

    def inverse_scale(self, input_params, y):
        """Map physical decision variables back to normalized ``[-1, 1]`` coordinates."""
        del input_params
        y = self._as_batch_tensor(y)
        lower, upper = self._bounds_for_width(y.shape[-1])
        return 2.0 * (y - lower) / (upper - lower) - 1.0

    def complete_partial(self, input_params, y_partial):
        """Complete a partial physical decision vector so that ``A y = input_params``."""
        input_params = self._as_batch_tensor(input_params)
        y_partial = self._as_batch_tensor(y_partial)
        if y_partial.shape[-1] == self.nvar:
            return y_partial
        if y_partial.shape[-1] != self.partial_dim:
            raise ValueError(f"Expected partial decision width {self.partial_dim}, got {y_partial.shape[-1]}.")
        if self.n_eq == 0:
            return y_partial
        rhs = input_params - y_partial @ self.A_partial.T
        completed_other = rhs @ self.A_other_inv.T
        y_full = y_partial.new_empty(y_partial.shape[0], self.nvar)
        y_full[:, self.partial_vars_idx] = y_partial
        y_full[:, self.other_vars] = completed_other
        return y_full

    def eq_resid(self, input_params, y):
        input_params = self._as_batch_tensor(input_params)
        y = self._as_batch_tensor(y)
        if self.n_eq == 0:
            return y.new_zeros((y.shape[0], 0))
        return y @ self.A.T - input_params

    def constraint_residual_xy(self, input_params, y, clip=True):
        y = self._as_batch_tensor(y)
        eq = self.eq_resid(input_params, y).abs()
        ineq = self.ineq_resid(input_params, y, clip=clip)
        return torch.cat([eq, ineq], dim=1)

    def check_feasibility(self, input_params, y):
        return self.constraint_residual_xy(input_params, y, clip=True)

    def box_resid(self, y, *, clip=True):
        y = self._as_batch_tensor(y)
        resid = torch.cat([self.L - y, y - self.U], dim=1)
        return torch.clamp(resid, min=0.0) if clip else resid

    @abstractmethod
    def ineq_resid(self, input_params, y, clip=True):
        """Return inequality residuals in ``g(input_params, y) <= 0`` form."""

    @abstractmethod
    def objective_xy(self, input_params, y, objective_batch=None):
        """Return objective values for a batch of inputs and decisions."""

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

    def _bounds_for_width(self, width):
        width = int(width)
        if width == self.nvar:
            return self.L, self.U
        if width == self.partial_dim:
            return self.L[self.partial_vars_idx], self.U[self.partial_vars_idx]
        raise ValueError(f"Cannot scale width {width}; expected {self.nvar} or {self.partial_dim}.")


def _as_bound_tensor(value, width, *, device, dtype):
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(width).clone()
    if tensor.shape != (width,):
        raise ValueError(f"Bounds must be scalar or shape ({width},), got {tuple(tensor.shape)}.")
    return tensor


def _make_generator(seed, device):
    generator_device = device if torch.device(device).type != "mps" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _randn(shape, generator, device, dtype):
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def _sample_full_row_rank_matrix(n_rows, n_cols, generator, device, dtype):
    if n_rows == 0:
        return torch.empty(0, n_cols, device=device, dtype=dtype)
    for _ in range(128):
        matrix = _randn((n_rows, n_cols), generator, device, dtype) / max(float(n_cols) ** 0.5, 1.0)
        if torch.linalg.matrix_rank(matrix).item() == n_rows:
            return matrix
    raise RuntimeError("Failed to sample a full-row-rank equality matrix.")


def _select_partial_variables(A, generator):
    n_eq, n_var = A.shape
    all_idx = torch.arange(n_var, device=A.device)
    if n_eq == 0:
        return all_idx, all_idx.new_empty(0)
    best_other = None
    best_score = None
    for _ in range(256):
        perm = torch.randperm(n_var, generator=generator, device=A.device)
        other = torch.sort(perm[:n_eq]).values
        block = A[:, other]
        sign, logabsdet = torch.linalg.slogdet(block)
        if sign.abs().item() == 0:
            continue
        score = logabsdet.item()
        if best_score is None or score > best_score:
            best_score = score
            best_other = other
    if best_other is None:
        raise RuntimeError("Failed to select invertible equality-completion variables.")
    mask = torch.ones(n_var, dtype=torch.bool, device=A.device)
    mask[best_other] = False
    partial = all_idx[mask]
    return partial, best_other
