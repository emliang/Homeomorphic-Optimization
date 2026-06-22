"""Synthetic joint chance-constrained linear problem family."""

from __future__ import annotations

import copy

import numpy as np
import torch

from homopt.problems.base import ParametricProblemBase, ProblemInstanceBatch
from homopt.utils import resolve_torch_dtype

EPSILON = 1e-6


def _as_tensor(value, *, device=None, dtype=torch.float32):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _vector_config(value, dim, *, default):
    if value is None:
        return np.full(dim, float(default), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.full(dim, float(array.item()), dtype=np.float32)
    if array.shape != (dim,):
        raise ValueError(f"Expected shape {(dim,)}, got {array.shape}")
    return array.astype(np.float32)


def _evaluate_chance_metrics(problem, x_input, y_value):
    x_tensor = _as_tensor(x_input, device=getattr(problem, "device", None)).view(1, -1)
    y_tensor = _as_tensor(y_value, device=x_tensor.device).view(1, -1)
    feasibility = problem.compute_scenario_feasibility(x_tensor, y_tensor)
    feasibility_rate = float(feasibility.mean().item())
    residual = problem.constraint_residual_xy(x_tensor, y_tensor, clip=True)
    max_violation = float(residual.max().item())
    return {
        "feasibility_rate": feasibility_rate,
        "chance_constraint_satisfied": bool(
            feasibility_rate >= (1.0 - float(problem.epsilon) - 1e-12) and max_violation <= 1e-6
        ),
        "max_violation": max_violation,
    }


class JCCLinearProblem(ParametricProblemBase):
    """Parametric chance-constrained linear family with synthetic sampled instances."""

    name = "JCCLinearProblem"

    def __init__(self, config=None):
        self.config = {
            "n_var": 8,
            "n_input_dim": 4,
            "n_ineq": 4,
            "n_scenarios": 20,
            "epsilon": 0.1,
            "seed": 2025,
            "x_lower": 0.0,
            "x_upper": None,
            "y_lower": 0.0,
            "y_upper": 2.0,
            "scenario_scale": 0.1,
            "input_margin": 0.1,
            "min_input_span": 0.2,
        }
        self.config.update(dict(config or {}))
        self._build_family()
        self.current_input = self.fixed_input0.clone()
        self.is_parametric = True

    def _build_family(self):
        rng = np.random.RandomState(int(self.config["seed"]))
        self.nvar = int(self.config["n_var"])
        self.n_input_dim = int(self.config["n_input_dim"])
        self.n_ineq = int(self.config["n_ineq"])
        self.n_scenarios = int(self.config["n_scenarios"])
        self.epsilon = float(self.config["epsilon"])
        self.n_constraints = int(self.n_ineq + 2 * self.nvar + 1)

        lower = _vector_config(self.config.get("y_lower"), self.nvar, default=0.0)
        upper = _vector_config(self.config.get("y_upper"), self.nvar, default=2.0)
        span = np.maximum(upper - lower, 1e-3)
        fixed_x0 = lower + 0.65 * span

        A = rng.uniform(0.2, 1.0, size=(self.n_input_dim, self.nvar)).astype(np.float32)
        W = rng.uniform(0.0, float(self.config["scenario_scale"]), size=(self.n_scenarios, self.n_input_dim)).astype(np.float32)
        G = rng.uniform(-0.5, 0.75, size=(self.n_ineq, self.nvar)).astype(np.float32) if self.n_ineq > 0 else np.zeros((0, self.nvar), dtype=np.float32)
        h_margin = rng.uniform(0.1, 0.4, size=(self.n_ineq,)).astype(np.float32) if self.n_ineq > 0 else np.zeros((0,), dtype=np.float32)
        h = (G @ fixed_x0 + h_margin).astype(np.float32)
        p = rng.uniform(-1.0, 1.0, size=(self.nvar,)).astype(np.float32)

        input_lower = _vector_config(self.config.get("x_lower"), self.n_input_dim, default=0.0)
        raw_input_upper = (A @ fixed_x0 - W.max(axis=0) - float(self.config["input_margin"])).astype(np.float32)
        min_input_span = float(self.config["min_input_span"])
        derived_input_upper = np.maximum(raw_input_upper, input_lower + min_input_span)
        input_upper_cfg = self.config.get("x_upper")
        input_upper = (
            derived_input_upper
            if input_upper_cfg is None
            else np.maximum(_vector_config(input_upper_cfg, self.n_input_dim, default=1.0), input_lower + min_input_span)
        )

        self.p_np = p
        self.A_np = A
        self.W_np = W
        self.G_np = G
        self.h_np = h
        self.input_L_np = input_lower
        self.input_U_np = input_upper
        self.L_np = lower
        self.U_np = upper

        self.p = torch.tensor(self.p_np, dtype=torch.float32)
        self.A = torch.tensor(self.A_np, dtype=torch.float32)
        self.W = torch.tensor(self.W_np, dtype=torch.float32)
        self.G = torch.tensor(self.G_np, dtype=torch.float32)
        self.h = torch.tensor(self.h_np, dtype=torch.float32)
        self.input_L = torch.tensor(self.input_L_np, dtype=torch.float32)
        self.input_U = torch.tensor(self.input_U_np, dtype=torch.float32)
        self.L = torch.tensor(self.L_np, dtype=torch.float32)
        self.U = torch.tensor(self.U_np, dtype=torch.float32)
        self.fixed_x0 = torch.tensor(fixed_x0, dtype=torch.float32)
        self.fixed_L = self.L.clone()
        self.fixed_U = self.U.clone()
        self.fixed_input0 = 0.5 * (self.input_L + self.input_U)
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def __str__(self):
        return (
            f"JCCLinearProblem-{self.nvar}-{self.n_ineq}-{self.n_input_dim}-"
            f"{self.n_scenarios}"
        )

    def to_device(self, device):
        device = torch.device(device)
        for attr_name in (
            "p",
            "A",
            "W",
            "G",
            "h",
            "input_L",
            "input_U",
            "L",
            "U",
            "fixed_x0",
            "fixed_L",
            "fixed_U",
            "fixed_input0",
            "current_input",
        ):
            if hasattr(self, attr_name):
                setattr(self, attr_name, getattr(self, attr_name).to(device=device))
        self.device = device
        return self

    def to_dtype(self, dtype):
        dtype = resolve_torch_dtype(dtype, default=torch.float32)
        for attr_name in (
            "p",
            "A",
            "W",
            "G",
            "h",
            "input_L",
            "input_U",
            "L",
            "U",
            "fixed_x0",
            "fixed_L",
            "fixed_U",
            "fixed_input0",
            "current_input",
        ):
            if hasattr(self, attr_name):
                setattr(self, attr_name, getattr(self, attr_name).to(dtype=dtype))
        self.dtype = dtype
        return self

    def sample_instances(self, n_samples, seed=2025, **kwargs):
        del kwargs
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        u = torch.rand((int(n_samples), self.n_input_dim), generator=generator, dtype=torch.float32)
        samples = self.input_L.cpu() + u * (self.input_U.cpu() - self.input_L.cpu())
        return samples.to(device=self.device, dtype=self.dtype)

    def sample_instance_batch(self, n_instances=1, seed=2025, **kwargs):
        samples = self.sample_instances(n_instances, seed=seed, **kwargs)
        return ProblemInstanceBatch(
            family=type(self).__name__,
            inputs=samples,
            objectives=None,
            n_instances=int(samples.shape[0]),
            metadata={
                "seed": int(seed),
                "n_scenarios": int(self.n_scenarios),
                "n_input_dim": int(self.n_input_dim),
            },
        )

    def build_instance(self, input_data, objective_data=None):
        del objective_data
        bound = copy.copy(self)
        bound.current_input = self._coerce_input_batch(input_data, device=self.device, batch_size=None).view(-1)
        return bound

    def _coerce_input_batch(self, input_batch, device, batch_size=None):
        x = _as_tensor(input_batch, device=device, dtype=self.dtype)
        if x.ndim == 1:
            x = x.view(1, -1)
        if x.ndim != 2 or x.shape[1] != self.n_input_dim:
            raise ValueError(f"Expected input batch shape (*, {self.n_input_dim}), got {tuple(x.shape)}")
        if batch_size is None or x.shape[0] == batch_size:
            return x
        if x.shape[0] == 1:
            return x.expand(batch_size, -1)
        raise ValueError(f"Input batch size {x.shape[0]} does not match decision batch size {batch_size}.")

    def objective_x(self, y):
        return torch.matmul(y, self.p.view(-1, 1))

    def objective_xy(self, input_params, y):
        del input_params
        return self.objective_x(y)

    def _deterministic_residual(self, y):
        pieces = []
        if self.n_ineq > 0:
            pieces.append(torch.matmul(y, self.G.T) - self.h.view(1, -1))
        pieces.append(self.L.view(1, -1) - y)
        pieces.append(y - self.U.view(1, -1))
        return torch.cat(pieces, dim=1)

    def compute_scenario_k_residual(self, input_params, y, scenario_idx, clip=True):
        input_batch = self._coerce_input_batch(input_params, y.device, batch_size=y.shape[0])
        residual = input_batch + self.W[scenario_idx].view(1, -1) - torch.matmul(y, self.A.T)
        return torch.clamp(residual, min=0.0) if clip else residual

    def compute_scenario_feasibility(self, input_params, y):
        input_batch = self._coerce_input_batch(input_params, y.device, batch_size=y.shape[0])
        scenario_feasible = torch.zeros((y.shape[0], self.n_scenarios), dtype=y.dtype, device=y.device)
        for scenario_idx in range(self.n_scenarios):
            residual = self.compute_scenario_k_residual(input_batch, y, scenario_idx, clip=True)
            scenario_feasible[:, scenario_idx] = (residual.max(dim=1)[0] <= EPSILON).to(dtype=y.dtype)
        return scenario_feasible

    def constraint_x(self, y, clip=True):
        return self.constraint_residual_xy(self.current_input.view(1, -1), y, clip=clip)

    def constraint_residual_xy(self, input_params, y, clip=True):
        input_batch = self._coerce_input_batch(input_params, y.device, batch_size=y.shape[0])
        deterministic = self._deterministic_residual(y)
        feasibility = self.compute_scenario_feasibility(input_batch, y)
        feasibility_rate = feasibility.mean(dim=1, keepdim=True)
        chance_violation = (1.0 - float(self.epsilon)) - feasibility_rate
        residual = torch.cat([deterministic, chance_violation], dim=1)
        return torch.clamp(residual, min=0.0) if clip else residual


class JCCIMProblem(JCCLinearProblem):
    """Legacy-style joint chance-constrained inventory/IM linear family.

    This family keeps the same package-native parametric contract as
    ``JCCLinearProblem`` while matching the old Homeomorphic_Projection JCCIM
    structure more closely:

    - linear objective ``p^T y``
    - sampled scenario constraints ``A y >= x + w_i``
    - deterministic side constraints ``G y <= h`` and box bounds

    The current package uses the same exact-solver baselines for both families,
    so this class only owns the family generation logic and legacy-facing shape
    conventions.
    """

    name = "JCCIMProblem"

    def __init__(self, config=None):
        translated = dict(config or {})
        if "n_eq" in translated and "n_input_dim" not in translated:
            translated["n_input_dim"] = translated.pop("n_eq")
        super().__init__(config=translated)
        self.n_eq = int(self.n_input_dim)

    def _build_family(self):
        rng = np.random.RandomState(int(self.config["seed"]))
        self.nvar = int(self.config["n_var"])
        self.n_input_dim = int(self.config["n_input_dim"])
        self.n_eq = int(self.n_input_dim)
        self.n_ineq = int(self.config["n_ineq"])
        self.n_scenarios = int(self.config["n_scenarios"])
        self.epsilon = float(self.config["epsilon"])
        self.n_constraints = int(self.n_ineq + 2 * self.nvar + 1)

        lower = _vector_config(self.config.get("y_lower"), self.nvar, default=0.0)
        upper = _vector_config(self.config.get("y_upper"), self.nvar, default=2.0)
        span = np.maximum(upper - lower, 1e-3)
        fixed_x0 = lower + 0.55 * span

        Q = np.diag(rng.uniform(0.0, 0.5, size=(self.nvar,)).astype(np.float32))
        p = rng.uniform(-1.0, 1.0, size=(self.nvar,)).astype(np.float32)
        A = rng.uniform(-1.0, 1.0, size=(self.n_input_dim, self.nvar)).astype(np.float32)
        row_norm = np.linalg.norm(A, axis=1, keepdims=True)
        A = A / np.maximum(row_norm, 1e-2)
        W = rng.uniform(0.0, float(self.config["scenario_scale"]), size=(self.n_scenarios, self.n_input_dim)).astype(np.float32)
        G = rng.uniform(-1.0, 1.0, size=(self.n_ineq, self.nvar)).astype(np.float32) if self.n_ineq > 0 else np.zeros((0, self.nvar), dtype=np.float32)
        h_margin = rng.uniform(0.15, 0.45, size=(self.n_ineq,)).astype(np.float32) if self.n_ineq > 0 else np.zeros((0,), dtype=np.float32)
        h = (G @ fixed_x0 + h_margin).astype(np.float32)

        input_margin = float(self.config["input_margin"])
        feasible_ceiling = (A @ fixed_x0 - W.max(axis=0) - input_margin).astype(np.float32)
        input_lower_cfg = self.config.get("x_lower")
        input_upper_cfg = self.config.get("x_upper")
        min_input_span = float(self.config["min_input_span"])
        if input_upper_cfg is None:
            input_upper = feasible_ceiling
        else:
            input_upper = _vector_config(input_upper_cfg, self.n_input_dim, default=1.0)
        if input_lower_cfg is None:
            input_lower = input_upper - np.maximum(np.abs(input_upper) * 0.35, min_input_span)
        else:
            input_lower = _vector_config(input_lower_cfg, self.n_input_dim, default=0.0)
        input_upper = np.maximum(input_upper, input_lower + min_input_span)

        self.Q_np = Q.astype(np.float32)
        self.p_np = p
        self.A_np = A
        self.W_np = W
        self.G_np = G
        self.h_np = h
        self.input_L_np = input_lower.astype(np.float32)
        self.input_U_np = input_upper.astype(np.float32)
        self.L_np = lower
        self.U_np = upper

        self.Q = torch.tensor(self.Q_np, dtype=torch.float32)
        self.p = torch.tensor(self.p_np, dtype=torch.float32)
        self.A = torch.tensor(self.A_np, dtype=torch.float32)
        self.W = torch.tensor(self.W_np, dtype=torch.float32)
        self.G = torch.tensor(self.G_np, dtype=torch.float32)
        self.h = torch.tensor(self.h_np, dtype=torch.float32)
        self.input_L = torch.tensor(self.input_L_np, dtype=torch.float32)
        self.input_U = torch.tensor(self.input_U_np, dtype=torch.float32)
        self.L = torch.tensor(self.L_np, dtype=torch.float32)
        self.U = torch.tensor(self.U_np, dtype=torch.float32)
        self.fixed_x0 = torch.tensor(fixed_x0, dtype=torch.float32)
        self.fixed_L = self.L.clone()
        self.fixed_U = self.U.clone()
        self.fixed_input0 = 0.5 * (self.input_L + self.input_U)
        self.current_input = self.fixed_input0.clone()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def __str__(self):
        return (
            f"JCCIMProblem-{self.nvar}-{self.n_ineq}-{self.n_eq}-"
            f"{self.n_scenarios}"
        )


__all__ = [
    "JCCIMProblem",
    "JCCLinearProblem",
]
