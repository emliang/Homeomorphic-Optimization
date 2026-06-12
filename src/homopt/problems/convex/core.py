"""Shared helpers for deterministic convex problem families."""

import torch

torch.set_default_dtype(torch.float32)


def _square(value):
    return value * value


def _radial_quadratic_dual_value(u, Q, p_eff, scale=1.0):
    scale_tensor = torch.as_tensor(scale, device=u.device, dtype=u.dtype).view(1, 1)
    py = torch.sum(p_eff * u, dim=-1, keepdim=True)
    yQy = torch.sum((u @ Q) * u, dim=-1, keepdim=True)
    discriminant = (py + 1) ** 2 + 2 * scale_tensor * yQy
    root = (py + 1 + torch.sqrt(torch.clamp(discriminant, min=0))) / (2 * scale_tensor)
    return torch.where(discriminant >= 0, torch.clamp(root, min=0), torch.zeros_like(root))


def _tensor_or_none(value):
    return torch.tensor(value, dtype=torch.float32) if value is not None else None


def _load_convex_core_config(problem, config):
    """Populate the shared deterministic convex family fields from config."""
    problem.prob_para = config
    problem.Q = torch.tensor(config['Q'], dtype=torch.float32)
    problem.p = torch.tensor(config['p'], dtype=torch.float32)
    problem.U = torch.tensor(config['U'], dtype=torch.float32)
    problem.L = torch.tensor(config['L'], dtype=torch.float32)
    problem.A = _tensor_or_none(config['A'])
    problem.b = _tensor_or_none(config['b'])
    problem.Qq = _tensor_or_none(config['Qq'])
    problem.pq = _tensor_or_none(config['pq'])
    problem.bq = _tensor_or_none(config['bq'])
    problem.G = _tensor_or_none(config['G'])
    problem.h = _tensor_or_none(config['h'])
    problem.C = _tensor_or_none(config['C'])
    problem.d = _tensor_or_none(config['d'])
    problem.nvar = problem.p.shape[0]
    problem.n_soc = problem.G.shape[0] if problem.G is not None else 0
    problem.n_lin = problem.A.shape[0] if problem.A is not None else 0
    problem.n_qua = problem.Qq.shape[0] if problem.Qq is not None else 0
    problem.ncon = problem.n_soc + problem.n_lin + problem.n_qua + problem.nvar * 2
    problem.ineq_cons = range(problem.ncon)

__all__ = [
    "_load_convex_core_config",
    "_radial_quadratic_dual_value",
    "_square",
    "_tensor_or_none",
]
