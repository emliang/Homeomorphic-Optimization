"""MaxCut-specific gauge map."""

from __future__ import annotations

import numpy as np
import torch

from .map import GaugeMap


class GaugeMapMaxCut(GaugeMap):
    def __init__(self, convex_set, p_norm=np.inf, x_origin=None, smooth=False):
        super().__init__(convex_set, p_norm, x_origin, smooth)
        self.node = convex_set.node
        self.edge = convex_set.edge
        self.I = torch.eye(len(self.node))
        self.smooth = smooth

    def _supports_explicit(self):
        return False

    def _closed_form_point_to_boundary_distance(self, u):
        batch_size = u.shape[0]
        S = torch.zeros(batch_size, len(self.node), len(self.node), device=u.device)
        S[:, self.constraint_set.upper_triangle_index[:, 0], self.constraint_set.upper_triangle_index[:, 1]] = u
        S = S + S.transpose(1, 2)
        eigenvals = torch.linalg.eigvalsh(-S)
        if self.smooth:
            hard_max = torch.max(eigenvals, dim=-1, keepdim=True)[0]
            active = eigenvals >= hard_max - self.smooth_tie_tol
            shifted = torch.where(
                active,
                (eigenvals - hard_max) / self.smooth_temperature,
                torch.full_like(eigenvals, float("-inf")),
            )
            max_eigenval = hard_max + self.smooth_temperature * torch.logsumexp(shifted, dim=-1, keepdim=True)
        else:
            max_eigenval = torch.max(eigenvals, dim=-1)[0]
        return max_eigenval.view(batch_size, 1)

    def power_method(self, A, max_iter=50, tol=1e-3):
        batch_size, n = A.shape[0], A.shape[1]
        x = torch.randn(batch_size, n, 1, device=A.device)
        x = x / torch.norm(x, dim=1, keepdim=True)
        for _ in range(max_iter):
            x_new = torch.matmul(A, x)
            x_new = x_new / torch.norm(x_new, dim=1, keepdim=True)
            error = torch.abs(x_new - x).max(1)[0]
            if error.max() < tol:
                break
            x = x_new
        max_eigenval = torch.sum(x * torch.matmul(A, x), dim=1, keepdim=True)
        max_eigenval[error > tol] = 0
        return max_eigenval

