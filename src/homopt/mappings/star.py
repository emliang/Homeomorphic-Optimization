"""Star-domain mapping implementation."""

from __future__ import annotations

import torch

from homopt.utils import move_tensors_to_device


class StarMap:
    """Homeomorphic mapping between the unit ball and a star-shaped domain."""

    def __init__(self, star_set, p_norm=2, x_origin=(0, 0)):
        self.alpha = star_set.alpha
        self.num_star = star_set.num_star
        # Optimizers and visualization helpers use this to define the source ball.
        self.p_norm = p_norm
        self.center = torch.tensor(x_origin, dtype=torch.float32).view(1, -1)

    def to_device(self, device):
        return move_tensors_to_device(self, device)

    def scaling(self, x):
        norms = torch.norm(x, dim=-1, p=2, keepdim=True)
        values = torch.ones_like(norms)
        valid = (norms > 0).view(-1)
        if valid.any():
            x_valid = x[valid]
            theta = x_valid[:, [1]] / x_valid[:, [0]]
            values[valid] = 1 + self.alpha * torch.sin(self.num_star * torch.atan(theta))
        return values

    def forward(self, z, method="autograd"):
        del method
        return z * self.scaling(z)

    def inverse(self, x):
        return x / self.scaling(x)

    def gauge(self, x):
        return torch.norm(x, dim=-1, p=2, keepdim=True) / self.scaling(x)


class PolyStarMap:
    """Radial map for the intersection of a convex gauge set and a star set."""

    def __init__(self, poly_map, star_map, p_norm=2, eps=1e-12):
        self.poly_map = poly_map
        self.star_map = star_map
        self.p_norm = p_norm
        self.center = torch.zeros_like(poly_map.center)
        self._eps = float(eps)

    def to_device(self, device):
        self.poly_map.to_device(device)
        self.star_map.to_device(device)
        return move_tensors_to_device(self, device)

    def _direction(self, x):
        norm = torch.norm(x, dim=-1, p=2, keepdim=True)
        direction = x / norm.clamp_min(self._eps)
        basis = torch.zeros_like(direction)
        basis[:, :1] = 1.0
        return norm, torch.where(norm > self._eps, direction, basis)

    def _boundary_beta(self, direction, method="autograd"):
        poly_beta = self.poly_map.gauge(direction, method=method)
        star_beta = self.star_map.gauge(direction)
        return torch.maximum(poly_beta, star_beta)

    def forward(self, z, method="autograd", return_state=False):
        if method not in {"autograd", "explicit"}:
            raise ValueError(f"Unsupported PolyStarMap forward method: {method}.")
        norm, direction = self._direction(z)
        beta = self._boundary_beta(direction, method=method).clamp_min(self._eps)
        x = torch.where(norm > self._eps, z / beta, torch.zeros_like(z))
        if return_state:
            return x, None
        return x

    def inverse(self, x, method="autograd"):
        if method not in {"autograd", "explicit"}:
            raise ValueError(f"Unsupported PolyStarMap inverse method: {method}.")
        norm = torch.norm(x, dim=-1, p=2, keepdim=True)
        gauge = self.gauge(x, method=method)
        return torch.where(norm > self._eps, x * gauge / norm.clamp_min(self._eps), torch.zeros_like(x))

    def gauge(self, x, method="autograd"):
        if method not in {"autograd", "explicit"}:
            raise ValueError(f"Unsupported PolyStarMap gauge method: {method}.")
        norm = torch.norm(x, dim=-1, p=2, keepdim=True)
        poly_gauge = self.poly_map.gauge(x, method=method)
        star_gauge = self.star_map.gauge(x)
        gauge = torch.maximum(poly_gauge, star_gauge)
        return torch.where(norm > self._eps, gauge, torch.zeros_like(gauge))

    def vjp(self, z, grad_x, method="autograd", state=None):
        del state
        if method not in {"autograd", "explicit"}:
            raise ValueError(f"Unsupported PolyStarMap vjp method: {method}.")
        with torch.enable_grad():
            z_detached = z.detach().requires_grad_(True)
            x = self.forward(z_detached, method=method)
            return torch.autograd.grad(x, z_detached, grad_outputs=grad_x, create_graph=False)[0]
