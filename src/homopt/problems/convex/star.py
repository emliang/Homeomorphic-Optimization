"""Two-dimensional star and poly-star toy problem families."""

import torch

from homopt.problems.base import TensorRuntimeMixin

from .core import _radial_quadratic_dual_value, _square


class ToyStarOpt(TensorRuntimeMixin):
    def __init__(self, alpha=1, num_star=4) -> None:
        self.alpha = alpha
        self.num_star = num_star
        self.obj_center = torch.tensor([1, 1]).view(1, -1)
        self.w = torch.tensor([0.3, 0.7]).view(1, -1)
        self.nvar = 2
        self.ncon = 1
        self.Q = torch.diag(self.w.view(-1))
        self.p =  (-2 * self.w * self.obj_center).view(1, -1)
        self.d = torch.sum(self.w * self.obj_center**2)
        self.ineq_cons = range(self.ncon)
        self.eq_cons = None
    def __str__(self):
        return f'Star_{self.num_star}_{self.alpha}'

    def scaling(self, x):
        """Compute the scaling factor gamma."""
        theta = x[:,[1]]/x[:,[0]]
        return 1 + self.alpha * torch.sin(self.num_star * torch.atan(theta))
    # Forward and inverse transformations
    # def psi_forward(self, z):
    #     """Transform from z-space to x-space."""
    #     return z * self.scaling(z)

    # def psi_inverse(self, x):
    #     """Transform from x-space to z-space."""
    #     return x / self.scaling(x)

    #    # Gradient functions
    def gradient_scaling(self, x):
        theta = x[:,[1]]/x[:,[0]]
        grad = self.alpha * self.num_star * torch.cos(self.num_star * torch.atan(theta)) \
                /  torch.sum(x**2, dim=-1,keepdim=True) \
                * torch.cat([-x[:,[1]], x[:,[0]]],dim=-1)
        return grad

    # Objective functions
    def objective_x(self, x):
        """Compute objective in x-space."""
        obj = self.w * (x - self.obj_center)**2
        return torch.sum(obj, dim=-1, keepdim=True)


    def gradient_objective_x(self, x):
        return 2 * self.w * (x - self.obj_center)

    def objective_z(self, z, hom_map=None, hom_map_method="autograd"):
        """Compute objective in z-space."""
        x = hom_map.forward(z, method=hom_map_method)
        return self.objective_x(x)

    def gradient_objective_z(self, z, hom_map=None, auto_grad=True, hom_map_method="autograd"):
        if auto_grad:
            z_detached = z.detach().requires_grad_(True)
            obj = self.objective_z(z_detached, hom_map, hom_map_method=hom_map_method)
            return torch.autograd.grad(obj, z_detached, create_graph=False)[0]
        else:
            gamma = hom_map.scaling(z)
            x = z * gamma
            grad_scaling = hom_map.gradient_scaling(z)
            return self.w * (x - self.obj_center) * (gamma + z * grad_scaling)

    def constraint_z(self, z, hom_map=None):
        return torch.norm(z, dim=-1, p=2, keepdim=True) - 1

    def gradient_constraint_z(self, z, hom_map=None):
        return z / torch.norm(z, dim=-1, p=2, keepdim=True)

    def constraint_x(self, x, clip=True):
        cons = torch.norm(x, dim=-1, p=2, keepdim=True) - self.scaling(x)
        if clip:
            return torch.clamp(cons, min=0)
        else:
            return cons

    def gradient_constraint_x(self, x, clip=True, dual_var=None):
        grad = x / torch.norm(x, dim=-1, p=2, keepdim=True) - self.gradient_scaling(x)
        if clip:
            if self.constraint_x(x) == 0:
                grad = grad * 0
        else:
            grad = grad * dual_var
        return grad

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None, method="explicit"):
        del method
        residual = self.constraint_x(x, clip=False)
        weights = dual_var
        if penalty_coef is not None:
            weights = weights + penalty_coef * torch.clamp(residual, min=0)
        grad = self.gradient_objective_x(x) + self.gradient_constraint_x(x, clip=False, dual_var=weights)
        if proximal_coef is not None and x_outer is not None:
            grad = grad + proximal_coef * (x - x_outer)
        return grad

    def gradient_penalty_x(self, x):
        x_detached = x.detach().requires_grad_(True)
        violation = 0.5 * _square(self.constraint_x(x_detached, clip=True)).sum(-1)
        grad = torch.autograd.grad(violation, x_detached, create_graph=False)[0]
        return grad

    def radial_primal_obj(self, x, hom_map=None):
        """Positive radial primal payoff used by the generalized RD baseline."""
        if hom_map is None:
            raise ValueError('hom_map is None')
        center = hom_map.center.to(device=x.device, dtype=x.dtype)
        return 1 + self.objective_x(center) - self.objective_x(x)

    def radial_dual_objective(self, x, hom_map=None):
        """Generalized radial dual objective using the active feasible-set gauge."""
        if hom_map is None:
            raise ValueError('hom_map is None')
        center = hom_map.center.to(device=x.device, dtype=x.dtype)
        u = x - center
        q_eff = 2 * self.Q.to(device=x.device, dtype=x.dtype)
        p_eff = self.gradient_objective_x(center).to(device=x.device, dtype=x.dtype)
        obj = _radial_quadratic_dual_value(u, q_eff, p_eff)
        cons = hom_map.gauge(u)
        return torch.max(obj, cons)

    def radial_dual_objective_gradient(self, x, hom_map=None, method="autograd"):
        """Compute a subgradient of the generalized radial dual objective."""
        if hom_map is None:
            raise ValueError('hom_map is None')
        if method == 'autograd':
            x_detached = x.detach().requires_grad_(True)
            obj = self.radial_dual_objective(x_detached, hom_map)
            return torch.autograd.grad(obj, x_detached, create_graph=False)[0]
        raise NotImplementedError


class PolyStarOpt(TensorRuntimeMixin):
    """Two-dimensional toy problem on the intersection of poly and star sets."""

    def __init__(self, poly_problem, star_problem) -> None:
        if poly_problem.nvar != 2 or star_problem.nvar != 2:
            raise ValueError("PolyStarOpt currently supports only two-dimensional toy problems.")
        self.poly_problem = poly_problem
        self.star_problem = star_problem
        self.nvar = 2
        self.ncon = int(poly_problem.ncon) + int(star_problem.ncon)
        self.n_eq = 0
        self.ineq_cons = range(self.ncon)
        self.eq_cons = None
        self.Q = star_problem.Q
        self.p = star_problem.p
        self.d = star_problem.d
        self.obj_center = star_problem.obj_center
        self.w = star_problem.w
        self.alpha = star_problem.alpha
        self.num_star = star_problem.num_star

    def to_device(self, device):
        self.poly_problem.to_device(device)
        self.star_problem.to_device(device)
        return super().to_device(device)

    def __str__(self):
        return f"PolyStar_{self.num_star}_{self.alpha}"

    def objective_x(self, x):
        return self.star_problem.objective_x(x)

    def gradient_objective_x(self, x):
        return self.star_problem.gradient_objective_x(x)

    def constraint_x(self, x, clip=True):
        residual = torch.cat(
            [
                self.poly_problem.constraint_x(x, clip=False),
                self.star_problem.constraint_x(x, clip=False),
            ],
            dim=1,
        )
        return torch.clamp(residual, min=0) if clip else residual

    def gradient_constraint_x(self, x, clip=True, dual_var=None):
        residual = self.constraint_x(x, clip=False)
        weights = torch.ones_like(residual) if dual_var is None else dual_var
        if clip:
            weights = weights * (residual > 0).to(dtype=weights.dtype)

        poly_end = int(self.poly_problem.ncon)
        poly_grad = self.poly_problem._explicit_inequality_weighted_gradient_x(x, weights[:, :poly_end])
        star_grad = self.star_problem.gradient_constraint_x(
            x,
            clip=False,
            dual_var=weights[:, poly_end:],
        )
        return poly_grad + star_grad

    def lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        objective = self.objective_x(x)
        residual = self.constraint_x(x, clip=False)
        dual = (dual_var * residual).sum(-1, keepdim=True)
        penalty = (
            0.5 * penalty_coef * _square(torch.clamp(residual, min=0)).sum(-1, keepdim=True)
            if penalty_coef is not None
            else 0
        )
        proximal = (
            0.5 * proximal_coef * _square(x - x_outer).sum(-1, keepdim=True)
            if proximal_coef is not None and x_outer is not None
            else 0
        )
        return objective + dual + penalty + proximal

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None, method="explicit"):
        del method
        residual = self.constraint_x(x, clip=False)
        weights = dual_var
        if penalty_coef is not None:
            weights = weights + penalty_coef * torch.clamp(residual, min=0)
        grad = self.gradient_objective_x(x) + self.gradient_constraint_x(x, clip=False, dual_var=weights)
        if proximal_coef is not None and x_outer is not None:
            grad = grad + proximal_coef * (x - x_outer)
        return grad

    def gradient_penalty_x(self, x):
        x_detached = x.detach().requires_grad_(True)
        violation = 0.5 * _square(self.constraint_x(x_detached, clip=True)).sum(-1)
        return torch.autograd.grad(violation, x_detached, create_graph=False)[0]

    def objective_z(self, z, hom_map=None, hom_map_method="autograd"):
        if hom_map is None:
            raise ValueError("hom_map is required for z-space objective evaluation.")
        return self.objective_x(hom_map.forward(z, method=hom_map_method))

    def gradient_objective_z(self, z, hom_map=None, method="autograd", hom_map_method="autograd", x=None, hom_state=None):
        if method == "autograd":
            z_detached = z.detach().requires_grad_(True)
            obj = self.objective_z(z_detached, hom_map, hom_map_method=hom_map_method)
            return torch.autograd.grad(obj, z_detached, create_graph=False)[0]
        if method == "explicit":
            if hom_map is None:
                raise ValueError("hom_map is required for explicit z-space objective gradient.")
            if x is None:
                x, hom_state = hom_map.forward(z, method=hom_map_method, return_state=True)
            return hom_map.vjp(z, self.gradient_objective_x(x), method=hom_map_method, state=hom_state)
        raise NotImplementedError

__all__ = ["PolyStarOpt", "ToyStarOpt"]
