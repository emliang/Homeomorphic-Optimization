"""Convex/toy problem implementations and deterministic instance builders."""

import torch
import numpy as np

from homopt.problems.base import FixedProblemParametricAdapter, TensorRuntimeMixin

torch.set_default_dtype(torch.float32)


def _square(value):
    return value * value


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


###################################################################
# Projection programming: Same constraint, L2 objective
###################################################################
class ProjProblem(TensorRuntimeMixin):
    def __init__(self, x_init, problem):
        self.x_init = torch.as_tensor(x_init).detach().clone().view(1, -1)
        self.problem = problem
        self.nvar = problem.nvar
        self.ncon = problem.ncon
        self.n_eq = 0
        self.ineq_cons = range(self.ncon)
        self.eq_cons = None

    def objective_x(self, x):
        return torch.sum((x - self.x_init)**2, dim=-1, keepdim=True)

    def gradient_objective_x(self, x):
        return 2*(x - self.x_init)

    def constraint_x(self, x, clip=True):
        return self.problem.constraint_x(x, clip)

    def gradient_constraint_x(self, x, clip=True, dual_var=None):
        return self.problem.gradient_constraint_x(x, clip = clip, dual_var=dual_var)

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

###################################################################
# Linear Optimization Oracle: Same constraint, Linear Objective
###################################################################
class LinearProblem(TensorRuntimeMixin):
    def __init__(self, x_init, problem):
        self.x_init = torch.as_tensor(x_init).detach().clone().view(1, -1)
        self.problem = problem
        self.nvar = problem.nvar
        self.ncon = problem.ncon
        self.n_eq = 0
        # Store the linear objective coefficients
        self.gradient =self.problem.gradient_objective_x(self.x_init)
        self.ineq_cons = range(self.ncon)
        self.eq_cons = None
    def objective_x(self, x):
        return torch.sum(self.gradient * x, dim=-1, keepdim=True)

    def gradient_objective_x(self, x):
        return self.gradient

    def constraint_x(self, x, clip=True):
        return self.problem.constraint_x(x, clip)

    def gradient_constraint_x(self, x, clip=True, dual_var=None):
        return self.problem.gradient_constraint_x(x, clip=clip, dual_var=dual_var)

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


###################################################################
# Two-dim toy examples
###################################################################
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
        """Compute primal objective: 1 - 0.5 * Q^TxQ - p^T x"""
        if hom_map is None:
            raise ValueError('hom_map is None')
        else:
            return 1 - self.objective_x(x-hom_map.center) + self.d
        # translation to be positive and maximize its negative

    def radial_dual_objective(self, x, hom_map=None):
        """Compute radial dual objective"""
        if hom_map is None:
            raise ValueError('hom_map is None')
        else:
            py = torch.sum(self.p * (x-hom_map.center), dim=-1)
            yQy = torch.sum(((x - hom_map.center) @ self.Q) * (x-hom_map.center), dim=-1)
            discriminant = (py + 1)**2 + 2 * yQy
            obj = (py + 1 + torch.sqrt(discriminant)) / (2)
            obj = torch.clamp(obj, min=0)
            cons = hom_map.gauge(x-hom_map.center)
        return torch.max(obj, cons)

    def radial_dual_objective_gradient(self, x, hom_map=None, method="autograd"):
        """Compute gradient of radial dual objective"""
        if hom_map is None:
            raise ValueError('hom_map is None')
        else:
            if method == 'autograd':
                x_detached = x.detach().requires_grad_(True)
                obj = self.radial_dual_objective(x_detached, hom_map)
                grad_x = torch.autograd.grad(obj, x_detached, create_graph=False)[0]
            else:
                raise NotImplementedError
        return grad_x


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



###################################################################
# Convex Programming
###################################################################
class ConvexOpt(TensorRuntimeMixin):
    def __init__(self, config) -> None:
        """
        min_x 1/2 x^T Q x + p x
        s.t.  L < x < U
              Ax < b,
              || Gx + h ||_2 < Cx + d
        """
        _load_convex_core_config(self, config)
        self.n_eq = 0
        self.eq_cons = None

    def __str__(self):
        return 'SOCP'

    def to_float32(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.float())
        return self

    def objective_x(self, x):
        return torch.sum(0.5 * (x @ self.Q) * x + self.p * x, dim=-1, keepdim=True)

    def inequality_constraint_terms_x(self, x):
        """Assemble the shared inequality residual blocks for convex problem families."""
        resids = []
        if self.A is not None:
            lin_res = torch.matmul(x, self.A.T) - self.b
            resids.append(lin_res)
        if self.Qq is not None:
            q = 0.5 * torch.einsum("bi,mij,bj->bm", x, self.Qq, x)
            p = torch.matmul(x, self.pq.T)
            quad_red = q + p - self.bq
            resids.append(quad_red)
        if self.G is not None:
            q = torch.norm(torch.einsum("mkn,bn->bmk", self.G, x) + self.h.unsqueeze(0), dim=-1, p=2)
            p = torch.matmul(x, self.C.T) + self.d
            soc_red = q - p
            resids.append(soc_red)
        lbound = self.L - x
        ubound = x - self.U
        resids += [lbound, ubound]
        return resids

    def gradient_objective_x(self, x):
        """Compute gradient of objective function with respect to x
        Args:
            x: input points (batch_size x n)
        Returns:
            gradient (batch_size x n)
        """
        return x @ self.Q + self.p

    def constraint_x(self, x, clip=True):
        """
         ||G^T x + h||_2 <= C^T x + d
         G: m * k * n
         h: m * k
         y: m * n
         C: m * n
         d: m * 1
        """
        resids = torch.cat(self.inequality_constraint_terms_x(x), dim=1)
        if clip:
            return torch.clamp(resids, min=0)
        else:
            return resids

    def gradient_constraint_x(self, x, clip=True, dual_var = None, method = 'autograd'):
        """Compute gradient of constraint function with respect to x
        Args:
            x: input points (batch_size x n)
        Returns:
            gradient (batch_size x num_constraints x n)
        """
        if method == 'autograd':
            x_detached = x.detach().requires_grad_(True)
            if dual_var is None:
                violation = self.constraint_x(x_detached).sum(-1)
            else:
                violation = (self.constraint_x(x_detached, clip=clip) * dual_var).sum(-1)
            grad = torch.autograd.grad(violation, x_detached, create_graph=False)[0]
        elif method == 'finite_diff':
            epsilon = 1e-6
            grad = torch.zeros_like(x)
            for i in range(x.shape[1]): # iterate over each variable
                delta = torch.zeros_like(x)
                delta[:, i] = epsilon
                obj_plus = self.constraint_x(x + delta)
                obj_minus = self.constraint_x(x - delta)
                grad[:, i] = (obj_plus - obj_minus) / (2 * epsilon)
        else:
            batch_size = x.shape[0]
            # 1. SOC constraints gradient: ||G_i^T x + h_i||_2 <= c_i^T x + d_i
            grad = []
            if self.G is not None:
                grad_soc_list = []
                for i in range(self.G.shape[0]):  # iterate over each SOC constraint
                    # Get components for i-th SOC constraint
                    Gi = self.G[i]  # k x n
                    hi = self.h[i]  # k
                    ci = self.C[i]  # n

                    # Compute G_i^T x + h_i
                    Gix = torch.matmul(x, Gi.T)  # batch x k
                    Gix_h = Gix + hi.unsqueeze(0)  # batch x k

                    # Compute norm and normalized vector
                    norms = torch.norm(Gix_h, dim=1, p=2, keepdim=True)  # batch x 1

                    # Avoid division by zero
                    mask = norms > 1e-10
                    normalized = torch.zeros_like(Gix_h)
                    normalized[mask.flatten()] = Gix_h[mask.flatten()] / norms[mask.flatten()]

                    # Gradient for i-th SOC constraint: G_i * normalized - c_i
                    grad_i = torch.matmul(normalized.unsqueeze(1), Gi) - ci.unsqueeze(0)  # batch x 1 x n
                    grad_soc_list.append(grad_i)

                # Stack all SOC constraint gradients
                grad_soc = torch.cat(grad_soc_list, dim=1)  # batch x m_soc x n
                grad.append(grad_soc)

            # 2. Linear constraints gradient
            if self.A is not None:
                grad_lin = self.A  # m x n
                grad.append(grad_lin)
            # 3. Box constraints gradient
            grad_lower = -torch.eye(x.shape[1], device=x.device)  # n x n
            grad_upper = torch.eye(x.shape[1], device=x.device)   # n x n
            grad.append(grad_lower)
            grad.append(grad_upper)

            # Combine all gradients
            grad = torch.cat([  grad ], dim=1)
            if dual_var is None:
                grad =  grad.sum(1)
            else:
                grad = (grad * dual_var.unsqueeze(-1)).sum(1)
        return grad

    def lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        objective = self.objective_x(x)
        residual = self.constraint_x(x, clip=False)
        dual = (dual_var * residual).sum(-1)
        penalty = (
            0.5 * penalty_coef * _square(torch.clamp(residual, min=0)).sum(-1)
            if penalty_coef is not None
            else 0
        )
        proximal = (
            0.5 * proximal_coef * _square(x - x_outer).sum(-1)
            if proximal_coef is not None and x_outer is not None
            else 0
        )
        return objective + dual + penalty + proximal

    def _explicit_inequality_weighted_gradient_x(self, x, weights):
        grad = torch.zeros_like(x)
        cursor = 0
        if self.A is not None:
            next_cursor = cursor + self.n_lin
            grad = grad + torch.matmul(weights[:, cursor:next_cursor], self.A)
            cursor = next_cursor
        if self.Qq is not None:
            next_cursor = cursor + self.n_qua
            qsym = 0.5 * (self.Qq + self.Qq.transpose(1, 2))
            quad_grads = torch.einsum("mij,bj->bmi", qsym, x) + self.pq.unsqueeze(0)
            grad = grad + torch.einsum("bm,bmn->bn", weights[:, cursor:next_cursor], quad_grads)
            cursor = next_cursor
        if self.G is not None:
            next_cursor = cursor + self.n_soc
            gx_h = torch.einsum("mkn,bn->bmk", self.G, x) + self.h.unsqueeze(0)
            gx_norm = torch.norm(gx_h, dim=-1, keepdim=True).clamp_min(1e-12)
            soc_grads = torch.einsum("bmk,mkn->bmn", gx_h / gx_norm, self.G) - self.C.unsqueeze(0)
            grad = grad + torch.einsum("bm,bmn->bn", weights[:, cursor:next_cursor], soc_grads)
            cursor = next_cursor
        lower_weights = weights[:, cursor:cursor + self.nvar]
        cursor += self.nvar
        upper_weights = weights[:, cursor:cursor + self.nvar]
        return grad - lower_weights + upper_weights

    def _explicit_lagrangian_x_gradient(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        residual = self.constraint_x(x, clip=False)
        weights = dual_var
        if penalty_coef is not None:
            weights = weights + penalty_coef * torch.clamp(residual, min=0)
        grad = self.gradient_objective_x(x) + self._explicit_inequality_weighted_gradient_x(x, weights)
        if proximal_coef is not None and x_outer is not None:
            grad = grad + proximal_coef * (x - x_outer)
        return grad

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None, method="autograd"):
        if method == "autograd":
            x_detached = x.detach().requires_grad_(True)
            lagrangian = self.lagrangian_x(x_detached, dual_var, penalty_coef, proximal_coef, x_outer)
            return torch.autograd.grad(lagrangian, x_detached, create_graph=False)[0]
        if method == "explicit":
            return self._explicit_lagrangian_x_gradient(x, dual_var, penalty_coef, proximal_coef, x_outer)
        raise NotImplementedError

    def gradient_penalty_x(self, x):
        x_detached = x.detach().requires_grad_(True)
        violation = 0.5 * _square(self.constraint_x(x_detached, clip=True)).sum(-1)
        grad = torch.autograd.grad(violation, x_detached, create_graph=False)[0]
        return grad

    def objective_z(self, z, hom_map=None, hom_map_method="autograd"):
        x = hom_map.forward(z, method=hom_map_method)
        return self.objective_x(x)

    def gradient_objective_z(self, z, hom_map=None, method="autograd", hom_map_method="autograd", x=None, hom_state=None):
        """Compute gradient of objective function with respect to z using chain rule
        Args:
            z: input points in unit ball (1 x n)
            gauge_map: homeomorphism mapping from unit ball to feasible set
        Returns:
            gradient (1 x n)
        """
        if method == 'autograd':
            z_detached = z.detach().requires_grad_(True)
            obj = self.objective_z(z_detached, hom_map, hom_map_method=hom_map_method)
            grad_z = torch.autograd.grad(obj, z_detached, create_graph=False)[0]
        elif method == 'explicit':
            if hom_map is None:
                raise ValueError("hom_map is required for explicit z-space objective gradient.")
            if x is None:
                x, hom_state = hom_map.forward(z, method=hom_map_method, return_state=True)
            grad_x = self.gradient_objective_x(x)
            grad_z = hom_map.vjp(z, grad_x, method=hom_map_method, state=hom_state)
        elif method == 'zeroorder':
            epsilon = 1e-6
            delta = torch.rand_like(z)
            delta = delta / torch.norm(delta, dim=1, keepdim=True)
            obj_plus = self.objective_z(z + epsilon * delta, hom_map)
            obj_minus = self.objective_z(z - epsilon * delta, hom_map)
            grad_z = (obj_plus - obj_minus) / (2 * epsilon) * delta
        elif method == 'finite_diff':
            epsilon = 1e-6
            batch_size, dim = z.shape

            # Create a perturbation tensor based on the identity matrix:
            #   - eye has shape (dim, dim)
            #   - multiplying it by epsilon gives the per-coordinate perturbation.
            #   - unsqueeze and expand to shape (batch_size, dim, dim) so that each sample gets its own copy.
            eye = torch.eye(dim, device=z.device, dtype=z.dtype)
            perturb = epsilon * eye.unsqueeze(0).expand(batch_size, -1, -1)

            # Expand z to match the perturb tensor shape. z_expanded has shape (batch_size, 1, dim)
            # and will be broadcast to (batch_size, dim, dim)
            z_expanded = z.unsqueeze(1)

            # Generate the perturbed inputs for plus and minus directions.
            # Each sample now has `dim` perturbations, one for each coordinate, resulting in shape (batch_size, dim, dim)
            z_plus = z_expanded + perturb
            z_minus = z_expanded - perturb

            # Flatten the first two dimensions from (batch_size, dim, dim) to (batch_size * dim, dim)
            z_plus_flat = z_plus.reshape(-1, dim)
            z_minus_flat = z_minus.reshape(-1, dim)

            # Compute the objective function for all perturbed inputs in one batch call.
            # It is assumed that `self.objective_z` returns a tensor of shape (batch_size * dim,) or (batch_size * dim, 1)
            obj_plus = self.objective_z(z_plus_flat, hom_map)
            obj_minus = self.objective_z(z_minus_flat, hom_map)

            # If necessary, squeeze the last dimension to ensure the shape is (batch_size * dim,)
            if obj_plus.dim() > 1:
                obj_plus = obj_plus.squeeze(-1)
            if obj_minus.dim() > 1:
                obj_minus = obj_minus.squeeze(-1)

            # Reshape the results back to (batch_size, dim)
            obj_plus = obj_plus.reshape(batch_size, dim)
            obj_minus = obj_minus.reshape(batch_size, dim)

            # Compute the finite-difference numerical gradient in a vectorized manner.
            grad_z = (obj_plus - obj_minus) / (2 * epsilon)
        else:
            raise NotImplementedError
        return grad_z

    def radial_primal_obj(self, x, hom_map=None):
        """Compute primal objective: 1 - 0.5 * Q^TxQ - p^T x"""
        if hom_map is None:
            raise ValueError('hom_map is None')
        else:
            return 1 - self.objective_x(x-hom_map.center)
        # translation to be positive and maximize its negative

    def radial_dual_objective(self, x, hom_map=None):
        """Compute radial dual objective"""
        if hom_map is None:
            raise ValueError('hom_map is None')
        else:
            py = torch.sum(self.p * (x-hom_map.center), dim=-1)
            yQy = torch.sum(((x - hom_map.center) @ self.Q) * (x-hom_map.center), dim=-1)
            discriminant = (py + 1)**2 + 2 * yQy
            obj = (py + 1 + torch.sqrt(discriminant)) / (2)
            obj = torch.clamp(obj, min=0)
            cons = hom_map.gauge(x-hom_map.center)
        return torch.max(obj, cons)

    def radial_dual_objective_gradient(self, x, hom_map=None, method="autograd"):
        """Compute gradient of radial dual objective"""
        if hom_map is None:
            raise ValueError('hom_map is None')
        else:
            if method == 'autograd':
                x_detached = x.detach().requires_grad_(True)
                obj = self.radial_dual_objective(x_detached, hom_map)
                grad_x = torch.autograd.grad(obj, x_detached, create_graph=False)[0]
            else:
                raise NotImplementedError
        return grad_x


class ConvexOptEq(ConvexOpt):
    """
    ConvexOpt plus optional linear equality constraints.

    min_x  1/2 x^T Q x + p^T x
    s.t.   L <= x <= U
           A x <= b
           1/2 x^T Qq_i x + pq_i^T x <= bq_i
           ||G_i x + h_i||_2 <= C_i^T x + d_i
           A_eq x = b_eq
    """
    def __init__(self, config) -> None:
        _load_convex_core_config(self, config)
        self.A_eq = _tensor_or_none(config.get('A_eq')) # m_eq * n
        self.b_eq = _tensor_or_none(config.get('b_eq')) # m_eq
        if (self.A_eq is None) != (self.b_eq is None):
            raise ValueError("ConvexOptEq requires both A_eq and b_eq, or neither.")
        self.n_eq = self.A_eq.shape[0] if self.A_eq is not None else 0
        self.eq_cons = range(self.ncon, self.ncon+self.n_eq)

    def eq_constraint_x(self, x):
        if self.A_eq is not None:
            return torch.matmul(x, self.A_eq.T) - self.b_eq
        return x.new_zeros((x.shape[0], 0))

    def build_explicit_lagrangian_z_outer_cache(
        self,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        x_outer=None,
        proximal_space="z",
    ):
        q_eff = self.Q
        p_eff = self.p.view(1, -1) if self.p.dim() == 1 else self.p

        if self.n_eq > 0:
            p_eff = p_eff + torch.matmul(dual_var, self.A_eq)
            if penalty_coef is not None:
                q_eff = q_eff + penalty_coef * torch.matmul(self.A_eq.T, self.A_eq)
                b_eq_term = torch.matmul(self.b_eq.view(1, -1), self.A_eq)
                p_eff = p_eff - penalty_coef * b_eq_term

        if proximal_coef is not None and proximal_space == "x":
            if x_outer is None:
                raise ValueError("x_outer is required when proximal_space='x'.")
            q_eff = q_eff + proximal_coef * torch.eye(
                self.nvar,
                device=self.device,
                dtype=self.Q.dtype,
            )
            p_eff = p_eff - proximal_coef * x_outer

        return {
            "q_eff": q_eff,
            "p_eff": p_eff,
        }

    def constraint_x(self, x, clip = True, eq_cons = True):
        """
         ||G^T x + h||_2 <= C^T x + d
         G: m * k * n
         h: m * k
         y: m * n
         C: m * n
         d: m * 1
        """
        resids = self.inequality_constraint_terms_x(x)

        if eq_cons and self.n_eq > 0:
            eq_res = self.eq_constraint_x(x)
            resids += [eq_res]
        resids = torch.cat(resids, dim=1)
        if clip:
            resids[:, self.ineq_cons] = torch.clamp(resids[:, self.ineq_cons], min=0)
        return resids

    def lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        """
        Dual term, penalty term, proximal term (For all constraints eq and ineq)
        """
        objective = self.objective_x(x)
        constraint_resid = self.constraint_x(x, clip=False, eq_cons=True)
        if penalty_coef is not None:
            ineq_penalty = _square(torch.clamp(constraint_resid[:, self.ineq_cons], min=0)).sum(-1)
            if self.n_eq > 0:
                eq_penalty = _square(constraint_resid[:, self.eq_cons]).sum(-1)
            else:
                eq_penalty = 0
            penalty = 0.5 * penalty_coef * (ineq_penalty + eq_penalty)
        else:
            penalty = 0
        if proximal_coef is not None:
            proximal = 0.5 * proximal_coef * _square(x - x_outer).sum(-1)
        else:
            proximal = 0
        dual = (dual_var * constraint_resid).sum(-1)
        lagrangian = objective + penalty + dual + proximal
        return lagrangian

    def _explicit_lagrangian_x_gradient(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        grad = self.gradient_objective_x(x)
        ineq_resid = self.constraint_x(x, clip=False, eq_cons=False)
        ineq_weights = dual_var[:, self.ineq_cons]
        if penalty_coef is not None:
            ineq_weights = ineq_weights + penalty_coef * torch.clamp(ineq_resid, min=0)
        grad = grad + self._explicit_inequality_weighted_gradient_x(x, ineq_weights)
        if self.n_eq > 0:
            eq_weights = dual_var[:, self.eq_cons]
            if penalty_coef is not None:
                eq_weights = eq_weights + penalty_coef * self.eq_constraint_x(x)
            grad = grad + torch.matmul(eq_weights, self.A_eq)
        if proximal_coef is not None and x_outer is not None:
            grad = grad + proximal_coef * (x - x_outer)
        return grad

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None, method="autograd"):
        """
        Gradient of Lagrangian
        """
        # auto_
        if method == 'autograd':
            x_detached = x.detach().requires_grad_(True)
            lagrangian = self.lagrangian_x(x_detached, dual_var, penalty_coef, proximal_coef, x_outer)
            grad_x = torch.autograd.grad(lagrangian, x_detached, create_graph=False)[0]
        elif method == 'explicit':
            grad_x = self._explicit_lagrangian_x_gradient(x, dual_var, penalty_coef, proximal_coef, x_outer)
        else:
            raise NotImplementedError
        return grad_x

    def lagrangian_z(
        self,
        z,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        hom_map=None,
        z_outer=None,
        x_outer=None,
        proximal_space="z",
        hom_map_method="autograd",
    ):
        """
        Dual term, penalty term, proximal term: (For eq only)
        """
        x = hom_map.forward(z, method=hom_map_method)
        objective = self.objective_x(x)
        eq_constraint = self.eq_constraint_x(x)
        # print(eq_constraint)
        if penalty_coef is not None:
            eq_penalty = 0.5 * penalty_coef * _square(eq_constraint).sum(-1)
        else:
            eq_penalty = 0
        if proximal_coef is not None:
            if proximal_space == "z":
                proximal = 0.5 * proximal_coef * _square(z - z_outer).sum(-1)
            elif proximal_space == "x":
                proximal = 0.5 * proximal_coef * _square(x - x_outer).sum(-1)
            else:
                raise ValueError(f"Unsupported proximal_space: {proximal_space}")
        else:
            proximal = 0
        eq_dual = (dual_var * eq_constraint).sum(-1)
        lagrangian = objective + eq_penalty + eq_dual + proximal
        return lagrangian

    def _explicit_lagrangian_z_gradient(
        self,
        z,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        hom_map=None,
        z_outer=None,
        x_outer=None,
        proximal_space="z",
        hom_map_method="autograd",
        x=None,
        hom_state=None,
        outer_cache=None,
    ):
        if hom_map is None:
            raise ValueError("hom_map is required for explicit z-space lagrangian gradient.")

        if x is None:
            x, hom_state = hom_map.forward(z, method=hom_map_method, return_state=True)
        if outer_cache is not None:
            grad_x = torch.matmul(x, outer_cache["q_eff"]) + outer_cache["p_eff"]
        else:
            grad_x = self.gradient_objective_x(x)
            if self.n_eq > 0:
                eq_weights = dual_var
                if penalty_coef is not None:
                    eq_weights = eq_weights + penalty_coef * self.eq_constraint_x(x)
                grad_x = grad_x + torch.matmul(eq_weights, self.A_eq)
            if proximal_coef is not None and proximal_space == "x":
                if x_outer is None:
                    raise ValueError("x_outer is required when proximal_space='x'.")
                grad_x = grad_x + proximal_coef * (x - x_outer)

        grad_z = hom_map.vjp(z, grad_x, method=hom_map_method, state=hom_state)
        if proximal_coef is not None and proximal_space == "z":
            if z_outer is None:
                raise ValueError("z_outer is required when proximal_space='z'.")
            grad_z = grad_z + proximal_coef * (z - z_outer)
        return grad_z

    def gradient_lagrangian_z(
        self,
        z,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        hom_map=None,
        z_outer=None,
        x_outer=None,
        proximal_space="z",
        method="autograd",
        hom_map_method="autograd",
        x=None,
        hom_state=None,
        outer_cache=None,
    ):
        """
        Gradient of Lagrangian
        """
        if method == 'autograd':
            z_detached = z.detach().requires_grad_(True)
            lagrangian = self.lagrangian_z(
                z_detached,
                dual_var,
                penalty_coef,
                proximal_coef,
                hom_map,
                z_outer,
                x_outer,
                proximal_space,
                hom_map_method,
            )
            grad_z = torch.autograd.grad(lagrangian, z_detached, create_graph=False)[0]
        elif method == 'explicit':
            grad_z = self._explicit_lagrangian_z_gradient(
                z,
                dual_var,
                penalty_coef=penalty_coef,
                proximal_coef=proximal_coef,
                hom_map=hom_map,
                z_outer=z_outer,
                x_outer=x_outer,
                proximal_space=proximal_space,
                hom_map_method=hom_map_method,
                x=x,
                hom_state=hom_state,
                outer_cache=outer_cache,
            )
        else:
            raise NotImplementedError
        return grad_z


class ParametricConvexProblem(FixedProblemParametricAdapter):
    """Learning-route view for deterministic convex families.

    This adapter treats a fixed convex optimization problem as a degenerate
    parametric family: the external instance input is accepted for API
    consistency, while objective/constraint evaluation is delegated to the
    wrapped convex problem on the decision variable itself.
    """

    def __init__(self, convex_problem):
        self.convex_problem = convex_problem
        super().__init__(convex_problem, name=type(convex_problem).__name__)

###################################################################
# Max-Cut SDP
###################################################################
class MaxCutSDP(TensorRuntimeMixin):
    def __init__(self, config) -> None:
        """
        standard Max-Cut SDP formulation:
            max_X sum(1-x_ij)/2
            s.t. x_ii =1, X>>0
        equivalent formulation:
            max_y sum(1-x_k)/2
            s.t.  -1 < x < 1
                  I + sum (x_k A_k) >> 0,
        """
        self.prob_para = config
           
        self.node ,self.edge, self.weights = config['node'], config['edge'], config['weights']
        self.num_node = len(config['node'])
        self.num_edge = len(config['edge'])
        self.weights = torch.tensor(self.weights, dtype=torch.float32)

        upper_triangle_index = [(i,j) for i in range(self.num_node) for j in range(self.num_node) if i<j]
        self.upper_triangle_index = np.array(upper_triangle_index)
        edge_index = []
        for k, (i,j) in enumerate(upper_triangle_index):
            if (i,j) in config['edge'] or (j,i) in config['edge']:
                edge_index.append(k)
        self.edge_index = np.array(edge_index)
        # Create adjacency matrix with weights

        self.nvar = (self.num_node ** 2 - self.num_node) // 2
        self.ncon = 1
        self.L = torch.ones(self.nvar, dtype=torch.float32) * -1
        self.U = torch.ones(self.nvar, dtype=torch.float32) * 1

        self.Q = torch.zeros(self.num_node, self.num_node, dtype=torch.float32)
        self.p = torch.zeros(self.nvar, dtype=torch.float32)
        self.p[self.edge_index] = self.weights / 2
        self.b = (self.weights).sum() / 2

    def __str__(self):
        return 'MaxCutSDP'

    def objective_x(self, x):
        return -torch.sum(self.weights * (1 - x[:, self.edge_index]), dim=1, keepdim=True) / 2

    def gradient_objective_x(self, x):
        """Compute gradient of objective function with respect to x
        Args:
            x: input points (batch_size x n)
        Returns:
            gradient (batch_size x n)
        """
        grad = torch.zeros_like(x).to(x.device)
        grad[:, self.edge_index] = self.weights * 0.5  # Negative because we're minimizing
        return grad

    def regularized_objective_x(self, x, reg = 1e-6):
        return self.objective_x(x) + reg * (x**2).sum().view(1,-1)

    def constraint_x(self, x, clip=True):
        """Compute constraint violations for the SDP constraint X >> 0
        Args:
            x: input points (batch_size x n)
            clip: whether to clip negative values to 0
        Returns:
            violations (batch_size x 1)
        """
        # batch_size = x.shape[0]
        # violations = torch.zeros(1, 1, device=x.device)

        batch_size = x.shape[0]
        # Construct the symmetric matrix X for each batch item.
        X = torch.zeros((batch_size, self.num_node, self.num_node), device=x.device, dtype=x.dtype)
        X[:, self.upper_triangle_index[:, 0], self.upper_triangle_index[:, 1]] = x
        X = X + X.transpose(1, 2) + torch.eye(self.num_node, device=x.device, dtype=x.dtype).unsqueeze(0)

        # Compute eigenvalues
        try:
            eigenvals = torch.linalg.eigvalsh(X)
            min_eigenval = torch.min(eigenvals, dim=1, keepdim=True).values
            # Constraint violation is -min_eigenval when positive
            violations = torch.clamp(-min_eigenval, min=0) if clip else -min_eigenval
        except:
            # If eigendecomposition fails, use a large violation
            violations = torch.full((batch_size, 1), 1e6, device=x.device, dtype=x.dtype)
        
        return violations

    def gradient_constraint_x(self, x, clip=True, dual_var=None, method='autograd'):
        """Compute gradient of constraint function with respect to x
        Args:
            x: input points (batch_size x n)
            dual_var: dual variables (batch_size x num_constraints)
            method: method to compute gradient
        Returns:
            gradient (batch_size x n)
        """
        if method == 'autograd':
            x_detached = x.detach().requires_grad_(True)
            if dual_var is None:
                violation = self.constraint_x(x_detached, clip=True).sum(-1)
            else:
                violation = (self.constraint_x(x_detached, clip=False) * dual_var).sum(-1)
            grad = torch.autograd.grad(violation, x_detached, create_graph=False)[0]
        else:
            # Finite difference approximation
            epsilon = 1e-6
            grad = torch.zeros_like(x)
            for i in range(x.shape[1]):
                delta = torch.zeros_like(x)
                delta[:, i] = epsilon
                obj_plus = self.constraint_x(x + delta)
                obj_minus = self.constraint_x(x - delta)
                grad[:, i] = (obj_plus - obj_minus) / (2 * epsilon)
        return grad

    def gradient_penalty_x(self, x):
        x_detached = x.detach().requires_grad_(True)
        violation = 0.5 * _square(self.constraint_x(x_detached, clip=True)).sum(-1)
        grad = torch.autograd.grad(violation, x_detached, create_graph=False)[0]
        return grad

    def objective_z(self, z, hom_map=None, hom_map_method="autograd"):
        """Compute objective in transformed space
        Args:
            z: input points in unit ball (batch_size x n)
            hom_map: homeomorphism mapping from unit ball to feasible set
        Returns:
            objective value (batch_size x 1)
        """
        x = hom_map.forward(z, method=hom_map_method)
        # return self.objective_x(x)
        return self.regularized_objective_x(x, reg=1e-5) #+ 1e-5 * z.square().sum().view(1,-1)

    def gradient_objective_z(self, z, hom_map=None, method="autograd", hom_map_method="autograd"):
        """Compute gradient of objective function with respect to z using chain rule
        Args:
            z: input points in unit ball (batch_size x n)
            hom_map: homeomorphism mapping from unit ball to feasible set
            method: method to compute gradient
        Returns:
            gradient (batch_size x n)
        """
        if method == 'autograd':
            z_detached = z.detach().requires_grad_(True)
            obj = self.objective_z(z_detached, hom_map, hom_map_method=hom_map_method)
            grad_z = torch.autograd.grad(obj, z_detached, create_graph=False)[0]
        elif method == 'explicit':
            if hom_map is None:
                raise ValueError("hom_map is required for explicit MaxCut z-space objective gradient.")
            x, hom_state = hom_map.forward(z, method=hom_map_method, return_state=True)
            grad_x = self.gradient_objective_x(x) + 2e-5 * x
            grad_z = hom_map.vjp(z, grad_x, method=hom_map_method, state=hom_state)
        elif method == 'finite_diff':
            epsilon = 1e-6
            grad_z = torch.zeros_like(z)
            for i in range(z.shape[1]):
                delta = torch.zeros_like(z)
                delta[:, i] = epsilon
                obj_plus = self.objective_z(z + delta, hom_map)
                obj_minus = self.objective_z(z - delta, hom_map)
                grad_z[:, i] = (obj_plus - obj_minus) / (2 * epsilon)
        else:
            raise NotImplementedError
        return grad_z

    def radial_primal_obj(self, x, hom_map=None):
        """Compute primal objective: 1 - 0.5 * Q^TxQ - p^T x"""
        if hom_map is None:
            raise ValueError('hom_map is None')
        else:
            return self.objective_x(x)
            # translation to be positive and maximize its negative

    def radial_dual_objective(self, x, hom_map=None):
        """Compute radial dual objective"""
        if hom_map is None:
            raise ValueError('hom_map is None')
        else:
            py = torch.sum(self.p * (x), dim=-1)
            discriminant = (py + 1) ** 2
            obj = (py + 1 + torch.sqrt(discriminant)) / (2*self.b)
            obj = torch.clamp(obj, min=0)
            cons = hom_map.gauge(x)
        return torch.max(obj, cons)

    def radial_dual_objective_gradient(self, x, hom_map=None, method="autograd"):
        """Compute gradient of radial dual objective"""
        if hom_map is None:
            raise ValueError('hom_map is None')
        else:
            if method == 'autograd':
                x_detached = x.detach().requires_grad_(True)
                obj = self.radial_dual_objective(x_detached, hom_map)
                grad_x = torch.autograd.grad(obj, x_detached, create_graph=False)[0]
            else:
                raise NotImplementedError
        return grad_x

class BMSDP(MaxCutSDP):
    def __init__(self, config, rank=1) -> None:
        super().__init__(config)
        self.rank = rank
        self.ncon = self.num_node #+ self.num_node ** 2
        self.nvar = self.num_node * self.rank
        self.L =  -1
        self.U =  1
        self.edge_index = np.array([[i,j] for (i,j) in self.edge])
        self.eq_cons = range(self.ncon)
        self.ineq_cons = None

    def __str__(self):
        return 'MaxCutSDP'

    def objective_x(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_node, self.rank)
        X = torch.matmul(x, x.transpose(1, 2))
        obj = - torch.sum(self.weights * (1 - X[:, self.edge_index[:,0], self.edge_index[:,1]])).view(-1,1) / 2
        return obj
    
    def gradient_objective_x(self, x, auto_grad=True):
        """
        Compute the gradient of the objective function with respect to x
        Args:
            x: input tensor of shape (batch_size x num_node*rank)
        Returns:
            gradient: tensor of same shape as x
        """
        if auto_grad:
            x = x.detach().requires_grad_(True)
            obj = self.objective_x(x)
            grad = torch.autograd.grad(obj, x, create_graph=False)[0]
        # else:
        # batch_size = x.shape[0]
        # x_reshaped = x.view(batch_size, self.num_node, self.rank)
        # # Create adjacency matrix with weights
        # # adj_matrix = torch.zeros(self.num_node, self.num_node, device=x.device)
        # # adj_matrix[self.edge_index[0], self.edge_index[1]] = self.weights
        # # Gradient: dobj/dx = (1/2) * A @ x where A is the weighted adjacency matrix
        # grad_reshaped = 0.5 * torch.matmul(self.adj_matrix.unsqueeze(0), x_reshaped)
        # grad = grad_reshaped.view(batch_size, -1)
        return grad
    
    def constraint_x(self, x, clip=True):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_node, self.rank)
        X = torch.matmul(x, x.transpose(1, 2))
        ## diagnal elements: 1
        diag_elements = torch.diagonal(X, dim1=1, dim2=2)
        diag_violation = (diag_elements - 1).view(batch_size, -1)
        if clip:
            diag_violation = (diag_violation).abs()
        return diag_violation

    def lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        objective = self.objective_x(x)
        residual = self.constraint_x(x, clip=False)
        dual = (dual_var * residual).sum(-1)
        penalty = (
            0.5 * penalty_coef * _square(residual).sum(-1)
            if penalty_coef is not None
            else 0
        )
        proximal = (
            0.5 * proximal_coef * _square(x - x_outer).sum(-1)
            if proximal_coef is not None and x_outer is not None
            else 0
        )
        return objective + dual + penalty + proximal

    def _explicit_lagrangian_x_gradient(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        batch_size = x.shape[0]
        x_view = x.view(batch_size, self.num_node, self.rank)
        grad_view = torch.zeros_like(x_view)
        edge_index = torch.as_tensor(self.edge_index, device=x.device, dtype=torch.long)
        edge_i = edge_index[:, 0]
        edge_j = edge_index[:, 1]
        weights = self.weights.to(device=x.device, dtype=x.dtype).view(1, -1, 1)
        grad_view.index_add_(1, edge_i, 0.5 * weights * x_view[:, edge_j, :])
        grad_view.index_add_(1, edge_j, 0.5 * weights * x_view[:, edge_i, :])

        residual = self.constraint_x(x, clip=False)
        constraint_weights = dual_var
        if penalty_coef is not None:
            constraint_weights = constraint_weights + penalty_coef * residual
        grad_view = grad_view + 2.0 * constraint_weights.unsqueeze(-1) * x_view
        grad = grad_view.reshape(batch_size, -1)
        if proximal_coef is not None and x_outer is not None:
            grad = grad + proximal_coef * (x - x_outer)
        return grad

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None, method="autograd"):
        if method == "explicit":
            return self._explicit_lagrangian_x_gradient(x, dual_var, penalty_coef, proximal_coef, x_outer)
        if method != "autograd":
            raise NotImplementedError
        x_detached = x.detach().requires_grad_(True)
        lagrangian = self.lagrangian_x(x_detached, dual_var, penalty_coef, proximal_coef, x_outer)
        return torch.autograd.grad(lagrangian, x_detached, create_graph=False)[0]

    # def gradient_constraint_x(self, x, clip=True, dual_var=None, method='autograd'):
    #     # violation = (self.constraint_x(x, clip=False) * dual_var).sum(-1)
    #     batch_size = x.shape[0]
    #     x_reshaped = x.view(batch_size, self.num_node, self.rank)
    #     grad_reshaped = 2 * dual_var.unsqueeze(-1) * x_reshaped
    #     # Flatten back to original shape
    #     grad = grad_reshaped.view(batch_size, -1)
    #     return grad

    def gradient_penalty_x(self, x):
        # batch_size = x.shape[0]
        # x_reshaped = x.view(batch_size, self.num_node, self.rank)
        # # X = x @ x^T
        # X = torch.matmul(x_reshaped, x_reshaped.transpose(1, 2))
        # # Diagonal violation: diag(X) - 1
        # diag_elements = torch.diagonal(X, dim1=1, dim2=2)  # (batch_size, num_node)
        # diag_violation = diag_elements - 1  # (batch_size, num_node)
        # # Explicit gradient computation
        # # For penalty L = sum_i (X_ii - 1)^2
        # # dL/dx_ik = 4 * (X_ii - 1) * x_ik
        # # Broadcast diag_violation to match x_reshaped dimensions
        # grad_reshaped = 4 * diag_violation.unsqueeze(-1) * x_reshaped  # (batch_size, num_node, rank)
        # # Flatten back to original shape
        # grad = grad_reshaped.view(batch_size, -1)
        x_detached = x.detach().requires_grad_(True)
        violation = 0.5 * _square(self.constraint_x(x_detached)).sum(-1)
        grad = torch.autograd.grad(violation, x_detached, create_graph=False)[0]
        return grad



###################################################################
# Problem instance generation
###################################################################

def _convex_problem_rng(seed):
    return np.random.RandomState(seed) if seed is not None else np.random.RandomState()


def _positive_margin(rng, size, scale=0.1):
    if isinstance(size, tuple):
        return np.abs(rng.randn(*size)) * scale
    return np.abs(rng.randn(size)) * scale


def _local_distance_margin(rng, gradient_norm, scale=0.1, eps=1e-12):
    return _positive_margin(rng, gradient_norm.shape, scale=scale) * np.maximum(gradient_norm, eps)


def _normalize_rows(matrix, eps=1e-12):
    if matrix is None:
        return None
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def _normalize_soc_tensor(tensor, eps=1e-12):
    if tensor is None:
        return None
    flat = tensor.reshape(tensor.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    return tensor / np.maximum(norms.reshape(-1, 1, 1), eps)


def _sample_diagonal_quadratic_terms(rng, count, num_var, *, quad_diag_lower=1e-2, quad_diag_upper=1.0):
    if quad_diag_lower < 0:
        raise ValueError("quad_diag_lower must be nonnegative.")
    if quad_diag_upper < quad_diag_lower:
        raise ValueError("quad_diag_upper must be greater than or equal to quad_diag_lower.")
    q_diag = rng.uniform(quad_diag_lower, quad_diag_upper, size=(count, num_var))
    q_stack = np.zeros((count, num_var, num_var))
    diag_idx = np.arange(num_var)
    q_stack[:, diag_idx, diag_idx] = q_diag
    return q_stack


def _sample_low_rank_quadratic_terms(rng, count, num_var, *, low_rank_quad_ridge=1e-2):
    if low_rank_quad_ridge < 0:
        raise ValueError("low_rank_quad_ridge must be nonnegative.")
    basis_cols = max(int(num_var**0.5), 1)
    basis = rng.randn(count, num_var, basis_cols)
    q_stack = np.einsum("mnk,mpk->mnp", basis, basis) / num_var
    q_stack += low_rank_quad_ridge * np.eye(num_var)[None, :, :]
    return q_stack


def _sample_quadratic_terms(
    rng,
    count,
    num_var,
    *,
    quadratic_type="diagonal",
    quad_diag_lower=1e-2,
    quad_diag_upper=1.0,
    low_rank_quad_ridge=1e-2,
):
    if quadratic_type == "diagonal":
        return _sample_diagonal_quadratic_terms(
            rng,
            count,
            num_var,
            quad_diag_lower=quad_diag_lower,
            quad_diag_upper=quad_diag_upper,
        )
    if quadratic_type == "low_rank":
        return _sample_low_rank_quadratic_terms(
            rng,
            count,
            num_var,
            low_rank_quad_ridge=low_rank_quad_ridge,
        )
    raise ValueError("quadratic_type must be one of: 'diagonal', 'low_rank'.")


def _sample_convex_objective(
    rng,
    num_var,
    obj,
    *,
    objective_quadratic_type="diagonal",
    quad_diag_lower=1e-2,
    quad_diag_upper=1.0,
    low_rank_quad_ridge=1e-2,
):
    if obj == "quad":
        q_matrix = _sample_quadratic_terms(
            rng,
            1,
            num_var,
            quadratic_type=objective_quadratic_type,
            quad_diag_lower=quad_diag_lower,
            quad_diag_upper=quad_diag_upper,
            low_rank_quad_ridge=low_rank_quad_ridge,
        )[0]
    elif obj == "low_rank_quad":
        q_matrix = _sample_quadratic_terms(
            rng,
            1,
            num_var,
            quadratic_type="low_rank",
            low_rank_quad_ridge=low_rank_quad_ridge,
        )[0]
    elif obj == "linear":
        q_matrix = np.zeros((num_var, num_var))
    else:
        raise ValueError("obj must be one of: 'linear', 'quad', 'low_rank_quad'.")
    linear_term = rng.randn(num_var) / np.sqrt(max(num_var, 1))
    return q_matrix, linear_term


def _sample_box_interior_anchor(rng, num_var, x_lower, x_upper, interior_ratio=0.8):
    if x_upper <= x_lower:
        raise ValueError("x_upper must be strictly greater than x_lower.")
    span = x_upper - x_lower
    center = 0.5 * (x_lower + x_upper)
    half_width = 0.5 * span * interior_ratio
    return center + rng.uniform(-half_width, half_width, size=num_var)


def _sample_linear_equalities(rng, n_lin_eq, num_var, x_ref):
    if n_lin_eq <= 0:
        return None, None
    a_eq = _normalize_rows(rng.randn(n_lin_eq, num_var))
    b_eq = np.dot(a_eq, x_ref)
    return a_eq, b_eq


def _sample_linear_inequalities(rng, num_linear_cons, num_var, x_ref, margin_scale):
    if num_linear_cons <= 0:
        return None, None
    a_matrix = _normalize_rows(rng.randn(num_linear_cons, num_var))
    grad_norm = np.linalg.norm(a_matrix, axis=1)
    b_vector = np.dot(a_matrix, x_ref) + _local_distance_margin(rng, grad_norm, scale=margin_scale)
    return a_matrix, b_vector


def _sample_quadratic_inequalities(
    rng,
    num_qua_cons,
    num_var,
    x_ref,
    margin_scale,
    *,
    constraint_quadratic_type="diagonal",
    quad_diag_lower=1e-2,
    quad_diag_upper=1.0,
    low_rank_quad_ridge=1e-2,
):
    if num_qua_cons <= 0:
        return None, None, None
    q_stack = _sample_quadratic_terms(
        rng,
        num_qua_cons,
        num_var,
        quadratic_type=constraint_quadratic_type,
        quad_diag_lower=quad_diag_lower,
        quad_diag_upper=quad_diag_upper,
        low_rank_quad_ridge=low_rank_quad_ridge,
    )
    p_stack = rng.randn(num_qua_cons, num_var) / np.sqrt(max(num_var, 1))
    q_at_ref = 0.5 * np.einsum("i,mij,j->m", x_ref, q_stack, x_ref)
    p_at_ref = np.einsum("mi,i->m", p_stack, x_ref)
    grad_at_ref = np.einsum("mij,j->mi", q_stack, x_ref) + p_stack
    grad_norm = np.linalg.norm(grad_at_ref, axis=1)
    b_stack = q_at_ref + p_at_ref + _local_distance_margin(rng, grad_norm, scale=margin_scale)
    return q_stack, p_stack, b_stack


def _sample_soc_inequalities(rng, num_soc_cons, num_var, x_ref, margin_scale):
    if num_soc_cons <= 0:
        return None, None, None, None
    soc_dim = num_var
    g_tensor = _normalize_soc_tensor(rng.randn(num_soc_cons, soc_dim, num_var))
    h_tensor = rng.randn(num_soc_cons, soc_dim) / np.sqrt(max(num_var, 1))
    c_matrix = _normalize_rows(rng.randn(num_soc_cons, num_var))
    gx_ref = np.einsum("mkn,n->mk", g_tensor, x_ref) + h_tensor
    d_vector = np.linalg.norm(gx_ref, ord=2, axis=1) - np.dot(c_matrix, x_ref)
    gx_norm = np.linalg.norm(gx_ref, ord=2, axis=1, keepdims=True)
    unit = gx_ref / np.maximum(gx_norm, 1e-12)
    grad_at_ref = np.einsum("mk,mkn->mn", unit, g_tensor) - c_matrix
    d_vector = d_vector + _local_distance_margin(rng, np.linalg.norm(grad_at_ref, axis=1), scale=margin_scale)
    return g_tensor, h_tensor, c_matrix, d_vector


def _sample_box_bounds(num_var, x_lower, x_upper):
    lower = np.full(num_var, x_lower)
    upper = np.full(num_var, x_upper)
    return lower, upper

def _nonnegative_int(params, key, default=None):
    value = params.get(key, default)
    if value is None:
        value = 0
    value = int(value)
    if value < 0:
        raise ValueError(f"{key} must be nonnegative.")
    return value


def _positive_int(params, key):
    value = int(params[key])
    if value < 1:
        raise ValueError(f"{key} must be positive.")
    return value


def _parse_convex_instance_params(params):
    """Normalize and validate deterministic convex instance controls."""

    num_var = _positive_int(params, "n_var")
    x_lower = float(params["x_lower"])
    x_upper = float(params["x_upper"])
    if x_upper <= x_lower:
        raise ValueError("x_upper must be strictly greater than x_lower.")
    margin_scale = float(params.get("margin_scale", 0.1))
    if margin_scale < 0:
        raise ValueError("margin_scale must be nonnegative.")
    anchor_interior_ratio = float(params.get("anchor_interior_ratio", 0.8))
    if not 0 < anchor_interior_ratio <= 1:
        raise ValueError("anchor_interior_ratio must be in (0, 1].")
    return {
        "num_var": num_var,
        "num_linear_cons": _nonnegative_int(params, "n_linear_cons"),
        "num_soc_cons": _nonnegative_int(params, "n_soc_cons"),
        "num_qua_cons": _nonnegative_int(params, "n_qua_cons"),
        "n_lin_eq": _nonnegative_int(params, "n_lin_eq", default=0),
        "x_lower": x_lower,
        "x_upper": x_upper,
        "margin_scale": margin_scale,
        "anchor_interior_ratio": anchor_interior_ratio,
        "quad_diag_lower": float(params.get("quad_diag_lower", 1e-2)),
        "quad_diag_upper": float(params.get("quad_diag_upper", 1.0)),
        "low_rank_quad_ridge": float(params.get("low_rank_quad_ridge", 1e-2)),
        "objective_quadratic_type": str(params.get("objective_quadratic_type", "diagonal")),
        "constraint_quadratic_type": str(params.get("constraint_quadratic_type", "diagonal")),
        "obj": str(params["obj"]),
        "seed": params.get("seed"),
    }


def create_test_problem(params):
    """Create a deterministic convex instance config for `ConvexOpt`/`ConvexOptEq`."""

    controls = _parse_convex_instance_params(params)
    rng = _convex_problem_rng(controls["seed"])

    q_matrix, linear_term = _sample_convex_objective(
        rng,
        controls["num_var"],
        controls["obj"],
        objective_quadratic_type=controls["objective_quadratic_type"],
        quad_diag_lower=controls["quad_diag_lower"],
        quad_diag_upper=controls["quad_diag_upper"],
        low_rank_quad_ridge=controls["low_rank_quad_ridge"],
    )
    x_ref = _sample_box_interior_anchor(
        rng,
        controls["num_var"],
        controls["x_lower"],
        controls["x_upper"],
        interior_ratio=controls["anchor_interior_ratio"],
    )
    a_eq, b_eq = _sample_linear_equalities(rng, controls["n_lin_eq"], controls["num_var"], x_ref)
    a_matrix, b_vector = _sample_linear_inequalities(
        rng,
        controls["num_linear_cons"],
        controls["num_var"],
        x_ref,
        controls["margin_scale"],
    )
    q_stack, p_stack, bq_stack = _sample_quadratic_inequalities(
        rng,
        controls["num_qua_cons"],
        controls["num_var"],
        x_ref,
        controls["margin_scale"],
        constraint_quadratic_type=controls["constraint_quadratic_type"],
        quad_diag_lower=controls["quad_diag_lower"],
        quad_diag_upper=controls["quad_diag_upper"],
        low_rank_quad_ridge=controls["low_rank_quad_ridge"],
    )
    g_tensor, h_tensor, c_matrix, d_vector = _sample_soc_inequalities(
        rng,
        controls["num_soc_cons"],
        controls["num_var"],
        x_ref,
        controls["margin_scale"],
    )
    lower, upper = _sample_box_bounds(controls["num_var"], controls["x_lower"], controls["x_upper"])

    return {
        "Q": q_matrix,
        "p": linear_term,
        "A": a_matrix,
        "b": b_vector,
        "A_eq": a_eq,
        "b_eq": b_eq,
        "Qq": q_stack,
        "pq": p_stack,
        "bq": bq_stack,
        "G": g_tensor,
        "h": h_tensor,
        "C": c_matrix,
        "d": d_vector,
        "L": lower,
        "U": upper,
    }

def create_maxcut_problem(para):
    """Create a test convex optimization problem
    Args:
        n: dimension of node
        alpha: sparsity of edge
    Returns:
        config: dictionary containing problem parameters
    """
    n = para['n']
    alpha = para['alpha']
    config = {
        'n': n,
        'alpha': alpha,
    }
    use_weights = para.get('use_weights', False)
    np.random.seed(para['seed'])
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "create_maxcut_problem requires networkx. Install with: pip install -e .[research]"
        ) from exc

    Graph = nx.erdos_renyi_graph(n, alpha, seed=para['seed'])
    edge = list(Graph.edges())
    if use_weights:
        edge_weight = np.random.rand(len(edge))
    else:
        edge_weight = np.ones(len(edge))
    config['edge'] = edge
    config['node'] = list(range(n))
    config['weights'] = np.array(edge_weight)
    return config
