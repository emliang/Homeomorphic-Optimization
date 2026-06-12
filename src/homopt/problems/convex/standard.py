"""Standard deterministic convex optimization problem families."""

import torch

from homopt.problems.base import FixedProblemParametricAdapter, TensorRuntimeMixin

from .core import _load_convex_core_config, _radial_quadratic_dual_value, _square, _tensor_or_none


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
        q_eff = self.Q.to(device=x.device, dtype=x.dtype)
        p = self.p.to(device=x.device, dtype=x.dtype).view(1, -1)
        p_eff = center @ q_eff + p
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

__all__ = ["ConvexOpt", "ConvexOptEq", "ParametricConvexProblem"]
