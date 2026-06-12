"""Deterministic single-instance QCQP problem."""

import torch

from homopt.problems.base import TensorRuntimeMixin

torch.set_default_dtype(torch.float32)


class QCOpt(TensorRuntimeMixin):
    def __init__(self, config) -> None:
        self.prob_para = [
            config['Q'],
            config['p'],
            config['A'],
            config['b'],
            config['Qq'],
            config['pq'],
            config['bq'],
            config['L'],
            config['U'],
            config.get('R', 100)
        ]

        self.Q = torch.as_tensor(config['Q'], dtype=torch.float32)
        self.p = torch.as_tensor(config['p'], dtype=torch.float32)

        self.U = torch.as_tensor(config['U'], dtype=torch.float32)
        self.L = torch.as_tensor(config['L'], dtype=torch.float32)
        # Linear constraints
        self.A = torch.as_tensor(config['A'], dtype=torch.float32) if config['A'] is not None else None # m * n
        self.b = torch.as_tensor(config['b'], dtype=torch.float32) if config['b'] is not None else None # m * 1
        # quadratic constraints
        self.Qq = torch.as_tensor(config['Qq'], dtype=torch.float32) if config['Qq'] is not None else None # m * n * n
        self.pq = torch.as_tensor(config['pq'], dtype=torch.float32) if config['pq'] is not None else None # m * n
        self.bq = torch.as_tensor(config['bq'], dtype=torch.float32) if config['bq'] is not None else None # m * 1
        self.R = torch.as_tensor(config['R'], dtype=torch.float32) if config['R'] is not None else None # 1 * 1

        self.nvar = self.p.shape[0]
        # linear + quadratic + l/u
        self.n_lin = self.A.shape[0] if self.A is not None else 0
        self.n_qua = self.Qq.shape[0] if self.Qq is not None else 0
        self.ncon = self.n_lin + self.n_qua + self.nvar * 2 + 1 # linear + quadratic + box constraints
        self.ineq_cons = range(self.ncon)
        self.eq_cons = None

    def __str__(self):
        return 'NonConvex_QC_Opt'

    def ineq_resid(self, input, x, clip=True):
        """Compute inequality residual"""
        """
        Compute constraint violations for:
        - Linear constraints: Ax <= b
        - Quadratic constraints: 1/2 x^T Qq_i x + pq_i^T x <= bq_i
        - Box constraints: L <= x <= U
        """
        resids = []

        # Linear constraints
        if self.A is not None:
            lin_res = torch.matmul(x, self.A.T) - self.b
            resids.append(lin_res)

        # Quadratic constraints
        if self.Qq is not None:
            q = 0.5 * torch.sum(torch.matmul(self.Qq, x.T).permute(2, 0, 1) * x, dim=-1)
            p = torch.matmul(x, self.pq.T)
            quad_red = q + p - self.bq
            resids.append(quad_red)

        # Box constraints
        lbound = self.L - x
        ubound = x - self.U
        r_ball = (x**2).sum(-1, keepdim=True) - self.R**2
        resids += [lbound, ubound, r_ball]

        resids = torch.cat(resids, dim=1)
        if clip:
            return torch.clamp(resids, min=0)
        else:
            return resids

    def objective_x(self, x):
        return torch.sum(0.5 * (x @ self.Q) * x + self.p * x, dim=-1, keepdim=True)

    def gradient_objective_x(self, x):
        """Compute gradient of objective function with respect to x
        Args:
            x: input points (batch_size x n)
        Returns:
            gradient (batch_size x n)
        """
        sym_q = 0.5 * (self.Q + self.Q.T)
        return x @ sym_q + self.p

    def constraint_x(self, x, clip=True):
        """
        Compute constraint violations for:
        - Linear constraints: Ax <= b
        - Quadratic constraints: 1/2 x^T Qq_i x + pq_i^T x <= bq_i
        - Box constraints: L <= x <= U
        - R-ball constraint: x^2 <= R^2
        """
        resids = []

        # Linear constraints
        if self.A is not None:
            lin_res = torch.matmul(x, self.A.T) - self.b
            resids.append(lin_res)

        # Quadratic constraints
        if self.Qq is not None:
            q = 0.5 * torch.sum(torch.matmul(self.Qq, x.T).permute(2, 0, 1) * x, dim=-1)
            p = torch.matmul(x, self.pq.T)
            quad_red = q + p - self.bq
            resids.append(quad_red)

        # Box constraints
        lbound = self.L - x
        ubound = x - self.U
        resids += [lbound, ubound]

        # R-ball constraint
        r_ball_res = (x**2).sum(-1, keepdim=True) - self.R**2
        resids.append(r_ball_res)

        resids = torch.cat(resids, dim=1)
        if clip:
            return torch.clamp(resids, min=0)
        else:
            return resids

    def gradient_penalty_x(self, x):
        x_detached = x.detach().requires_grad_(True)
        violation = 0.5 * (self.constraint_x(x_detached, clip=True) * self.constraint_x(x_detached, clip=True)).sum(-1)
        grad = torch.autograd.grad(violation, x_detached, create_graph=False)[0]
        return grad

    def _constraint_jacobian_x(self, x):
        batch_size, nvar = x.shape
        grad_components = []

        if self.A is not None:
            grad_components.append(self.A.unsqueeze(0).expand(batch_size, -1, -1))

        if self.Qq is not None:
            sym_q = 0.5 * (self.Qq + self.Qq.transpose(-1, -2))
            grad_quad = torch.einsum("bn,mnk->bmk", x, sym_q) + self.pq.unsqueeze(0)
            grad_components.append(grad_quad)

        eye = torch.eye(nvar, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        grad_components.append(-eye)
        grad_components.append(eye)
        grad_components.append(2.0 * x.unsqueeze(1))

        return torch.cat(grad_components, dim=1)

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
                violation = self.constraint_x(x_detached, clip=clip).sum(-1)
            else:
                violation = (self.constraint_x(x_detached, clip=clip) * dual_var).sum(-1)
            grad = torch.autograd.grad(violation, x_detached, create_graph=False)[0]
        elif method == 'finite_diff':
            epsilon = 1e-6
            grad = torch.zeros_like(x)
            for i in range(x.shape[1]): # iterate over each variable
                delta = torch.zeros_like(x)
                delta[:, i] = epsilon
                obj_plus = self.constraint_x(x + delta, clip=clip)
                obj_minus = self.constraint_x(x - delta, clip=clip)
                grad[:, i] = (obj_plus - obj_minus).sum(-1) / (2 * epsilon)
        else:
            grad = self._constraint_jacobian_x(x)
            if clip:
                active = (self.constraint_x(x, clip=False) > 0).to(dtype=x.dtype).unsqueeze(-1)
                grad = grad * active
            if dual_var is None:
                grad = grad.sum(1)
            else:
                grad = (grad * dual_var.unsqueeze(-1)).sum(1)
        return grad

    def gradient_lagrangian_x(
        self,
        x,
        dual_var,
        *,
        penalty_coef=None,
        proximal_coef=None,
        x_outer=None,
        method="explicit",
    ):
        if method == "autograd":
            x_detached = x.detach().requires_grad_(True)
            residual = self.constraint_x(x_detached, clip=False)
            objective = self.objective_x(x_detached)
            lagrangian = objective
            if dual_var is not None:
                lagrangian = lagrangian + (residual * dual_var).sum(dim=1, keepdim=True)
            if penalty_coef is not None and float(penalty_coef) != 0.0:
                violation = torch.clamp(residual, min=0)
                lagrangian = lagrangian + 0.5 * float(penalty_coef) * (violation * violation).sum(dim=1, keepdim=True)
            if proximal_coef is not None and x_outer is not None and float(proximal_coef) != 0.0:
                lagrangian = lagrangian + 0.5 * float(proximal_coef) * ((x_detached - x_outer) ** 2).sum(dim=1, keepdim=True)
            return torch.autograd.grad(lagrangian.sum(), x_detached, create_graph=False)[0]

        if method != "explicit":
            raise ValueError(f"Unsupported QCOpt lagrangian gradient method: {method}")

        residual = self.constraint_x(x, clip=False)
        jacobian = self._constraint_jacobian_x(x)
        grad = self.gradient_objective_x(x)
        if dual_var is not None:
            grad = grad + torch.einsum("bm,bmn->bn", dual_var, jacobian)
        if penalty_coef is not None and float(penalty_coef) != 0.0:
            violation = torch.clamp(residual, min=0)
            grad = grad + float(penalty_coef) * torch.einsum("bm,bmn->bn", violation, jacobian)
        if proximal_coef is not None and x_outer is not None and float(proximal_coef) != 0.0:
            grad = grad + float(proximal_coef) * (x - x_outer)
        return grad

    def objective_z(self, z, hom_map=None, hom_map_method="autograd"):
        x = hom_map.forward(z, method=hom_map_method)
        return self.objective_x(x)

    def gradient_objective_z(self, z, hom_map=None, method="autograd", hom_map_method="autograd"):
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

__all__ = ["QCOpt"]
