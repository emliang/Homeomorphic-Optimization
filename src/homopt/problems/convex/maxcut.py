"""Max-Cut SDP and Burer-Monteiro problem families."""

import numpy as np
import torch

from homopt.problems.base import TensorRuntimeMixin

from .core import _square


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

        # Compute eigenvalues.  A failed eigendecomposition is a numerical
        # error in the active solver path, not a large synthetic violation.
        try:
            eigenvals = torch.linalg.eigvalsh(X)
            min_eigenval = torch.min(eigenvals, dim=1, keepdim=True).values
            # Constraint violation is -min_eigenval when positive
            violations = torch.clamp(-min_eigenval, min=0) if clip else -min_eigenval
        except RuntimeError as exc:
            raise RuntimeError("MaxCut PSD eigendecomposition failed.") from exc
        
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
            raise ValueError(f"Unsupported MaxCut constraint gradient method: {method}")
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

    def gradient_objective_z(self, z, hom_map=None, method="autograd", hom_map_method="autograd", x=None, hom_state=None):
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
            if x is None:
                x, hom_state = hom_map.forward(z, method=hom_map_method, return_state=True)
            grad_x = self.gradient_objective_x(x) + 2e-5 * x
            grad_z = hom_map.vjp(z, grad_x, method=hom_map_method, state=hom_state)
        else:
            raise ValueError(f"Unsupported MaxCut z-objective gradient method: {method}")
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
        p_eff = self.gradient_objective_x(center).to(device=x.device, dtype=x.dtype)
        obj = torch.clamp(1 + torch.sum(p_eff * u, dim=-1, keepdim=True), min=0)
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

    def lift_to_sdp_decision(self, factor):
        """Lift Burer-Monteiro factors ``Y`` to the MaxCut SDP coordinates.

        The returned vector contains the strict upper triangle of ``Y Y^T`` in
        the coordinate ordering used by :class:`MaxCutSDP`.  It is intended for
        shared evaluation and reporting, not for the factor-space optimizer.
        """

        if factor.ndim != 2 or factor.shape[1] != self.nvar:
            raise ValueError(
                "BMSDP factor must have shape (batch, num_node * rank); "
                f"received {tuple(factor.shape)}."
            )
        y = factor.reshape(factor.shape[0], self.num_node, self.rank)
        matrix = y @ y.transpose(1, 2)
        indices = torch.as_tensor(self.upper_triangle_index, device=factor.device, dtype=torch.long)
        return matrix[:, indices[:, 0], indices[:, 1]]
    
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
            "create_maxcut_problem requires networkx. Install with: pip install -r requirements.txt"
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

__all__ = ["BMSDP", "MaxCutSDP", "create_maxcut_problem"]
