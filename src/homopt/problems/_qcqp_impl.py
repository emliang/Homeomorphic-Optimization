"""Migrated QCQP/parametric problem implementation from the legacy utils module."""

import torch
import numpy as np

from homopt.problems.base import LearningProblemAdapter, ParametricProblemBase, ProblemInstance, TensorRuntimeMixin

torch.set_default_dtype(torch.float32)


###################################################################
# Non-convex QCQP
# objective function:  1/2 x^T Q x + p^T x
# constraint function: Ax <= b, 1/2 x^T Qq_i x + pq_i^T x <= bq_i (potentially non-convex)
#                      L <= x <= U or x \in B(0,U)
###################################################################
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

def _spectral_quadratic_matrix(eigenvals, *, scale):
    """Build a symmetric quadratic matrix without forming an explicit diagonal matmul."""

    eigenvals = np.asarray(eigenvals, dtype=float)
    nvar = int(eigenvals.shape[0])
    U, _ = np.linalg.qr(np.random.randn(nvar, nvar))
    matrix = np.einsum("ir,r,jr->ij", U, eigenvals, U, optimize=True) / float(scale)
    matrix = 0.5 * (matrix + matrix.T)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("Generated QCQP quadratic matrix contains non-finite values.")
    return matrix


def create_test_QC_problem(para):
    """Create a test non-convex quadratic constraint optimization problem
    Args:
        para: dictionary containing problem parameters
            - n_var: dimension of variable x
            - n_linear_cons: number of linear constraints
            - n_qua_cons: number of quadratic constraints
            - obj: 'quad' for quadratic objective, 'linear' for linear
            - nonconvex_ratio: fraction of quadratic constraints that are non-convex
            - seed: random seed
    Returns:
        config: dictionary containing problem parameters
    """
    num_var = para['n_var']
    num_linear_cons = para['n_linear_cons']
    num_qua_cons = para['n_qua_cons']
    nonconvex_ratio = para.get('nonconvex_ratio', 0.5)  # Default 50% non-convex
    objective_convexity = bool(para.get('objective_convexity', False))
    constraint_convexity = bool(para.get('constraint_convexity', False))


    if para['seed'] is not None:
        np.random.seed(para['seed'])

    """Objective parameters"""
    if para['obj'] == 'quad':
        if objective_convexity:
            eigenvals = np.abs(np.random.randn(num_var)) + 0.1
        else:
            eigenvals = np.random.uniform(-1, 1, num_var)
        Q = _spectral_quadratic_matrix(eigenvals, scale=num_var)
    else:
        Q = np.zeros((num_var, num_var))

    p = np.random.randn(num_var) / num_var
    x0 = np.random.randn(num_var) * 0.1  # Smaller initial point for stability
    # Ensure initial point is within bounds
    x0 = np.clip(x0, para['x_lower'] + 0.1, para['x_upper'] - 0.1)

    """Linear constraint parameters"""
    if num_linear_cons > 0:
        A = np.random.randn(num_linear_cons, num_var)
        b = A @ x0 + np.abs(np.random.randn(num_linear_cons)) * 0.1  # Ensure feasibility
    else:
        A = b = None

    """Non-convex quadratic constraint parameters"""
    if num_qua_cons > 0:
        Qq = np.zeros((num_qua_cons, num_var, num_var))
        pq = np.random.randn(num_qua_cons, num_var) / num_var
        bq = np.zeros(num_qua_cons)

        # Determine which constraints are non-convex
        num_nonconvex = 0 if constraint_convexity else int(num_qua_cons * nonconvex_ratio)
        nonconvex_indices = np.random.choice(num_qua_cons, num_nonconvex, replace=False)

        for i in range(num_qua_cons):
            if i in nonconvex_indices:
                # Create non-convex quadratic constraint (indefinite matrix)
                eigenvals_q = np.random.uniform(-1, 1, num_var)
            else:
                # Create convex quadratic constraint (positive semidefinite)
                eigenvals_q = np.abs(np.random.randn(num_var)) + 0.1  # Ensure positive

            Qq[i] = _spectral_quadratic_matrix(eigenvals_q, scale=num_var)

            # Set bq to ensure initial feasibility with some margin
            quad_val = 0.5 * np.einsum("i,ij,j->", x0, Qq[i], x0, optimize=True) + float(np.dot(pq[i], x0))
            if not np.isfinite(quad_val):
                raise FloatingPointError("Generated QCQP quadratic offset contains a non-finite value.")
            bq[i] = quad_val + np.abs(np.random.randn()) * 0.1
    else:
        Qq = pq = bq = None


    """Box constraint parameters"""
    L = np.ones(num_var) * para['x_lower']
    U = np.ones(num_var) * para['x_upper']
    R = para.get('R', 100)

    config = {
        'Q': Q,
        'p': p,
        # linear constraints
        'A': A,
        'b': b,
        # quadratic constraints (potentially non-convex)
        'Qq': Qq,
        'pq': pq,
        'bq': bq,
        # box constraints
        'L': L,
        'U': U,
        'R': R,
        # additional info
        'x0': x0,  # Initial feasible point
        'nonconvex_ratio': nonconvex_ratio,
        'nonconvex_indices': nonconvex_indices if num_qua_cons > 0 else None
    }

    return config

###################################################################
# Parametric Non-Convex QCQP for MDH training and INN-PGD testing
###################################################################
class NonConvexQCProblem(TensorRuntimeMixin, ParametricProblemBase):
    """
    Generator for non-convex quadratic constraint problems for training MDH mapping.
    This class creates diverse QC problems and provides input samples (quadratic matrices)
    for training the homeomorphic mapping from unit ball to constraint sets.
    """

    def __init__(self, problem_args):
        """
        Initialize the problem generator
        Args:
            n_var: dimension of decision variables
            n_qua_cons: number of quadratic constraints
            n_linear_cons: number of linear constraints
        """
        self.nvar = problem_args['n_var']
        self.n_qua = problem_args['n_qua_cons']
        self.n_lin = problem_args['n_linear_cons']
        self.rank = problem_args.get('rank', self.nvar//2)
        self.constraint_convexity = problem_args.get('constraint_convexity', False)
        self.objective_convexity = problem_args.get('objective_convexity', False)
        # Input parameters: Qq matrices + pq vectors + bq scalars
        # Qq: n_qua_cons * n_var * n_var
        # pq: n_qua_cons * n_var
        # bq: n_qua_cons
        # self.npara = self.n_qua * (self.nvar * self.nvar + self.nvar + 1)  # This is for input parameter dimensions, not constraint count
        self.npara = (self.nvar * self.nvar + self.nvar + 1)  # This is for input parameter dimensions, not constraint count

        # Create fixed base problem parameters (everything except Qq)
        base_para = {
            'n_var': self.nvar,
            'n_linear_cons': self.n_lin,
            'n_qua_cons': self.n_qua,
            'obj': problem_args['obj'],
            'nonconvex_ratio': problem_args['nonconvex_ratio'],  # Fixed ratio
            'x_lower': problem_args['x_lower'],         # Fixed bounds
            'x_upper': problem_args['x_upper'],
            'R': problem_args.get('R', 100),
            'seed': problem_args['seed'],
            'objective_convexity': self.objective_convexity,
            'constraint_convexity': self.constraint_convexity,
        }

        # Create base configuration with fixed parameters
        base_config = create_test_QC_problem(base_para)
        self.base_config = base_config
        # Store fixed parameters (everything except Qq, pq, bq)
        self.fixed_Q = base_config['Q']      # Fixed objective matrix
        self.fixed_p = base_config['p']      # Fixed objective vector
        self.fixed_A = base_config['A']      # Fixed linear constraint matrix
        self.fixed_b = base_config['b']      # Fixed linear constraint vector
        self.fixed_L = base_config['L']      # Fixed lower bounds
        self.fixed_U = base_config['U']      # Fixed upper bounds
        self.fixed_x0 = base_config['x0']    # Fixed initial point
        self.fixed_R = base_config['R']      # Fixed R-ball constraint radius
        self.is_parametric = True
        self.device = torch.device('cpu')
        self.dtype = torch.get_default_dtype()
        # base_problem = QCOpt(base_config)

    def to_dtype(self, dtype):
        """Persist a preferred floating-point dtype for generated samples."""
        self.dtype = dtype
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor) and attr.is_floating_point():
                setattr(self, attr_name, attr.to(dtype=dtype))
        return self

    # def generate_problem_samples(
    #         self,
    #         n_samples,
    #         sample_obj=True,
    #         rng=None,
    #         # Landscape knobs
    #         neg_frac_constr=0.4,  # fraction of negative eigenvalues in constraints when nonconvex
    #         neg_frac_obj=0.5,  # fraction of negative eigenvalues in objective when nonconvex
    #         psd_mix_constr=0.2,  # PSD stabilization mix for constraints
    #         psd_mix_obj=0.1,  # PSD stabilization mix for objective
    #         center_scale=None,  # scale of constraint centers d_q; default ~ radius/2
    #         ball_radius=None,  # size of main feasible region used to set bq scales
    #         rho_scale=(0.4, 1.0),  # relative size of constraint radii rho to the main radius
    #         linear_scale_constr=1.0,  # scaling of pq
    #         linear_scale_obj=None,  # scaling of p
    #         obj_scale=1.0,  # overall scale of Q objective
    #         seed=None
    # ):
    #     """
    #     Modify your existing code to create random QCQPs with many local optima.
    #     Keeps the same final tensor shapes and concatenation scheme:
    #       input_samples = [Qq_flat, pq_flat, bq_flat]
    #       obj_samples   = [Q_flat, p]
    #     This uses centered ellipsoidal constraints (x - d)^T S (x - d) - rho^2 <= 0,
    #     folded into Qq, pq, bq.
    #     """
    #     if rng is None:
    #         rng = np.random.default_rng(seed)
    #     else:
    #         # Allow passing numpy RandomState/Generator
    #         try:
    #             rng = np.random.default_rng(rng.integers(0, 2 ** 32 - 1))
    #         except Exception:
    #             pass
    #
    #     nvar = self.nvar
    #     n_qua = self.n_qua
    #     rank = self.rank
    #
    #     # Defaults for scales
    #     if ball_radius is None:
    #         # Use fixed_U/L if available to infer a scale
    #         if hasattr(self, 'fixed_U') and hasattr(self, 'fixed_L'):
    #             # rough radius as half the range norm
    #             span = np.array(self.fixed_U) - np.array(self.fixed_L)
    #             ball_radius = float(np.linalg.norm(span) / max(1, np.sqrt(nvar)))
    #             ball_radius = max(ball_radius, 1.0)
    #         else:
    #             ball_radius = 5.0
    #     if center_scale is None:
    #         center_scale = 0.5 * ball_radius
    #     if linear_scale_obj is None:
    #         linear_scale_obj = 1.0 / max(1, nvar)
    #
    #     # x0 initial points (kept at zero as in your code)
    #     x0 = rng.normal(size=(n_samples, nvar)) * 0.0
    #
    #     # Constraint quadratics Qq: batched low-rank with optional indefiniteness
    #     Qq = _low_rank_quadratic_batch(
    #         rng=rng,
    #         n_samples=n_samples,
    #         n_qua=n_qua,
    #         nvar=nvar,
    #         rank=rank,
    #         convex=bool(getattr(self, 'constraint_convexity', False)),
    #         neg_frac=neg_frac_constr,
    #         psd_mix=psd_mix_constr,
    #         scale=1.0
    #     )  # (S, Q, n, n)
    #
    #     # Add centered-ellipsoid structure: (x - d)^T S (x - d) - rho^2 <= 0
    #     # We encode into standard quadratic form: x^T S x + (-2 S d)^T x + d^T S d - rho^2 <= 0
    #     d_q = rng.normal(scale=center_scale, size=(n_samples, n_qua, nvar))  # centers
    #     Sd = np.einsum('sqnm,sqm->sqn', Qq, d_q)  # S d, shape (S,Q,n)
    #     pq = -2.0 * Sd
    #     # Radii rho: between rho_scale[0]*R and rho_scale[1]*R
    #     rho = rng.uniform(rho_scale[0] * ball_radius, rho_scale[1] * ball_radius, size=(n_samples, n_qua))
    #     # Constant term: d^T S d - rho^2
    #     dSd = np.einsum('sqm,sqn->sq', d_q, Sd)  # (S,Q)
    #     bq = (dSd - rho ** 2)[..., np.newaxis]  # (S,Q,1)
    #
    #     # Optionally add a ball constraint as one of the quadratics if you want a compact set
    #     # Example (commented): replace first constraint with ||x||^2 - R^2 <= 0
    #     # I = np.eye(nvar)
    #     # Qq[:, 0] = I
    #     # pq[:, 0] = 0.0
    #     # bq[:, 0, 0] = -ball_radius**2
    #
    #     # Add modest linear noise to pq to break symmetry and add complexity
    #     if linear_scale_constr != 0.0:
    #         pq = pq + linear_scale_constr * rng.normal(size=pq.shape)
    #
    #     # Flatten and concatenate constraint parameters
    #     Qq_flat = Qq.reshape(n_samples, n_qua, -1)  # (S,Q,n*n)
    #     pq_flat = pq.reshape(n_samples, n_qua, -1)  # (S,Q,n)
    #     bq_flat = bq.reshape(n_samples, n_qua, -1)  # (S,Q,1)
    #     input_samples_np = np.concatenate([Qq_flat, pq_flat, bq_flat], axis=2)  # (S,Q,n*n+n+1)
    #     input_samples = torch.tensor(input_samples_np, dtype=torch.float32)

        # obj_samples = None
        # if sample_obj:
        #     # Objective matrix Q
        #     Q_obj = _low_rank_quadratic_batch(
        #         rng=rng,
        #         n_samples=n_samples,
        #         n_qua=1,
        #         nvar=nvar,
        #         rank=rank,
        #         convex=bool(getattr(self, 'objective_convexity', False)),
        #         neg_frac=neg_frac_obj,
        #         psd_mix=psd_mix_obj,
        #         scale=obj_scale
        #     )[:, 0, :, :]  # (S, n, n)
        #
        #     # Random linear term p
        #     p = rng.normal(size=(n_samples, nvar)) * linear_scale_obj
        #
        #     Q_flat = Q_obj.reshape(n_samples, -1)
        #     obj_np = np.concatenate([Q_flat, p], axis=1)
        #     obj_samples = torch.tensor(obj_np, dtype=torch.float32)
        #
        #     self.obj_samples = obj_samples
        #
        #     self.input_samples = input_samples
        # return input_samples, obj_samples

    def generate_problem_samples_torch(self, n_samples=1000, seed=None, sample_obj=True, device=None, dtype=None):
        """
        Generate diverse QC problem instances for training.
        Only quadratic constraint parameters (Qq, pq, bq) are varied, all other parameters are fixed.

        Args:
            n_samples: number of different problem instances to generate
            seed: random seed for reproducibility
        Returns:
            input_samples: tensor of shape (n_samples, ncon) containing flattened [Qq, pq, bq] parameters
            base_problem: single QCOpt problem instance with fixed parameters
        """

        if seed is not None:
            torch.manual_seed(seed)

        device = self.device if device is None else torch.device(device)
        dtype = self.dtype if dtype is None else dtype

        """ Random Constraint Parameter """
        # Generate all x0 at once: (n_samples, nvar)
        # x0 = torch.zeros(n_samples, self.nvar)
        # x0 = torch.clamp(x0, self.fixed_L + 1, self.fixed_U - 1)

        # Generate all eigenvalues at once: (n_samples, n_qua, rank)
        if self.constraint_convexity:
            eigenvals = torch.abs(torch.randn(n_samples, self.n_qua, self.rank, device=device, dtype=dtype))
        else:
            eigenvals = torch.randn(n_samples, self.n_qua, self.rank, device=device, dtype=dtype)

        # Generate all random matrices U_q at once: (n_samples, n_qua, nvar, rank)
        U_q = torch.randn(n_samples, self.n_qua, self.nvar, self.rank, device=device, dtype=dtype)
        U_q = U_q / torch.norm(U_q, dim=-2, keepdim=True)
        # Scale U_q by eigenvalues
        U_q_scaled = U_q * eigenvals.unsqueeze(-2)
        # Matrix multiplication using einsum
        Qq = torch.einsum('snir,snjr->snij', U_q_scaled, U_q) / self.nvar

        # Generate all pq at once: (n_samples, n_qua, nvar)
        pq = torch.randn(n_samples, self.n_qua, self.nvar, device=device, dtype=dtype) / self.nvar

        # Generate all bq at once: (n_samples, n_qua)
        bq = (torch.norm(pq, dim=-1, keepdim=True) \
              + torch.max(torch.abs(eigenvals), dim=-1, keepdim=True)[0] ** 2)

        # Reshape and concatenate: (n_samples, n_qua, nvar*nvar + nvar + 1)
        Qq_flat = Qq.view(n_samples, self.n_qua, -1)
        pq_flat = pq.view(n_samples, self.n_qua, -1)
        bq_flat = bq.view(n_samples, self.n_qua, -1)

        input_samples = torch.cat([Qq_flat, pq_flat, bq_flat], dim=2)

        ################################################################################
        obj_samples = None
        if sample_obj:
            """ Random Objective Parameter """
            # Generate all eigenvalues at once: (n_samples, rank)
            if self.objective_convexity:
                obj_eigenvals = torch.abs(torch.randn(n_samples, self.rank, device=device, dtype=dtype))
            else:
                obj_eigenvals = torch.randn(n_samples, self.rank, device=device, dtype=dtype)

            # Generate all random matrices U at once: (n_samples, nvar, rank)
            U = torch.randn(n_samples, self.nvar, self.rank, device=device, dtype=dtype)
            U = U / torch.norm(U, dim=-2, keepdim=True)
            U_scaled = U * obj_eigenvals.unsqueeze(1)  # broadcast eigenvals
            Q = torch.einsum('sir,sjr->sij', U_scaled, U) / (self.nvar)

            # Generate all p at once: (n_samples, nvar)
            p = torch.randn(n_samples, self.nvar, device=device, dtype=dtype) / self.nvar

            # Concatenate Q and p: (n_samples, nvar*nvar + nvar)
            Q_flat = Q.view(n_samples, -1)
            obj_samples = torch.cat([Q_flat, p], dim=1)

            self.obj_samples = obj_samples
            self.input_samples = input_samples

        return input_samples, obj_samples


    def generate_problem_samples(self, n_samples=1000, seed=None, sample_obj=True, device=None, dtype=None):
        """
        Generate diverse QC problem instances for training.
        Only quadratic constraint parameters (Qq, pq, bq) are varied, all other parameters are fixed.

        Args:
            n_samples: number of different problem instances to generate
            seed: random seed for reproducibility
        Returns:
            input_samples: tensor of shape (n_samples, ncon) containing flattened [Qq, pq, bq] parameters
            base_problem: single QCOpt problem instance with fixed parameters
        """

        if seed is not None:
            np.random.seed(seed)

        device = self.device if device is None else torch.device(device)
        dtype = self.dtype if dtype is None else dtype

        """ Random Constraint Parameter """
        # Generate all x0 at once: (n_samples, nvar)
        # x0 = np.random.randn(n_samples, self.nvar) * 0.
        # x0 = np.clip(x0, self.fixed_L + 1, self.fixed_U - 1)

        # Generate all eigenvalues at once: (n_samples, n_qua, rank)
        if self.constraint_convexity:
            eigenvals = np.abs(np.random.randn(n_samples, self.n_qua, self.rank))
        else:
            eigenvals = np.random.randn(n_samples, self.n_qua, self.rank)

        # Generate all random matrices U_q at once: (n_samples, n_qua, nvar, rank)
        U_q = np.random.randn(n_samples, self.n_qua, self.nvar, self.rank)
        U_q = U_q / np.linalg.norm(U_q, axis=-2, keepdims=True)
        # First: U_q @ diag(eigenvals) -> (n_samples, n_qua, nvar, rank)
        # Then: (U_q @ diag(eigenvals)) @ U_q.T -> (n_samples, n_qua, nvar, nvar)
        # Qq = np.matmul(U_q * eigenvals[:, :, np.newaxis, :], np.transpose(U_q, (0, 1, 3, 2)))/ self.nvar
        U_q_scaled = U_q * eigenvals[:, :, np.newaxis, :]
        Qq = np.einsum('snir,snjr->snij', U_q_scaled, U_q) / self.nvar
        # Generate all pq at once: (n_samples, n_qua, nvar)
        pq = np.random.randn(n_samples, self.n_qua, self.nvar) / self.nvar

        # Compute all quad_val at once: (n_samples, n_qua)
        # 0.5 * x0.T @ Qq @ x0 + pq @ x0
        # quad_val = 0.5 * np.einsum('si,snij,sj->sn', x0, Qq, x0) + np.einsum('sni,si->sn', pq, x0)

        # Generate all bq at once: (n_samples, n_qua)
        bq = (np.linalg.norm(pq, axis=-1, keepdims=True) \
             + np.max(np.abs(eigenvals), axis=-1, keepdims=True) **2)
        # bq = np.abs(quad_val) + np.abs(np.random.randn(n_samples, self.n_qua)) * self.fixed_U[0]

        # Reshape and concatenate: (n_samples, n_qua, nvar*nvar + nvar + 1)
        Qq_flat = Qq.reshape(n_samples, self.n_qua, -1)
        pq_flat = pq.reshape(n_samples, self.n_qua, -1)
        bq_flat = bq.reshape(n_samples, self.n_qua, -1)

        input_samples = np.concatenate([Qq_flat, pq_flat, bq_flat], axis=2)
        input_samples = torch.tensor(input_samples, dtype=dtype, device=device)

        ################################################################################
        obj_samples = None
        if sample_obj:
            """ Random Objective Parameter """
            # Generate all eigenvalues at once: (n_samples, rank)
            if self.objective_convexity:
                obj_eigenvals = np.abs(np.random.randn(n_samples, self.rank))
            else:
                obj_eigenvals = np.random.randn(n_samples, self.rank)

            # Generate all random matrices U at once: (n_samples, nvar, rank)
            U = np.random.randn(n_samples, self.nvar, self.rank)
            U = U / np.linalg.norm(U, axis=-2, keepdims=True)
            U_scaled = U * obj_eigenvals[:, np.newaxis, :]  # broadcast eigenvals
            Q = np.einsum('sir,sjr->sij', U_scaled, U) / (self.nvar)
            # Q = np.matmul(U * obj_eigenvals[:, np.newaxis, :], np.transpose(U, (0, 2, 1))) / self.nvar

            # Generate all p at once: (n_samples, nvar)
            p = np.random.randn(n_samples, self.nvar) / self.nvar

            # Concatenate Q and p: (n_samples, nvar*nvar + nvar)
            Q_flat = Q.reshape(n_samples, -1)
            obj_samples = np.concatenate([Q_flat, p], axis=1)
            obj_samples = torch.tensor(obj_samples, dtype=dtype, device=device)

            self.obj_samples = obj_samples
            self.input_samples = input_samples

        return input_samples, obj_samples

    # def generate_problem_samples(self, n_samples=1000, seed=None, sample_obj=True):
    #     """
    #     Generate diverse QC problem instances for training.
    #     Only quadratic constraint parameters (Qq, pq, bq) are varied, all other parameters are fixed.
    #
    #     Args:
    #         n_samples: number of different problem instances to generate
    #         seed: random seed for reproducibility
    #     Returns:
    #         input_samples: tensor of shape (n_samples, ncon) containing flattened [Qq, pq, bq] parameters
    #         base_problem: single QCOpt problem instance with fixed parameters
    #     """
    #
    #
    #     if seed is not None:
    #         np.random.seed(seed)
    #
    #     """ Random Constraint Parameter """
    #     input_samples = []
    #     for i in range(n_samples):
    #         # Generate diverse quadratic constraint parameters: Qq, pq, bq
    #         Qq = np.zeros((self.n_qua, self.nvar, self.nvar))
    #         pq = np.zeros((self.n_qua, self.nvar))
    #         bq = np.zeros(self.n_qua)
    #
    #         x0 = np.random.randn(self.nvar) * 0.01
    #         # Ensure initial point is within bounds
    #         x0 = np.clip(x0, self.fixed_L + 1, self.fixed_U - 1)
    #         # U_q, _ = np.linalg.qr(np.random.randn(self.nvar, self.nvar))
    #
    #         for j in range(self.n_qua):
    #             # Generate random eigenvalues (mix of positive and negative for non-convexity)
    #             if self.constraint_convexity:
    #                 eigenvals = np.abs(np.random.randn(self.rank))
    #             else:
    #                 eigenvals = np.random.randn(self.rank)
    #             U_q = np.random.randn(self.nvar, self.rank)
    #             Qq[j] = U_q @ np.diag(eigenvals) @ U_q.T / self.nvar
    #             # Generate random linear term for quadratic constraint
    #             pq[j] = np.random.randn(self.nvar) / self.nvar
    #             # Generate feasible bound: bq > current constraint value at x0
    #             quad_val = 0.5 * x0.T @ Qq[j] @ x0 + pq[j] @ x0
    #             bq[j] = quad_val + np.abs(np.random.randn()) * 10
    #
    #         # Concatenate all parameters into single input vector
    #         # n_quad * (n_var * n_var + n_var + 1)
    #         input_vector = np.concatenate([Qq.reshape(self.n_qua, -1),
    #                                        pq.reshape(self.n_qua, -1),
    #                                        bq.reshape(self.n_qua, -1)], axis=1)
    #         # input_vector = np.concatenate([Qq.flatten(), pq.flatten(), bq.flatten()])
    #         input_samples.append(input_vector)
    #
    #     input_samples = torch.tensor(np.array(input_samples), dtype=torch.float32)
    #
    #     obj_samples = None
    #     if sample_obj:
    #         """ Random Objective Parameter """
    #         obj_samples = []
    #         # U, _ = np.linalg.qr(np.random.randn(self.nvar, self.nvar))
    #         for i in range(n_samples):
    #             # Generate potentially non-convex objective
    #             if self.objective_convexity:
    #                 eigenvals = np.abs(np.random.randn(self.rank))
    #             else:
    #                 eigenvals = np.random.randn(self.rank)  # Mix of positive and negative eigenvalues
    #             # # Ensure at least one negative eigenvalue for non-convexity
    #             # U = np.random.randn(self.nvar, self.nvar)
    #             U = np.random.randn(self.nvar, self.rank)
    #             Q = U @ np.diag(eigenvals) @ U.T / self.nvar
    #             # Q = U @ np.diag(eigenvals) @ U.T / self.nvar
    #             p = np.random.randn(self.nvar) / self.nvar
    #             input_vector = np.concatenate([Q.flatten(), p.flatten()])
    #             obj_samples.append(input_vector)
    #         obj_samples = torch.tensor(np.array(obj_samples), dtype=torch.float32)
    #         self.obj_samples = obj_samples
    #         self.input_samples = input_samples
    #     return input_samples, obj_samples

    def formulate_problem(self, input_param, objective_param=None):
        # qq_size = self.n_qua * self.nvar * self.nvar
        # pq_size = self.n_qua * self.nvar
        # bq_size = self.n_qua
        # Unpack input parameters for this instance
        # input_vector = input_param.view(-1).cpu().numpy().astype(np.float32)
        # qq_flat = input_vector[:qq_size]
        # pq_flat = input_vector[qq_size:qq_size + pq_size]
        # bq_flat = input_vector[qq_size + pq_size:qq_size + pq_size + bq_size]

        # Reshape to original dimensions
        # Qq = qq_flat.reshape(self.n_qua, self.nvar, self.nvar).astype(np.float32)
        # pq = pq_flat.reshape(self.n_qua, self.nvar).astype(np.float32)
        # bq = bq_flat.reshape(self.n_qua).astype(np.float32)

        qq_size = self.nvar * self.nvar
        pq_size = self.nvar
        bq_size = 1
        Qq = input_param[:, :qq_size]
        pq = input_param[:, qq_size:qq_size + pq_size]
        bq = input_param[:, qq_size + pq_size:qq_size + pq_size + bq_size]
        Qq = Qq.reshape(self.n_qua, self.nvar, self.nvar).astype(np.float32)
        pq = pq.reshape(self.n_qua, self.nvar).astype(np.float32)
        bq = bq.reshape(self.n_qua).astype(np.float32)

        if objective_param is not None:
            objective_param = np.asarray(objective_param, dtype=np.float32).reshape(-1)
            q_size = self.nvar * self.nvar
            Q = objective_param[:q_size].reshape(self.nvar, self.nvar).astype(np.float32)
            p = objective_param[q_size:q_size + self.nvar].astype(np.float32)
        else:
            Q = np.array(self.fixed_Q, dtype=np.float32)
            p = np.array(self.fixed_p, dtype=np.float32)

        # Create problem configuration for this instance (ensure consistent dtypes)
        config = {
            'Q': Q,    # Objective matrix
            'p': p,    # Objective vector
            'A': np.array(self.fixed_A, dtype=np.float32) if self.fixed_A is not None else None,    # Linear constraint matrix
            'b': np.array(self.fixed_b, dtype=np.float32) if self.fixed_b is not None else None,    # Linear constraint vector
            'Qq': Qq.astype(np.float32),             # Quadratic constraint matrices
            'pq': pq.astype(np.float32),             # Quadratic constraint vectors
            'bq': bq.astype(np.float32),             # Quadratic constraint bounds
            'L': np.array(self.fixed_L, dtype=np.float32),    # Lower bounds
            'U': np.array(self.fixed_U, dtype=np.float32),     # Upper bounds
            'R': np.array(self.fixed_R, dtype=np.float32)     # R-ball constraint radius

        }
        # Create QCOpt problem instance
        problem = QCOpt(config)
        return problem

    def build_instance(self, input_data, objective_data=None):
        device = input_data.device if torch.is_tensor(input_data) else self.device
        if torch.is_tensor(input_data):
            input_param = input_data.detach().cpu().numpy().astype(np.float32)
        else:
            input_param = np.asarray(input_data, dtype=np.float32)
        if objective_data is None:
            objective_param = None
        elif torch.is_tensor(objective_data):
            objective_param = objective_data.detach().cpu().numpy().astype(np.float32)
        else:
            objective_param = np.asarray(objective_data, dtype=np.float32)
        problem = self.formulate_problem(input_param, objective_param=objective_param)
        if hasattr(problem, "to_device"):
            problem = problem.to_device(device)
        if hasattr(problem, "to_dtype"):
            problem = problem.to_dtype(self.dtype)
        return problem

    def bind_instance(self, instance):
        if isinstance(instance, ProblemInstance):
            return self.build_instance(instance.input_data, objective_data=instance.objective_data)
        return self.build_instance(instance)

    def scale(self, input_batch, x):
        """Scale from [0, 1] to original variable space using fixed bounds"""
        # Use the fixed bounds from the base problem
        fixed_L = torch.as_tensor(self.fixed_L, dtype=x.dtype, device=x.device)
        fixed_U = torch.as_tensor(self.fixed_U, dtype=x.dtype, device=x.device)
        # Scale from [-1, 1] to [L, U]
        return (x + 1) / 2 * (fixed_U - fixed_L) + fixed_L

    def complete_partial(self, input_batch, x):
        """Complete partial decision to full decision (identity for this case)"""
        return x

    def ineq_resid(self, input_batch, x):
        """
        Compute inequality constraint residuals using variable Qq, pq, bq and fixed other parameters.

        OPTIMIZED VERSION: Uses batch computation instead of iteration for significant performance gains.

        Key optimizations:
        - Vectorized parameter extraction and reshaping
        - Einstein summation (einsum) for efficient batch matrix operations
        - Single-pass constraint evaluation for all samples
        - Eliminated for-loops over batch dimension

        Args:
            input_batch: tensor of shape (batch_size, ncon) containing [Qq, pq, bq] parameters
            x: tensor of shape (batch_size, nvar) containing decision variables

        Returns:
            violations: tensor of shape (batch_size, total_constraints) containing constraint violations
        """
        batch_size = x.shape[0]
        input_size = input_batch.shape[0]

        if batch_size != input_size:
            # Handle broadcasting if needed
            if input_size == 1:
                input_batch = input_batch.repeat(batch_size, 1)
            else:
                raise ValueError(f"Batch size mismatch: x has {batch_size}, input has {input_size}")

        # Convert fixed parameters to tensors on the same device
        device = x.device
        fixed_L = torch.as_tensor(self.fixed_L, dtype=x.dtype, device=device)
        fixed_U = torch.as_tensor(self.fixed_U, dtype=x.dtype, device=device)

        # Calculate sizes for unpacking input parameters
        # qq_size = self.n_qua * self.nvar * self.nvar
        # pq_size = self.n_qua * self.nvar
        # bq_size = self.n_qua


        # Extract all parameters in batch
        # input_batch shape: (batch_size, n_qua_cons * (n_var*n_var + n_var + 1))
        # qq_flat_batch = input_batch[:, :qq_size]  # (batch_size, n_qua * n_var * n_var)
        # pq_flat_batch = input_batch[:, qq_size:qq_size + pq_size]  # (batch_size, n_qua * n_var)
        # bq_flat_batch = input_batch[:, qq_size + pq_size:qq_size + pq_size + bq_size]  # (batch_size, n_qua)
        # Reshape to original dimensions
        # Qq_batch = qq_flat_batch.reshape(batch_size, self.n_qua, self.nvar, self.nvar)  # (batch_size, n_qua, n_var, n_var)
        # pq_batch = pq_flat_batch.reshape(batch_size, self.n_qua, self.nvar)  # (batch_size, n_qua, n_var)
        # bq_batch = bq_flat_batch.reshape(batch_size, self.n_qua)  # (batch_size, n_qua)

        qq_size = self.nvar * self.nvar
        pq_size = self.nvar
        bq_size = 1

        Qq_batch = input_batch[:, :, :qq_size]
        pq_batch = input_batch[:, :, qq_size:qq_size + pq_size]
        bq_batch = input_batch[:, :, qq_size + pq_size:qq_size + pq_size + bq_size]
        Qq_batch = Qq_batch.view(batch_size, self.n_qua, self.nvar, self.nvar)
        pq_batch = pq_batch.view(batch_size, self.n_qua, self.nvar)
        bq_batch = bq_batch.view(batch_size, self.n_qua)

        # Compute quadratic constraint violations in batch: 0.5 * x^T * Qq_j * x + pq_j^T * x <= bq_j
        # x shape: (batch_size, n_var)
        # Qq_batch shape: (batch_size, n_qua, n_var, n_var)

        # Expand x for batch matrix multiplication: (batch_size, 1, n_var) -> (batch_size, n_qua, n_var)
        x_expanded = x.unsqueeze(1).expand(-1, self.n_qua, -1)  # (batch_size, n_qua, n_var)

        # Quadratic terms: 0.5 * x^T * Qq * x
        # Using einsum for efficient batch computation: 'bji,bjik,bjk->bj'
        quad_vals = 0.5 * torch.einsum('bji,bjik,bjk->bj', x_expanded, Qq_batch, x_expanded)  # (batch_size, n_qua)

        # Linear terms: pq^T * x
        # Using einsum: 'bjk,bk->bj'
        linear_vals = torch.einsum('bjk,bk->bj', pq_batch, x)  # (batch_size, n_qua)

        # Total quadratic constraint values
        total_quad_vals = quad_vals + linear_vals  # (batch_size, n_qua)
        quad_violations = total_quad_vals - bq_batch  # (batch_size, n_qua)

        # Box constraints using fixed bounds (batch computation)
        lbound_viol = fixed_L.unsqueeze(0) - x  # (batch_size, n_var)
        ubound_viol = x - fixed_U.unsqueeze(0)  # (batch_size, n_var)

        # Linear constraints (if any)
        if hasattr(self, 'fixed_A') and self.fixed_A is not None:
            fixed_A = torch.as_tensor(self.fixed_A, dtype=x.dtype, device=device)
            fixed_b = torch.as_tensor(self.fixed_b, dtype=x.dtype, device=device)
            lin_vals = torch.matmul(x, fixed_A.T)  # (batch_size, n_lin)
            lin_violations = lin_vals - fixed_b.unsqueeze(0)  # (batch_size, n_lin)
            all_viols = torch.cat([quad_violations, lin_violations, lbound_viol, ubound_viol], dim=1)
        else:
            all_viols = torch.cat([quad_violations, lbound_viol, ubound_viol], dim=1)

        # R-ball constraint
        r_ball_viol = (x**2).sum(-1, keepdim=True) - self.fixed_R**2  # (batch_size, 1)
        all_viols = torch.cat([all_viols, r_ball_viol], dim=1)

        return all_viols

    def objective(self, x):
        """Compute objective function"""
        fixed_Q = torch.as_tensor(self.fixed_Q, dtype=x.dtype, device=x.device)
        fixed_p = torch.as_tensor(self.fixed_p, dtype=x.dtype, device=x.device)
        return 0.5 * torch.sum(x @ fixed_Q * x, dim=1) + torch.sum(fixed_p * x, dim=1)

    def objective_xy(self, input_batch, x, objective_batch=None):
        """Learning-route objective view on the same QCQP decision variable."""
        del input_batch
        if objective_batch is None:
            return self.objective(x).view(-1, 1)
        if not torch.is_tensor(objective_batch):
            objective_batch = torch.as_tensor(objective_batch, dtype=x.dtype, device=x.device)
        else:
            objective_batch = objective_batch.to(dtype=x.dtype, device=x.device)
        if objective_batch.ndim == 1:
            objective_batch = objective_batch.view(1, -1)
        if objective_batch.shape[0] != x.shape[0]:
            if objective_batch.shape[0] == 1:
                objective_batch = objective_batch.expand(x.shape[0], -1)
            else:
                raise ValueError(
                    f"Objective batch size mismatch: x has {x.shape[0]}, objective has {objective_batch.shape[0]}"
                )
        q_size = self.nvar * self.nvar
        q_batch = objective_batch[:, :q_size].reshape(x.shape[0], self.nvar, self.nvar)
        p_batch = objective_batch[:, q_size:q_size + self.nvar]
        quad = 0.5 * torch.einsum("bi,bij,bj->b", x, q_batch, x)
        linear = torch.einsum("bi,bi->b", p_batch, x)
        return (quad + linear).view(-1, 1)

    def constraint_residual_xy(self, input_batch, x, clip=True):
        """Learning-route constraint residual view using the parametric QC instance input."""
        residual = self.ineq_resid(input_batch, x)
        return torch.clamp(residual, min=0.0) if clip else residual

    def violations(self, input_batch, x):
        """Compute violations"""
        return self.ineq_resid(input_batch, x).max(dim=1)[0].view(-1, 1)

    def check_feasibility(self, input_batch, x):
        """Check feasibility of points"""
        violations = self.ineq_resid(input_batch, x)
        return torch.clamp(violations, min=0)


class NonConvexQCLearningProblem(LearningProblemAdapter):
    """Compatibility shim for legacy callers that still construct a QCQP learning view."""

    def __init__(self, qc_problem):
        self.qc_problem = qc_problem
        super().__init__(qc_problem, name=type(qc_problem).__name__)





def _random_indefinite_eigs(rng, size, neg_frac=0.5, pos_range=(0.5, 2.0), neg_range=(0.5, 2.0)):
    """
    Draw mixed-sign eigenvalues with a target negative fraction.
    Returns shape = size, values shuffled.
    """
    total = np.prod(size)
    num_neg = int(np.round(neg_frac * total))
    num_pos = total - num_neg
    pos = rng.uniform(pos_range[0], pos_range[1], size=num_pos)
    neg = -rng.uniform(neg_range[0], neg_range[1], size=num_neg)
    vals = np.concatenate([pos, neg])
    rng.shuffle(vals)
    return vals.reshape(size)

def _random_orthonormal(rng, n, k):
    """
    Returns U in R^{n x k} with orthonormal columns (QR of Gaussian).
    """
    A = rng.normal(size=(n, k))
    # QR; ensure deterministic sign on R diagonal to stabilize
    Q, R = np.linalg.qr(A, mode='reduced')
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    Q = Q * signs
    return Q

def _low_rank_quadratic_batch(rng, n_samples, n_qua, nvar, rank,
                              convex=False, neg_frac=0.5,
                              psd_mix=0.3, scale=1.0):
    """
    Build batched symmetric Q matrices via low-rank U diag(eig) U^T,
    optionally mixed with a PSD component to stabilize conditioning.
    Returns Q: (S, Q, n, n)
    """
    # Orthonormal U for better conditioning
    # We draw different U per (sample, constraint) for variety
    U_q = np.zeros((n_samples, n_qua, nvar, rank))
    for s in range(n_samples):
        for q in range(n_qua):
            U_q[s, q] = _random_orthonormal(rng, nvar, rank)

    if convex:
        # strictly nonnegative eigenvalues
        eigenvals = np.abs(rng.normal(loc=1.0, scale=0.5, size=(n_samples, n_qua, rank)))
    else:
        eigenvals = _random_indefinite_eigs(rng, (n_samples, n_qua, rank), neg_frac=neg_frac)

    # Base indefinite/PSD component
    U_scaled = U_q * eigenvals[:, :, np.newaxis, :]  # (S,Q,n,r)
    Q_base = np.einsum('sqnr,sqmr->sqnm', U_scaled, U_q)  # (S,Q,n,n)

    # Optional PSD mix-in for stability: add A^T A scaled
    if psd_mix > 0.0:
        A = rng.normal(size=(n_samples, n_qua, nvar, rank))
        PSD = np.einsum('sqnr,sqmr->sqnm', A, A)  # (S,Q,n,n), PSD
        Q = (1.0 - psd_mix) * Q_base + psd_mix * PSD
    else:
        Q = Q_base

    # Symmetrize and scale
    Q = 0.5 * (Q + np.swapaxes(Q, -1, -2))
    Q = Q / max(1, nvar) * scale
    return Q





    
###################################################################
# chance constrained optimization
###################################################################
class ChanceConstraint_Opt(TensorRuntimeMixin):
    def __init__(self, config) -> None:
        """
        min_x 1/2 px
        s.t.  L < x < U
             1/N sum( I (Ax < b + w_i) ) > 1-\\delta,
        """
        self.prob_para = [
            config['Q'],
            config['p'],
            config['A'], 
            config['b'],
            config.get('L'),
            config.get('U'),
            config['b_noise_samples'],
            config['A_noise_samples'],
            config['delta']
        ]
        
        self.Q = torch.as_tensor(config['Q'])
        self.p = torch.as_tensor(config['p'])
        self.A = torch.as_tensor(config['A'])
        self.b = torch.as_tensor(config['b'])
        self.L = torch.as_tensor(config['L'])
        self.U = torch.as_tensor(config['U'])
        self.b_noise_samples = torch.as_tensor(config['b_noise_samples'])  # w_i samples
        self.A_noise_samples = torch.as_tensor(config['A_noise_samples'])  # w_i samples
        self.delta = config['delta']  # probability threshold
        
        self.nvar = self.p.shape[0]
        self.ncon = 2 * self.nvar + 1  # L/U bounds + chance constraint

    def objective_x(self, x):
        """Compute the objective function value
        Args:
            x: input points (batch_size x n)
        Returns:
            objective value (batch_size x 1)
        """
        return torch.sum(0.5 * (x @ self.Q) * x + self.p * x, dim=-1, keepdim=True)
    
    def gradient_objective_x(self, x):
        """Compute gradient of objective function
        Args:
            x: input points (batch_size x n)
        Returns:
            gradient (batch_size x n)
        """
        return x @ self.Q + self.p
    
    def constraint_x(self, x):
        """Compute constraint violations
        Args:
            x: input points (batch_size x n)
        Returns:
            violations (batch_size x num_constraints)
        """
        # Box constraints
        lbound = self.L - x
        ubound = x - self.U
        
        # Chance constraint: 1/N sum(I(Ax < b + w_i)) > 1-delta
        # For each noise sample, check if Ax < b + w_i
        N = self.b_noise_samples.shape[0]
        batch_size = x.shape[0]
        
        # Compute Ax for each batch element
        # Ax = torch.matmul(x, self.A.T)  # batch_size x constraints
        
        # Extend b for broadcasting
        b_expanded = self.b.unsqueeze(0).expand(batch_size, -1)  # batch_size x constraints
        
        # Check constraint satisfaction for each noise sample
        satisfied_count = torch.zeros(batch_size, 1, device=x.device)
        
        for i in range(N):
            b_noise = self.b_noise_samples[i]  # batch_size x constraints
            A_noise = self.A_noise_samples[i]
            is_satisfied = (torch.matmul(x, (self.A+A_noise).T) < b_expanded + b_noise).all(dim=1, keepdim=True).float()
            satisfied_count += is_satisfied
        
        # Probability of satisfaction
        prob_satisfied = satisfied_count / N
        
        # Violation: we want prob_satisfied > 1-delta
        # So violation is (1-delta) - prob_satisfied when positive
        chance_violation = torch.clamp((1 - self.delta) - prob_satisfied, min=0)
        
        # Combine all constraints
        return torch.cat([lbound, ubound, chance_violation], dim=1)
    
    def objective_z(self, z, hom_map=None, hom_map_method="autograd"):
        """Compute objective in transformed space
        Args:
            z: input points in unit ball (batch_size x n)
            hom_map: homeomorphism mapping from unit ball to feasible set
        Returns:
            objective value (batch_size x 1)
        """
        x = hom_map.forward(z, method=hom_map_method)
        return self.objective_x(x)

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
        elif method == 'zeroorder':
            epsilon = 1e-6
            delta = torch.rand_like(z)
            delta = delta / torch.norm(delta, dim=1, keepdim=True)
            obj_plus = self.objective_z(z + epsilon * delta, hom_map)
            obj_minus = self.objective_z(z - epsilon * delta, hom_map)
            grad_z = (obj_plus - obj_minus) / (2 * epsilon) * delta
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
