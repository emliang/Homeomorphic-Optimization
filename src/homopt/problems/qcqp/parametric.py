"""Parametric non-convex QCQP family for learning workflows."""

import numpy as np
import torch

from homopt.problems.base import ParametricProblemBase, ProblemInstance, TensorRuntimeMixin

from .deterministic import QCOpt
from .generators import create_test_QC_problem

torch.set_default_dtype(torch.float32)


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

    def scale(self, input_params, u):
        """Scale normalized coordinates from ``[-1, 1]`` to the decision bounds."""
        del input_params
        fixed_L = torch.as_tensor(self.fixed_L, dtype=u.dtype, device=u.device)
        fixed_U = torch.as_tensor(self.fixed_U, dtype=u.dtype, device=u.device)
        return (u + 1) / 2 * (fixed_U - fixed_L) + fixed_L

    def complete_partial(self, input_params, y):
        """Complete partial decision to full decision (identity for QCQP)."""
        del input_params
        return y

    def ineq_resid(self, input_params, y):
        """
        Compute inequality constraint residuals using variable Qq, pq, bq and fixed other parameters.

        OPTIMIZED VERSION: Uses batch computation instead of iteration for significant performance gains.

        Key optimizations:
        - Vectorized parameter extraction and reshaping
        - Einstein summation (einsum) for efficient batch matrix operations
        - Single-pass constraint evaluation for all samples
        - Eliminated for-loops over batch dimension

        Args:
            input_params: tensor of shape (batch_size, ncon) containing [Qq, pq, bq] parameters
            y: tensor of shape (batch_size, nvar) containing decision variables

        Returns:
            violations: tensor of shape (batch_size, total_constraints) containing constraint violations
        """
        batch_size = y.shape[0]
        input_size = input_params.shape[0]

        if batch_size != input_size:
            # Handle broadcasting if needed
            if input_size == 1:
                input_params = input_params.repeat(batch_size, 1)
            else:
                raise ValueError(f"Batch size mismatch: y has {batch_size}, input has {input_size}")

        # Convert fixed parameters to tensors on the same device
        device = y.device
        fixed_L = torch.as_tensor(self.fixed_L, dtype=y.dtype, device=device)
        fixed_U = torch.as_tensor(self.fixed_U, dtype=y.dtype, device=device)

        # Calculate sizes for unpacking input parameters
        # qq_size = self.n_qua * self.nvar * self.nvar
        # pq_size = self.n_qua * self.nvar
        # bq_size = self.n_qua


        # Extract all parameters in batch
        # input_params shape: (batch_size, n_qua_cons * (n_var*n_var + n_var + 1))
        # qq_flat_batch = input_params[:, :qq_size]  # (batch_size, n_qua * n_var * n_var)
        # pq_flat_batch = input_params[:, qq_size:qq_size + pq_size]  # (batch_size, n_qua * n_var)
        # bq_flat_batch = input_params[:, qq_size + pq_size:qq_size + pq_size + bq_size]  # (batch_size, n_qua)
        # Reshape to original dimensions
        # Qq_batch = qq_flat_batch.reshape(batch_size, self.n_qua, self.nvar, self.nvar)  # (batch_size, n_qua, n_var, n_var)
        # pq_batch = pq_flat_batch.reshape(batch_size, self.n_qua, self.nvar)  # (batch_size, n_qua, n_var)
        # bq_batch = bq_flat_batch.reshape(batch_size, self.n_qua)  # (batch_size, n_qua)

        qq_size = self.nvar * self.nvar
        pq_size = self.nvar
        bq_size = 1

        Qq_batch = input_params[:, :, :qq_size]
        pq_batch = input_params[:, :, qq_size:qq_size + pq_size]
        bq_batch = input_params[:, :, qq_size + pq_size:qq_size + pq_size + bq_size]
        Qq_batch = Qq_batch.view(batch_size, self.n_qua, self.nvar, self.nvar)
        pq_batch = pq_batch.view(batch_size, self.n_qua, self.nvar)
        bq_batch = bq_batch.view(batch_size, self.n_qua)

        # Compute quadratic constraint violations in batch: 0.5 * y^T Qq_j y + pq_j^T y <= bq_j
        # y shape: (batch_size, n_var)
        # Qq_batch shape: (batch_size, n_qua, n_var, n_var)

        # Expand y for batch matrix multiplication: (batch_size, 1, n_var) -> (batch_size, n_qua, n_var)
        y_expanded = y.unsqueeze(1).expand(-1, self.n_qua, -1)  # (batch_size, n_qua, n_var)

        # Quadratic terms: 0.5 * y^T Qq y
        # Using einsum for efficient batch computation: 'bji,bjik,bjk->bj'
        quad_vals = 0.5 * torch.einsum('bji,bjik,bjk->bj', y_expanded, Qq_batch, y_expanded)  # (batch_size, n_qua)

        # Linear terms: pq^T y
        # Using einsum: 'bjk,bk->bj'
        linear_vals = torch.einsum('bjk,bk->bj', pq_batch, y)  # (batch_size, n_qua)

        # Total quadratic constraint values
        total_quad_vals = quad_vals + linear_vals  # (batch_size, n_qua)
        quad_violations = total_quad_vals - bq_batch  # (batch_size, n_qua)

        # Box constraints using fixed bounds (batch computation)
        lbound_viol = fixed_L.unsqueeze(0) - y  # (batch_size, n_var)
        ubound_viol = y - fixed_U.unsqueeze(0)  # (batch_size, n_var)

        # Linear constraints (if any)
        if hasattr(self, 'fixed_A') and self.fixed_A is not None:
            fixed_A = torch.as_tensor(self.fixed_A, dtype=y.dtype, device=device)
            fixed_b = torch.as_tensor(self.fixed_b, dtype=y.dtype, device=device)
            lin_vals = torch.matmul(y, fixed_A.T)  # (batch_size, n_lin)
            lin_violations = lin_vals - fixed_b.unsqueeze(0)  # (batch_size, n_lin)
            all_viols = torch.cat([quad_violations, lin_violations, lbound_viol, ubound_viol], dim=1)
        else:
            all_viols = torch.cat([quad_violations, lbound_viol, ubound_viol], dim=1)

        # R-ball constraint
        r_ball_viol = (y**2).sum(-1, keepdim=True) - self.fixed_R**2  # (batch_size, 1)
        all_viols = torch.cat([all_viols, r_ball_viol], dim=1)

        return all_viols

    def objective(self, x):
        """Compute objective function"""
        fixed_Q = torch.as_tensor(self.fixed_Q, dtype=x.dtype, device=x.device)
        fixed_p = torch.as_tensor(self.fixed_p, dtype=x.dtype, device=x.device)
        return 0.5 * torch.sum(x @ fixed_Q * x, dim=1) + torch.sum(fixed_p * x, dim=1)

    def objective_xy(self, input_params, y, objective_batch=None):
        """Learning-route objective view on the same QCQP decision variable."""
        del input_params
        if objective_batch is None:
            return self.objective(y).view(-1, 1)
        if not torch.is_tensor(objective_batch):
            objective_batch = torch.as_tensor(objective_batch, dtype=y.dtype, device=y.device)
        else:
            objective_batch = objective_batch.to(dtype=y.dtype, device=y.device)
        if objective_batch.ndim == 1:
            objective_batch = objective_batch.view(1, -1)
        if objective_batch.shape[0] != y.shape[0]:
            if objective_batch.shape[0] == 1:
                objective_batch = objective_batch.expand(y.shape[0], -1)
            else:
                raise ValueError(
                    f"Objective batch size mismatch: y has {y.shape[0]}, objective has {objective_batch.shape[0]}"
                )
        q_size = self.nvar * self.nvar
        q_batch = objective_batch[:, :q_size].reshape(y.shape[0], self.nvar, self.nvar)
        p_batch = objective_batch[:, q_size:q_size + self.nvar]
        quad = 0.5 * torch.einsum("bi,bij,bj->b", y, q_batch, y)
        linear = torch.einsum("bi,bi->b", p_batch, y)
        return (quad + linear).view(-1, 1)

    def constraint_residual_xy(self, input_params, y, clip=True):
        """Learning-route constraint residual view using the parametric QC instance input."""
        residual = self.ineq_resid(input_params, y)
        return torch.clamp(residual, min=0.0) if clip else residual

    def violations(self, input_params, y):
        """Compute violations"""
        return self.ineq_resid(input_params, y).max(dim=1)[0].view(-1, 1)

    def check_feasibility(self, input_params, y):
        """Check feasibility of points"""
        violations = self.ineq_resid(input_params, y)
        return torch.clamp(violations, min=0)

__all__ = ["NonConvexQCProblem"]
