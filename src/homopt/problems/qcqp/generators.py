"""QCQP deterministic instance generators."""

import numpy as np

from .spectral import _spectral_quadratic_matrix


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

__all__ = ["create_test_QC_problem"]
