"""Deterministic convex instance generators."""

import numpy as np


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

__all__ = ["create_test_problem"]
