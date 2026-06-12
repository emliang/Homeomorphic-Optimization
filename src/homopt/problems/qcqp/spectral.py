"""Spectral helpers for QCQP instance generation."""

import numpy as np


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

__all__ = [
    "_low_rank_quadratic_batch",
    "_random_indefinite_eigs",
    "_random_orthonormal",
    "_spectral_quadratic_matrix",
]
