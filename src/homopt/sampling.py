"""Sampling utilities with optional low-discrepancy Hammersly sampling."""

from __future__ import annotations

import numpy as np

__all__ = [
    'sampling_sphere',
    'sampling_square',
    'sampling_sphere_surface',
    'sampling_square_surface',
    'sampling_body',
    'sampling_surface',
]


def _load_skopt_sampler():
    try:
        from skopt.sampler import Hammersly
        from skopt.space import Space
    except Exception:  # pragma: no cover - optional dependency
        return None, None
    return Space, Hammersly


def _unit_box_samples(n_samples, n_dim, low=0.0, high=1.0):
    Space, Hammersly = _load_skopt_sampler()
    if Space is not None and Hammersly is not None:
        dims = [(low, high) for _ in range(n_dim)]
        sampler = Hammersly()
        return np.asarray(sampler.generate(Space(dims).dimensions, n_samples), dtype=np.float32)
    return np.random.uniform(low, high, size=(n_samples, n_dim)).astype(np.float32)


def _normalize_rows(x, ord_value):
    norms = np.linalg.norm(x, ord=ord_value, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def sampling_sphere(n_samples, n_dim):
    x = _unit_box_samples(n_samples, n_dim, 0.0, 1.0)
    r = x[:, [0]] ** (1.0 / n_dim)
    for n in range(1, n_dim):
        if n == 1:
            x[:, n] = x[:, n] * 2 * np.pi
        else:
            x[:, n] = 1 - 2 * x[:, n]
    x_s = []
    for n in range(1, n_dim):
        if n == 1:
            x_s = np.column_stack([np.cos(x[:, n:n + 1]), np.sin(x[:, n:n + 1])])
        else:
            x_s = np.column_stack([np.sqrt(1 - x[:, n:n + 1] ** 2) * x_s, x[:, n:n + 1]])
    return np.multiply(x_s, r).astype(np.float32)


def sampling_square(n_samples, n_dim):
    return _unit_box_samples(n_samples, n_dim, -1.0, 1.0)


def sampling_sphere_surface(n_samples, n_dim):
    x = np.random.randn(n_samples, n_dim).astype(np.float32)
    return _normalize_rows(x, 2).astype(np.float32)


def sampling_square_surface(n_samples, n_dim):
    x = np.random.randn(n_samples, n_dim).astype(np.float32)
    return _normalize_rows(x, np.inf).astype(np.float32)


def sampling_body(n_samples, n_dim, shape, lu=(-1, 1)):
    if shape == 'sphere':
        x = sampling_sphere(n_samples, n_dim)
        return x + (lu[0] + lu[1]) / 2
    if shape == 'square':
        x = sampling_square(n_samples, n_dim)
        return (x + 1) * (lu[1] - lu[0]) / 2 + lu[0]
    raise ValueError(f'Unknown shape: {shape}')


def sampling_surface(n_samples, n_dim, shape, lu=(-1, 1)):
    if shape == 'sphere':
        x = sampling_sphere_surface(n_samples, n_dim)
        return x + (lu[0] + lu[1]) / 2
    if shape == 'square':
        x = sampling_square_surface(n_samples, n_dim)
        return (x + 1) * (lu[1] - lu[0]) / 2 + lu[0]
    raise ValueError(f'Unknown shape: {shape}')
