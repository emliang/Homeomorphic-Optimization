import numpy as np

from homopt.models.sampling import (
    sampling_body,
    sampling_sphere,
    sampling_sphere_surface,
    sampling_square,
    sampling_square_surface,
    sampling_surface,
)


def test_sampling_square_shape_and_bounds():
    x = sampling_square(32, 3)
    assert x.shape == (32, 3)
    assert np.all(x <= 1.0 + 1e-6)
    assert np.all(x >= -1.0 - 1e-6)


def test_sampling_sphere_radius_within_unit_ball():
    x = sampling_sphere(64, 3)
    radii = np.linalg.norm(x, axis=1)
    assert x.shape == (64, 3)
    assert np.all(radii <= 1.0 + 1e-5)


def test_sampling_sphere_surface_has_unit_l2_norm():
    x = sampling_sphere_surface(64, 4)
    norms = np.linalg.norm(x, axis=1)
    assert x.shape == (64, 4)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_sampling_square_surface_has_unit_linf_norm():
    x = sampling_square_surface(64, 4)
    norms = np.linalg.norm(x, ord=np.inf, axis=1)
    assert x.shape == (64, 4)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_sampling_body_and_surface_dispatch():
    xb = sampling_body(16, 2, 'square', lu=(-2, 2))
    xs = sampling_surface(16, 2, 'square', lu=(-2, 2))
    assert xb.shape == (16, 2)
    assert xs.shape == (16, 2)
    assert np.all(xb <= 2.0 + 1e-6)
    assert np.all(xb >= -2.0 - 1e-6)
    assert np.all(np.linalg.norm(xs / 2.0, ord=np.inf, axis=1) <= 1.0 + 1e-5)


def test_sampling_sphere_surface_small_batch_smoke():
    x = sampling_sphere_surface(8, 3)
    assert x.shape == (8, 3)
