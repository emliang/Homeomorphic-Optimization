import numpy as np
import torch

from homopt.optim import AdamOptimizer, GDOptimizer, NormalizedGDOptimizer
from homopt.problems import create_maxcut_problem, create_test_problem


def test_gd_optimizer_momentum_trace_matches_expected_values():
    grad = torch.tensor([[2.0, -4.0]], dtype=torch.float32)
    opt = GDOptimizer(beta1=0.5)
    first = opt.step(grad)
    second = opt.step(grad)
    assert torch.allclose(first, torch.tensor([[1.0, -2.0]], dtype=torch.float32))
    assert torch.allclose(second, torch.tensor([[1.5, -3.0]], dtype=torch.float32))


def test_adam_optimizer_constant_gradient_normalizes_to_unit_scale():
    grad = torch.tensor([[3.0, -6.0]], dtype=torch.float32)
    opt = AdamOptimizer()
    first = opt.step(grad)
    second = opt.step(grad)
    expected = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    assert torch.allclose(first, expected, atol=1e-6)
    assert torch.allclose(second, expected, atol=1e-6)


def test_normalized_gd_optimizer_scales_by_l2_norm_per_sample():
    grad = torch.tensor([[3.0, -4.0]], dtype=torch.float32)
    opt = NormalizedGDOptimizer()
    assert torch.allclose(opt.step(grad), torch.tensor([[0.6, -0.8]], dtype=torch.float32))


def test_create_test_problem_is_seed_reproducible():
    params = {
        "seed": 11,
        "n_var": 4,
        "n_linear_cons": 1,
        "n_soc_cons": 1,
        "n_qua_cons": 1,
        "obj": "quad",
        "x_lower": -2,
        "x_upper": 2,
    }
    first = create_test_problem(params)
    second = create_test_problem(params)
    assert np.allclose(first["Q"], second["Q"])
    assert np.allclose(first["p"], second["p"])


def test_create_maxcut_problem_is_seed_reproducible():
    params = {
        "seed": 13,
        "n": 6,
        "alpha": 0.4,
        "use_weights": True,
    }
    first = create_maxcut_problem(params)
    second = create_maxcut_problem(params)
    assert first["edge"] == second["edge"]
    assert np.allclose(first["weights"], second["weights"])
