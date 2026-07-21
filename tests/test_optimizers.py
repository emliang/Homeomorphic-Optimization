import torch
import pytest

from homopt.optim import AdamOptimizer, GDOptimizer, NormalizedGDOptimizer, run_algorithm


def test_gd_optimizer_step_smoke():
    grad = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    opt = GDOptimizer(beta1=0.5)
    out = opt.step(grad)
    assert out.shape == grad.shape
    assert torch.isfinite(out).all()


def test_adam_optimizer_step_shape_is_preserved():
    grad = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    opt = AdamOptimizer()
    out = opt.step(grad)
    assert out.shape == grad.shape
    assert torch.isfinite(out).all()


def test_normalized_gd_optimizer_step_has_unit_row_norm():
    grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32)
    opt = NormalizedGDOptimizer()
    out = opt.step(grad)
    assert torch.allclose(out[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(out[1], torch.zeros(2))


def test_run_algorithm_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        run_algorithm('UNKNOWN', problem=None, params={'common': {'seed': 0}})
