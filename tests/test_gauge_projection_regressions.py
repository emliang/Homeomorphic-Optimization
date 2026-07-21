"""Regression tests for implicit general-convex gauge gradients and latent balls."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from homopt.mappings.gauge import GaugeMap
from homopt.optim.alm.hom_alm import HomALMOptimizer
from homopt.optim.core.tensors import _project_to_ball
from homopt.optim.first_order.hom_pgd import HomPGDOptimizer


class _GeneralEllipseSet:
    def __init__(self):
        self.nvar = 2
        self.A = self.b = None
        self.L = self.U = None
        self.G = self.h = self.C = self.d = None
        self.Qq = self.pq = self.bq = None

    def general_convex_constraint_x(self, x):
        return x[:, :1].square() / 9.0 + x[:, 1:].square() / 4.0 - 1.0

    def general_convex_constraint_gradient_x(self, x):
        gradient = torch.stack((2.0 * x[:, 0] / 9.0, 2.0 * x[:, 1] / 4.0), dim=-1)
        return gradient.unsqueeze(1)


def _central_difference(loss_fn, point, *, step=1e-6):
    gradient = torch.empty_like(point)
    for column in range(point.shape[1]):
        offset = torch.zeros_like(point)
        offset[:, column] = step
        gradient[:, column] = (loss_fn(point + offset) - loss_fn(point - offset)) / (2.0 * step)
    return gradient


def test_general_convex_default_gradient_uses_implicit_bisection_vjp():
    mapping = GaugeMap(_GeneralEllipseSet(), p_norm=2, general_bisection_iterations=80)
    z_default = torch.tensor([[0.8, 0.6]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[0.7, 1.3]], dtype=torch.float64)

    loss_default = (weights * mapping.forward(z_default).square()).sum()
    gradient_default = torch.autograd.grad(loss_default, z_default)[0]

    z_explicit = z_default.detach().clone().requires_grad_(True)
    loss_explicit = (weights * mapping.forward(z_explicit, method="explicit").square()).sum()
    gradient_explicit = torch.autograd.grad(loss_explicit, z_explicit)[0]

    def loss_at(point):
        return (weights * mapping.forward(point).square()).sum()

    gradient_finite_difference = _central_difference(loss_at, z_default.detach())
    assert mapping.last_forward_mode == "explicit"
    assert torch.allclose(gradient_default, gradient_explicit, atol=1e-8, rtol=1e-7)
    assert torch.allclose(gradient_default, gradient_finite_difference, atol=2e-5, rtol=2e-5)


def test_general_convex_default_gradient_stays_implicit_when_returning_state():
    mapping = GaugeMap(_GeneralEllipseSet(), p_norm=2, general_bisection_iterations=80)
    z = torch.tensor([[0.8, 0.6]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[0.7, 1.3]], dtype=torch.float64)

    x, state = mapping.forward(z, return_state=True)
    gradient = torch.autograd.grad((weights * x.square()).sum(), z)[0]

    def loss_at(point):
        return (weights * mapping.forward(point).square()).sum()

    assert state is not None
    assert torch.allclose(gradient, _central_difference(loss_at, z.detach()), atol=2e-5, rtol=2e-5)


def test_general_convex_rejects_unsupported_non_euclidean_gradient_path():
    mapping = GaugeMap(_GeneralEllipseSet(), p_norm=1.5)
    with pytest.raises(NotImplementedError, match="p_norm=2"):
        mapping.forward(torch.tensor([[0.8, 0.6]], dtype=torch.float64))


@pytest.mark.parametrize("p_norm", [1.0, 1.5, 2.0, 3.0, np.inf])
def test_homeomorphic_optimizers_enforce_selected_latent_p_ball(p_norm):
    z = torch.tensor([[3.0, -4.0], [0.25, -0.5]], dtype=torch.float64)
    shell = SimpleNamespace(hom_map=SimpleNamespace(p_norm=p_norm))

    expected = _project_to_ball(z, p_norm)
    projected_pgd = HomPGDOptimizer.project_to_ball(shell, z)
    projected_alm = HomALMOptimizer.project_to_ball(shell, z)

    assert torch.allclose(projected_pgd, expected)
    assert torch.allclose(projected_alm, expected)
    norms = torch.norm(expected, dim=-1, p=p_norm)
    assert torch.all(norms <= 1.0 + 1e-12)
    assert torch.allclose(expected[1], z[1])


@pytest.mark.parametrize("p_norm", [0.0, 0.5, -1.0, -np.inf, np.nan, "2"])
def test_latent_ball_projection_rejects_invalid_p_norm(p_norm):
    with pytest.raises(ValueError, match="p_norm"):
        _project_to_ball(torch.ones(1, 2), p_norm)
