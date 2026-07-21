import torch

from homopt.mappings.gauge import GaugeMap


class DummyBoxSet:
    def __init__(self):
        self.nvar = 2
        self.G = None
        self.h = None
        self.C = None
        self.d = None
        self.Qq = None
        self.pq = None
        self.bq = None
        self.A = None
        self.b = None
        self.L = torch.tensor([-1.0, -1.0], dtype=torch.float32)
        self.U = torch.tensor([1.0, 1.0], dtype=torch.float32)

    def constraint_x(self, x):
        upper = x - self.U.view(1, -1)
        lower = self.L.view(1, -1) - x
        return torch.cat([upper, lower], dim=1)


class DummyMixedSet:
    def __init__(self):
        self.nvar = 2
        self.A = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        self.b = torch.tensor([[1.2, 1.2]], dtype=torch.float32)
        self.L = torch.tensor([-1.0, -1.0], dtype=torch.float32)
        self.U = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.G = None
        self.h = None
        self.C = None
        self.d = None
        self.Qq = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        self.pq = torch.zeros((1, 2), dtype=torch.float32)
        self.bq = torch.tensor([2.0], dtype=torch.float32)

    def constraint_x(self, x):
        del x
        return torch.zeros((1, 1), dtype=torch.float32)


class DummyGeneralBallSet:
    def __init__(self):
        self.nvar = 2
        self.G = None
        self.h = None
        self.C = None
        self.d = None
        self.Qq = None
        self.pq = None
        self.bq = None
        self.A = None
        self.b = None
        self.L = None
        self.U = None
        self.radius = 2.0

    def constraint_x(self, x):
        return self.general_convex_constraint_x(x)

    def general_convex_constraint_x(self, x):
        return (x.square().sum(dim=1, keepdim=True) - self.radius**2)

    def general_convex_constraint_gradient_x(self, x):
        return (2.0 * x).unsqueeze(1)


def test_gaugemap_box_roundtrip():
    box = DummyBoxSet()
    mapping = GaugeMap(box, p_norm=2)
    z = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    x = mapping.forward(z)
    z_recovered = mapping.inverse(x)
    assert torch.allclose(z, z_recovered, atol=1e-5)


def test_gaugemap_explicit_box_backward_matches_autograd_multi_batch():
    box = DummyBoxSet()
    mapping = GaugeMap(box, p_norm=2)
    z = torch.tensor([[0.25, -0.5], [0.4, 0.1], [-0.2, -0.6], [0.75, 0.2]], dtype=torch.float32)

    z_auto = z.clone().requires_grad_(True)
    loss_auto = (mapping.forward(z_auto, method="autograd") ** 2).sum()
    grad_auto = torch.autograd.grad(loss_auto, z_auto)[0]

    z_exp = z.clone().requires_grad_(True)
    loss_exp = (mapping.forward(z_exp, method="explicit") ** 2).sum()
    grad_exp = torch.autograd.grad(loss_exp, z_exp)[0]

    assert torch.allclose(grad_exp, grad_auto, atol=1e-4, rtol=1e-4)


def test_gaugemap_explicit_mixed_constraints_matches_autograd():
    mixed = DummyMixedSet()
    mapping = GaugeMap(mixed, p_norm=2)
    z = torch.tensor([[0.3, 0.15], [0.45, -0.2]], dtype=torch.float32)

    z_auto = z.clone().requires_grad_(True)
    loss_auto = (mapping.forward(z_auto, method="autograd") ** 2).sum()
    grad_auto = torch.autograd.grad(loss_auto, z_auto)[0]

    z_exp = z.clone().requires_grad_(True)
    loss_exp = (mapping.forward(z_exp, method="explicit") ** 2).sum()
    grad_exp = torch.autograd.grad(loss_exp, z_exp)[0]

    assert torch.allclose(grad_exp, grad_auto, atol=1e-4, rtol=1e-4)


def test_gaugemap_general_convex_bisection_forward_and_backward():
    problem = DummyGeneralBallSet()
    mapping = GaugeMap(problem, p_norm=2, general_bisection_iterations=80)
    z = torch.tensor([[0.3, 0.4], [-0.2, 0.1]], dtype=torch.float32, requires_grad=True)

    x_default = mapping.forward(z.detach())
    assert torch.allclose(x_default, problem.radius * z.detach(), atol=1e-5, rtol=1e-5)

    x = mapping.forward(z, method="explicit")
    assert torch.allclose(x, problem.radius * z, atol=1e-5, rtol=1e-5)

    loss = x.square().sum()
    grad = torch.autograd.grad(loss, z)[0]
    expected_grad = 2.0 * problem.radius**2 * z.detach()
    assert torch.allclose(grad, expected_grad, atol=1e-4, rtol=1e-4)
