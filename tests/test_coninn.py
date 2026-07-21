import torch

from homopt.models import (
    CubeGaugeCvxINN,
    CvxINN,
    INN,
    SphereReparam,
    SphereReparamGaugeCvxINN,
    TanhGaugeCvxINN,
    combine_local_stretch_objective,
    initialize_inn_as_identity,
    local_log_stretch_loss,
    train_cvxinn_min_distortion,
)
from homopt.models.flows.coupling import CombinedActNormLU
from homopt.mappings.gauge import GaugeMap
from homopt.viz import evaluate_coninn_exactness


class IdentityInvertible(torch.nn.Module):
    def forward(self, z, c=None):
        del c
        return z

    def inverse(self, x, c=None):
        del c
        return x


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


class AffineInvertible(torch.nn.Module):
    def __init__(self):
        super().__init__()
        matrix = torch.tensor([[1.3, 0.2], [-0.15, 0.9]], dtype=torch.float32)
        bias = torch.tensor([0.25, -0.1], dtype=torch.float32)
        self.register_buffer("matrix", matrix)
        self.register_buffer("matrix_inv", torch.inverse(matrix))
        self.register_buffer("bias", bias)

    def forward(self, z, c=None):
        del c
        return torch.matmul(z, self.matrix.T) + self.bias.view(1, -1)

    def inverse(self, x, c=None):
        del c
        return torch.matmul(x - self.bias.view(1, -1), self.matrix_inv.T)


def test_cvxinn_export_is_available():
    assert CvxINN is not None
    assert CubeGaugeCvxINN is not None
    assert SphereReparam is not None
    assert SphereReparamGaugeCvxINN is not None
    assert TanhGaugeCvxINN is not None


def test_cvxinn_roundtrip_over_box_constraint():
    model = CvxINN(IdentityInvertible(), GaugeMap(DummyBoxSet(), p_norm=2))
    z = torch.tensor([[0.2, -0.3], [0.5, 0.1]], dtype=torch.float32)

    x = model(z)
    z_back = model.inverse(x)

    cons = DummyBoxSet().constraint_x(x)
    assert x.shape == z.shape
    assert torch.all(cons <= 1e-5)
    assert torch.allclose(z_back, z, atol=1e-5)


def test_tanh_gauge_cvxinn_roundtrip_over_box_constraint():
    model = TanhGaugeCvxINN.from_convex_set(IdentityInvertible(), DummyBoxSet())
    z = torch.tensor([[0.2, -0.3], [0.5, 0.1]], dtype=torch.float32)

    x = model(z)
    z_back = model.inverse(x)

    cons = DummyBoxSet().constraint_x(x)
    assert x.shape == z.shape
    assert torch.all(cons <= 1e-5)
    assert torch.allclose(z_back, z, atol=1e-5)


def test_cube_gauge_cvxinn_roundtrip_over_box_constraint():
    model = CubeGaugeCvxINN.from_convex_set(IdentityInvertible(), DummyBoxSet())
    z = torch.tensor([[0.2, -0.3], [0.5, 0.1]], dtype=torch.float32)

    x = model(z)
    z_back = model.inverse(x)

    cons = DummyBoxSet().constraint_x(x)
    assert x.shape == z.shape
    assert torch.all(cons <= 1e-5)
    assert torch.allclose(z_back, z, atol=1e-5)


def test_sphere_reparam_roundtrip_preserves_radius():
    model = SphereReparam(2)
    z = torch.tensor([[0.2, -0.3], [0.5, 0.1]], dtype=torch.float32)

    out = model(z)
    inv = model.inverse(out)

    assert torch.allclose(torch.norm(out, dim=1), torch.norm(z, dim=1), atol=1e-6)
    assert torch.allclose(inv, z, atol=1e-5)


def test_sphere_reparam_gauge_cvxinn_roundtrip_over_box_constraint():
    model = SphereReparamGaugeCvxINN.from_convex_set(SphereReparam(2), DummyBoxSet())
    z = torch.tensor([[0.2, -0.3], [0.5, 0.1]], dtype=torch.float32)

    x = model(z)
    z_back = model.inverse(x)

    cons = DummyBoxSet().constraint_x(x)
    assert x.shape == z.shape
    assert torch.all(cons <= 1e-5)
    assert torch.allclose(z_back, z, atol=1e-5)


def test_cvxinn_exactness_evaluator_reports_feasible_roundtrip_and_ball_image():
    model = CvxINN(AffineInvertible(), GaugeMap(DummyBoxSet(), p_norm=2))
    stats = evaluate_coninn_exactness(model, DummyBoxSet(), n_interior=512, n_boundary=128, seed=2030)

    assert stats["max_constraint_violation"] <= 1e-5
    assert stats["max_boundary_violation"] <= 1e-5
    assert stats["max_roundtrip_error"] <= 1e-5
    assert stats["max_inner_ball_radius"] < 1.0


def test_local_log_stretch_loss_is_near_zero_for_identity_map():
    z = torch.tensor([[0.2, -0.1], [-0.25, 0.35], [0.1, 0.15]], dtype=torch.float32)
    loss, stats = local_log_stretch_loss(
        lambda x: x,
        z,
        num_directions=6,
        fd_eps=1e-3,
        domain_shape="ball",
        aggregate="max",
    )

    assert abs(loss.item()) < 1e-4
    assert stats["stretch"].min().item() > 0


def test_soft_spread_objective_is_finite_and_nonnegative():
    z = torch.tensor([[0.2, -0.1], [-0.25, 0.35], [0.1, 0.15]], dtype=torch.float32)
    loss, stats = local_log_stretch_loss(
        lambda x: x * torch.tensor([1.0, 1.2], dtype=x.dtype, device=x.device),
        z,
        num_directions=6,
        fd_eps=1e-3,
        domain_shape="ball",
        aggregate="spread",
    )
    soft_loss = combine_local_stretch_objective(
        loss,
        stats,
        objective="variance_plus_soft_spread",
        soft_temperature=0.1,
    )
    assert torch.isfinite(soft_loss)
    assert soft_loss.item() >= 0.0


def test_spread_weight_changes_combined_objective():
    z = torch.tensor([[0.2, -0.1], [-0.25, 0.35], [0.1, 0.15]], dtype=torch.float32)
    loss, stats = local_log_stretch_loss(
        lambda x: x * torch.tensor([1.0, 1.2], dtype=x.dtype, device=x.device),
        z,
        num_directions=6,
        fd_eps=1e-3,
        domain_shape="ball",
        aggregate="spread",
    )
    low = combine_local_stretch_objective(loss, stats, objective="variance_plus_spread", spread_weight=0.1)
    high = combine_local_stretch_objective(loss, stats, objective="variance_plus_spread", spread_weight=0.5)
    assert torch.isfinite(low) and torch.isfinite(high)
    assert high.item() > low.item()


def test_train_cvxinn_min_distortion_smoke_runs():
    model, record = train_cvxinn_min_distortion(
        DummyBoxSet(),
        total_iteration=3,
        batch_size=32,
        h_dim=8,
        num_layer=1,
        stretch_num_directions=4,
        identity_init_inner=True,
        stretch_objective="variance_plus_spread",
        record_every=1,
        seed=2030,
    )

    assert isinstance(model, CvxINN)
    assert len(record["loss_list"]) == 3
    assert all(torch.isfinite(torch.tensor(record["loss_list"])).tolist())
    assert record["identity_init_inner"] is True
    assert record["stretch_objective"] == "variance_plus_spread"


def test_train_cvxinn_min_distortion_cube_wrapper_uses_cube_latent_domain():
    model, record = train_cvxinn_min_distortion(
        DummyBoxSet(),
        total_iteration=3,
        batch_size=32,
        h_dim=8,
        num_layer=1,
        stretch_num_directions=4,
        identity_init_inner=True,
        stretch_objective="variance_plus_spread",
        wrapper_type="cube_reparam_gauge",
        record_every=1,
        seed=2032,
    )

    assert isinstance(model, CubeGaugeCvxINN)
    assert record["latent_domain_shape"] == "cube"


def test_train_cvxinn_min_distortion_sphere_reparam_wrapper_smoke_runs():
    model, record = train_cvxinn_min_distortion(
        DummyBoxSet(),
        total_iteration=3,
        batch_size=32,
        stretch_num_directions=4,
        identity_init_inner=True,
        stretch_objective="variance_plus_spread",
        wrapper_type="sphere_reparam_gauge",
        record_every=1,
        seed=2033,
    )

    assert isinstance(model, SphereReparamGaugeCvxINN)
    assert record["latent_domain_shape"] == "ball"


def test_train_cvxinn_min_distortion_supports_early_stopping():
    model, record = train_cvxinn_min_distortion(
        DummyBoxSet(),
        total_iteration=20,
        batch_size=16,
        h_dim=8,
        num_layer=1,
        stretch_num_directions=4,
        identity_init_inner=True,
        stretch_objective="variance_plus_spread",
        record_every=1,
        eval_every=1,
        eval_batch_size=64,
        early_stop_patience=2,
        early_stop_min_delta=1e-8,
        early_stop_min_iterations=1,
        lr=0.0,
        seed=2031,
    )

    assert isinstance(model, CvxINN)
    assert record["stop_reason"] == "early_stopping"
    assert record["final_iteration"] < record["total_iteration"]
    assert record["best_iteration"] <= record["final_iteration"]
    assert len(record["eval_history"]) >= 2


def test_combined_actnorm_lu_roundtrip_is_exact():
    layer = CombinedActNormLU(2).eval()
    z = torch.tensor([[0.2, -0.1], [0.5, 0.1], [-0.3, 0.25]], dtype=torch.float32)

    out, _ = layer(z)
    inv, _ = layer(out, mode="inverse")

    assert torch.allclose(inv, z, atol=1e-5)


def test_initialize_inn_as_identity_makes_forward_inverse_consistent():
    model = INN(nin=2, nhid=8, cin=1, nl=2, inv='coupling', Con_type='MLP').eval()
    initialize_inn_as_identity(model)
    z = torch.tensor([[0.2, -0.1], [0.5, 0.1], [-0.3, 0.25]], dtype=torch.float32)
    c = torch.zeros((z.shape[0], 1), dtype=torch.float32)

    out = model(z, c)
    inv = model.inverse(out, c)

    assert torch.allclose(out, z, atol=1e-5)
    assert torch.allclose(inv, z, atol=1e-5)


def test_initialize_made_inn_as_identity_makes_forward_inverse_consistent():
    model = INN(nin=2, nhid=8, cin=1, nl=2, inv='made', Con_type='MLP').eval()
    initialize_inn_as_identity(model)
    z = torch.tensor([[0.2, -0.1], [0.5, 0.1], [-0.3, 0.25]], dtype=torch.float32)
    c = torch.zeros((z.shape[0], 1), dtype=torch.float32)

    out = model(z, c)
    inv = model.inverse(out, c)

    assert torch.allclose(out, z, atol=1e-5)
    assert torch.allclose(inv, z, atol=1e-5)


def test_initialize_residual_inn_as_identity_makes_forward_inverse_consistent():
    model = INN(nin=2, nhid=8, cin=1, nl=2, inv='residual', Con_type='MLP').eval()
    initialize_inn_as_identity(model)
    z = torch.tensor([[0.2, -0.1], [0.5, 0.1], [-0.3, 0.25]], dtype=torch.float32)
    c = torch.zeros((z.shape[0], 1), dtype=torch.float32)

    out = model(z, c)
    inv = model.inverse(out, c)

    assert torch.allclose(out, z, atol=1e-5)
    assert torch.allclose(inv, z, atol=1e-4)


def test_residual_inn_can_skip_training_stats_in_training_mode():
    model = INN(nin=2, nhid=8, cin=1, nl=1, inv='residual', Con_type='MLP')
    initialize_inn_as_identity(model)
    model.track_training_stats = False
    model.train()
    z = torch.tensor([[0.2, -0.1], [0.5, 0.1]], dtype=torch.float32)
    c = torch.zeros((z.shape[0], 1), dtype=torch.float32)

    out = model(z, c)

    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, z, atol=1e-5)
