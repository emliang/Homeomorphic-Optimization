import numpy as np
import pytest
import torch

from homopt.mappings.gauge import GaugeMap, soft_maximum
from homopt.mappings.gauge.roots import _max_real_quadratic_root
from homopt.mappings.gauge.state import explicit_forward_state
from homopt.mappings.star import StarMap
from homopt.problems import ConvexOptEq, create_test_problem


class DummyStarSet:
    alpha = 0.2
    num_star = 4


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


class DummySOCSet:
    def __init__(self):
        self.nvar = 2
        self.A = None
        self.b = None
        self.L = None
        self.U = None
        self.Qq = None
        self.pq = None
        self.bq = None
        self.G = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        self.h = torch.zeros((1, 2), dtype=torch.float32)
        self.C = torch.zeros((1, 2), dtype=torch.float32)
        self.d = torch.tensor([2.0], dtype=torch.float32)

    def constraint_x(self, x):
        del x
        return torch.zeros((1, 1), dtype=torch.float32)


class DummyQuadSet:
    def __init__(self):
        self.nvar = 2
        self.A = None
        self.b = None
        self.L = None
        self.U = None
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


def test_starmap_forward_inverse_roundtrip():
    star = StarMap(DummyStarSet())
    z = torch.tensor([[0.2, 0.1]], dtype=torch.float32)
    x = star.forward(z)
    z_recovered = star.inverse(x)
    assert torch.allclose(z, z_recovered, atol=1e-5)


def test_gaugemap_box_roundtrip():
    box = DummyBoxSet()
    mapping = GaugeMap(box, p_norm=2)
    z = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    x = mapping.forward(z)
    z_recovered = mapping.inverse(x)
    assert torch.allclose(z, z_recovered, atol=1e-5)


def test_soft_maximum_bounds():
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    soft = soft_maximum(x, dim=-1)
    hard = torch.max(x, dim=-1)[0]
    assert soft.shape == hard.shape
    assert torch.all(soft >= hard - 1e-4)


def test_max_real_quadratic_root_handles_near_zero_constant_term():
    A = torch.tensor([[-2.682209e-7]], dtype=torch.float64, requires_grad=True)
    B = torch.tensor([[0.3873385190963745]], dtype=torch.float64)
    C = torch.tensor([[-0.5703201293945312]], dtype=torch.float64)

    root = _max_real_quadratic_root(A, B, C, eps=1e-12)
    grad = torch.autograd.grad(root.sum(), A)[0]

    assert root.item() == pytest.approx(0.679156, rel=1e-5)
    assert torch.isfinite(grad).all()
    assert abs(float(grad.item())) < 10.0


def test_gaugemap_explicit_linear_box_backward_matches_autograd():
    box = DummyBoxSet()
    mapping = GaugeMap(box, p_norm=2)

    z_auto = torch.tensor([[0.25, -0.5]], dtype=torch.float32, requires_grad=True)
    x_auto = mapping.forward(z_auto, method="autograd")
    loss_auto = (x_auto ** 2).sum()
    grad_auto = torch.autograd.grad(loss_auto, z_auto)[0]

    z_exp = torch.tensor([[0.25, -0.5]], dtype=torch.float32, requires_grad=True)
    x_exp = mapping.forward(z_exp, method="explicit")
    loss_exp = (x_exp ** 2).sum()
    grad_exp = torch.autograd.grad(loss_exp, z_exp)[0]

    assert torch.allclose(grad_exp, grad_auto, atol=1e-4, rtol=1e-4)


def test_gaugemap_explicit_soc_backward_matches_autograd():
    soc = DummySOCSet()
    mapping = GaugeMap(soc, p_norm=2)

    z_auto = torch.tensor([[0.3, 0.2]], dtype=torch.float32, requires_grad=True)
    x_auto = mapping.forward(z_auto, method="autograd")
    loss_auto = (x_auto ** 2).sum()
    grad_auto = torch.autograd.grad(loss_auto, z_auto)[0]

    z_exp = torch.tensor([[0.3, 0.2]], dtype=torch.float32, requires_grad=True)
    x_exp = mapping.forward(z_exp, method="explicit")
    loss_exp = (x_exp ** 2).sum()
    grad_exp = torch.autograd.grad(loss_exp, z_exp)[0]

    assert torch.allclose(grad_exp, grad_auto, atol=1e-4, rtol=1e-4)


def test_gaugemap_explicit_quadratic_backward_matches_autograd():
    quad = DummyQuadSet()
    mapping = GaugeMap(quad, p_norm=2)

    z_auto = torch.tensor([[0.4, -0.1]], dtype=torch.float32, requires_grad=True)
    x_auto = mapping.forward(z_auto, method="autograd")
    loss_auto = (x_auto ** 2).sum()
    grad_auto = torch.autograd.grad(loss_auto, z_auto)[0]

    z_exp = torch.tensor([[0.4, -0.1]], dtype=torch.float32, requires_grad=True)
    x_exp = mapping.forward(z_exp, method="explicit")
    loss_exp = (x_exp ** 2).sum()
    grad_exp = torch.autograd.grad(loss_exp, z_exp)[0]

    assert torch.allclose(grad_exp, grad_auto, atol=1e-4, rtol=1e-4)


def test_gaugemap_vjp_explicit_soc_matches_autograd():
    soc = DummySOCSet()
    mapping = GaugeMap(soc, p_norm=2)
    z = torch.tensor([[0.3, 0.2]], dtype=torch.float32)
    grad_x = torch.tensor([[1.0, -0.5]], dtype=torch.float32)

    grad_auto = mapping.vjp(z, grad_x, method="autograd")
    grad_exp = mapping.vjp(z, grad_x, method="explicit")

    assert torch.allclose(grad_exp, grad_auto, atol=1e-5, rtol=1e-5)


def test_gaugemap_vjp_explicit_quadratic_matches_autograd():
    quad = DummyQuadSet()
    mapping = GaugeMap(quad, p_norm=2)
    z = torch.tensor([[0.4, -0.1]], dtype=torch.float32)
    grad_x = torch.tensor([[0.5, 1.25]], dtype=torch.float32)

    grad_auto = mapping.vjp(z, grad_x, method="autograd")
    grad_exp = mapping.vjp(z, grad_x, method="explicit")

    assert torch.allclose(grad_exp, grad_auto, atol=1e-5, rtol=1e-5)


def test_gaugemap_explicit_backward_matches_autograd_multi_batch():
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


def test_gaugemap_explicit_requires_supported_geometry():
    box = DummyBoxSet()
    mapping = GaugeMap(box, p_norm=1.5)
    z = torch.tensor([[0.2, -0.35]], dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="explicit forward"):
        mapping.forward(z, method="explicit")


def test_gaugemap_explicit_active_tie_smooth_matches_autograd():
    box = DummyBoxSet()
    mapping = GaugeMap(box, p_norm=2, smooth=True)
    z = torch.tensor([[0.2, 0.35]], dtype=torch.float32)

    z_auto = z.clone().requires_grad_(True)
    loss_auto = (mapping.forward(z_auto, method="autograd") ** 2).sum()
    grad_auto = torch.autograd.grad(loss_auto, z_auto)[0]

    z_exp = z.clone().requires_grad_(True)
    loss_exp = (mapping.forward(z_exp, method="explicit") ** 2).sum()
    grad_exp = torch.autograd.grad(loss_exp, z_exp)[0]

    assert mapping.last_forward_mode == "explicit"
    assert torch.allclose(grad_exp, grad_auto, atol=1e-5, rtol=1e-5)


def test_gaugemap_explicit_active_tie_smooth_mixed_constraints_matches_autograd():
    mixed = DummyMixedSet()
    mapping = GaugeMap(mixed, p_norm=2, smooth=True, smooth_tie_tol=1e-7, smooth_temperature=1e-2)
    z = torch.tensor([[0.3, 0.15], [0.45, -0.2]], dtype=torch.float32)

    z_auto = z.clone().requires_grad_(True)
    loss_auto = (mapping.forward(z_auto, method="autograd") ** 2).sum()
    grad_auto = torch.autograd.grad(loss_auto, z_auto)[0]

    z_exp = z.clone().requires_grad_(True)
    loss_exp = (mapping.forward(z_exp, method="explicit") ** 2).sum()
    grad_exp = torch.autograd.grad(loss_exp, z_exp)[0]

    assert mapping.last_forward_mode == "explicit"
    assert torch.allclose(grad_exp, grad_auto, atol=1e-5, rtol=1e-5)


def test_gaugemap_smooth_excludes_non_tied_candidates_from_logsumexp():
    box = DummyBoxSet()
    z = torch.tensor([[0.1, 0.9]], dtype=torch.float32)
    hard_mapping = GaugeMap(box, p_norm=2, smooth=False)
    smooth_mapping = GaugeMap(box, p_norm=2, smooth=True, smooth_tie_tol=1e-8, smooth_temperature=0.5)

    _, hard_state = hard_mapping.forward(z, method="explicit", return_state=True)
    _, smooth_state = smooth_mapping.forward(z, method="explicit", return_state=True)

    assert torch.allclose(smooth_state["beta"], hard_state["beta"], atol=1e-7, rtol=1e-7)


def test_gaugemap_smooth_applies_logsumexp_to_tied_active_candidates():
    box = DummyBoxSet()
    temperature = 1e-2
    z = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    hard_mapping = GaugeMap(box, p_norm=2, smooth=False)
    smooth_mapping = GaugeMap(box, p_norm=2, smooth=True, smooth_tie_tol=1e-8, smooth_temperature=temperature)

    _, hard_state = hard_mapping.forward(z, method="explicit", return_state=True)
    _, smooth_state = smooth_mapping.forward(z, method="explicit", return_state=True)

    expected_beta = hard_state["beta"] + temperature * torch.log(torch.tensor([[2.0]], dtype=torch.float32))
    assert torch.allclose(smooth_state["beta"], expected_beta, atol=1e-7, rtol=1e-7)


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


def test_gaugemap_vjp_explicit_matches_autograd():
    mixed = DummyMixedSet()
    mapping = GaugeMap(mixed, p_norm=2)
    z = torch.tensor([[0.3, 0.15], [0.45, -0.2]], dtype=torch.float32)
    grad_x = torch.tensor([[1.0, -0.5], [0.25, 0.75]], dtype=torch.float32)

    grad_auto = mapping.vjp(z, grad_x, method="autograd")
    grad_exp = mapping.vjp(z, grad_x, method="explicit")

    assert torch.allclose(grad_exp, grad_auto, atol=1e-5, rtol=1e-5)


def test_gaugemap_autograd_vjp_works_inside_no_grad():
    mixed = DummyMixedSet()
    mapping = GaugeMap(mixed, p_norm=2)
    z = torch.tensor([[0.3, 0.15], [0.45, -0.2]], dtype=torch.float32)
    grad_x = torch.tensor([[1.0, -0.5], [0.25, 0.75]], dtype=torch.float32)

    expected = mapping.vjp(z, grad_x, method="autograd")
    with torch.no_grad():
        actual = mapping.vjp(z, grad_x, method="autograd")

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_gaugemap_records_forward_mode_stats():
    mixed = DummyMixedSet()
    mapping = GaugeMap(mixed, p_norm=2)
    mapping.reset_forward_stats()
    z = torch.tensor([[0.3, 0.15]], dtype=torch.float32)

    mapping.forward(z, method="explicit")
    assert mapping.last_forward_mode == "explicit"
    assert mapping.forward_mode_counts["explicit"] == 1

    mapping.forward(z, method="autograd")
    assert mapping.last_forward_mode == "autograd"
    assert mapping.forward_mode_counts["autograd"] == 1


def test_gaugemap_explicit_vjp_requires_supported_geometry():
    mapping = GaugeMap(DummyBoxSet(), p_norm=1.5)
    z = torch.tensor([[0.2, -0.35]], dtype=torch.float32)
    grad_x = torch.ones_like(z)

    with pytest.raises(NotImplementedError, match="explicit forward"):
        mapping.vjp(z, grad_x, method="explicit")


def test_gaugemap_rejects_unknown_method():
    mapping = GaugeMap(DummyBoxSet(), p_norm=2)
    z = torch.tensor([[0.2, -0.35]], dtype=torch.float32)

    with pytest.raises(ValueError, match="Unsupported GaugeMap forward method"):
        mapping.forward(z, method="typo")
    with pytest.raises(ValueError, match="Unsupported GaugeMap vjp method"):
        mapping.vjp(z, torch.ones_like(z), method="typo")
    with pytest.raises(ValueError, match="Unsupported GaugeMap gauge method"):
        mapping.gauge(z, method="typo")
    with pytest.raises(ValueError, match="Unsupported GaugeMap inverse method"):
        mapping.inverse(z, method="typo")


def test_gaugemap_default_explicit_gradient_rule_preserves_fast_path():
    mapping = GaugeMap(DummyMixedSet(), p_norm=2)

    assert mapping.explicit_gradient_rule == "polynomial"


def test_gaugemap_explicit_convexopteq_lagrangian_gradient_matches_autograd():
    seed = 2025
    n_var = 4
    np.random.seed(seed)
    np.random.randn(n_var, int(n_var**0.5))
    np.random.randn(n_var) / n_var
    x_origin = np.random.randn(n_var)

    config = create_test_problem(
        {
            "obj": "quad",
            "seed": seed,
            "n_var": n_var,
            "n_linear_cons": 1,
            "n_soc_cons": 0,
            "n_qua_cons": 1,
            "n_lin_eq": 1,
            "x_lower": -5.0,
            "x_upper": 5.0,
        }
    )
    problem = ConvexOptEq(config).to_device("cpu")
    mapping = GaugeMap(problem, p_norm=2, x_origin=x_origin)
    z = torch.tensor(
        [[0.20, -0.13, 0.17, -0.08]],
        dtype=torch.float32,
    )
    dual_var = torch.zeros((1, problem.n_eq), dtype=torch.float32)

    grad_auto = problem.gradient_lagrangian_z(
        z,
        dual_var,
        penalty_coef=10.0,
        hom_map=mapping,
        method="autograd",
        hom_map_method="autograd",
    )
    grad_explicit = problem.gradient_lagrangian_z(
        z,
        dual_var,
        penalty_coef=10.0,
        hom_map=mapping,
        method="explicit",
        hom_map_method="explicit",
    )

    assert torch.allclose(grad_explicit, grad_auto, atol=1e-7, rtol=1e-5)


def test_gaugemap_explicit_convexopteq_outer_cache_matches_direct_explicit():
    seed = 2026
    n_var = 4
    np.random.seed(seed)
    x_origin = np.random.randn(n_var)

    config = create_test_problem(
        {
            "obj": "quad",
            "seed": seed,
            "n_var": n_var,
            "n_linear_cons": 1,
            "n_soc_cons": 0,
            "n_qua_cons": 1,
            "n_lin_eq": 1,
            "x_lower": -5.0,
            "x_upper": 5.0,
        }
    )
    problem = ConvexOptEq(config).to_device("cpu")
    mapping = GaugeMap(problem, p_norm=2, x_origin=x_origin)
    z = torch.tensor([[0.18, -0.11, 0.14, -0.05]], dtype=torch.float32)
    dual_var = torch.zeros((1, problem.n_eq), dtype=torch.float32)
    x, state = mapping.forward(z, method="explicit", return_state=True)
    outer_cache = problem.build_explicit_lagrangian_z_outer_cache(dual_var, penalty_coef=10.0)

    grad_direct = problem.gradient_lagrangian_z(
        z,
        dual_var,
        penalty_coef=10.0,
        hom_map=mapping,
        method="explicit",
        hom_map_method="explicit",
        x=x,
        hom_state=state,
    )
    grad_cached = problem.gradient_lagrangian_z(
        z,
        dual_var,
        penalty_coef=10.0,
        hom_map=mapping,
        method="explicit",
        hom_map_method="explicit",
        x=x,
        hom_state=state,
        outer_cache=outer_cache,
    )

    assert torch.allclose(grad_cached, grad_direct, atol=1e-7, rtol=1e-5)


def test_gaugemap_explicit_convex_soc_mixed_objective_gradient_matches_autograd():
    seed = 2025
    n_var = 4
    np.random.seed(seed)
    np.random.randn(n_var, int(n_var**0.5))
    np.random.randn(n_var) / n_var
    x_origin = np.random.randn(n_var)

    config = create_test_problem(
        {
            "obj": "quad",
            "seed": seed,
            "n_var": n_var,
            "n_linear_cons": 1,
            "n_soc_cons": 1,
            "n_qua_cons": 1,
            "n_lin_eq": 0,
            "x_lower": -5.0,
            "x_upper": 5.0,
        }
    )
    problem = ConvexOptEq(config).to_device("cpu")
    mapping = GaugeMap(problem, p_norm=2, x_origin=x_origin)
    z = torch.tensor([[0.31, -0.48, 0.37, -0.25]], dtype=torch.float32)

    grad_auto = problem.gradient_objective_z(z, mapping, method="autograd", hom_map_method="autograd")
    grad_explicit = problem.gradient_objective_z(z, mapping, method="explicit", hom_map_method="explicit")

    assert torch.allclose(grad_explicit, grad_auto, atol=2e-5, rtol=1e-5)


def test_gaugemap_explicit_center_objective_gradient_matches_autograd():
    seed = 2026
    n_var = 4
    x_origin = np.array([0.2, -0.1, 0.15, -0.05])

    config = create_test_problem(
        {
            "obj": "quad",
            "seed": seed,
            "n_var": n_var,
            "n_linear_cons": 1,
            "n_soc_cons": 1,
            "n_qua_cons": 1,
            "n_lin_eq": 0,
            "x_lower": -5.0,
            "x_upper": 5.0,
        }
    )
    problem = ConvexOptEq(config).to_device("cpu")
    mapping = GaugeMap(problem, p_norm=2, x_origin=x_origin)
    z = torch.zeros((1, n_var), dtype=torch.float32)

    grad_auto = problem.gradient_objective_z(z, mapping, method="autograd", hom_map_method="autograd")
    grad_explicit = problem.gradient_objective_z(z, mapping, method="explicit", hom_map_method="explicit")

    assert torch.linalg.norm(grad_auto) > 1e-6
    assert torch.allclose(grad_explicit, grad_auto, atol=1e-6, rtol=1e-6)


def test_gaugemap_implicit_gradient_reconstruction_matches_polynomial_formula():
    config = create_test_problem(
        {
            "obj": "quad",
            "seed": 2031,
            "n_var": 4,
            "n_linear_cons": 2,
            "n_soc_cons": 2,
            "n_qua_cons": 2,
            "n_lin_eq": 1,
            "x_lower": -5.0,
            "x_upper": 5.0,
        }
    )
    problem = ConvexOptEq(config).to_device("cpu")
    mapping = GaugeMap(problem, p_norm=2, x_origin=np.array([0.2, -0.1, 0.15, -0.05]))
    z = torch.tensor(
        [
            [0.31, -0.48, 0.37, -0.25],
            [-0.24, 0.35, -0.18, 0.41],
            [0.18, 0.11, -0.44, -0.29],
        ],
        dtype=torch.float32,
    )
    terms = mapping._get_static_terms(z.device, z.dtype)
    common = {
        "center": terms["x0"],
        "eps": mapping._eps,
        "tie_tol": 1e-7,
        "A": terms.get("A"),
        "inv_slack": terms.get("inv_slack"),
        "inv_lower_denom": terms.get("inv_lower_denom"),
        "inv_upper_denom": terms.get("inv_upper_denom"),
        "G": terms.get("G"),
        "C": terms.get("C"),
        "Gx0": terms.get("Gx0"),
        "Cx0": terms.get("Cx0"),
        "soc_gradB_const": terms.get("soc_gradB_const"),
        "soc_Ccoef": terms.get("soc_Ccoef"),
        "Qsym": terms.get("Qsym"),
        "pq": terms.get("pq"),
        "quad_gradB_const": terms.get("quad_gradB_const"),
        "quad_Ccoef": terms.get("Cq0"),
    }
    implicit_state = explicit_forward_state(z, gradient_rule="implicit", **common)
    hybrid_state = explicit_forward_state(z, gradient_rule="hybrid", **common)
    polynomial_state = explicit_forward_state(z, gradient_rule="polynomial", **common)

    grad_x = torch.tensor(
        [
            [0.7, -0.2, 0.3, -0.5],
            [-0.1, 0.6, -0.4, 0.2],
            [0.3, 0.1, -0.6, 0.5],
        ],
        dtype=torch.float32,
    )
    implicit_vjp = mapping.vjp(z, grad_x, method="explicit", state=implicit_state)
    hybrid_vjp = mapping.vjp(z, grad_x, method="explicit", state=hybrid_state)
    polynomial_vjp = mapping.vjp(z, grad_x, method="explicit", state=polynomial_state)

    assert torch.equal(implicit_state["best_family"], polynomial_state["best_family"])
    assert torch.equal(implicit_state["best_index"], polynomial_state["best_index"])
    assert torch.equal(hybrid_state["best_family"], polynomial_state["best_family"])
    assert torch.equal(hybrid_state["best_index"], polynomial_state["best_index"])
    assert torch.allclose(implicit_state["x"], polynomial_state["x"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(hybrid_state["x"], polynomial_state["x"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(implicit_state["best_grad_u"], polynomial_state["best_grad_u"], atol=2e-5, rtol=2e-5)
    assert torch.allclose(hybrid_state["best_grad_u"], polynomial_state["best_grad_u"], atol=2e-5, rtol=2e-5)
    assert torch.allclose(implicit_vjp, polynomial_vjp, atol=2e-5, rtol=2e-5)
    assert torch.allclose(hybrid_vjp, polynomial_vjp, atol=2e-5, rtol=2e-5)


def test_convexopteq_explicit_lagrangian_x_gradient_matches_autograd():
    config = create_test_problem(
        {
            "obj": "quad",
            "seed": 2026,
            "n_var": 4,
            "n_linear_cons": 2,
            "n_soc_cons": 1,
            "n_qua_cons": 1,
            "n_lin_eq": 1,
            "x_lower": -5.0,
            "x_upper": 5.0,
        }
    )
    problem = ConvexOptEq(config).to_device("cpu")
    x = torch.tensor([[0.31, -0.17, 0.23, -0.11]], dtype=torch.float32)
    x_outer = torch.tensor([[0.10, -0.20, 0.05, 0.15]], dtype=torch.float32)
    dual_var = torch.linspace(0.05, 0.45, steps=problem.ncon + problem.n_eq).view(1, -1)

    grad_auto = problem.gradient_lagrangian_x(
        x,
        dual_var,
        penalty_coef=2.5,
        proximal_coef=0.3,
        x_outer=x_outer,
        method="autograd",
    )
    grad_explicit = problem.gradient_lagrangian_x(
        x,
        dual_var,
        penalty_coef=2.5,
        proximal_coef=0.3,
        x_outer=x_outer,
        method="explicit",
    )

    assert torch.allclose(grad_explicit, grad_auto, atol=1e-5, rtol=1e-5)
