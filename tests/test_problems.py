import json
from pathlib import Path
import numpy as np
import pytest
import torch

from homopt.mappings import GaugeMap
import homopt.experiments.benchmarks.stiefel.compare as stiefel_benchmarks
from homopt.experiments.benchmarks.stiefel.setup import (
    STIEFEL_REFERENCE_CACHE_VERSION,
    build_stiefel_initial_point,
    build_stiefel_problem,
    resolve_stiefel_restricted_rows,
    stiefel_reference_cache_key,
    stiefel_reference_cache_path,
)
from homopt.optim.registry import _algorithm_specs
from homopt.optim import (
    StiefelALMEQPGDOptimizer,
    StiefelRetractionALMOptimizer,
    StiefelRetractionOptimizer,
)
from homopt.optim.manifolds.stiefel import _project_stiefel_tangent
from homopt.problems import (
    ConvexOpt,
    ConvexOptEq,
    BMSDP,
    MaxCutSDP,
    ProjProblem,
    StiefelProblem,
    ToyStarOpt,
    create_constrained_pca_stiefel_config,
    create_test_problem,
)
from homopt.problems.convex.generators import (
    _sample_convex_objective,
    _sample_quadratic_inequalities,
    _sample_soc_inequalities,
)
from homopt.solvers import StiefelPyomoIPOPTSolver


def _payload_metrics(payload):
    return payload["metrics"]


def _load_stiefel_hom_alm_compare_module():
    import importlib.util
    import sys
    from pathlib import Path

    script_dir = Path(__file__).resolve().parents[1] / "scripts" / "hom_alm"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("run_stiefel_eq_compare", script_dir / "run_stiefel_eq_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_test_problem_defaults_work():
    config = create_test_problem({
        'seed': 0,
        'n_var': 2,
        'n_linear_cons': 1,
        'n_soc_cons': 0,
        'n_qua_cons': 0,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
    })
    assert 'Q' in config and 'p' in config


def test_bmsdp_lift_uses_the_maxcut_sdp_coordinate_order():
    config = {"node": [0, 1, 2], "edge": [(0, 1)], "weights": np.asarray([1.0])}
    sdp_problem = MaxCutSDP(config)
    factor_problem = BMSDP(config, rank=2)
    factors = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    lifted = factor_problem.lift_to_sdp_decision(factors)
    assert lifted.shape == (1, sdp_problem.nvar)
    assert torch.allclose(lifted, torch.tensor([[0.0, 1.0, 1.0]]))
    assert torch.allclose(factor_problem.objective_x(factors), sdp_problem.objective_x(lifted))


def test_create_test_problem_quad_uses_controlled_diagonal_spectrum():
    config = create_test_problem({
        'seed': 0,
        'n_var': 5,
        'n_linear_cons': 0,
        'n_soc_cons': 0,
        'n_qua_cons': 0,
        'obj': 'quad',
        'quad_diag_lower': 0.2,
        'quad_diag_upper': 0.3,
        'x_lower': -1,
        'x_upper': 1,
    })
    q_matrix = config['Q']
    q_diag = np.diag(q_matrix)
    assert np.allclose(q_matrix, np.diag(q_diag))
    assert np.all(q_diag >= 0.2)
    assert np.all(q_diag <= 0.3)


def test_create_test_problem_quad_objective_can_use_low_rank_spectrum():
    ridge = 1e-2
    config = create_test_problem({
        'seed': 2,
        'n_var': 8,
        'n_linear_cons': 0,
        'n_soc_cons': 0,
        'n_qua_cons': 0,
        'obj': 'quad',
        'objective_quadratic_type': 'low_rank',
        'low_rank_quad_ridge': ridge,
        'x_lower': -1,
        'x_upper': 1,
    })
    q_matrix = config['Q']
    off_diag = q_matrix.copy()
    diag_idx = np.arange(q_matrix.shape[0])
    off_diag[diag_idx, diag_idx] = 0.0

    assert np.max(np.abs(off_diag)) > 1e-8
    assert np.linalg.eigvalsh(q_matrix).min() >= ridge - 1e-10


@pytest.mark.parametrize("obj", ["linear", "quad", "low_rank_quad"])
def test_convex_objective_linear_term_uses_sqrt_dimension_scale(obj):
    seed = 13
    num_var = 8

    _, linear_term = _sample_convex_objective(
        np.random.RandomState(seed),
        num_var,
        obj,
        quad_diag_lower=0.05,
        quad_diag_upper=0.1,
        low_rank_quad_ridge=0.01,
    )
    rng = np.random.RandomState(seed)
    if obj == "quad":
        rng.uniform(0.05, 0.1, size=num_var)
    elif obj == "low_rank_quad":
        rng.randn(num_var, max(int(num_var**0.5), 1))
    expected_linear_term = rng.randn(num_var) / np.sqrt(num_var)

    assert np.allclose(linear_term, expected_linear_term)


def test_create_test_problem_quadratic_constraints_use_diagonal_controlled_spectrum():
    config = create_test_problem({
        'seed': 3,
        'n_var': 6,
        'n_linear_cons': 0,
        'n_soc_cons': 0,
        'n_qua_cons': 3,
        'obj': 'quad',
        'quad_diag_lower': 0.05,
        'quad_diag_upper': 0.08,
        'x_lower': -1,
        'x_upper': 1,
    })
    q_constraints = config['Qq']
    q_diag = np.diagonal(q_constraints, axis1=1, axis2=2)
    off_diag = q_constraints.copy()
    diag_idx = np.arange(q_constraints.shape[1])
    off_diag[:, diag_idx, diag_idx] = 0.0

    assert np.allclose(off_diag, 0.0)
    assert np.all(q_diag >= 0.05)
    assert np.all(q_diag <= 0.08)


def test_create_test_problem_quadratic_constraints_can_use_low_rank_spectrum():
    ridge = 1e-2
    config = create_test_problem({
        'seed': 4,
        'n_var': 8,
        'n_linear_cons': 0,
        'n_soc_cons': 0,
        'n_qua_cons': 3,
        'obj': 'quad',
        'constraint_quadratic_type': 'low_rank',
        'low_rank_quad_ridge': ridge,
        'x_lower': -1,
        'x_upper': 1,
    })
    q_constraints = config['Qq']
    off_diag = q_constraints.copy()
    diag_idx = np.arange(q_constraints.shape[1])
    off_diag[:, diag_idx, diag_idx] = 0.0

    assert np.max(np.abs(off_diag)) > 1e-8
    assert np.linalg.eigvalsh(q_constraints).min() >= ridge - 1e-10


def test_quadratic_constraint_linear_term_uses_sqrt_dimension_scale():
    seed = 11
    num_var = 7
    num_qua_cons = 4
    x_ref = np.linspace(-0.2, 0.3, num_var)

    _, p_stack, _ = _sample_quadratic_inequalities(
        np.random.RandomState(seed),
        num_qua_cons,
        num_var,
        x_ref,
        margin_scale=0.1,
        quad_diag_lower=0.05,
        quad_diag_upper=0.1,
    )
    rng = np.random.RandomState(seed)
    rng.uniform(0.05, 0.1, size=(num_qua_cons, num_var))
    expected_p_stack = rng.randn(num_qua_cons, num_var) / np.sqrt(num_var)

    assert np.allclose(p_stack, expected_p_stack)


def test_create_test_problem_constraint_margins_are_local_distance_scaled():
    x_ref = np.linspace(-0.4, 0.6, 5)
    margin_scale = 0.15

    q_stack, p_stack, bq_stack = _sample_quadratic_inequalities(
        np.random.RandomState(4),
        3,
        5,
        x_ref,
        margin_scale,
        quad_diag_lower=0.05,
        quad_diag_upper=0.1,
    )
    q_value = 0.5 * np.matmul(np.matmul(x_ref, q_stack), x_ref)
    p_value = p_stack @ x_ref
    q_margin = bq_stack - q_value - p_value
    q_grad_norm = np.linalg.norm(np.einsum("mij,j->mi", q_stack, x_ref) + p_stack, axis=1)

    g_tensor, h_tensor, c_matrix, d_vector = _sample_soc_inequalities(
        np.random.RandomState(5),
        3,
        5,
        x_ref,
        margin_scale,
    )
    soc_value = np.linalg.norm(g_tensor @ x_ref + h_tensor, ord=2, axis=1) - c_matrix @ x_ref
    soc_margin = d_vector - soc_value
    gx_ref = g_tensor @ x_ref + h_tensor
    unit = gx_ref / np.maximum(np.linalg.norm(gx_ref, ord=2, axis=1, keepdims=True), 1e-12)
    soc_grad_norm = np.linalg.norm(np.einsum("mk,mkn->mn", unit, g_tensor) - c_matrix, axis=1)

    assert np.all(q_margin >= 0.0)
    assert np.all(soc_margin >= 0.0)
    assert np.all(np.isfinite(q_margin / np.maximum(q_grad_norm, 1e-12)))
    assert np.all(np.isfinite(soc_margin / np.maximum(soc_grad_norm, 1e-12)))


def test_create_test_problem_low_rank_quad_adds_ridge():
    ridge = 1e-2
    config = create_test_problem({
        'seed': 0,
        'n_var': 5,
        'n_linear_cons': 0,
        'n_soc_cons': 0,
        'n_qua_cons': 0,
        'obj': 'low_rank_quad',
        'low_rank_quad_ridge': ridge,
        'x_lower': -1,
        'x_upper': 1,
    })
    min_eig = np.linalg.eigvalsh(config['Q']).min()
    assert min_eig >= ridge - 1e-10


def test_create_test_problem_rejects_unknown_objective_name():
    with pytest.raises(ValueError, match="obj must be one of"):
        create_test_problem({
            'seed': 0,
            'n_var': 2,
            'n_linear_cons': 0,
            'n_soc_cons': 0,
            'n_qua_cons': 0,
            'obj': 'legacy_quad',
            'x_lower': -1,
            'x_upper': 1,
        })


def test_create_test_problem_validates_anchor_ratio():
    with pytest.raises(ValueError, match="anchor_interior_ratio"):
        create_test_problem({
            'seed': 0,
            'n_var': 2,
            'n_linear_cons': 0,
            'n_soc_cons': 0,
            'n_qua_cons': 0,
            'obj': 'linear',
            'x_lower': -1,
            'x_upper': 1,
            'anchor_interior_ratio': 1.5,
        })


def test_convexopt_objective_and_constraints_smoke():
    config = create_test_problem({
        'seed': 1,
        'n_var': 2,
        'n_linear_cons': 1,
        'n_soc_cons': 0,
        'n_qua_cons': 0,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
    })
    problem = ConvexOpt(config)
    x = torch.zeros(1, problem.nvar)
    assert problem.objective_x(x).shape == (1, 1)
    assert problem.constraint_x(x).ndim == 2


def test_radial_dual_general_gauge_uses_centered_objective_and_maps_feasible():
    config = create_test_problem({
        'seed': 1,
        'n_var': 2,
        'n_linear_cons': 0,
        'n_soc_cons': 0,
        'n_qua_cons': 0,
        'obj': 'quad',
        'x_lower': -2,
        'x_upper': 2,
    })
    problem = ConvexOpt(config)
    center = torch.tensor([[0.25, -0.10]], dtype=torch.float32)
    mapping = GaugeMap(problem, p_norm=2, x_origin=center, smooth=False)
    y = torch.tensor([[1.40, -0.85]], dtype=torch.float32)

    radial_value = problem.radial_dual_objective(y, mapping)
    displacement = y - center
    p_eff = center @ problem.Q + problem.p.view(1, -1)
    py = torch.sum(p_eff * displacement, dim=-1, keepdim=True)
    y_q_y = torch.sum((displacement @ problem.Q) * displacement, dim=-1, keepdim=True)
    expected_objective_radial = (py + 1 + torch.sqrt((py + 1) ** 2 + 2 * y_q_y)) / 2
    expected = torch.maximum(expected_objective_radial, mapping.gauge(displacement))
    recovered = center + displacement / radial_value

    assert radial_value.shape == (1, 1)
    assert torch.allclose(problem.radial_primal_obj(center, mapping), torch.ones(1, 1), atol=1e-6)
    assert torch.allclose(radial_value, expected, atol=1e-6)
    assert float(problem.constraint_x(recovered).max()) <= 1e-6


def test_toy_star_problem_basic_behavior():
    prob = ToyStarOpt(alpha=0.2, num_star=4)
    x = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    obj = prob.objective_x(x)
    cons = prob.constraint_x(x)
    assert obj.shape == (1, 1)
    assert cons.shape == (1, 1)


def test_proj_problem_wraps_constraint_and_objective():
    config = create_test_problem({
        'seed': 2,
        'n_var': 2,
        'n_linear_cons': 1,
        'n_soc_cons': 0,
        'n_qua_cons': 0,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
    })
    base = ConvexOpt(config)
    proj = ProjProblem([0.1, -0.2], base)
    x = torch.zeros(1, 2)
    assert proj.objective_x(x).shape == (1, 1)
    assert proj.constraint_x(x).ndim == 2


def test_convexopteq_eq_constraints_shape():
    config = create_test_problem({
        'seed': 3,
        'n_var': 2,
        'n_linear_cons': 1,
        'n_soc_cons': 0,
        'n_qua_cons': 0,
        'n_lin_eq': 1,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
    })
    problem = ConvexOptEq(config)
    x = torch.zeros(1, 2)
    eq = problem.eq_constraint_x(x)
    assert eq.shape == (1, 1)


def test_convexopteq_without_equalities_returns_empty_eq_residual():
    config = create_test_problem({
        'seed': 3,
        'n_var': 2,
        'n_linear_cons': 1,
        'n_soc_cons': 0,
        'n_qua_cons': 0,
        'n_lin_eq': 0,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
    })
    problem = ConvexOptEq(config)
    x = torch.zeros(1, 2)
    eq = problem.eq_constraint_x(x)
    assert eq.shape == (1, 0)


def test_convexopteq_reuses_convexopt_inequality_residual_assembly():
    config = create_test_problem({
        'seed': 4,
        'n_var': 2,
        'n_linear_cons': 1,
        'n_soc_cons': 0,
        'n_qua_cons': 1,
        'n_lin_eq': 1,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
    })
    problem = ConvexOptEq(config)
    x = torch.zeros(1, 2)
    ineq_only = torch.cat(problem.inequality_constraint_terms_x(x), dim=1)
    all_cons = problem.constraint_x(x, clip=False, eq_cons=True)
    assert torch.allclose(ineq_only, all_cons[:, problem.ineq_cons])


def _stiefel_config(n_cols=2):
    return {
        "A_obj": torch.diag(torch.tensor([3.0, 2.0, 1.0])),
        "n_cols": n_cols,
        "L": -2.0,
        "U": 2.0,
        "norm_radius": 2.0,
    }


def test_stiefel_hom_alm_compare_default_initial_point_is_nonzero_and_convex_feasible():
    module = _load_stiefel_hom_alm_compare_module()
    params = module.merge_params(module.PARAMS, {"device": "cpu"})
    problem, _, runtime_dtype = build_stiefel_problem(params)
    init = build_stiefel_initial_point(problem, params, runtime_dtype)

    assert init.shape == (1, problem.nvar)
    assert init.norm().item() > 0
    assert torch.clamp(problem.constraint_x(init, clip=False, eq_cons=False), min=0).max().item() == pytest.approx(0.0)


def test_stiefel_hom_alm_compare_resolves_top_loading_restricted_rows():
    covariance = torch.diag(torch.tensor([5.0, 1.0, 4.0, 2.0]))
    rows = resolve_stiefel_restricted_rows(
        {
            "restricted_rows": "top_loading",
            "restricted_row_count": 2,
            "n_cols": 2,
        },
        covariance,
    )

    assert rows == [0, 2]


def test_stiefel_hom_alm_compare_uses_scale_label_for_output_names():
    module = _load_stiefel_hom_alm_compare_module()
    scale_label = module.scale_label(module.PARAMS)

    assert scale_label == module.PARAMS["scale_label"]
    assert module.OUTPUT_DIR == Path(__file__).resolve().parents[1] / "results" / "hom_alm" / "stiefel_eq_compare"
    assert stiefel_benchmarks._visualization_prefix(module.PARAMS) == f"stiefel_{scale_label}"
    assert stiefel_benchmarks.STIEFEL_PLOT_ALGORITHM_ORDER == [
        "Penalty",
        "Prox-Penalty",
        "ALM",
        "Prox-ALM",
        "ALM-EQ-PGD",
        "StiefelRetraction",
        "StiefelRetractionPenalty",
        "StiefelRetractionALM",
        "Hom-ALM",
        "Prox-Hom-ALM",
    ]
    assert stiefel_benchmarks.STIEFEL_PLOT_LABELS["Hom-ALM"] == "Hom-ALM"
    assert stiefel_benchmarks.STIEFEL_PLOT_LABELS["Prox-Hom-ALM"] == "Hom-PALM"
    assert stiefel_benchmarks.STIEFEL_PLOT_LABELS["ALM-EQ-PGD"] == "ALM-Eq-PGD"
    assert stiefel_benchmarks.STIEFEL_PLOT_LABELS["Prox-Penalty"] == "PPP"


def test_convex_eq_compare_uses_scale_label_for_output_names():
    import importlib.util
    import sys
    from pathlib import Path

    script_dir = Path(__file__).resolve().parents[1] / "scripts" / "hom_alm"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("run_convex_eq_compare", script_dir / "run_convex_eq_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    expected_scale_label = module.convex_problem_scale_label(module.PARAMS)
    assert expected_scale_label == "socp_eq_n10_q5_eq5"
    assert module.OUTPUT_DIR == Path(__file__).resolve().parents[1] / "results" / "hom_alm" / "convex_eq_compare"
    assert module.PARAMS["algorithms"] == module.BASE_PARAMS["algorithms"]
    assert set(module.PARAMS["algorithms"]).issubset(set(_algorithm_specs()))
    assert module.PARAMS["method_labels"]["Prox-Hom-ALM"] == "Hom-PALM"
    instance_labels = [
        module.convex_problem_scale_label(module.merge_params(module.BASE_PARAMS, overrides))
        for label, overrides in module.INSTANCE_OVERRIDES
        if label is None
    ]
    assert all(label.startswith("socp_eq_") for label in instance_labels)


def test_socp_eq_empty_instance_list_runs_current_config(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    import homopt.experiments.script_runtime as script_runtime

    script_dir = scripts_root / "hom_alm"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("run_convex_eq_compare_empty", script_dir / "run_convex_eq_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    def fake_run_labeled_benchmark_instances(
        name,
        benchmark,
        base_params,
        instances,
        *,
        repo_root=None,
        label_builder=None,
        family=None,
        parent_output_dir=None,
    ):
        del benchmark, family, repo_root
        captured.update(
            name=name,
            base_params=base_params,
            instances=instances,
            label_builder=label_builder,
            parent_output_dir=parent_output_dir,
        )
        return "ran-current-config"

    monkeypatch.setattr(script_runtime, "run_labeled_benchmark_instances", fake_run_labeled_benchmark_instances)

    assert module.main([]) == "ran-current-config"
    assert captured["name"] == module.EXPERIMENT_NAME
    assert captured["base_params"] is module.PARAMS
    assert captured["instances"] == []
    assert captured["label_builder"] is module.convex_problem_scale_label
    assert captured["parent_output_dir"] == module.OUTPUT_DIR


def test_convex_ineq_compare_uses_scale_label_for_output_names():
    import importlib.util
    import sys
    from pathlib import Path

    script_dir = Path(__file__).resolve().parents[1] / "scripts" / "hom_pgd"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("run_convex_ineq_compare", script_dir / "run_convex_ineq_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scale_label = module.convex_problem_scale_label(module.PARAMS)
    assert scale_label == "socp_n10_soc10"
    assert module.OUTPUT_DIR == Path(__file__).resolve().parents[1] / "results" / "hom_pgd" / "convex_ineq_compare"
    assert module.PARAMS["algorithms"] == ["PGD", "FW", "ALM", "RD", "Hom-PGD"]
    assert module.PARAMS["method_labels"]["Hom-PGD"] == "Hom-PGD"


def test_stiefel_solver_budgets_follow_common_outer_inner_config():
    module = _load_stiefel_hom_alm_compare_module()
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "max_iterations": 7,
            "outer_common": {
                "max_running_time": 11,
                "convergence_threshold": 1e-5,
            },
            "inner_solver_common": {
                "inner_stopping_rule": "fixed",
                "inner_iterations": 3,
                "inner_stepsize_rule": "constant",
                "inner_lr_decay": 0.7,
            },
        },
    )

    nested = stiefel_benchmarks._stiefel_nested_solver_base_params(params)
    penalty = stiefel_benchmarks._stiefel_penalty_solver_base_params(params)
    alm_params = stiefel_benchmarks._build_alm_params(params)

    assert nested["outer_iterations"] == 7
    assert nested["inner_iterations"] == 3
    assert nested["max_running_time"] == 11
    assert nested["convergence_threshold"] == pytest.approx(1e-5)
    assert nested["learning_rate"] == pytest.approx(params["common_config"]["learning_rate"])
    assert nested["stepsize_rule"] == "constant"
    assert nested["lr_decay"] == pytest.approx(0.7)
    assert penalty["outer_iterations"] == 7
    assert penalty["inner_iterations"] == 3
    assert penalty["max_running_time"] == 11
    assert penalty["convergence_threshold"] == pytest.approx(1e-5)
    assert penalty["learning_rate"] == pytest.approx(params["common_config"]["learning_rate"])
    assert penalty["penalty_coef"] == pytest.approx(params["outer_common"]["penalty_coef"])
    assert penalty["penalty_growth"] == pytest.approx(params["outer_common"]["penalty_growth"])
    assert alm_params["ALM"].get("use_proximal", False) is False
    assert alm_params["Prox-ALM"]["use_proximal"] is True
    assert alm_params["Hom-ALM"].get("use_proximal", False) is False
    assert alm_params["Prox-Hom-ALM"]["use_proximal"] is True

    invalid_params = module.merge_params(
        module.BASE_PARAMS,
        {"common_config": {"outer_lr_decay": 0.9}},
    )
    with pytest.raises(ValueError, match="Unsupported common_config keys"):
        stiefel_benchmarks._outer_config(invalid_params)


def test_stiefel_algorithm_config_is_merged():
    module = _load_stiefel_hom_alm_compare_module()
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithm_config": {
                "Prox-Hom-ALM": {
                    "hom_map_gradient": "explicit",
                    "inner_learning_rate": 3e-4,
                    "custom_flag": "from_algorithm_config",
                },
                "StiefelRetractionPenalty": {
                    "penalty_growth": 1.07,
                    "penalty_coef": 5.0,
                },
            },
        },
    )

    algorithm_config = stiefel_benchmarks._algorithm_config(params)
    alm_params = stiefel_benchmarks._build_alm_params(params)

    assert algorithm_config["Prox-Hom-ALM"]["custom_flag"] == "from_algorithm_config"
    assert algorithm_config["Prox-Hom-ALM"]["inner_learning_rate"] == pytest.approx(3e-4)
    assert algorithm_config["StiefelRetractionPenalty"]["penalty_growth"] == pytest.approx(1.07)
    assert algorithm_config["StiefelRetractionPenalty"]["penalty_coef"] == pytest.approx(5.0)
    assert alm_params["Prox-Hom-ALM"]["custom_flag"] == "from_algorithm_config"
    assert alm_params["Prox-Hom-ALM"]["inner_learning_rate"] == pytest.approx(3e-4)


def test_stiefel_hom_alm_compare_writes_shared_visualization_artifacts(tmp_path):
    module = _load_stiefel_hom_alm_compare_module()
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["StiefelRetraction"],
            "device": "cpu",
            "verbose": False,
            "visualize": True,
            "include_reference_solver": False,
            "reference_need_opt": False,
            "visualization_prefix": "stiefel_smoke",
            "max_iterations": 3,
                "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
            "algorithm_config": {
                "StiefelRetraction": {
                    "learning_rate": 0.1,
                    "stepsize_rule": "constant",
                    "convergence_threshold": 1e-12,
                    "inequality_penalty_coef": 10.0,
                }
            },
        },
    )

    payload = stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    assert payload["artifacts"]["objective_convergence_iteration"] == "artifacts/stiefel_smoke_objective_convergence_by_iter.pdf"
    assert (
        payload["artifacts"]["objective_convergence_iteration_logx"]
        == "artifacts/stiefel_smoke_objective_convergence_by_iter_logx.pdf"
    )
    assert (
        payload["artifacts"]["objective_convergence_time_logx"]
        == "artifacts/stiefel_smoke_objective_convergence_by_time_logx.pdf"
    )
    assert (tmp_path / payload["artifacts"]["objective_convergence_iteration"]).exists()
    assert (tmp_path / payload["artifacts"]["objective_convergence_iteration_logx"]).exists()
    assert (tmp_path / payload["artifacts"]["objective_convergence_time_logx"]).exists()
    assert (tmp_path / payload["artifacts"]["objective_convergence_time"]).exists()
    assert (tmp_path / payload["artifacts"]["runtime_summary"]).exists()
    assert (tmp_path / payload["artifacts"]["metric_traces"]).exists()


def test_stiefel_visualization_reference_label_is_plain_ipopt(monkeypatch, tmp_path):
    module = _load_stiefel_hom_alm_compare_module()
    captured = {}

    def fake_save_comparison_visualizations(**kwargs):
        captured.update(kwargs)
        return {"runtime_summary": "artifacts/runtime.pdf"}

    monkeypatch.setattr(stiefel_benchmarks, "save_comparison_visualizations", fake_save_comparison_visualizations)
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["StiefelRetraction"],
            "device": "cpu",
            "verbose": False,
            "visualize": True,
            "include_reference_solver": False,
            "reference_need_opt": False,
            "max_iterations": 1,
                "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
            "algorithm_config": {
                "StiefelRetraction": {
                    "learning_rate": 0.1,
                    "stepsize_rule": "constant",
                    "convergence_threshold": 1e-12,
                    "inequality_penalty_coef": 10.0,
                }
            },
        },
    )

    stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    assert captured["reference_label"] == "IPOPT"
    assert captured["violation_y_min"] == pytest.approx(params["outer_common"]["convergence_threshold"])


def test_stiefel_hom_alm_compare_visualize_only_reuses_existing_records(monkeypatch, tmp_path):
    module = _load_stiefel_hom_alm_compare_module()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    records_dir = artifact_dir / "records"
    records_dir.mkdir()
    records = {
        "StiefelRetractionPenalty": {
            "runtime_total": 0.1,
            "extras": {
                "objective_traj": np.array([-0.1, -0.2]),
                "eq_violation_traj": np.array([0.0, 0.0]),
                "ineq_violation_traj": np.array([1e-3, 1e-6]),
                "outer_iter_time": np.array([0.01]),
            },
        }
    }
    summaries = {
        "StiefelRetractionPenalty": {
            "final_objective": -0.2,
            "status": "max_iterations",
            "record_path": "artifacts/records/StiefelRetractionPenalty.npy",
        }
    }
    np.save(records_dir / "StiefelRetractionPenalty.npy", records["StiefelRetractionPenalty"], allow_pickle=True)
    (artifact_dir / "summary.json").write_text(json.dumps(summaries), encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "algorithms": ["StiefelRetractionPenalty"],
                "records": {
                    "StiefelRetractionPenalty": "artifacts/records/StiefelRetractionPenalty.npy",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "name": "stiefel_hom_alm_compare_smoke",
                "objective": -0.2,
                "feasible": True,
                "artifacts": {"records": "artifacts/records"},
                "metrics": {
                    "algorithms": ["StiefelRetractionPenalty"],
                    "requested_algorithms": ["StiefelRetractionPenalty"],
                    "reference_objective": -0.25,
                    "results": summaries,
                },
                "context": {},
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_save_comparison_visualizations(**kwargs):
        captured.update(kwargs)
        return {"runtime_summary": "artifacts/stiefel_runtime_summary.pdf"}

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("visualize_only must not rerun solvers")

    monkeypatch.setattr(stiefel_benchmarks, "save_comparison_visualizations", fake_save_comparison_visualizations)
    monkeypatch.setattr(stiefel_benchmarks, "run_algorithm", fail_if_called)
    monkeypatch.setattr(stiefel_benchmarks, "_run_stiefel_ipopt", fail_if_called)
    monkeypatch.setattr(stiefel_benchmarks, "_run_stiefel_retraction_penalty", fail_if_called)

    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["StiefelRetractionPenalty"],
            "device": "cpu",
            "verbose": False,
            "visualize": True,
            "visualize_only": True,
            "show_convergence_legend": False,
            "include_reference_solver": False,
            "reference_need_opt": False,
                "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
        },
    )

    payload = stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    assert _payload_metrics(payload)["visualization_only"] is True
    assert payload["artifacts"]["runtime_summary"] == "artifacts/stiefel_runtime_summary.pdf"
    assert set(captured["records"]) == {"StiefelRetractionPenalty"}
    assert np.allclose(
        captured["records"]["StiefelRetractionPenalty"]["extras"]["ineq_violation_traj"],
        [1e-3, 1e-6],
    )
    assert captured["algorithms"] == ["StiefelRetractionPenalty"]
    assert captured["reference_objective"] == pytest.approx(-0.25)
    assert captured["show_convergence_legend"] is False


def test_stiefel_hom_alm_compare_accepts_retraction_alm_algorithm(tmp_path):
    module = _load_stiefel_hom_alm_compare_module()
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["StiefelRetractionALM"],
            "device": "cpu",
            "verbose": False,
            "visualize": False,
            "include_reference_solver": False,
            "reference_need_opt": False,
            "max_iterations": 2,
            "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
            "algorithm_config": {
                "StiefelRetractionALM": {
                    "learning_rate": 0.1,
                    "inner_iterations": 2,
                    "stepsize_rule": "constant",
                    "convergence_threshold": 1e-12,
                    "penalty_coef": 5.0,
                }
            },
        },
    )

    payload = stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    metrics = _payload_metrics(payload)
    assert "StiefelRetractionALM" in metrics["results"]
    assert metrics["results"]["StiefelRetractionALM"]["status"] in {"converged", "max_iterations"}


def test_stiefel_hom_alm_compare_accepts_prox_penalty_algorithm(tmp_path):
    module = _load_stiefel_hom_alm_compare_module()
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["Prox-Penalty"],
            "device": "cpu",
            "verbose": False,
            "visualize": False,
            "include_reference_solver": False,
            "reference_need_opt": False,
            "max_iterations": 2,
            "common_config": {
                "learning_rate": 0.01,
                "stepsize_rule": "constant",
            },
            "outer_common": {
                "max_running_time": 10,
                "convergence_threshold": 1e-12,
            },
            "inner_solver_common": {
                "inner_iterations": 2,
                "inner_iterations_min": 2,
                "inner_iterations_max": 2,
                "inner_stepsize_rule": "constant",
                "inner_learning_rate": 0.01,
                "momentum": 0.0,
            },
            "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
        },
    )

    payload = stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    metrics = _payload_metrics(payload)
    assert "Prox-Penalty" in metrics["results"]
    result = metrics["results"]["Prox-Penalty"]
    assert result["iterations"] == 2
    assert result["final_full_violation"] >= 0.0
    optimizer_params = stiefel_benchmarks._build_alm_params(params)
    assert optimizer_params["Prox-Penalty"]["use_lagrangian"] is False
    assert optimizer_params["Prox-Penalty"]["use_penalty"] is True
    assert optimizer_params["Prox-Penalty"]["use_proximal"] is True


def test_stiefel_retraction_alias_uses_penalty_config(tmp_path):
    module = _load_stiefel_hom_alm_compare_module()
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["StiefelRetraction"],
            "device": "cpu",
            "verbose": False,
            "visualize": False,
            "include_reference_solver": False,
            "reference_need_opt": False,
            "max_iterations": 2,
            "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
            "algorithm_config": {
                "StiefelRetractionPenalty": {
                    "learning_rate": 0.1,
                    "stepsize_rule": "constant",
                    "convergence_threshold": 1e-12,
                    "inequality_penalty_coef": 7.0,
                }
            },
        },
    )

    payload = stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    metrics = _payload_metrics(payload)
    assert "StiefelRetraction" in metrics["results"]
    assert metrics["results"]["StiefelRetraction"]["status"] in {"converged", "max_iterations"}


def test_stiefel_hom_alm_compare_treats_ipopt_as_reference_solver(monkeypatch, tmp_path):
    module = _load_stiefel_hom_alm_compare_module()
    calls = {"count": 0}

    def fake_ipopt(problem, params, init_point):
        del params, init_point
        calls["count"] += 1
        solution = np.array([[0.0, 1.0, 0.0]], dtype=float)
        x = torch.tensor(solution, dtype=problem.A_obj.dtype, device=problem.device)
        return {
            "solution": solution,
            "status": "optimal",
            "objective": float(problem.objective_x(x).item()),
            "runtime_total": 0.25,
            "violation": 0.0,
            "extras": {},
        }

    monkeypatch.setattr(stiefel_benchmarks, "_run_stiefel_ipopt", fake_ipopt)
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["StiefelIPOPT", "StiefelRetractionPenalty"],
            "device": "cpu",
            "verbose": False,
            "visualize": False,
            "max_iterations": 3,
            "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
            "algorithm_config": {
                "StiefelRetractionPenalty": {
                    "learning_rate": 0.1,
                    "stepsize_rule": "constant",
                    "convergence_threshold": 1e-12,
                    "inequality_penalty_coef": 10.0,
                }
            },
        },
    )

    payload = stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    metrics = _payload_metrics(payload)
    assert metrics["reference_objective"] == pytest.approx(-2.0)
    assert metrics["reference_solver_time"] == pytest.approx(0.25)
    assert metrics["reference_cache_enabled"] is True
    assert metrics["reference_cache_hit"] is False
    assert metrics["reference_cache_path"] == "artifacts/stiefel_reference_context_cache.npy"
    assert "StiefelIPOPT" not in metrics["results"]
    assert metrics["algorithms"] == ["StiefelRetractionPenalty"]
    assert metrics["requested_algorithms"] == ["StiefelIPOPT", "StiefelRetractionPenalty"]
    assert metrics["comparison_rows"][0]["method"] == "StiefelIPOPT"
    assert metrics["comparison_rows"][0]["route"] == "solver"
    diagnostics = metrics["reference_solution_diagnostics"]
    assert diagnostics["available"] is True
    assert diagnostics["counts_by_type"]["soc"]["total"] == 1
    assert diagnostics["counts_by_type"]["equality"]["violated"] == 0
    assert diagnostics["box_total"] == 6
    assert diagnostics["group_total"] == 1
    assert diagnostics["active_group_total"] == 0
    assert diagnostics["violated_group_total"] == 0

    cached_payload = stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    assert calls["count"] == 1
    cached_metrics = _payload_metrics(cached_payload)
    assert cached_metrics["reference_cache_hit"] is True
    assert cached_metrics["reference_cached_solver_time"] == pytest.approx(0.25)
    assert cached_metrics["reference_cached_opt_solver_time"] == pytest.approx(0.25)
    assert cached_metrics["reference_objective"] == pytest.approx(-2.0)


def test_stiefel_hom_alm_compare_defaults_to_ipopt_reference_solver(monkeypatch, tmp_path):
    module = _load_stiefel_hom_alm_compare_module()

    def fake_ipopt(problem, params, init_point):
        del params, init_point
        solution = np.array([[0.0, 1.0, 0.0]], dtype=float)
        x = torch.tensor(solution, dtype=problem.A_obj.dtype, device=problem.device)
        return {
            "solution": solution,
            "status": "optimal",
            "objective": float(problem.objective_x(x).item()),
            "runtime_total": 0.25,
            "violation": 0.0,
            "extras": {},
        }

    monkeypatch.setattr(stiefel_benchmarks, "_run_stiefel_ipopt", fake_ipopt)
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["StiefelRetractionPenalty"],
            "device": "cpu",
            "verbose": False,
            "visualize": False,
            "max_iterations": 1,
            "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
            "algorithm_config": {
                "StiefelRetractionPenalty": {
                    "learning_rate": 0.1,
                    "stepsize_rule": "constant",
                    "convergence_threshold": 1e-12,
                    "inequality_penalty_coef": 10.0,
                }
            },
        },
    )

    payload = stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    metrics = _payload_metrics(payload)
    assert metrics["reference_need_opt"] is True
    assert metrics["requested_algorithms"] == ["StiefelRetractionPenalty"]
    assert metrics["comparison_rows"][0]["method"] == "StiefelIPOPT"
    assert metrics["comparison_rows"][0]["route"] == "solver"
    assert "StiefelIPOPT" not in metrics["results"]


def test_stiefel_hom_alm_compare_supports_ipopt_only_reference_run(monkeypatch, tmp_path):
    module = _load_stiefel_hom_alm_compare_module()

    def fake_ipopt(problem, params, init_point):
        del params, init_point
        solution = np.array([[0.0, 1.0, 0.0]], dtype=float)
        x = torch.tensor(solution, dtype=problem.A_obj.dtype, device=problem.device)
        return {
            "solution": solution,
            "status": "optimal",
            "objective": float(problem.objective_x(x).item()),
            "runtime_total": 0.25,
            "violation": 0.0,
            "extras": {},
        }

    monkeypatch.setattr(stiefel_benchmarks, "_run_stiefel_ipopt", fake_ipopt)
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["StiefelIPOPT"],
            "device": "cpu",
            "verbose": False,
            "visualize": False,
            "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
        },
    )

    payload = stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    metrics = _payload_metrics(payload)
    assert metrics["algorithms"] == []
    assert metrics["requested_algorithms"] == ["StiefelIPOPT"]
    assert metrics["comparison_rows"] == [metrics["instance_rows"][0]]
    assert metrics["comparison_rows"][0]["method"] == "StiefelIPOPT"
    assert metrics["reference_objective"] == pytest.approx(-2.0)


def test_stiefel_reference_cache_without_solution_recomputes_ipopt(monkeypatch, tmp_path):
    module = _load_stiefel_hom_alm_compare_module()
    calls = {"count": 0}

    def fake_ipopt(problem, params, init_point):
        del params, init_point
        calls["count"] += 1
        solution = np.array([[0.0, 1.0, 0.0]], dtype=float)
        x = torch.tensor(solution, dtype=problem.A_obj.dtype, device=problem.device)
        return {
            "solution": solution,
            "status": "optimal",
            "objective": float(problem.objective_x(x).item()),
            "runtime_total": 0.25,
            "violation": 0.0,
            "extras": {},
        }

    monkeypatch.setattr(stiefel_benchmarks, "_run_stiefel_ipopt", fake_ipopt)
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["StiefelIPOPT"],
            "device": "cpu",
            "verbose": False,
            "visualize": False,
            "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
        },
    )
    cache_path = stiefel_reference_cache_path(tmp_path, cache_reference=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_key = stiefel_reference_cache_key(params)
    np.save(
        cache_path,
        {
            "version": STIEFEL_REFERENCE_CACHE_VERSION,
            "entries": {
                cache_key: {
                    "result": {
                        "solution": None,
                        "status": "stale_missing_solution",
                    }
                }
            },
        },
        allow_pickle=True,
    )

    with pytest.warns(RuntimeWarning, match="has no solution; recomputing IPOPT"):
        payload = stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)

    assert calls["count"] == 1
    metrics = _payload_metrics(payload)
    assert metrics["reference_cache_hit"] is False
    assert metrics["reference_objective"] == pytest.approx(-2.0)


def test_stiefel_reference_failure_is_explicit(monkeypatch, tmp_path):
    module = _load_stiefel_hom_alm_compare_module()

    def fake_ipopt(problem, params, init_point):
        del problem, params, init_point
        return {
            "solution": None,
            "status": "ipopt_unavailable",
            "message": "ipopt executable not found",
            "runtime_total": 0.0,
            "violation": None,
            "extras": {},
        }

    monkeypatch.setattr(stiefel_benchmarks, "_run_stiefel_ipopt", fake_ipopt)
    params = module.merge_params(
        module.BASE_PARAMS,
        {
            "algorithms": ["StiefelRetractionPenalty"],
            "device": "cpu",
            "verbose": False,
            "visualize": False,
            "reference_cache": False,
            "problem_config": {
                "data_source": "synthetic",
                "n_rows": 3,
                "n_cols": 1,
                "spectrum": [3.0, 2.0, 1.0],
                "box_lower": -1.5,
                "box_upper": 1.5,
                "restricted_rows": [0],
                "group_budget": 0.5,
            },
        },
    )

    with pytest.raises(RuntimeError, match="StiefelIPOPT reference solve failed"):
        stiefel_benchmarks.stiefel_algorithm_comparison(output_dir=tmp_path, **params)


def test_stiefel_problem_objective_equality_and_constraints():
    problem = StiefelProblem(_stiefel_config(n_cols=2))
    x = torch.eye(3, 2).reshape(1, -1)

    assert problem.nvar == 6
    assert problem.n_eq == 4
    assert problem.objective_x(x).shape == (1, 1)
    assert torch.allclose(problem.objective_x(x), torch.tensor([[-5.0]]))
    assert torch.allclose(problem.eq_constraint_x(x), torch.zeros(1, 4))

    ineq = problem.constraint_x(x, clip=False, eq_cons=False)
    all_cons = problem.constraint_x(x, clip=False, eq_cons=True)
    assert ineq.shape == (1, problem.ncon)
    assert all_cons.shape == (1, problem.ncon + problem.n_eq)
    assert torch.all(ineq <= 0)


def test_stiefel_problem_explicit_lagrangian_gradient_matches_autograd():
    problem = StiefelProblem(_stiefel_config(n_cols=2))
    x = torch.tensor([[0.2, -0.1, 0.3, 0.4, -0.2, 0.1]], dtype=torch.float32)
    dual = torch.linspace(-0.3, 0.4, problem.ncon + problem.n_eq).view(1, -1)

    grad_auto = problem.gradient_lagrangian_x(
        x,
        dual,
        penalty_coef=1.7,
        proximal_coef=0.2,
        x_outer=torch.zeros_like(x),
        method="autograd",
    )
    grad_explicit = problem.gradient_lagrangian_x(
        x,
        dual,
        penalty_coef=1.7,
        proximal_coef=0.2,
        x_outer=torch.zeros_like(x),
        method="explicit",
    )

    assert torch.allclose(grad_auto, grad_explicit, atol=1e-5)


def test_stiefel_problem_explicit_lagrangian_gradient_matches_autograd_with_group_norm():
    problem = StiefelProblem(
        {
            "A_obj": torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])),
            "n_cols": 2,
            "L": -1.5,
            "U": 1.5,
            "group_norm_constraints": [
                {
                    "name": "restricted_first_row",
                    "rows": [0],
                    "budget": 0.25,
                }
            ],
        }
    )
    x = torch.tensor([[0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.1, -0.3]], dtype=torch.float32)
    dual = torch.linspace(-0.4, 0.5, problem.ncon + problem.n_eq).view(1, -1)

    grad_auto = problem.gradient_lagrangian_x(
        x,
        dual,
        penalty_coef=2.0,
        proximal_coef=0.1,
        x_outer=torch.zeros_like(x),
        method="autograd",
    )
    grad_explicit = problem.gradient_lagrangian_x(
        x,
        dual,
        penalty_coef=2.0,
        proximal_coef=0.1,
        x_outer=torch.zeros_like(x),
        method="explicit",
    )

    assert torch.allclose(grad_auto, grad_explicit, atol=1e-5)


def test_stiefel_problem_explicit_z_lagrangian_gradient_matches_autograd():
    problem = StiefelProblem(
        {
            "A_obj": torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])),
            "n_cols": 2,
            "L": -1.5,
            "U": 1.5,
            "group_norm_constraints": [
                {
                    "name": "restricted_first_row",
                    "rows": [0],
                    "budget": 0.35,
                }
            ],
        }
    )
    mapping = GaugeMap(problem, p_norm=2, x_origin=torch.zeros(problem.nvar), smooth=False)
    z = torch.tensor([[0.10, -0.15, 0.25, 0.20, -0.10, 0.18, 0.05, -0.12]], dtype=torch.float32)
    dual = torch.linspace(-0.2, 0.3, problem.n_eq).view(1, -1)
    z_outer = torch.zeros_like(z)

    grad_auto = problem.gradient_lagrangian_z(
        z,
        dual,
        penalty_coef=1.3,
        proximal_coef=0.2,
        hom_map=mapping,
        z_outer=z_outer,
        proximal_space="z",
        hom_map_method="autograd",
    )
    x, state = mapping.forward(z, method="explicit", return_state=True)
    grad_explicit = problem.gradient_lagrangian_z(
        z,
        dual,
        penalty_coef=1.3,
        proximal_coef=0.2,
        hom_map=mapping,
        z_outer=z_outer,
        proximal_space="z",
        hom_map_method="explicit",
        method="explicit",
        x=x,
        hom_state=state,
    )

    assert torch.allclose(grad_auto, grad_explicit, atol=1e-4)


def test_stiefel_problem_explicit_z_objective_gradient_matches_autograd():
    problem = StiefelProblem(
        {
            "A_obj": torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])),
            "n_cols": 2,
            "L": -1.5,
            "U": 1.5,
            "group_norm_constraints": [{"rows": [0], "budget": 0.35}],
        }
    )
    mapping = GaugeMap(problem, p_norm=2, x_origin=torch.zeros(problem.nvar), smooth=False)
    z = torch.tensor([[0.10, -0.15, 0.25, 0.20, -0.10, 0.18, 0.05, -0.12]], dtype=torch.float32)
    x, state = mapping.forward(z, method="explicit", return_state=True)

    grad_auto = problem.gradient_objective_z(z, mapping, method="autograd", hom_map_method="autograd")
    grad_explicit = problem.gradient_objective_z(
        z,
        mapping,
        method="explicit",
        hom_map_method="explicit",
        x=x,
        hom_state=state,
    )

    assert torch.allclose(grad_explicit, grad_auto, atol=1e-4, rtol=1e-4)


def test_stiefel_problem_convex_side_works_with_gauge_map():
    problem = StiefelProblem(_stiefel_config(n_cols=1))
    mapping = GaugeMap(problem, p_norm=2, x_origin=torch.zeros(problem.nvar), smooth=False)
    z = torch.tensor(
        [
            [0.8, 0.1, -0.2],
            [-0.4, 0.5, 0.3],
        ],
        dtype=torch.float32,
    )

    x = mapping.forward(z, method="explicit")
    ineq = problem.constraint_x(x, clip=False, eq_cons=False)

    assert x.shape == z.shape
    assert torch.all(ineq <= 1e-5)


def test_stiefel_problem_can_model_active_box_constraints():
    lower = torch.full((3,), -1.5)
    upper = torch.full((3,), 1.5)
    lower[0] = -0.2
    upper[0] = 0.2
    problem = StiefelProblem(
        {
            "A_obj": torch.diag(torch.tensor([3.0, 2.0, 1.0])),
            "n_cols": 1,
            "L": lower,
            "U": upper,
            "norm_radius": 1.0,
        }
    )
    unconstrained_stiefel_point = torch.tensor([[1.0, 0.0, 0.0]])
    tight_feasible_point = torch.tensor([[0.2, (1.0 - 0.2**2) ** 0.5, 0.0]])

    assert problem.eq_constraint_x(unconstrained_stiefel_point).abs().max() <= 1e-6
    assert problem.constraint_x(unconstrained_stiefel_point, clip=True, eq_cons=False).max() == torch.tensor(0.8)
    assert problem.eq_constraint_x(tight_feasible_point).abs().max() <= 1e-6
    assert problem.constraint_x(tight_feasible_point, clip=True, eq_cons=False).max() <= 1e-6


def test_stiefel_problem_can_model_active_feature_group_norm_constraints():
    problem = StiefelProblem(
        {
            "A_obj": torch.diag(torch.tensor([3.0, 2.0, 1.0])),
            "n_cols": 1,
            "L": -1.5,
            "U": 1.5,
            "group_norm_constraints": [
                {
                    "name": "restricted_first_feature",
                    "rows": [0],
                    "budget": 0.2,
                }
            ],
        }
    )
    unconstrained_stiefel_point = torch.tensor([[1.0, 0.0, 0.0]])
    group_feasible_point = torch.tensor([[0.2, (1.0 - 0.2**2) ** 0.5, 0.0]])

    assert problem.n_soc == 1
    assert problem.eq_constraint_x(unconstrained_stiefel_point).abs().max() <= 1e-6
    assert problem.constraint_x(unconstrained_stiefel_point, clip=True, eq_cons=False).max() == torch.tensor(0.8)
    assert problem.eq_constraint_x(group_feasible_point).abs().max() <= 1e-6
    assert problem.constraint_x(group_feasible_point, clip=True, eq_cons=False).max() <= 1e-6


def test_constrained_pca_stiefel_config_can_normalize_top_sum_objective():
    config = create_constrained_pca_stiefel_config(
        covariance=torch.diag(torch.tensor([4.0, 2.0, 1.0])),
        n_cols=2,
        objective_normalization="top_sum",
        L=-1.5,
        U=1.5,
    )
    problem = StiefelProblem(config)
    x = torch.eye(3, 2).reshape(1, -1)

    assert config["objective_normalization"] == "top_sum"
    assert config["objective_scale"] == pytest.approx(6.0)
    assert torch.allclose(problem.A_obj, torch.diag(torch.tensor([4.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0])))
    assert problem.objective_x(x).item() == pytest.approx(-1.0)


def test_constrained_pca_stiefel_config_builds_group_norm_problem():
    data = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.5, 1.0],
            [-2.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    config = create_constrained_pca_stiefel_config(
        data=data,
        n_cols=1,
        restricted_rows=[0],
        group_budget=0.3,
        L=-1.5,
        U=1.5,
    )
    problem = StiefelProblem(config)

    assert problem.A_obj.shape == (3, 3)
    assert problem.group_norm_constraints[0]["rows"] == [0]
    assert problem.group_norm_constraints[0]["budget"] == 0.3
    assert problem.n_soc == 1


def test_stiefel_retraction_solver_preserves_orthogonality():
    problem = StiefelProblem(_stiefel_config(n_cols=1))
    solver = StiefelRetractionOptimizer(problem)

    result = solver.solve_result(learning_rate=0.2, max_iterations=80, seed=0)
    x = torch.tensor(result["solution"], dtype=torch.float32)

    assert result["objective"] < -2.9
    assert problem.eq_constraint_x(x).abs().max() < 1e-5
    assert problem.constraint_x(x, clip=True, eq_cons=False).max() <= 1e-5


def test_stiefel_retraction_solver_inequality_penalty_reduces_group_violation():
    problem = StiefelProblem(
        {
            "A_obj": torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])),
            "n_cols": 2,
            "L": -1.5,
            "U": 1.5,
            "group_norm_constraints": [{"rows": [0, 1], "budget": 0.45}],
        }
    )
    solver = StiefelRetractionOptimizer(problem)

    no_penalty = solver.solve_result(learning_rate=0.1, max_iterations=80, seed=0)
    with_penalty = solver.solve_result(
        learning_rate=0.1,
        max_iterations=80,
        inequality_penalty_coef=100.0,
        seed=0,
    )

    assert with_penalty["extras"]["ineq_violation"] < no_penalty["extras"]["ineq_violation"]
    assert with_penalty["extras"]["inequality_penalty_coef"] == pytest.approx(100.0)


def test_stiefel_retraction_penalty_plot_timing_matches_metric_traj():
    problem = StiefelProblem(_stiefel_config(n_cols=1))
    solver = StiefelRetractionOptimizer(problem)

    result = solver.solve_result(
        learning_rate=10.0,
        outer_iterations=3,
        inner_iterations=5,
        stepsize_rule="adaptive",
        convergence_threshold=0.0,
        seed=0,
    )
    extras = result["extras"]

    assert len(extras["iter_time"]) == len(extras["objective_traj"]) - 1
    assert len(extras["inner_iter_time"]) >= len(extras["iter_time"])


def test_stiefel_retraction_alm_plot_timing_matches_metric_traj():
    problem = StiefelProblem(_stiefel_config(n_cols=1))
    solver = StiefelRetractionALMOptimizer(problem)

    result = solver.solve_result(
        learning_rate=10.0,
        outer_iterations=3,
        inner_iterations=5,
        stepsize_rule="adaptive",
        convergence_threshold=0.0,
        seed=0,
    )
    extras = result["extras"]

    assert len(extras["iter_time"]) == len(extras["objective_traj"]) - 1
    assert len(extras["inner_iter_time"]) >= len(extras["iter_time"])


def test_stiefel_retraction_penalty_gradient_matches_autograd_objective():
    problem = StiefelProblem(
        {
            "A_obj": torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])),
            "n_cols": 2,
            "L": -1.5,
            "U": 1.5,
            "group_norm_constraints": [{"rows": [0, 1], "budget": 0.45}],
        }
    )
    solver = StiefelRetractionOptimizer(problem)
    x = solver._initial_point(seed=0)
    penalty_coef = 7.0

    explicit_grad = solver._riemannian_penalized_gradient_x(x, penalty_coef)
    x_autograd = x.detach().requires_grad_(True)
    loss = solver._penalized_objective_x(x_autograd, penalty_coef).sum()
    euclidean_grad = torch.autograd.grad(loss, x_autograd, create_graph=False)[0]
    autograd_grad = _project_stiefel_tangent(problem, x_autograd, euclidean_grad)

    assert torch.allclose(explicit_grad, autograd_grad, atol=1e-5, rtol=1e-5)


def test_stiefel_retraction_penalty_solver_increases_penalty_across_outer_loops():
    problem = StiefelProblem(_stiefel_config(n_cols=1))
    solver = StiefelRetractionOptimizer(problem)

    result = solver.solve_result(
        learning_rate=0.1,
        outer_iterations=3,
        inner_iterations=1,
        penalty_coef=2.0,
        penalty_growth=2.0,
        max_penalty=5.0,
        convergence_threshold=0.0,
        seed=0,
    )

    assert result["extras"]["outer_iterations"] == 3
    assert result["extras"]["inner_iterations"] == 1
    assert result["extras"]["penalty_traj"].tolist() == pytest.approx([2.0, 4.0, 5.0])
    assert result["extras"]["penalty_coef"] == pytest.approx(5.0)


def test_stiefel_retraction_alm_solver_updates_inequality_duals():
    problem = StiefelProblem(
        {
            "A_obj": torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])),
            "n_cols": 2,
            "L": -1.5,
            "U": 1.5,
            "group_norm_constraints": [{"rows": [0, 1], "budget": 0.45}],
        }
    )
    solver = StiefelRetractionALMOptimizer(problem)
    baseline = StiefelRetractionOptimizer(problem).solve_result(
        learning_rate=0.1,
        max_iterations=80,
        inequality_penalty_coef=0.0,
        seed=0,
    )

    result = solver.solve_result(
        learning_rate=0.1,
        outer_iterations=20,
        inner_iterations=5,
        penalty_coef=10.0,
        penalty_growth=1.2,
        convergence_threshold=1e-8,
        seed=0,
    )

    assert result["extras"]["ineq_violation"] < baseline["extras"]["ineq_violation"]
    assert result["extras"]["dual_norm"] > 0.0
    assert result["extras"]["dual_norm_traj"][-1] > 0.0
    assert problem.eq_constraint_x(torch.tensor(result["solution"], dtype=torch.float32)).abs().max() < 1e-5


def test_stiefel_retraction_alm_requires_stationarity_to_converge():
    problem = StiefelProblem(_stiefel_config(n_cols=1))
    solver = StiefelRetractionALMOptimizer(problem)

    result = solver.solve_result(
        learning_rate=0.0,
        outer_iterations=1,
        inner_iterations=1,
        penalty_coef=1.0,
        convergence_threshold=1e-8,
        seed=0,
    )

    assert result["extras"]["ineq_violation"] <= 1e-8
    assert result["extras"]["final_first_order_lagrangian_gap"] > 1e-8
    assert result["status"] == "max_iterations"


def test_stiefel_solver_verbose_prints_compact_iteration_table(capsys):
    problem = StiefelProblem(_stiefel_config(n_cols=1))
    solver = StiefelRetractionOptimizer(problem)

    solver.solve_result(learning_rate=0.1, max_iterations=1, verbose=True, verbose_interval=1, seed=0)
    output = capsys.readouterr().out

    assert "outer" in output
    assert "obj" in output
    assert "eqvio" in output
    assert "ineqvio" in output
    assert "lr" in output
    assert "dual_norm" in output
    assert "penalty" in output
    assert "lag_gap" in output
    assert "iter_time" in output
    assert "inner_err" not in output
    assert "dual_lr" not in output
    assert "total_time" not in output


def test_stiefel_alm_eq_pgd_projects_convex_side_with_cvxpy():
    pytest.importorskip("cvxpy")
    problem = StiefelProblem(
        {
            "A_obj": torch.diag(torch.tensor([3.0, 2.0, 1.0])),
            "n_cols": 1,
            "L": -1.5,
            "U": 1.5,
            "norm_radius": 1.0,
            "group_norm_constraints": [{"rows": [0], "budget": 0.2}],
        }
    )
    solver = StiefelALMEQPGDOptimizer(problem)
    x_init = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    result = solver.solve_result(
        x_init=x_init,
        learning_rate=0.1,
        outer_iterations=12,
        inner_iterations=3,
        convergence_threshold=1e-4,
        penalty_coef=2.0,
        penalty_growth=1.2,
        max_penalty=20.0,
        projection_solver_priority=("CLARABEL", "SCS"),
        seed=0,
    )
    x = torch.tensor(result["solution"], dtype=torch.float32)

    assert result["objective"] is not None
    assert result["extras"]["projection_time_total"] > 0.0
    assert result["extras"]["ineq_violation"] <= 1e-5
    assert problem.constraint_x(x, clip=True, eq_cons=False).max() <= 1e-5
    assert result["extras"]["eq_violation"] < 0.5


def test_stiefel_pyomo_ipopt_solver_solves_small_group_constrained_problem():
    import shutil

    pytest.importorskip("pyomo.environ")
    if shutil.which("ipopt") is None:
        pytest.skip("ipopt executable is not available")

    problem = StiefelProblem(
        {
            "A_obj": torch.diag(torch.tensor([3.0, 2.0, 1.0])),
            "n_cols": 1,
            "L": -1.5,
            "U": 1.5,
            "group_norm_constraints": [{"rows": [0], "budget": 0.2}],
        }
    )
    solver = StiefelPyomoIPOPTSolver(problem)

    result = solver.solve_result(
        x_init=torch.tensor([[0.0, 1.0, 0.0]]),
        solver_options={"tol": 1e-8, "max_iter": 300, "print_level": 0},
        multi_start=2,
    )

    assert result["objective"] is not None
    assert result["extras"]["eq_violation"] < 1e-5
    assert result["extras"]["ineq_violation"] < 1e-5
    assert result["extras"]["multi_start"] == 2
