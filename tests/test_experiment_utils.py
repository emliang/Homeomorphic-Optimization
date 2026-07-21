from pathlib import Path

import numpy as np
import pytest
import torch

from homopt.experiments import (
    ExperimentResult,
    load_result,
    run_and_record,
    save_result,
)
from homopt.experiments.common.reports import (
    convex_solution_diagnostics,
    summarize_single_problem_run_record,
)
from homopt.experiments.common.config import (
    apply_config_groups,
    normalize_algorithm_config_group,
    normalize_alm_algorithm_config,
    normalize_jcc_problem_config,
)
from homopt.experiments.method_specs import (
    base_common_params,
    build_convex_penalty_method_kwargs,
    build_eq_alm_method_params,
    build_eq_cvxpy_baseline_params,
    build_first_order_method_params,
    build_first_order_subproblem_params,
    build_hom_penalty_method_params,
    build_penalty_method_params,
    normalize_jcc_lagrangian_config,
)
from homopt.experiments.common.labels import CONVEX_METHOD_LABELS, STIEFEL_METHOD_LABELS
from homopt.experiments.common.naming import (
    labeled_artifact_prefix,
    labeled_run_name,
    sanitize_label,
    scale_label,
)
import homopt.experiments.script_runtime as script_runtime
import homopt.learning.training.cache as training_cache
from homopt.mappings.base import BaseMap
from homopt.optim.base import BaseOptimizer
from homopt.problems.base import BaseProblem
from homopt.viz.comparison import _outer_per_iter_time_value


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyProblem(BaseProblem):
    nvar = 2

    def objective_x(self, x):
        return x.sum()

    def constraint_x(self, x, clip=True):
        return x


class DummyMap(BaseMap):
    def forward(self, z):
        return z + 1

    def inverse(self, x):
        return x - 1


class DummyOptimizer(BaseOptimizer):
    def optimize(self, initial_point=None, verbose=False, seed=2025):
        return {'x': initial_point, 'seed': seed}


class TinyConvexProblem:
    nvar = 2
    n_eq = 1

    def __init__(self):
        self.Q = torch.eye(2)
        self.A = torch.tensor([[1.0, 0.0]])
        self.b = torch.tensor([1.0])
        self.Qq = torch.eye(2).view(1, 2, 2)
        self.pq = torch.zeros(1, 2)
        self.bq = torch.tensor([0.5])
        self.G = torch.tensor([[[1.0, 0.0]]])
        self.h = torch.zeros(1, 1)
        self.C = torch.tensor([[0.0, 1.0]])
        self.d = torch.tensor([0.5])
        self.L = torch.tensor([0.0, -1.0])
        self.U = torch.tensor([1.0, 2.0])

    def eq_constraint_x(self, x):
        return x[:, :1] + x[:, 1:2] - 1.0


class _CacheData:
    device = torch.device("cpu")
    dtype = torch.float32


class _TinyModel:
    def __init__(self):
        self.to_args = None

    def to(self, **kwargs):
        self.to_args = kwargs
        return self


def test_load_or_train_mapping_keeps_stale_checkpoint_load_errors_fatal(tmp_path):
    checkpoint = tmp_path / "inn_mapping.pt"
    record = tmp_path / "inn_training_record.npy"
    checkpoint.write_bytes(b"stale checkpoint")
    np.save(record, {"old": True})

    def load_fn(path, map_location=None):
        del path, map_location
        raise ModuleNotFoundError("No module named 'homopt.models._inn_impl'", name="homopt.models._inn_impl")

    with pytest.raises(ModuleNotFoundError):
        training_cache.load_or_train_mapping(
            data=_CacheData(),
            args={},
            save_dir=tmp_path,
            retrain=False,
            model_filename="inn_mapping.pt",
            record_filename="inn_training_record.npy",
            load_fn=load_fn,
            save_fn=lambda model, save_dir, filename: Path(save_dir) / filename,
            train_fn=lambda data, args, save_dir: (_TinyModel(), {"new": True}),
        )


def test_load_or_train_mapping_keeps_unknown_module_load_errors_fatal(tmp_path):
    checkpoint = tmp_path / "inn_mapping.pt"
    record = tmp_path / "inn_training_record.npy"
    checkpoint.write_bytes(b"bad checkpoint")
    np.save(record, {"old": True})

    def load_fn(path, map_location=None):
        del path, map_location
        raise ModuleNotFoundError("No module named 'external_package'", name="external_package")

    with pytest.raises(ModuleNotFoundError):
        training_cache.load_or_train_mapping(
            data=_CacheData(),
            args={},
            save_dir=tmp_path,
            retrain=False,
            model_filename="inn_mapping.pt",
            record_filename="inn_training_record.npy",
            load_fn=load_fn,
            save_fn=lambda model, save_dir, filename: Path(save_dir) / filename,
            train_fn=lambda data, args, save_dir: (_TinyModel(), {}),
        )


def test_base_interfaces_can_be_subclassed():
    problem = DummyProblem()
    mapping = DummyMap()
    optimizer = DummyOptimizer()
    assert problem.nvar == 2
    assert mapping.forward(1) == 2
    assert mapping.inverse(2) == 1
    assert optimizer.optimize(initial_point=3)['x'] == 3


def test_experiment_scale_label_naming_helpers_are_canonical():
    params = {"scale_label": "SOCP EQ n=10 / q=2"}

    assert sanitize_label(params["scale_label"]) == "socp_eq_n_10_q_2"
    assert scale_label(params) == "socp_eq_n_10_q_2"
    assert labeled_run_name("convex_eq_compare", params) == "convex_eq_compare_socp_eq_n_10_q_2"
    assert labeled_artifact_prefix("socp_eq", params) == "socp_eq_n_10_q_2"
    assert labeled_artifact_prefix("stiefel", {"scale_label": "small_digits"}) == "stiefel_small_digits"
    assert labeled_artifact_prefix("socp", params, explicit_prefix="paper_socp") == "paper_socp"


def test_experiment_method_labels_distinguish_proximal_algorithms():
    assert CONVEX_METHOD_LABELS["ALM"] == "ALM"
    assert CONVEX_METHOD_LABELS["Prox-ALM"] == "PALM"
    assert CONVEX_METHOD_LABELS["Hom-ALM"] == "Hom-ALM"
    assert CONVEX_METHOD_LABELS["Prox-Hom-ALM"] == "Hom-PALM"
    assert CONVEX_METHOD_LABELS["Prox-ALM-EQ"] == "PALM-Eq"
    assert STIEFEL_METHOD_LABELS["StiefelRetractionALM"] == "Retraction-ALM"


def test_script_merge_params_falls_back_to_cpu_without_cuda(monkeypatch, capsys):
    script_runtime._CUDA_FALLBACK_WARNED.clear()
    monkeypatch.setattr(script_runtime.torch.cuda, "is_available", lambda: False)

    params = script_runtime.merge_params(
        {"device": "cuda:1", "nested": {"device": "auto"}},
        {"other": 1},
    )

    assert params["device"] == "cpu"
    assert params["nested"]["device"] == "cpu"
    out = capsys.readouterr().out
    assert "CUDA is unavailable" in out
    assert "using device='cpu'" in out


def test_outer_per_iter_runtime_rejects_nested_record_without_outer_trace():
    record = {
        "runtime_total": 10.0,
        "iter_time": [0.01, 0.01, 0.01],
        "extras": {
            "outer_iterations": 5,
            "inner_iterations": 100,
        },
    }

    with pytest.raises(ValueError, match="requires explicit outer_iter_time"):
        _outer_per_iter_time_value(record)


def test_single_problem_summary_uses_x_solved_objective_and_violation():
    problem = DummyProblem()
    summary = summarize_single_problem_run_record(
        problem,
        {
            "obj_traj": [999.0],
            "cons_traj": [123.0],
            "iter_time": [0.01],
            "outer_iter_time": [0.012],
            "inner_iter_time": [0.008],
            "solver_iter_time": [0.006],
            "total_wall_time": 0.02,
            "last_trans_time": 0.003,
            "initial_transform_time": 0.001,
            "final_transform_time": 0.002,
            "initial_ip_time": 0.004,
            "x_solved": torch.tensor([[2.0, 3.0]]),
        },
        reference_objective=1.0,
        include_total_iter_time=True,
    )
    assert summary["optimizer_reported_final_objective"] == pytest.approx(999.0)
    assert summary["final_objective"] == pytest.approx(5.0)
    assert summary["objective_gap"] == pytest.approx(4.0)
    assert summary["optimizer_reported_final_violation"] == pytest.approx(123.0)
    assert summary["final_violation"] == pytest.approx(3.0)
    assert summary["total_iter_time"] == pytest.approx(0.01)
    assert summary["total_wall_time"] == pytest.approx(0.02)
    assert summary["total_outer_iter_time"] == pytest.approx(0.012)
    assert summary["mean_inner_iter_time"] == pytest.approx(0.008)
    assert summary["mean_solver_iter_time"] == pytest.approx(0.006)
    assert summary["last_trans_time"] == pytest.approx(0.003)
    assert summary["initial_transform_time"] == pytest.approx(0.001)
    assert summary["final_transform_time"] == pytest.approx(0.002)
    assert summary["initial_ip_time"] == pytest.approx(0.004)


def test_convex_solution_diagnostics_reports_active_constraint_types():
    diagnostics = convex_solution_diagnostics(TinyConvexProblem(), torch.tensor([1.0, 0.0]), active_tol=1e-6)

    assert diagnostics["available"] is True
    assert diagnostics["counts_by_type"]["linear"]["active"] == 1
    assert diagnostics["counts_by_type"]["linear"]["violated"] == 0
    assert diagnostics["counts_by_type"]["quadratic"]["active"] == 1
    assert diagnostics["counts_by_type"]["soc"]["active"] == 1
    assert diagnostics["counts_by_type"]["soc"]["violated"] == 1
    assert diagnostics["counts_by_type"]["box_upper"]["active"] == 1
    assert diagnostics["counts_by_type"]["box_lower"]["active"] == 0
    assert diagnostics["counts_by_type"]["equality"]["violated"] == 0
    assert diagnostics["active_inequality_total"] == 4


def test_experiment_result_roundtrip(tmp_path):
    result = ExperimentResult(
        name='demo',
        objective=1.25,
        feasible=True,
        metrics={'iters': 3},
    )
    save_result(result, tmp_path, config={'seed': 7})
    loaded = load_result(tmp_path)
    assert loaded.name == 'demo'
    assert loaded.objective == 1.25
    assert loaded.feasible is True
    assert loaded.metrics['iters'] == 3
    assert (tmp_path / 'config.json').exists()


def test_run_and_record_with_dict_payload(tmp_path):
    out_dir = tmp_path / 'run1'

    def run_fn():
        return {'objective': 2.0, 'feasible': False, 'metrics': {'iters': 10}}

    result = run_and_record('trial', run_fn, out_dir, config={'lr': 1e-3})
    assert result.name == 'trial'
    assert result.objective == 2.0
    assert result.feasible is False
    assert result.metrics['iters'] == 10
    assert 'elapsed_sec' in result.metrics
    assert (out_dir / 'result.json').exists()


def test_run_and_record_rejects_top_level_metric_fields(tmp_path):
    def run_fn():
        return {'objective': 2.0, 'feasible': False, 'iters': 10}

    with pytest.raises(ValueError, match="unknown top-level fields"):
        run_and_record('trial', run_fn, tmp_path / 'run_invalid')


def test_run_and_record_rejects_legacy_top_level_metric_escape(tmp_path):
    def run_fn():
        return {'objective': 2.0, 'feasible': False, '_allow_top_level_metrics': True, 'iters': 10}

    with pytest.raises(ValueError, match="unknown top-level fields"):
        run_and_record('trial', run_fn, tmp_path / 'run_invalid')


def test_run_and_record_merges_nested_metrics_and_context(tmp_path):
    out_dir = tmp_path / 'run_nested_metrics'

    def run_fn():
        return {
            'objective': 3.0,
            'feasible': True,
            'metrics': {'iters': 4, 'extra': 9},
        }

    result = run_and_record(
        'trial',
        run_fn,
        out_dir,
        config={'lr': 1e-3},
        context={'entrypoint': 'unit_test'},
    )
    assert result.metrics['iters'] == 4
    assert result.metrics['extra'] == 9
    assert result.context['entrypoint'] == 'unit_test'


def test_run_and_record_preserves_artifacts_from_dict_payload(tmp_path):
    out_dir = tmp_path / 'run_with_artifacts'

    def run_fn():
        return {
            'objective': 1.0,
            'feasible': True,
            'artifacts': {'trace': 'artifacts/trace.npy'},
            'metrics': {'iters': 2},
        }

    result = run_and_record('trial', run_fn, out_dir)
    assert result.artifacts['trace'] == 'artifacts/trace.npy'
    assert result.metrics['iters'] == 2


def test_run_and_record_with_result_payload(tmp_path):
    out_dir = tmp_path / 'run2'

    def run_fn():
        return ExperimentResult(name='custom', objective=0.5, feasible=True)

    result = run_and_record('ignored', run_fn, out_dir)
    assert result.name == 'custom'
    assert result.objective == 0.5
    assert result.feasible is True
    assert Path(out_dir / 'result.json').exists()


def test_penalty_method_helpers_share_canonical_outer_penalty_defaults():
    common = base_common_params(
        seed=0,
        max_iterations=5,
        max_running_time=10,
        learning_rate=1e-3,
        stepsize_rule="adaptive",
        lr_decay=0.99,
        min_lr=1e-6,
    )
    alm = build_penalty_method_params(common, outer_iterations=3, inner_iterations=2)
    penalty = build_penalty_method_params(
        common,
        outer_iterations=3,
        inner_iterations=2,
        use_lagrangian=False,
        use_penalty=True,
    )
    prox_penalty = build_penalty_method_params(
        common,
        outer_iterations=3,
        inner_iterations=2,
        use_lagrangian=False,
        use_penalty=True,
        use_proximal=True,
    )
    convex_hom_kwargs = build_convex_penalty_method_kwargs(
        common,
        outer_iterations=3,
        dual_learning_rate=5e-3,
        proximal_space="z",
    )
    alm_eq = build_eq_alm_method_params(common, outer_iterations=3)
    prox_alm_eq = build_eq_alm_method_params(common, outer_iterations=3, use_proximal=True)
    prox_penalty_eq = build_eq_alm_method_params(
        common,
        outer_iterations=3,
        use_lagrangian=False,
        use_proximal=True,
    )
    hom_alm = build_hom_penalty_method_params(common, outer_iterations=3, inner_iterations=2)
    jcc = normalize_jcc_lagrangian_config(
        max_iterations=3,
        lagrangian_baseline_config={
            "inner_iterations": 2,
            "learning_rate": 1e-3,
            "inner_learning_rate": 1e-3,
            "dual_learning_rate": 1e-1,
            "lr_decay": 0.99,
            "inner_lr_decay": 0.99,
            "min_lr": 1e-6,
            "inner_min_lr": 1e-6,
            "penalty_coef": 10.0,
            "penalty_growth": 1.1,
            "proximal_coef": 0.1,
            "max_penalty": 1e2,
            "max_dual": 1e2,
            "max_running_time": 10,
            "convergence_threshold": 1e-5,
            "stepsize_rule": "adaptive",
            "inner_stepsize_rule": "constant",
            "inner_solver": "gd",
            "lagrangian_gradient": "autograd",
            "proximal_space": "x",
            "opt": "gd",
            "momentum": 0.0,
            "verbose": False,
            "verbose_interval": 50,
        },
    )

    for payload in (alm, penalty, prox_penalty, alm_eq, prox_alm_eq, prox_penalty_eq, hom_alm, jcc):
        assert payload["outer_iterations"] == 3
        assert payload["dual_learning_rate"] == 1e-1
        assert payload["penalty_coef"] == 10.0
        assert payload["penalty_growth"] == 1.1
        assert payload["proximal_coef"] == 0.1
        assert payload["max_penalty"] == 1e2
        assert payload["max_dual"] == 1e2

    assert alm["inner_iterations"] >= 2
    assert convex_hom_kwargs["inner_iterations"] == 50
    assert convex_hom_kwargs["inner_solver"] == "gd"
    assert convex_hom_kwargs["inner_proximal_update_iterations"] == 1
    assert convex_hom_kwargs["inner_proximal_coef"] == 0.0

    assert convex_hom_kwargs["inner_proximal_space"] == "z"
    assert convex_hom_kwargs["acceleration_space"] == "z"
    assert convex_hom_kwargs["outer_stepsize_rule"] == "adaptive"
    assert convex_hom_kwargs["inner_stepsize_rule"] == "constant"
    assert convex_hom_kwargs["dual_learning_rate"] == pytest.approx(5e-3)
    assert hom_alm["inner_iterations"] >= 2
    assert jcc["inner_iterations"] >= 2
    assert alm["proximal_space"] == "x"
    assert alm["lagrangian_gradient"] == "explicit"
    assert penalty["use_lagrangian"] is False
    assert penalty["use_penalty"] is True
    assert penalty["use_proximal"] is False
    assert prox_penalty["use_lagrangian"] is False
    assert prox_penalty["use_penalty"] is True
    assert prox_penalty["use_proximal"] is True
    assert alm_eq["opt_type"] == "ALM-EQ"
    assert prox_alm_eq["use_lagrangian"] is True
    assert prox_alm_eq["use_proximal"] is True
    assert prox_penalty_eq["use_lagrangian"] is False
    assert prox_penalty_eq["use_proximal"] is True
    for inactive_key in (
        "inner_solver",
        "inner_iterations",
        "inner_proximal_update_iterations",
        "inner_proximal_coef",
        "inner_proximal_space",
        "acceleration_method",
        "acceleration_space",
        "outer_stepsize_rule",
        "inner_stepsize_rule",
        "proximal_space",
    ):
        assert inactive_key not in alm_eq
    assert alm["inner_solver"] == "gd"
    assert alm["inner_proximal_update_iterations"] == 1
    assert alm["inner_proximal_coef"] == 0.0
    assert alm["inner_proximal_space"] == "x"
    assert alm["acceleration_method"] == "none"
    assert alm["acceleration_space"] == "x"
    assert hom_alm["proximal_space"] == "z"
    assert hom_alm["lagrangian_gradient"] == "explicit"
    assert hom_alm["hom_map_gradient"] == "explicit"
    assert jcc["proximal_space"] == "x"
    assert alm["outer_stepsize_rule"] == "adaptive"
    assert hom_alm["outer_stepsize_rule"] == "adaptive"
    assert jcc["outer_stepsize_rule"] == "adaptive"
    assert alm["outer_lr_decay"] == pytest.approx(0.99)
    assert hom_alm["outer_lr_decay"] == pytest.approx(0.99)
    assert jcc["outer_lr_decay"] == pytest.approx(0.99)
    assert alm["inner_stepsize_rule"] == "constant"
    assert hom_alm["inner_stepsize_rule"] == "constant"
    assert jcc["inner_stepsize_rule"] == "constant"
    assert hom_alm["inner_solver"] == "gd"
    assert hom_alm["inner_proximal_update_iterations"] == 1
    assert hom_alm["inner_proximal_coef"] == 0.0
    assert hom_alm["inner_proximal_space"] == "z"
    assert hom_alm["acceleration_method"] == "none"
    assert hom_alm["acceleration_space"] == "z"
    assert alm["opt_type"] == "ALM"
    assert hom_alm["opt_type"] == "Hom-ALM"

    prox_hom_alm = build_hom_penalty_method_params(
        common,
        outer_iterations=3,
        inner_iterations=6,
        inner_solver="prox_gd",
        inner_proximal_update_iterations=2,
        inner_proximal_coef=0.5,
        inner_proximal_space="x",
        acceleration_method="nag",
        acceleration_space="x",
    )
    assert prox_hom_alm["inner_solver"] == "prox_gd"
    assert prox_hom_alm["inner_proximal_update_iterations"] == 2
    assert prox_hom_alm["inner_proximal_coef"] == 0.5
    assert prox_hom_alm["inner_proximal_space"] == "x"
    assert prox_hom_alm["acceleration_method"] == "nag"
    assert prox_hom_alm["acceleration_space"] == "x"

    eq_family = build_eq_cvxpy_baseline_params(common, outer_iterations=3)
    assert list(eq_family) == ["Penalty-EQ", "Prox-Penalty-EQ", "ALM-EQ", "Prox-ALM-EQ"]
    assert eq_family["Penalty-EQ"]["use_lagrangian"] is False
    assert eq_family["Penalty-EQ"]["use_penalty"] is True
    assert eq_family["Penalty-EQ"]["use_proximal"] is False
    assert eq_family["Prox-Penalty-EQ"]["use_lagrangian"] is False
    assert eq_family["Prox-Penalty-EQ"]["use_proximal"] is True
    assert eq_family["ALM-EQ"]["use_lagrangian"] is True
    assert eq_family["ALM-EQ"]["use_proximal"] is False
    assert eq_family["Prox-ALM-EQ"]["use_lagrangian"] is True
    assert eq_family["Prox-ALM-EQ"]["use_proximal"] is True
    for payload in eq_family.values():
        assert payload["outer_iterations"] == 3
        assert payload["penalty_coef"] == 10.0
        assert payload["penalty_growth"] == 1.1
        for inactive_key in (
            "inner_solver",
            "inner_iterations",
            "inner_proximal_update_iterations",
            "inner_proximal_coef",
            "inner_proximal_space",
            "acceleration_method",
            "acceleration_space",
            "outer_stepsize_rule",
            "inner_stepsize_rule",
            "proximal_space",
        ):
            assert inactive_key not in payload


def test_jcc_problem_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unsupported JCC problem_config keys"):
        normalize_jcc_problem_config(
            n_scenarios=2,
            epsilon=0.1,
            demand_std=0.05,
            seed=7,
            problem_config={"unused": 1},
        )


def test_jcc_problem_config_preserves_pglib_loader_controls(tmp_path):
    config = normalize_jcc_problem_config(
        n_scenarios=2,
        epsilon=0.1,
        demand_std=0.05,
        seed=7,
        problem_config={
            "pglib_case_name": "custom_case",
            "pglib_data_dir": tmp_path,
            "download_if_missing": False,
        },
    )

    assert config["pglib_case_name"] == "custom_case"
    assert config["pglib_data_dir"] == str(tmp_path)
    assert config["download_if_missing"] is False


def test_exact_equality_baselines_reject_unsupported_algorithm_controls():
    for config in (
        {"learning_rate": 1e-3},
        {"first_order_lagrangian_gap_threshold": 1e-6},
    ):
        with pytest.raises(ValueError, match="Unsupported ALM-EQ config keys"):
            normalize_alm_algorithm_config("ALM-EQ", config)

    assert normalize_alm_algorithm_config(
        "ALM-EQ",
        {"stepsize_rule": "adaptive", "lr_decay": 0.99},
    ) == {}

    with pytest.raises(ValueError, match="Unsupported ALM config keys"):
        normalize_alm_algorithm_config("ALM", {"solver_verbose": True})


def test_first_order_helpers_expose_embedded_subproblem_configs():
    common = base_common_params(
        seed=0,
        max_iterations=5,
        max_running_time=10,
        learning_rate=1e-3,
        stepsize_rule="adaptive",
        lr_decay=0.99,
        min_lr=1e-6,
    )
    params = build_first_order_method_params(
        common,
        projection_outer_iterations=3,
        projection_inner_iterations=4,
        linearization_outer_iterations=5,
        linearization_inner_iterations=6,
    )

    proj_cfg = params["PGD"]["projection_subproblem"]
    loo_cfg = params["FW"]["linearization_subproblem"]
    direct_proj_cfg = build_first_order_subproblem_params(
        common,
        outer_iterations=3,
        inner_iterations=4,
    )
    assert params["PGD"]["projection_outer_iterations"] == 3
    assert params["PGD"]["projection_inner_iterations"] == 4
    assert params["PGD"]["acceleration_space"] == "x"
    assert params["Hom-PGD"]["hom_map_gradient"] == "explicit"
    assert params["Hom-PGD"]["acceleration_space"] == "z"
    assert params["Hom-PGD"]["use_proximal"] is False
    assert params["Hom-PGD"]["proximal_update_iterations"] == 1
    assert params["Hom-PGD"]["proximal_coef"] == 0.0
    assert params["Hom-PGD"]["proximal_space"] == "z"
    assert proj_cfg["outer_iterations"] == 3
    assert proj_cfg["inner_iterations"] == 4
    assert proj_cfg == direct_proj_cfg
    assert proj_cfg["learning_rate"] == 1e-3
    assert proj_cfg["dual_learning_rate"] == 1e-2
    assert proj_cfg["inner_solver"] == "gd"
    assert proj_cfg["inner_proximal_update_iterations"] == 1
    assert proj_cfg["inner_proximal_coef"] == 0.0
    assert proj_cfg["inner_proximal_space"] == "x"
    assert proj_cfg["acceleration_method"] == "none"
    assert proj_cfg["acceleration_space"] == "x"
    assert proj_cfg["outer_stepsize_rule"] == "adaptive"
    assert proj_cfg["inner_stepsize_rule"] == "constant"
    assert loo_cfg["outer_iterations"] == 5
    assert loo_cfg["inner_iterations"] == 6
    assert loo_cfg["learning_rate"] == 1e-3
    assert loo_cfg["dual_learning_rate"] == 1e-2
    assert loo_cfg["inner_solver"] == "gd"
    assert loo_cfg["inner_proximal_space"] == "x"
    assert loo_cfg["acceleration_method"] == "none"
    assert loo_cfg["acceleration_space"] == "x"
    assert params["FW"]["linearization_outer_iterations"] == 5
    assert params["FW"]["linearization_inner_iterations"] == 6
    assert params["FW"]["acceleration_space"] == "x"
    assert params["RD"]["acceleration_space"] == "x"
    assert "proj_outer_iter" not in params["PGD"]
    assert "loo_subproblem" not in params["FW"]


def test_first_order_public_subproblem_iteration_overrides_sync_embedded_configs():
    normalized = normalize_algorithm_config_group(
        algorithm_config={
            "PGD": {
                "projection_max_iterations": 7,
                "projection_inner_iterations": 8,
            },
            "FW": {
                "linearization_max_iterations": 9,
                "linearization_inner_iterations": 10,
            },
        }
    )

    assert normalized["PGD"]["projection_outer_iterations"] == 7
    assert normalized["PGD"]["projection_subproblem"]["outer_iterations"] == 7
    assert normalized["PGD"]["projection_subproblem"]["inner_iterations"] == 8
    assert "projection_max_iterations" not in normalized["PGD"]
    assert normalized["FW"]["linearization_outer_iterations"] == 9
    assert normalized["FW"]["linearization_subproblem"]["outer_iterations"] == 9
    assert normalized["FW"]["linearization_subproblem"]["inner_iterations"] == 10
    assert "linearization_max_iterations" not in normalized["FW"]


def test_first_order_subproblem_overrides_preserve_canonical_defaults():
    common = base_common_params(
        seed=0,
        max_iterations=5,
        max_running_time=17,
        learning_rate=1e-3,
        stepsize_rule="adaptive",
        lr_decay=0.99,
        min_lr=1e-6,
    )
    params = {
        "common": common,
        **build_first_order_method_params(
            common,
            projection_outer_iterations=3,
            projection_inner_iterations=4,
            linearization_outer_iterations=3,
            linearization_inner_iterations=4,
        ),
    }
    updated = apply_config_groups(
        params,
        algorithm_config={
            "PGD": {"projection_max_iterations": 7, "projection_inner_iterations": 8},
            "FW": {"linearization_max_iterations": 9, "linearization_inner_iterations": 10},
        },
    )

    assert updated["PGD"]["projection_subproblem"]["outer_iterations"] == 7
    assert updated["PGD"]["projection_subproblem"]["inner_iterations"] == 8
    assert updated["PGD"]["projection_subproblem"]["max_running_time"] == 17
    assert updated["FW"]["linearization_subproblem"]["outer_iterations"] == 9
    assert updated["FW"]["linearization_subproblem"]["inner_iterations"] == 10
    assert updated["FW"]["linearization_subproblem"]["max_running_time"] == 17


def test_first_order_public_subproblem_iteration_overrides_reject_old_outer_names():
    with pytest.raises(ValueError, match="projection_outer_iterations"):
        normalize_algorithm_config_group(
            algorithm_config={"PGD": {"projection_outer_iterations": 7}}
        )
    with pytest.raises(ValueError, match="linearization_outer_iterations"):
        normalize_algorithm_config_group(
            algorithm_config={"FW": {"linearization_outer_iterations": 7}}
        )
