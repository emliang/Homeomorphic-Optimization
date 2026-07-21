import json
import inspect
import importlib.util
from types import SimpleNamespace
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from _common import merge_params  # noqa: E402
from inn_pgd._configs import qcqp_inn_pgd_config  # noqa: E402
from homopt.experiments import examples, hom_alm, hom_pgd, inn_pgd, learning_postprocess
import homopt.records.artifacts as artifact_helpers
import homopt.experiments.common.convex_reference as convex_reference_common
import homopt.experiments.common.config as config_common
import homopt.experiments.common.inn_baselines as inn_baselines
import homopt.experiments.common.run_records as benchmark_record_common
import homopt.experiments.benchmarks.jcc.compare as jcc_opf_benchmarks
import homopt.experiments.benchmarks.jcc.methods as jcc_methods
import homopt.experiments.benchmarks.jcc.setup as jcc_setup
import homopt.experiments.benchmarks.jcc.linear as jcc_parametric_benchmarks
import homopt.experiments.benchmarks.parametric_registry as parametric_registry
import homopt.experiments.benchmarks.qcqp.training as qcqp_inn_common
import homopt.experiments.benchmarks.qcqp.inn_pgd as qcqp_inn_comparison_benchmarks
import homopt.experiments.benchmarks.qcqp.sweep as qcqp_inn_sweep_benchmarks
import homopt.experiments.benchmarks.convex_parametric as convex_parametric_learning_benchmarks
import homopt.experiments.benchmarks.convex as convex_ineq_benchmarks
import homopt.experiments.benchmarks.maxcut as maxcut_benchmarks
import homopt.experiments.benchmarks.star as toy_star_benchmarks
from homopt.experiments.common.labels import CONVEX_METHOD_LABELS
from homopt.experiments.runner import run_and_record
from homopt.learning.training import build_learning_mapping_payload, build_predictor_training_payload
from homopt.viz.comparison import build_comparison_traces


class _FakeINNMapping(torch.nn.Module):
    def forward(self, z, c=None):
        del c
        return z


def _qcqp_inn_params(**overrides):
    return qcqp_inn_pgd_config(overrides)


def _kwargs_for(function, params):
    signature = inspect.signature(function)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(params)
    return {key: value for key, value in params.items() if key in signature.parameters}


def _inn_training_params(**overrides):
    return _kwargs_for(inn_pgd.inn_training_benchmark, _qcqp_inn_params(**overrides))


def _qcqp_inn_comparison_params(**overrides):
    return _kwargs_for(qcqp_inn_comparison_benchmarks.qcqp_inn_comparison, _qcqp_inn_params(**overrides))



def _run_workload(name, workload, kwargs, tmp_path):
    output_dir = tmp_path / "benchmarks" / name
    signature = inspect.signature(workload)
    call_kwargs = dict(kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()) or "output_dir" in signature.parameters:
        call_kwargs["output_dir"] = output_dir
    result = run_and_record(
        name=name,
        run_fn=lambda: workload(**call_kwargs),
        output_dir=output_dir,
        config={"entrypoint": "test_workload", "kwargs": kwargs},
        context={"entrypoint": "test_workload"},
    )
    return result, output_dir


def test_learning_mapping_payload_uses_shared_inn_ipnn_schema():
    problem_args = {"n_var": 2, "n_qua_cons": 1}
    train_args = {"n_samples": 8, "batch_size": 2, "total_iteration": 3, "pre_training": 1}

    inn_payload = build_learning_mapping_payload(
        "inn",
        problem_args,
        {"h_dim": 8},
        train_args,
        runtime_device="cpu",
        runtime_dtype=torch.float64,
        ensure_results_save_freq=True,
    )
    ipnn_payload = build_learning_mapping_payload(
        "ipnn",
        problem_args,
        {"h_dim": 8},
        train_args,
        runtime_device="cpu",
        runtime_dtype=torch.float64,
        ensure_results_save_freq=True,
    )

    assert "model_config" in inn_payload
    assert "ipnn_model_config" in ipnn_payload
    assert inn_payload["model_config"]["device"] == "cpu"
    assert ipnn_payload["ipnn_model_config"]["device"] == "cpu"
    assert inn_payload["model_config"]["dtype"] is torch.float64
    assert ipnn_payload["ipnn_model_config"]["dtype"] is torch.float64
    assert inn_payload["model_config"]["pre_training"] == 1
    assert ipnn_payload["ipnn_model_config"]["pre_training"] == 1
    assert "resultsSaveFreq" in inn_payload["model_config"]
    assert "resultsSaveFreq" in ipnn_payload["ipnn_model_config"]


def test_predictor_training_payload_uses_shared_train_loop_schema():
    problem_args = {"n_var": 2, "n_ineq": 3}
    model_args = {"h_dim": 8, "approach": "supervise", "supervise_target": "dataset"}
    train_args = {
        "n_samples": 8,
        "batch_size": 2,
        "total_iteration": 3,
        "pre_training": 1,
        "resultsSaveFreq": 2,
    }

    payload = build_predictor_training_payload(problem_args, model_args, train_args)

    model_payload = payload["predictor_model_config"]
    assert payload["problem_config"] == problem_args
    assert model_payload["h_dim"] == 8
    assert model_payload["n_samples"] == 8
    assert model_payload["batch_size"] == 2
    assert model_payload["total_iteration"] == 3
    assert model_payload["pre_training"] == 1
    assert model_payload["resultsSaveFreq"] == 2


def _payload_metrics(payload):
    return payload["metrics"]


def _qcqp_inn_case_and_instances(output_dir, **params):
    params = _qcqp_inn_params(**params)
    case_context = qcqp_inn_common._build_qcqp_inn_case_context(output_dir, params)
    instance_batch = qcqp_inn_common._sample_qcqp_inn_test_instances(case_context)
    return case_context, instance_batch


def test_poly_star_benchmark_writes_artifacts(tmp_path):
    result, output_dir = _run_workload('poly_star_smoke', hom_pgd.poly_star_benchmark, {'algorithms': ['Hom-PGD'], 'max_iterations': 4, 'max_running_time': 5, 'reference_solver': 'grid', 'reference_grid_size': 51}, tmp_path)
    assert result.name == "poly_star_smoke"
    assert result.artifacts["records"] == "artifacts/records"
    assert (output_dir / result.artifacts["records"]).exists()
    assert (output_dir / result.artifacts["summary"]).exists()
    assert (output_dir / result.artifacts["manifest"]).exists()
    assert result.metrics["reference_objective"] is not None
    assert result.metrics["reference_label"] in {"GridSearch", "MOSEK"}


def test_poly_star_benchmark_supports_explicit_poly_and_star_sets():
    for problem_type, expected_problem in [
        ("poly", "toy_poly"),
        ("star", "toy_star"),
        ("poly_star", "toy_poly_star"),
    ]:
        payload = hom_pgd.poly_star_benchmark(
            problem_type=problem_type,
            algorithms=["Hom-PGD"],
            max_iterations=2,
            max_running_time=5,
            device="cpu",
            dtype="float32",
            reference_solver="grid",
            reference_grid_size=51,
            output_dir=None,
        )
        metrics = _payload_metrics(payload)
        assert metrics["problem"] == expected_problem
        assert metrics["nvar"] == 2
        assert metrics["algorithms"] == ["Hom-PGD"]
        assert metrics["reference_objective"] is not None


def test_poly_star_toy_context_reuses_poly_problem_and_gauge_center():
    problem, hom_map = toy_star_benchmarks.build_poly_star_toy_context(
        problem_type="poly",
        poly_config={
            "obj": "quad",
            "n_var": 2,
            "n_linear_cons": 5,
            "n_soc_cons": 0,
            "n_qua_cons": 0,
            "x_lower": -2.0,
            "x_upper": 2.0,
            "margin_scale": 0.75,
            "anchor_interior_ratio": 0.15,
        },
        seed=2030,
        device="cpu",
        dtype="float32",
    )
    z = torch.tensor([[0.2, -0.3]], dtype=torch.float32)
    x = hom_map.forward(z)
    z_round_trip = hom_map.inverse(x)

    assert problem.nvar == 2
    assert hom_map.center.shape == (1, 2)
    assert torch.allclose(z_round_trip, z, atol=1e-5)


def test_poly_star_benchmark_uses_one_shared_gauge_center_start(monkeypatch):
    starts = []

    def capture_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del name, params, hom_map
        starts.append(np.asarray(init_point, dtype=float).reshape(-1))
        return {
            "objective": 0.0,
            "violation": 0.0,
            "obj_traj": [0.0],
            "cons_traj": [0.0],
            "eq_violation_traj": [0.0],
            "ineq_violation_traj": [0.0],
            "iter_time": [0.0],
            "x_traj": [np.asarray(init_point, dtype=float).reshape(-1)],
            "x_solved": np.asarray(init_point, dtype=float).reshape(1, -1),
        }

    monkeypatch.setattr(toy_star_benchmarks, "run_algorithm", capture_run_algorithm)
    toy_star_benchmarks.poly_star_benchmark(
        problem_type="star",
        algorithms=["PGD", "RD", "Hom-PGD"],
        max_iterations=1,
        max_running_time=5,
        device="cpu",
        dtype="float32",
        reference_solver="none",
        output_dir=None,
    )

    assert len(starts) == 3
    assert all(np.allclose(start, starts[0]) for start in starts[1:])
    assert np.allclose(starts[0], np.zeros(2))


def test_poly_star_benchmark_can_return_the_exact_run_context():
    payload, context = toy_star_benchmarks.poly_star_benchmark(
        problem_type="star",
        algorithms=["Hom-PGD"],
        max_iterations=2,
        max_running_time=5,
        device="cpu",
        dtype="float32",
        reference_solver="none",
        output_dir=None,
        return_context=True,
    )

    assert payload["metrics"]["algorithms"] == ["Hom-PGD"]
    assert context["problem"].nvar == 2
    assert context["hom_map"].center.shape == (1, 2)
    assert set(context["records"]) == {"Hom-PGD"}


def test_poly_star_auto_reference_uses_ipopt_for_star(monkeypatch):
    captured = {}

    def fake_ipopt_reference(problem, *, solver, options, num_starts, reference_label):
        captured["problem"] = problem
        captured["solver"] = solver
        captured["options"] = options
        captured["num_starts"] = num_starts
        captured["reference_label"] = reference_label
        return {
            "label": reference_label or "IPOPT",
            "objective": 0.25,
            "solution": np.array([0.5, 0.5]),
            "violation": 0.0,
            "status": "optimal",
            "runtime": 0.1,
        }

    monkeypatch.setattr(toy_star_benchmarks, "_toy_ipopt_reference", fake_ipopt_reference)
    payload = hom_pgd.poly_star_benchmark(
        problem_type="star",
        algorithms=["Hom-PGD"],
        max_iterations=1,
        max_running_time=5,
        device="cpu",
        dtype="float32",
        reference_solver="auto",
        reference_ipopt_solver="mock-ipopt",
        reference_ipopt_options={"tol": 1e-7},
        reference_ipopt_num_starts=3,
        output_dir=None,
    )
    metrics = _payload_metrics(payload)
    assert metrics["reference_label"] == "IPOPT"
    assert metrics["reference_objective"] == pytest.approx(0.25)
    assert captured["solver"] == "mock-ipopt"
    assert captured["options"] == {"tol": 1e-7}
    assert captured["num_starts"] == 3


def test_inn_training_benchmark_writes_model_artifacts(tmp_path):
    result, output_dir = _run_workload('inn_training_smoke', inn_pgd.inn_training_benchmark, _inn_training_params(n_samples=8, batch_size=4, total_iteration=2, max_iterations=3), tmp_path)
    assert "training_record" in result.artifacts
    assert (output_dir / result.artifacts["training_record"]).exists()
    assert (output_dir / result.artifacts["model"]).exists()
    assert "training_penalty_final" in result.metrics


def test_hom_alm_equality_scope_is_not_plotted_as_inequality():
    records = {
        "Prox-Hom-ALM": {
            "obj_traj": np.array([1.0, 0.5, 0.25]),
            "cons_traj": np.array([1.0, 0.1, 1e-3]),
            "iter_time": [0.1, 0.2],
            "violation_scope": "equality",
        }
    }
    traces = build_comparison_traces(None, records, ["Prox-Hom-ALM"])
    trace = traces["Prox-Hom-ALM"]

    assert np.allclose(trace["equality_violation"], [1.0, 0.1, 1e-3])
    assert np.allclose(trace["inequality_violation"], [0.0, 0.0, 0.0])
    assert np.allclose(trace["full_violation"], [1.0, 0.1, 1e-3])


def test_record_violation_split_helper_respects_explicit_constraint_scope():
    pure_ineq = SimpleNamespace(ncon=3, n_eq=0)
    ineq_record = {"cons_traj": np.array([0.5, 0.1])}
    benchmark_record_common.ensure_record_violation_split(pure_ineq, ineq_record)
    assert np.allclose(ineq_record["ineq_violation_traj"], [0.5, 0.1])
    assert np.allclose(ineq_record["eq_violation_traj"], [0.0, 0.0])
    assert ineq_record["violation_scope"] == "inequality"

    pure_eq = SimpleNamespace(ncon=2, eq_cons=range(2), ineq_cons=None)
    eq_record = {"cons_traj": np.array([0.4, 0.05])}
    benchmark_record_common.ensure_record_violation_split(pure_eq, eq_record)
    assert np.allclose(eq_record["eq_violation_traj"], [0.4, 0.05])
    assert np.allclose(eq_record["ineq_violation_traj"], [0.0, 0.0])
    assert eq_record["violation_scope"] == "equality"

    mixed = SimpleNamespace(ncon=3, n_eq=1, eq_cons=range(2, 3), ineq_cons=range(2))
    mixed_record = {"cons_traj": np.array([0.3, 0.02])}
    benchmark_record_common.ensure_record_violation_split(mixed, mixed_record)
    assert "eq_violation_traj" not in mixed_record
    assert "ineq_violation_traj" not in mixed_record


def test_comparison_traces_require_explicit_split_violation_for_scalar_records():
    records = {
        "PGD": {
            "obj_traj": np.array([1.0, 0.5, 0.25]),
            "cons_traj": np.array([1.0, 0.1, 1e-3]),
            "iter_time": [0.1, 0.2],
        }
    }

    with pytest.raises(ValueError, match="split violation records"):
        build_comparison_traces(None, records, ["PGD"])


def test_comparison_traces_do_not_infer_hom_alm_scope_from_name():
    records = {
        "Hom-ALM": {
            "obj_traj": np.array([1.0, 0.5]),
            "cons_traj": np.array([0.1, 0.01]),
            "iter_time": [0.1],
        }
    }
    with pytest.raises(ValueError, match="split violation records"):
        build_comparison_traces(None, records, ["Hom-ALM"])


def test_comparison_traces_require_aligned_timing_records():
    records = {
        "PGD": {
            "obj_traj": np.array([1.0, 0.5, 0.25]),
            "eq_violation_traj": np.array([0.0, 0.0, 0.0]),
            "ineq_violation_traj": np.array([1.0, 0.1, 1e-3]),
            "cons_traj": np.array([1.0, 0.1, 1e-3]),
            "iter_time": [0.1],
        }
    }

    with pytest.raises(ValueError, match="metric values"):
        build_comparison_traces(None, records, ["PGD"])


def test_convex_ineq_visualization_accepts_fw_scalar_violation(tmp_path):
    payload = hom_pgd.convex_algorithm_comparison(
        problem_type="socp",
        algorithms=["FW"],
        n_var=6,
        n_soc_cons=3,
        max_iterations=2,
        max_running_time=5,
        device="cpu",
        dtype="float32",
        visualize=True,
        output_dir=tmp_path / "convex_ineq_fw",
    )
    assert _payload_metrics(payload)["results"]["FW"]["final_inequality_violation"] >= 0.0
    assert "full_violation_convergence_iteration" in payload["artifacts"]


def test_inn_training_benchmark_supports_non_2d_problem(tmp_path):
    result, output_dir = _run_workload('inn_training_non_2d_smoke', inn_pgd.inn_training_benchmark, _inn_training_params(n_samples=8, batch_size=4, total_iteration=2, max_iterations=2, problem_config={'n_var': 3, 'n_qua_cons': 2, 'n_linear_cons': 0}), tmp_path)
    assert result.metrics["latent_dim"] == 3
    assert result.metrics["decision_dim"] == 3
    assert (output_dir / result.artifacts["training_record"]).exists()
    assert (output_dir / result.artifacts["model"]).exists()


def test_inn_training_benchmark_accepts_canonical_config_groups(tmp_path):
    result, output_dir = _run_workload('inn_training_canonical_groups', inn_pgd.inn_training_benchmark, _inn_training_params(train_config={'n_samples': 8, 'batch_size': 4, 'total_iteration': 2}, max_iterations=2, problem_config={'n_var': 3, 'n_qua_cons': 2, 'n_linear_cons': 0}, model_config={'h_dim': 6}, optimizer_config={'momentum': 0.2}), tmp_path)
    assert result.metrics["latent_dim"] == 3
    assert result.metrics["decision_dim"] == 3
    assert (output_dir / result.artifacts["training_record"]).exists()
    assert (output_dir / result.artifacts["model"]).exists()


def test_inn_training_benchmark_writes_only_canonical_artifacts(tmp_path):
    result, output_dir = _run_workload('inn_training_canonical_artifacts', inn_pgd.inn_training_benchmark, _inn_training_params(n_samples=8, batch_size=4, total_iteration=2, max_iterations=3), tmp_path)
    del result
    artifact_dir = output_dir / "artifacts"
    cache_dir = output_dir.parent / "training_cache"
    assert list(cache_dir.glob("*/inn_training_record.npy"))
    assert list(cache_dir.glob("*/inn_mapping.pt"))
    assert not (artifact_dir / "inn_training_record.npy").exists()
    assert not (artifact_dir / "inn_mapping.pt").exists()
    assert not (artifact_dir / "training_record.npy").exists()
    assert not (artifact_dir / "mdh_mapping.pth").exists()
    assert not (artifact_dir / "mdh_mapping_coupling.pth").exists()
    assert not (artifact_dir / "inn_mapping_checkpoint.pt").exists()


def test_qcqp_inn_comparison_writes_only_canonical_artifacts(monkeypatch, tmp_path):
    class _FakeOptimizer:
        def __init__(self, problem, paras, model):
            self.problem = problem

        def optimize(self, input_params, objective_params, seed, initial_point=None):
            del seed
            assert objective_params is not None
            assert initial_point is not None
            n_samples = input_params.shape[0]
            n_var = self.problem.nvar
            assert tuple(initial_point.shape) == (n_samples, n_var)
            device = input_params.device
            dtype = input_params.dtype
            x_opt = torch.zeros(n_samples, n_var, device=device, dtype=dtype)
            decision_traj = torch.zeros(2, n_var, device=device, dtype=dtype)
            latent_traj = torch.zeros(2, n_var, device=device, dtype=dtype)
            obj_traj = torch.linspace(1.0, float(n_samples), n_samples, device=device, dtype=dtype)
            cons_traj = torch.zeros(n_samples, device=device, dtype=dtype)
            per_iter_time = [0.1, 0.2]
            return x_opt, decision_traj, latent_traj, obj_traj, cons_traj, per_iter_time

    def _fake_loader(
        mapping_type,
        data,
        *,
        problem_args,
        model_args,
        train_args,
        save_dir,
        retrain=False,
        runtime_device=None,
        runtime_dtype=None,
        ensure_results_save_freq=False,
    ):
        assert mapping_type == "inn"
        del data, problem_args, model_args, train_args, retrain, runtime_device, runtime_dtype, ensure_results_save_freq
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "inn_mapping.pt"
        record_path = save_dir / "inn_training_record.npy"
        model_path.write_bytes(b"fake")
        np.save(record_path, {"training_time_list": [0.25]})
        return {
            "model": _FakeINNMapping(),
            "training_record": {"training_time_list": [0.25]},
            "model_path": model_path,
            "record_path": record_path,
        }

    monkeypatch.setattr(qcqp_inn_common, "prepare_learning_mapping", _fake_loader)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "INNPGDOptimizer", _FakeOptimizer)

    result, output_dir = _run_workload('qcqp_inn_canonical_artifacts', inn_pgd.qcqp_inn_experiment, _qcqp_inn_params(n_var=4, n_qua_cons=2, n_linear_cons=0, num_test_instance=3, train_config={'n_samples': 8, 'batch_size': 2, 'total_iteration': 2}, max_iterations=2, retrain=True), tmp_path)
    artifact_dir = output_dir / "artifacts"
    cache_dir = output_dir.parent / "training_cache"
    assert (output_dir / result.artifacts["training_record"]).exists()
    assert (output_dir / result.artifacts["summary"]).exists()
    assert (output_dir / result.artifacts["manifest"]).exists()
    assert (output_dir / result.artifacts["records_dir"]).is_dir()
    assert list(cache_dir.glob("*/inn_training_record.npy"))
    assert list(cache_dir.glob("*/inn_mapping.pt"))
    assert not (artifact_dir / "inn_training_record.npy").exists()
    assert not (artifact_dir / "inn_mapping.pt").exists()
    assert (artifact_dir / "records" / "INN-PGD.npy").exists()
    assert (artifact_dir / "inn_qcqp_summary.json").exists()
    assert not (artifact_dir / "training_record.npy").exists()
    assert not (artifact_dir / "inn_pgd_record.npy").exists()
    assert not (artifact_dir / "qcqp_inn_summary.json").exists()


def test_qcqp_inn_training_cache_is_independent_of_result_label(tmp_path):
    params = _qcqp_inn_params(**{
        "n_var": 2,
        "n_qua_cons": 2,
        "n_linear_cons": 0,
        "num_test_instance": 2,
        "seed": 2025,
        "train_config": {"n_samples": 8, "batch_size": 2, "total_iteration": 2},
        "model_config": {"latent_init_shape": "sphere"},
    })
    inn_only_context = qcqp_inn_common._build_qcqp_inn_case_context(
        tmp_path / "qcqp_n2_q2_inn_only",
        {**params, "result_label": "qcqp_n2_q2_inn_only"},
    )
    baseline_context = qcqp_inn_common._build_qcqp_inn_case_context(
        tmp_path / "qcqp_n2_q2_baseline_compare",
        {**params, "result_label": "qcqp_n2_q2_baseline_compare", "lagrangian_baselines": ["ALM"]},
    )

    assert inn_only_context["save_dir"] == baseline_context["save_dir"]
    assert inn_only_context["save_dir"].parent == tmp_path / "training_cache"


def test_qcqp_inn_comparison_is_internal_not_config_entrypoint():
    assert "qcqp_inn_comparison" not in parametric_registry.PARAMETRIC_BENCHMARKS
    assert "qcqp_inn_experiment" in parametric_registry.PARAMETRIC_BENCHMARKS
    with pytest.raises(AttributeError):
        getattr(inn_pgd, "qcqp_inn_comparison")


def test_qcqp_inn_comparison_generates_2d_visualization(monkeypatch, tmp_path):
    captured = {}

    class _FakeOptimizer:
        def __init__(self, problem, paras, model):
            self.problem = problem

        def optimize(self, input_params, objective_params, seed, initial_point=None):
            del seed
            assert objective_params is not None
            assert initial_point is not None
            n_samples = input_params.shape[0]
            n_var = self.problem.nvar
            assert tuple(initial_point.shape) == (n_samples, n_var)
            device = input_params.device
            dtype = input_params.dtype
            x_opt = torch.zeros(n_samples, n_var, device=device, dtype=dtype)
            decision_traj = torch.zeros(2 * n_samples, n_var, device=device, dtype=dtype)
            latent_traj = torch.zeros(2 * n_samples, n_var, device=device, dtype=dtype)
            obj_traj = torch.tensor([1.0, 2.0, 0.5, 1.5], device=device, dtype=dtype)
            cons_traj = torch.zeros(2 * n_samples, device=device, dtype=dtype)
            per_iter_time = [0.1]
            return x_opt, decision_traj, latent_traj, obj_traj, cons_traj, per_iter_time

    def _fake_loader(
        mapping_type,
        data,
        *,
        problem_args,
        model_args,
        train_args,
        save_dir,
        retrain=False,
        runtime_device=None,
        runtime_dtype=None,
        ensure_results_save_freq=False,
    ):
        assert mapping_type == "inn"
        del data, problem_args, model_args, train_args, retrain, runtime_device, runtime_dtype, ensure_results_save_freq
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "inn_mapping.pt"
        record_path = save_dir / "inn_training_record.npy"
        model_path.write_bytes(b"fake")
        np.save(record_path, {"training_time_list": [0.25]})
        return {
            "model": _FakeINNMapping(),
            "training_record": {"training_time_list": [0.25]},
            "model_path": model_path,
            "record_path": record_path,
        }

    def _fake_visualize(**kwargs):
        captured["training_visualize"] = captured.get("training_visualize", 0) + 1
        captured["obj_shape"] = tuple(np.asarray(kwargs["obj_traj"]).shape)
        captured["decision_shape"] = tuple(np.asarray(kwargs["decision_traj"]).shape)
        captured["plot_optimizer_diagnostics"] = kwargs["plot_optimizer_diagnostics"]
        return {"training_metrics": "artifacts/inn_training_metrics.pdf"}

    def _fake_comparison_visualize(*args, **kwargs):
        del args
        captured["comparison_input_shape"] = tuple(kwargs["input_params"].shape)
        captured["comparison_objective_shape"] = tuple(kwargs["objective_params"].shape)
        return {"qcqp_2d_objective_iteration": "artifacts/qcqp_2d_objective_by_iter.pdf"}

    monkeypatch.setattr(qcqp_inn_common, "prepare_learning_mapping", _fake_loader)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "INNPGDOptimizer", _FakeOptimizer)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "save_inn_training_visualizations", _fake_visualize)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "save_qcqp_2d_comparison_visualizations", _fake_comparison_visualize)

    params = _qcqp_inn_params(**{
        "n_var": 2,
        "n_qua_cons": 2,
        "n_linear_cons": 0,
        "num_test_instance": 2,
        "train_config": {"n_samples": 8, "batch_size": 2, "total_iteration": 2},
    })
    case_context, instance_batch = _qcqp_inn_case_and_instances(tmp_path, **params)
    payload = qcqp_inn_comparison_benchmarks.qcqp_inn_comparison(
        **_kwargs_for(
            qcqp_inn_comparison_benchmarks.qcqp_inn_comparison,
            merge_params(params, {"max_iterations": 1, "retrain": True, "visualize": True}),
        ),
        case_context=case_context,
        instance_batch=instance_batch,
        output_dir=tmp_path,
    )

    assert payload["artifacts"]["training_metrics"] == "artifacts/inn_training_metrics.pdf"
    assert payload["artifacts"]["qcqp_2d_objective_iteration"] == "artifacts/qcqp_2d_objective_by_iter.pdf"
    assert captured["training_visualize"] == 1
    assert captured["obj_shape"] == (0,)
    assert captured["decision_shape"] == (0,)
    assert captured["comparison_input_shape"][0] == 1
    assert captured["comparison_objective_shape"][0] == 1
    assert captured["plot_optimizer_diagnostics"] is False


def test_qcqp_inn_comparison_visualizes_multiple_instances(monkeypatch, tmp_path):
    captured_indices = []
    training_visualize_calls = []

    class _FakeOptimizer:
        def __init__(self, problem, paras, model):
            self.problem = problem

        def optimize(self, input_params, objective_params, seed, initial_point=None):
            del seed
            assert objective_params is not None
            n_samples = input_params.shape[0]
            n_var = self.problem.nvar
            assert tuple(initial_point.shape) == (n_samples, n_var)
            device = input_params.device
            dtype = input_params.dtype
            x_opt = torch.zeros(n_samples, n_var, device=device, dtype=dtype)
            decision_traj = torch.zeros(2 * n_samples, n_var, device=device, dtype=dtype)
            latent_traj = torch.zeros(2 * n_samples, n_var, device=device, dtype=dtype)
            obj_traj = torch.zeros(2 * n_samples, device=device, dtype=dtype)
            cons_traj = torch.zeros(2 * n_samples, device=device, dtype=dtype)
            return x_opt, decision_traj, latent_traj, obj_traj, cons_traj, [0.1]

    def _fake_loader(
        mapping_type,
        data,
        *,
        problem_args,
        model_args,
        train_args,
        save_dir,
        retrain=False,
        runtime_device=None,
        runtime_dtype=None,
        ensure_results_save_freq=False,
    ):
        assert mapping_type == "inn"
        del data, problem_args, model_args, train_args, retrain, runtime_device, runtime_dtype, ensure_results_save_freq
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "inn_mapping.pt"
        record_path = save_dir / "inn_training_record.npy"
        model_path.write_bytes(b"fake")
        np.save(record_path, {"training_time_list": [0.25]})
        return {
            "model": _FakeINNMapping(),
            "training_record": {"training_time_list": [0.25]},
            "model_path": model_path,
            "record_path": record_path,
        }

    def _fake_visualize(**kwargs):
        training_visualize_calls.append(kwargs)
        assert kwargs["plot_optimizer_diagnostics"] is False
        assert "instance_idx" not in kwargs or kwargs["instance_idx"] is None
        return {"training_metrics": "artifacts/inn_training_metrics.pdf"}

    def _fake_comparison_visualize(*args, **kwargs):
        del args
        captured_indices.append(int(kwargs["instance_idx"]))
        return {"qcqp_2d_objective_iteration": f"artifacts/comparison_inst{kwargs['instance_idx']}.pdf"}

    monkeypatch.setattr(qcqp_inn_common, "prepare_learning_mapping", _fake_loader)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "INNPGDOptimizer", _FakeOptimizer)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "save_inn_training_visualizations", _fake_visualize)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "save_qcqp_2d_comparison_visualizations", _fake_comparison_visualize)

    params = _qcqp_inn_params(**{
        "n_var": 2,
        "n_qua_cons": 2,
        "n_linear_cons": 0,
        "num_test_instance": 2,
        "train_config": {"n_samples": 8, "batch_size": 2, "total_iteration": 2},
    })
    case_context, instance_batch = _qcqp_inn_case_and_instances(tmp_path, **params)
    payload = qcqp_inn_comparison_benchmarks.qcqp_inn_comparison(
        **_kwargs_for(
            qcqp_inn_comparison_benchmarks.qcqp_inn_comparison,
            merge_params(
                params,
                {"max_iterations": 1, "retrain": True, "visualize": True, "visualize_instance_idx": [0, 1]},
            ),
        ),
        case_context=case_context,
        instance_batch=instance_batch,
        output_dir=tmp_path,
    )

    assert captured_indices == [0, 1]
    assert len(training_visualize_calls) == 1
    assert payload["artifacts"]["training_metrics"] == "artifacts/inn_training_metrics.pdf"
    assert payload["artifacts"]["qcqp_2d_objective_iteration_inst0"] == "artifacts/comparison_inst0.pdf"
    assert payload["artifacts"]["qcqp_2d_objective_iteration_inst1"] == "artifacts/comparison_inst1.pdf"


def test_qcqp_inn_comparison_runs_optional_iterative_baselines(monkeypatch, tmp_path):
    captured = []

    class _FakeOptimizer:
        def __init__(self, problem, paras, model):
            self.problem = problem

        def optimize(self, input_params, objective_params, seed, initial_point=None):
            del seed
            assert objective_params is not None
            assert initial_point is not None
            n_samples = input_params.shape[0]
            n_var = self.problem.nvar
            assert tuple(initial_point.shape) == (n_samples, n_var)
            device = input_params.device
            dtype = input_params.dtype
            x_opt = torch.zeros(n_samples, n_var, device=device, dtype=dtype)
            obj_traj = torch.zeros(n_samples, device=device, dtype=dtype)
            cons_traj = torch.zeros(n_samples, device=device, dtype=dtype)
            return x_opt, [], [], obj_traj, cons_traj, [0.1]

    def _fake_loader(
        mapping_type,
        data,
        *,
        problem_args,
        model_args,
        train_args,
        save_dir,
        retrain=False,
        runtime_device=None,
        runtime_dtype=None,
        ensure_results_save_freq=False,
    ):
        assert mapping_type == "inn"
        del data, problem_args, model_args, train_args, retrain, runtime_device, runtime_dtype, ensure_results_save_freq
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "inn_mapping.pt"
        record_path = save_dir / "inn_training_record.npy"
        model_path.write_bytes(b"fake")
        np.save(record_path, {"training_time_list": [0.25]})
        return {
            "model": _FakeINNMapping(),
            "training_record": {"training_time_list": [0.25]},
            "model_path": model_path,
            "record_path": record_path,
        }

    def _fake_run_lagrangian_baseline(problem, init_point, params, *, method, seed):
        del seed
        captured.append(
            (
                method,
                params["use_lagrangian"],
                params["use_proximal"],
                tuple(init_point.shape),
                init_point.detach().cpu().clone(),
            )
        )
        return {
            "obj_traj": np.asarray([1.0]),
            "cons_traj": np.asarray([0.0]),
            "iter_time": [0.01],
            "x_solved": torch.zeros(1, problem.nvar, dtype=getattr(problem, "dtype", torch.float32)),
            "total_wall_time": 0.02,
        }

    monkeypatch.setattr(qcqp_inn_common, "prepare_learning_mapping", _fake_loader)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "INNPGDOptimizer", _FakeOptimizer)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "run_inn_lagrangian_baseline", _fake_run_lagrangian_baseline)

    params = _qcqp_inn_params(**{
        "n_var": 2,
        "n_qua_cons": 2,
        "n_linear_cons": 0,
        "num_test_instance": 1,
        "train_config": {"n_samples": 8, "batch_size": 1, "total_iteration": 2},
    })
    case_context, instance_batch = _qcqp_inn_case_and_instances(tmp_path, **params)
    payload = qcqp_inn_comparison_benchmarks.qcqp_inn_comparison(
        **_kwargs_for(
            qcqp_inn_comparison_benchmarks.qcqp_inn_comparison,
            merge_params(
                params,
                {
                    "max_iterations": 1,
                    "retrain": True,
                    "visualize": False,
                    "lagrangian_baselines": ["ALM", "Penalty", "Prox-Penalty"],
                },
            ),
        ),
        case_context=case_context,
        instance_batch=instance_batch,
        output_dir=tmp_path,
    )

    assert [item[0] for item in captured] == ["ALM", "Penalty", "Prox-Penalty"]
    assert captured[0][1:3] == (True, False)
    assert captured[1][1:3] == (False, False)
    assert captured[2][1:3] == (False, True)
    assert torch.allclose(captured[0][4], captured[1][4])
    assert torch.allclose(captured[1][4], captured[2][4])
    assert _payload_metrics(payload)["iterative_baseline_rows"][0]["method"] == "ALM"
    assert "iterative_baseline_json" in payload["artifacts"]


def test_qcqp_inn_comparison_visualizes_inn_only_2d(monkeypatch, tmp_path):
    captured = []

    class _FakeOptimizer:
        def __init__(self, problem, paras, model):
            self.problem = problem

        def optimize(self, input_params, objective_params, seed, initial_point=None):
            del objective_params, seed
            assert initial_point is not None
            n_samples = input_params.shape[0]
            n_var = self.problem.nvar
            device = input_params.device
            dtype = input_params.dtype
            x_opt = torch.zeros(n_samples, n_var, device=device, dtype=dtype)
            decision_traj = torch.zeros(2 * n_samples, n_var, device=device, dtype=dtype)
            latent_traj = torch.zeros(2 * n_samples, n_var, device=device, dtype=dtype)
            obj_traj = torch.zeros(2 * n_samples, device=device, dtype=dtype)
            cons_traj = torch.zeros(2 * n_samples, device=device, dtype=dtype)
            return x_opt, decision_traj, latent_traj, obj_traj, cons_traj, [0.1]

    def _fake_loader(
        mapping_type,
        data,
        *,
        problem_args,
        model_args,
        train_args,
        save_dir,
        retrain=False,
        runtime_device=None,
        runtime_dtype=None,
        ensure_results_save_freq=False,
    ):
        assert mapping_type == "inn"
        del data, problem_args, model_args, train_args, retrain, runtime_device, runtime_dtype, ensure_results_save_freq
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "inn_mapping.pt"
        record_path = save_dir / "inn_training_record.npy"
        model_path.write_bytes(b"fake")
        np.save(record_path, {"training_time_list": [0.25]})
        return {
            "model": _FakeINNMapping(),
            "training_record": {"training_time_list": [0.25]},
            "model_path": model_path,
            "record_path": record_path,
        }

    def _fake_training_visualize(**kwargs):
        assert kwargs["plot_optimizer_diagnostics"] is False
        assert "instance_idx" not in kwargs or kwargs["instance_idx"] is None
        return {"training_metrics": "artifacts/inn_training_metrics.pdf"}

    def _fake_comparison_visualize(*args, **kwargs):
        captured.append((int(kwargs["instance_idx"]), dict(kwargs["iterative_records"])))
        return {"qcqp_2d_objective_iteration": f"artifacts/comparison_inst{kwargs['instance_idx']}.pdf"}

    monkeypatch.setattr(qcqp_inn_common, "prepare_learning_mapping", _fake_loader)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "INNPGDOptimizer", _FakeOptimizer)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "save_inn_training_visualizations", _fake_training_visualize)
    monkeypatch.setattr(qcqp_inn_comparison_benchmarks, "save_qcqp_2d_comparison_visualizations", _fake_comparison_visualize)

    params = _qcqp_inn_params(**{
        "n_var": 2,
        "n_qua_cons": 2,
        "n_linear_cons": 0,
        "num_test_instance": 2,
        "train_config": {"n_samples": 8, "batch_size": 2, "total_iteration": 2},
    })
    case_context, instance_batch = _qcqp_inn_case_and_instances(tmp_path, **params)
    payload = qcqp_inn_comparison_benchmarks.qcqp_inn_comparison(
        **_kwargs_for(
            qcqp_inn_comparison_benchmarks.qcqp_inn_comparison,
            merge_params(
                params,
                {
                    "max_iterations": 1,
                    "retrain": True,
                    "visualize": True,
                    "visualize_instance_idx": [0, 1],
                    "lagrangian_baselines": [],
                },
            ),
        ),
        case_context=case_context,
        instance_batch=instance_batch,
        output_dir=tmp_path,
    )

    assert captured == [(0, {}), (1, {})]
    assert payload["artifacts"]["qcqp_2d_objective_iteration_inst0"] == "artifacts/comparison_inst0.pdf"
    assert payload["artifacts"]["qcqp_2d_objective_iteration_inst1"] == "artifacts/comparison_inst1.pdf"


def test_qcqp_inn_sensitivity_sweep_overlays_case_convergence(monkeypatch, tmp_path):
    captured = []
    phase_order = []
    seen_num_test_instance = []
    seen_case_convergence = []

    def _fake_prepare(case_context):
        del case_context
        phase_order.append("train")

    def _fake_qcqp_inn_comparison(output_dir=None, **params):
        phase_order.append("test")
        seen_num_test_instance.append(params.get("num_test_instance"))
        seen_case_convergence.append(params.get("plot_case_convergence"))
        record = {
            "obj_trajectory": np.asarray([2.0, 1.0]),
            "cons_trajectory": np.asarray([1.0, 0.0]),
            "per_iter_time": [0.1],
            "x_trajectory": np.zeros((2, 2)),
            "z_trajectory": np.zeros((2, 2)),
        }
        artifacts, _, _, _ = artifact_helpers.save_incremental_comparison_artifacts(
            output_dir,
            records={"INN-PGD": record},
            summaries={
                "INN-PGD": {
                    "final_objective": 1.0,
                    "final_violation": 0.0,
                    "total_wall_time": 0.1,
                    "total_iter_time": 0.1,
                    "iterations": 1,
                }
            },
            algorithm_order=["INN-PGD"],
        )
        return {
            "objective": 1.0,
            "final_violation": 0.0,
            "feasible": True,
            "iterations": 1,
            "artifacts": artifacts,
        }

    def _fake_sensitivity_visualize(output_dir, artifact_root_path, **kwargs):
        del output_dir, artifact_root_path
        captured.append(
            (
                kwargs["sweep_name"],
                [case["label"] for case in kwargs["cases"]],
                [case["record"] is not None for case in kwargs["cases"]],
                kwargs["visualize_instance_idx"],
            )
        )
        return {f"overlay_{kwargs['sweep_name']}": f"artifacts/{kwargs['sweep_name']}.pdf"}

    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "_prepare_qcqp_inn_training_context", _fake_prepare)
    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "qcqp_inn_comparison", _fake_qcqp_inn_comparison)
    monkeypatch.setattr(
        qcqp_inn_sweep_benchmarks,
        "save_qcqp_sensitivity_convergence_visualizations",
        _fake_sensitivity_visualize,
    )

    payload = qcqp_inn_sweep_benchmarks.qcqp_inn_sensitivity_sweep(
        **_qcqp_inn_params(
        sensitivity_sweeps={"num_layer": [1, 3, 5]},
        n_var=2,
        num_test_instance=1,
        visualize=True,
        visualize_instance_idx=[0],
        plot_case_convergence=False,
        ),
        output_dir=tmp_path,
    )

    assert phase_order == ["train", "train", "train", "test", "test", "test"]
    assert seen_num_test_instance == [1, 1, 1]
    assert seen_case_convergence == [False, False, False]
    assert captured == [("num_layer", ["num_layer=1", "num_layer=3", "num_layer=5"], [True, True, True], [0])]
    assert payload["artifacts"]["overlay_num_layer"] == "artifacts/num_layer.pdf"


def test_qcqp_visualize_instance_idx_all_tracks_num_test_instance_and_toy_defaults_to_one_instance():
    assert qcqp_inn_common._visualize_instance_indices("all", 3) == [0, 1, 2]
    assert qcqp_inn_common._visualize_instance_indices("ALL", 2) == [0, 1]

    repo_root = Path(__file__).resolve().parents[1]
    script_dir = repo_root / "scripts" / "inn_pgd"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("run_qcqp_inn_toy", script_dir / "run_qcqp_inn_toy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.PARAMS["num_test_instance"] == 3
    assert module.PARAMS["visualize_instance_idx"] == [0]
    assert qcqp_inn_common._visualize_instance_indices(
        module.PARAMS["visualize_instance_idx"],
        module.PARAMS["num_test_instance"],
    ) == [0]


def test_jcc_problem_benchmark_writes_feasibility_artifacts(tmp_path):
    result, output_dir = _run_workload('jcc_problem_smoke', inn_pgd.jcc_problem_benchmark, {'n_scenarios': 2, 'seed': 7}, tmp_path)
    assert result.metrics["n_scenarios"] == 2
    assert 0.0 <= result.metrics["feasibility_rate"] <= 1.0
    assert (output_dir / result.artifacts["scenario_feasibility"]).exists()
    assert (output_dir / result.artifacts["summary"]).exists()


def _jcc_lagrangian_baseline_config(**overrides):
    config = {
        "inner_iterations": 1,
        "learning_rate": 1e-3,
        "inner_learning_rate": 1e-3,
        "dual_learning_rate": 1e-1,
        "lr_decay": 0.999,
        "inner_lr_decay": 0.999,
        "min_lr": 1e-6,
        "inner_min_lr": 1e-6,
        "penalty_coef": 10.0,
        "penalty_growth": 1.1,
        "proximal_coef": 0.1,
        "max_penalty": 1e2,
        "max_dual": 1e2,
        "max_running_time": 5,
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
    }
    config.update(overrides)
    return config


def test_jcc_algorithm_comparison_reports_three_solver_baselines(tmp_path):
    result, output_dir = _run_workload('jcc_compare_smoke', inn_pgd.jcc_algorithm_comparison, {'n_scenarios': 2, 'seed': 7, 'max_iterations': 1, 'lagrangian_baseline_config': _jcc_lagrangian_baseline_config()}, tmp_path)
    assert "solver_status" in result.metrics
    assert "cvar_status" in result.metrics
    assert "scenario_status" in result.metrics
    assert "cvar_objective" in result.metrics
    assert "scenario_objective" in result.metrics
    assert any(row["route"] == "solver" for row in result.metrics["comparison_rows"])
    assert any(row["route"] == "iterative" for row in result.metrics["comparison_rows"])
    assert set(result.metrics["best_method"]) == {"route", "method"}
    assert set(result.metrics["comparison_summary"]) == {"objective", "feasible", "best_method"}
    assert len(result.metrics["comparison_summary_rows"]) == 6
    assert (output_dir / result.artifacts["solver_record"]).exists()
    assert (output_dir / result.artifacts["CVaR_record"]).exists()
    assert (output_dir / result.artifacts["scenario_record"]).exists()
    assert (output_dir / result.artifacts["summary"]).exists()
    assert (output_dir / result.artifacts["comparison_summary_json"]).exists()
    assert (output_dir / result.artifacts["comparison_summary_csv"]).exists()


def test_jcc_inn_pgd_uses_canonical_lagrangian_config_names():
    signature = inspect.signature(inn_pgd.jcc_algorithm_comparison)
    assert "lagrangian_baseline_config" in signature.parameters
    assert "lagrangian_baseline_config_overrides" not in signature.parameters
    assert "lagrangian_baselines" in signature.parameters
    assert "num_test_instance" in signature.parameters
    assert "inn_pgd_n_samples" not in signature.parameters
    assert "algorithm_config" not in signature.parameters
    assert "algorithm_config_overrides" not in signature.parameters

    repo_root = Path(__file__).resolve().parents[1]
    script_text = (repo_root / "scripts" / "inn_pgd" / "run_jcc_opf_compare.py").read_text()
    assert "inn_pgd_config" in script_text
    assert "from homopt.experiments.inn_pgd import jcc_algorithm_comparison" in script_text
    assert "homopt.experiments.hom_alm" not in script_text
    assert '"lagrangian_baseline_config"' in script_text
    assert '"lagrangian_baseline_config_overrides"' not in script_text
    assert '"num_test_instance"' in script_text
    assert "inn_pgd_n_samples" not in script_text
    assert "algorithm_config" not in script_text

    script_dir = repo_root / "scripts" / "inn_pgd"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("run_jcc_opf_compare", script_dir / "run_jcc_opf_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PARAMS["num_bus"] == 57
    assert module.PARAMS["run_inn_pgd"] is True
    assert module.PARAMS["lagrangian_baseline_config"]["lagrangian_gradient"] == "autograd"
    normalized_problem_config = config_common.normalize_jcc_problem_config(
        n_scenarios=module.PARAMS["n_scenarios"],
        epsilon=module.PARAMS["epsilon"],
        demand_std=module.PARAMS["demand_std"],
        seed=module.PARAMS["seed"],
        problem_config=module.PARAMS["problem_config"],
    )
    assert normalized_problem_config["pglib_data_dir"] is None
    assert normalized_problem_config["download_if_missing"] is True


def test_jcc_inn_config_rejects_training_keys_in_model_config():
    with pytest.raises(ValueError, match="use train_config"):
        jcc_methods._normalize_jcc_inn_configs(model_config={"batch_size": 4})


def test_jcc_inn_config_rejects_optimizer_iteration_budget():
    with pytest.raises(ValueError, match="optimizer_config must not set max_iterations"):
        jcc_methods._normalize_jcc_inn_configs(
            max_iterations=2,
            model_config={},
            optimizer_config={"max_iterations": 3},
            train_config={"n_samples": 4, "batch_size": 2, "total_iteration": 1},
        )


def test_inn_lagrangian_baseline_config_uses_top_level_max_iterations():
    baseline_config = _qcqp_inn_params()["lagrangian_baseline_config"]
    params = inn_baselines.build_inn_lagrangian_method_params(7, baseline_config, max_iterations=2)
    assert params["ALM"]["outer_iterations"] == 2
    assert params["Penalty"]["outer_iterations"] == 2
    assert params["Prox-Penalty"]["outer_iterations"] == 2

    with pytest.raises(ValueError, match="INN-PGD lagrangian_baseline_config must not set"):
        inn_baselines.build_inn_lagrangian_method_params(7, {**baseline_config, "max_iterations": 2}, max_iterations=2)
    with pytest.raises(ValueError, match="INN-PGD lagrangian_baseline_config must not set"):
        inn_baselines.build_inn_lagrangian_method_params(7, {**baseline_config, "outer_iterations": 2}, max_iterations=2)
    with pytest.raises(ValueError, match="Unsupported INN-PGD lagrangian_baseline_config keys"):
        inn_baselines.build_inn_lagrangian_method_params(
            7,
            {**baseline_config, "outer_stepsize_rule": "adaptive"},
            max_iterations=2,
        )


def test_jcc_algorithm_comparison_accepts_canonical_config_groups(tmp_path):
    result, output_dir = _run_workload('jcc_compare_canonical', inn_pgd.jcc_algorithm_comparison, {'seed': 7, 'max_iterations': 1, 'problem_config': {'n_scenarios': 2, 'demand_std': 0.04}, 'solver_configs': {'mixed_integer': {'verbose': False}, 'cvar': {'verbose': False}, 'scenario': {'verbose': False}}, 'lagrangian_baseline_config': _jcc_lagrangian_baseline_config()}, tmp_path)
    assert result.metrics["solver_status"] is not None
    assert result.metrics["cvar_status"] is not None
    assert result.metrics["scenario_status"] is not None
    assert len(result.metrics["comparison_summary_rows"]) == 6
    assert (output_dir / result.artifacts["summary"]).exists()
    assert (output_dir / result.artifacts["comparison_summary_json"]).exists()


def test_jcc_lagrangian_baselines_require_explicit_config(tmp_path):
    with pytest.raises(ValueError, match="lagrangian_baseline_config is required"):
        _run_workload(
            "jcc_compare_missing_lagrangian_config",
            inn_pgd.jcc_algorithm_comparison,
            {"seed": 7, "max_iterations": 1, "problem_config": {"n_scenarios": 2}},
            tmp_path,
        )


def test_jcc_solver_only_compare_does_not_require_lagrangian_config(tmp_path):
    result, _ = _run_workload(
        "jcc_compare_solver_only",
        inn_pgd.jcc_algorithm_comparison,
        {"seed": 7, "max_iterations": 1, "problem_config": {"n_scenarios": 2}, "lagrangian_baselines": []},
        tmp_path,
    )
    assert len(result.metrics["comparison_summary_rows"]) == 3


def test_jcc_algorithm_comparison_can_include_inn_pgd(monkeypatch, tmp_path):
    def _fake_inn_pgd(problem, **kwargs):
        del kwargs
        x_mid = ((problem.P_min + problem.P_max) / 2).detach().cpu().numpy()
        return {
            "x_opt": x_mid,
            "obj_traj": np.array([1.0, 0.8]),
            "cons_traj": np.array([0.1, 0.0]),
            "iter_time": [0.01, 0.02],
            "total_wall_time": 0.03,
            "final_objective": 0.8,
            "final_violation": 0.0,
            "chance_feasibility_rate": 1.0,
            "violation": 0.0,
            "feasible": True,
        }

    monkeypatch.setattr(jcc_opf_benchmarks, "_run_jcc_inn_pgd_variant", _fake_inn_pgd)
    result, output_dir = _run_workload('jcc_compare_with_inn_pgd', inn_pgd.jcc_algorithm_comparison, {'n_scenarios': 2, 'seed': 7, 'run_inn_pgd': True, 'max_iterations': 1, 'lagrangian_baseline_config': _jcc_lagrangian_baseline_config()}, tmp_path)

    methods = {row["method"] for row in result.metrics["comparison_rows"]}
    assert "INN-PGD" in methods
    assert result.metrics["inn_pgd_final_violation"] == 0.0
    assert (output_dir / result.artifacts["INN-PGD_record"]).exists()


def test_jcc_inn_training_residual_is_continuous_and_differentiable():
    problem = jcc_setup._build_jcc_problem(
        num_bus=57,
        n_scenarios=2,
        epsilon=0.1,
        demand_std=0.05,
        seed=7,
        runtime_device=torch.device("cpu"),
        runtime_dtype=torch.float32,
    )
    adapter = jcc_methods._JCCINNPGDAdapter(problem)
    inputs = problem.sample_instances(3, seed=7)
    z = torch.zeros(3, adapter.nvar, requires_grad=True)
    x = adapter.scale(inputs, z)

    residual = adapter.ineq_resid(inputs, x, clip=True)
    chance_violation = adapter.violations(inputs, x)

    assert residual.requires_grad
    assert residual.shape[0] == 3
    assert residual.shape[1] == problem.n_scenarios
    assert chance_violation.shape == (3, 1)
    grad = torch.autograd.grad(residual.sum(), z, allow_unused=False)[0]
    assert torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad) > 0


def test_jcc_inn_adapter_propagates_float64_runtime_dtype():
    problem = jcc_setup._build_jcc_problem(
        num_bus=30,
        n_scenarios=2,
        epsilon=0.1,
        demand_std=0.05,
        seed=7,
        runtime_device=torch.device("cpu"),
        runtime_dtype=torch.float64,
    )
    adapter = jcc_methods._JCCINNPGDAdapter(problem)
    inputs, _ = adapter.generate_problem_samples_torch(n_samples=2, dtype=torch.float64)
    z = torch.zeros(2, adapter.nvar, dtype=adapter.dtype, requires_grad=True)
    x = adapter.scale(inputs, z)

    objective = adapter.objective(x)
    residual = adapter.ineq_resid(inputs, x, clip=True)
    chance_violation = adapter.violations(inputs, x)
    grad = torch.autograd.grad(objective.sum(), z, allow_unused=False)[0]

    assert problem.dtype is torch.float64
    assert adapter.dtype is torch.float64
    assert inputs.dtype is torch.float64
    assert x.dtype is torch.float64
    assert objective.dtype is torch.float64
    assert residual.dtype is torch.float64
    assert chance_violation.dtype is torch.float64
    assert torch.isfinite(grad).all()


def test_jcc_inn_pgd_nonfinite_output_is_failed_status(monkeypatch, tmp_path):
    class _FakeAdapter:
        nvar = 3
        n_bus = 57
        n_scenarios = 2
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self, problem):
            self.problem = problem

        def sample_instance_batch(self, n_instances, seed):
            del seed
            return SimpleNamespace(inputs=torch.ones(int(n_instances), self.n_scenarios, self.n_bus))

    class _FakeProblem:
        config = {"epsilon": 0.1}

        def compute_scenario_feasibility(self, x, scenario_batch=None):
            raise AssertionError("nonfinite INN output should not be evaluated for scenario feasibility")

    class _FakeOptimizer:
        def __init__(self, problem, paras, model):
            del problem, paras, model

        def optimize(self, initial_point=None, input_params=None, seed=None):
            del initial_point, input_params, seed
            x_opt = torch.full((1, 3), float("nan"))
            obj_traj = torch.full((2,), float("nan"))
            cons_traj = torch.tensor([1.0, 1.0])
            return x_opt, [], torch.zeros(2, 3), obj_traj, cons_traj, [0.01]

    def _fake_loader(
        mapping_type,
        adapter,
        *,
        problem_args,
        model_args,
        train_args,
        save_dir,
        retrain=False,
        runtime_device=None,
        runtime_dtype=None,
        ensure_results_save_freq=False,
    ):
        assert mapping_type == "inn"
        del adapter, problem_args, model_args, train_args, retrain, runtime_device, runtime_dtype, ensure_results_save_freq
        model_path = Path(save_dir) / "inn_mapping.pt"
        record_path = Path(save_dir) / "inn_training_record.npy"
        return {
            "model": torch.nn.Linear(3, 3),
            "training_record": {},
            "model_path": model_path,
            "record_path": record_path,
        }

    monkeypatch.setattr(jcc_methods, "_JCCINNPGDAdapter", _FakeAdapter)
    monkeypatch.setattr(jcc_methods, "prepare_learning_mapping", _fake_loader)
    monkeypatch.setattr(jcc_methods, "INNPGDOptimizer", _FakeOptimizer)

    result = jcc_methods._run_jcc_inn_pgd_variant(
        _FakeProblem(),
        output_dir=tmp_path,
        seed=7,
        max_iterations=1,
        retrain=False,
        num_test_instance=1,
        model_config={"Con_type": "PI"},
        optimizer_config={
            "learning_rate": 0.1,
            "min_lr": 1e-6,
            "initial_latent_mode": "center",
        },
        train_config={"n_samples": 4, "batch_size": 2, "total_iteration": 1},
    )

    assert result["status"] == "failed_nonfinite"
    assert result["failure_reason"] == "non-finite x_opt, objective"
    assert result["x_opt"] is None
    assert result["final_objective"] is None
    assert result["final_violation"] == 1.0
    assert result["chance_feasibility_rate"] == 0.0
    assert result["violation"] == pytest.approx(0.9)
    assert result["feasible"] is False


def test_jcc_compare_prepares_inn_before_baselines(monkeypatch, tmp_path):
    call_order = []

    def _solver_result(name):
        return {
            "num_bus": 30,
            "n_scenarios": 2,
            "solver": name,
            "status": "optimal",
            "runtime_sec": 0.0,
            "objective_value": 1.0,
            "objective": 1.0,
            "feasibility_rate": 1.0,
            "chance_satisfied": True,
            "chance_constraint_satisfied": True,
            "x_optimal": np.zeros(1),
        }

    def _row(route, method, objective=1.0):
        return {
            "row_kind": "instance",
            "route": route,
            "method": method,
            "objective": objective,
            "feasible": True,
            "violation": 0.0,
            "runtime": 0.0,
            "objective_mean": objective,
            "feasibility_rate": 1.0,
            "violation_mean": 0.0,
            "runtime_mean": 0.0,
            "runtime_total": 0.0,
        }

    def _fake_inn_pgd(problem, **kwargs):
        del kwargs
        call_order.append("INN-PGD")
        x_mid = ((problem.P_min + problem.P_max) / 2).detach().cpu().numpy()
        return {
            "x_opt": x_mid,
            "obj_traj": np.array([1.0]),
            "cons_traj": np.array([0.0]),
            "iter_time": [0.0],
            "total_wall_time": 0.0,
            "final_objective": 0.9,
            "final_violation": 0.0,
            "chance_feasibility_rate": 1.0,
            "violation": 0.0,
            "feasible": True,
        }

    def _fake_solver_suite(**kwargs):
        del kwargs
        call_order.append("exact-solvers")
        return (
            {
                "mixed_integer": _solver_result("mixed_integer"),
                "cvar": _solver_result("cvar"),
                "scenario": _solver_result("scenario"),
            },
            [],
            [_row("solver", "mixed_integer"), _row("solver", "cvar"), _row("solver", "scenario")],
        )

    def _fake_lagrangian(problem, initial_point, base_args, *, method, seed):
        del problem, initial_point, base_args, seed
        call_order.append(method)
        return {
            "x_opt": np.zeros(1),
            "iter_time": [0.0],
            "total_wall_time": 0.0,
            "final_objective": 1.1,
            "final_violation": 0.0,
            "feasible": True,
        }

    monkeypatch.setattr(jcc_opf_benchmarks, "_run_jcc_inn_pgd_variant", _fake_inn_pgd)
    monkeypatch.setattr(jcc_opf_benchmarks, "_run_jcc_solver_suite", _fake_solver_suite)
    monkeypatch.setattr(jcc_opf_benchmarks, "run_inn_lagrangian_baseline", _fake_lagrangian)

    inn_pgd.jcc_algorithm_comparison(
        n_scenarios=2,
        seed=7,
        run_inn_pgd=True,
        lagrangian_baselines=["ALM"],
        lagrangian_baseline_config=_jcc_lagrangian_baseline_config(),
        output_dir=tmp_path,
    )

    assert call_order == ["INN-PGD", "exact-solvers", "ALM"]


def test_jcc_baseline_solver_sweep_exposes_comparison_rows(monkeypatch, tmp_path):
    class _FakeProblem:
        def __init__(self, num_bus, n_scenarios):
            self.n_bus = int(num_bus)
            self.n_scenarios = int(n_scenarios)

    class _BaseSolver:
        payload = {}

        def __init__(self, problem, solver_config=None):
            del solver_config
            self.problem = problem

        def solve(self):
            return dict(type(self).payload)

    class _MISolver(_BaseSolver):
        payload = {
            "status": "optimal",
            "objective_value": 1.2,
            "feasibility_rate": 0.8,
            "chance_constraint_satisfied": False,
        }

    class _CVaRSolver(_BaseSolver):
        payload = {
            "status": "optimal",
            "objective_value": 0.9,
            "feasibility_rate": 0.92,
            "chance_constraint_satisfied": True,
        }

    class _ScenarioSolver(_BaseSolver):
        payload = {
            "status": "optimal",
            "objective_value": 1.0,
            "feasibility_rate": 0.91,
            "chance_constraint_satisfied": True,
        }

    monkeypatch.setattr(
        jcc_opf_benchmarks,
        "_build_jcc_problem",
        lambda **kwargs: _FakeProblem(kwargs["num_bus"], kwargs["n_scenarios"]),
    )
    monkeypatch.setattr(jcc_methods, "JCCDCOPFSolver", _MISolver)
    monkeypatch.setattr(jcc_methods, "JCCDCOPFCVaRSolver", _CVaRSolver)
    monkeypatch.setattr(jcc_methods, "JCCDCOPFRobustScenarioSolver", _ScenarioSolver)

    payload = inn_pgd.jcc_baseline_solver_sweep(
        num_bus_list=[30],
        n_scenarios=[2],
        epsilon=0.1,
        output_dir=tmp_path,
        verbose=False,
    )

    metrics = _payload_metrics(payload)
    assert len(metrics["comparison_rows"]) == 3
    assert len(metrics["comparison_summary_rows"]) == 3
    assert metrics["best_method"] == {"route": "solver", "method": "cvar"}
    assert payload["objective"] == 0.9
    assert payload["feasible"] is True
    assert metrics["all_records_feasible"] is False
    assert (tmp_path / payload["artifacts"]["baseline_metrics_json"]).exists()
    assert (tmp_path / payload["artifacts"]["baseline_metrics_csv"]).exists()
    assert (tmp_path / payload["artifacts"]["comparison_summary_json"]).exists()
    assert (tmp_path / payload["artifacts"]["comparison_summary_csv"]).exists()


def test_convex_algorithm_comparison_writes_summary_artifacts(tmp_path):
    result, output_dir = _run_workload('convex_ineq_compare_smoke', hom_alm.convex_algorithm_comparison, {'problem_type': 'socp', 'algorithms': ['Hom-PGD'], 'n_var': 4, 'n_linear_cons': 1, 'n_soc_cons': 1, 'n_qua_cons': 0, 'max_iterations': 3, 'max_running_time': 10, 'learning_rate': 0.01, 'stepsize_rule': 'constant', 'smooth': False}, tmp_path)
    assert result.metrics["problem_type"] == "socp"
    assert "Hom-PGD" in result.metrics["results"]
    assert result.metrics["instance_rows"] == result.metrics["comparison_rows"]
    assert result.metrics["method_summary_rows"] == result.metrics["comparison_summary_rows"]
    assert len(result.metrics["comparison_rows"]) == 1
    assert result.metrics["comparison_rows"][0]["route"] == "iterative"
    assert result.metrics["comparison_rows"][0]["method"] == "Hom-PGD"
    assert set(result.metrics["comparison_summary"]) == {"objective", "feasible", "best_method"}
    assert result.metrics["best_method"] == {"route": "iterative", "method": "Hom-PGD"}
    assert (output_dir / result.artifacts["records"]).exists()
    assert (output_dir / result.artifacts["summary"]).exists()


def test_convex_eq_comparison_includes_reference_solver(tmp_path):
    result, output_dir = _run_workload('eq_compare_smoke', hom_alm.convex_algorithm_comparison, {'problem_type': 'socp_eq', 'algorithms': ['Hom-ALM'], 'n_var': 4, 'n_linear_cons': 1, 'n_qua_cons': 1, 'n_soc_cons': 0, 'n_lin_eq': 1, 'max_iterations': 2, 'max_running_time': 10, 'learning_rate': 0.01, 'stepsize_rule': 'constant'}, tmp_path)
    assert result.metrics["problem_type"] == "socp_eq"
    assert any(row["route"] == "solver" and row["method"] == "ConvexSolver" for row in result.metrics["comparison_rows"])
    assert any(row["route"] == "iterative" and row["method"] == "Hom-ALM" for row in result.metrics["comparison_rows"])
    assert any(row["route"] == "solver" and row["method"] == "ConvexSolver" for row in result.metrics["comparison_summary_rows"])
    assert result.metrics["reference_solver_violation"] >= 0.0
    assert (output_dir / result.artifacts["records"]).exists()
    assert (output_dir / result.artifacts["summary"]).exists()


def test_convex_problem_type_rejects_unknown_aliases():
    with pytest.raises(ValueError, match="socp_eq"):
        convex_reference_common.normalize_convex_problem_type("eq")


def test_convex_eq_comparison_can_skip_reference_opt(monkeypatch):
    captured = {}

    def fake_prepare_reference(problem, **kwargs):
        captured["need_opt"] = kwargs["need_opt"]
        return {
            "x_opt": None,
            "x_origin": np.zeros(problem.nvar, dtype=float),
            "objective_opt": None,
            "solver_violation": None,
            "solver_time": None,
            "ip_solver_time": 0.0,
            "hom_origin_constraints": kwargs["hom_origin_constraints"],
            "x_origin_eq_violation": 0.0,
            "hom_map": None,
        }

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del name, problem, params, hom_map, init_point
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 4)),
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    result = hom_alm.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["ALM"],
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=1,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=2,
        max_running_time=10,
        learning_rate=1e-2,
        stepsize_rule="constant",
        reference_need_opt=False,
        include_reference_solver=False,
    )

    assert captured["need_opt"] is False
    metrics = _payload_metrics(result)
    assert metrics["reference_objective"] is None
    assert not any(row["route"] == "solver" for row in metrics["comparison_rows"])


def test_convex_eq_include_reference_solver_forces_reference_opt(monkeypatch, capsys):
    captured = {}

    def fake_prepare_reference(problem, **kwargs):
        captured["need_opt"] = kwargs["need_opt"]
        return {
            "x_opt": np.zeros(problem.nvar, dtype=float),
            "x_origin": np.zeros(problem.nvar, dtype=float),
            "objective_opt": 0.0,
            "solver_violation": 0.0,
            "solver_time": 0.01,
            "ip_solver_time": 0.0,
            "hom_origin_constraints": kwargs["hom_origin_constraints"],
            "x_origin_eq_violation": 0.0,
            "hom_map": None,
        }

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del name, problem, params, hom_map, init_point
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 4)),
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    result = hom_alm.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["ALM"],
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=1,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=2,
        max_running_time=10,
        learning_rate=1e-2,
        stepsize_rule="constant",
        reference_need_opt=False,
        include_reference_solver=True,
        verbose=True,
    )

    assert captured["need_opt"] is True
    captured_output = capsys.readouterr().out
    assert "Preparing ConvexSolver reference" in captured_output
    assert "Computed ConvexSolver reference" in captured_output
    assert any(
        row["route"] == "solver" and row["method"] == "ConvexSolver"
        for row in _payload_metrics(result)["comparison_rows"]
    )


def test_convex_eq_comparison_exposes_alm_eq_baseline_config(monkeypatch):
    captured = {}

    def fake_prepare_reference(problem, **kwargs):
        return {
            "x_opt": None,
            "x_origin": np.zeros(problem.nvar, dtype=float),
            "objective_opt": None,
            "solver_violation": None,
            "solver_time": None,
            "ip_solver_time": 0.0,
            "hom_origin_constraints": kwargs["hom_origin_constraints"],
            "x_origin_eq_violation": 0.0,
            "hom_map": None,
        }

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del problem, hom_map, init_point
        captured.setdefault("names", []).append(name)
        captured["params"] = params
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 4)),
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    result = hom_alm.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["Penalty-EQ", "Prox-Penalty-EQ", "ALM-EQ", "Prox-ALM-EQ"],
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=1,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=2,
        max_running_time=10,
        learning_rate=1e-2,
        stepsize_rule="constant",
        reference_need_opt=False,
        include_reference_solver=False,
    )

    alm_eq = captured["params"]["ALM-EQ"]
    penalty_eq = captured["params"]["Penalty-EQ"]
    prox_penalty_eq = captured["params"]["Prox-Penalty-EQ"]
    prox_alm_eq = captured["params"]["Prox-ALM-EQ"]
    assert captured["names"] == ["Penalty-EQ", "Prox-Penalty-EQ", "ALM-EQ", "Prox-ALM-EQ"]
    assert penalty_eq["opt_type"] == "Penalty-EQ"
    assert penalty_eq["use_lagrangian"] is False
    assert penalty_eq["use_penalty"] is True
    assert penalty_eq["use_proximal"] is False
    assert prox_penalty_eq["opt_type"] == "Prox-Penalty-EQ"
    assert prox_penalty_eq["use_lagrangian"] is False
    assert prox_penalty_eq["use_penalty"] is True
    assert prox_penalty_eq["use_proximal"] is True
    assert alm_eq["opt_type"] == "ALM-EQ"
    assert alm_eq["outer_iterations"] == 2
    assert alm_eq["use_lagrangian"] is True
    assert alm_eq["use_penalty"] is True
    assert alm_eq["use_proximal"] is False
    assert prox_alm_eq["opt_type"] == "Prox-ALM-EQ"
    assert prox_alm_eq["use_lagrangian"] is True
    assert prox_alm_eq["use_penalty"] is True
    assert prox_alm_eq["use_proximal"] is True
    assert _payload_metrics(result)["algorithms"] == ["Penalty-EQ", "Prox-Penalty-EQ", "ALM-EQ", "Prox-ALM-EQ"]


def test_convex_prox_alm_aliases_have_distinct_proximal_flags(monkeypatch):
    captured = {}

    def fake_prepare_reference(problem, **kwargs):
        return {
            "x_opt": None,
            "x_origin": np.zeros(problem.nvar, dtype=float),
            "objective_opt": None,
            "solver_violation": None,
            "solver_time": None,
            "ip_solver_time": 0.0,
            "hom_origin_constraints": kwargs["hom_origin_constraints"],
            "x_origin_eq_violation": 0.0,
            "hom_map": None,
        }

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del problem, hom_map, init_point
        captured.setdefault("names", []).append(name)
        captured["params"] = params
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 4)),
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    result = hom_alm.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["Penalty", "Prox-Penalty", "ALM", "Prox-ALM", "Hom-ALM", "Prox-Hom-ALM"],
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=0,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=5,
        common_config={"learning_rate": 2e-3},
        outer_common={
            "convergence_threshold": 2e-6,
        },
        inner_solver_common={
            "inner_iterations": 3,
            "inner_learning_rate": 5e-4,
        },
        reference_need_opt=False,
        include_reference_solver=False,
    )

    assert captured["names"] == ["Penalty", "Prox-Penalty", "ALM", "Prox-ALM", "Hom-ALM", "Prox-Hom-ALM"]
    assert captured["params"]["Penalty"]["use_lagrangian"] is False
    assert captured["params"]["Penalty"]["use_penalty"] is True
    assert captured["params"]["Penalty"]["use_proximal"] is False
    assert captured["params"]["Prox-Penalty"]["use_lagrangian"] is False
    assert captured["params"]["Prox-Penalty"]["use_penalty"] is True
    assert captured["params"]["Prox-Penalty"]["use_proximal"] is True
    assert captured["params"]["ALM"]["use_proximal"] is False
    assert captured["params"]["Prox-ALM"]["use_proximal"] is True
    assert captured["params"]["Hom-ALM"]["use_proximal"] is False
    assert captured["params"]["Prox-Hom-ALM"]["use_proximal"] is True
    assert captured["params"]["Prox-ALM"]["opt_type"] == "ALM"
    assert captured["params"]["Prox-Hom-ALM"]["opt_type"] == "Hom-ALM"
    for name in ("Penalty", "Prox-Penalty", "ALM", "Prox-ALM", "Hom-ALM", "Prox-Hom-ALM"):
        assert captured["params"][name]["learning_rate"] == pytest.approx(2e-3)
        assert captured["params"][name]["outer_iterations"] == 5
        assert captured["params"][name]["convergence_threshold"] == pytest.approx(2e-6)
        assert captured["params"][name]["inner_iterations"] == 3
        assert captured["params"][name]["inner_learning_rate"] == pytest.approx(5e-4)
    assert _payload_metrics(result)["algorithms"] == ["Penalty", "Prox-Penalty", "ALM", "Prox-ALM", "Hom-ALM", "Prox-Hom-ALM"]


def test_prepare_convex_reference_context_supports_ineq_only_origin(monkeypatch):
    config = convex_reference_common.build_convex_problem_config(
        seed=2025,
        n_var=4,
        n_linear_cons=1,
        n_soc_cons=0,
        n_qua_cons=1,
        n_lin_eq=1,
    )
    problem = convex_reference_common.build_convex_problem(
        config=config,
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        problem_type="socp_eq",
    )
    calls = []

    def fake_solve_exact_result(solver, solve_type, **kwargs):
        del solver
        calls.append((solve_type, dict(kwargs)))
        return {
            "solution": np.zeros(problem.nvar, dtype=float),
            "objective": 0.0 if solve_type == "opt" else None,
            "runtime_total": 0.0,
            "violation": 0.0,
        }

    monkeypatch.setattr(convex_reference_common, "solve_exact_result", fake_solve_exact_result)
    reference = convex_reference_common.prepare_convex_reference_context(
        problem,
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        hom_origin_constraints="ineq",
    )

    assert calls[0] == ("opt", {})
    assert calls[1][0] == "ip"
    assert calls[1][1]["equality"] is False
    assert reference["hom_origin_constraints"] == "ineq"
    assert reference["x_origin_eq_violation"] is not None


def test_prepare_convex_reference_context_supports_geometric_origin(monkeypatch):
    config = convex_reference_common.build_convex_problem_config(
        seed=2025,
        n_var=4,
        n_linear_cons=1,
        n_soc_cons=0,
        n_qua_cons=1,
        n_lin_eq=1,
    )
    problem = convex_reference_common.build_convex_problem(
        config=config,
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        problem_type="socp_eq",
    )
    calls = []

    def fake_solve_exact_result(solver, solve_type, **kwargs):
        del solver
        calls.append((solve_type, dict(kwargs)))
        return {
            "solution": np.zeros(problem.nvar, dtype=float),
            "objective": 0.0 if solve_type == "opt" else None,
            "runtime_total": 0.0,
            "violation": 0.0,
        }

    monkeypatch.setattr(convex_reference_common, "solve_exact_result", fake_solve_exact_result)
    reference = convex_reference_common.prepare_convex_reference_context(
        problem,
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        hom_origin_constraints="ineq",
        ip_mode="geometric_central_ip",
    )

    assert calls[0] == ("opt", {})
    assert calls[1][0] == "geometric_central_ip"
    assert calls[1][1]["equality"] is False
    assert reference["origin_method"] == "geometric_central_ip"


def test_prepare_convex_reference_context_caches_solver_and_origin(monkeypatch, tmp_path):
    config = convex_reference_common.build_convex_problem_config(
        seed=2025,
        n_var=4,
        n_linear_cons=0,
        n_soc_cons=0,
        n_qua_cons=0,
        n_lin_eq=0,
        x_lower=-1.0,
        x_upper=1.0,
    )
    problem = convex_reference_common.build_convex_problem(
        config=config,
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        problem_type="socp",
    )
    calls = []

    def fake_solve_exact_result(solver, solve_type, **kwargs):
        del solver
        calls.append((solve_type, dict(kwargs)))
        if solve_type == "opt":
            return {
                "solution": np.full(problem.nvar, 0.25, dtype=float),
                "objective": 1.5,
                "runtime_total": 3.0,
                "violation": 0.0,
            }
        return {
            "solution": np.zeros(problem.nvar, dtype=float),
            "objective": None,
            "runtime_total": 4.0,
            "violation": 0.0,
        }

    monkeypatch.setattr(convex_reference_common, "solve_exact_result", fake_solve_exact_result)
    first = convex_reference_common.prepare_convex_reference_context(
        problem,
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        output_dir=tmp_path,
    )
    second = convex_reference_common.prepare_convex_reference_context(
        problem,
        runtime_device="cpu",
        runtime_dtype=torch.float32,
        output_dir=tmp_path,
    )

    assert [call[0] for call in calls] == ["opt", "ip"]
    assert first["reference_cache_hit"] is False
    assert second["reference_cache_hit"] is True
    assert np.allclose(second["x_opt"], np.full(problem.nvar, 0.25))
    assert np.allclose(second["x_origin"], np.zeros(problem.nvar))
    assert second["solver_time"] == pytest.approx(0.0)
    assert second["ip_solver_time"] == pytest.approx(0.0)
    assert second["cached_solver_time"] == pytest.approx(3.0)
    assert second["cached_ip_solver_time"] == pytest.approx(4.0)
    assert Path(second["reference_cache_path"]).exists()


def test_convex_common_config_refreshes_derived_solver_configs(monkeypatch):
    captured = {}

    def fake_prepare_reference(problem, **kwargs):
        return {
            "x_opt": None,
            "x_origin": np.zeros(problem.nvar, dtype=float),
            "objective_opt": None,
            "solver_violation": None,
            "solver_time": None,
            "ip_solver_time": 0.0,
            "hom_origin_constraints": kwargs["hom_origin_constraints"],
            "x_origin_eq_violation": 0.0,
            "hom_map": None,
        }

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del name, problem, hom_map, init_point
        captured["params"] = params
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 4)),
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    hom_alm.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["PGD"],
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=1,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=2,
        max_running_time=10,
        learning_rate=1e-2,
        stepsize_rule="adaptive",
        common_config={
            "stepsize_rule": "constant",
            "lr_decay": 0.5,
            "max_running_time": 12,
            "verbose_interval": 7,
        },
        reference_need_opt=False,
        include_reference_solver=False,
    )

    params = captured["params"]
    assert params["common"]["stepsize_rule"] == "constant"
    assert params["common"]["lr_decay"] == pytest.approx(0.5)
    assert params["common"]["max_running_time"] == 12
    pgd_subproblem = params["PGD"]["projection_subproblem"]
    fw_subproblem = params["FW"]["linearization_subproblem"]
    for subproblem in (pgd_subproblem, fw_subproblem):
        assert subproblem["outer_stepsize_rule"] == "constant"
        assert subproblem["outer_lr_decay"] == pytest.approx(0.5)
        assert subproblem["inner_lr_decay"] == pytest.approx(0.5)
        assert subproblem["max_running_time"] == 12
    assert params["ALM"]["outer_stepsize_rule"] == "constant"
    assert params["ALM"]["outer_lr_decay"] == pytest.approx(0.5)
    assert params["ALM"]["inner_lr_decay"] == pytest.approx(0.5)
    assert params["ALM"]["verbose_interval"] == 7
    assert params["Hom-ALM"]["outer_stepsize_rule"] == "constant"
    assert params["Hom-ALM"]["outer_lr_decay"] == pytest.approx(0.5)
    assert params["Hom-ALM"]["inner_lr_decay"] == pytest.approx(0.5)
    assert params["Hom-ALM"]["verbose_interval"] == 7


def test_convex_common_config_controls_hom_map_and_initial_point_mode(monkeypatch):
    captured = {}
    x_origin = np.ones(4, dtype=float)

    def fake_prepare_reference(problem, **kwargs):
        del problem
        captured["reference_kwargs"] = kwargs
        return {
            "x_opt": np.zeros(4, dtype=float),
            "x_origin": x_origin,
            "objective_opt": None,
            "solver_violation": None,
            "solver_time": None,
            "ip_solver_time": 0.0,
            "hom_origin_constraints": kwargs["hom_origin_constraints"],
            "x_origin_eq_violation": 0.0,
            "hom_map": None,
        }

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del name, problem, params, hom_map
        captured["init_point"] = init_point
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 4)),
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    hom_alm.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["Hom-ALM"],
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=0,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=2,
        common_config={
            "smooth": True,
            "initial_point_mode": "gauge_center",
        },
        smooth=False,
        reference_need_opt=False,
        include_reference_solver=False,
    )

    assert captured["reference_kwargs"]["smooth"] is True
    assert np.array_equal(captured["init_point"], x_origin)


def test_convex_initial_point_mode_random_is_shared_across_algorithms(monkeypatch):
    captured = []
    x_origin = np.ones(3, dtype=float)

    def fake_prepare_reference(problem, **kwargs):
        del problem
        return {
            "x_opt": np.zeros(3, dtype=float),
            "x_origin": x_origin,
            "objective_opt": None,
            "solver_violation": None,
            "solver_time": None,
            "ip_solver_time": 0.0,
            "hom_origin_constraints": kwargs["hom_origin_constraints"],
            "x_origin_eq_violation": 0.0,
            "hom_map": None,
        }

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del problem, params, hom_map
        captured.append((name, np.asarray(init_point, dtype=float).copy()))
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 3)),
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    hom_alm.convex_algorithm_comparison(
        problem_type="socp",
        algorithms=["PGD", "RD", "Hom-PGD"],
        n_var=3,
        n_linear_cons=1,
        n_qua_cons=0,
        n_soc_cons=0,
        max_iterations=2,
        common_config={"initial_point_mode": "random"},
        reference_need_opt=False,
        include_reference_solver=False,
    )

    assert [name for name, _ in captured] == ["PGD", "RD", "Hom-PGD"]
    assert all(np.array_equal(point, captured[0][1]) for _, point in captured)
    assert not np.array_equal(captured[0][1].reshape(-1), x_origin)


def test_convex_inner_solver_common_does_not_override_outer_lr_decay(monkeypatch):
    captured = {}

    def fake_prepare_reference(problem, **kwargs):
        return {
            "x_opt": None,
            "x_origin": np.zeros(problem.nvar, dtype=float),
            "objective_opt": None,
            "solver_violation": None,
            "solver_time": None,
            "ip_solver_time": 0.0,
            "hom_origin_constraints": kwargs["hom_origin_constraints"],
            "x_origin_eq_violation": 0.0,
            "hom_map": None,
        }

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del problem, hom_map, init_point
        captured.setdefault("names", []).append(name)
        captured["params"] = params
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 4)),
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    hom_alm.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["ALM", "Hom-ALM"],
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=1,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=2,
        max_running_time=10,
        learning_rate=1e-2,
        stepsize_rule="adaptive",
        common_config={
            "learning_rate": 2e-3,
            "lr_decay": 0.998,
            "stepsize_rule": "adaptive",
        },
        outer_common={
            "convergence_threshold": 1e-6,
        },
        inner_solver_common={
            "inner_iterations": 3,
            "inner_learning_rate": 5e-4,
            "inner_lr_decay": 0.99,
        },
        reference_need_opt=False,
        include_reference_solver=False,
    )

    params = captured["params"]
    for name in ("ALM", "Hom-ALM"):
        assert params[name]["learning_rate"] == pytest.approx(2e-3)
        assert params[name]["inner_learning_rate"] == pytest.approx(5e-4)
        assert params[name]["outer_stepsize_rule"] == "adaptive"
        assert params[name]["outer_lr_decay"] == pytest.approx(0.998)
        assert params[name]["inner_lr_decay"] == pytest.approx(0.99)


def test_convex_visualization_accepts_plot_order_and_labels(monkeypatch, tmp_path):
    captured = {}

    def fake_prepare_reference(problem, **kwargs):
        del problem, kwargs
        return {
            "x_opt": np.zeros((1, 4)),
            "x_origin": np.zeros(4, dtype=float),
            "objective_opt": 0.0,
            "solver_violation": 0.0,
            "solver_time": 0.0,
            "ip_solver_time": 0.0,
            "hom_origin_constraints": "inequality",
            "x_origin_eq_violation": 0.0,
            "hom_map": None,
        }

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del name, problem, params, hom_map, init_point
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 4)),
        }

    def fake_save_comparison_visualizations(**kwargs):
        captured.update(kwargs)
        return {"runtime_summary": "artifacts/runtime.pdf"}

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)
    monkeypatch.setattr(convex_ineq_benchmarks, "save_comparison_visualizations", fake_save_comparison_visualizations)

    result = hom_alm.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["Prox-Hom-ALM", "Prox-ALM"],
        plot_algorithm_order=["Prox-ALM", "Prox-Hom-ALM"],
        method_labels=CONVEX_METHOD_LABELS,
        reference_label="MOSEK",
        scale_label="tiny_eq",
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=0,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=2,
        common_config={"convergence_threshold": 1e-4},
        reference_need_opt=True,
        include_reference_solver=False,
        visualize=True,
        output_dir=tmp_path,
    )

    assert captured["algorithms"] == ["Prox-ALM", "Prox-Hom-ALM"]
    assert captured["prefix"] == "socp_eq_tiny_eq"
    assert captured["method_labels"] == CONVEX_METHOD_LABELS
    assert captured["reference_label"] == "MOSEK"
    assert captured["violation_y_min"] == pytest.approx(1e-4)
    metrics = _payload_metrics(result)
    assert metrics["method_labels"] == CONVEX_METHOD_LABELS
    assert metrics["reference_label"] == "MOSEK"
    assert metrics["scale_label"] == "tiny_eq"


def test_convex_visualize_only_reuses_existing_records(monkeypatch, tmp_path):
    artifact_dir = artifact_helpers.artifact_root(tmp_path)
    records_dir = artifact_dir / "records"
    records_dir.mkdir()
    records = {
        "Prox-Hom-ALM": {
            "obj_traj": [-1.0, -2.0],
            "cons_traj": [1e-2, 1e-6],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 4)),
        }
    }
    summaries = {
        "Prox-Hom-ALM": {
            "final_objective": -2.0,
            "final_violation": 1e-6,
            "record_path": "artifacts/records/Prox-Hom-ALM.npy",
        }
    }
    np.save(records_dir / "Prox-Hom-ALM.npy", records["Prox-Hom-ALM"], allow_pickle=True)
    (artifact_dir / "summary.json").write_text(json.dumps(summaries), encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": artifact_helpers.COMPARISON_STORAGE_VERSION,
                "algorithms": ["Prox-Hom-ALM"],
                "records": {"Prox-Hom-ALM": "artifacts/records/Prox-Hom-ALM.npy"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "name": "convex_eq_compare_tiny_eq",
                "objective": -2.0,
                "feasible": True,
                "artifacts": {"records": "artifacts/records"},
                "metrics": {
                    "algorithms": ["Prox-Hom-ALM"],
                    "requested_algorithms": ["Prox-Hom-ALM"],
                    "reference_objective": -2.1,
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
        return {"runtime_summary": "artifacts/socp_eq_runtime_summary.pdf"}

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("visualize_only must not rerun reference or algorithms")

    monkeypatch.setattr(convex_ineq_benchmarks, "save_comparison_visualizations", fake_save_comparison_visualizations)
    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fail_if_called)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fail_if_called)

    payload = convex_ineq_benchmarks.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["Prox-Hom-ALM"],
        plot_algorithm_order=["Prox-Hom-ALM"],
        method_labels=CONVEX_METHOD_LABELS,
        reference_label="MOSEK",
        scale_label="tiny_eq",
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=0,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=2,
        common_config={"convergence_threshold": 1e-4},
        visualize=True,
        visualize_only=True,
        output_dir=tmp_path,
    )

    assert _payload_metrics(payload)["visualization_only"] is True
    assert payload["artifacts"]["runtime_summary"] == "artifacts/socp_eq_runtime_summary.pdf"
    assert set(captured["records"]) == {"Prox-Hom-ALM"}
    assert captured["algorithms"] == ["Prox-Hom-ALM"]
    assert captured["reference_objective"] == pytest.approx(-2.1)
    assert captured["violation_y_min"] == pytest.approx(1e-4)


def test_incremental_comparison_storage_updates_per_algorithm(tmp_path):
    first_records = {
        "PGD": {
            "obj_traj": [1.0, 0.5],
            "cons_traj": [1e-2, 1e-4],
            "iter_time": [0.1, 0.2],
        }
    }
    first_summaries = {
        "PGD": {
            "final_objective": 0.5,
            "final_violation": 1e-4,
            "total_iter_time": 0.3,
            "iterations": 2,
            "long_debug_traj": list(range(32)),
        }
    }
    artifact_helpers.save_incremental_comparison_artifacts(
        tmp_path,
        records=first_records,
        summaries=first_summaries,
        algorithm_order=["PGD"],
        manifest_metadata={"scale_label": "tiny_eq"},
    )
    second_records = {
        "Prox-Hom-ALM": {
            "obj_traj": [0.9, 0.2],
            "cons_traj": [1e-3, 1e-6],
            "iter_time": [0.05, 0.05],
        }
    }
    second_summaries = {
        "Prox-Hom-ALM": {
            "final_objective": 0.2,
            "final_violation": 1e-6,
            "total_iter_time": 0.1,
            "iterations": 2,
        }
    }
    artifacts, views, merged_records, merged_summaries = artifact_helpers.save_incremental_comparison_artifacts(
        tmp_path,
        records=second_records,
        summaries=second_summaries,
        algorithm_order=["Prox-Hom-ALM"],
    )

    assert artifacts["records_dir"] == "artifacts/records"
    assert set(merged_records) == {"PGD", "Prox-Hom-ALM"}
    assert set(merged_summaries) == {"PGD", "Prox-Hom-ALM"}
    assert "long_debug_traj" not in merged_summaries["PGD"]
    assert views["best_method"]["method"] == "Prox-Hom-ALM"
    assert (tmp_path / "artifacts" / "records" / "PGD.npy").exists()
    assert (tmp_path / "artifacts" / "records" / "Prox-Hom-ALM.npy").exists()

    loaded_records, loaded_summaries, _, manifest = artifact_helpers.load_incremental_comparison_artifacts(
        tmp_path,
        algorithms=["PGD", "Prox-Hom-ALM"],
    )
    assert set(loaded_records) == {"PGD", "Prox-Hom-ALM"}
    assert set(loaded_summaries) == {"PGD", "Prox-Hom-ALM"}
    assert manifest["algorithms"] == ["PGD", "Prox-Hom-ALM"]


def test_incremental_comparison_storage_rejects_stale_manifest_schema(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "manifest.json").write_text(
        json.dumps({"version": 0, "algorithms": [], "records": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported comparison artifact schema"):
        artifact_helpers.load_incremental_comparison_artifacts(tmp_path)
    with pytest.raises(ValueError, match="Unsupported comparison artifact schema"):
        artifact_helpers.save_incremental_comparison_artifacts(
            tmp_path,
            records={},
            summaries={},
        )


def test_convex_alm_algorithm_config_is_normalized_by_algorithm_type(monkeypatch):
    captured = {}

    def fake_prepare_reference(problem, **kwargs):
        return {
            "x_opt": None,
            "x_origin": np.zeros(problem.nvar, dtype=float),
            "objective_opt": None,
            "solver_violation": None,
            "solver_time": None,
            "ip_solver_time": 0.0,
            "hom_origin_constraints": kwargs["hom_origin_constraints"],
            "x_origin_eq_violation": 0.0,
            "hom_map": None,
        }

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del problem, hom_map, init_point
        captured.setdefault("names", []).append(name)
        captured["params"] = params
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
            "x_traj": [],
            "x_solved": np.zeros((1, 4)),
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)
    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    hom_alm.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["ALM", "ALM-EQ", "Hom-ALM"],
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=1,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=2,
        max_running_time=10,
        learning_rate=1e-2,
        stepsize_rule="adaptive",
        common_config={
            "learning_rate": 2e-3,
            "stepsize_rule": "adaptive",
            "lr_decay": 0.998,
        },
        outer_common={
            "first_order_lagrangian_gap_threshold": 1e-6,
            "solver_verbose": True,
        },
        algorithm_config={
            "Hom-ALM": {"stepsize_rule": "diminish", "lr_decay": 0.95},
            "ALM-EQ": {"stepsize_rule": "diminish", "lr_decay": 0.95, "solver_verbose": False},
        },
        reference_need_opt=False,
        include_reference_solver=False,
    )

    params = captured["params"]
    assert params["ALM"]["learning_rate"] == pytest.approx(2e-3)
    assert params["ALM"]["outer_iterations"] == 2
    assert params["ALM"]["outer_stepsize_rule"] == "adaptive"
    assert params["ALM"]["outer_lr_decay"] == pytest.approx(0.998)
    assert params["Hom-ALM"]["outer_iterations"] == 2
    assert params["Hom-ALM"]["outer_stepsize_rule"] == "diminish"
    assert params["Hom-ALM"]["outer_lr_decay"] == pytest.approx(0.95)
    assert "stepsize_rule" not in params["ALM-EQ"]
    assert "lr_decay" not in params["ALM-EQ"]
    assert "learning_rate" not in params["ALM-EQ"]
    assert "first_order_lagrangian_gap_threshold" not in params["ALM-EQ"]
    assert params["ALM"]["first_order_lagrangian_gap_threshold"] == pytest.approx(1e-6)
    assert params["ALM-EQ"]["solver_verbose"] is False
    assert params["Penalty-EQ"]["solver_verbose"] is True
    assert "solver_verbose" not in params["ALM"]


def test_convex_alm_config_outer_lr_keys_are_rejected(monkeypatch):
    def fake_prepare_reference(problem, **kwargs):
        del problem, kwargs
        raise AssertionError("reference context should not be built after config validation fails")

    monkeypatch.setattr(convex_ineq_benchmarks, "prepare_convex_reference_context", fake_prepare_reference)

    with pytest.raises(ValueError, match="Unsupported outer_common keys"):
        hom_alm.convex_algorithm_comparison(
            problem_type="socp_eq",
            algorithms=["ALM"],
            n_var=4,
            n_linear_cons=1,
            n_qua_cons=1,
            n_soc_cons=0,
            n_lin_eq=1,
            max_iterations=2,
            max_running_time=10,
            outer_common={"outer_lr_decay": 0.9},
            reference_need_opt=False,
            include_reference_solver=False,
        )

    with pytest.raises(ValueError, match="Unsupported common_config keys"):
        hom_alm.convex_algorithm_comparison(
            problem_type="socp_eq",
            algorithms=["ALM"],
            n_var=4,
            n_linear_cons=1,
            n_qua_cons=1,
            n_soc_cons=0,
            n_lin_eq=1,
            max_iterations=2,
            max_running_time=10,
            common_config={"outer_lr_decay": 0.9},
            reference_need_opt=False,
            include_reference_solver=False,
        )

    with pytest.raises(ValueError, match="Unsupported inner_solver_common keys"):
        hom_alm.convex_algorithm_comparison(
            problem_type="socp_eq",
            algorithms=["ALM"],
            n_var=4,
            n_linear_cons=1,
            n_qua_cons=1,
            n_soc_cons=0,
            n_lin_eq=1,
            max_iterations=2,
            max_running_time=10,
            inner_solver_common={"learning_rate": 1e-3},
            reference_need_opt=False,
            include_reference_solver=False,
        )

    with pytest.raises(ValueError, match="Unsupported Hom-ALM config keys"):
        hom_alm.convex_algorithm_comparison(
            problem_type="socp_eq",
            algorithms=["Hom-ALM"],
            n_var=4,
            n_linear_cons=1,
            n_qua_cons=1,
            n_soc_cons=0,
            n_lin_eq=1,
            max_iterations=2,
            max_running_time=10,
            algorithm_config={"Hom-ALM": {"outer_stepsize_rule": "adaptive"}},
            reference_need_opt=False,
            include_reference_solver=False,
        )

    with pytest.raises(ValueError, match="outer_common must not set"):
        hom_alm.convex_algorithm_comparison(
            problem_type="socp_eq",
            algorithms=["ALM"],
            n_var=4,
            n_linear_cons=1,
            n_qua_cons=1,
            n_soc_cons=0,
            n_lin_eq=1,
            max_iterations=2,
            max_running_time=10,
            outer_common={"outer_iterations": 3},
            reference_need_opt=False,
            include_reference_solver=False,
        )

    with pytest.raises(ValueError, match="algorithm_config for ALM must not set"):
        hom_alm.convex_algorithm_comparison(
            problem_type="socp_eq",
            algorithms=["ALM"],
            n_var=4,
            n_linear_cons=1,
            n_qua_cons=1,
            n_soc_cons=0,
            n_lin_eq=1,
            max_iterations=2,
            max_running_time=10,
            algorithm_config={"ALM": {"outer_iterations": 3}},
            reference_need_opt=False,
            include_reference_solver=False,
        )


def test_convex_eq_comparison_exposes_hom_alm_penalty_config(monkeypatch):
    captured = {}

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del problem, hom_map, init_point
        captured.setdefault("names", []).append(name)
        captured["params"] = params
        return {
            "obj_traj": [1.0],
            "cons_traj": [123.0],
            "iter_time": [0.01],
            "x_traj": [],
            "last_trans_time": 0.0,
            "x_solved": np.zeros((1, 4)),
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    result = hom_alm.convex_algorithm_comparison(
        problem_type="socp_eq",
        algorithms=["ALM", "Hom-ALM"],
        n_var=4,
        n_linear_cons=1,
        n_qua_cons=1,
        n_soc_cons=0,
        n_lin_eq=1,
        max_iterations=2,
        max_running_time=10,
        learning_rate=1e-2,
        stepsize_rule="constant",
    )

    alm = captured["params"]["ALM"]
    hom_alm_params = captured["params"]["Hom-ALM"]
    assert captured["names"] == ["ALM", "Hom-ALM"]
    assert alm["inner_iterations"] == hom_alm_params["inner_iterations"] == 50
    assert alm["inner_solver"] == hom_alm_params["inner_solver"] == "gd"
    assert alm["inner_proximal_update_iterations"] == hom_alm_params["inner_proximal_update_iterations"] == 1
    assert alm["inner_proximal_coef"] == hom_alm_params["inner_proximal_coef"] == 0.0
    assert alm["dual_learning_rate"] == hom_alm_params["dual_learning_rate"] == pytest.approx(1e-1)
    assert alm["penalty_coef"] == hom_alm_params["penalty_coef"] == 10.0
    assert alm["penalty_growth"] == hom_alm_params["penalty_growth"] == 1.1
    assert alm["proximal_coef"] == hom_alm_params["proximal_coef"] == 0.1
    assert alm["proximal_space"] == "x"
    assert alm["inner_proximal_space"] == "x"
    assert alm["acceleration_space"] == "x"
    assert hom_alm_params["inner_iterations"] == 50
    assert hom_alm_params["proximal_space"] == "z"
    assert hom_alm_params["inner_proximal_space"] == "z"
    assert hom_alm_params["acceleration_space"] == "z"
    assert hom_alm_params["hom_map_gradient"] == "explicit"
    assert hom_alm_params["outer_stepsize_rule"] == "constant"
    assert hom_alm_params["inner_stepsize_rule"] == "constant"
    assert hom_alm_params["max_penalty"] == 1e2
    assert hom_alm_params["max_dual"] == 1e2
    metrics = _payload_metrics(result)
    summary = metrics["results"]["Hom-ALM"]
    assert summary["optimizer_reported_final_objective"] == pytest.approx(1.0)
    assert summary["optimizer_reported_final_violation"] == pytest.approx(123.0)
    assert summary["final_violation"] == pytest.approx(summary["final_full_violation"])
    assert "final_equality_violation" in summary
    row = next(row for row in metrics["comparison_rows"] if row["route"] == "iterative")
    assert row["violation"] == pytest.approx(row["full_violation"])
    assert row["optimizer_reported_violation"] == pytest.approx(123.0)


def test_maxcut_algorithm_comparison_writes_summary_artifacts(tmp_path):
    result, output_dir = _run_workload('maxcut_compare_smoke', hom_pgd.maxcut_algorithm_comparison, {'algorithms': ['Hom-PGD'], 'n': 6, 'alpha': 0.5, 'max_iterations': 2, 'max_running_time': 10}, tmp_path)
    assert result.metrics["n"] == 6
    assert "Hom-PGD" in result.metrics["results"]
    assert len(result.metrics["comparison_rows"]) == 1
    assert result.metrics["comparison_rows"][0]["route"] == "iterative"
    assert result.metrics["comparison_rows"][0]["method"] == "Hom-PGD"
    assert set(result.metrics["comparison_summary"]) == {"objective", "feasible", "best_method"}
    assert result.metrics["best_method"] == {"route": "iterative", "method": "Hom-PGD"}
    assert (output_dir / result.artifacts["records"]).exists()
    assert (output_dir / result.artifacts["summary"]).exists()


def test_learning_route_summary_runs_through_script_record(tmp_path):
    result, output_dir = _run_workload('learning_route_smoke', examples.learning_route_summary, {'seed': 7, 'n_var': 2, 'n_samples': 8, 'prediction_value': 0.0}, tmp_path)
    assert result.metrics["predictor"] == "constant"
    assert result.metrics["refiner"] == "projection"
    assert 0.0 <= result.metrics["raw_feasibility_rate"] <= 1.0
    assert result.metrics["refined_feasibility_rate"] >= result.metrics["raw_feasibility_rate"]
    assert (output_dir / "result.json").exists()


def test_qcqp_inn_experiment_uses_shared_best_case_selection(monkeypatch, tmp_path):
    phase_order = []

    def _fake_prepare(case_context):
        del case_context
        phase_order.append("train")

    def _fake_qcqp_inn_comparison(output_dir=None, **kwargs):
        del output_dir
        phase_order.append("test")
        assert "ipopt_solver_name" not in kwargs
        assert "ipopt_options" not in kwargs
        n_var = int(kwargs["n_var"])
        n_qua_cons = int(kwargs["n_qua_cons"])
        mapping = {
            (10, 10): {"objective": -1.0, "feasible": False, "final_violation": 1e-1, "iterations": 5},
            (20, 20): {"objective": -0.5, "feasible": True, "final_violation": 0.0, "iterations": 5},
            (30, 30): {"objective": -0.8, "feasible": True, "final_violation": 0.0, "iterations": 5},
        }
        return mapping[(n_var, n_qua_cons)]

    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "_prepare_qcqp_inn_training_context", _fake_prepare)
    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "qcqp_inn_comparison", _fake_qcqp_inn_comparison)
    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "_print_qcqp_instance_metrics", lambda *args, **kwargs: None)

    payload = inn_pgd.qcqp_inn_experiment(
        **_qcqp_inn_params(
            run_high_dim_sweep=True,
            high_dim_sweep_cases=[(10, 10), (20, 20), (30, 30)],
            high_dim_max_iter_factor=2,
            ipopt_solver_name="mock-ipopt",
            ipopt_options={"max_iter": 1},
        ),
        output_dir=tmp_path,
    )

    assert phase_order == ["train", "train", "train", "test", "test", "test"]
    assert payload["objective"] == -0.8
    assert payload["feasible"] is False
    assert _payload_metrics(payload)["best_case"] == {"n_var": 30, "n_qua_cons": 30}
    assert (tmp_path / payload["artifacts"]["sweep_summary_json"]).exists()
    assert (tmp_path / payload["artifacts"]["sweep_summary_csv"]).exists()


def test_qcqp_inn_experiment_filters_ipopt_options_for_single_case(monkeypatch, tmp_path):
    seen = {}

    def _fake_qcqp_inn_comparison(output_dir=None, **kwargs):
        seen["output_dir"] = output_dir
        seen["kwargs"] = dict(kwargs)
        return {"objective": -1.0, "feasible": True, "final_violation": 0.0, "iterations": 3}

    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "qcqp_inn_comparison", _fake_qcqp_inn_comparison)
    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "_print_qcqp_instance_metrics", lambda *args, **kwargs: None)

    payload = inn_pgd.qcqp_inn_experiment(
        **_qcqp_inn_params(
            run_high_dim_sweep=False,
            n_var=4,
            n_qua_cons=5,
            num_test_instance=1,
            ipopt_solver_name="mock-ipopt",
            ipopt_options={"max_iter": 1},
        ),
        output_dir=tmp_path,
    )

    assert payload["objective"] == -1.0
    assert seen["output_dir"] == tmp_path
    assert seen["kwargs"]["num_test_instance"] == 1
    assert "n_samples" not in seen["kwargs"]
    assert "ipopt_solver_name" not in seen["kwargs"]
    assert "ipopt_options" not in seen["kwargs"]


def test_qcqp_inn_experiment_uses_ipopt_objectives_for_convergence_gap(monkeypatch, tmp_path):
    seen = {}

    ipopt_rows = [
        {"index": 0, "objective": -2.0, "runtime": 0.1, "status": "optimal", "feasible": True},
        {"index": 1, "objective": -3.0, "runtime": 0.2, "status": "optimal", "feasible": True},
    ]

    def _fake_qcqp_inn_comparison(output_dir=None, **kwargs):
        seen["output_dir"] = output_dir
        seen["kwargs"] = dict(kwargs)
        return {"objective": -1.0, "feasible": True, "final_violation": 0.0, "iterations": 3, "artifacts": {}}

    def _fake_run_ipopt(problem, instance_batch, case_params):
        seen["ipopt_problem"] = problem
        seen["ipopt_instance_batch"] = instance_batch
        seen["ipopt_case_params"] = dict(case_params)
        seen["ipopt_n_samples"] = len(instance_batch)
        return ipopt_rows

    def _fake_summarize(payload, output_dir, case_params, *, ipopt_rows=None):
        del payload, output_dir, case_params
        seen["summarize_ipopt_rows"] = ipopt_rows
        return {"ipopt_obj_mean": -2.5, "ipopt_obj_best": -3.0}

    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "qcqp_inn_comparison", _fake_qcqp_inn_comparison)
    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "_run_qcqp_ipopt_baseline", _fake_run_ipopt)
    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "_summarize_qcqp_ipopt_comparison", _fake_summarize)
    monkeypatch.setattr(qcqp_inn_sweep_benchmarks, "_print_qcqp_instance_metrics", lambda *args, **kwargs: None)

    payload = inn_pgd.qcqp_inn_experiment(
        **_qcqp_inn_params(
            run_high_dim_sweep=False,
            compare_ipopt_baseline=True,
            n_var=4,
            n_qua_cons=5,
            num_test_instance=2,
            ipopt_solver_name="mock-ipopt",
            ipopt_options={"max_iter": 1},
        ),
        output_dir=None,
    )

    assert seen["kwargs"]["convergence_reference_objectives"] == [-2.0, -3.0]
    assert seen["kwargs"]["instance_batch"] is seen["ipopt_instance_batch"]
    assert seen["kwargs"]["case_context"]["data"] is seen["ipopt_problem"]
    assert seen["ipopt_case_params"]["ipopt_solver_name"] == "mock-ipopt"
    assert seen["ipopt_n_samples"] == 2
    assert seen["summarize_ipopt_rows"] is ipopt_rows
    assert _payload_metrics(payload)["ipopt_comparison_rows"] == [{"ipopt_obj_mean": -2.5, "ipopt_obj_best": -3.0}]


def test_jcc_linear_solver_benchmark_writes_comparison_artifacts(monkeypatch, tmp_path):
    seen = []

    class _BaseFakeSolver:
        objective = None
        feasible = False
        violation = None
        runtime_total = 0.0

        def __init__(self, problem, solver_config=None):
            del problem
            seen.append(dict(solver_config or {}))

        def solve_result(self, solve_type="opt", x_init=None, solver=None, options=None):
            del solve_type, x_init, solver, options
            return {
                "solution": np.zeros(6, dtype=float),
                "status": "optimal" if self.feasible else "failed",
                "objective": self.objective,
                "runtime_total": self.runtime_total,
                "violation": self.violation,
                "feasible": self.feasible,
                "extras": {},
            }

    class _FakeMixedIntegerSolver(_BaseFakeSolver):
        objective = 1.0
        feasible = True
        violation = 0.0
        runtime_total = 0.1

    class _FakeCVaRSolver(_BaseFakeSolver):
        objective = 1.2
        feasible = False
        violation = 0.2
        runtime_total = 0.05

    class _FakeScenarioSolver(_BaseFakeSolver):
        objective = 1.5
        feasible = True
        violation = 0.0
        runtime_total = 0.08

    monkeypatch.setattr(jcc_parametric_benchmarks, "JCCLinearSolver", _FakeMixedIntegerSolver)
    monkeypatch.setattr(jcc_parametric_benchmarks, "JCCLinearCVaRSolver", _FakeCVaRSolver)
    monkeypatch.setattr(jcc_parametric_benchmarks, "JCCLinearRobustScenarioSolver", _FakeScenarioSolver)

    result, output_dir = _run_workload('jcc_linear_solver_smoke', inn_pgd.jcc_linear_solver_benchmark, {'seed': 7, 'n_var': 6, 'n_input_dim': 3, 'n_ineq': 2, 'n_scenarios': 4, 'epsilon': 0.2, 'n_samples': 3, 'problem_config': {'scenario_scale': 0.05}, 'solver_configs': {'mixed_integer': {'solver': 'MOCK-MIP', 'solver_options': {'foo': 1}}, 'cvar': {'solver': 'MOCK-CVAR'}, 'scenario': {'solver': 'MOCK-SCEN'}}}, tmp_path)

    assert result.metrics["enabled_solvers"] == ["mixed_integer", "cvar", "scenario"]
    assert result.metrics["best_method"] == {"route": "solver", "method": "mixed_integer"}
    assert len(result.metrics["results"]) == 9
    assert result.metrics["solver_summary"]["mixed_integer"]["feasibility_rate"] == 1.0
    assert any(cfg.get("solver") == "MOCK-MIP" and cfg.get("solver_options") == {"foo": 1} for cfg in seen)
    assert (output_dir / result.artifacts["summary"]).exists()
    assert (output_dir / result.artifacts["rows_json"]).exists()
    assert (output_dir / result.artifacts["rows_csv"]).exists()
    assert (output_dir / result.artifacts["method_summary_json"]).exists()
    assert (output_dir / result.artifacts["method_summary_csv"]).exists()


def test_jcc_linear_solver_benchmark_supports_jccim_problem_family(monkeypatch, tmp_path):
    class _FakeSolver:
        def __init__(self, problem, solver_config=None):
            del solver_config
            self.problem = problem

        def solve_result(self, solve_type="opt", x_init=None, solver=None, options=None):
            del solve_type, x_init, solver, options
            return {
                "solution": np.zeros(self.problem.nvar, dtype=float),
                "status": "optimal",
                "objective": 1.0,
                "runtime_total": 0.01,
                "violation": 0.0,
                "feasible": True,
                "extras": {},
            }

    monkeypatch.setattr(jcc_parametric_benchmarks, "JCCLinearSolver", _FakeSolver)
    monkeypatch.setattr(jcc_parametric_benchmarks, "JCCLinearCVaRSolver", _FakeSolver)
    monkeypatch.setattr(jcc_parametric_benchmarks, "JCCLinearRobustScenarioSolver", _FakeSolver)

    result, output_dir = _run_workload('jccim_solver_smoke', inn_pgd.jcc_linear_solver_benchmark, {'seed': 7, 'problem_family': 'jccim', 'n_var': 6, 'n_input_dim': 3, 'n_ineq': 2, 'n_scenarios': 4, 'epsilon': 0.1, 'n_samples': 2}, tmp_path)

    assert result.metrics["problem_family"] == "jccim"
    assert result.metrics["enabled_solvers"] == ["mixed_integer", "cvar", "scenario"]
    assert len(result.metrics["results"]) == 6
    assert (output_dir / result.artifacts["summary"]).exists()


def test_convex_parametric_learning_benchmark_runs_predictor_postprocess(tmp_path):
    result, output_dir = _run_workload(
        "convex_parametric_learning_smoke",
        learning_postprocess.convex_parametric_learning_benchmark,
        {
            "seed": 13,
            "problem_type": "qp",
            "n_var": 5,
            "n_eq": 1,
            "n_ineq": 2,
            "n_samples": 3,
            "predictor_type": "constant",
            "prediction_value": 0.0,
            "baselines": ["predict_only", "predict_diff_projection"],
            "projection_config": {
                "proj_max_steps": 2,
                "corr_lr": 1e-3,
                "corr_momentum": 0.0,
                "proj_eps": 1e-6,
            },
            "device": "cpu",
            "dtype": "float32",
        },
        tmp_path,
    )
    assert result.metrics["problem_family"] == "convex_parametric"
    assert result.metrics["problem_type"] == "qp"
    assert result.metrics["workflow_stages"] == [
        "data_collection",
        "predictor_training_or_loading",
        "postprocess_evaluation",
    ]
    assert result.metrics["data_collection_time_sec"] >= 0.0
    assert result.metrics["predictor_train_time_sec"] == 0.0
    assert result.metrics["postprocess_train_time_sec"] == 0.0
    assert result.metrics["refine_time_sec"] >= 0.0
    assert set(result.metrics["enabled_baselines"]) == {"predict_only", "predict_diff_projection"}
    assert result.metrics["results"]["predict_diff_projection"]["refiner"] == "diff_projection"
    assert (output_dir / result.artifacts["summary"]).exists()
    assert (output_dir / result.artifacts["predictions"]).exists()


def test_acopf_parametric_learning_benchmark_runs_predictor_postprocess(tmp_path):
    result, output_dir = _run_workload(
        "acopf_parametric_learning_smoke",
        learning_postprocess.acopf_parametric_learning_benchmark,
        {
            "seed": 13,
            "case": 57,
            "dataset_samples": 10000,
            "n_samples": 2,
            "predictor_type": "constant",
            "prediction_value": 0.0,
            "baselines": ["predict_only"],
            "projection_config": {
                "proj_eps": 1e-6,
            },
            "device": "cpu",
            "dtype": "float32",
        },
        tmp_path,
    )
    assert result.metrics["problem_family"] == "acopf_parametric"
    assert result.metrics["case"] == 57
    assert result.metrics["nb"] == 57
    assert result.metrics["workflow_stages"] == [
        "data_collection",
        "predictor_training_or_loading",
        "postprocess_evaluation",
    ]
    assert result.metrics["data_collection_time_sec"] >= 0.0
    assert result.metrics["results"]["predict_only"]["refiner"] == "identity"
    assert (output_dir / result.artifacts["summary"]).exists()
    assert (output_dir / result.artifacts["predictions"]).exists()


def test_acopf_parametric_learning_benchmark_runs_pf_partial_predictor(tmp_path):
    result, output_dir = _run_workload(
        "acopf_parametric_pf_partial_smoke",
        learning_postprocess.acopf_parametric_learning_benchmark,
        {
            "seed": 13,
            "case": 57,
            "dataset_samples": 10000,
            "n_samples": 2,
            "predictor_type": "nn",
            "baselines": ["predict_only"],
            "problem_config": {
                "prediction_space": "pf_partial",
                "completion_config": {
                    "gradient_mode": "implicit",
                    "max_iters": 2,
                    "tol": 0.0,
                    "damping": 1e-5,
                    "allocation": "soft_cost",
                },
            },
            "predictor_model_config": {
                "h_dim": 16,
                "num_layer": 2,
                "outact": "tanh",
                "dropout": 0.0,
                "lr": 1e-4,
                "lr_decay": 1.0,
                "lr_decay_step": 1,
                "w_ineq": 1e-3,
                "w_obj": 1e-2,
                "approach": "unsupervise",
            },
            "predictor_train_config": {
                "n_samples": 4,
                "batch_size": 2,
                "total_iteration": 1,
                "pre_training": 0,
            },
            "device": "cpu",
            "dtype": "float32",
            "retrain": True,
        },
        tmp_path,
    )
    assert result.metrics["problem_family"] == "acopf_parametric"
    assert result.metrics["prediction_space"] == "pf_partial"
    assert result.metrics["partial_dim"] < result.metrics["n_var"]
    assert result.metrics["results"]["predict_only"]["refiner"] == "identity"
    assert (output_dir / result.artifacts["summary"]).exists()


def test_result_artifacts_are_recorded_in_result_json(tmp_path):
    _, output_dir = _run_workload('poly_star_artifact_json', hom_pgd.poly_star_benchmark, {'algorithms': ['Hom-PGD'], 'max_iterations': 2, 'reference_solver': 'grid', 'reference_grid_size': 51}, tmp_path)
    result_json = Path(output_dir / "result.json").read_text()
    assert "artifacts/records" in result_json
    assert "artifacts/manifest.json" in result_json


def test_solver_config_legacy_aliases_are_rejected():
    with pytest.raises(ValueError, match="Unsupported solver_config keys"):
        config_common.normalize_solver_run_config(solver_config={"name": "ipopt"})

    with pytest.raises(ValueError, match="Unsupported solver_configs keys"):
        jcc_setup.normalize_jcc_linear_solver_configs(
            solver_configs={"mixed_integer": {"options": {"tol": 1e-6}}}
        )


def test_benchmark_algorithm_config_reaches_runtime(monkeypatch):
    captured = {}

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del problem, hom_map, init_point
        captured["name"] = name
        captured["params"] = params
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    payload = hom_pgd.socp_hompgd_benchmark(
        max_iterations=3,
        common_config={"stepsize_rule": "diminish"},
        algorithm_config={"Hom-PGD": {"learning_rate": 5e-4, "momentum": 0.95}},
    )

    assert captured["name"] == "Hom-PGD"
    assert captured["params"]["common"]["stepsize_rule"] == "diminish"
    assert captured["params"]["Hom-PGD"]["learning_rate"] == 5e-4
    assert captured["params"]["Hom-PGD"]["momentum"] == 0.95
    metrics = _payload_metrics(payload)
    assert metrics["algorithms"] == ["Hom-PGD"]
    assert metrics["comparison_rows"][0]["route"] == "iterative"
    assert metrics["comparison_rows"][0]["method"] == "Hom-PGD"


def test_socp_hompgd_benchmark_accepts_canonical_config_groups(monkeypatch):
    captured = {}

    def fake_run_algorithm(name, problem, params, hom_map=None, init_point=None):
        del hom_map, init_point
        captured["name"] = name
        captured["params"] = params
        captured["problem"] = problem
        return {
            "obj_traj": [1.0],
            "cons_traj": [0.0],
            "iter_time": [0.01],
        }

    monkeypatch.setattr(convex_ineq_benchmarks, "run_algorithm", fake_run_algorithm)

    payload = hom_pgd.socp_hompgd_benchmark(
        max_iterations=3,
        common_config={"stepsize_rule": "diminish"},
        problem_config={"x_lower": -3, "x_upper": 3},
        algorithm_config={"Hom-PGD": {"learning_rate": 5e-4, "momentum": 0.95}},
    )

    assert captured["name"] == "Hom-PGD"
    assert captured["params"]["common"]["stepsize_rule"] == "diminish"
    assert captured["params"]["Hom-PGD"]["learning_rate"] == 5e-4
    assert captured["params"]["Hom-PGD"]["momentum"] == 0.95
    assert getattr(captured["problem"], "nvar", None) == 4
    metrics = _payload_metrics(payload)
    assert metrics["results"]["Hom-PGD"]["final_violation"] == pytest.approx(0.0)
    assert metrics["best_method"] == {"route": "iterative", "method": "Hom-PGD"}


def test_maxcut_algorithm_comparison_accepts_canonical_config_groups(tmp_path):
    result, output_dir = _run_workload('maxcut_canonical_groups', hom_pgd.maxcut_algorithm_comparison, {'algorithms': ['Hom-PGD'], 'n': 6, 'max_iterations': 2, 'max_running_time': 5, 'common_config': {'smooth': False}, 'algorithm_config': {'Hom-PGD': {'learning_rate': 0.001}}}, tmp_path)
    assert result.metrics["algorithms"] == ["Hom-PGD"]
    assert (output_dir / result.artifacts["records"]).exists()
    assert (output_dir / result.artifacts["summary"]).exists()
