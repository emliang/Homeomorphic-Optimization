from pathlib import Path

from homopt.experiments import ExperimentResult
from homopt.experiments.entrypoints import default_script_output_dir, print_result_summary


def test_default_script_output_dir_uses_results_root():
    assert default_script_output_dir("demo_run") == Path("results") / "demo_run"


def test_print_result_summary_emits_compact_comparison_table(capsys, tmp_path):
    result = ExperimentResult(
        name="demo",
        objective=1.234567,
        feasible=True,
        metrics={
            "comparison_summary_rows": [
                {
                    "method": "ALM",
                    "route": "iterative",
                    "objective_mean": 1.234567,
                    "equality_violation_mean": 1.234e-5,
                    "inequality_violation_mean": 0.0,
                    "first_order_lagrangian_gap_mean": 3.21e-4,
                    "avg_outer_iter_time_mean": 0.012345,
                    "mean_inner_iter_time_mean": 0.006789,
                    "initial_ip_time_mean": 0.1111,
                    "initial_transform_time_mean": 0.001234,
                    "final_transform_time_mean": 0.002345,
                    "total_wall_time_mean": 1.2345,
                    "runtime_total": 0.98765,
                    "hom_map_smooth": True,
                }
            ],
            "best_method": {"method": "ALM", "route": "iterative"},
            "reference_solution_diagnostics": {
                "available": True,
                "active_tol": 1e-5,
                "active_inequality_total": 2,
                "inequality_total": 5,
                "violated_inequality_total": 0,
                "active_box_total": 1,
                "box_total": 2,
                "violated_box_total": 0,
                "active_group_total": 1,
                "group_total": 1,
                "violated_group_total": 0,
                "counts_by_type": {
                    "linear": {"total": 2, "active": 1, "violated": 0, "max_residual": -1e-6},
                    "quadratic": {"total": 1, "active": 0, "violated": 0, "max_residual": -0.2},
                    "soc": {"total": 0, "active": 0, "violated": 0, "max_residual": 0.0},
                    "box_lower": {"total": 1, "active": 1, "violated": 0, "max_residual": 0.0},
                    "box_upper": {"total": 1, "active": 0, "violated": 0, "max_residual": -0.4},
                    "equality": {"total": 1, "active": 1, "violated": 0, "max_abs_residual": 2e-7},
                },
            },
            "reference_cache_enabled": True,
            "reference_cache_hit": True,
            "reference_cache_path": "artifacts/convex_reference_context_cache.npy",
            "reference_cache_load_time": 0.00123,
            "reference_cached_opt_solver_time": 0.1234,
            "reference_cached_origin_solver_time": 0.5678,
        },
    )

    print_result_summary(result, tmp_path)
    captured = capsys.readouterr().out

    assert "comparison:" in captured
    assert "reference_cache:" in captured
    assert "status=hit" in captured
    assert "artifacts/convex_reference_context_cache.npy" in captured
    assert "cached_opt=0.1234s" in captured
    assert "cached_origin=0.5678s" in captured
    assert "eq_vio" in captured
    assert "ineq_vio" in captured
    assert "lag_gap" in captured
    assert "avg_outer" in captured
    assert "avg_inner" in captured
    assert "cvxpy_ip" in captured
    assert "init_map" in captured
    assert "final_map" in captured
    assert "iter" in captured
    assert "wall" in captured
    assert "reference_solution_diagnostics:" in captured
    assert "active_ineq: 2/5" in captured
    assert "active_box: 1/2" in captured
    assert "active_group: 1/1" in captured
    assert "linear" in captured
    assert "quadratic" in captured
    assert "equality: 1" in captured
    assert "1.235" in captured
    assert "1.234e-05" in captured
    assert "0.000321" in captured
    assert "0.01235s" in captured
    assert "0.006789s" in captured
    assert "0.1111s" in captured
    assert "0.001234s" in captured
    assert "0.002345s" in captured
    assert "0.9877s" in captured
    assert "1.234s" in captured
    assert "smooth=True" in captured
    assert "best_method: ALM [iterative]" in captured


def test_print_result_summary_uses_violation_fallback_for_ineq_column(capsys, tmp_path):
    result = ExperimentResult(
        name="jcc",
        objective=1.0,
        feasible=False,
        metrics={
            "comparison_summary_rows": [
                {
                    "method": "INN-PGD",
                    "route": "iterative",
                    "objective_mean": float("nan"),
                    "violation_mean": 0.9,
                    "runtime_total": 1.0,
                    "total_wall_time": 1.0,
                }
            ],
            "best_method": {"method": "INN-PGD", "route": "iterative"},
        },
    )

    print_result_summary(result, tmp_path)
    captured = capsys.readouterr().out

    assert "INN-PGD [iterative]" in captured
    assert "nan" not in captured
    assert "       0.9" in captured
