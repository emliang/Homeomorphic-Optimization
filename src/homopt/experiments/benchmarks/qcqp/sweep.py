"""QCQP INN experiment and sensitivity sweep benchmarks."""

from __future__ import annotations

from pathlib import Path

from homopt.records.artifacts import artifact_mapping, artifact_ref, artifact_root, build_benchmark_payload, save_json, save_table_artifacts
from homopt.experiments.common.comparison import comparison_metric_source
from .reports import summarize_qcqp_sweep_rows
from .training import (
    _build_qcqp_inn_case_context,
    _load_qcqp_inn_record_from_result,
    _prepare_qcqp_inn_training_context,
    _resolve_qcqp_num_test_instance,
    _sample_qcqp_inn_test_instances,
)
from .inn_pgd import (
    _ipopt_reference_objectives,
    _print_qcqp_instance_metrics,
    _run_qcqp_ipopt_baseline,
    _save_qcqp_ipopt_comparison_summary,
    _save_qcqp_ipopt_instance_rows,
    _summarize_qcqp_ipopt_comparison,
    qcqp_inn_comparison,
)
from homopt.viz.inn_training import save_qcqp_sensitivity_convergence_visualizations


def _apply_qcqp_sensitivity_override(case_params, sweep_name, sweep_value):
    params = dict(case_params)
    model_sweeps = {
        "num_layer": int,
        "h_dim": int,
        "w_penalty": float,
        "w_distortion": float,
        "w_lipschitz": float,
        "lr": float,
    }
    optimizer_sweeps = {
        "feasibility_eps": float,
        "initial_latent_mode": str,
        "initial_latent_radius": float,
        "learning_rate": float,
        "momentum": float,
        "opt": str,
        "stepsize_rule": str,
    }
    if sweep_name in model_sweeps:
        params.setdefault("model_config", {})
        params["model_config"][sweep_name] = model_sweeps[sweep_name](sweep_value)
    elif sweep_name in optimizer_sweeps:
        params.setdefault("optimizer_config", {})
        params["optimizer_config"][sweep_name] = optimizer_sweeps[sweep_name](sweep_value)
    else:
        raise ValueError(f"Unsupported sensitivity sweep: {sweep_name}")
    return params


def _qcqp_sensitivity_case_tag(sweep_name, sweep_value):
    value = str(sweep_value).replace("/", "_")
    return f"{sweep_name}_{value}"


def _save_qcqp_sweep_summary(output_dir, rows):
    _, json_path, csv_path, _ = save_table_artifacts(
        output_dir,
        base_name="high_dim_sweep_summary",
        rows=rows,
        fieldnames=[
            "n_var",
            "n_qua_cons",
            "max_iterations",
            "objective",
            "feasible",
            "final_violation",
            "iterations",
            "output_dir",
        ],
    )
    return json_path, csv_path


def qcqp_inn_experiment(
    run_high_dim_sweep=False,
    high_dim_sweep_cases=None,
    high_dim_max_iter_factor=10,
    compare_ipopt_baseline=False,
    output_dir=None,
    config=None,
    **qcqp_params,
):
    """Run either a single QCQP INN case or a high-dimensional sweep."""

    ipopt_solver_name = str(qcqp_params.pop("ipopt_solver_name", "ipopt"))
    ipopt_options = dict(qcqp_params.pop("ipopt_options", {}) or {})
    qcqp_params = _resolve_qcqp_num_test_instance(qcqp_params)

    def _case_params_for_ipopt(base_params):
        return {
            **base_params,
            "ipopt_solver_name": ipopt_solver_name,
            "ipopt_options": ipopt_options,
        }

    if not bool(run_high_dim_sweep):
        case_context = _build_qcqp_inn_case_context(output_dir, qcqp_params)
        instance_batch = _sample_qcqp_inn_test_instances(case_context)
        precomputed_ipopt_rows = None
        if bool(compare_ipopt_baseline) and qcqp_params.get("convergence_reference_objective") is None:
            precomputed_ipopt_rows = _run_qcqp_ipopt_baseline(
                case_context["data"],
                instance_batch,
                _case_params_for_ipopt(qcqp_params),
            )
            qcqp_params = {
                **qcqp_params,
                "convergence_reference_objectives": _ipopt_reference_objectives(precomputed_ipopt_rows),
            }
        payload = qcqp_inn_comparison(
            output_dir=output_dir,
            case_context=case_context,
            instance_batch=instance_batch,
            **qcqp_params,
        )
        _print_qcqp_instance_metrics(payload, output_dir, int(qcqp_params.get("num_test_instance", 1)))
        if bool(compare_ipopt_baseline):
            compare_row = _summarize_qcqp_ipopt_comparison(
                payload,
                output_dir,
                _case_params_for_ipopt(qcqp_params),
                ipopt_rows=precomputed_ipopt_rows,
            )
            artifacts = dict(payload.get("artifacts", {}))
            if output_dir is not None:
                compare_json, compare_csv = _save_qcqp_ipopt_comparison_summary(output_dir, [compare_row])
                instance_json, instance_csv = _save_qcqp_ipopt_instance_rows(output_dir, precomputed_ipopt_rows or [])
                artifacts.update(
                    artifact_mapping(
                        output_dir,
                        ipopt_comparison_json=compare_json,
                        ipopt_comparison_csv=compare_csv,
                        ipopt_instance_json=instance_json,
                        ipopt_instance_csv=instance_csv,
                    )
                )
            payload = {
                **payload,
                "artifacts": artifacts,
                "metrics": {
                    **comparison_metric_source(payload),
                    "ipopt_comparison_rows": [compare_row],
                },
            }
        return payload

    factor = int(high_dim_max_iter_factor)
    sweep_cases = list(high_dim_sweep_cases or [])
    case_plan = []
    for n_var, n_qua_cons in sweep_cases:
        n_var = int(n_var)
        n_qua_cons = int(n_qua_cons)
        case_max_iterations = n_var * factor
        case_name = f"nvar_{n_var}_nqua_{n_qua_cons}"
        case_out = Path(output_dir) / case_name if output_dir is not None else None
        case_params = {
            **qcqp_params,
            "n_var": n_var,
            "n_qua_cons": n_qua_cons,
            "max_iterations": case_max_iterations,
        }
        case_context = _build_qcqp_inn_case_context(case_out, case_params)
        case_plan.append((case_name, case_out, case_params, case_max_iterations, case_context))

    print("[qcqp-compare] high-dim sweep train phase")
    for case_name, case_out, case_params, _, case_context in case_plan:
        print(f"[qcqp-compare] train case={case_name} output={case_out}")
        _prepare_qcqp_inn_training_context(case_context)

    rows = []
    compare_rows = []
    print("[qcqp-compare] high-dim sweep test phase")
    for case_name, case_out, case_params, case_max_iterations, case_context in case_plan:
        test_params = {**case_params, "retrain": False}
        instance_batch = _sample_qcqp_inn_test_instances(case_context)
        precomputed_ipopt_rows = None
        if bool(compare_ipopt_baseline) and test_params.get("convergence_reference_objective") is None:
            precomputed_ipopt_rows = _run_qcqp_ipopt_baseline(
                case_context["data"],
                instance_batch,
                _case_params_for_ipopt(test_params),
            )
            test_params = {
                **test_params,
                "convergence_reference_objectives": _ipopt_reference_objectives(precomputed_ipopt_rows),
            }
        print(f"[qcqp-compare] case={case_name} max_iterations={case_max_iterations} output={case_out}")
        payload = qcqp_inn_comparison(
            output_dir=case_out,
            case_context=case_context,
            instance_batch=instance_batch,
            **test_params,
        )
        _print_qcqp_instance_metrics(payload, case_out, int(test_params.get("num_test_instance", 1)))
        if bool(compare_ipopt_baseline):
            compare_row = _summarize_qcqp_ipopt_comparison(
                payload,
                case_out,
                _case_params_for_ipopt(test_params),
                ipopt_rows=precomputed_ipopt_rows,
            )
            if case_out is not None:
                _save_qcqp_ipopt_instance_rows(case_out, precomputed_ipopt_rows or [])
            compare_rows.append(compare_row)
            print(
                "[qcqp-compare] INN-PGD vs IPOPT | "
                f"inn_obj_mean={compare_row['inn_pgd_obj_mean']} "
                f"inn_time_mean={compare_row['inn_pgd_runtime_est_mean']}s "
                f"| ipopt_obj_mean={compare_row['ipopt_obj_mean']} "
                f"ipopt_time_mean={compare_row['ipopt_runtime_mean']}s "
                f"ipopt_success={compare_row['ipopt_success_rate']:.2f} "
                f"inn_feas={compare_row['inn_pgd_feasibility_rate']:.2f}"
            )
        metrics = comparison_metric_source(payload)
        row = {
            "n_var": int(test_params["n_var"]),
            "n_qua_cons": int(test_params["n_qua_cons"]),
            "max_iterations": case_max_iterations,
            "objective": payload.get("objective"),
            "feasible": payload.get("feasible"),
            "final_violation": metrics.get("final_violation"),
            "iterations": metrics.get("iterations"),
            "output_dir": None if case_out is None else str(case_out),
        }
        rows.append(row)
        print(
            f"[qcqp-compare] done case={case_name} "
            f"obj={row['objective']} feasible={row['feasible']} "
            f"vio={row['final_violation']}"
        )

    summary_json, summary_csv = _save_qcqp_sweep_summary(output_dir, rows)
    compare_json = compare_csv = None
    if compare_rows:
        compare_json, compare_csv = _save_qcqp_ipopt_comparison_summary(output_dir, compare_rows)
    sweep_summary = summarize_qcqp_sweep_rows(rows)
    artifacts = artifact_mapping(
        output_dir,
        sweep_summary_json=summary_json,
        sweep_summary_csv=summary_csv,
        ipopt_comparison_json=compare_json,
        ipopt_comparison_csv=compare_csv,
    )
    return build_benchmark_payload(
        objective=sweep_summary["objective"],
        feasible=sweep_summary["feasible"],
        artifacts=artifacts,
        num_cases=len(rows),
        best_case=sweep_summary["best_case"],
    )


def qcqp_inn_sensitivity_sweep(
    sensitivity_sweeps,
    output_dir=None,
    config=None,
    **base_params,
):
    """Run QCQP INN comparison over a small set of parameter sweeps."""

    artifact_dir = artifact_root(output_dir)
    base_params = _resolve_qcqp_num_test_instance(base_params)
    sweep_results = {}
    artifacts = {}
    for sweep_name, values in dict(sensitivity_sweeps).items():
        sweep_results[sweep_name] = []
        sweep_cases = []
        case_plan = []
        for value in values:
            case_params = _apply_qcqp_sensitivity_override(base_params, sweep_name, value)
            tag = _qcqp_sensitivity_case_tag(sweep_name, value)
            case_output_dir = Path(output_dir) / sweep_name / tag if output_dir is not None else None
            case_context = _build_qcqp_inn_case_context(case_output_dir, case_params)
            case_plan.append((value, tag, case_output_dir, case_params, case_context))
        print(f"[qcqp-sensitivity] train phase sweep={sweep_name}")
        for value, tag, case_output_dir, case_params, case_context in case_plan:
            del value
            print(f"[qcqp-sensitivity] train case={tag} output={case_output_dir}")
            _prepare_qcqp_inn_training_context(case_context)
        print(f"[qcqp-sensitivity] test phase sweep={sweep_name}")
        for value, tag, case_output_dir, case_params, case_context in case_plan:
            test_params = {**case_params, "retrain": False}
            instance_batch = _sample_qcqp_inn_test_instances(case_context)
            result = qcqp_inn_comparison(
                output_dir=case_output_dir,
                case_context=case_context,
                instance_batch=instance_batch,
                **test_params,
            )
            sweep_cases.append(
                {
                    "label": f"{sweep_name}={value}",
                    "value": value,
                    "tag": tag,
                    "record": _load_qcqp_inn_record_from_result(result, case_output_dir),
                }
            )
            sweep_results[sweep_name].append(
                {
                    "tag": tag,
                    "value": value,
                    "objective": result.get("objective"),
                    "final_violation": result.get("final_violation"),
                    "feasible": result.get("feasible"),
                    "iterations": result.get("iterations"),
                    "output_dir": None if case_output_dir is None else str(case_output_dir),
                }
            )
        if bool(base_params.get("visualize", False)) and artifact_dir is not None and int(base_params.get("n_var", 0)) == 2:
            artifacts.update(
                save_qcqp_sensitivity_convergence_visualizations(
                    output_dir,
                    artifact_dir,
                    sweep_name=sweep_name,
                    cases=sweep_cases,
                    n_samples=int(base_params.get("num_test_instance", 1)),
                    visualize_instance_idx=base_params.get("visualize_instance_idx", 0),
                    show_convergence_legend=base_params.get("show_convergence_legend", True),
                    violation_y_min=base_params.get("violation_y_min", 1e-8),
                    plot_objective_gap=base_params.get("plot_objective_gap", True),
                    plot_gap_plus_violation=base_params.get("plot_gap_plus_violation", True),
                    plot_runtime_summary=base_params.get("plot_runtime_summary", True),
                )
            )

    if artifact_dir is not None:
        summary_path = save_json(artifact_dir / "qcqp_sensitivity_summary.json", sweep_results)
        artifacts["summary"] = artifact_ref(output_dir, summary_path)

    return build_benchmark_payload(
        objective=0.0,
        feasible=True,
        artifacts=artifacts,
        metrics={"sweeps": sweep_results},
    )


__all__ = ["qcqp_inn_experiment", "qcqp_inn_sensitivity_sweep"]
