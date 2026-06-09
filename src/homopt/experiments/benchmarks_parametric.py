"""Parametric benchmark entrypoints.

This module provides the public benchmark surface for the parametric main
experiment track while reusing shared implementation helpers.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ._comparison_helpers import (
    build_comparison_views,
    make_comparison_row,
    summarize_comparison_rows_by_method,
    timing_extras_from_iter_time,
)
from ._benchmark_common import (
    artifact_mapping,
    artifact_ref,
    artifact_root,
    build_benchmark_payload,
    final_or_nan,
    resolve_runtime,
    save_json,
    save_incremental_comparison_artifacts,
    save_numpy,
    summarize_single_problem_run_record,
    save_table_artifacts,
    trajectory_dim,
)
from homopt.viz.inn_training import (
    save_inn_training_visualizations,
    save_qcqp_2d_comparison_visualizations,
    save_qcqp_sensitivity_convergence_visualizations,
)
from ._inn_baselines import (
    build_inn_lagrangian_method_params,
    inn_lagrangian_feasibility_tol,
    normalize_inn_lagrangian_baselines,
    run_inn_lagrangian_baseline,
)
from ._benchmark_parametric_common import (
    build_qcqp_ipnn_training_payload,
    build_qcqp_inn_training_payload,
    build_qcqp_predictor_training_payload,
    default_qcqp_ipnn_args,
    default_qcqp_inn_args,
    default_qcqp_optimizer_config,
    default_qcqp_predictor_args,
    default_qcqp_projection_args,
    extract_qcqp_inn_instance_metrics,
    load_or_train_decision_predictor,
    load_or_train_ipnn_mapping,
    load_qcqp_inn_record,
    normalize_jcc_linear_problem_config,
    normalize_jcc_linear_solver_configs,
    QCQP_INN_METHOD_NAME,
    summarize_qcqp_inn_record,
    load_or_train_inn_mapping,
    normalize_qcqp_inn_benchmark_config,
    normalize_qcqp_problem_config,
    normalize_qcqp_train_config,
    normalize_solver_run_config,
    training_time_from_record,
)
from ._qcqp_helpers import (
    build_qcqp_learning_baseline_output,
    build_qcqp_learning_benchmark_metrics,
    build_qcqp_learning_context,
    build_qcqp_problem_context,
    build_qcqp_route_comparison_views,
    build_qcqp_route_rows,
    normalize_qcqp_learning_baselines,
    QCQP_LEARNING_BASELINES,
    QCQP_LEARNING_RESULT_FIELDS,
    QCQP_ROUTE_COMPARISON_FIELDS,
    summarize_qcqp_sweep_rows,
)
from homopt.learning import (
    ConstantDecisionPredictor,
    DiffProjectionRefiner,
    ExactSolverProjectionRefiner,
    HomeomorphicProjectionRefiner,
    IPNNBisectionRefiner,
    IdentityRefiner,
    NeuralDecisionPredictor,
    RayBisectionRefiner,
    WarmStartSolverRefiner,
    summarize_prediction_route,
)
from homopt.models import INNPGDOptimizer
from homopt.problems import JCCDCOPFProblem, JCCIMProblem, JCCLinearProblem, as_learning_problem, bind_problem_instance, bind_singleton_problem_instance
from homopt.solvers import JCCLinearCVaRSolver, JCCLinearRobustScenarioSolver, JCCLinearSolver, QCQPSolver, solve_exact_result
from homopt.utils import cast_tensors_to_dtype, ensure_dir, set_global_seed


PARAMETRIC_BENCHMARKS = (
    "qcqp_inn_experiment",
    "qcqp_inn_sensitivity_sweep",
    "qcqp_learning_benchmark",
    "qcqp_route_comparison",
    "inn_training_benchmark",
    "jcc_linear_solver_benchmark",
    "jcc_problem_benchmark",
)


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
        params.setdefault("model_config_overrides", {})
        params["model_config_overrides"][sweep_name] = model_sweeps[sweep_name](sweep_value)
    elif sweep_name in optimizer_sweeps:
        params.setdefault("optimizer_config_overrides", {})
        params["optimizer_config_overrides"][sweep_name] = optimizer_sweeps[sweep_name](sweep_value)
    else:
        raise ValueError(f"Unsupported sensitivity sweep: {sweep_name}")
    return params


def _qcqp_sensitivity_case_tag(sweep_name, sweep_value):
    value = str(sweep_value).replace("/", "_")
    return f"{sweep_name}_{value}"


def _load_qcqp_inn_record_from_result(result, case_output_dir):
    del result
    if case_output_dir is None:
        return None
    return load_qcqp_inn_record(case_output_dir)


def _resolve_qcqp_num_test_instance(params):
    params = dict(params)
    if "n_samples" in params:
        raise ValueError("QCQP INN tests use num_test_instance; train_config['n_samples'] controls INN training.")
    if "batch_size" in params or "total_iteration" in params:
        raise ValueError("QCQP INN training controls use train_config, not top-level batch_size/total_iteration.")
    params["num_test_instance"] = int(params.get("num_test_instance", 1))
    return params


def _stable_qcqp_training_cache_label(*, problem_args, model_args, train_args, seed, dtype):
    base = (
        f"qcqp_n{int(problem_args['n_var'])}"
        f"_q{int(problem_args['n_qua_cons'])}"
        f"_seed{int(seed)}"
    )
    payload = {
        "version": 1,
        "seed": int(seed),
        "dtype": str(dtype),
        "problem": problem_args,
        "model": model_args,
        "train": train_args,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._")
    return f"{safe_base}_{digest}"


def _qcqp_training_cache_dir(output_dir, *, problem_args, model_args, train_args, seed, dtype):
    label = _stable_qcqp_training_cache_label(
        problem_args=problem_args,
        model_args=model_args,
        train_args=train_args,
        seed=seed,
        dtype=dtype,
    )
    if output_dir is None:
        root = Path("results") / "_scratch" / "qcqp" / "training_cache"
    else:
        root = Path(output_dir).parent / "training_cache"
    return ensure_dir(root / label)


def _save_qcqp_learning_artifacts(output_dir, *, metrics, rows, prediction_payload):
    artifact_dir = artifact_root(output_dir)
    if artifact_dir is None:
        return {}
    summary_path = save_json(artifact_dir / "qcqp_learning_summary.json", metrics)
    artifacts, _, _, _ = save_table_artifacts(
        output_dir,
        base_name="qcqp_learning_baselines",
        rows=rows,
        fieldnames=list(QCQP_LEARNING_RESULT_FIELDS),
        json_key="baselines_json",
        csv_key="baselines_csv",
        markdown_key="unused_markdown",
    )
    predictions_path = save_numpy(artifact_dir / "qcqp_learning_predictions.npy", prediction_payload)
    plot_artifacts = _save_qcqp_learning_plots(output_dir, rows)
    return {
        **artifacts,
        **plot_artifacts,
        "summary": artifact_ref(output_dir, summary_path),
        "predictions": artifact_ref(output_dir, predictions_path),
    }


def _learning_method_label(name):
    labels = {
        "predict_only": "Predict",
        "predict_ray_bisection": "Ray-Bisect",
        "predict_diff_projection": "D-Proj",
        "predict_exact_projection": "Proj",
        "predict_homeomorphic_projection": "H-Proj",
        "predict_ip_bisection": "B-Proj",
        "predict_warm_start": "WS",
    }
    return labels.get(str(name), str(name))


def _learning_bar_plot(output_dir, rows, *, key, ylabel, filename, positive_log=False):
    values = []
    labels = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if positive_log:
            value = max(value, 1e-12)
        labels.append(str(row["baseline"]))
        values.append(value)
    if not values:
        return None

    from homopt.viz.style import ALGORITHM_COLORS, PAPER_STYLE, apply_paper_axis_style, require_matplotlib, set_axis_labels
    from homopt.viz.artifacts import save_figure

    plt = require_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=PAPER_STYLE["runtime_figsize"])
    colors = [ALGORITHM_COLORS.get(label, "#4C78A8") for label in labels]
    ax.bar(np.arange(len(labels)), values, color=colors, alpha=0.9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([_learning_method_label(label) for label in labels], rotation=25, ha="right")
    if positive_log:
        ax.set_yscale("log")
    set_axis_labels(ax, "Method", ylabel)
    apply_paper_axis_style(ax)
    fig.tight_layout()
    path = artifact_root(output_dir) / filename
    save_figure(fig, path)
    plt.close(fig)
    return path


def _save_qcqp_learning_plots(output_dir, rows):
    if output_dir is None:
        return {}
    plot_specs = (
        ("learning_feasibility", "feasibility_rate", "Feasibility rate", "qcqp_learning_feasibility.pdf", False),
        ("learning_violation", "violation_mean", "Mean violation", "qcqp_learning_violation.pdf", True),
        ("learning_runtime", "runtime_mean", "Running time per instance (s)", "qcqp_learning_runtime.pdf", False),
        ("learning_objective_gap", "objective_gap_mean", "Relative optimality gap", "qcqp_learning_objective_gap.pdf", True),
        ("learning_solution_mse", "solution_mse", "Solution MSE", "qcqp_learning_solution_mse.pdf", True),
    )
    artifacts = {}
    for artifact_name, key, ylabel, filename, positive_log in plot_specs:
        path = _learning_bar_plot(output_dir, rows, key=key, ylabel=ylabel, filename=filename, positive_log=positive_log)
        if path is not None:
            artifacts[artifact_name] = artifact_ref(output_dir, path)
    return artifacts


JCC_LINEAR_SOLVER_RESULT_FIELDS = (
    "row_kind",
    "route",
    "method",
    "instance_id",
    "status",
    "objective",
    "feasible",
    "violation",
    "runtime",
    "objective_mean",
    "feasibility_rate",
    "violation_mean",
    "runtime_mean",
    "runtime_total",
)


def _save_jcc_linear_solver_artifacts(output_dir, *, metrics, rows, summary_rows):
    artifact_dir = artifact_root(output_dir)
    if artifact_dir is None:
        return {}
    summary_path = save_json(artifact_dir / "jcc_linear_solver_summary.json", metrics)
    row_artifacts, _, _, _ = save_table_artifacts(
        output_dir,
        base_name="jcc_linear_solver_rows",
        rows=rows,
        fieldnames=list(JCC_LINEAR_SOLVER_RESULT_FIELDS),
        json_key="rows_json",
        csv_key="rows_csv",
        markdown_key="unused_markdown",
    )
    summary_artifacts, _, _, _ = save_table_artifacts(
        output_dir,
        base_name="jcc_linear_solver_method_summary",
        rows=summary_rows,
        fieldnames=["route", "method", "objective_mean", "feasibility_rate", "violation_mean", "runtime_mean", "runtime_total", "num_instances"],
        json_key="method_summary_json",
        csv_key="method_summary_csv",
        markdown_key="unused_method_markdown",
    )
    return {
        **row_artifacts,
        **summary_artifacts,
        "summary": artifact_ref(output_dir, summary_path),
    }


def _save_qcqp_route_comparison_summary(output_dir, rows):
    markdown_lines = [
        "| Route | Method | Objective Mean | Feasibility Rate | Violation Mean | Runtime Mean (s) | Runtime Total (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        markdown_lines.append(
            "| {route} | {method} | {objective_mean} | {feasibility_rate} | {violation_mean} | {runtime_mean} | {runtime_total} |".format(
                route=str(row.get("route")),
                method=str(row.get("method")),
                objective_mean="--" if row.get("objective_mean") is None else f"{float(row['objective_mean']):.6g}",
                feasibility_rate="--" if row.get("feasibility_rate") is None else f"{float(row['feasibility_rate']):.4f}",
                violation_mean="--" if row.get("violation_mean") is None else f"{float(row['violation_mean']):.3e}",
                runtime_mean="--" if row.get("runtime_mean") is None else f"{float(row['runtime_mean']):.6g}",
                runtime_total="--" if row.get("runtime_total") is None else f"{float(row['runtime_total']):.6g}",
            )
        )
    _, json_path, csv_path, markdown_path = save_table_artifacts(
        output_dir,
        base_name="qcqp_route_comparison",
        rows=rows,
        fieldnames=list(QCQP_ROUTE_COMPARISON_FIELDS),
        markdown_text="\n".join(markdown_lines) + "\n",
    )
    return json_path, csv_path, markdown_path


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


def _save_qcqp_ipopt_comparison_summary(output_dir, rows):
    _, json_path, csv_path, _ = save_table_artifacts(
        output_dir,
        base_name="qcqp_inn_pgd_ipopt_comparison",
        rows=rows,
        fieldnames=[
            "n_var",
            "n_qua_cons",
            "n_samples",
            "inn_pgd_obj_mean",
            "inn_pgd_obj_best",
            "inn_pgd_runtime_est_mean",
            "inn_pgd_feasibility_rate",
            "ipopt_obj_mean",
            "ipopt_obj_best",
            "ipopt_runtime_mean",
            "ipopt_runtime_total",
            "ipopt_success_rate",
        ],
    )
    return json_path, csv_path


def _save_qcqp_ipopt_instance_rows(output_dir, rows):
    _, json_path, csv_path, _ = save_table_artifacts(
        output_dir,
        base_name="qcqp_inn_pgd_ipopt_instances",
        rows=rows,
        fieldnames=[
            "index",
            "status",
            "objective",
            "runtime",
            "feasible",
            "max_violation",
            "solver_status",
        ],
        json_key="ipopt_instance_json",
        csv_key="ipopt_instance_csv",
    )
    return json_path, csv_path


def _save_qcqp_iterative_baseline_summary(output_dir, rows):
    _, json_path, csv_path, _ = save_table_artifacts(
        output_dir,
        base_name="qcqp_iterative_baseline_comparison",
        rows=rows,
        fieldnames=[
            "method",
            "n_samples",
            "objective_mean",
            "objective_best",
            "violation_mean",
            "feasibility_rate",
            "runtime_mean",
            "runtime_total",
            "iterations_mean",
        ],
        json_key="iterative_baseline_json",
        csv_key="iterative_baseline_csv",
    )
    return json_path, csv_path


def _print_qcqp_instance_metrics(payload, output_dir, n_samples):
    rows = extract_qcqp_inn_instance_metrics(payload, output_dir, n_samples)
    if not rows:
        return
    est_runtime = rows[0]["runtime_est"]
    total_runtime = est_runtime * n_samples if n_samples > 0 and np.isfinite(est_runtime) else float("nan")
    print(f"[qcqp-compare] instance metrics (n_samples={n_samples}, total_runtime={total_runtime:.3f}s)")
    for row in rows:
        print(
            f"  inst={row['index']:02d} | objective={row['objective']:.6f} "
            f"| violation={row['violation']:.3e} | feasible={row['feasible']} "
            f"| runtime_est={row['runtime_est']:.3f}s"
        )


def _run_qcqp_ipopt_baseline(problem, instance_batch, case_params):
    solver_name = str(case_params.get("ipopt_solver_name", "ipopt"))
    solver_options = dict(case_params.get("ipopt_options", {}))
    rows = []
    n_samples = len(instance_batch)
    for idx in range(n_samples):
        bound_problem = bind_problem_instance(problem, instance_batch.get_instance(idx))
        solver = QCQPSolver(bound_problem.prob_para)
        result = solve_exact_result(
            solver,
            solve_config={
                "solve_type": "opt",
                "solver_name": solver_name,
                "solver_options": solver_options,
            },
        )
        extras = result.get("extras", {})
        rows.append(
            {
                "index": idx,
                "status": str(result.get("status")),
                "objective": result.get("objective"),
                "runtime": float(result.get("runtime_total", float("nan"))),
                "feasible": bool(result.get("feasible", False)),
                "max_violation": float(result.get("violation", float("inf"))),
                "solver_status": str(extras.get("solver_status")) if extras.get("solver_status") is not None else None,
            }
        )
    return rows


def _summarize_qcqp_ipopt_comparison(payload, output_dir, case_params, *, ipopt_rows=None):
    n_samples = int(case_params.get("num_test_instance", 1))
    inn_rows = extract_qcqp_inn_instance_metrics(payload, output_dir, n_samples)
    if ipopt_rows is None:
        raise ValueError("ipopt_rows must be computed from the shared QCQP test instance batch.")
    inn_obj = [r["objective"] for r in inn_rows if r.get("objective") is not None]
    inn_rt = [r["runtime_est"] for r in inn_rows if np.isfinite(r.get("runtime_est", np.nan))]
    ip_obj = [
        r["objective"]
        for r in ipopt_rows
        if r.get("objective") is not None and r.get("status") == "optimal" and r.get("feasible", False)
    ]
    ip_rt = [r["runtime"] for r in ipopt_rows if np.isfinite(r.get("runtime", np.nan))]
    ip_ok = [1.0 if (r.get("status") == "optimal" and r.get("feasible", False)) else 0.0 for r in ipopt_rows]
    return {
        "n_var": int(case_params["n_var"]),
        "n_qua_cons": int(case_params["n_qua_cons"]),
        "n_samples": n_samples,
        "inn_pgd_obj_mean": float(np.mean(inn_obj)) if inn_obj else None,
        "inn_pgd_obj_best": float(np.min(inn_obj)) if inn_obj else None,
        "inn_pgd_runtime_est_mean": float(np.mean(inn_rt)) if inn_rt else None,
        "inn_pgd_feasibility_rate": float(np.mean([1.0 if r.get("feasible", False) else 0.0 for r in inn_rows])) if inn_rows else 0.0,
        "ipopt_obj_mean": float(np.mean(ip_obj)) if ip_obj else None,
        "ipopt_obj_best": float(np.min(ip_obj)) if ip_obj else None,
        "ipopt_runtime_mean": float(np.mean(ip_rt)) if ip_rt else None,
        "ipopt_runtime_total": float(np.sum(ip_rt)) if ip_rt else None,
        "ipopt_success_rate": float(np.mean(ip_ok)) if ip_ok else 0.0,
    }


def _ipopt_reference_objectives(ipopt_rows):
    objectives = []
    for row in ipopt_rows:
        value = row.get("objective") if row.get("status") == "optimal" and row.get("feasible", False) else None
        objectives.append(float("nan") if value is None else float(value))
    return objectives


def _reference_objective_for_instance(reference_objective, reference_objectives, instance_idx):
    if reference_objective is not None:
        return reference_objective
    if reference_objectives is None:
        return None
    if int(instance_idx) >= len(reference_objectives):
        return None
    value = reference_objectives[int(instance_idx)]
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _qcqp_baseline_init_point(problem, generator):
    dtype = getattr(problem, "dtype", getattr(generator, "dtype", torch.float32))
    device = getattr(problem, "device", getattr(generator, "device", torch.device("cpu")))
    x0 = getattr(generator, "fixed_x0", None)
    if x0 is None:
        return torch.zeros(1, problem.nvar, device=device, dtype=dtype)
    return torch.as_tensor(x0, device=device, dtype=dtype).view(1, -1)


def _qcqp_inn_initial_latent(problem, input_params, *, mode="center", radius=1.0, seed=None):
    dtype = getattr(problem, "dtype", torch.float32)
    device = getattr(problem, "device", torch.device("cpu"))
    mode = str(mode or "center").strip().lower()
    if mode == "center":
        return torch.zeros(input_params.shape[0], problem.nvar, device=device, dtype=dtype)
    if mode == "sphere_boundary_random":
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(int(seed))
        directions = torch.randn(
            input_params.shape[0],
            problem.nvar,
            generator=generator,
            dtype=torch.float64,
            device="cpu",
        )
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (float(radius) * directions).to(device=device, dtype=dtype)
    raise ValueError(
        "Unsupported initial_latent_mode: "
        f"{mode}. Use 'center' or 'sphere_boundary_random'."
    )


def _qcqp_inn_initial_decision(problem, model, input_params, initial_latent):
    input_params = input_params.to(device=initial_latent.device, dtype=initial_latent.dtype)
    model.eval()
    with torch.no_grad():
        if hasattr(model, "forward_embedded"):
            condition_emb = model.embed_condition(input_params, initial_latent.shape[0]) if hasattr(model, "embed_condition") else None
            mapped = model.forward_embedded(initial_latent, condition_emb)
        else:
            mapped = model(initial_latent, input_params)
        if isinstance(mapped, tuple):
            mapped = mapped[0]
        scaled = problem.scale(input_params, mapped)
        return problem.complete_partial(input_params, scaled).detach()


def _run_qcqp_iterative_baselines(data, instance_batch, baselines, *, seed, baseline_config, initial_points=None):
    baselines = normalize_inn_lagrangian_baselines(baselines)
    if not baselines:
        return [], {}
    params = build_inn_lagrangian_method_params(seed, baseline_config)
    rows = []
    records = {}
    n_samples = len(instance_batch)
    feasibility_tol = inn_lagrangian_feasibility_tol(baseline_config)
    for method in baselines:
        method_records = []
        for idx in range(n_samples):
            problem = bind_problem_instance(data, instance_batch.get_instance(idx))
            if initial_points is None:
                init_point = _qcqp_baseline_init_point(problem, data)
            else:
                init_point = torch.as_tensor(
                    initial_points[idx : idx + 1],
                    device=getattr(problem, "device", data.device),
                    dtype=getattr(problem, "dtype", data.dtype),
                )
            record = run_inn_lagrangian_baseline(
                problem,
                init_point,
                params[method],
                method=method,
                seed=seed,
            )
            summary = summarize_single_problem_run_record(problem, record, include_total_iter_time=True)
            method_records.append(
                {
                    "index": idx,
                    "objective": summary.get("final_objective"),
                    "violation": summary.get("final_violation"),
                    "feasible": bool(summary.get("final_violation", float("inf")) <= feasibility_tol),
                    "runtime": float(summary.get("total_wall_time", summary.get("total_iter_time", float("nan")))),
                    "iterations": int(summary.get("iterations", 0)),
                    "record": record,
                }
            )
        objectives = [row["objective"] for row in method_records if row.get("objective") is not None]
        violations = [row["violation"] for row in method_records if row.get("violation") is not None]
        runtimes = [row["runtime"] for row in method_records if np.isfinite(row.get("runtime", np.nan))]
        iterations = [row["iterations"] for row in method_records]
        rows.append(
            {
                "method": method,
                "n_samples": n_samples,
                "objective_mean": float(np.mean(objectives)) if objectives else None,
                "objective_best": float(np.min(objectives)) if objectives else None,
                "violation_mean": float(np.mean(violations)) if violations else None,
                "feasibility_rate": float(np.mean([1.0 if row["feasible"] else 0.0 for row in method_records])) if method_records else 0.0,
                "runtime_mean": float(np.mean(runtimes)) if runtimes else None,
                "runtime_total": float(np.sum(runtimes)) if runtimes else None,
                "iterations_mean": float(np.mean(iterations)) if iterations else None,
            }
        )
        records[method] = method_records
    return rows, records


def _select_batched_history(value, n_samples, instance_idx=0):
    if n_samples <= 1:
        return value
    if hasattr(value, "detach"):
        total = int(value.shape[0])
        if total % int(n_samples) != 0:
            raise ValueError(
                f"Batched history length {total} is not divisible by num_test_instance={int(n_samples)}."
            )
        return value.reshape(total // int(n_samples), int(n_samples), *value.shape[1:])[:, int(instance_idx)]
    array = np.asarray(value)
    if array.shape[0] % int(n_samples) != 0:
        raise ValueError(
            f"Batched history length {array.shape[0]} is not divisible by num_test_instance={int(n_samples)}."
        )
    return array.reshape(array.shape[0] // int(n_samples), int(n_samples), *array.shape[1:])[:, int(instance_idx)]


def _visualize_instance_indices(value, n_samples):
    n_samples = max(1, int(n_samples))
    if value is None:
        raw_indices = [0]
    elif isinstance(value, (list, tuple, set)):
        raw_indices = list(value)
    else:
        raw_indices = [value]
    indices = []
    for raw in raw_indices:
        idx = int(raw)
        if idx < 0 or idx >= n_samples:
            raise ValueError(f"visualize_instance_idx={idx} is outside available range [0, {n_samples - 1}].")
        if idx not in indices:
            indices.append(idx)
    return indices or [0]


def _merge_instance_visualization_artifacts(target, source, instance_idx, multiple):
    if not multiple:
        target.update(source)
        return
    for key, value in source.items():
        target[f"{key}_inst{int(instance_idx)}"] = value


def _make_qcqp_learning_refiner(
    *,
    baseline,
    qc_problem,
    solver_run_config,
    projection_config,
    homeomorphic_model=None,
    ipnn_model=None,
):
    if baseline == "predict_only":
        return IdentityRefiner()
    if baseline == "predict_ray_bisection":
        return RayBisectionRefiner(anchor=qc_problem.fixed_x0, steps=projection_config["proj_max_steps"], tol=projection_config["proj_eps"])
    if baseline == "predict_exact_projection":
        return ExactSolverProjectionRefiner(
            qc_problem,
            solver_factory=QCQPSolver,
            solve_exact=solve_exact_result,
            solve_config=solver_run_config,
            tol=projection_config["proj_eps"],
        )
    if baseline == "predict_warm_start":
        return WarmStartSolverRefiner(
            qc_problem,
            solver_factory=QCQPSolver,
            solve_exact=solve_exact_result,
            solve_config=solver_run_config,
            tol=projection_config["proj_eps"],
        )
    if baseline == "predict_diff_projection":
        return DiffProjectionRefiner(
            steps=projection_config["proj_max_steps"],
            lr=projection_config["corr_lr"],
            momentum=projection_config["corr_momentum"],
            tol=projection_config["proj_eps"],
        )
    if baseline == "predict_homeomorphic_projection":
        if homeomorphic_model is None:
            raise ValueError("predict_homeomorphic_projection requires a trained homeomorphic model.")
        return HomeomorphicProjectionRefiner(
            homeomorphic_model,
            qc_problem,
            projection_config=projection_config,
            tol=projection_config["proj_eps"],
        )
    if baseline == "predict_ip_bisection":
        if ipnn_model is None:
            raise ValueError("predict_ip_bisection requires a trained IPNN model.")
        return IPNNBisectionRefiner(
            ipnn_model,
            qc_problem,
            projection_config=projection_config,
            tol=projection_config["proj_eps"],
        )
    raise ValueError(f"Unsupported qcqp learning baseline: {baseline}")


def _prepare_qcqp_postprocess_resources(
    *,
    enabled,
    seed,
    n_var,
    n_qua_cons,
    n_linear_cons,
    n_samples,
    runtime_device,
    runtime_dtype,
    problem_args,
    context,
    save_dir,
    output_dir,
    retrain,
    model_config,
    model_config_overrides,
    train_config,
    train_config_overrides,
    ipnn_model_config,
    ipnn_model_config_overrides,
    ipnn_train_config,
    ipnn_train_config_overrides,
):
    resources = {
        "homeomorphic_model": None,
        "homeomorphic_train_time": 0.0,
        "ipnn_model": None,
        "ipnn_train_time": 0.0,
    }
    artifacts = {}

    if "predict_homeomorphic_projection" in enabled:
        homeo_cfg = normalize_qcqp_inn_benchmark_config(
            seed=seed,
            n_var=n_var,
            n_qua_cons=n_qua_cons,
            n_linear_cons=n_linear_cons,
            n_samples=max(int(n_samples), 64),
            batch_size=min(64, max(int(n_samples), 1)),
            total_iteration=1000,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
            base_inn_args=default_qcqp_inn_args(),
            base_optimizer_config=default_qcqp_optimizer_config(),
            problem_config=problem_args,
            model_config=model_config,
            model_config_overrides=model_config_overrides,
            train_config=train_config,
            train_config_overrides=train_config_overrides,
        )
        homeo_args = build_qcqp_inn_training_payload(
            homeo_cfg["problem"],
            homeo_cfg["model"],
            homeo_cfg["train"],
            ensure_results_save_freq=True,
        )
        model, record, model_path, record_path = load_or_train_inn_mapping(
            context["qc_problem"],
            homeo_args,
            save_dir / "homeomorphic_projection",
            retrain=retrain,
        )
        resources["homeomorphic_model"] = model
        resources["homeomorphic_train_time"] = training_time_from_record(record)
        artifacts.update(
            artifact_mapping(
                output_dir,
                homeomorphic_model=model_path,
                homeomorphic_training_record=record_path,
            )
        )

    if "predict_ip_bisection" in enabled:
        ipnn_train_args = normalize_qcqp_train_config(
            n_samples=max(int(n_samples), 64),
            batch_size=min(64, max(int(n_samples), 1)),
            total_iteration=1000,
            train_config=ipnn_train_config,
            train_config_overrides=ipnn_train_config_overrides,
        )
        ipnn_args = build_qcqp_ipnn_training_payload(
            problem_args,
            {
                **default_qcqp_ipnn_args(),
                **(ipnn_model_config or {}),
                **(ipnn_model_config_overrides or {}),
                "device": str(runtime_device),
                "dtype": runtime_dtype,
            },
            ipnn_train_args,
            ensure_results_save_freq=True,
        )
        model, record, model_path, record_path = load_or_train_ipnn_mapping(
            context["qc_problem"],
            ipnn_args,
            save_dir / "ipnn_projection",
            retrain=retrain,
        )
        resources["ipnn_model"] = model
        resources["ipnn_train_time"] = training_time_from_record(record)
        artifacts.update(
            artifact_mapping(
                output_dir,
                ipnn_model=model_path,
                ipnn_training_record=record_path,
            )
        )

    return resources, artifacts


def _prepare_qcqp_predictor_resource(
    *,
    predictor_type,
    prediction_value,
    seed,
    n_samples,
    runtime_device,
    runtime_dtype,
    problem_args,
    context,
    save_dir,
    output_dir,
    retrain,
    predictor_model_config,
    predictor_model_config_overrides,
    predictor_train_config,
    predictor_train_config_overrides,
):
    predictor_key = str(predictor_type).strip().lower()
    if predictor_key in {"constant", "constant_decision"}:
        return {
            "predictor": ConstantDecisionPredictor(
                decision_dim=int(context["qc_problem"].nvar),
                value=prediction_value,
            ),
            "train_time": 0.0,
        }, {}

    if predictor_key not in {"nn", "nn_decision", "neural", "neural_decision"}:
        raise ValueError(f"Unsupported qcqp predictor_type: {predictor_type}")

    predictor_train_args = normalize_qcqp_train_config(
        n_samples=max(int(n_samples), 64),
        batch_size=min(64, max(int(n_samples), 1)),
        total_iteration=1000,
        train_config=predictor_train_config,
        train_config_overrides=predictor_train_config_overrides,
    )
    predictor_args = build_qcqp_predictor_training_payload(
        problem_args,
        {
            **default_qcqp_predictor_args(),
            **(predictor_model_config or {}),
            **(predictor_model_config_overrides or {}),
            "seed": seed,
            "device": str(runtime_device),
            "dtype": runtime_dtype,
        },
        predictor_train_args,
        ensure_results_save_freq=True,
    )
    model, record, model_path, record_path = load_or_train_decision_predictor(
        context["qc_problem"],
        predictor_args,
        save_dir / "decision_predictor",
        retrain=retrain,
    )
    predictor = NeuralDecisionPredictor(model, context["qc_problem"])
    artifacts = artifact_mapping(
        output_dir,
        predictor_model=model_path,
        predictor_training_record=record_path,
    )
    return {
        "predictor": predictor,
        "train_time": training_time_from_record(record),
    }, artifacts


def _run_qcqp_learning_baseline(
    *,
    baseline,
    context,
    n_samples,
    predictor,
    predictor_train_time=0.0,
    solver_run_config,
    projection_config,
    homeomorphic_model=None,
    homeomorphic_train_time=0.0,
    ipnn_model=None,
    ipnn_train_time=0.0,
    reference_y=None,
):
    qc_problem = context["qc_problem"]
    problem = as_learning_problem(qc_problem)
    input_samples = context["instance_batch"].inputs
    refiner = _make_qcqp_learning_refiner(
        baseline=baseline,
        qc_problem=qc_problem,
        solver_run_config=solver_run_config,
        projection_config=projection_config,
        homeomorphic_model=homeomorphic_model,
        ipnn_model=ipnn_model,
    )

    start = perf_counter()
    summary = summarize_prediction_route(
        problem,
        input_samples,
        predictor,
        refiner,
        tol=projection_config["proj_eps"],
        include_predictions=True,
        objective_batch=context["instance_batch"].objectives,
        reference_y=reference_y,
    )
    runtime_sec = float(perf_counter() - start)
    raw_predictions = summary.pop("raw_predictions")
    refined_predictions = summary.pop("refined_predictions")
    row, result, prediction_entry = build_qcqp_learning_baseline_output(
        baseline=baseline,
        summary=summary,
        runtime_sec=runtime_sec,
        n_samples=n_samples,
        raw_predictions=raw_predictions,
        refined_predictions=refined_predictions,
        predictor_train_time=predictor_train_time,
        postprocess_train_time=homeomorphic_train_time if baseline == "predict_homeomorphic_projection" else (
            ipnn_train_time if baseline == "predict_ip_bisection" else 0.0
        ),
    )
    result["predictor_train_time_sec"] = float(predictor_train_time)
    if baseline == "predict_homeomorphic_projection":
        result["train_time_sec"] = float(homeomorphic_train_time)
        result["postprocess_train_time_sec"] = float(homeomorphic_train_time)
    elif baseline == "predict_ip_bisection":
        result["train_time_sec"] = float(ipnn_train_time)
        result["postprocess_train_time_sec"] = float(ipnn_train_time)
    else:
        result["train_time_sec"] = 0.0
        result["postprocess_train_time_sec"] = 0.0
    result["refine_time_sec"] = float(runtime_sec)
    return row, result, prediction_entry


def _compute_qcqp_reference_solutions(context, solver_run_config):
    qc_problem = context["qc_problem"]
    instance_batch = context["instance_batch"]
    inputs = instance_batch.inputs
    references = []
    solver_rows = []
    total_runtime = 0.0
    for idx in range(len(instance_batch)):
        instance = instance_batch.get_instance(idx)
        bound_problem = bind_problem_instance(qc_problem, instance)
        solver = QCQPSolver(bound_problem.prob_para)
        start = perf_counter()
        result = solve_exact_result(solver, solve_config={"solve_type": "opt", **solver_run_config})
        elapsed = float(perf_counter() - start)
        total_runtime += elapsed
        solution = result.get("solution")
        if solution is None:
            raise RuntimeError(f"Reference solver returned no solution for learning test instance {idx}.")
        solution_tensor = torch.as_tensor(solution, dtype=inputs.dtype, device=inputs.device).view(1, -1)
        references.append(solution_tensor)
        objective = result.get("objective_value")
        if objective is None:
            with torch.no_grad():
                objective = float(qc_problem.objective_xy(inputs[idx].view(1, -1), solution_tensor, objective_batch=instance.objective_data).reshape(-1)[0].item())
        solver_rows.append(
            {
                "instance_id": int(idx),
                "status": result.get("status"),
                "objective": float(objective),
                "runtime_sec": elapsed,
            }
        )
    return {
        "solutions": torch.cat(references, dim=0),
        "rows": solver_rows,
        "runtime_total": float(total_runtime),
        "runtime_mean": float(total_runtime / max(len(instance_batch), 1)),
    }


def inn_training_benchmark(
    seed=2025,
    n_samples=16,
    batch_size=4,
    total_iteration=4,
    max_iterations=None,
    problem_config=None,
    problem_config_overrides=None,
    model_config=None,
    model_config_overrides=None,
    optimizer_config=None,
    optimizer_config_overrides=None,
    train_config=None,
    train_config_overrides=None,
    visualize=False,
    visualize_mdh_mapping=False,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Package-native INN training smoke with artifact output."""

    train_args = normalize_qcqp_train_config(
        n_samples=n_samples,
        batch_size=batch_size,
        total_iteration=total_iteration,
        train_config=train_config,
        train_config_overrides=train_config_overrides,
    )
    params = {
        "seed": seed,
        "n_var": int((problem_config or {}).get("n_var", 2)),
        "n_qua_cons": int((problem_config or {}).get("n_qua_cons", 1)),
        "n_linear_cons": int((problem_config or {}).get("n_linear_cons", 0)),
        "num_test_instance": 1,
        "max_iterations": 5 if max_iterations is None else int(max_iterations),
        "problem_config": problem_config,
        "problem_config_overrides": problem_config_overrides,
        "model_config": model_config,
        "model_config_overrides": model_config_overrides,
        "optimizer_config": optimizer_config,
        "optimizer_config_overrides": optimizer_config_overrides,
        "train_config": train_args,
        "retrain": True,
        "visualize": visualize,
        "visualize_mdh_mapping": visualize_mdh_mapping,
        "lagrangian_baselines": [],
        "compare_ipopt_baseline": False,
        "output_dir": output_dir,
        "device": device,
        "dtype": dtype,
    }
    return qcqp_inn_experiment(**params)


def _build_qcqp_inn_case_context(output_dir, params):
    params = _resolve_qcqp_num_test_instance(dict(params))
    seed = int(params.get("seed", 2025))
    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(
        params.get("device"),
        params.get("dtype"),
        default_device="auto",
    )
    artifact_dir = artifact_root(output_dir)
    normalized = normalize_qcqp_inn_benchmark_config(
        seed=seed,
        n_var=int(params.get("n_var", 2)),
        n_qua_cons=int(params.get("n_qua_cons", 3)),
        n_linear_cons=int(params.get("n_linear_cons", 0)),
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        n_samples=10000,
        batch_size=256,
        total_iteration=10000,
        base_inn_args=default_qcqp_inn_args(),
        base_optimizer_config=default_qcqp_optimizer_config(),
        problem_config=params.get("problem_config"),
        problem_config_overrides=params.get("problem_config_overrides"),
        model_config=params.get("model_config"),
        model_config_overrides=params.get("model_config_overrides"),
        optimizer_config=params.get("optimizer_config"),
        optimizer_config_overrides=params.get("optimizer_config_overrides"),
        train_config=params.get("train_config"),
        train_config_overrides=params.get("train_config_overrides"),
    )
    problem_args = normalized["problem"]
    problem_context = build_qcqp_problem_context(
        seed=seed,
        problem_args=problem_args,
        device=runtime_device,
        dtype=runtime_dtype,
    )
    data = problem_context["qc_problem"]
    training_payload = build_qcqp_inn_training_payload(
        problem_args,
        normalized["model"],
        normalized["train"],
        ensure_results_save_freq=True,
    )
    save_dir = _qcqp_training_cache_dir(
        output_dir,
        problem_args=problem_args,
        model_args=normalized["model"],
        train_args=normalized["train"],
        seed=seed,
        dtype=runtime_dtype,
    )
    return {
        "params": params,
        "seed": seed,
        "runtime_device": runtime_device,
        "runtime_dtype": runtime_dtype,
        "artifact_dir": artifact_dir,
        "save_dir": save_dir,
        "normalized": normalized,
        "problem_args": problem_args,
        "model_args": normalized["model"],
        "optimizer_args": normalized["optimizer"],
        "train_args": normalized["train"],
        "data": data,
        "training_payload": training_payload,
    }


def jcc_problem_benchmark(
    num_bus=30,
    n_scenarios=3,
    demand_std=0.05,
    seed=2025,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Package-native JCC smoke that records scenario feasibility data."""

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    problem = JCCDCOPFProblem(
        num_bus=num_bus,
        config={
            "n_scenarios": n_scenarios,
            "demand_std": demand_std,
            "seed": seed,
        },
    ).to_device(runtime_device)
    problem = cast_tensors_to_dtype(problem, runtime_dtype)
    problem = bind_singleton_problem_instance(problem, seed=seed)
    midpoint = ((problem.P_min + problem.P_max) / 2).view(1, -1)
    objective = float(problem.objective_x(midpoint).item())
    chance_violation = float(problem.constraint_x(midpoint).item())
    robust_violation = float(problem.robust_constraint_x(midpoint).item())
    scenario_feasibility = problem.compute_scenario_feasibility(midpoint).detach().cpu().numpy()
    feasibility_rate = float(np.mean(scenario_feasibility))

    artifact_dir = artifact_root(output_dir)
    artifacts = {}
    if artifact_dir is not None:
        scenario_path = save_numpy(artifact_dir / "jcc_scenarios.npy", scenario_feasibility)
        summary_path = save_json(
            artifact_dir / "jcc_summary.json",
            {
                "feasibility_rate": feasibility_rate,
                "objective": objective,
                "chance_violation": chance_violation,
                "robust_violation": robust_violation,
                "n_gen_vars": int(problem.n_gen_vars),
                "n_scenarios": int(problem.n_scenarios),
            },
        )
        artifacts = artifact_mapping(output_dir, scenario_feasibility=scenario_path, summary=summary_path)

    return build_benchmark_payload(
        objective=objective,
        feasible=chance_violation <= 1e-5,
        artifacts=artifacts,
        chance_violation=chance_violation,
        robust_violation=robust_violation,
        feasibility_rate=feasibility_rate,
        n_gen_vars=int(problem.n_gen_vars),
        n_scenarios=int(problem.n_scenarios),
        num_bus=int(problem.n_bus),
        seed=seed,
    )


def jcc_linear_solver_benchmark(
    seed=2025,
    problem_family="jcc_linear",
    n_var=8,
    n_input_dim=4,
    n_ineq=4,
    n_scenarios=20,
    epsilon=0.1,
    n_samples=8,
    problem_config=None,
    problem_config_overrides=None,
    solver_configs=None,
    solver_config_overrides=None,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Parametric JCC linear benchmark comparing the three exact solver baselines."""

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    problem_args = normalize_jcc_linear_problem_config(
        problem_family=problem_family,
        seed=seed,
        n_var=n_var,
        n_input_dim=n_input_dim,
        n_ineq=n_ineq,
        n_scenarios=n_scenarios,
        epsilon=epsilon,
        problem_config=problem_config,
        problem_config_overrides=problem_config_overrides,
    )
    solver_cfgs = normalize_jcc_linear_solver_configs(
        solver_configs=solver_configs,
        solver_config_overrides=solver_config_overrides,
    )
    resolved_family = str(problem_args.get("problem_family", "jcc_linear")).strip().lower()
    problem_cls = JCCLinearProblem if resolved_family == "jcc_linear" else JCCIMProblem
    problem = problem_cls(config=problem_args).to_device(runtime_device).to_dtype(runtime_dtype)
    instance_batch = problem.sample_instance_batch(n_instances=n_samples, seed=seed)

    solver_specs = (
        ("mixed_integer", JCCLinearSolver, solver_cfgs["mixed_integer"]),
        ("cvar", JCCLinearCVaRSolver, solver_cfgs["cvar"]),
        ("scenario", JCCLinearRobustScenarioSolver, solver_cfgs["scenario"]),
    )
    rows = []
    for idx in range(len(instance_batch)):
        bound_problem = bind_problem_instance(problem, instance_batch.get_instance(idx))
        for method, solver_cls, cfg in solver_specs:
            solver = solver_cls(bound_problem, solver_config=cfg)
            result = solve_exact_result(solver)
            rows.append(
                {
                    **make_comparison_row(
                        route="solver",
                        method=method,
                        objective=result.get("objective"),
                        feasible=bool(result.get("feasible")),
                        violation=result.get("violation"),
                        runtime_total=float(result.get("runtime_total") if result.get("runtime_total") is not None else 0.0),
                        total_wall_time=result.get("runtime_total"),
                    ),
                    "instance_id": int(idx),
                    "status": str(result.get("status", "unknown")),
                }
            )

    summary_rows = summarize_comparison_rows_by_method(rows)
    comparison_views = build_comparison_views(rows, summary_rows=summary_rows)
    solver_summary = {
        row["method"]: {
            "objective_mean": row["objective_mean"],
            "feasibility_rate": row["feasibility_rate"],
            "violation_mean": row["violation_mean"],
            "runtime_mean": row["runtime_mean"],
            "runtime_total": row["runtime_total"],
            "num_instances": row["num_instances"],
        }
        for row in summary_rows
    }
    metrics = {
        "seed": int(seed),
        "problem_family": resolved_family,
        "n_samples": int(n_samples),
        "n_var": int(problem.nvar),
        "n_input_dim": int(problem.n_input_dim),
        "n_ineq": int(problem.n_ineq),
        "n_scenarios": int(problem.n_scenarios),
        "epsilon": float(problem.epsilon),
        "enabled_solvers": [name for name, _, _ in solver_specs],
        "results": comparison_views["comparison_rows"],
        "instance_rows": comparison_views["instance_rows"],
        "method_summary_rows": comparison_views["method_summary_rows"],
        "comparison_rows": comparison_views["comparison_rows"],
        "comparison_summary": comparison_views["comparison_summary"],
        "comparison_summary_rows": comparison_views["comparison_summary_rows"],
        "best_method": comparison_views["best_method"],
        "solver_summary": solver_summary,
    }
    artifacts = _save_jcc_linear_solver_artifacts(
        output_dir,
        metrics=metrics,
        rows=rows,
        summary_rows=summary_rows,
    )
    return build_benchmark_payload(
        objective=comparison_views["objective"],
        feasible=comparison_views["feasible"],
        artifacts=artifacts,
        **metrics,
    )


def qcqp_inn_comparison(
    seed=2025,
    n_var=2,
    n_qua_cons=3,
    n_linear_cons=0,
    num_test_instance=1,
    max_iterations=10,
    problem_config=None,
    problem_config_overrides=None,
    model_config=None,
    model_config_overrides=None,
    optimizer_config=None,
    optimizer_config_overrides=None,
    train_config=None,
    train_config_overrides=None,
    retrain=False,
    output_dir=None,
    device=None,
    dtype=None,
    visualize=True,
    visualize_instance_idx=0,
    visualize_mdh_mapping=False,
    convergence_reference_objective=None,
    convergence_reference_objectives=None,
    show_convergence_legend=True,
    violation_y_min=1e-8,
    plot_case_convergence=True,
    plot_objective_gap=True,
    plot_gap_plus_violation=True,
    plot_runtime_summary=True,
    lagrangian_baselines=None,
    lagrangian_baseline_config=None,
    case_context=None,
    instance_batch=None,
):
    """Package-owned QCQP INN comparison for nonconvex inequality problems."""

    if case_context is None:
        raise ValueError(
            "qcqp_inn_comparison requires an explicit case_context. "
            "Use qcqp_inn_experiment or _build_qcqp_inn_case_context so all solvers share the same problem."
        )
    seed = case_context["seed"]
    runtime_device = case_context["runtime_device"]
    runtime_dtype = case_context["runtime_dtype"]
    artifact_dir = case_context["artifact_dir"]
    save_dir = case_context["save_dir"]
    run_artifact_dir = artifact_dir if artifact_dir is not None else save_dir
    problem_args = case_context["problem_args"]
    optimizer_args = case_context["optimizer_args"]
    data = case_context["data"]
    n_samples = int(case_context["params"]["num_test_instance"])
    n_var = int(problem_args["n_var"])
    n_qua_cons = int(problem_args["n_qua_cons"])
    model, training_record, model_path, record_path = _load_or_train_qcqp_inn_case(case_context, retrain=retrain)

    if instance_batch is None:
        raise ValueError(
            "qcqp_inn_comparison requires an explicit instance_batch. "
            "Sample it once with _sample_qcqp_inn_test_instances and pass it to every compared solver."
        )
    if len(instance_batch) != n_samples:
        raise ValueError(f"Expected {n_samples} QCQP test instances, got {len(instance_batch)}.")
    initial_latent = _qcqp_inn_initial_latent(
        data,
        instance_batch.inputs,
        mode=optimizer_args.get("initial_latent_mode", "center"),
        radius=optimizer_args.get("initial_latent_radius", 1.0),
        seed=seed,
    )
    initial_decision = _qcqp_inn_initial_decision(data, model, instance_batch.inputs, initial_latent)
    optimizer = INNPGDOptimizer(problem=data, paras={**optimizer_args, "max_iterations": max_iterations}, model=model)
    optimize_start = perf_counter()
    x_opt, decision_traj, latent_traj, obj_traj, cons_traj, per_iter_time = optimizer.optimize(
        initial_point=initial_latent,
        input_params=instance_batch.inputs,
        objective_params=instance_batch.objectives,
        seed=seed,
    )
    total_wall_time = perf_counter() - optimize_start
    inn_record = {
        "x_opt": x_opt.detach().cpu().numpy(),
        "x_trajectory": decision_traj.detach().cpu().numpy() if hasattr(decision_traj, "detach") else [],
        "z_trajectory": latent_traj.detach().cpu().numpy() if hasattr(latent_traj, "detach") else [],
        "obj_trajectory": obj_traj.detach().cpu().numpy(),
        "cons_trajectory": cons_traj.detach().cpu().numpy(),
        "per_iter_time": per_iter_time,
        "total_wall_time": total_wall_time,
        "initial_latent_mode": str(optimizer_args.get("initial_latent_mode", "center")),
        "initial_latent_radius": float(optimizer_args.get("initial_latent_radius", 1.0)),
    }
    summary_path = run_artifact_dir / "inn_qcqp_summary.json"
    summary = {
        "final_objective": float(obj_traj.detach().cpu().view(-1)[-int(n_samples) :].mean().item()),
        "final_violation": float(cons_traj.detach().cpu().view(-1)[-int(n_samples) :].max().item()),
        "iterations": int(len(per_iter_time)),
        "n_var": int(n_var),
        "n_qua_cons": int(n_qua_cons),
        "n_samples": int(n_samples),
        "training_penalty_final": final_or_nan(training_record.get("penalty_list", [])),
        "training_volume_final": final_or_nan(training_record.get("volume_list", [])),
        "latent_dim": trajectory_dim(latent_traj, data.nvar),
        "decision_dim": trajectory_dim(decision_traj, data.nvar),
        "x_norm": float(torch.norm(x_opt).item()),
        "initial_latent_mode": str(optimizer_args.get("initial_latent_mode", "center")),
        "initial_latent_radius": float(optimizer_args.get("initial_latent_radius", 1.0)),
        **timing_extras_from_iter_time(per_iter_time, total_wall_time=total_wall_time),
    }
    save_json(summary_path, summary)
    aggregate = summarize_qcqp_inn_record(inn_record, n_samples)
    artifacts = artifact_mapping(
        output_dir,
        model=model_path,
        training_record=record_path,
        summary=summary_path,
    )
    visualize_indices = _visualize_instance_indices(visualize_instance_idx, n_samples)
    multi_visualize = len(visualize_indices) > 1
    if bool(visualize) and artifact_dir is not None and int(n_var) == 2:
        for instance_idx in visualize_indices:
            instance_artifacts = save_inn_training_visualizations(
                output_dir=output_dir,
                artifact_root_path=artifact_dir,
                training_record=training_record,
                obj_traj=_select_batched_history(obj_traj, n_samples, instance_idx),
                cons_traj=_select_batched_history(cons_traj, n_samples, instance_idx),
                decision_traj=_select_batched_history(decision_traj, n_samples, instance_idx),
                problem=data,
                input_params=instance_batch.inputs[instance_idx : instance_idx + 1],
                objective_params=instance_batch.objectives[instance_idx : instance_idx + 1],
                instance_idx=instance_idx,
                plot_optimizer_diagnostics=False,
            )
            _merge_instance_visualization_artifacts(artifacts, instance_artifacts, instance_idx, multi_visualize)
        if bool(visualize_mdh_mapping):
            from homopt.viz import visualize_mdh_mapping_transformation

            mdh_base = artifact_dir / "mdh_mapping.pdf"
            fig = visualize_mdh_mapping_transformation(
                model,
                data,
                save_path=str(mdh_base),
                seed=seed,
            )
            if fig is not None:
                import matplotlib.pyplot as plt

                plt.close(fig)
                mdh_path = artifact_dir / "mdh_mapping_mdh_mapping_visualization.pdf"
                if mdh_path.exists():
                    artifacts["mdh_mapping_transformation"] = artifact_ref(output_dir, mdh_path)

    iterative_rows, iterative_records = _run_qcqp_iterative_baselines(
        data,
        instance_batch,
        lagrangian_baselines,
        seed=seed,
        baseline_config=lagrangian_baseline_config,
        initial_points=initial_decision,
    )
    if iterative_rows:
        iterative_json, iterative_csv = _save_qcqp_iterative_baseline_summary(output_dir, iterative_rows)
        artifacts.update(
            artifact_mapping(
                output_dir,
                iterative_baseline_json=iterative_json,
                iterative_baseline_csv=iterative_csv,
            )
        )

    comparison_records = {QCQP_INN_METHOD_NAME: inn_record, **iterative_records}
    comparison_summaries = {
        QCQP_INN_METHOD_NAME: {
            **summary,
            **aggregate,
            "final_objective": summary["final_objective"],
            "final_violation": summary["final_violation"],
        }
    }
    for row in iterative_rows:
        method = row["method"]
        comparison_summaries[method] = {
            "final_objective": row.get("objective_mean"),
            "final_violation": row.get("violation_mean"),
            "total_wall_time": row.get("runtime_total"),
            "total_iter_time": row.get("runtime_total"),
            "iterations": row.get("iterations_mean"),
            **row,
        }
    record_artifacts, _, _, _ = save_incremental_comparison_artifacts(
        output_dir,
        records=comparison_records,
        summaries=comparison_summaries,
        algorithm_order=[QCQP_INN_METHOD_NAME, *normalize_inn_lagrangian_baselines(lagrangian_baselines)],
        manifest_metadata={
            "problem": "qcqp_inn",
            "n_var": n_var,
            "n_qua_cons": n_qua_cons,
            "n_samples": n_samples,
        },
    )
    artifacts.update(record_artifacts)

    if bool(visualize) and artifact_dir is not None and int(n_var) == 2:
        for instance_idx in visualize_indices:
            instance_artifacts = save_qcqp_2d_comparison_visualizations(
                output_dir,
                artifact_dir,
                problem=data,
                model=model,
                input_params=instance_batch.inputs[instance_idx : instance_idx + 1],
                objective_params=instance_batch.objectives[instance_idx : instance_idx + 1],
                inn_record=inn_record,
                iterative_records=iterative_records,
                n_samples=n_samples,
                instance_idx=instance_idx,
                reference_objective=_reference_objective_for_instance(
                    convergence_reference_objective,
                    convergence_reference_objectives,
                    instance_idx,
                ),
                show_convergence_legend=show_convergence_legend,
                violation_y_min=violation_y_min,
                plot_convergence=plot_case_convergence,
                plot_objective_gap=plot_objective_gap,
                plot_gap_plus_violation=plot_gap_plus_violation,
                plot_runtime_summary=plot_runtime_summary,
            )
            _merge_instance_visualization_artifacts(artifacts, instance_artifacts, instance_idx, multi_visualize)

    return build_benchmark_payload(
        objective=aggregate.get("objective_mean") if aggregate.get("objective_mean") is not None else summary["final_objective"],
        feasible=aggregate.get("feasibility_rate", 0.0) >= 1.0,
        artifacts=artifacts,
        **aggregate,
        **summary,
        iterative_baseline_rows=iterative_rows,
    )


def qcqp_learning_benchmark(
    seed=7,
    n_var=10,
    n_qua_cons=10,
    n_linear_cons=0,
    n_samples=8,
    predictor_type="constant",
    prediction_value=1.5,
    baselines=None,
    problem_config=None,
    problem_config_overrides=None,
    retrain=False,
    predictor_model_config=None,
    predictor_model_config_overrides=None,
    predictor_train_config=None,
    predictor_train_config_overrides=None,
    model_config=None,
    model_config_overrides=None,
    train_config=None,
    train_config_overrides=None,
    ipnn_model_config=None,
    ipnn_model_config_overrides=None,
    ipnn_train_config=None,
    ipnn_train_config_overrides=None,
    projection_config=None,
    projection_config_overrides=None,
    solver_config=None,
    solver_config_overrides=None,
    warmstart_solver_name="ipopt",
    warmstart_solver_options=None,
    compute_reference=False,
    output_dir=None,
    device=None,
    dtype=None,
    context=None,
):
    """Package-owned QCQP learning-route benchmark using shared result/eval surfaces."""

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    enabled = normalize_qcqp_learning_baselines(QCQP_LEARNING_BASELINES if baselines is None else baselines)
    if not enabled:
        raise ValueError("qcqp_learning_benchmark requires at least one baseline.")

    results = {}
    prediction_payload = {}
    rows = []
    if context is not None:
        problem_args = dict(context["problem_args"])
        n_samples = len(context["instance_batch"])
        n_var = int(problem_args["n_var"])
        n_qua_cons = int(problem_args["n_qua_cons"])
        n_linear_cons = int(problem_args.get("n_linear_cons", 0))
    else:
        problem_args = normalize_qcqp_problem_config(
            seed=seed,
            n_var=n_var,
            n_qua_cons=n_qua_cons,
            n_linear_cons=n_linear_cons,
            problem_config=problem_config,
            problem_config_overrides=problem_config_overrides,
        )
    solver_run_config = normalize_solver_run_config(
        solver_name=warmstart_solver_name,
        solver_options=warmstart_solver_options,
        solver_config=solver_config,
        solver_config_overrides=solver_config_overrides,
    )
    projection_args = {
        **default_qcqp_projection_args(),
        **(projection_config or {}),
        **(projection_config_overrides or {}),
    }
    if context is None:
        context = build_qcqp_learning_context(
            seed=seed,
            n_var=n_var,
            n_qua_cons=n_qua_cons,
            n_linear_cons=n_linear_cons,
            n_samples=n_samples,
            problem_args=problem_args,
            device=runtime_device,
            dtype=runtime_dtype,
            sample_obj=("predict_warm_start" in enabled or bool(compute_reference)),
        )
    reference = None
    reference_artifacts = {}
    if compute_reference:
        reference = _compute_qcqp_reference_solutions(context, solver_run_config)
        reference_path = save_json(artifact_root(output_dir) / "qcqp_learning_reference_solver.json", reference["rows"]) if output_dir is not None else None
        reference_artifacts = artifact_mapping(output_dir, reference_solver=reference_path)
    artifact_dir = artifact_root(output_dir)
    save_dir = artifact_dir if artifact_dir is not None else ensure_dir(Path("results") / "_scratch" / "qcqp_learning")
    predictor_resource, predictor_artifacts = _prepare_qcqp_predictor_resource(
        predictor_type=predictor_type,
        prediction_value=prediction_value,
        seed=seed,
        n_samples=n_samples,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        problem_args=problem_args,
        context=context,
        save_dir=save_dir,
        output_dir=output_dir,
        retrain=retrain,
        predictor_model_config=predictor_model_config,
        predictor_model_config_overrides=predictor_model_config_overrides,
        predictor_train_config=predictor_train_config,
        predictor_train_config_overrides=predictor_train_config_overrides,
    )
    resources, baseline_artifacts = _prepare_qcqp_postprocess_resources(
        enabled=enabled,
        seed=seed,
        n_var=n_var,
        n_qua_cons=n_qua_cons,
        n_linear_cons=n_linear_cons,
        n_samples=n_samples,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        problem_args=problem_args,
        context=context,
        save_dir=save_dir,
        output_dir=output_dir,
        retrain=retrain,
        model_config=model_config,
        model_config_overrides=model_config_overrides,
        train_config=train_config,
        train_config_overrides=train_config_overrides,
        ipnn_model_config=ipnn_model_config,
        ipnn_model_config_overrides=ipnn_model_config_overrides,
        ipnn_train_config=ipnn_train_config,
        ipnn_train_config_overrides=ipnn_train_config_overrides,
    )

    for baseline in enabled:
        row, result, prediction_entry = _run_qcqp_learning_baseline(
            baseline=baseline,
            context=context,
            n_samples=n_samples,
            predictor=predictor_resource["predictor"],
            predictor_train_time=predictor_resource["train_time"],
            solver_run_config=solver_run_config,
            projection_config=projection_args,
            homeomorphic_model=resources["homeomorphic_model"],
            homeomorphic_train_time=resources["homeomorphic_train_time"],
            ipnn_model=resources["ipnn_model"],
            ipnn_train_time=resources["ipnn_train_time"],
            reference_y=None if reference is None else reference["solutions"],
        )
        rows.append(row)
        prediction_payload[baseline] = prediction_entry
        results[baseline] = result

    metrics = build_qcqp_learning_benchmark_metrics(
        results=results,
        enabled_baselines=enabled,
        seed=seed,
        n_var=n_var,
        n_qua_cons=n_qua_cons,
        n_linear_cons=n_linear_cons,
        n_samples=n_samples,
        prediction_value=prediction_value,
    )
    if reference is not None:
        metrics["reference_solver_runtime_total"] = reference["runtime_total"]
        metrics["reference_solver_runtime_mean"] = reference["runtime_mean"]

    artifacts = _save_qcqp_learning_artifacts(
        output_dir,
        metrics=metrics,
        rows=rows,
        prediction_payload=prediction_payload,
    )
    artifacts.update(predictor_artifacts)
    artifacts.update(baseline_artifacts)
    artifacts.update(reference_artifacts)

    return build_benchmark_payload(
        objective=float(metrics["refined_objective_mean"]),
        feasible=metrics["refined_feasibility_rate"] >= 1.0 - 1e-12,
        artifacts=artifacts,
        **metrics,
    )


def _validate_qcqp_route_shared_params(iterative_cfg, learning_cfg):
    shared = dict(iterative_cfg)
    for key in ("seed", "n_var", "n_qua_cons", "n_linear_cons", "problem_config", "problem_config_overrides", "device", "dtype"):
        if key in learning_cfg:
            if key in shared and shared[key] != learning_cfg[key]:
                raise ValueError(f"QCQP route comparison requires shared {key}; got different iterative and learning values.")
            shared[key] = learning_cfg[key]
    learning_samples = learning_cfg.get("n_samples")
    iterative_samples = iterative_cfg.get("num_test_instance")
    if learning_samples is not None and iterative_samples is not None and int(learning_samples) != int(iterative_samples):
        raise ValueError(
            "QCQP route comparison requires learning_params['n_samples'] to match "
            "iterative_params['num_test_instance']."
        )
    if iterative_samples is None:
        shared["num_test_instance"] = int(learning_samples) if learning_samples is not None else int(shared.get("num_test_instance", 1))
    return shared


def _qcqp_learning_context_from_inn_case(case_context, instance_batch):
    return {
        "qc_problem": case_context["data"],
        "instance_batch": instance_batch,
        "problem_args": case_context["problem_args"],
    }


def qcqp_route_comparison(
    output_dir=None,
    iterative_params=None,
    learning_params=None,
):
    """Run QCQP iterative and learning routes under one comparison surface."""

    iterative_cfg = dict(iterative_params or {})
    learning_cfg = dict(learning_params or {})
    shared_params = _validate_qcqp_route_shared_params(iterative_cfg, learning_cfg)

    iterative_out = Path(output_dir) / "iterative" if output_dir is not None else None
    learning_out = Path(output_dir) / "learning" if output_dir is not None else None

    shared_context = _build_qcqp_inn_case_context(iterative_out, shared_params)
    shared_instances = _sample_qcqp_inn_test_instances(shared_context)
    learning_context = _qcqp_learning_context_from_inn_case(shared_context, shared_instances)

    iterative_payload = qcqp_inn_comparison(
        output_dir=iterative_out,
        case_context=shared_context,
        instance_batch=shared_instances,
        **iterative_cfg,
    )
    learning_payload = qcqp_learning_benchmark(
        output_dir=learning_out,
        context=learning_context,
        n_samples=len(shared_instances),
        **{key: value for key, value in learning_cfg.items() if key != "n_samples"},
    )

    rows = build_qcqp_route_rows(
        iterative_payload=iterative_payload,
        learning_results=learning_payload.get("results", {}),
    )

    artifacts = {}
    if output_dir is not None:
        artifact_json, artifact_csv, artifact_md = _save_qcqp_route_comparison_summary(output_dir, rows)
        artifacts = artifact_mapping(
            output_dir,
            summary_json=artifact_json,
            summary_csv=artifact_csv,
            summary_markdown=artifact_md,
        )

    learning_results = learning_payload.get("results", {})
    comparison_views = build_qcqp_route_comparison_views(
        rows=rows,
        iterative_payload=iterative_payload,
        learning_results=learning_results,
    )

    return build_benchmark_payload(
        objective=comparison_views["objective"],
        feasible=comparison_views["feasible"],
        artifacts=artifacts,
        results=comparison_views["results"],
        instance_rows=comparison_views["instance_rows"],
        method_summary_rows=comparison_views["method_summary_rows"],
        comparison_rows=comparison_views["comparison_rows"],
        comparison_summary=comparison_views["comparison_summary"],
        comparison_summary_rows=comparison_views["comparison_summary_rows"],
        iterative=comparison_views["iterative"],
        learning=comparison_views["learning"],
        learning_summary=comparison_views["learning_summary"],
        best_method=comparison_views["best_method"],
    )

def _load_or_train_qcqp_inn_case(case_context, *, retrain):
    return load_or_train_inn_mapping(
        case_context["data"],
        case_context["training_payload"],
        case_context["save_dir"],
        retrain=bool(retrain),
    )


def _sample_qcqp_inn_test_instances(case_context):
    """Sample the test instance batch once and pass it to every compared solver."""

    return case_context["data"].sample_instance_batch(
        n_instances=int(case_context["params"]["num_test_instance"]),
        seed=int(case_context["seed"]),
        sample_obj=True,
        device=case_context["runtime_device"],
        dtype=case_context["runtime_dtype"],
    )


def _prepare_qcqp_inn_training_case(output_dir, params):
    """Prepare the INN mapping for one QCQP case without running test baselines."""

    case_context = _build_qcqp_inn_case_context(output_dir, params)
    return _load_or_train_qcqp_inn_case(case_context, retrain=case_context["params"].get("retrain", False))


def _prepare_qcqp_inn_training_context(case_context):
    """Prepare the INN mapping from an already-built QCQP case context."""

    return _load_or_train_qcqp_inn_case(
        case_context,
        retrain=case_context["params"].get("retrain", False),
    )


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
                "ipopt_comparison_rows": [compare_row],
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
        row = {
            "n_var": int(test_params["n_var"]),
            "n_qua_cons": int(test_params["n_qua_cons"]),
            "max_iterations": case_max_iterations,
            "objective": payload.get("objective"),
            "feasible": payload.get("feasible"),
            "final_violation": payload.get("final_violation"),
            "iterations": payload.get("iterations"),
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

__all__ = [
    "PARAMETRIC_BENCHMARKS",
    "inn_training_benchmark",
    "jcc_linear_solver_benchmark",
    "jcc_problem_benchmark",
    "qcqp_inn_experiment",
    "qcqp_inn_sensitivity_sweep",
    "qcqp_learning_benchmark",
    "qcqp_route_comparison",
]
