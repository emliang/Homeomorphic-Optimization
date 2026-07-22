"""QCQP INN single-case comparison benchmark."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from homopt.records.artifacts import (
    artifact_mapping,
    artifact_ref,
    build_benchmark_payload,
    save_incremental_comparison_artifacts,
    save_json,
    save_table_artifacts,
)
from .reports import (
    QCQP_INN_METHOD_NAME,
    extract_qcqp_inn_instance_metrics,
    summarize_qcqp_inn_record,
)
from homopt.experiments.common.comparison import timing_extras_from_iter_time
from homopt.experiments.common.inn_baselines import (
    build_inn_lagrangian_method_params,
    inn_lagrangian_feasibility_tol,
    normalize_inn_lagrangian_baselines,
    run_inn_lagrangian_baseline,
)
from .training import (
    _load_or_train_qcqp_inn_case,
    _merge_instance_visualization_artifacts,
    _select_batched_history,
    _visualize_instance_indices,
)
from homopt.experiments.common.runtime import final_or_nan, trajectory_dim
from homopt.experiments.common.reports import summarize_single_problem_run_record
from homopt.optim import INNPGDOptimizer
from homopt.problems import bind_problem_instance
from homopt.solvers import QCQPSolver, solve_exact_result
from homopt.viz.inn_training import save_inn_training_visualizations, save_qcqp_2d_comparison_visualizations


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
    n_samples = int(case_params["num_test_instance"])
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


def _run_qcqp_iterative_baselines(
    data,
    instance_batch,
    baselines,
    *,
    seed,
    max_iterations,
    baseline_config,
    initial_points=None,
):
    baselines = normalize_inn_lagrangian_baselines(baselines)
    if not baselines:
        return [], {}
    params = build_inn_lagrangian_method_params(seed, baseline_config, max_iterations=max_iterations)
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


def qcqp_inn_comparison(
    seed=2025,
    n_var=2,
    n_qua_cons=3,
    n_linear_cons=0,
    num_test_instance=1,
    max_iterations=10,
    problem_config=None,
    model_config=None,
    optimizer_config=None,
    train_config=None,
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
        training_artifacts = save_inn_training_visualizations(
            output_dir=output_dir,
            artifact_root_path=artifact_dir,
            training_record=training_record,
            obj_traj=[],
            cons_traj=[],
            decision_traj=[],
            plot_optimizer_diagnostics=False,
        )
        artifacts.update(training_artifacts)
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
        max_iterations=max_iterations,
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
        run_identity={
            "benchmark": "qcqp_inn_comparison",
            "seed": seed,
            "problem_config": problem_args,
            "model_config": case_context["model_args"],
            "training_config": case_context["train_args"],
            "optimizer_config": optimizer_args,
            "max_iterations": max_iterations,
            "lagrangian_baseline_config": lagrangian_baseline_config,
            "runtime": {"device": str(runtime_device), "dtype": str(runtime_dtype)},
            "test_instances": {
                "inputs": instance_batch.inputs,
                "objectives": instance_batch.objectives,
            },
        },
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


__all__ = [
    "qcqp_inn_comparison",
    "_ipopt_reference_objectives",
    "_reference_objective_for_instance",
    "_run_qcqp_ipopt_baseline",
    "_save_qcqp_ipopt_comparison_summary",
    "_save_qcqp_ipopt_instance_rows",
    "_summarize_qcqp_ipopt_comparison",
]
