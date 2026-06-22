"""QCQP learning-route benchmark entrypoints."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from homopt.records.artifacts import artifact_mapping, artifact_root, build_benchmark_payload, save_json
from homopt.experiments.common.config import normalize_solver_run_config, require_explicit_config
from homopt.learning.training import prepare_decision_predictor, prepare_learning_mapping
from homopt.learning.training.cache import training_time_from_record
from homopt.experiments.common.learning_reports import save_parametric_learning_artifacts
from homopt.experiments.common.parametric_runtime import run_prediction_refinement_route
from .reports import (
    QCQP_LEARNING_BASELINES,
    build_qcqp_learning_baseline_output,
    build_qcqp_learning_benchmark_metrics,
    normalize_qcqp_learning_baselines,
)
from .setup import (
    build_qcqp_learning_context,
    normalize_qcqp_inn_benchmark_config,
    normalize_qcqp_problem_config,
    normalize_qcqp_train_config,
)
from homopt.experiments.common.runtime import resolve_runtime
from homopt.learning import (
    ConstantDecisionPredictor,
    DiffProjectionRefiner,
    ExactSolverProjectionRefiner,
    HomeomorphicProjectionRefiner,
    IPNNBisectionRefiner,
    IdentityRefiner,
    InitializedOptSolverRefiner,
    NeuralDecisionPredictor,
    RayBisectionRefiner,
)
from homopt.problems import bind_problem_instance
from homopt.solvers import QCQPSolver, solve_exact_result
from homopt.experiments.common.io import ensure_dir
from homopt.utils import set_global_seed


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
    if baseline == "predict_initialized_opt":
        return InitializedOptSolverRefiner(
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
    train_config,
    ipnn_model_config,
    ipnn_train_config,
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
            base_inn_args={},
            base_optimizer_config={},
            problem_config=problem_args,
            model_config=require_explicit_config(model_config, "model_config"),
            train_config=require_explicit_config(train_config, "train_config"),
        )
        homeo_resource = prepare_learning_mapping(
            "inn",
            context["qc_problem"],
            problem_args=homeo_cfg["problem"],
            model_args=homeo_cfg["model"],
            train_args=homeo_cfg["train"],
            save_dir=save_dir / "homeomorphic_projection",
            retrain=retrain,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
            ensure_results_save_freq=True,
        )
        model = homeo_resource["model"]
        record = homeo_resource["training_record"]
        model_path = homeo_resource["model_path"]
        record_path = homeo_resource["record_path"]
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
        )
        ipnn_resource = prepare_learning_mapping(
            "ipnn",
            context["qc_problem"],
            problem_args=problem_args,
            model_args=require_explicit_config(ipnn_model_config, "ipnn_model_config"),
            train_args=ipnn_train_args,
            save_dir=save_dir / "ipnn_projection",
            retrain=retrain,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
            ensure_results_save_freq=True,
        )
        model = ipnn_resource["model"]
        record = ipnn_resource["training_record"]
        model_path = ipnn_resource["model_path"]
        record_path = ipnn_resource["record_path"]
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
    predictor_train_config,
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
    )
    predictor_resource = prepare_decision_predictor(
        context["qc_problem"],
        problem_args=problem_args,
        model_args={
            **require_explicit_config(predictor_model_config, "predictor_model_config"),
            "seed": seed,
        },
        train_args=predictor_train_args,
        save_dir=save_dir / "decision_predictor",
        retrain=retrain,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        ensure_results_save_freq=True,
    )
    model = predictor_resource["model"]
    record = predictor_resource["training_record"]
    model_path = predictor_resource["model_path"]
    record_path = predictor_resource["record_path"]
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
    problem = qc_problem
    input_samples = context["instance_batch"].inputs
    refiner = _make_qcqp_learning_refiner(
        baseline=baseline,
        qc_problem=qc_problem,
        solver_run_config=solver_run_config,
        projection_config=projection_config,
        homeomorphic_model=homeomorphic_model,
        ipnn_model=ipnn_model,
    )

    summary, runtime_sec = run_prediction_refinement_route(
        problem=problem,
        input_samples=input_samples,
        predictor=predictor,
        refiner=refiner,
        tol=projection_config["proj_eps"],
        objective_batch=context["instance_batch"].objectives,
        reference_y=reference_y,
    )
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
    retrain=False,
    predictor_model_config=None,
    predictor_train_config=None,
    model_config=None,
    train_config=None,
    ipnn_model_config=None,
    ipnn_train_config=None,
    projection_config=None,
    solver_config=None,
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
        )
    solver_run_config = normalize_solver_run_config(
        solver_name=warmstart_solver_name,
        solver_options=warmstart_solver_options,
        solver_config=solver_config,
    )
    projection_args = {
        **require_explicit_config(projection_config, "projection_config"),
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
            sample_obj=("predict_initialized_opt" in enabled or bool(compute_reference)),
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
        predictor_train_config=predictor_train_config,
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
        train_config=train_config,
        ipnn_model_config=ipnn_model_config,
        ipnn_train_config=ipnn_train_config,
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

    artifacts = save_parametric_learning_artifacts(
        output_dir,
        metrics=metrics,
        rows=rows,
        prediction_payload=prediction_payload,
        artifact_prefix="qcqp_learning",
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


__all__ = [
    "qcqp_learning_benchmark",
]
