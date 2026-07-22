"""MaxCut SDP single-problem benchmarks."""

from __future__ import annotations

import copy
import numpy as np
import torch

from homopt.records.artifacts import (
    build_benchmark_payload,
    build_visualize_only_benchmark_payload,
    load_visualize_only_comparison_artifacts,
    save_incremental_comparison_artifacts,
)
from homopt.experiments.common.config import (
    apply_config_groups,
    enforce_alm_outer_iteration_budget,
    normalize_single_common_config,
    reject_algorithm_iteration_budget_keys,
)
from homopt.experiments.method_specs import build_first_order_method_params, build_penalty_method_params
from homopt.experiments.common.run_records import ensure_record_violation_split, summarize_run_record
from homopt.experiments.common.runtime import resolve_runtime
from homopt.mappings import GaugeMapMaxCut
from homopt.optim import run_algorithm
from homopt.problems import BMSDP, MaxCutSDP, create_maxcut_problem
from homopt.solvers import MaxCutSolver, solve_exact_result
from homopt.utils import cast_tensors_to_dtype, set_global_seed
from homopt.viz import save_comparison_visualizations


def _evaluate_bm_record_in_sdp_space(sdp_problem, factor_problem, record, *, rank):
    """Return a comparison record evaluated in the original SDP coordinates."""

    factor_trajectory = np.asarray(record.get("x_traj", []), dtype=float)
    if factor_trajectory.size == 0:
        raise RuntimeError(
            "Burer-Monteiro comparison requires an explicit factor trajectory. "
            "Set track_decisions=True for the ALM baseline."
        )
    factor_trajectory = factor_trajectory.reshape(-1, factor_problem.nvar)
    factor_tensor = torch.as_tensor(
        factor_trajectory,
        dtype=sdp_problem.p.dtype,
        device=sdp_problem.device,
    )
    with torch.no_grad():
        sdp_trajectory = factor_problem.lift_to_sdp_decision(factor_tensor)
        objective = sdp_problem.objective_x(sdp_trajectory).reshape(-1).detach().cpu().numpy()
        violation = sdp_problem.constraint_x(sdp_trajectory, clip=True).reshape(-1).detach().cpu().numpy()

        solved_factor = torch.as_tensor(
            np.asarray(record["x_solved"], dtype=float).reshape(1, -1),
            dtype=sdp_problem.p.dtype,
            device=sdp_problem.device,
        )
        solved_sdp = factor_problem.lift_to_sdp_decision(solved_factor).detach().cpu().numpy()

    evaluated = copy.deepcopy(record)
    evaluated.update(
        {
            "x_traj": sdp_trajectory.detach().cpu().numpy(),
            "x_solved": solved_sdp,
            "obj_traj": objective,
            "cons_traj": violation,
            "ineq_violation_traj": violation,
            "eq_violation_traj": np.zeros_like(violation),
            "violation_scope": "inequality",
            "decision_space": "burer_monteiro_factor",
            "evaluation_space": "maxcut_sdp",
            "factor_rank": int(rank),
            "factor_x_traj": factor_trajectory,
            "factor_x_solved": np.asarray(record["x_solved"], dtype=float),
            "factor_obj_traj": np.asarray(record.get("obj_traj", []), dtype=float),
            "factor_cons_traj": np.asarray(record.get("cons_traj", []), dtype=float),
        }
    )
    return evaluated


def maxcut_algorithm_comparison(
    algorithms=None,
    common_config=None,
    algorithm_config=None,
    seed=2025,
    expr_id=0,
    n=10,
    alpha=0.5,
    max_iterations=200,
    max_running_time=30,
    output_dir=None,
    device=None,
    dtype=None,
    visualize=True,
    visualize_only=False,
    visualization_prefix="maxcut_sdp",
    method_labels=None,
    reference_label="SDP",
    show_convergence_legend=True,
):
    """Package-owned MaxCut-SDP comparison runner."""

    del expr_id
    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    reject_algorithm_iteration_budget_keys(algorithm_config, context="algorithm_config")
    algorithms = list(algorithms or ["PGD", "ALM-log", "ALM-bp", "RD", "Hom-PGD"])

    common_params = {
        "seed": seed,
        "learning_rate": 1e-3,
        "convergence_threshold": 1e-6,
        "max_iterations": max_iterations,
        "max_running_time": max_running_time,
        "opt": "gd",
        "stepsize_rule": "adaptive",
        "acceleration_method": "none",
        "acceleration_space": "z",
        "smooth": True,
        "momentum": 0.0,
        "lr_decay": 0.999,
        "min_lr": 1e-6,
    }
    common_params = normalize_single_common_config(
        common=common_params,
        common_config=common_config,
    )
    effective_max_iterations = int(common_params["max_iterations"])
    params = {
        "common": common_params,
        **build_first_order_method_params(
            common_params,
            projection_outer_iterations=10,
            projection_inner_iterations=100,
            include_fw=False,
            include_rd=True,
            hom_p_norm=2,
            hom_momentum=0.99,
        ),
        "ALM": build_penalty_method_params(
            common_params,
            outer_iterations=effective_max_iterations,
            inner_iterations=100,
            dual_learning_rate=0.1,
            penalty_coef=10.0,
            penalty_growth=1.1,
            proximal_coef=0.1,
            proximal_space="x",
            outer_stepsize_rule=common_params["stepsize_rule"],
            inner_stepsize_rule="constant",
            max_penalty=1e2,
            max_dual=1e2,
            use_lagrangian=True,
            use_penalty=True,
            use_proximal=False,
        ),
    }
    params["ALM"]["track_decisions"] = True
    params["Hom-PGD"]["hom_map_gradient"] = "autograd"
    params = apply_config_groups(
        params,
        algorithm_config=algorithm_config,
    )
    params = enforce_alm_outer_iteration_budget(params, max_iterations=effective_max_iterations)

    config = create_maxcut_problem({"seed": seed, "n": n, "alpha": alpha})
    base_problem = cast_tensors_to_dtype(MaxCutSDP(config).to_device(runtime_device), runtime_dtype)

    if visualize_only:
        visual_state = load_visualize_only_comparison_artifacts(
            output_dir,
            algorithms=algorithms,
        )
        records = visual_state["records"]
        summaries = visual_state["summaries"]
        previous_result = visual_state["previous_result"]
        artifacts = visual_state["artifacts"]
        stored_algorithms = visual_state["stored_algorithms"]
        stored_metadata = dict(visual_state["manifest"].get("metadata", {}) or {})
        stored_reference_objective = previous_result.get("metrics", {}).get(
            "reference_objective",
            stored_metadata.get("reference_objective"),
        )
        if stored_reference_objective is not None and previous_result.get("metrics", {}).get("reference_objective") is None:
            previous_result = {
                **previous_result,
                "metrics": {
                    **dict(previous_result.get("metrics", {}) or {}),
                    "reference_objective": stored_reference_objective,
                    "reference_label": stored_metadata.get("reference_label", reference_label),
                },
            }
        if visualize and output_dir is not None:
            artifacts.update(
                save_comparison_visualizations(
                    problem=base_problem,
                    records=records,
                    algorithms=stored_algorithms,
                    output_dir=output_dir,
                    prefix=visualization_prefix,
                    reference_objective=stored_reference_objective,
                    reference_label=reference_label,
                    method_labels=method_labels,
                    violation_y_min=float(common_params["convergence_threshold"]),
                    show_convergence_legend=show_convergence_legend,
                )
            )
        return build_visualize_only_benchmark_payload(
            previous_result=previous_result,
            summaries=summaries,
            artifacts=artifacts,
            requested_algorithms=algorithms,
            algorithms=stored_algorithms,
        )

    solver = MaxCutSolver(base_problem.prob_para)
    solver_result = solve_exact_result(solver, "opt")
    x_opt = solver_result["solution"]
    x_opt_tensor = torch.tensor(
        x_opt[base_problem.upper_triangle_index[:, 0], base_problem.upper_triangle_index[:, 1]],
        dtype=runtime_dtype,
    ).view(1, -1)
    obj_opt = float(base_problem.objective_x(x_opt_tensor).detach().cpu().view(-1)[0].item())
    hom_map = GaugeMapMaxCut(
        base_problem,
        p_norm=params["Hom-PGD"].get("hom_p_norm", 2),
        smooth=bool(common_params.get("smooth", True)),
    ).to_device(runtime_device)
    hom_map = cast_tensors_to_dtype(hom_map, runtime_dtype)

    def _run_maxcut_record(algorithm):
        if algorithm.startswith("ALM-"):
            rank_key = algorithm.split("-", 1)[1]
            if rank_key == "log":
                rank = int(np.ceil(np.log(base_problem.num_node)))
            elif rank_key == "bp":
                rank = int(np.ceil(np.sqrt(2 * base_problem.num_node)))
            else:
                rank = base_problem.num_node
            problem = cast_tensors_to_dtype(BMSDP(config, rank=rank).to_device(runtime_device), runtime_dtype)
            init_point = np.random.default_rng(seed + rank).normal(size=(1, problem.nvar)) / problem.num_node
            record = run_algorithm("ALM", problem, params, hom_map=hom_map, init_point=init_point)
            return _evaluate_bm_record_in_sdp_space(base_problem, problem, record, rank=rank)
        problem = cast_tensors_to_dtype(MaxCutSDP(config).to_device(runtime_device), runtime_dtype)
        init_point = np.zeros((1, problem.nvar))
        record = run_algorithm(algorithm, problem, params, hom_map=hom_map, init_point=init_point)
        record["violation_scope"] = "inequality"
        ensure_record_violation_split(problem, record)
        return record

    records = {}
    summaries = {}
    for algorithm in algorithms:
        record = _run_maxcut_record(algorithm)
        records[algorithm] = record
        summaries[algorithm] = summarize_run_record(record, include_total_iter_time=True)

    artifacts, comparison_views, records, summaries = save_incremental_comparison_artifacts(
        output_dir,
        records=records,
        summaries=summaries,
        algorithm_order=records.keys(),
        run_identity={
            "benchmark": "maxcut_algorithm_comparison",
            "problem_config": config,
            "common_config": params["common"],
            "runtime": {"device": str(runtime_device), "dtype": str(runtime_dtype)},
            "reference_label": reference_label,
        },
        manifest_metadata={
            "problem_type": "maxcut_sdp",
            "reference_label": reference_label,
            "reference_objective": obj_opt,
        },
    )
    if visualize and output_dir is not None:
        artifacts.update(
            save_comparison_visualizations(
                problem=base_problem,
                records=records,
                algorithms=algorithms,
                output_dir=output_dir,
                prefix=visualization_prefix,
                reference_objective=obj_opt,
                reference_label=reference_label,
                method_labels=method_labels,
                violation_y_min=float(common_params["convergence_threshold"]),
                show_convergence_legend=show_convergence_legend,
            )
        )

    return build_benchmark_payload(
        objective=comparison_views["objective"],
        feasible=comparison_views["feasible"],
        artifacts=artifacts,
        algorithms=algorithms,
        method_labels=dict(method_labels or {}),
        reference_label=reference_label,
        reference_objective=obj_opt,
        results=summaries,
        instance_rows=comparison_views["instance_rows"],
        method_summary_rows=comparison_views["method_summary_rows"],
        comparison_rows=comparison_views["comparison_rows"],
        comparison_summary=comparison_views["comparison_summary"],
        comparison_summary_rows=comparison_views["comparison_summary_rows"],
        best_method=comparison_views["best_method"],
        n=int(n),
        alpha=float(alpha),
        seed=seed,
    )


__all__ = ["maxcut_algorithm_comparison"]
