"""Toy star-domain single-problem benchmarks."""

from __future__ import annotations

import numpy as np
import torch
import warnings
from time import perf_counter

from homopt.records.artifacts import (
    build_benchmark_payload,
    load_visualize_only_comparison_artifacts,
    save_incremental_comparison_artifacts,
)
from homopt.experiments.common.config import (
    apply_config_groups,
    enforce_alm_outer_iteration_budget,
    merged,
    normalize_single_common_config,
    reject_algorithm_iteration_budget_keys,
)
from homopt.experiments.common.run_records import ensure_record_violation_split, summarize_run_record
from homopt.experiments.common.runtime import resolve_runtime
from homopt.experiments.common.single_problem import _ineq_algorithm_params
from homopt.mappings import GaugeMap, PolyStarMap
from homopt.mappings.star import StarMap
from homopt.optim import run_algorithm
from homopt.problems import ConvexOpt, PolyStarOpt, ToyStarOpt, create_test_problem
from homopt.solvers import ConvexSolver
from homopt.solvers.pyomo import PYOMO_AVAILABLE, PYOMO_NLP_SOLVER, SolverStatus, TerminationCondition, pyo
from homopt.utils import cast_tensors_to_dtype, set_global_seed
from homopt.viz import save_convex_2d_visualizations


def _origin_polytope_config(poly_config, *, seed):
    config = dict(poly_config or {})
    n_linear = int(config.get("n_linear_cons", 6))
    x_lower = float(config.get("x_lower", -2.0))
    x_upper = float(config.get("x_upper", 2.0))
    if x_lower >= 0 or x_upper <= 0:
        raise ValueError("poly_star intersection requires poly_config bounds to contain the origin.")
    radius = float(config.get("poly_radius", 1.0))
    radius_jitter = float(config.get("poly_radius_jitter", 0.0))
    if radius <= 0:
        raise ValueError("poly_radius must be positive.")
    if radius_jitter < 0:
        raise ValueError("poly_radius_jitter must be nonnegative.")

    rng = np.random.RandomState(config.get("seed", seed))
    if n_linear > 0:
        angle_offset = float(config.get("angle_offset", rng.uniform(0.0, 2.0 * np.pi / max(n_linear, 1))))
        angles = angle_offset + np.linspace(0.0, 2.0 * np.pi, n_linear, endpoint=False)
        a_matrix = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        jitter = rng.uniform(-radius_jitter, radius_jitter, size=n_linear)
        b_vector = np.maximum(radius * (1.0 + jitter), 1e-6)
    else:
        a_matrix = None
        b_vector = None

    return {
        "Q": np.eye(2),
        "p": np.zeros(2),
        "U": np.full(2, x_upper),
        "L": np.full(2, x_lower),
        "A": a_matrix,
        "b": b_vector,
        "Qq": None,
        "pq": None,
        "bq": None,
        "G": None,
        "h": None,
        "C": None,
        "d": None,
    }


def _toy_problem_tensor_dtype(problem):
    if hasattr(problem, "Q") and torch.is_tensor(problem.Q):
        return problem.Q.dtype
    return torch.float32


def _toy_max_violation(problem, x_value):
    if x_value is None:
        return None
    x = torch.as_tensor(
        np.asarray(x_value, dtype=float).reshape(1, -1),
        dtype=_toy_problem_tensor_dtype(problem),
        device=problem.device,
    )
    with torch.no_grad():
        residual = problem.constraint_x(x, clip=True)
    return float(residual.max().detach().cpu().item()) if residual.numel() else 0.0


def _toy_grid_reference(problem, *, grid_size=801):
    """Dense-grid reference for nonconvex two-dimensional toy problems."""

    if int(getattr(problem, "nvar", 0)) != 2:
        return None
    if hasattr(problem, "L") and hasattr(problem, "U"):
        lower = np.asarray(problem.L.detach().cpu().numpy(), dtype=float).reshape(-1)
        upper = np.asarray(problem.U.detach().cpu().numpy(), dtype=float).reshape(-1)
    elif hasattr(problem, "poly_problem"):
        lower = np.asarray(problem.poly_problem.L.detach().cpu().numpy(), dtype=float).reshape(-1)
        upper = np.asarray(problem.poly_problem.U.detach().cpu().numpy(), dtype=float).reshape(-1)
    else:
        radius = 1.0 + abs(float(getattr(problem, "alpha", 0.0)))
        lower = np.asarray([-radius, -radius], dtype=float)
        upper = np.asarray([radius, radius], dtype=float)

    axes = [np.linspace(float(lower[i]), float(upper[i]), int(grid_size)) for i in range(2)]
    mesh = np.stack(np.meshgrid(*axes, indexing="xy"), axis=-1).reshape(-1, 2)
    x = torch.as_tensor(mesh, dtype=_toy_problem_tensor_dtype(problem), device=problem.device)
    best_x = None
    best_obj = np.inf
    batch_size = 200000
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            residual = problem.constraint_x(xb, clip=False)
            residual = torch.nan_to_num(residual, nan=np.inf, posinf=np.inf, neginf=-np.inf)
            feasible = (residual <= 1e-8).all(dim=1)
            if not feasible.any():
                continue
            obj = problem.objective_x(xb).reshape(-1)
            obj = torch.nan_to_num(obj, nan=np.inf, posinf=np.inf, neginf=np.inf)
            feasible_obj = torch.where(feasible, obj, torch.full_like(obj, np.inf))
            value, idx = torch.min(feasible_obj, dim=0)
            if float(value.item()) < best_obj:
                best_obj = float(value.item())
                best_x = xb[int(idx.item())].detach().cpu().numpy()
    if best_x is None:
        return None
    return {
        "label": "GridSearch",
        "objective": best_obj,
        "solution": np.asarray(best_x, dtype=float),
        "violation": _toy_max_violation(problem, best_x),
        "status": "grid_reference",
        "runtime": None,
    }


def _toy_ipopt_default_options():
    return {
        "max_iter": 3000,
        "tol": 1e-8,
        "print_level": 0,
        "sb": "yes",
        "acceptable_tol": 1e-6,
        "acceptable_iter": 15,
        "mu_strategy": "adaptive",
        "nlp_scaling_method": "gradient-based",
    }


def _toy_ipopt_starts(problem, *, num_starts):
    alpha = abs(float(getattr(problem, "alpha", 0.0)))
    radius = max(0.25, 0.8 * (1.0 + alpha))
    angles = np.linspace(-np.pi, np.pi, int(num_starts), endpoint=False)
    return [(radius, float(theta)) for theta in angles]


def _pyomo_value_or_none(expr):
    try:
        return float(pyo.value(expr))
    except Exception:
        return None


def _toy_ipopt_reference(problem, *, solver=PYOMO_NLP_SOLVER, options=None, num_starts=16, reference_label=None):
    """IPOPT reference for the nonconvex two-dimensional star toy problems."""

    if not PYOMO_AVAILABLE or pyo is None:
        raise ImportError("Pyomo is required for toy IPOPT references. Install with: pip install -r requirements.txt")
    if int(getattr(problem, "nvar", 0)) != 2:
        raise ValueError("Toy IPOPT reference currently supports only two-dimensional problems.")

    opt = pyo.SolverFactory(solver)
    if not opt.available():
        raise RuntimeError(f"Solver {solver} is not available for toy reference solve.")
    solver_options = _toy_ipopt_default_options()
    if options:
        solver_options.update(options)
    for key, value in solver_options.items():
        opt.options[key] = value

    star_problem = getattr(problem, "star_problem", problem)
    poly_problem = getattr(problem, "poly_problem", None)
    if poly_problem is not None and (poly_problem.Qq is not None or poly_problem.G is not None):
        raise ValueError("Toy IPOPT poly-star reference supports linear/box polytope constraints only.")

    alpha = float(star_problem.alpha)
    num_star = int(star_problem.num_star)
    center = np.asarray(star_problem.obj_center.detach().cpu().numpy(), dtype=float).reshape(-1)
    weights = np.asarray(star_problem.w.detach().cpu().numpy(), dtype=float).reshape(-1)
    radius_upper = max(1e-3, 1.0 + abs(alpha))
    starts = _toy_ipopt_starts(problem, num_starts=max(1, int(num_starts)))

    lower = upper = a_matrix = b_vector = None
    if poly_problem is not None:
        lower = np.asarray(poly_problem.L.detach().cpu().numpy(), dtype=float).reshape(-1)
        upper = np.asarray(poly_problem.U.detach().cpu().numpy(), dtype=float).reshape(-1)
        if poly_problem.A is not None:
            a_matrix = np.asarray(poly_problem.A.detach().cpu().numpy(), dtype=float)
            b_vector = np.asarray(poly_problem.b.detach().cpu().numpy(), dtype=float).reshape(-1)

    best = None
    status_log = []
    start_time = perf_counter()
    for r0, theta0 in starts:
        model = pyo.ConcreteModel()
        model.r = pyo.Var(bounds=(0.0, radius_upper), initialize=float(np.clip(r0, 1e-8, radius_upper)))
        model.theta = pyo.Var(bounds=(-np.pi, np.pi), initialize=theta0)
        x0 = model.r * pyo.cos(model.theta)
        x1 = model.r * pyo.sin(model.theta)
        model.objective = pyo.Objective(
            expr=weights[0] * (x0 - center[0]) ** 2 + weights[1] * (x1 - center[1]) ** 2,
            sense=pyo.minimize,
        )
        model.star_constraint = pyo.Constraint(expr=model.r <= 1.0 + alpha * pyo.sin(num_star * model.theta))
        if lower is not None and upper is not None:
            model.box_constraints = pyo.ConstraintList()
            model.box_constraints.add(x0 >= lower[0])
            model.box_constraints.add(x0 <= upper[0])
            model.box_constraints.add(x1 >= lower[1])
            model.box_constraints.add(x1 <= upper[1])
        if a_matrix is not None:
            model.linear_constraints = pyo.ConstraintList()
            for row, rhs in zip(a_matrix, b_vector):
                model.linear_constraints.add(row[0] * x0 + row[1] * x1 <= rhs)

        try:
            results = opt.solve(model, tee=False)
            status = str(results.solver.status)
            termination = str(results.solver.termination_condition)
            status_log.append(f"{status}:{termination}")
            x_candidate = np.asarray([pyo.value(x0), pyo.value(x1)], dtype=float)
            objective = _pyomo_value_or_none(model.objective)
        except Exception as exc:
            status_log.append(f"error:{exc}")
            continue
        if objective is None or not np.isfinite(objective) or not np.all(np.isfinite(x_candidate)):
            continue
        violation = _toy_max_violation(problem, x_candidate)
        is_optimal = (
            results.solver.status == SolverStatus.ok
            and results.solver.termination_condition == TerminationCondition.optimal
        )
        if violation is not None and violation <= 1e-5:
            candidate = {
                "label": reference_label or "IPOPT",
                "objective": objective,
                "solution": x_candidate,
                "violation": violation,
                "status": "optimal" if is_optimal else f"{status}:{termination}",
                "runtime": perf_counter() - start_time,
            }
            if best is None or candidate["objective"] < best["objective"]:
                best = candidate

    if best is None:
        raise RuntimeError(
            "IPOPT did not produce a feasible toy reference solution. "
            f"Observed solver outcomes: {status_log[:5]}"
        )
    return best


def _toy_reference_solution(
    problem_type,
    problem,
    *,
    reference_solver="auto",
    reference_label=None,
    grid_size=801,
    ipopt_solver=PYOMO_NLP_SOLVER,
    ipopt_options=None,
    ipopt_num_starts=16,
):
    solver_name = str(reference_solver or "none").lower()
    if solver_name == "none":
        return None
    if solver_name == "auto":
        solver_name = "convex" if problem_type == "poly" else "ipopt"
    if solver_name in {"convex", "cvxpy", "mosek"}:
        if problem_type != "poly":
            raise ValueError("Convex reference solver is only valid for problem_type='poly'.")
        result = ConvexSolver(problem.prob_para).solve_result("opt")
        return {
            "label": reference_label or "MOSEK",
            "objective": result.get("objective"),
            "solution": result.get("solution"),
            "violation": result.get("violation"),
            "status": result.get("status"),
            "runtime": result.get("runtime_total"),
        }
    if solver_name in {"ipopt", "pyomo-ipopt", "pyomo_ipopt"}:
        if problem_type == "poly":
            raise ValueError("Use reference_solver='convex' for problem_type='poly'.")
        return _toy_ipopt_reference(
            problem,
            solver=ipopt_solver,
            options=ipopt_options,
            num_starts=ipopt_num_starts,
            reference_label=reference_label,
        )
    if solver_name in {"grid", "grid_search", "gridsearch"}:
        result = _toy_grid_reference(problem, grid_size=grid_size)
        if result is not None and reference_label is not None:
            result["label"] = reference_label
        return result
    raise ValueError("reference_solver must be 'auto', 'none', 'convex', 'ipopt', or 'grid'.")


def build_poly_star_toy_context(
    *,
    alpha=0.3,
    num_star=4,
    problem_type="star",
    poly_config=None,
    seed=2025,
    device=None,
    dtype=None,
    hom_p_norm=2,
):
    """Construct a two-dimensional toy problem and its matching homeomorphic map.

    This is the shared construction boundary for runnable toy benchmarks and
    tutorials.  It intentionally does not choose or run a reference solver.
    """

    set_global_seed(seed)
    runtime_device, runtime_dtype = resolve_runtime(device, dtype, default_device="cpu")
    problem_type = str(problem_type).lower()
    if problem_type not in {"poly", "star", "poly_star"}:
        raise ValueError("problem_type must be 'poly', 'star', or 'poly_star'")

    if problem_type == "star":
        problem = cast_tensors_to_dtype(
            ToyStarOpt(alpha=alpha, num_star=num_star).to_device(runtime_device),
            runtime_dtype,
        )
        hom_map = cast_tensors_to_dtype(StarMap(problem, p_norm=hom_p_norm).to_device(runtime_device), runtime_dtype)
    elif problem_type == "poly_star":
        poly_problem = cast_tensors_to_dtype(
            ConvexOpt(_origin_polytope_config(poly_config, seed=seed)).to_device(runtime_device),
            runtime_dtype,
        )
        star_problem = cast_tensors_to_dtype(
            ToyStarOpt(alpha=alpha, num_star=num_star).to_device(runtime_device),
            runtime_dtype,
        )
        problem = cast_tensors_to_dtype(PolyStarOpt(poly_problem, star_problem).to_device(runtime_device), runtime_dtype)
        poly_map = cast_tensors_to_dtype(
            GaugeMap(poly_problem, p_norm=hom_p_norm, x_origin=np.zeros(2)).to_device(runtime_device),
            runtime_dtype,
        )
        star_map = cast_tensors_to_dtype(
            StarMap(star_problem, p_norm=hom_p_norm).to_device(runtime_device),
            runtime_dtype,
        )
        hom_map = cast_tensors_to_dtype(
            PolyStarMap(poly_map, star_map, p_norm=hom_p_norm).to_device(runtime_device),
            runtime_dtype,
        )
    else:
        default_poly_config = {
            "obj": "quad",
            "seed": seed,
            "n_var": 2,
            "n_linear_cons": 4,
            "n_soc_cons": 0,
            "n_qua_cons": 0,
            "x_lower": -2.0,
            "x_upper": 2.0,
            "margin_scale": 0.2,
        }
        config = create_test_problem(merged(default_poly_config, poly_config or {}))
        problem = cast_tensors_to_dtype(ConvexOpt(config).to_device(runtime_device), runtime_dtype)
        x_origin = ConvexSolver(problem.prob_para).solve("central_ip")
        hom_map = cast_tensors_to_dtype(
            GaugeMap(problem, p_norm=hom_p_norm, x_origin=x_origin).to_device(runtime_device),
            runtime_dtype,
        )
    return problem, hom_map


def solve_poly_star_toy_reference(
    problem_type,
    problem,
    *,
    reference_solver="auto",
    reference_label=None,
    reference_grid_size=801,
    reference_ipopt_solver=PYOMO_NLP_SOLVER,
    reference_ipopt_options=None,
    reference_ipopt_num_starts=16,
):
    """Solve one toy instance with the explicitly requested reference method."""

    return _toy_reference_solution(
        problem_type,
        problem,
        reference_solver=reference_solver,
        reference_label=reference_label,
        grid_size=reference_grid_size,
        ipopt_solver=reference_ipopt_solver,
        ipopt_options=reference_ipopt_options,
        ipopt_num_starts=reference_ipopt_num_starts,
    )


def _resolve_toy_initial_point(problem, hom_map, mode, *, seed):
    """Build one shared decision-space start for a toy comparison."""

    normalized = str(mode or "gauge_center").lower()
    if normalized == "gauge_center":
        return hom_map.center.detach().cpu().numpy()
    if normalized == "random":
        rng = np.random.default_rng(seed)
        direction = rng.standard_normal(int(problem.nvar))
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        z = torch.as_tensor(
            (0.2 + 0.6 * float(rng.random())) * direction.reshape(1, -1),
            dtype=_toy_problem_tensor_dtype(problem),
            device=problem.device,
        )
        with torch.no_grad():
            return hom_map.forward(z, method="autograd").detach().cpu().numpy()
    raise ValueError("initial_point_mode must be 'gauge_center' or 'random'.")


def poly_star_benchmark(
    algorithms=None,
    common_config=None,
    algorithm_config=None,
    alpha=0.3,
    num_star=4,
    problem_type="star",
    poly_config=None,
    max_iterations=25,
    max_running_time=5,
    learning_rate=1e-2,
    stepsize_rule="constant",
    lr_decay=0.999,
    min_lr=1e-6,
    seed=2025,
    output_dir=None,
    device=None,
    dtype=None,
    visualize=True,
    visualize_only=False,
    show_constraint_notation=False,
    reference_solver="auto",
    reference_label=None,
    reference_grid_size=801,
    reference_ipopt_solver=PYOMO_NLP_SOLVER,
    reference_ipopt_options=None,
    reference_ipopt_num_starts=16,
    return_context=False,
):
    """Small package-native toy benchmark for polytope, star, or intersection sets.

    ``return_context=True`` is intended for in-process instructional use. It
    returns the normal serializable payload together with the exact problem,
    map, reference, and records used during the run.
    """

    reject_algorithm_iteration_budget_keys(algorithm_config, context="algorithm_config")
    problem_type = str(problem_type).lower()
    if problem_type not in {"poly", "star", "poly_star"}:
        raise ValueError("problem_type must be 'poly', 'star', or 'poly_star'")
    algorithms = list(algorithms or ["Hom-PGD"])
    effective_common_config = normalize_single_common_config(
        common_config=common_config,
    )
    params = _ineq_algorithm_params(
        seed,
        max_iterations,
        max_running_time,
        learning_rate,
        stepsize_rule,
        lr_decay,
        min_lr,
        common_config=effective_common_config,
    )
    params = apply_config_groups(
        params,
        algorithm_config=algorithm_config,
    )
    effective_max_iterations = int(params["common"]["max_iterations"])
    params = enforce_alm_outer_iteration_budget(params, max_iterations=effective_max_iterations)
    hom_p_norm = params.get("Hom-PGD", {}).get("hom_p_norm", 2)
    problem, hom_map = build_poly_star_toy_context(
        alpha=alpha,
        num_star=num_star,
        problem_type=problem_type,
        poly_config=poly_config,
        seed=seed,
        device=device,
        dtype=dtype,
        hom_p_norm=hom_p_norm,
    )

    reference = solve_poly_star_toy_reference(
        problem_type,
        problem,
        reference_solver=reference_solver,
        reference_label=reference_label,
        reference_grid_size=reference_grid_size,
        reference_ipopt_solver=reference_ipopt_solver,
        reference_ipopt_options=reference_ipopt_options,
        reference_ipopt_num_starts=reference_ipopt_num_starts,
    )
    reference_payload = {}
    if reference is not None and reference.get("objective") is not None:
        reference_payload = {
            "reference_label": reference.get("label"),
            "reference_objective": float(reference["objective"]),
            "reference_solution": np.asarray(reference.get("solution"), dtype=float).reshape(-1).tolist(),
            "reference_violation": reference.get("violation"),
            "reference_status": reference.get("status"),
            "reference_runtime": reference.get("runtime"),
        }
    init_point = _resolve_toy_initial_point(
        problem,
        hom_map,
        params["common"].get("initial_point_mode", "gauge_center"),
        seed=seed,
    )

    existing_artifacts = {}
    if visualize_only:
        visual_state = load_visualize_only_comparison_artifacts(
            output_dir,
            algorithms=algorithms,
        )
        records = visual_state["records"]
        summaries = visual_state["summaries"]
        existing_artifacts = visual_state["artifacts"]
        previous_metrics = visual_state["metrics"]
        if not reference_payload and previous_metrics.get("reference_objective") is not None:
            reference_payload = {
                "reference_label": previous_metrics.get("reference_label"),
                "reference_objective": previous_metrics.get("reference_objective"),
            }
    else:
        records = {}
        summaries = {}
        for algorithm in algorithms:
            result = run_algorithm(algorithm, problem, params, hom_map=hom_map, init_point=init_point)
            ensure_record_violation_split(problem, result)
            records[algorithm] = result
            summaries[algorithm] = summarize_run_record(
                result,
                reference_objective=reference_payload.get("reference_objective"),
                include_total_iter_time=True,
            )

    artifacts, _, records, summaries = save_incremental_comparison_artifacts(
        output_dir,
        records=records,
        summaries=summaries,
        algorithm_order=algorithms,
        manifest_metadata={"problem_family": f"toy_{problem_type}"},
    )
    artifacts = {**existing_artifacts, **artifacts}

    if visualize and output_dir is not None:
        viz_artifacts = save_convex_2d_visualizations(
            problem=problem,
            records=records,
            algorithms=algorithms,
            output_dir=output_dir,
            hom_map=hom_map,
            payload=reference_payload,
            prefix=f"toy_{problem_type}",
            include_individual=True,
            show_constraint_notation=show_constraint_notation,
            reference_label=reference_payload.get("reference_label"),
            violation_y_min=float(params["common"]["convergence_threshold"]),
        )
        artifacts = {**artifacts, **viz_artifacts}

    primary = algorithms[0]
    payload = build_benchmark_payload(
        objective=summaries[primary]["final_objective"],
        feasible=summaries[primary]["final_violation"] <= 1e-5,
        artifacts=artifacts,
        algorithms=list(records.keys()),
        benchmarks=summaries,
        **reference_payload,
        nvar=int(problem.nvar),
        problem=f"toy_{problem_type}",
        seed=seed,
    )
    if return_context:
        return payload, {
            "problem": problem,
            "hom_map": hom_map,
            "reference": reference,
            "records": records,
            "summaries": summaries,
            "params": params,
            "initial_point": init_point,
        }
    return payload


__all__ = ["build_poly_star_toy_context", "poly_star_benchmark", "solve_poly_star_toy_reference"]
