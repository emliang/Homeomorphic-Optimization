"""Optimizer algorithm registry and dispatch implementation."""

from __future__ import annotations

import time

from homopt.optim.common import _collect_optimizer_timing_metrics, _pack_run_result
from homopt.optim.first_order import FrankWolfeOptimizer, HomPGDOptimizer, PGDOptimizer, RadialDualOptimizer
from homopt.optim.hom_alm import HomALMOptimizer
from homopt.optim.lagrangian import EqualityConstrainedALMOptimizer, LagrangianOptimizer


_EQ_CVXPY_ALGORITHMS = ('Penalty-EQ', 'Prox-Penalty-EQ', 'ALM-EQ', 'Prox-ALM-EQ')


def _algorithm_specs():
    def _params_for_algorithm(params, name, *, force_proximal=False):
        if name not in params:
            raise KeyError(f"Missing optimizer parameters for algorithm: {name}")
        algorithm_params = dict(params[name])
        algorithm_params["opt_type"] = name
        if force_proximal:
            if "use_proximal" in algorithm_params and not bool(algorithm_params["use_proximal"]):
                raise ValueError(f"{name} requires use_proximal=True.")
            algorithm_params["use_proximal"] = True
            if "proximal_coef" not in algorithm_params:
                raise KeyError(f"Missing proximal_coef for proximal algorithm: {name}")
        return algorithm_params

    specs = {
        'PGD': (lambda problem, params, hom_map: PGDOptimizer(problem, params['PGD']), False),
        'Penalty': (
            lambda problem, params, hom_map: LagrangianOptimizer(
                problem,
                _params_for_algorithm(params, "Penalty"),
            ),
            False,
        ),
        'Prox-Penalty': (
            lambda problem, params, hom_map: LagrangianOptimizer(
                problem,
                _params_for_algorithm(params, "Prox-Penalty", force_proximal=True),
            ),
            False,
        ),
        'ALM': (lambda problem, params, hom_map: LagrangianOptimizer(problem, params['ALM']), False),
        'Prox-ALM': (
            lambda problem, params, hom_map: LagrangianOptimizer(
                problem,
                _params_for_algorithm(params, "Prox-ALM", force_proximal=True),
            ),
            False,
        ),
        'Hom-PGD': (lambda problem, params, hom_map: HomPGDOptimizer(problem, params['Hom-PGD'], hom_map=hom_map), True),
        'Hom-ALM': (lambda problem, params, hom_map: HomALMOptimizer(problem, params['Hom-ALM'], hom_map=hom_map), True),
        'Prox-Hom-ALM': (
            lambda problem, params, hom_map: HomALMOptimizer(
                problem,
                _params_for_algorithm(params, "Prox-Hom-ALM", force_proximal=True),
                hom_map=hom_map,
            ),
            True,
        ),
        'FW': (lambda problem, params, hom_map: FrankWolfeOptimizer(problem, params['FW']), False),
        'RD': (lambda problem, params, hom_map: RadialDualOptimizer(problem, params['RD'], hom_map=hom_map), True),
    }
    specs.update({
        name: (
            lambda problem, params, hom_map, algorithm=name: EqualityConstrainedALMOptimizer(
                problem,
                _params_for_algorithm(params, algorithm, force_proximal=algorithm.startswith("Prox-")),
            ),
            False,
        )
        for name in _EQ_CVXPY_ALGORITHMS
    })
    return specs


def run_algorithm(name, problem, params, hom_map=None, init_point=None):
    """Helper function to run a specific algorithm and collect results"""
    print(f'Running {name}')
    specs = _algorithm_specs()
    if name not in specs:
        raise ValueError(f"Unknown algorithm: {name}")
    seed = params['common']['seed']
    builder, returns_transform_time = specs[name]
    optimizer = builder(problem, params, hom_map)
    if hom_map is not None and hasattr(hom_map, "reset_forward_stats"):
        hom_map.reset_forward_stats()
    run_start_time = time.perf_counter()
    result = optimizer.optimize(init_point, verbose=bool(params["common"].get("verbose", False)), seed=seed)
    total_wall_time = time.perf_counter() - run_start_time
    extra_metrics = {"total_wall_time": total_wall_time}
    if name in {"Hom-ALM", "Prox-Hom-ALM"}:
        extra_metrics["violation_scope"] = "equality"
    extra_metrics.update(_collect_optimizer_timing_metrics(optimizer))
    if hom_map is not None and hasattr(hom_map, "last_forward_mode"):
        extra_metrics.update(
            {
                "hom_map_forward_mode": hom_map.last_forward_mode,
                "hom_map_forward_mode_counts": dict(getattr(hom_map, "forward_mode_counts", {})),
                "hom_map_smooth": bool(getattr(hom_map, "smooth", False)),
                "hom_map_smooth_tie_tol": float(getattr(hom_map, "smooth_tie_tol", 0.0)),
                "hom_map_smooth_temperature": float(getattr(hom_map, "smooth_temperature", 0.0)),
            }
        )
    if returns_transform_time:
        final_decision, decision_trajectory, objective_trajectory, violation_trajectory, per_iter_time, last_trans_time = result
        latent_trajectory = getattr(optimizer, "last_z_trajectory", None)
        return _pack_run_result(
            final_decision,
            decision_trajectory,
            objective_trajectory,
            violation_trajectory,
            per_iter_time,
            last_trans_time=last_trans_time,
            latent_trajectory=latent_trajectory,
            extra_metrics=extra_metrics,
        )
    final_decision, decision_trajectory, objective_trajectory, violation_trajectory, per_iter_time = result
    return _pack_run_result(
        final_decision,
        decision_trajectory,
        objective_trajectory,
        violation_trajectory,
        per_iter_time,
        extra_metrics=extra_metrics,
    )



__all__ = [
    "EqualityConstrainedALMOptimizer",
    "FrankWolfeOptimizer",
    "HomALMOptimizer",
    "HomPGDOptimizer",
    "LagrangianOptimizer",
    "PGDOptimizer",
    "RadialDualOptimizer",
    "_algorithm_specs",
    "run_algorithm",
]
