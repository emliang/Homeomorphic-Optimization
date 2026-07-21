"""Shared experiment configuration normalization helpers."""

from __future__ import annotations

import copy


def merged(base, overrides=None):
    if base is None:
        payload = {}
    elif isinstance(base, dict):
        payload = copy.deepcopy(base)
    else:
        payload = dict(base)
    if overrides:
        payload.update(copy.deepcopy(dict(overrides)))
    return payload


def deep_merged(base, overrides=None):
    payload = copy.deepcopy(base) if isinstance(base, dict) else ({} if base is None else dict(base))
    for key, value in dict(overrides or {}).items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = deep_merged(payload[key], value)
        else:
            payload[key] = copy.deepcopy(value)
    return payload


def _reject_config_keys(config, *, keys, context, message):
    invalid = sorted(set(keys).intersection(config))
    if not invalid:
        return
    raise ValueError(f"Unsupported {context} keys: {invalid}. {message}")


def normalize_inner_solver_common_config(inner_solver_common):
    """Keep inner-solver controls from overwriting outer ALM controls."""

    if not inner_solver_common:
        return inner_solver_common
    config = copy.deepcopy(inner_solver_common)
    _reject_config_keys(
        config,
        keys=("learning_rate", "lr_decay", "stepsize_rule"),
        context="inner_solver_common",
        message="Use inner_learning_rate, inner_lr_decay, and inner_stepsize_rule.",
    )
    return config


def normalize_outer_common_config(outer_common, *, context="outer_common"):
    """Normalize shared ALM outer-loop controls before algorithm-level merges."""

    if not outer_common:
        return outer_common
    config = copy.deepcopy(outer_common)
    _reject_config_keys(
        config,
        keys=(
            "learning_rate",
            "lr_decay",
            "stepsize_rule",
            "outer_learning_rate",
            "outer_lr_decay",
            "outer_stepsize_rule",
        ),
        context=context,
        message=(
            "Set shared primal step controls in common_config. "
            "Use algorithm_config with learning_rate, lr_decay, or stepsize_rule for per-algorithm overrides."
        ),
    )
    return config


ALM_ITERATIVE_ALGORITHMS = ("Penalty", "Prox-Penalty", "ALM", "Prox-ALM", "Hom-ALM", "Prox-Hom-ALM")
ALM_CVXPY_ALGORITHMS = ("Penalty-EQ", "Prox-Penalty-EQ", "ALM-EQ", "Prox-ALM-EQ")
ALM_FAMILY_ALGORITHMS = (*ALM_ITERATIVE_ALGORITHMS, *ALM_CVXPY_ALGORITHMS)
ALM_CVXPY_CONFIG_KEYS = {
    "max_running_time",
    "outer_iterations",
    "convergence_threshold",
    "dual_learning_rate",
    "penalty_coef",
    "penalty_growth",
    "proximal_coef",
    "max_penalty",
    "max_dual",
    "use_lagrangian",
    "use_penalty",
    "use_proximal",
    "return_best_violation",
    "check_outer_objective_change",
    "outer_objective_change_threshold",
    "min_outer_iterations",
    "solver_options",
    "subproblem_time_limit_sec",
    "solver_verbose",
}

# These controls are meaningful only for the exact-subproblem equality
# baselines.  ``outer_common`` may contain the union of method-family controls,
# but direct iterative overrides must not accept and then discard them.
ALM_CVXPY_ONLY_CONFIG_KEYS = {
    "check_outer_objective_change",
    "outer_objective_change_threshold",
    "min_outer_iterations",
    "solver_options",
    "subproblem_time_limit_sec",
    "solver_verbose",
}

# These public aliases describe first-order step schedules.  They are useful
# in a shared algorithm-config payload, but exact CVXPY subproblems do not
# perform a primal gradient update.  Drop only these known no-op aliases after
# public-key normalization; all other unsupported explicit keys still fail.
ALM_CVXPY_IGNORED_PUBLIC_SCHEDULE_KEYS = {
    "outer_lr_decay",
    "outer_stepsize_rule",
}


def _outer_common_for_alm_algorithm(algorithm_name, outer_common):
    """Route shared outer controls to the ALM variants that support them.

    ``outer_common`` deliberately covers the union of iterative and exact
    equality-baseline controls.  Gradient-inner-loop controls such as
    ``first_order_lagrangian_gap_threshold`` therefore remain available to
    iterative methods but must not leak into the CVXPY-backed ``*-EQ``
    variants.  Direct per-algorithm overrides are validated separately by
    :func:`normalize_alm_algorithm_config`; its only intentional no-op filter
    is the documented public schedule aliases for exact baselines.
    """

    if algorithm_name in ALM_ITERATIVE_ALGORITHMS:
        return {
            key: copy.deepcopy(value)
            for key, value in outer_common.items()
            if key not in ALM_CVXPY_ONLY_CONFIG_KEYS
        }
    if algorithm_name in ALM_CVXPY_ALGORITHMS:
        return {
            key: copy.deepcopy(value)
            for key, value in outer_common.items()
            if key in ALM_CVXPY_CONFIG_KEYS
        }
    return copy.deepcopy(outer_common)


def _normalize_alm_public_algorithm_config(config, *, context):
    normalized = copy.deepcopy(config)
    _reject_config_keys(
        normalized,
        keys=("outer_learning_rate", "outer_lr_decay", "outer_stepsize_rule"),
        context=context,
        message="Use learning_rate, lr_decay, and stepsize_rule; ALM builders map them to outer-loop fields.",
    )
    if "lr_decay" in normalized:
        normalized["outer_lr_decay"] = normalized.pop("lr_decay")
    if "stepsize_rule" in normalized:
        normalized["outer_stepsize_rule"] = normalized.pop("stepsize_rule")
    return normalized


def normalize_alm_algorithm_config(algorithm_name, config, *, outer_config_normalized=False):
    """Normalize ALM-family per-algorithm config before merging."""

    if not config:
        return config
    normalized_outer = copy.deepcopy(config) if outer_config_normalized else None
    if algorithm_name in ALM_ITERATIVE_ALGORITHMS:
        normalized = normalized_outer or _normalize_alm_public_algorithm_config(
            config,
            context=f"{algorithm_name} config",
        )
        if not outer_config_normalized:
            unsupported = sorted(set(normalized).intersection(ALM_CVXPY_ONLY_CONFIG_KEYS))
            if unsupported:
                raise ValueError(
                    f"Unsupported {algorithm_name} config keys: {unsupported}. "
                    "These controls apply only to the CVXPY-backed *-EQ baselines."
                )
        return normalized
    if algorithm_name in ALM_CVXPY_ALGORITHMS:
        normalized = normalized_outer or _normalize_alm_public_algorithm_config(
            config,
            context=f"{algorithm_name} config",
        )
        unknown_keys = sorted(
            set(normalized) - ALM_CVXPY_CONFIG_KEYS - ALM_CVXPY_IGNORED_PUBLIC_SCHEDULE_KEYS
        )
        if unknown_keys:
            raise ValueError(f"Unsupported {algorithm_name} config keys: {unknown_keys}")
        return {
            key: value
            for key, value in normalized.items()
            if key in ALM_CVXPY_CONFIG_KEYS
        }
    return config


def merge_alm_outer_common(params, outer_common):
    """Apply shared outer-loop controls to every active ALM-family config."""

    if not outer_common:
        return params
    normalized_outer = normalize_outer_common_config(outer_common)
    for name in ALM_FAMILY_ALGORITHMS:
        if name in params and isinstance(params[name], dict):
            supported_outer = _outer_common_for_alm_algorithm(name, normalized_outer)
            params[name] = merged(
                params[name],
                normalize_alm_algorithm_config(name, supported_outer, outer_config_normalized=True),
            )
    return params


def merge_alm_inner_solver_common(params, inner_solver_common):
    """Apply gradient-inner-loop controls only to iterative ALM variants."""

    if not inner_solver_common:
        return params
    normalized_inner = normalize_inner_solver_common_config(inner_solver_common)
    for name in ALM_ITERATIVE_ALGORITHMS:
        if name in params and isinstance(params[name], dict):
            params[name] = merged(params[name], normalized_inner)
    return params


def apply_alm_common_configs(params, *, outer_common=None, inner_solver_common=None):
    """Apply shared ALM outer and inner config in the canonical order."""

    params = merge_alm_outer_common(params, outer_common)
    return merge_alm_inner_solver_common(params, inner_solver_common)


def normalize_alm_common_config_groups(*, outer_common=None, inner_solver_common=None):
    """Normalize ALM outer/inner shared config groups at one call site."""

    return (
        normalize_outer_common_config(outer_common),
        normalize_inner_solver_common_config(inner_solver_common),
    )


def reject_iteration_budget_keys(config, *, context):
    invalid = sorted({"outer_iterations", "max_iterations"}.intersection(dict(config or {})))
    if invalid:
        raise ValueError(
            f"{context} must not set {invalid}. "
            "Use common max_iterations as the shared outer-loop budget."
        )


def reject_algorithm_iteration_budget_keys(config_group, *, context):
    for name, config in dict(config_group or {}).items():
        if name not in ALM_FAMILY_ALGORITHMS or not isinstance(config, dict):
            continue
        invalid = sorted({"outer_iterations", "max_iterations"}.intersection(config))
        if invalid:
            raise ValueError(
                f"{context} for {name} must not set {invalid}. "
                "Use common max_iterations as the shared outer-loop budget."
            )


def align_outer_iterations_with_max_iterations(
    outer_common=None,
    *,
    max_iterations=None,
    context="outer_common",
):
    """Use max_iterations as the canonical outer-loop budget for ALM methods."""

    reject_iteration_budget_keys(outer_common, context=context)
    config = copy.deepcopy(outer_common) if outer_common else {}
    if max_iterations is None:
        return config
    max_iter = int(max_iterations)
    config["outer_iterations"] = max_iter
    return config


def enforce_alm_outer_iteration_budget(params, *, max_iterations, context="algorithm_config"):
    """Reject ALM-family configs whose outer loop budget diverges from max_iterations."""

    max_iter = int(max_iterations)
    for name in ALM_FAMILY_ALGORITHMS:
        config = params.get(name)
        if not isinstance(config, dict):
            continue
        if "max_iterations" in config:
            raise ValueError(
                f"{context} for {name} must not set max_iterations. "
                "Use common max_iterations as the shared outer-loop budget."
            )
        if "outer_iterations" not in config:
            continue
        if "outer_iterations" in config and int(config["outer_iterations"]) != max_iter:
            raise ValueError(
                f"Conflicting {context} parameters for {name}: "
                f"max_iterations={max_iter!r} and outer_iterations={config['outer_iterations']!r}. "
                "Use common max_iterations as the shared outer-loop budget."
            )
    return params


def apply_config_groups(
    params,
    common_config=None,
    problem_config=None,
    algorithm_config=None,
):
    """Apply common/problem/algorithm config groups once at the final merge boundary."""

    common_config = normalize_single_common_config(
        common_config=common_config,
    )
    problem_config = normalize_single_problem_config(
        problem_config=problem_config,
    )
    algorithm_config = normalize_algorithm_config_group(
        algorithm_config=algorithm_config,
    )
    if "common" in params:
        params["common"] = merged(params["common"], common_config)
    if common_config:
        for key, value in list(params.items()):
            if isinstance(value, dict) and "opt_type" in value:
                # Common controls can be shared across methods, but exact
                # equality baselines accept only their explicit solver contract.
                shared = common_config
                if key in ALM_CVXPY_ALGORITHMS:
                    shared = {
                        name: setting
                        for name, setting in common_config.items()
                        if name in ALM_CVXPY_CONFIG_KEYS
                    }
                params[key] = merged(value, normalize_alm_algorithm_config(key, shared))
    if "prob" in params:
        params["prob"] = merged(params["prob"], problem_config)
    if algorithm_config:
        unknown_algorithms = sorted(name for name in algorithm_config if name not in params)
        if unknown_algorithms:
            raise ValueError(f"Unknown algorithm_config entries: {unknown_algorithms}")
        for name, config in algorithm_config.items():
            if name in params and isinstance(params[name], dict):
                # PGD/FW expose nested projection/oracle subproblem controls.
                # Preserve the canonical subproblem defaults when applying a
                # public iteration override instead of replacing that nested
                # configuration wholesale.
                params[name] = deep_merged(params[name], normalize_alm_algorithm_config(name, config))
    return params


def normalize_single_common_config(
    *,
    common=None,
    common_config=None,
):
    payload = merged(common, common_config)
    _reject_config_keys(
        payload,
        keys=("outer_learning_rate", "outer_lr_decay", "outer_stepsize_rule"),
        context="common_config",
        message="Use learning_rate, lr_decay, and stepsize_rule; ALM builders map them to outer-loop fields.",
    )
    if "lr_decay" in payload:
        payload["outer_lr_decay"] = payload["lr_decay"]
        payload.setdefault("inner_lr_decay", payload["lr_decay"])
    if "min_lr" in payload:
        payload.setdefault("inner_min_lr", payload["min_lr"])
    if "stepsize_rule" in payload:
        payload["outer_stepsize_rule"] = payload["stepsize_rule"]
    return payload


def normalize_single_problem_config(
    *,
    problem_config=None,
):
    return merged(None, problem_config)


def normalize_single_algorithm_config(
    *,
    algorithm_config=None,
):
    normalized = deep_merged(None, algorithm_config)
    for name, config in list(normalized.items()):
        if not isinstance(config, dict):
            continue
        reject_iteration_budget_keys(config, context=f"algorithm_config for {name}")
        normalized[name] = normalize_first_order_subproblem_config(name, config)
    return normalized


def _sync_subproblem_iterations(config, *, public_max_key, old_outer_key, inner_key, subproblem_key, context):
    normalized = copy.deepcopy(config)
    _reject_config_keys(
        normalized,
        keys=(old_outer_key,),
        context=context,
        message=f"Use {public_max_key}; it is mapped to the subproblem outer loop.",
    )
    subproblem = copy.deepcopy(normalized.get(subproblem_key) or {})
    if public_max_key in normalized:
        value = int(normalized.pop(public_max_key))
        normalized[old_outer_key] = value
        subproblem["outer_iterations"] = value
    if inner_key in normalized:
        subproblem["inner_iterations"] = int(normalized[inner_key])
    if subproblem:
        normalized[subproblem_key] = merged(normalized.get(subproblem_key), subproblem)
    return normalized


def normalize_first_order_subproblem_config(algorithm_name, config):
    """Normalize public PGD/FW subproblem iteration knobs into solver configs."""

    if not config:
        return config
    if algorithm_name == "PGD":
        return _sync_subproblem_iterations(
            config,
            public_max_key="projection_max_iterations",
            old_outer_key="projection_outer_iterations",
            inner_key="projection_inner_iterations",
            subproblem_key="projection_subproblem",
            context="PGD config",
        )
    if algorithm_name == "FW":
        return _sync_subproblem_iterations(
            config,
            public_max_key="linearization_max_iterations",
            old_outer_key="linearization_outer_iterations",
            inner_key="linearization_inner_iterations",
            subproblem_key="linearization_subproblem",
            context="FW config",
        )
    return config


def normalize_algorithm_config_group(
    *,
    algorithm_config=None,
):
    """Merge per-algorithm config groups."""

    return normalize_single_algorithm_config(
        algorithm_config=algorithm_config,
    )


def normalize_solver_run_config(
    *,
    solver_name="ipopt",
    solver_options=None,
    solver_config=None,
):
    normalized = {
        "solver_name": solver_name,
        "solver_options": dict(solver_options or {}),
    }
    if solver_config:
        translated = dict(solver_config)
        _reject_config_keys(
            translated,
            keys=("name", "options"),
            context="solver_config",
            message="Use solver_name and solver_options.",
        )
        normalized = merged(normalized, translated)
    normalized["solver_name"] = str(normalized.get("solver_name", solver_name))
    normalized["solver_options"] = dict(normalized.get("solver_options") or {})
    return normalized


def normalize_jcc_problem_config(
    *,
    n_scenarios,
    epsilon,
    demand_std,
    seed,
    problem_config=None,
):
    supported_keys = {
        "n_scenarios",
        "epsilon",
        "demand_std",
        "seed",
        "pglib_case_name",
        "pglib_data_dir",
        "download_if_missing",
    }
    problem_cfg = {
        "n_scenarios": int(n_scenarios),
        "epsilon": float(epsilon),
        "demand_std": float(demand_std),
        "seed": int(seed),
        # Keep PGLib input controls in the canonical JCC config.  They affect
        # the actual optimization instance and must reach the problem loader.
        "pglib_case_name": None,
        "pglib_data_dir": None,
        "download_if_missing": True,
    }
    if problem_config:
        overrides = dict(problem_config)
        unknown = sorted(set(overrides) - supported_keys)
        if unknown:
            raise ValueError(f"Unsupported JCC problem_config keys: {unknown}")
        problem_cfg.update(overrides)
    problem_cfg["n_scenarios"] = int(problem_cfg["n_scenarios"])
    problem_cfg["seed"] = int(problem_cfg["seed"])
    problem_cfg["epsilon"] = float(problem_cfg["epsilon"])
    problem_cfg["demand_std"] = float(problem_cfg["demand_std"])
    case_name = problem_cfg.get("pglib_case_name")
    data_dir = problem_cfg.get("pglib_data_dir")
    problem_cfg["pglib_case_name"] = None if case_name is None else str(case_name)
    problem_cfg["pglib_data_dir"] = None if data_dir is None else str(data_dir)
    problem_cfg["download_if_missing"] = bool(problem_cfg["download_if_missing"])
    return problem_cfg


def normalize_jcc_solver_configs(
    *,
    solver_configs=None,
):
    canonical = {
        "mixed_integer": {"solver": "GUROBI", "M": 1000.0, "verbose": False},
        "cvar": {"solver": "MOSEK", "verbose": False},
        "scenario": {"solver": "MOSEK", "verbose": False},
    }
    if solver_configs:
        for name, cfg in dict(solver_configs).items():
            if name not in canonical:
                raise ValueError(f"Unsupported JCC solver config group: {name}")
            canonical[name] = merged(canonical[name], cfg)
    return canonical


def require_explicit_config(config, name):
    if config is None:
        raise ValueError(f"{name} must be provided explicitly by the experiment script.")
    return config


__all__ = [
    "ALM_CVXPY_ALGORITHMS",
    "ALM_FAMILY_ALGORITHMS",
    "ALM_ITERATIVE_ALGORITHMS",
    "align_outer_iterations_with_max_iterations",
    "apply_alm_common_configs",
    "apply_config_groups",
    "deep_merged",
    "enforce_alm_outer_iteration_budget",
    "merge_alm_inner_solver_common",
    "merge_alm_outer_common",
    "merged",
    "normalize_algorithm_config_group",
    "normalize_alm_algorithm_config",
    "normalize_alm_common_config_groups",
    "normalize_inner_solver_common_config",
    "normalize_jcc_problem_config",
    "normalize_jcc_solver_configs",
    "normalize_first_order_subproblem_config",
    "normalize_outer_common_config",
    "normalize_single_algorithm_config",
    "normalize_single_common_config",
    "normalize_single_problem_config",
    "normalize_solver_run_config",
    "require_explicit_config",
    "reject_algorithm_iteration_budget_keys",
    "reject_iteration_budget_keys",
]
