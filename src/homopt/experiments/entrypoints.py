"""Shared entrypoints for script-style experiment runs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union

from homopt.utils import ensure_dir

from .context import ExperimentContext
from .runner import run_and_record


def _fmt_summary_scalar(value, *, digits=4, default="-"):
    if value is None:
        return default
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _comparison_summary_rows(result):
    metrics = dict(getattr(result, "metrics", {}) or {})
    instance_rows = list(metrics.get("comparison_rows") or [])
    if instance_rows:
        unique_labels = {(row.get("route"), row.get("method")) for row in instance_rows}
        if len(instance_rows) == len(unique_labels):
            return instance_rows
    summary_rows = list(metrics.get("comparison_summary_rows") or [])
    if summary_rows:
        return summary_rows
    return instance_rows


def _comparison_row_value(row, summary_key, raw_key):
    if summary_key in row:
        return row.get(summary_key)
    if raw_key == "avg_outer_iter_time" and raw_key not in row:
        total_time = row.get("runtime_total", row.get("runtime_mean", row.get("runtime")))
        iterations = row.get("iterations")
        if total_time is not None and iterations not in (None, 0):
            return float(total_time) / float(iterations)
    return row.get(raw_key)


def _format_seconds(value):
    scalar = _fmt_summary_scalar(value, digits=4)
    return "-" if scalar == "-" else f"{scalar}s"


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _print_comparison_summary(result):
    rows = _comparison_summary_rows(result)
    if not rows:
        return
    best = (getattr(result, "metrics", {}) or {}).get("best_method") or {}
    print("comparison:")
    header = (
        f"  {'method':<26}"
        f" {'obj':>10}"
        f" {'eq_vio':>10}"
        f" {'ineq_vio':>10}"
        f" {'lag_gap':>10}"
        f" {'avg_outer':>11}"
        f" {'avg_inner':>11}"
        f" {'cvxpy_ip':>10}"
        f" {'init_map':>10}"
        f" {'final_map':>10}"
        f" {'iter':>10}"
        f" {'wall':>10}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for row in rows:
        method = row.get("method", "?")
        route = row.get("route")
        label = f"{method} [{route}]" if route else str(method)
        objective = _comparison_row_value(row, "objective_mean", "objective")
        eq_violation = _comparison_row_value(row, "equality_violation_mean", "equality_violation")
        ineq_violation = _comparison_row_value(row, "inequality_violation_mean", "inequality_violation")
        lagrangian_gap = _comparison_row_value(row, "first_order_lagrangian_gap_mean", "first_order_lagrangian_gap")
        avg_outer_iter_time = _comparison_row_value(row, "avg_outer_iter_time_mean", "avg_outer_iter_time")
        avg_inner_iter_time = _first_present(
            _comparison_row_value(row, "mean_inner_iter_time_mean", "mean_inner_iter_time"),
            _comparison_row_value(row, "mean_solver_iter_time_mean", "mean_solver_iter_time"),
        )
        initial_ip_time = _comparison_row_value(row, "initial_ip_time_mean", "initial_ip_time")
        initial_transform_time = _comparison_row_value(row, "initial_transform_time_mean", "initial_transform_time")
        final_transform_time = _comparison_row_value(row, "final_transform_time_mean", "final_transform_time")
        iter_time = row.get("runtime_total", row.get("runtime_mean", row.get("runtime")))
        wall_time = _comparison_row_value(row, "total_wall_time_mean", "total_wall_time")
        hom_mode = row.get("hom_map_forward_mode")
        hom_smooth = row.get("hom_map_smooth")
        hom_suffix = ""
        if hom_mode:
            hom_suffix += f"  mode={hom_mode}"
        if hom_smooth is not None:
            hom_suffix += f"  smooth={bool(hom_smooth)}"
        print(
            f"  {label:<26}"
            f" {_fmt_summary_scalar(objective):>10}"
            f" {_fmt_summary_scalar(eq_violation):>10}"
            f" {_fmt_summary_scalar(ineq_violation):>10}"
            f" {_fmt_summary_scalar(lagrangian_gap):>10}"
            f" {_format_seconds(avg_outer_iter_time):>11}"
            f" {_format_seconds(avg_inner_iter_time):>11}"
            f" {_format_seconds(initial_ip_time):>10}"
            f" {_format_seconds(initial_transform_time):>10}"
            f" {_format_seconds(final_transform_time):>10}"
            f" {_format_seconds(iter_time):>10}"
            f" {_format_seconds(wall_time):>10}"
            f"{hom_suffix}"
        )
    if best:
        route = best.get("route")
        suffix = f" [{route}]" if route else ""
        print(f"best_method: {best.get('method')}{suffix}")


def _print_reference_solution_diagnostics(result):
    diagnostics = (getattr(result, "metrics", {}) or {}).get("reference_solution_diagnostics") or {}
    if not diagnostics or not diagnostics.get("available"):
        return
    counts = diagnostics.get("counts_by_type") or {}
    print("reference_solution_diagnostics:")
    print(
        "  "
        f"active_ineq: {diagnostics.get('active_inequality_total', 0)}/{diagnostics.get('inequality_total', 0)}, "
        f"violated_ineq: {diagnostics.get('violated_inequality_total', 0)}, "
        f"active_tol: {_fmt_summary_scalar(diagnostics.get('active_tol'), digits=2)}"
    )
    if "box_total" in diagnostics or "group_total" in diagnostics:
        print(
            "  "
            f"active_box: {diagnostics.get('active_box_total', 0)}/{diagnostics.get('box_total', 0)}, "
            f"violated_box: {diagnostics.get('violated_box_total', 0)}, "
            f"active_group: {diagnostics.get('active_group_total', 0)}/{diagnostics.get('group_total', 0)}, "
            f"violated_group: {diagnostics.get('violated_group_total', 0)}"
        )
    equality = counts.get("equality") or {}
    if equality.get("total", 0):
        print(
            "  "
            f"equality: {equality.get('total', 0)}, "
            f"violated: {equality.get('violated', 0)}, "
            f"max_abs: {_fmt_summary_scalar(equality.get('max_abs_residual'))}"
        )
    print("  active_by_type:")
    for key in ("linear", "quadratic", "soc", "box_lower", "box_upper"):
        block = counts.get(key) or {}
        if block.get("total", 0) == 0:
            continue
        print(
            "    "
            f"{key:<10} "
            f"{block.get('active', 0):>4}/{block.get('total', 0):<4} "
            f"violated={block.get('violated', 0):<4} "
            f"max_res={_fmt_summary_scalar(block.get('max_residual'))}"
        )


def _print_reference_cache_summary(result):
    metrics = getattr(result, "metrics", {}) or {}
    if "reference_cache_hit" not in metrics and not metrics.get("reference_cache_path"):
        return
    enabled = bool(metrics.get("reference_cache_enabled", True))
    if not enabled:
        status = "disabled"
    else:
        status = "hit" if metrics.get("reference_cache_hit") else "miss"
    parts = [f"status={status}"]
    path = metrics.get("reference_cache_path")
    if path:
        parts.append(f"path={path}")
    load_time = metrics.get("reference_cache_load_time")
    if load_time is not None:
        parts.append(f"load={_format_seconds(load_time)}")
    cached_opt_time = metrics.get("reference_cached_opt_solver_time")
    if cached_opt_time is not None:
        parts.append(f"cached_opt={_format_seconds(cached_opt_time)}")
    cached_origin_time = metrics.get("reference_cached_origin_solver_time")
    if cached_origin_time is not None:
        parts.append(f"cached_origin={_format_seconds(cached_origin_time)}")
    print("reference_cache:", " ".join(parts))


def print_result_summary(result, output_dir, preset=None):
    if preset is not None:
        print("preset:", preset)
    print("name:", result.name)
    print("output_dir:", output_dir)
    has_comparison = bool((getattr(result, "metrics", {}) or {}).get("comparison_rows"))
    if has_comparison:
        print("best_objective:", result.objective)
        print("best_feasible:", result.feasible)
    else:
        print("objective:", result.objective)
        print("feasible:", result.feasible)
    _print_reference_cache_summary(result)
    _print_reference_solution_diagnostics(result)
    _print_comparison_summary(result)


def default_script_output_dir(name: str, output_root: Union[str, Path, None] = None) -> Path:
    root = Path(output_root) if output_root is not None else Path("results")
    return root / name


def record_script_run(name, params, run_fn, output_dir=None):
    target_dir = ensure_dir(Path(output_dir or default_script_output_dir(name)))
    if not isinstance(params, Mapping):
        raise ValueError("Script params must be a mapping.")
    script_params = dict(params)
    context = ExperimentContext.from_script(name=name, params=script_params, output_dir=target_dir)
    result = run_and_record(
        name=name,
        run_fn=run_fn,
        output_dir=target_dir,
        config={"entrypoint": "script", "script_params": script_params},
        context=context.to_dict(),
    )
    print_result_summary(result, target_dir)
    return result, target_dir


__all__ = [
    "default_script_output_dir",
    "print_result_summary",
    "record_script_run",
]
