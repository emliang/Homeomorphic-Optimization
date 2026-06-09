"""Shared benchmark helpers for the two main experiment tracks."""

from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import os
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from homopt.experiments._comparison_helpers import build_single_instance_comparison_views, make_comparison_row
from homopt.mappings import GaugeMap
from homopt.optim._defaults import build_first_order_subproblem_defaults
from homopt.problems import ConvexOpt, ConvexOptEq, bind_singleton_problem_instance, create_test_problem
from homopt.solvers import ConvexSolver, solve_exact_result
from homopt.utils import cast_tensors_to_dtype, ensure_dir, resolve_torch_device, resolve_torch_dtype


# Artifact and payload helpers

def artifact_root(output_dir):
    if output_dir is None:
        return None
    return ensure_dir(Path(output_dir) / "artifacts")


def artifact_ref(output_dir, path):
    if output_dir is None or path is None:
        return None
    output_path = Path(output_dir)
    artifact_path = Path(path)
    try:
        return str(artifact_path.relative_to(output_path))
    except ValueError:
        return os.path.relpath(artifact_path, output_path)


def artifact_mapping(output_dir, **named_paths):
    return {
        name: artifact_ref(output_dir, path)
        for name, path in named_paths.items()
        if path is not None
    }


def build_benchmark_payload(*, objective, feasible, artifacts=None, **metrics):
    return {
        "objective": objective,
        "feasible": feasible,
        "artifacts": dict(artifacts or {}),
        "_allow_top_level_metrics": True,
        **metrics,
    }


def build_visualize_only_benchmark_payload(
    *,
    previous_result,
    summaries,
    artifacts,
    requested_algorithms,
    algorithms=None,
    results_key="results",
):
    previous_metrics = dict(previous_result.get("metrics", {}) or {})
    carry_metrics = {
        key: value
        for key, value in previous_metrics.items()
        if key not in {results_key, "results", "algorithms", "requested_algorithms", "visualization_only"}
    }
    effective_algorithms = list(algorithms or previous_metrics.get("algorithms") or [])
    return build_benchmark_payload(
        objective=previous_result.get("objective", previous_metrics.get("objective")),
        feasible=previous_result.get("feasible", previous_metrics.get("feasible")),
        artifacts=artifacts,
        **carry_metrics,
        results=summaries,
        algorithms=effective_algorithms,
        requested_algorithms=previous_metrics.get("requested_algorithms", list(requested_algorithms)),
        visualization_only=True,
    )


# Scalar summaries
SCALAR_TIMING_FIELDS = (
    "total_wall_time",
    "last_trans_time",
    "initial_transform_time",
    "final_transform_time",
    "initial_ip_time",
)
SEQUENCE_TIMING_FIELDS = ("outer_iter_time", "inner_iter_time", "solver_iter_time")

def final_scalar(value):
    array = np.asarray(value).reshape(-1)
    return float(array[-1])


def final_or_nan(value):
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return float("nan")
    return float(array[-1])


def _add_timing_summary(summary, record):
    for key in SCALAR_TIMING_FIELDS:
        if record.get(key) is not None:
            summary[key] = float(record.get(key) or 0.0)
    for key in SEQUENCE_TIMING_FIELDS:
        values = np.asarray(record.get(key, []), dtype=float).reshape(-1)
        if values.size:
            summary[key] = values.tolist()
            summary[f"total_{key}"] = float(values.sum())
            summary[f"mean_{key}"] = float(values.mean())


def summarize_run_record(record, *, reference_objective=None, include_total_iter_time=False):
    final_objective = final_scalar(record["obj_traj"])
    final_violation = final_scalar(record["cons_traj"])
    summary = {
        "final_objective": final_objective,
        "final_violation": final_violation,
        "iterations": int(len(record["iter_time"])),
    }
    if include_total_iter_time:
        summary["total_iter_time"] = float(sum(record["iter_time"]))
    _add_timing_summary(summary, record)
    if reference_objective is not None:
        if abs(float(reference_objective)) <= 1e-12:
            summary["objective_gap"] = None
        else:
            summary["objective_gap"] = float(abs(final_objective - float(reference_objective)) / abs(float(reference_objective)))
    return summary


def _supports_keyword(callable_obj, keyword):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _problem_tensor_kwargs(problem):
    for attr_name in ("Q", "p", "L", "A_obj"):
        tensor = getattr(problem, attr_name, None)
        if torch.is_tensor(tensor):
            return {"device": tensor.device, "dtype": tensor.dtype}
    return {"device": getattr(problem, "device", torch.device("cpu")), "dtype": torch.float32}


def _coerce_decision_tensor(problem, decision):
    if decision is None:
        return None
    kwargs = _problem_tensor_kwargs(problem)
    x = torch.as_tensor(decision, **kwargs)
    if x.ndim == 1:
        x = x.view(1, -1)
    if x.ndim != 2 or x.shape[-1] != int(getattr(problem, "nvar", x.shape[-1])):
        return None
    return x


def _call_problem_constraint(problem, x, *, include_equalities, clip):
    constraint_x = problem.constraint_x
    kwargs = {}
    if _supports_keyword(constraint_x, "clip"):
        kwargs["clip"] = clip
    if _supports_keyword(constraint_x, "eq_cons"):
        kwargs["eq_cons"] = bool(include_equalities)
    return constraint_x(x, **kwargs)


def _max_or_zero(tensor):
    if tensor is None or tensor.numel() == 0:
        return 0.0
    return float(tensor.max().detach().cpu().item())


def constraint_violation_summary(problem, decision):
    """Compute reporting metrics with a consistent full/ineq/eq split."""

    x = _coerce_decision_tensor(problem, decision)
    if x is None:
        return {}

    residual = _call_problem_constraint(problem, x, include_equalities=True, clip=False)
    if not torch.is_tensor(residual):
        residual = torch.as_tensor(residual, **_problem_tensor_kwargs(problem))
    if residual.ndim == 1:
        residual = residual.view(1, -1)

    ineq_cons = getattr(problem, "ineq_cons", None)
    eq_cons = getattr(problem, "eq_cons", None)
    if ineq_cons is None and eq_cons is None:
        inequality_residual = residual
        equality_residual = None
    else:
        ineq_idx = [] if ineq_cons is None else list(ineq_cons)
        eq_idx = [] if eq_cons is None else list(eq_cons)
        inequality_residual = residual[:, ineq_idx] if ineq_idx else residual.new_zeros((residual.shape[0], 0))
        equality_residual = residual[:, eq_idx] if eq_idx else None

    inequality_violation = _max_or_zero(torch.clamp(inequality_residual, min=0))
    equality_violation = None if equality_residual is None else _max_or_zero(equality_residual.abs())
    split_values = [inequality_violation]
    if equality_violation is not None:
        split_values.append(equality_violation)
    return {
        "final_full_violation": max(split_values),
        "final_inequality_violation": inequality_violation,
        "final_equality_violation": equality_violation,
    }


def _residual_block_diagnostics(residual, active_tol):
    residual = residual.detach()
    if residual.ndim == 1:
        residual = residual.view(1, -1)
    if residual.numel() == 0:
        return {"total": 0, "active": 0, "violated": 0, "max_residual": 0.0}
    flat = residual.reshape(-1)
    return {
        "total": int(flat.numel()),
        "active": int((flat >= -active_tol).sum().detach().cpu().item()),
        "violated": int((flat > active_tol).sum().detach().cpu().item()),
        "max_residual": float(flat.max().detach().cpu().item()),
    }


def _empty_residual_block_diagnostics(x, active_tol):
    return _residual_block_diagnostics(x.new_zeros((1, 0)), active_tol)


def _sum_block_diagnostics(*blocks):
    nonempty = [block for block in blocks if block is not None]
    if not nonempty:
        return {"total": 0, "active": 0, "violated": 0, "max_residual": 0.0}
    total = int(sum(block.get("total", 0) for block in nonempty))
    max_values = [
        float(block.get("max_residual", 0.0))
        for block in nonempty
        if block.get("total", 0)
    ]
    return {
        "total": total,
        "active": int(sum(block.get("active", 0) for block in nonempty)),
        "violated": int(sum(block.get("violated", 0) for block in nonempty)),
        "max_residual": max(max_values) if max_values else 0.0,
    }


def _stiefel_group_norm_diagnostics(problem, x, active_tol):
    groups = getattr(problem, "group_norm_constraints", None) or ()
    if not groups:
        return None
    residuals = []
    for group in groups:
        indices = torch.as_tensor(group["flat_indices"], device=x.device, dtype=torch.long)
        residuals.append(torch.norm(x[:, indices], dim=-1, p=2) - float(group["budget"]))
    return _residual_block_diagnostics(torch.stack(residuals, dim=1), active_tol)


def convex_solution_diagnostics(problem, decision, *, active_tol=1e-5):
    """Report active-set structure at a ConvexOpt decision.

    Inequality residuals follow the convention residual <= 0. A constraint is
    counted as active when residual >= -active_tol and violated when residual >
    active_tol. Equalities are reported separately; all equality rows are active
    by definition and `violated` counts rows with |residual| > active_tol.
    """

    x = _coerce_decision_tensor(problem, decision)
    if x is None:
        return {"available": False, "active_tol": float(active_tol)}
    if x.shape[0] != 1:
        x = x[:1]

    diagnostics = {
        "available": True,
        "active_tol": float(active_tol),
        "nvar": int(getattr(problem, "nvar", x.shape[-1])),
        "counts_by_type": {},
    }
    counts = diagnostics["counts_by_type"]

    if getattr(problem, "A", None) is not None:
        counts["linear"] = _residual_block_diagnostics(torch.matmul(x, problem.A.T) - problem.b, active_tol)
    else:
        counts["linear"] = _empty_residual_block_diagnostics(x, active_tol)

    if getattr(problem, "Qq", None) is not None:
        q = 0.5 * torch.einsum("bi,mij,bj->bm", x, problem.Qq, x)
        p = torch.matmul(x, problem.pq.T)
        counts["quadratic"] = _residual_block_diagnostics(q + p - problem.bq, active_tol)
    else:
        counts["quadratic"] = _empty_residual_block_diagnostics(x, active_tol)

    if getattr(problem, "G", None) is not None:
        q = torch.norm(torch.einsum("mkn,bn->bmk", problem.G, x) + problem.h.unsqueeze(0), dim=-1, p=2)
        p = torch.matmul(x, problem.C.T) + problem.d
        counts["soc"] = _residual_block_diagnostics(q - p, active_tol)
    else:
        counts["soc"] = _empty_residual_block_diagnostics(x, active_tol)

    if getattr(problem, "L", None) is not None and getattr(problem, "U", None) is not None:
        counts["box_lower"] = _residual_block_diagnostics(problem.L - x, active_tol)
        counts["box_upper"] = _residual_block_diagnostics(x - problem.U, active_tol)
    else:
        counts["box_lower"] = _empty_residual_block_diagnostics(x, active_tol)
        counts["box_upper"] = _empty_residual_block_diagnostics(x, active_tol)
    box_counts = _sum_block_diagnostics(counts["box_lower"], counts["box_upper"])
    diagnostics["active_box_total"] = box_counts["active"]
    diagnostics["box_total"] = box_counts["total"]
    diagnostics["violated_box_total"] = box_counts["violated"]
    diagnostics["box_max_residual"] = box_counts["max_residual"]

    group_counts = _stiefel_group_norm_diagnostics(problem, x, active_tol)
    if group_counts is not None:
        diagnostics["active_group_total"] = group_counts["active"]
        diagnostics["group_total"] = group_counts["total"]
        diagnostics["violated_group_total"] = group_counts["violated"]
        diagnostics["group_max_residual"] = group_counts["max_residual"]

    eq_residual = None
    if getattr(problem, "n_eq", 0) > 0 and hasattr(problem, "eq_constraint_x"):
        eq_residual = problem.eq_constraint_x(x).detach()
    if eq_residual is None or eq_residual.numel() == 0:
        equality = {"total": 0, "active": 0, "violated": 0, "max_abs_residual": 0.0}
    else:
        eq_flat = eq_residual.abs().reshape(-1)
        equality = {
            "total": int(eq_flat.numel()),
            "active": int(eq_flat.numel()),
            "violated": int((eq_flat > active_tol).sum().detach().cpu().item()),
            "max_abs_residual": float(eq_flat.max().detach().cpu().item()),
        }
    counts["equality"] = equality

    inequality_blocks = [value for key, value in counts.items() if key != "equality"]
    diagnostics["active_inequality_total"] = int(sum(block["active"] for block in inequality_blocks))
    diagnostics["inequality_total"] = int(sum(block["total"] for block in inequality_blocks))
    diagnostics["violated_inequality_total"] = int(sum(block["violated"] for block in inequality_blocks))
    diagnostics["active_total"] = diagnostics["active_inequality_total"] + equality["active"]
    diagnostics["constraint_total"] = diagnostics["inequality_total"] + equality["total"]
    diagnostics["violated_total"] = diagnostics["violated_inequality_total"] + equality["violated"]
    return diagnostics


def objective_summary(problem, decision):
    """Compute the reported objective at the same decision used for violation."""

    x = _coerce_decision_tensor(problem, decision)
    if x is None:
        return {}

    objective = problem.objective_x(x)
    if not torch.is_tensor(objective):
        objective = torch.as_tensor(objective, **_problem_tensor_kwargs(problem))
    return {"final_objective": float(objective.detach().reshape(-1).mean().cpu().item())}


def summarize_single_problem_run_record(
    problem,
    record,
    *,
    reference_objective=None,
    include_total_iter_time=False,
):
    """Summarize a fixed-problem run using full violation for reporting.

    Optimizers may record an internal violation trajectory tailored to their
    update rule. For example, Hom-ALM records equality-only violation because
    the homeomorphism handles inequalities. Benchmark rows should still compare
    all methods using the same full/equality/inequality residual split.
    """

    summary = summarize_run_record(
        record,
        reference_objective=reference_objective,
        include_total_iter_time=include_total_iter_time,
    )
    objective_at_solution = objective_summary(problem, record.get("x_solved"))
    if objective_at_solution:
        summary["optimizer_reported_final_objective"] = summary["final_objective"]
        summary.update(objective_at_solution)
        if reference_objective is not None:
            if abs(float(reference_objective)) <= 1e-12:
                summary["objective_gap"] = None
            else:
                summary["objective_gap"] = float(
                    abs(summary["final_objective"] - float(reference_objective)) / abs(float(reference_objective))
                )
    split = constraint_violation_summary(problem, record.get("x_solved"))
    if split:
        summary["optimizer_reported_final_violation"] = summary["final_violation"]
        summary.update(split)
        summary["final_violation"] = split["final_full_violation"]
    if record.get("hom_map_forward_mode") is not None:
        summary["hom_map_forward_mode"] = record.get("hom_map_forward_mode")
    if record.get("hom_map_forward_mode_counts") is not None:
        summary["hom_map_forward_mode_counts"] = record.get("hom_map_forward_mode_counts")
    for key in ("hom_map_smooth", "hom_map_smooth_tie_tol", "hom_map_smooth_temperature"):
        if record.get(key) is not None:
            summary[key] = record.get(key)
    if record.get("final_first_order_lagrangian_gap") is not None:
        summary["final_first_order_lagrangian_gap"] = float(record["final_first_order_lagrangian_gap"])
    if record.get("first_order_lagrangian_gap_traj") is not None:
        values = np.asarray(record.get("first_order_lagrangian_gap_traj"), dtype=float).reshape(-1)
        if values.size:
            summary["first_order_lagrangian_gap_traj"] = values.tolist()
            summary["final_first_order_lagrangian_gap"] = float(values[-1])
    if record.get("lagrangian_gradient") is not None:
        summary["lagrangian_gradient"] = record.get("lagrangian_gradient")
    if record.get("hom_map_gradient") is not None:
        summary["hom_map_gradient"] = record.get("hom_map_gradient")
    return summary


# Artifact writers

def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def save_numpy(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, payload, allow_pickle=True)
    return path


def save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


COMPARISON_STORAGE_VERSION = 1
COMPARISON_RECORDS_DIRNAME = "records"
COMPARISON_MANIFEST_NAME = "manifest.json"
COMPARISON_SUMMARY_NAME = "summary.json"


def _safe_algorithm_filename(name):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    if not safe:
        digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:12]
        return f"algorithm_{digest}.npy"
    return f"{safe}.npy"


def _json_scalar(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return None
    if hasattr(value, "item"):
        try:
            item = value.item()
        except Exception:
            return None
        if item is value:
            return None
        return _json_scalar(item)
    return None


def _slim_json_payload(payload):
    scalar = _json_scalar(payload)
    if scalar is not None or payload is None:
        return scalar
    if isinstance(payload, dict):
        slim = {}
        for key, value in payload.items():
            slim_value = _slim_json_payload(value)
            if slim_value is not None or value is None:
                slim[str(key)] = slim_value
        return slim
    if isinstance(payload, (list, tuple)):
        if len(payload) <= 8:
            values = []
            for value in payload:
                scalar_value = _json_scalar(value)
                if scalar_value is None and value is not None:
                    return None
                values.append(scalar_value)
            return values
        return None
    return None


def slim_comparison_summary(summary):
    return _slim_json_payload(dict(summary or {}))


def comparison_storage_paths(output_dir):
    root = artifact_root(output_dir)
    if root is None:
        return None
    return {
        "root": root,
        "records_dir": root / COMPARISON_RECORDS_DIRNAME,
        "manifest": root / COMPARISON_MANIFEST_NAME,
        "summary": root / COMPARISON_SUMMARY_NAME,
    }


def _read_json_or_empty(path):
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _merge_order(*groups):
    seen = set()
    merged_order = []
    for group in groups:
        for name in list(group or []):
            if name in seen:
                continue
            seen.add(name)
            merged_order.append(name)
    return merged_order


def _absolute_record_path(output_dir, record_ref):
    path = Path(record_ref)
    if path.is_absolute():
        return path
    return Path(output_dir) / path


def load_incremental_comparison_artifacts(
    output_dir,
    *,
    algorithms=None,
):
    """Load comparison records from the manifest-backed per-algorithm store."""

    if output_dir is None:
        raise ValueError("visualize_only=True requires output_dir so existing records can be loaded.")
    paths = comparison_storage_paths(output_dir)
    manifest = _read_json_or_empty(paths["manifest"])
    if not manifest:
        raise FileNotFoundError(f"Missing comparison manifest artifact: {paths['manifest']}")
    summaries = _read_json_or_empty(paths["summary"])
    result_path = Path(output_dir) / "result.json"
    previous_result = _read_json_or_empty(result_path)
    requested = list(algorithms or manifest.get("algorithms") or [])
    record_refs = dict(manifest.get("records") or {})
    missing = [name for name in requested if name not in record_refs]
    if missing:
        raise FileNotFoundError(
            "Missing comparison records for requested algorithms: "
            + ", ".join(str(name) for name in missing)
        )
    records = {}
    for name in requested:
        record_path = _absolute_record_path(output_dir, record_refs[name])
        if not record_path.exists():
            raise FileNotFoundError(f"Missing comparison record for {name}: {record_path}")
        records[name] = np.load(record_path, allow_pickle=True).item()
    loaded_summaries = {name: summaries.get(name, {}) for name in requested}
    return records, loaded_summaries, previous_result, manifest


def save_incremental_comparison_artifacts(
    output_dir,
    *,
    records,
    summaries,
    algorithm_order=None,
    manifest_metadata=None,
):
    """Save comparison records per algorithm and keep a lightweight manifest/summary."""

    root_paths = comparison_storage_paths(output_dir)
    if root_paths is None:
        comparison_views = build_single_instance_comparison_views(
            summaries,
            route="iterative",
            runtime_key="total_iter_time",
        )
        return {}, comparison_views, dict(records), dict(summaries)

    records_dir = root_paths["records_dir"]
    records_dir.mkdir(parents=True, exist_ok=True)
    old_manifest = _read_json_or_empty(root_paths["manifest"])
    old_summaries = _read_json_or_empty(root_paths["summary"])
    old_record_refs = dict(old_manifest.get("records") or {})

    record_refs = dict(old_record_refs)
    slim_summaries = dict(old_summaries)
    for algorithm, record in dict(records or {}).items():
        record_path = records_dir / _safe_algorithm_filename(algorithm)
        save_numpy(record_path, record)
        record_refs[algorithm] = artifact_ref(output_dir, record_path)
        summary = slim_comparison_summary(summaries.get(algorithm, {}))
        summary["record_path"] = record_refs[algorithm]
        slim_summaries[algorithm] = summary

    stored_algorithms = _merge_order(
        old_manifest.get("algorithms"),
        algorithm_order,
        records.keys(),
        record_refs.keys(),
    )
    stored_algorithms = [name for name in stored_algorithms if name in record_refs]
    manifest = {
        "version": COMPARISON_STORAGE_VERSION,
        "algorithms": stored_algorithms,
        "records": {name: record_refs[name] for name in stored_algorithms},
    }
    if manifest_metadata:
        manifest["metadata"] = _slim_json_payload(manifest_metadata)
    save_json(root_paths["manifest"], manifest)
    save_json(root_paths["summary"], {name: slim_summaries[name] for name in stored_algorithms})

    merged_records = {}
    for algorithm in stored_algorithms:
        if algorithm in records:
            merged_records[algorithm] = records[algorithm]
            continue
        record_path = _absolute_record_path(output_dir, record_refs[algorithm])
        if record_path.exists():
            merged_records[algorithm] = np.load(record_path, allow_pickle=True).item()
    merged_summaries = {name: slim_summaries[name] for name in stored_algorithms if name in slim_summaries}

    artifacts = {
        "records": artifact_ref(output_dir, records_dir),
        "records_dir": artifact_ref(output_dir, records_dir),
        "manifest": artifact_ref(output_dir, root_paths["manifest"]),
        "summary": artifact_ref(output_dir, root_paths["summary"]),
    }
    comparison_views = build_single_instance_comparison_views(
        merged_summaries,
        route="iterative",
        runtime_key="total_iter_time",
    )
    return artifacts, comparison_views, merged_records, merged_summaries


def save_table_artifacts(
    output_dir,
    *,
    base_name,
    rows,
    fieldnames,
    markdown_text=None,
    json_key="summary_json",
    csv_key="summary_csv",
    markdown_key="summary_markdown",
):
    root = artifact_root(output_dir)
    if root is None:
        return {}, None, None, None
    json_path = save_json(root / f"{base_name}.json", rows)
    csv_path = write_csv(root / f"{base_name}.csv", rows, fieldnames=fieldnames)
    markdown_path = None
    if markdown_text is not None:
        markdown_path = save_text(root / f"{base_name}.md", markdown_text)
    artifacts = artifact_mapping(
        output_dir,
        **{
            json_key: json_path,
            csv_key: csv_path,
            markdown_key: markdown_path,
        },
    )
    return artifacts, json_path, csv_path, markdown_path


# Comparison helpers
#
# `run_algorithm_suite(...)` is route-agnostic.
# The single-instance comparison row helpers below are currently only used by
# the single/fixed-problem track.

def run_algorithm_suite(algorithms, run_record, summarize_record):
    records = {}
    summaries = {}
    for algorithm in algorithms:
        record = run_record(algorithm)
        records[algorithm] = record
        summaries[algorithm] = summarize_record(record)
    return records, summaries


def make_single_instance_solver_row(*, method, objective, violation, runtime_total, feasibility_tol=1e-5, **extras):
    extras.setdefault("total_wall_time", runtime_total)
    return make_comparison_row(
        route="solver",
        method=method,
        objective=objective,
        feasible=(violation is not None and float(violation) <= float(feasibility_tol)),
        violation=violation,
        runtime_total=runtime_total,
        **extras,
    )


# Runtime and small conversion helpers

def trajectory_dim(traj, fallback):
    if torch.is_tensor(traj) and traj.ndim == 2 and traj.shape[1] > 0:
        return int(traj.shape[1])
    return int(fallback)


def resolve_runtime(device=None, dtype=None, *, default_device="cpu", default_dtype=torch.float32):
    runtime_device = resolve_torch_device(device, default=default_device)
    runtime_dtype = resolve_torch_dtype(dtype, default=default_dtype)
    return runtime_device, runtime_dtype


def as_builtin(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return value.item() if hasattr(value, "item") else value


def ensure_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


# Config normalization helpers
#
# Boundary:
# - `merged(...)` is the generic deep-merge primitive shared across both main
#   experiment tracks.
# - `normalize_single_*` is owned by the single/fixed-problem track.
# - `normalize_solver_run_config(...)` is shared by solver-backed routes,
#   including QCQP warmstart baselines and other exact-solver call sites.
# - `normalize_jcc_*` is single-track-specific; QCQP/INN-specific config
#   assembly belongs in `_benchmark_parametric_common.py`.

def merged(base, overrides=None):
    if base is None:
        payload = {}
    elif isinstance(base, dict):
        payload = copy.deepcopy(base)
    else:
        payload = dict(base)
    if overrides:
        payload.update(copy.deepcopy(overrides))
    return payload


def deep_merged(base, overrides=None):
    """Recursively merge nested config dictionaries."""

    if base is None:
        payload = {}
    elif isinstance(base, dict):
        payload = copy.deepcopy(base)
    else:
        payload = dict(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = deep_merged(payload[key], value)
        else:
            payload[key] = copy.deepcopy(value)
    return payload


def normalize_inner_solver_common_config(inner_solver_common):
    """Keep inner-solver controls from overwriting outer ALM controls."""

    if not inner_solver_common:
        return inner_solver_common
    config = copy.deepcopy(inner_solver_common)
    _normalize_config_alias(
        config,
        alias="learning_rate",
        canonical="inner_learning_rate",
        context="inner_solver_common",
    )
    _normalize_config_alias(config, alias="lr_decay", canonical="inner_lr_decay", context="inner_solver_common")
    _normalize_config_alias(
        config,
        alias="stepsize_rule",
        canonical="inner_stepsize_rule",
        context="inner_solver_common",
    )
    return config


def _normalize_config_alias(config, *, alias, canonical, context):
    if alias not in config:
        return
    alias_value = config.pop(alias)
    if canonical in config and config[canonical] != alias_value:
        raise ValueError(
            f"Conflicting {context} parameters: {alias}={alias_value!r} "
            f"and {canonical}={config[canonical]!r}."
        )
    config[canonical] = alias_value


def normalize_outer_common_config(outer_common, *, context="outer_common"):
    """Keep outer-loop ALM controls explicit before algorithm-level merges."""

    if not outer_common:
        return outer_common
    config = copy.deepcopy(outer_common)
    _normalize_config_alias(config, alias="outer_learning_rate", canonical="learning_rate", context=context)
    _normalize_config_alias(config, alias="lr_decay", canonical="outer_lr_decay", context=context)
    _normalize_config_alias(config, alias="stepsize_rule", canonical="outer_stepsize_rule", context=context)
    return config


ALM_ITERATIVE_ALGORITHMS = ("Penalty", "Prox-Penalty", "ALM", "Prox-ALM", "Hom-ALM", "Prox-Hom-ALM")
ALM_CVXPY_ALGORITHMS = ("Penalty-EQ", "Prox-Penalty-EQ", "ALM-EQ", "Prox-ALM-EQ")
ALM_FAMILY_ALGORITHMS = (*ALM_ITERATIVE_ALGORITHMS, *ALM_CVXPY_ALGORITHMS)
ALM_CVXPY_OVERRIDE_KEYS = {
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
ALM_CVXPY_IGNORED_OVERRIDE_KEYS = {
    "learning_rate",
    "outer_lr_decay",
    "outer_stepsize_rule",
    "check_first_order_lagrangian_gap",
    "first_order_lagrangian_gap_threshold",
    "outer_first_order_gap_threshold",
}


def normalize_alm_algorithm_override_config(algorithm_name, overrides):
    """Normalize ALM-family per-algorithm overrides before merging."""

    if not overrides:
        return overrides
    if algorithm_name in ALM_ITERATIVE_ALGORITHMS:
        return normalize_outer_common_config(overrides, context=f"{algorithm_name} override")
    if algorithm_name in ALM_CVXPY_ALGORITHMS:
        config = normalize_outer_common_config(overrides, context=f"{algorithm_name} override")
        unknown_keys = sorted(set(config) - ALM_CVXPY_OVERRIDE_KEYS - ALM_CVXPY_IGNORED_OVERRIDE_KEYS)
        if unknown_keys:
            raise ValueError(f"Unsupported {algorithm_name} override keys: {unknown_keys}")
        return {key: value for key, value in config.items() if key in ALM_CVXPY_OVERRIDE_KEYS}
    return overrides


def merge_alm_outer_common(params, outer_common):
    """Apply shared outer-loop controls to every active ALM-family config."""

    if not outer_common:
        return params
    normalized_outer = normalize_outer_common_config(outer_common)
    for name in ALM_FAMILY_ALGORITHMS:
        if name in params and isinstance(params[name], dict):
            params[name] = merged(
                params[name],
                normalize_alm_algorithm_override_config(name, normalized_outer),
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


def apply_param_overrides(
    params,
    common_config=None,
    problem_config=None,
    algorithm_config=None,
    *,
    common_overrides=None,
    common_config_overrides=None,
    problem_overrides=None,
    problem_config_overrides=None,
    algorithm_params=None,
    algorithm_config_overrides=None,
):
    """Apply common/problem/algorithm override groups once at the final merge boundary."""

    common_config = normalize_single_common_config(
        common_config=common_config,
        common_overrides=common_overrides,
        common_config_overrides=common_config_overrides,
    )
    problem_config = normalize_single_problem_config(
        problem_config=problem_config,
        problem_overrides=problem_overrides,
        problem_config_overrides=problem_config_overrides,
    )
    algorithm_config = normalize_algorithm_config_group(
        algorithm_config=algorithm_config,
        algorithm_params=algorithm_params,
        algorithm_config_overrides=algorithm_config_overrides,
    )
    if "common" in params:
        params["common"] = merged(params["common"], common_config)
    if common_config:
        for key, value in list(params.items()):
            if isinstance(value, dict) and "opt_type" in value:
                params[key] = merged(value, normalize_alm_algorithm_override_config(key, common_config))
    if "prob" in params:
        params["prob"] = merged(params["prob"], problem_config)
    if algorithm_config:
        unknown_algorithms = sorted(name for name in algorithm_config if name not in params)
        if unknown_algorithms:
            raise ValueError(f"Unknown algorithm_config entries: {unknown_algorithms}")
        for name, overrides in algorithm_config.items():
            if name in params and isinstance(params[name], dict):
                params[name] = merged(params[name], normalize_alm_algorithm_override_config(name, overrides))
    return params


# Single/fixed-problem config helpers

def normalize_single_common_config(
    *,
    common=None,
    common_overrides=None,
    common_config=None,
    common_config_overrides=None,
):
    payload = merged(common, common_config)
    payload = merged(payload, common_overrides)
    payload = merged(payload, common_config_overrides)
    return payload


def normalize_single_problem_config(
    *,
    problem_overrides=None,
    problem_config=None,
    problem_config_overrides=None,
):
    payload = merged(problem_overrides, problem_config)
    payload = merged(payload, problem_config_overrides)
    return payload


def normalize_single_algorithm_config(
    *,
    algorithm_params=None,
    algorithm_config=None,
    algorithm_config_overrides=None,
):
    payload = deep_merged(algorithm_params, algorithm_config)
    payload = deep_merged(payload, algorithm_config_overrides)
    return payload


def normalize_algorithm_config_group(
    *,
    algorithm_params=None,
    algorithm_config=None,
    algorithm_config_overrides=None,
):
    """Merge per-algorithm config groups."""

    return normalize_single_algorithm_config(
        algorithm_params=algorithm_params,
        algorithm_config=algorithm_config,
        algorithm_config_overrides=algorithm_config_overrides,
    )

# Shared exact-solver config helper

def normalize_solver_run_config(
    *,
    solver_name="ipopt",
    solver_options=None,
    solver_config=None,
    solver_config_overrides=None,
):
    normalized = {
        "solver_name": solver_name,
        "solver_options": dict(solver_options or {}),
    }
    for group in (solver_config, solver_config_overrides):
        if not group:
            continue
        translated = dict(group)
        if "name" in translated and "solver_name" not in translated:
            translated["solver_name"] = translated.pop("name")
        if "options" in translated and "solver_options" not in translated:
            translated["solver_options"] = translated.pop("options")
        normalized = merged(normalized, translated)
    normalized["solver_name"] = str(normalized.get("solver_name", solver_name))
    normalized["solver_options"] = dict(normalized.get("solver_options") or {})
    return normalized


# JCC-specific config helpers

def normalize_jcc_problem_config(
    *,
    n_scenarios,
    epsilon,
    demand_std,
    seed,
    problem_config=None,
    problem_config_overrides=None,
):
    problem_cfg = {
        "n_scenarios": int(n_scenarios),
        "epsilon": float(epsilon),
        "demand_std": float(demand_std),
        "seed": int(seed),
    }
    for payload in (problem_config, problem_config_overrides):
        if not payload:
            continue
        problem_cfg.update(dict(payload))
    problem_cfg["n_scenarios"] = int(problem_cfg["n_scenarios"])
    problem_cfg["seed"] = int(problem_cfg["seed"])
    problem_cfg["epsilon"] = float(problem_cfg["epsilon"])
    problem_cfg["demand_std"] = float(problem_cfg["demand_std"])
    return problem_cfg


def normalize_jcc_solver_configs(
    *,
    solver_configs=None,
    solver_config_overrides=None,
):
    canonical = {
        "mixed_integer": {"solver": "GUROBI", "M": 1000.0, "verbose": False},
        "cvar": {"solver": "MOSEK", "verbose": False},
        "scenario": {"solver": "MOSEK", "verbose": False},
    }
    for payload in (solver_configs, solver_config_overrides):
        if not payload:
            continue
        for name, cfg in dict(payload).items():
            if name not in canonical:
                raise ValueError(f"Unsupported JCC solver config group: {name}")
            canonical[name] = merged(canonical[name], cfg)
    return canonical


def normalize_jcc_lagrangian_config(
    *,
    lagrangian_baseline_config=None,
    lagrangian_baseline_config_overrides=None,
):
    shared_controls = penalty_family_shared_defaults()
    lag_args = normalize_penalty_method_config(
        penalty_method_args={
            "learning_rate": 1e-3,
            "outer_iterations": 50,
            "inner_iterations": 10,
            "max_running_time": 300,
            "convergence_threshold": 1e-5,
            "outer_stepsize_rule": "adaptive",
            "inner_stepsize_rule": "constant",
            "outer_lr_decay": 0.999,
            "inner_lr_decay": 0.999,
            "momentum": 0.0,
            "proximal_space": "x",
            "lagrangian_gradient": "autograd",
            **shared_controls,
            "opt": "gd",
        },
    )
    for payload in (lagrangian_baseline_config, lagrangian_baseline_config_overrides):
        lag_args = normalize_penalty_method_config(
            penalty_method_args=lag_args,
            penalty_method_args_overrides=normalize_alm_algorithm_override_config("ALM", payload),
        )
    return lag_args


def penalty_family_shared_defaults(
    *,
    dual_learning_rate=1e-1,
    penalty_coef=10.0,
    penalty_growth=1.1,
    proximal_coef=0.1,
    max_penalty=1e2,
    max_dual=1e2,
    use_lagrangian=True,
    use_penalty=True,
    use_proximal=False,
):
    return {
        "dual_learning_rate": dual_learning_rate,
        "penalty_coef": penalty_coef,
        "penalty_growth": penalty_growth,
        "proximal_coef": proximal_coef,
        "max_penalty": max_penalty,
        "max_dual": max_dual,
        "use_lagrangian": use_lagrangian,
        "use_penalty": use_penalty,
        "use_proximal": use_proximal,
    }


def default_penalty_method_config():
    return {
        "learning_rate": 1e-3,
        "outer_iterations": 50,
        "inner_iterations": 10,
        "max_running_time": 300,
        "convergence_threshold": 1e-6,
        "outer_stepsize_rule": "adaptive",
        "inner_stepsize_rule": "constant",
        "inner_solver": "gd",
        "inner_stopping_rule": "fixed",
        "inner_proximal_update_iterations": 1,
        "inner_proximal_coef": 0.0,
        "inner_proximal_space": "x",
        "acceleration_method": "none",
        "acceleration_space": "x",
        "outer_lr_decay": 0.999,
        "inner_lr_decay": 0.999,
        "momentum": 0.0,
        "proximal_space": "x",
        **penalty_family_shared_defaults(),
        "opt": "gd",
    }


def normalize_penalty_method_config(
    *,
    penalty_method_args=None,
    penalty_method_args_overrides=None,
):
    penalty_method_args = normalize_outer_common_config(
        penalty_method_args,
        context="penalty method config",
    )
    penalty_method_args_overrides = normalize_outer_common_config(
        penalty_method_args_overrides,
        context="penalty method override",
    )
    cfg = merged(default_penalty_method_config(), penalty_method_args)
    cfg = merged(cfg, penalty_method_args_overrides)
    return cfg


# Algorithm parameter builders
#
# These builders are currently single-track-only and mainly serve the
# convex/SOCP/poly-star/MaxCut iterative benchmarks plus the Hom-ALM variants.

CONVEX_ALM_INNER_ITERATIONS = 50

EQ_CVXPY_BASELINE_SPECS = {
    "Penalty-EQ": {
        "use_lagrangian": False,
        "use_penalty": True,
        "use_proximal": False,
    },
    "Prox-Penalty-EQ": {
        "use_lagrangian": False,
        "use_penalty": True,
        "use_proximal": True,
    },
    "ALM-EQ": {
        "use_lagrangian": True,
        "use_penalty": True,
        "use_proximal": False,
    },
    "Prox-ALM-EQ": {
        "use_lagrangian": True,
        "use_penalty": True,
        "use_proximal": True,
    },
}

EQ_CVXPY_BASELINE_KEYS = (
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
)


def build_first_order_subproblem_params(
    common,
    *,
    outer_iterations,
    inner_iterations,
    overrides=None,
):
    """Canonical Lagrangian subproblem config for PGD projection and FW LOO."""

    return normalize_penalty_method_config(
        penalty_method_args=build_first_order_subproblem_defaults(
            common,
            outer_iterations=outer_iterations,
            inner_iterations=inner_iterations,
        ),
        penalty_method_args_overrides=overrides,
    )


def base_common_params(seed, max_iterations, max_running_time, learning_rate, stepsize_rule, lr_decay):
    return {
        "seed": seed,
        "convergence_threshold": 1e-6,
        "max_iterations": max_iterations,
        "max_running_time": max_running_time,
        "opt": "gd",
        "learning_rate": learning_rate,
        "stepsize_rule": stepsize_rule,
        "hom_p_norm": 2,
        "hom_map_explicit_gradient_rule": "polynomial",
        "hom_map_smooth_tie_tol": 1e-7,
        "hom_map_smooth_temperature": 1e-4,
        "momentum": 0.0,
        "warm_start": False,
        "acceleration_method": "none",
        "acceleration_space": "z",
        "smooth": False,
        "lr_decay": lr_decay,
    }


def build_first_order_method_params(
    common,
    *,
    projection_outer_iterations,
    projection_inner_iterations,
    linearization_outer_iterations=None,
    linearization_inner_iterations=None,
    projection_subproblem=None,
    linearization_subproblem=None,
    include_fw=True,
    include_rd=True,
    hom_p_norm=2,
    hom_map_gradient="explicit",
    hom_momentum=None,
):
    pgd_projection_subproblem = build_first_order_subproblem_params(
        common,
        outer_iterations=projection_outer_iterations,
        inner_iterations=projection_inner_iterations,
        overrides=projection_subproblem,
    )
    effective_linearization_outer_iterations = (
        linearization_outer_iterations
        if linearization_outer_iterations is not None
        else projection_outer_iterations
    )
    effective_linearization_inner_iterations = (
        linearization_inner_iterations
        if linearization_inner_iterations is not None
        else projection_inner_iterations
    )
    fw_linearization_subproblem = build_first_order_subproblem_params(
        common,
        outer_iterations=effective_linearization_outer_iterations,
        inner_iterations=effective_linearization_inner_iterations,
        overrides=linearization_subproblem,
    )
    x_space_common = {**common, "acceleration_space": "x"}
    params = {
        "PGD": {
            **x_space_common,
            "opt_type": "PGD",
            "projection_outer_iterations": projection_outer_iterations,
            "projection_inner_iterations": projection_inner_iterations,
            "projection_subproblem": pgd_projection_subproblem,
        },
        "Hom-PGD": {
            **common,
            "opt_type": "Hom-PGD",
            "hom_p_norm": hom_p_norm,
            "hom_map_gradient": hom_map_gradient,
            "use_proximal": False,
            "proximal_update_iterations": 1,
            "proximal_coef": 0.0,
            "proximal_space": "z",
        },
    }
    if hom_momentum is not None:
        params["Hom-PGD"]["momentum"] = hom_momentum
    if include_fw:
        params["FW"] = {
            **x_space_common,
            "opt_type": "FW",
            "linearization_outer_iterations": effective_linearization_outer_iterations,
            "linearization_inner_iterations": effective_linearization_inner_iterations,
            "linearization_subproblem": fw_linearization_subproblem,
        }
    if include_rd:
        params["RD"] = {
            **x_space_common,
            "opt_type": "RD",
        }
    return params


def filter_params(common, keys):
    return {key: common[key] for key in keys if key in common}


def optional_params(**kwargs):
    return {key: value for key, value in kwargs.items() if value is not None}


def build_convex_penalty_method_kwargs(
    common,
    *,
    outer_iterations,
    inner_iterations=CONVEX_ALM_INNER_ITERATIONS,
    dual_learning_rate=1e-1,
    proximal_space="x",
    use_lagrangian=True,
    use_penalty=True,
    use_proximal=False,
):
    """Canonical penalty/ALM defaults for fixed convex comparison experiments."""

    return {
        "outer_iterations": outer_iterations,
        "inner_iterations": inner_iterations,
        "proximal_space": proximal_space,
        "outer_stepsize_rule": common.get("outer_stepsize_rule", common["stepsize_rule"]),
        "inner_stepsize_rule": "constant",
        "inner_solver": "gd",
        "inner_proximal_update_iterations": 1,
        "inner_proximal_coef": 0.0,
        "inner_proximal_space": proximal_space,
        "inner_restart": False,
        "acceleration_space": proximal_space,
        **penalty_family_shared_defaults(
            dual_learning_rate=dual_learning_rate,
            use_lagrangian=use_lagrangian,
            use_penalty=use_penalty,
            use_proximal=use_proximal,
        ),
    }


def build_penalty_method_params(
    common,
    *,
    outer_iterations=None,
    inner_iterations=None,
    dual_learning_rate=None,
    penalty_coef=None,
    penalty_growth=None,
    proximal_coef=None,
    proximal_space=None,
    inner_learning_rate=None,
    outer_lr_decay=None,
    inner_lr_decay=None,
    outer_stepsize_rule=None,
    inner_stepsize_rule=None,
    inner_solver=None,
    inner_proximal_update_iterations=None,
    inner_proximal_coef=None,
    inner_proximal_space=None,
    inner_restart=None,
    acceleration_method=None,
    acceleration_space=None,
    lagrangian_gradient="explicit",
    max_penalty=None,
    max_dual=None,
    use_lagrangian=True,
    use_penalty=True,
    use_proximal=False,
):
    params = normalize_penalty_method_config(
        penalty_method_args={
            **filter_params(
                common,
                (
                    "learning_rate",
                    "inner_learning_rate",
                    "max_running_time",
                    "outer_lr_decay",
                    "inner_lr_decay",
                    "convergence_threshold",
                    "outer_stepsize_rule",
                    "opt",
                    "momentum",
                    "verbose_interval",
                    "inner_restart",
                    "acceleration_method",
                    "acceleration_space",
                ),
            ),
            "outer_lr_decay": common.get("outer_lr_decay", common.get("lr_decay")),
            "inner_lr_decay": common.get("inner_lr_decay", common.get("lr_decay")),
            "use_lagrangian": use_lagrangian,
            "use_penalty": use_penalty,
            "use_proximal": use_proximal,
        },
        penalty_method_args_overrides={
            **optional_params(
                outer_iterations=outer_iterations,
                inner_iterations=inner_iterations,
                dual_learning_rate=dual_learning_rate,
                penalty_coef=penalty_coef,
                penalty_growth=penalty_growth,
                proximal_coef=proximal_coef,
                proximal_space=proximal_space,
                inner_learning_rate=inner_learning_rate,
                outer_lr_decay=outer_lr_decay,
                inner_lr_decay=inner_lr_decay,
                outer_stepsize_rule=outer_stepsize_rule,
                inner_stepsize_rule=inner_stepsize_rule,
                inner_solver=inner_solver,
                inner_proximal_update_iterations=inner_proximal_update_iterations,
                inner_proximal_coef=inner_proximal_coef,
                inner_restart=inner_restart,
                acceleration_method=acceleration_method,
                max_penalty=max_penalty,
                max_dual=max_dual,
            ),
            **{"inner_proximal_space": inner_proximal_space if inner_proximal_space is not None else "x"},
            **{"acceleration_space": acceleration_space if acceleration_space is not None else "x"},
            "lagrangian_gradient": lagrangian_gradient,
        },
    )
    params["opt_type"] = "ALM"
    return params


def build_eq_alm_method_params(
    common,
    *,
    opt_type="ALM-EQ",
    outer_iterations=None,
    dual_learning_rate=None,
    penalty_coef=None,
    penalty_growth=None,
    proximal_coef=None,
    max_penalty=None,
    max_dual=None,
    use_lagrangian=True,
    use_penalty=True,
    use_proximal=False,
):
    params = normalize_penalty_method_config(
        penalty_method_args={
            **filter_params(
                common,
                (
                    "max_running_time",
                    "convergence_threshold",
                ),
            ),
            "inner_iterations": 1,
            "proximal_space": "x",
            "use_lagrangian": use_lagrangian,
            "use_penalty": use_penalty,
            "use_proximal": use_proximal,
        },
        penalty_method_args_overrides={
            **optional_params(
                outer_iterations=outer_iterations,
                dual_learning_rate=dual_learning_rate,
                penalty_coef=penalty_coef,
                penalty_growth=penalty_growth,
                proximal_coef=proximal_coef,
                max_penalty=max_penalty,
                max_dual=max_dual,
            ),
        },
    )
    return {
        **{key: params[key] for key in EQ_CVXPY_BASELINE_KEYS},
        "opt_type": opt_type,
    }


def build_eq_cvxpy_baseline_params(common, *, outer_iterations):
    return {
        name: build_eq_alm_method_params(
            common,
            opt_type=name,
            outer_iterations=outer_iterations,
            **penalty_family_shared_defaults(**flags),
        )
        for name, flags in EQ_CVXPY_BASELINE_SPECS.items()
    }


def build_hom_penalty_method_params(
    common,
    *,
    outer_iterations=None,
    inner_iterations=None,
    dual_learning_rate=None,
    penalty_coef=None,
    penalty_growth=None,
    proximal_coef=None,
    proximal_space=None,
    inner_learning_rate=None,
    outer_lr_decay=None,
    inner_lr_decay=None,
    outer_stepsize_rule=None,
    inner_stepsize_rule=None,
    inner_solver=None,
    inner_proximal_update_iterations=None,
    inner_proximal_coef=None,
    inner_proximal_space=None,
    inner_restart=None,
    acceleration_method=None,
    acceleration_space=None,
    hom_map_gradient="explicit",
    max_penalty=None,
    max_dual=None,
    use_lagrangian=True,
    use_penalty=True,
    use_proximal=False,
):
    params = build_penalty_method_params(
        common,
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
        dual_learning_rate=dual_learning_rate,
        penalty_coef=penalty_coef,
        penalty_growth=penalty_growth,
        proximal_coef=proximal_coef,
        proximal_space=proximal_space if proximal_space is not None else "z",
        inner_learning_rate=inner_learning_rate,
        outer_lr_decay=outer_lr_decay,
        inner_lr_decay=inner_lr_decay,
        outer_stepsize_rule=outer_stepsize_rule,
        inner_stepsize_rule=inner_stepsize_rule,
        inner_solver=inner_solver,
        inner_proximal_update_iterations=inner_proximal_update_iterations,
        inner_proximal_coef=inner_proximal_coef,
        inner_proximal_space=inner_proximal_space if inner_proximal_space is not None else "z",
        inner_restart=inner_restart,
        acceleration_method=acceleration_method,
        acceleration_space=acceleration_space if acceleration_space is not None else "z",
        max_penalty=max_penalty,
        max_dual=max_dual,
        use_lagrangian=use_lagrangian,
        use_penalty=use_penalty,
        use_proximal=use_proximal,
    )
    params["hom_map_gradient"] = hom_map_gradient
    params["opt_type"] = "Hom-ALM"
    return params


# Convex problem and reference-context helpers
#
# These helpers are owned by the single/fixed-problem benchmark track.
# Parametric QCQP/JCC instance construction lives in the problem layer and in
# `_benchmark_parametric_common.py`.

def build_convex_problem_config(
    *,
    seed,
    n_var,
    n_linear_cons,
    n_soc_cons,
    n_qua_cons,
    n_lin_eq=0,
    obj="quad",
    x_lower=-5.0,
    x_upper=5.0,
    margin_scale=0.1,
    anchor_interior_ratio=0.8,
    objective_quadratic_type="diagonal",
    constraint_quadratic_type="diagonal",
    quad_diag_lower=1e-2,
    quad_diag_upper=1.0,
    low_rank_quad_ridge=1e-2,
    problem_overrides=None,
):
    config = create_test_problem(
        {
            "obj": obj,
            "seed": seed,
            "n_var": n_var,
            "n_linear_cons": n_linear_cons,
            "n_qua_cons": n_qua_cons,
            "n_soc_cons": n_soc_cons,
            "n_lin_eq": n_lin_eq,
            "x_lower": x_lower,
            "x_upper": x_upper,
            "margin_scale": margin_scale,
            "anchor_interior_ratio": anchor_interior_ratio,
            "objective_quadratic_type": objective_quadratic_type,
            "constraint_quadratic_type": constraint_quadratic_type,
            "quad_diag_lower": quad_diag_lower,
            "quad_diag_upper": quad_diag_upper,
            "low_rank_quad_ridge": low_rank_quad_ridge,
        }
    )
    if problem_overrides:
        config.update(problem_overrides)
    return config


def normalize_convex_problem_type(problem_type):
    normalized = str(problem_type).lower()
    if normalized not in {"socp", "socp_eq"}:
        raise ValueError("problem_type must be one of: 'socp', 'socp_eq'.")
    return normalized


def build_convex_problem(
    *,
    config,
    runtime_device,
    runtime_dtype,
    problem_type="socp",
):
    problem_type = normalize_convex_problem_type(problem_type)
    problem = ConvexOptEq(config) if problem_type == "socp_eq" else ConvexOpt(config)
    return cast_tensors_to_dtype(problem.to_device(runtime_device), runtime_dtype)


def build_convex_hom_map(
    problem,
    *,
    runtime_device,
    runtime_dtype,
    x_origin,
    smooth=False,
    hom_p_norm=2,
    hom_map_explicit_gradient_rule="polynomial",
    hom_map_smooth_tie_tol=1e-7,
    hom_map_smooth_temperature=1e-4,
):
    return cast_tensors_to_dtype(
        GaugeMap(
            problem,
            p_norm=hom_p_norm,
            x_origin=x_origin,
            smooth=smooth,
            explicit_gradient_rule=hom_map_explicit_gradient_rule,
            smooth_tie_tol=hom_map_smooth_tie_tol,
            smooth_temperature=hom_map_smooth_temperature,
        ).to_device(runtime_device),
        runtime_dtype,
    )


def normalize_hom_origin_constraints(value):
    aliases = {
        "full": "full",
        "all": "full",
        "eq": "full",
        "equality": "full",
        "ineq": "ineq",
        "inequality": "ineq",
        "inequalities": "ineq",
    }
    key = str(value).lower()
    if key not in aliases:
        raise ValueError("hom_origin_constraints must be one of: 'full', 'ineq'.")
    return aliases[key]


def _origin_equality_violation(problem, x_origin):
    if x_origin is None or getattr(problem, "n_eq", 0) <= 0:
        return None
    x_tensor = torch.as_tensor(x_origin, dtype=problem.Q.dtype, device=problem.Q.device).view(1, -1)
    return float(problem.eq_constraint_x(x_tensor).abs().max().detach().cpu().item())


def _project_to_equalities(problem, x):
    if getattr(problem, "n_eq", 0) <= 0 or getattr(problem, "A_eq", None) is None:
        return x
    residual = x @ problem.A_eq.T - problem.b_eq
    gram_inv = torch.linalg.pinv(problem.A_eq @ problem.A_eq.T)
    return x - residual @ gram_inv @ problem.A_eq


def _approximate_chebyshev_origin(
    problem,
    *,
    include_equality=False,
    iterations=200,
    learning_rate=5e-2,
    temperature=1e-2,
):
    """Approximate an interior origin for GaugeMap without a CVXPY center solve."""

    device = getattr(problem, "device", torch.device("cpu"))
    dtype = getattr(problem.Q, "dtype", torch.float32)
    lower = problem.L.to(device=device, dtype=dtype).view(1, -1)
    upper = problem.U.to(device=device, dtype=dtype).view(1, -1)
    x = (0.5 * (lower + upper)).detach()
    if include_equality:
        x = _project_to_equalities(problem, x).clamp(lower, upper)
    x_var = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_var], lr=float(learning_rate))
    best_x = x_var.detach().clone()
    best_score = float("inf")
    temp = float(max(temperature, 1e-8))

    for _ in range(int(max(iterations, 1))):
        optimizer.zero_grad()
        ineq_residual = _call_problem_constraint(problem, x_var, include_equalities=False, clip=False)
        smooth_max_residual = temp * torch.logsumexp(ineq_residual / temp, dim=1).mean()
        loss = smooth_max_residual
        if include_equality and getattr(problem, "n_eq", 0) > 0:
            eq_residual = problem.eq_constraint_x(x_var)
            loss = loss + 100.0 * torch.mean(eq_residual * eq_residual)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x_var.clamp_(lower, upper)
            if include_equality:
                x_var.copy_(_project_to_equalities(problem, x_var).clamp(lower, upper))
            score = float(
                _call_problem_constraint(problem, x_var, include_equalities=False, clip=False)
                .max()
                .detach()
                .cpu()
                .item()
            )
            if include_equality and getattr(problem, "n_eq", 0) > 0:
                score += float(problem.eq_constraint_x(x_var).abs().max().detach().cpu().item())
            if score < best_score:
                best_score = score
                best_x = x_var.detach().clone()

    return best_x.view(-1).detach().cpu().numpy()


_CONVEX_REFERENCE_CACHE_VERSION = 1
_CONVEX_REFERENCE_CACHE_FILENAME = "convex_reference_context_cache.npy"


def _hash_reference_value(hasher, value):
    if value is None:
        hasher.update(b"<none>")
        return
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"array")
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(array.tobytes())
        return
    if isinstance(value, dict):
        hasher.update(b"dict")
        for key in sorted(value):
            hasher.update(str(key).encode("utf-8"))
            _hash_reference_value(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(f"seq:{len(value)}".encode("utf-8"))
        for item in value:
            _hash_reference_value(hasher, item)
        return
    hasher.update(repr(value).encode("utf-8"))


def _convex_reference_cache_key(
    problem,
    *,
    ip_mode,
    ip_eps,
    hom_origin_constraints,
    hom_origin,
    origin_include_equality,
):
    hasher = hashlib.sha256()
    hasher.update(f"version:{_CONVEX_REFERENCE_CACHE_VERSION}".encode("utf-8"))
    _hash_reference_value(hasher, getattr(problem, "prob_para", {}))
    _hash_reference_value(
        hasher,
        {
            "ip_mode": str(ip_mode),
            "ip_eps": float(ip_eps),
            "hom_origin_constraints": str(hom_origin_constraints),
            "hom_origin": None
            if hom_origin is None
            else np.asarray(hom_origin, dtype=float).reshape(-1),
            "origin_include_equality": bool(origin_include_equality),
        },
    )
    return hasher.hexdigest()


def _convex_reference_cache_path(output_dir):
    root = artifact_root(output_dir)
    if root is None:
        return None
    return root / _CONVEX_REFERENCE_CACHE_FILENAME


def _load_convex_reference_cache(cache_path, cache_key):
    if cache_path is None or not Path(cache_path).exists():
        return None, 0.0
    start_time = perf_counter()
    try:
        cache = np.load(cache_path, allow_pickle=True).item()
    except (OSError, ValueError, EOFError, AttributeError):
        return None, perf_counter() - start_time
    if not isinstance(cache, dict) or cache.get("version") != _CONVEX_REFERENCE_CACHE_VERSION:
        return None, perf_counter() - start_time
    entries = cache.get("entries", {})
    if not isinstance(entries, dict) or cache_key not in entries:
        return None, perf_counter() - start_time
    return copy.deepcopy(entries[cache_key]), perf_counter() - start_time


def _save_convex_reference_cache(cache_path, cache_key, payload):
    if cache_path is None:
        return
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {"version": _CONVEX_REFERENCE_CACHE_VERSION, "entries": {}}
    if cache_path.exists():
        try:
            loaded = np.load(cache_path, allow_pickle=True).item()
            if isinstance(loaded, dict) and loaded.get("version") == _CONVEX_REFERENCE_CACHE_VERSION:
                cache = loaded
                cache.setdefault("entries", {})
        except (OSError, ValueError, EOFError, AttributeError):
            pass
    cache["entries"][cache_key] = copy.deepcopy(payload)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, cache, allow_pickle=True)
    tmp_path.replace(cache_path)


def _reference_payload_from_values(
    *,
    x_opt,
    objective_opt,
    solver_violation,
    solver_time,
    x_origin,
    ip_solver_time,
    origin_method,
):
    payload = {"origin_method": origin_method}
    if x_opt is not None:
        payload.update(
            {
                "x_opt": np.asarray(x_opt, dtype=float).reshape(-1),
                "objective_opt": objective_opt,
                "solver_violation": solver_violation,
                "solver_time": solver_time,
            }
        )
    if x_origin is not None:
        payload.update(
            {
                "x_origin": np.asarray(x_origin, dtype=float).reshape(-1),
                "ip_solver_time": ip_solver_time,
            }
        )
    return payload


def prepare_convex_reference_context(
    problem,
    *,
    runtime_device,
    runtime_dtype,
    hom_p_norm=2,
    smooth=False,
    hom_map_explicit_gradient_rule="polynomial",
    hom_map_smooth_tie_tol=1e-7,
    hom_map_smooth_temperature=1e-4,
    need_opt=True,
    ip_mode="ip",
    ip_eps=1e-3,
    hom_origin_constraints="full",
    hom_origin=None,
    output_dir=None,
    cache_reference=True,
):
    hom_origin_constraints = normalize_hom_origin_constraints(hom_origin_constraints)
    origin_include_equality = hom_origin_constraints == "full"
    solver = ConvexSolver(problem.prob_para)
    cache_path = _convex_reference_cache_path(output_dir) if cache_reference else None
    cache_key = _convex_reference_cache_key(
        problem,
        ip_mode=ip_mode,
        ip_eps=ip_eps,
        hom_origin_constraints=hom_origin_constraints,
        hom_origin=hom_origin,
        origin_include_equality=origin_include_equality,
    )
    cached_payload, cache_load_time = _load_convex_reference_cache(cache_path, cache_key)
    cached_payload = cached_payload or {}
    cache_dirty = False
    reference_cache_hit = False
    cached_solver_time = None
    cached_ip_solver_time = None

    x_opt = obj_opt = solver_violation = solver_time = None
    if need_opt:
        if "x_opt" in cached_payload:
            x_opt = np.asarray(cached_payload["x_opt"], dtype=float).reshape(-1)
            obj_opt = cached_payload.get("objective_opt")
            solver_violation = cached_payload.get("solver_violation")
            cached_solver_time = cached_payload.get("solver_time")
            solver_time = 0.0
            reference_cache_hit = True
        else:
            solver_result = solve_exact_result(solver, "opt")
            x_opt = solver_result["solution"]
            solver_time = solver_result["runtime_total"]
            obj_opt = float(solver_result["objective"]) if solver_result["objective"] is not None else None
            solver_violation = float(solver_result["violation"]) if solver_result["violation"] is not None else None
            cached_solver_time = solver_time
            cache_dirty = True
    origin_method = str(ip_mode)
    if hom_origin is not None:
        x_origin = np.asarray(hom_origin, dtype=float).reshape(-1)
        ip_solver_time = 0.0
        origin_method = "provided"
    elif "x_origin" in cached_payload:
        x_origin = np.asarray(cached_payload["x_origin"], dtype=float).reshape(-1)
        origin_method = cached_payload.get("origin_method", origin_method)
        cached_ip_solver_time = cached_payload.get("ip_solver_time")
        ip_solver_time = 0.0
        reference_cache_hit = True
    elif ip_mode == "central_ip":
        ip_result = solve_exact_result(solver, "central_ip", equality=origin_include_equality)
        x_origin = ip_result["solution"]
        ip_solver_time = ip_result["runtime_total"]
        cached_ip_solver_time = ip_solver_time
        cache_dirty = True
    elif ip_mode == "geometric_central_ip":
        ip_result = solve_exact_result(solver, "geometric_central_ip", equality=origin_include_equality)
        x_origin = ip_result["solution"]
        ip_solver_time = ip_result["runtime_total"]
        cached_ip_solver_time = ip_solver_time
        cache_dirty = True
    elif str(ip_mode).lower() in {"chebyshev", "chebyshev_approx", "approx_chebyshev"}:
        start_time = perf_counter()
        x_origin = _approximate_chebyshev_origin(problem, include_equality=origin_include_equality)
        ip_solver_time = perf_counter() - start_time
        origin_method = "chebyshev_approx"
        cached_ip_solver_time = ip_solver_time
        cache_dirty = True
    else:
        ip_result = solve_exact_result(solver, "ip", eps=ip_eps, equality=origin_include_equality)
        x_origin = ip_result["solution"]
        ip_solver_time = ip_result["runtime_total"]
        cached_ip_solver_time = ip_solver_time
        cache_dirty = True
    if cache_dirty and cache_path is not None:
        next_payload = dict(cached_payload)
        next_payload.update(
            _reference_payload_from_values(
                x_opt=x_opt,
                objective_opt=obj_opt,
                solver_violation=solver_violation,
                solver_time=cached_solver_time if need_opt else cached_payload.get("solver_time"),
                x_origin=x_origin if hom_origin is None else cached_payload.get("x_origin"),
                ip_solver_time=cached_ip_solver_time
                if hom_origin is None
                else cached_payload.get("ip_solver_time"),
                origin_method=origin_method
                if hom_origin is None
                else cached_payload.get("origin_method", origin_method),
            )
        )
        _save_convex_reference_cache(cache_path, cache_key, next_payload)
    x_origin_eq_violation = _origin_equality_violation(problem, x_origin)
    hom_map = build_convex_hom_map(
        problem,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        x_origin=x_origin,
        smooth=smooth,
        hom_p_norm=hom_p_norm,
        hom_map_explicit_gradient_rule=hom_map_explicit_gradient_rule,
        hom_map_smooth_tie_tol=hom_map_smooth_tie_tol,
        hom_map_smooth_temperature=hom_map_smooth_temperature,
    )
    cache_path_value = str(cache_path) if cache_path is not None and Path(cache_path).exists() else None
    return {
        "x_opt": x_opt,
        "x_origin": x_origin,
        "objective_opt": obj_opt,
        "solver_violation": solver_violation,
        "solver_time": solver_time,
        "ip_solver_time": ip_solver_time,
        "origin_method": origin_method,
        "hom_origin_constraints": hom_origin_constraints,
        "x_origin_eq_violation": x_origin_eq_violation,
        "hom_map": hom_map,
        "reference_cache_hit": reference_cache_hit,
        "reference_cache_path": cache_path_value,
        "reference_cache_key": cache_key,
        "reference_cache_load_time": cache_load_time,
        "cached_solver_time": cached_solver_time,
        "cached_ip_solver_time": cached_ip_solver_time,
    }
