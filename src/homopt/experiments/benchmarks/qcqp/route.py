"""QCQP iterative-vs-learning route comparison benchmark."""

from __future__ import annotations

import inspect
from pathlib import Path

from homopt.records.artifacts import artifact_mapping, build_benchmark_payload, save_table_artifacts
from homopt.experiments.common.comparison import comparison_metric_source
from .reports import (
    QCQP_ROUTE_COMPARISON_FIELDS,
    build_qcqp_route_comparison_views,
    build_qcqp_route_rows,
)
from .training import _build_qcqp_inn_case_context, _sample_qcqp_inn_test_instances
from .inn_pgd import qcqp_inn_comparison
from .learning import qcqp_learning_benchmark


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


def _validate_qcqp_route_shared_params(iterative_cfg, learning_cfg):
    shared = dict(iterative_cfg)
    for key in ("seed", "n_var", "n_qua_cons", "n_linear_cons", "problem_config", "device", "dtype"):
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

    comparison_keys = set(inspect.signature(qcqp_inn_comparison).parameters)
    iterative_payload = qcqp_inn_comparison(
        output_dir=iterative_out,
        case_context=shared_context,
        instance_batch=shared_instances,
        **{key: value for key, value in iterative_cfg.items() if key in comparison_keys},
    )
    learning_payload = qcqp_learning_benchmark(
        output_dir=learning_out,
        context=learning_context,
        n_samples=len(shared_instances),
        **{key: value for key, value in learning_cfg.items() if key not in {"context", "n_samples"}},
    )

    learning_metrics = comparison_metric_source(learning_payload)
    rows = build_qcqp_route_rows(
        iterative_payload=iterative_payload,
        learning_results=learning_metrics.get("results", {}),
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

    learning_results = learning_metrics.get("results", {})
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


__all__ = ["qcqp_route_comparison"]
