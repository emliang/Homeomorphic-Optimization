"""Artifact and benchmark payload helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from homopt.records.comparison import build_single_instance_comparison_views
from homopt.utils.io import ensure_dir


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
        "metrics": dict(metrics),
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
COMPARISON_IDENTITY_VERSION = 1


def _identity_json_value(value):
    """Return a deterministic, JSON-safe representation for identity hashing.

    Comparison identities describe the shared experimental instance, not the
    potentially large records saved for each method.  Arrays and tensors are
    represented by shape, dtype, and content digest so explicit problem data
    remains part of the identity without bloating ``manifest.json``.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return {"__float__": "nan"}
        if np.isposinf(value):
            return {"__float__": "inf"}
        if np.isneginf(value):
            return {"__float__": "-inf"}
        return value
    if isinstance(value, np.generic):
        return _identity_json_value(value.item())
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, (torch.dtype, torch.device)):
        return {"__torch_type__": str(value)}
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest()}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        }
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            }
        }
    if isinstance(value, dict):
        return {
            str(key): _identity_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, range)):
        return [_identity_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_identity_json_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    raise TypeError(
        "Comparison run identities must contain deterministic configuration values; "
        f"got unsupported value of type {type(value).__name__}."
    )


def build_comparison_run_identity(payload):
    """Build the immutable identity stored with incremental comparison records."""

    canonical_payload = _identity_json_value(payload)
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return {
        "version": COMPARISON_IDENTITY_VERSION,
        "fingerprint": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "payload": canonical_payload,
    }


def _require_matching_comparison_run_identity(manifest, identity, path):
    stored_identity = manifest.get("run_identity")
    if not isinstance(stored_identity, dict):
        raise ValueError(
            f"Existing comparison artifacts at {path} lack a run identity. "
            "Rerun the comparison into a fresh output directory before adding methods."
        )
    if stored_identity.get("version") != COMPARISON_IDENTITY_VERSION:
        raise ValueError(
            f"Unsupported comparison run identity at {path}: "
            f"version={stored_identity.get('version')!r}; "
            f"expected {COMPARISON_IDENTITY_VERSION}."
        )
    if stored_identity.get("fingerprint") != identity["fingerprint"]:
        raise ValueError(
            f"Comparison artifact run identity mismatch at {path}. Existing records belong to "
            "a different base experiment; use a fresh output directory instead of merging them."
        )


def _require_current_comparison_manifest(manifest, path):
    version = manifest.get("version")
    if version != COMPARISON_STORAGE_VERSION:
        raise ValueError(
            f"Unsupported comparison artifact schema at {path}: version={version!r}; "
            f"expected {COMPARISON_STORAGE_VERSION}. Rerun the experiment to regenerate artifacts."
        )
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


def ordered_algorithm_subset(available_algorithms, preferred_order=None):
    """Return available algorithms, optionally front-loaded by a preferred plot order."""

    available = list(available_algorithms or [])
    if preferred_order is None:
        return available
    preferred = [algorithm for algorithm in preferred_order if algorithm in available]
    preferred.extend(algorithm for algorithm in available if algorithm not in preferred)
    return preferred


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
    _require_current_comparison_manifest(manifest, paths["manifest"])
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


def load_visualize_only_comparison_artifacts(
    output_dir,
    *,
    algorithms=None,
    plot_algorithm_order=None,
):
    """Load existing comparison state for a visualize-only benchmark pass."""

    records, summaries, previous_result, manifest = load_incremental_comparison_artifacts(
        output_dir,
        algorithms=algorithms,
    )
    stored_algorithms = list(records.keys()) or list(manifest.get("algorithms") or algorithms or [])
    return {
        "records": records,
        "summaries": summaries,
        "previous_result": previous_result,
        "manifest": manifest,
        "artifacts": dict(previous_result.get("artifacts", {}) or {}),
        "stored_algorithms": stored_algorithms,
        "plot_algorithms": ordered_algorithm_subset(stored_algorithms, plot_algorithm_order),
        "metrics": dict(previous_result.get("metrics", {}) or {}),
    }


def save_incremental_comparison_artifacts(
    output_dir,
    *,
    records,
    summaries,
    algorithm_order=None,
    manifest_metadata=None,
    run_identity=None,
):
    """Save comparison records per algorithm and keep a lightweight manifest/summary.

    ``run_identity`` must capture the shared instance and comparison settings.
    It deliberately excludes the selected method list so compatible partial
    method runs can be added later, while preventing records from distinct
    instances/configurations from being merged into one result directory.
    """

    root_paths = comparison_storage_paths(output_dir)
    if root_paths is None:
        comparison_views = build_single_instance_comparison_views(
            summaries,
            route="iterative",
            runtime_key="total_iter_time",
        )
        return {}, comparison_views, dict(records), dict(summaries)

    old_manifest = _read_json_or_empty(root_paths["manifest"])
    if old_manifest:
        _require_current_comparison_manifest(old_manifest, root_paths["manifest"])
    if run_identity is None:
        raise ValueError(
            "save_incremental_comparison_artifacts requires run_identity when output_dir is set. "
            "Pass the shared problem and comparison configuration used by every method."
        )
    identity = build_comparison_run_identity(run_identity)
    if old_manifest:
        _require_matching_comparison_run_identity(old_manifest, identity, root_paths["manifest"])
    old_summaries = _read_json_or_empty(root_paths["summary"])
    old_record_refs = dict(old_manifest.get("records") or {})
    records_dir = root_paths["records_dir"]
    records_dir.mkdir(parents=True, exist_ok=True)

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
        "run_identity": identity,
        "algorithms": stored_algorithms,
        "records": {name: record_refs[name] for name in stored_algorithms},
    }
    metadata = old_manifest.get("metadata")
    if manifest_metadata is not None:
        metadata = _slim_json_payload(manifest_metadata)
    if metadata is not None:
        manifest["metadata"] = metadata
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


__all__ = [
    "artifact_mapping",
    "artifact_ref",
    "artifact_root",
    "build_comparison_run_identity",
    "build_benchmark_payload",
    "build_visualize_only_benchmark_payload",
    "comparison_storage_paths",
    "load_incremental_comparison_artifacts",
    "load_visualize_only_comparison_artifacts",
    "ordered_algorithm_subset",
    "save_incremental_comparison_artifacts",
    "save_json",
    "save_numpy",
    "save_table_artifacts",
    "save_text",
    "slim_comparison_summary",
    "write_csv",
]
