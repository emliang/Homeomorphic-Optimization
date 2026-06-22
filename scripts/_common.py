"""Bootstrap and re-export shared helpers for script experiment entrypoints."""

from pathlib import Path
import os
import sys


def _find_repo_root(start):
    for path in (start, *start.parents):
        if (path / "src" / "homopt").exists():
            return path
    raise RuntimeError(f"Could not locate HomOPT repo root from {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
MPLCONFIGDIR = REPO_ROOT / "results" / "_cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from homopt.experiments.script_runtime import (  # noqa: E402
    SCRIPT_ONLY_PARAM_KEYS,
    apply_runtime_device_fallback,
    benchmark_params,
    convex_problem_scale_label,
    labeled_child_output_dir,
    make_benchmark_entrypoint as _make_benchmark_entrypoint,
    merge_param_layers,
    merge_params,
    result_subfolder_label,
    run_labeled_benchmark_instance as _run_labeled_benchmark_instance,
    run_labeled_benchmark_instances as _run_labeled_benchmark_instances,
    run_script_experiment as _run_script_experiment,
    script_family as _script_family,
    script_output_dir as _script_output_dir,
    script_run_output_dir as _script_run_output_dir,
)
from homopt.experiments.common.naming import sanitize_label, scale_label  # noqa: E402


def script_family(script_file):
    return _script_family(script_file, scripts_root=SCRIPTS_ROOT)


def script_output_dir(name, output_root=None, family=None):
    return _script_output_dir(name, repo_root=REPO_ROOT, output_root=output_root, family=family)


def script_run_output_dir(name, params=None, output_root=None, family=None):
    return _script_run_output_dir(
        name,
        repo_root=REPO_ROOT,
        params=params,
        output_root=output_root,
        family=family,
    )


def run_script_experiment(name, params, run_fn, output_dir=None, family=None):
    return _run_script_experiment(
        name,
        params,
        run_fn,
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        family=family,
    )


def make_benchmark_entrypoint(
    name,
    benchmark,
    params,
    *,
    output_dir=None,
    family=None,
    instances=None,
    label_builder=None,
):
    return _make_benchmark_entrypoint(
        name,
        benchmark,
        params,
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        family=family,
        instances=instances,
        label_builder=label_builder,
    )


def run_labeled_benchmark_instance(
    name,
    benchmark,
    base_params,
    label,
    overrides=None,
    *,
    label_builder=None,
    family=None,
    parent_output_dir=None,
):
    return _run_labeled_benchmark_instance(
        name,
        benchmark,
        base_params,
        label,
        overrides,
        repo_root=REPO_ROOT,
        label_builder=label_builder,
        family=family,
        parent_output_dir=parent_output_dir,
    )


def run_labeled_benchmark_instances(
    name,
    benchmark,
    base_params,
    instances,
    *,
    label_builder=None,
    family=None,
    parent_output_dir=None,
):
    return _run_labeled_benchmark_instances(
        name,
        benchmark,
        base_params,
        instances,
        repo_root=REPO_ROOT,
        label_builder=label_builder,
        family=family,
        parent_output_dir=parent_output_dir,
    )


__all__ = [
    "MPLCONFIGDIR",
    "REPO_ROOT",
    "SCRIPT_ONLY_PARAM_KEYS",
    "SCRIPTS_ROOT",
    "SRC_ROOT",
    "apply_runtime_device_fallback",
    "benchmark_params",
    "convex_problem_scale_label",
    "labeled_child_output_dir",
    "make_benchmark_entrypoint",
    "merge_param_layers",
    "merge_params",
    "result_subfolder_label",
    "run_labeled_benchmark_instance",
    "run_labeled_benchmark_instances",
    "run_script_experiment",
    "script_family",
    "script_output_dir",
    "script_run_output_dir",
    "sanitize_label",
    "scale_label",
]
