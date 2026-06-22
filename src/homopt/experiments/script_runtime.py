"""Runtime helpers for directly runnable experiment scripts."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from homopt.experiments import default_script_output_dir, record_script_run
from homopt.experiments.common.naming import labeled_run_name, sanitize_label, scale_label


_CUDA_FALLBACK_WARNED = set()
SCRIPT_ONLY_PARAM_KEYS = {"result_label", "scale_label"}


def _warn_cuda_fallback(original, replacement, reason):
    key = (str(original), str(replacement), reason)
    if key in _CUDA_FALLBACK_WARNED:
        return
    _CUDA_FALLBACK_WARNED.add(key)
    print(
        f"[HomOPT scripts] Requested device={original!r} but {reason}; "
        f"using device={replacement!r}."
    )


def _device_with_cuda_fallback(value):
    if isinstance(value, torch.device):
        device_text = str(value)
    else:
        device_text = str(value).strip().lower() if value is not None else ""
    if not device_text:
        return value
    if device_text == "auto":
        if torch.cuda.is_available():
            return value
        _warn_cuda_fallback(value, "cpu", "CUDA is unavailable")
        return "cpu"
    if not device_text.startswith("cuda"):
        return value
    if not torch.cuda.is_available():
        _warn_cuda_fallback(value, "cpu", "CUDA is unavailable")
        return "cpu"
    if ":" in device_text:
        try:
            index = int(device_text.split(":", 1)[1])
        except ValueError:
            return value
        if index >= torch.cuda.device_count():
            _warn_cuda_fallback(value, "cpu", f"CUDA device index {index} is unavailable")
            return "cpu"
    return value


def _apply_runtime_device_fallback(value):
    if isinstance(value, dict):
        return {
            key: _device_with_cuda_fallback(val) if key == "device" else _apply_runtime_device_fallback(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_apply_runtime_device_fallback(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_apply_runtime_device_fallback(item) for item in value)
    return value


def _merge_params_raw(base, overrides=None):
    result = copy.deepcopy(base or {})
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_params_raw(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_params(base, overrides=None):
    """Deep-merge overrides into a copy of base params."""

    return _apply_runtime_device_fallback(_merge_params_raw(base, overrides or {}))


def merge_param_layers(*layers):
    """Deep-merge multiple config layers, then normalize runtime fields once."""

    result = {}
    for layer in layers:
        result = _merge_params_raw(result, layer or {})
    return _apply_runtime_device_fallback(result)


def apply_runtime_device_fallback(params):
    """Normalize script runtime devices after ad hoc param edits."""

    return _apply_runtime_device_fallback(copy.deepcopy(params or {}))


def convex_problem_scale_label(params):
    """Build a convex experiment label from the final problem dimensions."""

    pieces = [
        str((params or {}).get("problem_type", "socp")),
        f"n{int(params['n_var'])}",
    ]
    n_linear = int((params or {}).get("n_linear_cons", 0) or 0)
    n_soc = int((params or {}).get("n_soc_cons", 0) or 0)
    n_qua = int((params or {}).get("n_qua_cons", 0) or 0)
    n_eq = int((params or {}).get("n_lin_eq", 0) or 0)
    if n_linear:
        pieces.append(f"lin{n_linear}")
    if n_soc:
        pieces.append(f"soc{n_soc}")
    if n_qua:
        pieces.append(f"q{n_qua}")
    if n_eq:
        pieces.append(f"eq{n_eq}")
    return "_".join(pieces)


def script_family(script_file, *, scripts_root):
    """Return the experiment family from a script path under scripts/."""

    path = Path(script_file).resolve()
    try:
        relative = path.relative_to(Path(scripts_root).resolve())
    except ValueError as exc:
        raise ValueError(f"Script path is not under scripts/: {path}") from exc
    return relative.parts[0]


def script_output_dir(name, *, repo_root, output_root=None, family=None):
    root = Path(repo_root) / output_root if output_root is not None else Path(repo_root) / "results"
    if family is not None:
        root = root / str(family)
    return default_script_output_dir(name, output_root=root)


def result_subfolder_label(params):
    """Return the run-level subfolder label for script outputs."""

    explicit_label = (params or {}).get("result_label")
    if explicit_label:
        return sanitize_label(explicit_label, fallback="")
    label = scale_label(params, fallback="")
    if label:
        return label
    problem_type = (params or {}).get("problem_type")
    if problem_type:
        return sanitize_label(problem_type, fallback="")
    return ""


def benchmark_params(params):
    """Return params intended for the package benchmark call, excluding script metadata."""

    return {
        key: value
        for key, value in dict(params or {}).items()
        if key not in SCRIPT_ONLY_PARAM_KEYS
    }


def script_run_output_dir(name, *, repo_root, params=None, output_root=None, family=None):
    output_dir = script_output_dir(name, repo_root=repo_root, output_root=output_root, family=family)
    label = result_subfolder_label(params)
    return output_dir / label if label else output_dir


def labeled_child_output_dir(parent_output_dir, params=None):
    label = result_subfolder_label(params)
    parent = Path(parent_output_dir)
    return parent / label if label else parent


def run_script_experiment(name, params, run_fn, *, repo_root, output_dir=None, family=None):
    return record_script_run(
        name=name,
        params=params,
        run_fn=run_fn,
        output_dir=output_dir or script_run_output_dir(name, repo_root=repo_root, params=params, family=family),
    )


def _labeled_params(base_params, label, overrides=None, *, label_builder=None):
    params = merge_params(base_params, overrides or {})
    if label:
        params["scale_label"] = str(label)
    elif label_builder is not None:
        params["scale_label"] = label_builder(params)
    return params


def make_benchmark_entrypoint(
    name,
    benchmark,
    params,
    *,
    repo_root,
    output_dir=None,
    family=None,
    instances=None,
    label_builder=None,
):
    if instances is None:
        target_output_dir = output_dir or script_run_output_dir(
            name,
            repo_root=repo_root,
            params=params,
            family=family,
        )

        def run():
            return benchmark(output_dir=target_output_dir, **benchmark_params(params))

        def main():
            return run_script_experiment(
                name=name,
                params=params,
                run_fn=run,
                repo_root=repo_root,
                output_dir=target_output_dir,
                family=family,
            )

        return run, main, target_output_dir

    base_params = params
    target_output_dir = output_dir or script_output_dir(name, repo_root=repo_root, family=family)

    def run(label=None, overrides=None):
        return run_labeled_benchmark_instance(
            name,
            benchmark,
            base_params,
            label,
            overrides,
            repo_root=repo_root,
            label_builder=label_builder,
            family=family,
            parent_output_dir=target_output_dir,
        )

    def main(selected_instances=None):
        return run_labeled_benchmark_instances(
            name,
            benchmark,
            base_params,
            instances if selected_instances is None else selected_instances,
            repo_root=repo_root,
            label_builder=label_builder,
            family=family,
            parent_output_dir=target_output_dir,
        )

    return run, main, target_output_dir


def run_labeled_benchmark_instance(
    name,
    benchmark,
    base_params,
    label,
    overrides=None,
    *,
    repo_root,
    label_builder=None,
    family=None,
    parent_output_dir=None,
):
    params = _labeled_params(base_params, label, overrides, label_builder=label_builder)
    run_name = labeled_run_name(name, params)
    output_dir = (
        labeled_child_output_dir(parent_output_dir, params=params)
        if parent_output_dir is not None
        else script_run_output_dir(name, repo_root=repo_root, params=params, family=family)
    )

    def _run():
        return benchmark(output_dir=output_dir, **benchmark_params(params))

    return run_script_experiment(
        name=run_name,
        params=params,
        run_fn=_run,
        repo_root=repo_root,
        output_dir=output_dir,
        family=family,
    )


def run_labeled_benchmark_instances(
    name,
    benchmark,
    base_params,
    instances,
    *,
    repo_root,
    label_builder=None,
    family=None,
    parent_output_dir=None,
):
    if not instances:
        return [
            run_labeled_benchmark_instance(
                name,
                benchmark,
                base_params,
                None,
                None,
                repo_root=repo_root,
                label_builder=label_builder,
                family=family,
                parent_output_dir=parent_output_dir,
            )
        ]
    results = []
    for index, instance in enumerate(instances):
        if not isinstance(instance, (tuple, list)) or len(instance) != 2:
            raise ValueError(
                "Each benchmark instance must be a (scale_label, overrides) pair; "
                f"entry {index} is {instance!r}."
            )
        label, overrides = instance
        display_params = merge_params(base_params, overrides or {})
        display_label = label or (label_builder(display_params) if label_builder is not None else "unlabeled")
        print(f"\n=== Running {display_label} ===")
        results.append(
            run_labeled_benchmark_instance(
                name,
                benchmark,
                base_params,
                label,
                overrides,
                repo_root=repo_root,
                label_builder=label_builder,
                family=family,
                parent_output_dir=parent_output_dir,
            )
        )
    return results


__all__ = [
    "SCRIPT_ONLY_PARAM_KEYS",
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
]
