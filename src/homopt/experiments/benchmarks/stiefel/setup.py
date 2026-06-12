"""Stiefel benchmark problem construction and reference cache helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import torch

from homopt.experiments.common.artifacts import artifact_root
from homopt.experiments.common.config import normalize_algorithm_config_group
from homopt.experiments.common.runtime import resolve_runtime
from homopt.mappings import GaugeMap
from homopt.problems import StiefelProblem, create_constrained_pca_stiefel_config
from homopt.utils import cast_tensors_to_dtype


STIEFEL_REFERENCE_CACHE_FILENAME = "stiefel_reference_context_cache.npy"
STIEFEL_REFERENCE_CACHE_VERSION = 1

def load_stiefel_data(problem_params):
    source = str(problem_params.get("data_source", "synthetic")).lower()
    if source == "synthetic":
        return None
    data_home = str(Path(problem_params.get("data_home", "data/sklearn_data")).resolve())
    from sklearn.datasets import (
        fetch_lfw_people,
        fetch_olivetti_faces,
        fetch_openml,
        load_breast_cancer,
        load_digits,
        load_iris,
        load_wine,
    )

    loaders = {
        "sklearn_digits": load_digits,
        "sklearn_wine": load_wine,
        "sklearn_breast_cancer": load_breast_cancer,
        "sklearn_iris": load_iris,
    }
    if source in loaders:
        data = loaders[source]().data.astype(np.float32)
    elif source == "sklearn_olivetti_faces":
        data = fetch_olivetti_faces(
            data_home=data_home,
            shuffle=bool(problem_params.get("shuffle", False)),
            random_state=int(problem_params.get("random_state", 0)),
            download_if_missing=bool(problem_params.get("download_if_missing", False)),
        ).data.astype(np.float32)
    elif source == "sklearn_lfw_people":
        data = fetch_lfw_people(
            data_home=data_home,
            resize=float(problem_params.get("lfw_resize", 0.25)),
            min_faces_per_person=int(problem_params.get("min_faces_per_person", 0)),
            color=bool(problem_params.get("color", False)),
            download_if_missing=bool(problem_params.get("download_if_missing", False)),
        ).data.astype(np.float32)
    elif source in {"openml_mnist", "openml_fashion_mnist"}:
        if not bool(problem_params.get("download_if_missing", False)):
            raise ValueError(
                f"{source!r} may require network access; set problem_config.download_if_missing=True explicitly."
            )
        data = fetch_openml(
            "mnist_784" if source == "openml_mnist" else "Fashion-MNIST",
            version=int(problem_params.get("openml_version", 1)),
            as_frame=False,
            parser="auto",
            data_home=data_home,
        ).data.astype(np.float32)
    else:
        raise ValueError(
            "problem.data_source must be one of: 'synthetic', 'sklearn_digits', "
            "'sklearn_wine', 'sklearn_breast_cancer', 'sklearn_iris', "
            "'sklearn_olivetti_faces', 'sklearn_lfw_people', 'openml_mnist', "
            "'openml_fashion_mnist'."
        )
    max_samples = problem_params.get("max_samples")
    if max_samples is not None:
        data = data[: int(max_samples)]
    if bool(problem_params.get("standardize", True)):
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        data = (data - mean) / np.maximum(std, 1e-6)
    return data


def pca_covariance_from_data(data):
    data_tensor = torch.as_tensor(data, dtype=torch.float32)
    centered = data_tensor - data_tensor.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(int(centered.shape[0]) - 1, 1)


def resolve_stiefel_restricted_rows(problem_params, covariance_tensor):
    restricted_rows = problem_params.get("restricted_rows")
    if not isinstance(restricted_rows, str):
        return restricted_rows
    mode = restricted_rows.lower()
    if mode not in {"top_loading", "top_loadings", "pca_top_loading"}:
        raise ValueError(
            "problem.restricted_rows string mode must be one of: "
            "'top_loading', 'top_loadings', or 'pca_top_loading'."
        )
    if covariance_tensor is None:
        raise ValueError("top-loading restricted rows require a PCA covariance matrix.")
    row_count = int(problem_params.get("restricted_row_count", 0))
    if row_count <= 0:
        raise ValueError("problem.restricted_row_count must be positive for top-loading restricted rows.")
    row_count = min(row_count, int(covariance_tensor.shape[0]))
    n_cols = int(problem_params["n_cols"])
    eigvals, eigvecs = torch.linalg.eigh(covariance_tensor)
    top_indices = torch.argsort(eigvals, descending=True)[:n_cols]
    top_vectors = eigvecs[:, top_indices]
    row_energy = torch.sum(top_vectors * top_vectors, dim=1)
    return sorted(int(index) for index in torch.topk(row_energy, row_count).indices.detach().cpu().tolist())


def build_stiefel_config(problem_params):
    data = load_stiefel_data(problem_params)
    if data is not None:
        covariance_tensor = pca_covariance_from_data(data)
        return create_constrained_pca_stiefel_config(
            covariance=covariance_tensor,
            n_cols=int(problem_params["n_cols"]),
            restricted_rows=resolve_stiefel_restricted_rows(problem_params, covariance_tensor),
            group_budget=problem_params.get("group_budget"),
            L=float(problem_params["box_lower"]),
            U=float(problem_params["box_upper"]),
            norm_radius=problem_params.get("norm_radius"),
            objective_normalization=problem_params.get("objective_normalization"),
            objective_scale=problem_params.get("objective_scale"),
        )

    n_rows = int(problem_params["n_rows"])
    n_cols = int(problem_params["n_cols"])
    spectrum = problem_params.get("spectrum")
    if spectrum is None:
        spectrum = np.linspace(float(n_rows), 1.0, n_rows).tolist()
    if len(spectrum) != n_rows:
        raise ValueError("problem.spectrum length must equal problem.n_rows.")

    config = {
        "A_obj": torch.diag(torch.as_tensor(spectrum, dtype=torch.float32)),
        "n_cols": n_cols,
        "L": float(problem_params["box_lower"]),
        "U": float(problem_params["box_upper"]),
    }
    if problem_params.get("norm_radius") is not None:
        config["norm_radius"] = float(problem_params["norm_radius"])

    restricted_rows = problem_params.get("restricted_rows")
    group_budget = problem_params.get("group_budget")
    if restricted_rows is not None or group_budget is not None:
        if restricted_rows is None or group_budget is None:
            raise ValueError("problem.restricted_rows and problem.group_budget must be provided together.")
        config["group_norm_constraints"] = [
            {
                "name": "restricted_rows",
                "rows": [int(row) for row in restricted_rows],
                "budget": float(group_budget),
            }
        ]
    return config


def move_stiefel_problem(problem, runtime_device, runtime_dtype):
    return cast_tensors_to_dtype(problem.to_device(runtime_device), runtime_dtype)


def build_stiefel_problem(params):
    runtime_device, runtime_dtype = resolve_runtime(params.get("device"), params.get("dtype"))
    problem = move_stiefel_problem(
        StiefelProblem(build_stiefel_config(params["problem_config"])),
        runtime_device,
        runtime_dtype,
    )
    return problem, runtime_device, runtime_dtype


def build_stiefel_hom_map(problem, params, runtime_dtype):
    mapping_params = params["mapping"]
    origin_mode = str(mapping_params.get("x_origin", "zero")).lower()
    if origin_mode != "zero":
        raise ValueError("Only mapping.x_origin='zero' is currently supported.")
    x_origin = torch.zeros(problem.nvar, device=problem.device, dtype=runtime_dtype)
    hom_map = GaugeMap(
        problem,
        p_norm=mapping_params.get("hom_p_norm", 2),
        x_origin=x_origin,
        smooth=bool(mapping_params.get("smooth", False)),
        explicit_gradient_rule=mapping_params.get("hom_map_explicit_gradient_rule", "polynomial"),
        smooth_tie_tol=mapping_params.get("hom_map_smooth_tie_tol", 1e-7),
        smooth_temperature=mapping_params.get("hom_map_smooth_temperature", 1e-4),
    )
    return cast_tensors_to_dtype(hom_map.to_device(problem.device), runtime_dtype)


def build_stiefel_initial_point(problem, params, runtime_dtype):
    init_config = copy.deepcopy(params.get("initialization") or {})
    mode = str(init_config.get("mode", "random_ball")).lower()
    if mode == "zero":
        return torch.zeros(1, problem.nvar, device=problem.device, dtype=runtime_dtype)
    if mode != "random_ball":
        raise ValueError("initialization.mode must be 'random_ball' or 'zero'.")
    scale = float(init_config.get("scale", 0.1))
    if scale <= 0:
        raise ValueError("initialization.scale must be positive.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(params["seed"]))
    point = torch.randn(1, problem.nvar, generator=generator, dtype=runtime_dtype)
    point = point.to(problem.device)
    norm = point.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return scale * point / norm


def stiefel_reference_solver_params(params):
    algorithm_config = normalize_algorithm_config_group(
        algorithm_config=params.get("algorithm_config"),
    )
    solver_params = copy.deepcopy(algorithm_config.get("StiefelIPOPT", {}))
    solver_params.setdefault("seed", int(params["seed"]))
    return solver_params


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, range)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.generic):
        return value.item()
    return value


def stiefel_reference_cache_key(params):
    payload = {
        "problem_config": params.get("problem_config"),
        "reference_solver": stiefel_reference_solver_params(params),
        "seed": params.get("seed"),
        "initialization": params.get("initialization"),
    }
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stiefel_reference_cache_path(output_dir, cache_reference):
    if output_dir is None or not cache_reference:
        return None
    return artifact_root(output_dir) / STIEFEL_REFERENCE_CACHE_FILENAME


def load_stiefel_reference_cache(cache_path, cache_key):
    if cache_path is None or not Path(cache_path).exists():
        return None, 0.0
    start = perf_counter()
    try:
        cache = np.load(cache_path, allow_pickle=True).item()
    except (OSError, ValueError, EOFError, AttributeError) as exc:
        warnings.warn(f"Ignoring unreadable Stiefel reference cache {cache_path}: {exc}", RuntimeWarning)
        return None, perf_counter() - start
    if not isinstance(cache, dict) or cache.get("version") != STIEFEL_REFERENCE_CACHE_VERSION:
        return None, perf_counter() - start
    payload = dict(cache.get("entries", {})).get(cache_key)
    return copy.deepcopy(payload), perf_counter() - start


def save_stiefel_reference_cache(cache_path, cache_key, payload):
    if cache_path is None or payload is None:
        return
    cache_path = Path(cache_path)
    cache = {"version": STIEFEL_REFERENCE_CACHE_VERSION, "entries": {}}
    if cache_path.exists():
        try:
            loaded = np.load(cache_path, allow_pickle=True).item()
            if isinstance(loaded, dict) and loaded.get("version") == STIEFEL_REFERENCE_CACHE_VERSION:
                cache = loaded
        except (OSError, ValueError, EOFError, AttributeError) as exc:
            warnings.warn(f"Overwriting unreadable Stiefel reference cache {cache_path}: {exc}", RuntimeWarning)
            cache = {"version": STIEFEL_REFERENCE_CACHE_VERSION, "entries": {}}
    cache.setdefault("entries", {})[cache_key] = copy.deepcopy(payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp.npy")
    np.save(tmp_path, cache, allow_pickle=True)
    tmp_path.replace(cache_path)
