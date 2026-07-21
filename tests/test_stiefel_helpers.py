import numpy as np
import pytest

from homopt.experiments.benchmarks.stiefel.setup import (
    build_stiefel_config,
    build_stiefel_initial_point,
    build_stiefel_problem,
    load_stiefel_data,
    load_stiefel_reference_cache,
    save_stiefel_reference_cache,
    stiefel_reference_cache_key,
)


def _synthetic_params():
    return {
        "device": "cpu",
        "dtype": "float32",
        "seed": 123,
        "problem_config": {
            "data_source": "synthetic",
            "n_rows": 5,
            "n_cols": 2,
            "box_lower": -1.0,
            "box_upper": 1.0,
            "restricted_rows": [0, 1],
            "group_budget": 1.0,
        },
        "initialization": {"mode": "random_ball", "scale": 0.25},
        "algorithm_config": {
            "StiefelIPOPT": {
                "solver": "ipopt",
                "solver_options": {"tol": 1e-6},
            }
        },
    }


def test_build_stiefel_config_for_synthetic_problem():
    config = build_stiefel_config(_synthetic_params()["problem_config"])

    assert config["n_cols"] == 2
    assert config["A_obj"].shape == (5, 5)
    assert config["group_norm_constraints"] == [
        {"name": "restricted_rows", "rows": [0, 1], "budget": 1.0}
    ]


def test_build_stiefel_problem_and_initial_point_are_deterministic():
    params = _synthetic_params()
    problem, _runtime_device, runtime_dtype = build_stiefel_problem(params)

    first = build_stiefel_initial_point(problem, params, runtime_dtype).detach().cpu().numpy()
    second = build_stiefel_initial_point(problem, params, runtime_dtype).detach().cpu().numpy()

    assert problem.nvar == 10
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(np.linalg.norm(first), 0.25, rtol=1e-5)


def test_openml_stiefel_data_requires_explicit_download():
    with pytest.raises(ValueError, match="download_if_missing=True"):
        load_stiefel_data({"data_source": "openml_mnist", "download_if_missing": False})


def test_stiefel_reference_cache_roundtrip(tmp_path):
    params = _synthetic_params()
    cache_key = stiefel_reference_cache_key(params)
    cache_path = tmp_path / "reference.npy"

    save_stiefel_reference_cache(cache_path, cache_key, {"result": {"objective": -1.0}})
    payload, load_time = load_stiefel_reference_cache(cache_path, cache_key)

    assert payload == {"result": {"objective": -1.0}}
    assert load_time >= 0.0
    assert not list(tmp_path.glob("*.tmp*"))
