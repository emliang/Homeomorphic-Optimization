"""AC-OPF predictor plus post-processing benchmarks."""

from __future__ import annotations

from homopt.experiments.common.parametric_learning import ParametricLearningSpec, run_parametric_learning_benchmark
from homopt.learning import DiffProjectionRefiner, IdentityRefiner
from homopt.problems import ParametricACOPFProblem


ACOPF_PARAMETRIC_LEARNING_BASELINES = (
    "predict_only",
    "predict_diff_projection",
)


def _build_acopf_parametric_problem(
    *,
    seed,
    dataset_path=None,
    case=57,
    dataset_samples=10000,
    load_std=0.05,
    pglib_case_name=None,
    pglib_data_dir=None,
    download_if_missing=True,
    branch_constraints=True,
    enforce_slack_angle=True,
    problem_config=None,
    device,
    dtype,
):
    config = dict(problem_config or {})
    return ParametricACOPFProblem(
        dataset_path=dataset_path,
        case=int(case) if dataset_path is None else None,
        dataset_samples=int(dataset_samples),
        load_std=float(load_std),
        pglib_case_name=pglib_case_name,
        pglib_data_dir=pglib_data_dir,
        download_if_missing=download_if_missing,
        seed=seed,
        branch_constraints=branch_constraints,
        enforce_slack_angle=enforce_slack_angle,
        device=device,
        dtype=dtype,
        **config,
    )


def _acopf_parametric_metadata(
    problem,
    *,
    seed,
    dataset_path=None,
    case=57,
    dataset_samples=10000,
    load_std=0.05,
    pglib_case_name=None,
    pglib_data_dir=None,
    download_if_missing=True,
    branch_constraints=True,
    enforce_slack_angle=True,
    **kwargs,
):
    del kwargs
    return {
        "problem_family": "acopf_parametric",
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "case_source": "dataset_path" if dataset_path is not None else "pglib-opf",
        "case": int(case),
        "dataset_samples": int(dataset_samples),
        "load_std": float(load_std),
        "pglib_case_name": pglib_case_name,
        "pglib_data_dir": None if pglib_data_dir is None else str(pglib_data_dir),
        "download_if_missing": bool(download_if_missing),
        "prediction_space": str(getattr(problem, "prediction_space", "full")),
        "n_var": int(problem.nvar),
        "partial_dim": int(getattr(problem, "partial_dim", problem.nvar)),
        "n_input": int(problem.xdim),
        "n_eq": int(problem.neq),
        "n_ineq": int(problem.nineq),
        "nb": int(problem.nb),
        "ng": int(problem.ng),
        "nl": int(problem.nl),
        "branch_constraints": bool(branch_constraints),
        "enforce_slack_angle": bool(enforce_slack_angle),
        "seed": int(seed),
    }


def _make_acopf_parametric_refiner(baseline, projection_config):
    if baseline == "predict_only":
        return IdentityRefiner()
    if baseline == "predict_diff_projection":
        cfg = dict(projection_config or {})
        return DiffProjectionRefiner(
            steps=int(cfg.get("proj_max_steps", cfg.get("steps", 30))),
            lr=float(cfg.get("corr_lr", cfg.get("lr", 1e-4))),
            momentum=float(cfg.get("corr_momentum", cfg.get("momentum", 0.5))),
            tol=float(cfg.get("proj_eps", cfg.get("tol", 1e-6))),
        )
    raise ValueError(f"Unsupported AC-OPF parametric baseline: {baseline}")


ACOPF_PARAMETRIC_LEARNING_SPEC = ParametricLearningSpec(
    family="acopf-parametric",
    artifact_prefix="acopf_parametric_learning",
    supported_baselines=ACOPF_PARAMETRIC_LEARNING_BASELINES,
    build_problem=_build_acopf_parametric_problem,
    make_refiner=_make_acopf_parametric_refiner,
    metadata=_acopf_parametric_metadata,
)


def acopf_parametric_learning_benchmark(
    seed=2026,
    case=57,
    dataset_samples=10000,
    dataset_path=None,
    load_std=0.05,
    pglib_case_name=None,
    pglib_data_dir=None,
    download_if_missing=True,
    n_samples=8,
    predictor_type="constant",
    prediction_value=0.0,
    baselines=None,
    problem_config=None,
    branch_constraints=True,
    enforce_slack_angle=True,
    predictor_model_config=None,
    predictor_train_config=None,
    projection_config=None,
    retrain=False,
    output_dir=None,
    device=None,
    dtype=None,
):
    """Run predictor/postprocess baselines for dataset-backed AC-OPF."""

    return run_parametric_learning_benchmark(
        spec=ACOPF_PARAMETRIC_LEARNING_SPEC,
        seed=seed,
        n_samples=n_samples,
        predictor_type=predictor_type,
        prediction_value=prediction_value,
        baselines=baselines,
        predictor_model_config=predictor_model_config,
        predictor_train_config=predictor_train_config,
        projection_config=projection_config,
        retrain=retrain,
        output_dir=output_dir,
        device=device,
        dtype=dtype,
        problem_kwargs={
            "case": case,
            "dataset_samples": dataset_samples,
            "dataset_path": dataset_path,
            "load_std": load_std,
            "pglib_case_name": pglib_case_name,
            "pglib_data_dir": pglib_data_dir,
            "download_if_missing": download_if_missing,
            "branch_constraints": branch_constraints,
            "enforce_slack_angle": enforce_slack_angle,
            "problem_config": problem_config,
        },
    )


__all__ = [
    "ACOPF_PARAMETRIC_LEARNING_BASELINES",
    "ACOPF_PARAMETRIC_LEARNING_SPEC",
    "acopf_parametric_learning_benchmark",
]
