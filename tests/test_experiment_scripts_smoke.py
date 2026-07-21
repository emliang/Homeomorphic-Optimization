from pathlib import Path
import importlib
import inspect
import json
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _common import benchmark_params  # noqa: E402


EXPERIMENT_SCRIPTS = (
    ("scripts.hom_alm.run_convex_eq_2d", "hom_alm", "convex_eq_2d", False),
    ("scripts.hom_alm.run_convex_eq_compare", "hom_alm", "convex_eq_compare", True),
    ("scripts.hom_alm.run_stiefel_eq_compare", "hom_alm", "stiefel_eq_compare", True),
    ("scripts.hom_pgd.run_adversarial_attack", "hom_pgd", "adversarial_attack", False),
    ("scripts.hom_pgd.run_convex_ineq_compare", "hom_pgd", "convex_ineq_compare", True),
    ("scripts.hom_pgd.run_maxcut_sdp_compare", "hom_pgd", "maxcut_sdp_compare", False),
    ("scripts.hom_pgd.run_poly_star_compare", "hom_pgd", "poly_star_compare", False),
    ("scripts.inn_pgd.run_jcc_opf_compare", "inn_pgd", "jcc_opf_compare", False),
    ("scripts.inn_pgd.run_qcqp_inn_ablation", "inn_pgd", "qcqp_inn_ablation", False),
    ("scripts.inn_pgd.run_qcqp_inn_compare", "inn_pgd", "qcqp_inn_compare", False),
    ("scripts.inn_pgd.run_qcqp_inn_toy", "inn_pgd", "qcqp_inn_toy", False),
    (
        "scripts.learning_postprocess.run_acopf_predictor_postprocess",
        "learning_postprocess",
        "acopf_predictor_postprocess",
        False,
    ),
    (
        "scripts.learning_postprocess.run_convex_parametric_predictor_postprocess",
        "learning_postprocess",
        "convex_parametric_predictor_postprocess",
        False,
    ),
)
EXPERIMENT_SCRIPT_MODULES = {case[0] for case in EXPERIMENT_SCRIPTS}


def _import_module(module_name):
    return importlib.import_module(module_name)


def test_every_experiment_script_has_smoke_case():
    script_files = sorted(
        path
        for folder in ("hom_alm", "hom_pgd", "inn_pgd", "learning_postprocess")
        for path in (SCRIPTS_ROOT / folder).glob("run_*.py")
    )
    discovered = {
        ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        for path in script_files
    }

    assert discovered == EXPERIMENT_SCRIPT_MODULES


def test_all_experiment_scripts_have_standard_entrypoints():
    for module_name, family, experiment_name, is_multi_instance in EXPERIMENT_SCRIPTS:
        module = _import_module(module_name)

        assert module.EXPERIMENT_NAME == experiment_name
        assert module.EXPERIMENT_FAMILY == family
        assert callable(module.BENCHMARK)
        assert isinstance(module.PARAMS, dict)
        assert callable(module.run)
        assert callable(module.main)

        output_dir = Path(module.OUTPUT_DIR)
        parent = REPO_ROOT / "results" / family / experiment_name
        if is_multi_instance:
            assert output_dir == parent
        else:
            assert output_dir.parent == parent
            assert output_dir.name


def test_all_experiment_script_params_match_benchmark_signatures():
    for module_name, *_ in EXPERIMENT_SCRIPTS:
        module = _import_module(module_name)
        signature = inspect.signature(module.BENCHMARK)
        accepts_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if accepts_kwargs:
            continue

        benchmark_keys = set(benchmark_params(module.PARAMS))
        accepted_keys = set(signature.parameters)
        unexpected = sorted(benchmark_keys - accepted_keys)

        assert not unexpected, f"{module_name} passes unsupported benchmark params: {unexpected}"
        assert "output_dir" in accepted_keys, f"{module_name} benchmark must accept output_dir"


def test_convex_eq_2d_loads_manifest_backed_record_directory(tmp_path):
    module = _import_module("homopt.experiments.benchmarks.convex_eq_2d")
    artifacts = tmp_path / "artifacts"
    records_dir = artifacts / "records"
    records_dir.mkdir(parents=True)

    expected = {
        "ALM": {"objective_values": [1.0]},
        "Hom-ALM": {"objective_values": [0.5]},
    }
    for algorithm, record in expected.items():
        np.save(records_dir / f"{algorithm}.npy", record, allow_pickle=True)

    (artifacts / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "algorithms": ["ALM", "Hom-ALM"],
                "records": {
                    "ALM": "artifacts/records/ALM.npy",
                    "Hom-ALM": "artifacts/records/Hom-ALM.npy",
                },
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "summary.json").write_text("{}", encoding="utf-8")

    loaded = module._load_records(tmp_path)

    assert loaded == expected


def test_experiment_scripts_do_not_hand_load_comparison_records():
    forbidden_fragments = (
        "np.load",
        "records_path",
        'payload["artifacts"]["records"]',
        "payload['artifacts']['records']",
    )
    offenders = []
    for folder in ("hom_alm", "hom_pgd", "inn_pgd", "learning_postprocess"):
        for script_path in (SCRIPTS_ROOT / folder).glob("run_*.py"):
            text = script_path.read_text(encoding="utf-8")
            if any(fragment in text for fragment in forbidden_fragments):
                offenders.append(str(script_path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_experiment_scripts_do_not_default_to_visualize_only():
    for module_name, *_ in EXPERIMENT_SCRIPTS:
        module = _import_module(module_name)
        assert bool(module.PARAMS.get("visualize_only", False)) is False


def _walk_mapping_keys(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key)
            yield from _walk_mapping_keys(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            yield from _walk_mapping_keys(value)


def test_experiment_script_params_do_not_expose_internal_outer_loop_aliases():
    forbidden_keys = {
        "outer_iterations",
        "outer_learning_rate",
        "outer_lr_decay",
        "outer_min_lr",
        "outer_stepsize_rule",
        "projection_outer_iterations",
        "linearization_outer_iterations",
    }
    offenders = {}
    for module_name, *_ in EXPERIMENT_SCRIPTS:
        module = _import_module(module_name)
        present = sorted(forbidden_keys.intersection(_walk_mapping_keys(module.PARAMS)))
        if present:
            offenders[module_name] = present

    assert offenders == {}


def test_experiment_script_params_do_not_carry_debug_reference_grid_config():
    offenders = []
    for module_name, *_ in EXPERIMENT_SCRIPTS:
        module = _import_module(module_name)
        if "reference_grid_size" in set(_walk_mapping_keys(module.PARAMS)):
            offenders.append(module_name)

    assert offenders == []


def test_poly_star_scripts_use_problem_identity_labels():
    compare = _import_module("scripts.hom_pgd.run_poly_star_compare")

    assert compare.PARAMS["result_label"] == "star_a0p5_k4_seed2030"


def test_convex_parametric_postprocess_stays_on_predictor_postprocess_route():
    postprocess = _import_module("scripts.learning_postprocess.run_convex_parametric_predictor_postprocess")
    configs = _import_module("scripts.learning_postprocess._configs")

    assert postprocess.PARAMS["problem_type"] == "qp"
    assert postprocess.PARAMS["problem_config"] == configs.CONVEX_PARAMETRIC_PROBLEM_CONFIG["problem_config"]
    assert postprocess.PARAMS["result_label"] == "qp_n10_eq2_ineq10_samples32"
    assert postprocess.PARAMS["predictor_type"] == "nn"
    assert postprocess.PARAMS["predictor_model_config"]["approach"] == "unsupervise"
    assert "supervise_target" not in postprocess.PARAMS["predictor_model_config"]
    assert postprocess.PARAMS["baselines"] == [
        "predict_only",
        "predict_diff_projection",
    ]
    assert "model_config" not in postprocess.PARAMS
    assert "train_config" not in postprocess.PARAMS


def test_acopf_parametric_postprocess_stays_on_predictor_postprocess_route():
    postprocess = _import_module("scripts.learning_postprocess.run_acopf_predictor_postprocess")

    assert postprocess.PARAMS["case"] == 57
    assert postprocess.PARAMS["dataset_samples"] == 10000
    assert postprocess.PARAMS["dataset_path"] is None
    assert postprocess.PARAMS["result_label"] == "acopf_case57_samples10000_eval32"
    assert postprocess.PARAMS["problem_config"]["prediction_space"] == "pf_partial"
    assert postprocess.PARAMS["problem_config"]["completion_config"]["gradient_mode"] == "implicit"
    assert postprocess.PARAMS["predictor_type"] == "nn"
    assert postprocess.PARAMS["predictor_model_config"]["approach"] == "unsupervise"
    assert "supervise_target" not in postprocess.PARAMS["predictor_model_config"]
    assert postprocess.PARAMS["baselines"] == [
        "predict_only",
        "predict_diff_projection",
    ]
    assert "model_config" not in postprocess.PARAMS
    assert "train_config" not in postprocess.PARAMS
