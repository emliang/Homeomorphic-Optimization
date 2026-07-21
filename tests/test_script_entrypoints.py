from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_HELPERS_ROOT = REPO_ROOT / "scripts"

if str(SCRIPT_HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_HELPERS_ROOT))


import _common as script_common  # noqa: E402
import homopt.experiments.script_runtime as script_runtime  # noqa: E402


def test_script_output_uses_problem_or_scale_subfolder():
    assert script_common.script_run_output_dir(
        "maxcut_sdp_compare",
        params={"result_label": "maxcut_n10_alpha0.5", "problem_type": "maxcut_sdp"},
        family="hom_pgd",
    ) == REPO_ROOT / "results" / "hom_pgd" / "maxcut_sdp_compare" / "maxcut_n10_alpha0_5"

    assert script_common.script_run_output_dir(
        "poly_star_compare",
        params={"problem_type": "poly_star"},
        family="hom_pgd",
    ) == REPO_ROOT / "results" / "hom_pgd" / "poly_star_compare" / "poly_star"

    assert script_common.script_run_output_dir(
        "stiefel_eq_compare",
        params={"scale_label": "small_digits", "problem_type": "stiefel"},
        family="hom_alm",
    ) == REPO_ROOT / "results" / "hom_alm" / "stiefel_eq_compare" / "small_digits"


def test_script_output_paths_do_not_use_script_runs_bucket():
    output_dir = script_common.script_run_output_dir(
        "convex_eq_compare",
        params={"problem_type": "socp_eq", "n_var": 10},
        family="hom_alm",
    )

    assert "script_runs" not in output_dir.parts


def test_script_metadata_is_not_passed_to_benchmark():
    seen = {}

    def fake_benchmark(**kwargs):
        seen.update(kwargs)
        return {"objective": 0.0, "feasible": True}

    run, _main, output_dir = script_common.make_benchmark_entrypoint(
        "metadata_strip_smoke",
        fake_benchmark,
        {
            "result_label": "case_a",
            "scale_label": "small",
            "problem_type": "poly",
            "n_var": 2,
        },
        output_dir=REPO_ROOT / "results" / "_test" / "metadata_strip_smoke",
    )

    run()

    assert seen["output_dir"] == output_dir
    assert seen["problem_type"] == "poly"
    assert seen["n_var"] == 2
    assert "result_label" not in seen
    assert "scale_label" not in seen


def test_multi_layer_param_merge_applies_cuda_fallback_once(monkeypatch, capsys):
    monkeypatch.setattr(script_runtime.torch.cuda, "is_available", lambda: False)
    script_runtime._CUDA_FALLBACK_WARNED.clear()

    params = script_common.merge_param_layers(
        {"device": "auto", "dtype": "float32"},
        {"n_var": 2},
        {"device": "cuda:0"},
    )

    captured = capsys.readouterr().out
    assert params["device"] == "cpu"
    assert captured.count("Requested device=") == 1
    assert "device='cuda:0'" in captured
    assert "device='auto'" not in captured


def test_multi_instance_entrypoint_reports_parent_output_dir():
    def fake_benchmark(**_kwargs):
        return {"objective": 0.0, "feasible": True}

    _run, _main, output_dir = script_common.make_benchmark_entrypoint(
        "convex_ineq_compare",
        fake_benchmark,
        {"problem_type": "socp", "n_var": 10},
        family="hom_pgd",
        instances=[(None, {"n_var": 20})],
        label_builder=lambda params: f"n{params['n_var']}",
    )

    assert output_dir == REPO_ROOT / "results" / "hom_pgd" / "convex_ineq_compare"


def test_multi_instance_entrypoint_uses_explicit_parent_output_dir(tmp_path):
    seen_output_dirs = []

    def fake_benchmark(**kwargs):
        seen_output_dirs.append(Path(kwargs["output_dir"]))
        return {"objective": 0.0, "feasible": True}

    parent_dir = tmp_path / "runs"
    _run, main, output_dir = script_common.make_benchmark_entrypoint(
        "convex_ineq_compare",
        fake_benchmark,
        {"problem_type": "socp", "n_var": 10},
        output_dir=parent_dir,
        family="hom_pgd",
        instances=[(None, {"n_var": 20})],
        label_builder=lambda params: f"n{params['n_var']}",
    )

    main()

    assert output_dir == parent_dir
    assert seen_output_dirs == [parent_dir / "n20"]
