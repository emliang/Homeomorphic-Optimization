from pathlib import Path

from homopt.experiments import (
    EXPERIMENT_FAMILIES,
    SCRIPT_EXPERIMENTS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_scripts_have_no_legacy_wrappers():
    assert not list((REPO_ROOT / "scripts").glob("**/run_legacy_*.py"))


def test_script_experiment_catalog_matches_active_scripts():
    script_paths = {
        str(path.relative_to(REPO_ROOT))
        for folder in ("hom_pgd", "hom_alm", "inn_pgd", "learning_postprocess")
        for path in (REPO_ROOT / "scripts" / folder).glob("run_*.py")
    }
    assert set(SCRIPT_EXPERIMENTS.values()) == script_paths
    for name, path in SCRIPT_EXPERIMENTS.items():
        assert name in Path(path).stem


def test_experiment_families_reference_active_scripts():
    family_scripts = set()
    for family in EXPERIMENT_FAMILIES.values():
        scripts = family["scripts"]
        assert scripts
        for script_name in scripts:
            assert script_name in SCRIPT_EXPERIMENTS
        family_scripts.update(scripts)

    assert set(SCRIPT_EXPERIMENTS).issubset(family_scripts)
    assert set(EXPERIMENT_FAMILIES["inn_pgd"]["scripts"]) == {
        "qcqp_inn_compare",
        "qcqp_inn_ablation",
        "qcqp_inn_toy",
        "jcc_opf_compare",
    }


def test_experiment_catalog_is_script_only():
    assert set(SCRIPT_EXPERIMENTS)
    assert not any("preset" in name for name in SCRIPT_EXPERIMENTS)
