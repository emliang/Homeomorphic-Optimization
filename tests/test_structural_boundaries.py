from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_files(root):
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _contains_runtime_import(text, patterns):
    return any(re.search(pattern, text, flags=re.MULTILINE) for pattern in patterns)


def test_src_package_has_no_runtime_utils_imports():
    src_root = REPO_ROOT / "src" / "homopt"
    offending = []
    for path in _python_files(src_root):
        text = path.read_text()
        if _contains_runtime_import(text, [r"^\s*from utils(?:\.|\s)", r"^\s*import utils(?:\b|\.)"]):
            offending.append(str(path.relative_to(REPO_ROOT)))
    assert offending == []


def test_script_wrappers_have_no_legacy_runtime_imports():
    scripts_root = REPO_ROOT / "scripts"
    offending = []
    for path in _python_files(scripts_root):
        text = path.read_text()
        if _contains_runtime_import(
            text,
            [
                r"legacy_presets",
                r"^\s*from utils(?:\.|\s)",
                r"^\s*import utils(?:\b|\.)",
            ],
        ):
                offending.append(str(path.relative_to(REPO_ROOT)))
    assert offending == []


def test_script_runtime_logic_lives_in_src_package():
    script_common = (REPO_ROOT / "scripts" / "_common.py").read_text()
    runtime_module = REPO_ROOT / "src" / "homopt" / "experiments" / "script_runtime.py"
    runtime_text = runtime_module.read_text()

    assert runtime_module.exists()
    assert "from homopt.experiments.script_runtime import" in script_common
    assert "import torch" not in script_common
    assert "def _device_with_cuda_fallback" not in script_common
    assert "def merge_params" not in script_common
    assert "def _device_with_cuda_fallback" in runtime_text
    assert "def merge_params" in runtime_text
    assert "def make_benchmark_entrypoint" in runtime_text
    assert "fallback_params" not in script_common
    assert "fallback_params" not in runtime_text


def test_optimizer_registry_does_not_encode_result_tuple_shapes():
    registry_source = (REPO_ROOT / "src" / "homopt" / "optim" / "registry.py").read_text()
    assert "returns_transform_time" not in registry_source
    assert "last_trans_time = result" not in registry_source
    assert "final_decision, decision_trajectory" not in registry_source


def test_scripts_tree_only_contains_active_families():
    allowed = {"hom_pgd", "hom_alm", "inn_pgd", "learning_postprocess"}
    folders = {path.name for path in (REPO_ROOT / "scripts").iterdir() if path.is_dir() and path.name != "__pycache__"}
    assert folders.issubset(allowed)


def test_script_family_shared_settings_use_configs_modules():
    scripts_root = REPO_ROOT / "scripts"
    retired_names = {"_defaults.py", "_presets.py"}
    offenders = []
    for folder in ("hom_pgd", "hom_alm", "inn_pgd", "learning_postprocess"):
        for path in (scripts_root / folder).glob("_*.py"):
            if path.name in retired_names:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_script_config_modules_do_not_mutate_import_paths():
    scripts_root = REPO_ROOT / "scripts"
    offenders = []
    for path in scripts_root.glob("*/_configs.py"):
        text = path.read_text()
        if "sys.path" in text or "SCRIPT_ROOT" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_experiment_scripts_use_family_config_imports():
    scripts_root = REPO_ROOT / "scripts"
    offending = []
    for folder in ("hom_pgd", "hom_alm", "inn_pgd", "learning_postprocess"):
        for path in _python_files(scripts_root / folder):
            text = path.read_text()
            if "from _configs import" in text or "SCRIPT_DIR" in text:
                offending.append(str(path.relative_to(REPO_ROOT)))
    assert offending == []


def test_experiment_scripts_import_family_benchmarks_not_compat_facade():
    scripts_root = REPO_ROOT / "scripts"
    offending = []
    for folder in ("hom_pgd", "hom_alm", "inn_pgd", "learning_postprocess"):
        for path in _python_files(scripts_root / folder):
            text = path.read_text()
            if (
                "homopt.experiments.benchmarks import" in text
                or "homopt.experiments.benchmarks." in text
            ):
                offending.append(str(path.relative_to(REPO_ROOT)))
    assert offending == []


def test_lower_layers_do_not_import_experiments():
    src_root = REPO_ROOT / "src" / "homopt"
    lower_layer_roots = (
        "records",
        "solvers",
        "learning",
        "viz",
        "models",
        "optim",
        "problems",
        "mappings",
        "utils",
    )
    offending = []
    for folder in lower_layer_roots:
        for path in _python_files(src_root / folder):
            text = path.read_text()
            if "homopt.experiments" in text:
                offending.append(str(path.relative_to(REPO_ROOT)))
    assert offending == []


def test_experiment_scripts_do_not_import_workload_implementation_modules():
    scripts_root = REPO_ROOT / "scripts"
    implementation_modules = (
        "homopt.experiments.adversarial import",
        "homopt.experiments.benchmarks_single import",
        "homopt.experiments.benchmarks_parametric import",
    )
    offending = []
    for folder in ("hom_pgd", "hom_alm", "inn_pgd", "learning_postprocess"):
        for path in _python_files(scripts_root / folder):
            text = path.read_text()
            if any(pattern in text for pattern in implementation_modules):
                offending.append(str(path.relative_to(REPO_ROOT)))
    assert offending == []


def test_public_experiment_surface_is_script_only():
    experiments_root = REPO_ROOT / "src" / "homopt" / "experiments"
    retired_modules = {
        "benchmarks.py",
        "config.py",
        "schema.py",
        "cli.py",
        "main.py",
        "_benchmark_registry.py",
        "_parsers.py",
    }
    assert not [path.name for path in experiments_root.glob("*.py") if path.name in retired_modules]


def test_experiment_context_has_no_config_entrypoint_compatibility():
    context_text = (REPO_ROOT / "src" / "homopt" / "experiments" / "context.py").read_text()
    entrypoints_text = (REPO_ROOT / "src" / "homopt" / "experiments" / "entrypoints.py").read_text()

    for snippet in ("from_config", "config_path", "preset"):
        assert snippet not in context_text
        assert snippet not in entrypoints_text


def test_project_metadata_has_no_retired_console_scripts():
    text = (REPO_ROOT / "pyproject.toml").read_text()
    assert "[project.scripts]" not in text
    assert "homopt.experiments.cli:main" not in text
    assert "homopt.experiments.main:main" not in text


def test_models_namespace_has_no_coninn_compat_alias():
    models_init = (REPO_ROOT / "src" / "homopt" / "models" / "__init__.py").read_text()
    assert not (REPO_ROOT / "src" / "homopt" / "models" / "_inn_impl.py").exists()
    inn_impl = (REPO_ROOT / "src" / "homopt" / "models" / "inn.py").read_text()
    assert "ConINN" not in models_init
    assert "ConINN =" not in inn_impl


def test_models_namespace_does_not_export_optimizers_or_postprocess_helpers():
    models_init = (REPO_ROOT / "src" / "homopt" / "models" / "__init__.py").read_text()
    forbidden = (
        "INNPGDOptimizer",
        "pgd_transformed_space",
        "homeomorphic_bisection",
        "interior_point_bisection",
        "homeo_bisection",
        "ip_bisection",
    )
    for snippet in forbidden:
        assert snippet not in models_init


def test_cvxinn_family_lives_in_dedicated_subpackage():
    models_root = REPO_ROOT / "src" / "homopt" / "models"
    package_root = REPO_ROOT / "src" / "homopt"
    assert not (package_root / "sampling.py").exists()
    assert (models_root / "sampling.py").exists()
    for retired_name in ("_inn_impl.py", "_ipnn_impl.py", "_predictor_impl.py", "_nn_common.py", "_local_stretch.py"):
        assert not (models_root / retired_name).exists()
    inn_impl = (models_root / "inn.py").read_text()
    cvx_models = (REPO_ROOT / "src" / "homopt" / "models" / "cvxinn" / "models.py").read_text()
    cvx_training = (REPO_ROOT / "src" / "homopt" / "models" / "cvxinn" / "training.py").read_text()

    forbidden_in_inn_impl = (
        "class CvxINN",
        "class TanhGaugeCvxINN",
        "class CubeGaugeCvxINN",
        "class SphereReparam",
        "class SphereReparamGaugeCvxINN",
        "def train_cvxinn_min_distortion",
    )
    for snippet in forbidden_in_inn_impl:
        assert snippet not in inn_impl

    learning_root = REPO_ROOT / "src" / "homopt" / "learning"
    optim_root = REPO_ROOT / "src" / "homopt" / "optim"
    inn_training = (learning_root / "training" / "inn.py").read_text()
    inn_pgd = (optim_root / "inn" / "pgd.py").read_text()
    assert "class INN" in inn_impl
    assert "def train_mdh_mapping" not in inn_impl
    assert "homeomorphic_bisection" not in inn_impl
    assert "class INNPGDOptimizer" not in inn_impl
    assert not (models_root / "inn_optimization.py").exists()
    assert "def train_mdh_mapping" in inn_training
    assert "def pgd_transformed_space" in inn_pgd
    assert "class INNPGDOptimizer" in inn_pgd
    assert "class CvxINN" in cvx_models
    assert "class SphereReparamGaugeCvxINN" in cvx_models
    assert "def train_cvxinn_min_distortion" in cvx_training


def test_learning_training_optimizer_schedule_is_shared():
    training_root = REPO_ROOT / "src" / "homopt" / "learning" / "training"
    utility_text = (training_root / "utils.py").read_text()
    assert "def build_adamw_with_step_scheduler" in utility_text

    offenders = []
    for path in _python_files(training_root):
        if path.name == "utils.py":
            continue
        text = path.read_text()
        if "optim.AdamW" in text or "StepLR" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_learning_training_cache_does_not_hide_stale_checkpoint_errors():
    cache_text = (REPO_ROOT / "src" / "homopt" / "learning" / "training" / "cache.py").read_text()
    forbidden = (
        "_RETIRED_MODEL_MODULES",
        "_is_retired_model_checkpoint_error",
        "Ignoring stale model checkpoint",
        "references retired module",
    )
    for snippet in forbidden:
        assert snippet not in cache_text


def test_inn_condition_encoders_live_outside_core_inn_model():
    models_root = REPO_ROOT / "src" / "homopt" / "models"
    inn_text = (models_root / "inn.py").read_text()
    condition_text = (models_root / "condition.py").read_text()
    models_init = (models_root / "__init__.py").read_text()

    for snippet in (
        "class ResBlock",
        "class MLP",
        "class PINN",
        "class Mixer",
        "class QuadMixer",
        "def build_condition_encoder",
        "def normalize_condition_encoder_type",
    ):
        assert snippet in condition_text
        assert snippet not in inn_text

    assert "build_condition_encoder" in inn_text
    assert "normalize_condition_encoder_type" in inn_text
    assert "build_condition_encoder" in models_init
    assert "normalize_condition_encoder_type" in models_init


def test_convex_problem_families_live_in_dedicated_subpackage():
    problems_root = REPO_ROOT / "src" / "homopt" / "problems"
    assert not (problems_root / "_convex_impl.py").exists()

    problems_init = (problems_root / "__init__.py").read_text()
    standard_text = (problems_root / "convex" / "standard.py").read_text()
    wrappers_text = (problems_root / "convex" / "wrappers.py").read_text()
    star_text = (problems_root / "convex" / "star.py").read_text()
    maxcut_text = (problems_root / "convex" / "maxcut.py").read_text()
    generators_text = (problems_root / "convex" / "generators.py").read_text()

    assert '"ConvexOpt": ("homopt.problems.convex.standard", "ConvexOpt")' in problems_init
    assert '"create_test_problem": ("homopt.problems.convex.generators", "create_test_problem")' in problems_init
    assert "class ConvexOpt" in standard_text
    assert "class ConvexOptEq" in standard_text
    assert "class ProjProblem" in wrappers_text
    assert "class LinearProblem" in wrappers_text
    assert "class ToyStarOpt" in star_text
    assert "class PolyStarOpt" in star_text
    assert "class MaxCutSDP" in maxcut_text
    assert "class BMSDP" in maxcut_text
    assert "def create_maxcut_problem" in maxcut_text
    assert "def create_test_problem" in generators_text


def test_qcqp_problem_family_lives_in_dedicated_subpackage():
    problems_root = REPO_ROOT / "src" / "homopt" / "problems"
    assert not (problems_root / "_qcqp_impl.py").exists()

    problems_init = (problems_root / "__init__.py").read_text()
    deterministic_text = (problems_root / "qcqp" / "deterministic.py").read_text()
    parametric_text = (problems_root / "qcqp" / "parametric.py").read_text()
    generators_text = (problems_root / "qcqp" / "generators.py").read_text()
    spectral_text = (problems_root / "qcqp" / "spectral.py").read_text()

    assert '"QCOpt": ("homopt.problems.qcqp.deterministic", "QCOpt")' in problems_init
    assert '"NonConvexQCProblem": ("homopt.problems.qcqp.parametric", "NonConvexQCProblem")' in problems_init
    assert '"create_test_QC_problem": ("homopt.problems.qcqp.generators", "create_test_QC_problem")' in problems_init
    assert "class QCOpt" in deterministic_text
    assert "class NonConvexQCProblem" in parametric_text
    assert "def create_test_QC_problem" in generators_text
    assert "def _spectral_quadratic_matrix" in spectral_text


def test_jcc_problem_family_lives_in_dedicated_subpackage():
    problems_root = REPO_ROOT / "src" / "homopt" / "problems"
    assert not (problems_root / "_jcc_opf_impl.py").exists()
    assert not (problems_root / "_jcc_linear_impl.py").exists()

    problems_init = (problems_root / "__init__.py").read_text()
    jcc_init = (problems_root / "jcc" / "__init__.py").read_text()
    opf_text = (problems_root / "jcc" / "opf.py").read_text()
    linear_text = (problems_root / "jcc" / "linear.py").read_text()

    assert '"JCCDCOPFProblem": ("homopt.problems.jcc.opf", "JCCDCOPFProblem")' in problems_init
    assert '"JCCLinearProblem": ("homopt.problems.jcc.linear", "JCCLinearProblem")' in problems_init
    assert '"JCCIMProblem": ("homopt.problems.jcc.linear", "JCCIMProblem")' in problems_init
    assert '"solve_jcc_dcopt": ("homopt.problems.jcc.opf", "solve_jcc_dcopt")' in problems_init
    assert "from .opf import JCCDCOPFProblem, solve_jcc_dcopt" in jcc_init
    assert "class JCCDCOPFProblem" in opf_text
    assert "class JCCLinearProblem" in linear_text
    assert "class JCCIMProblem" in linear_text
    for forbidden in (
        "class JCCDCOPFSolver",
        "class JCCDCOPFCVaRSolver",
        "class JCCDCOPFRobustScenarioSolver",
        "_JCCCVXPYModelBuilder",
        "_solve_cvxpy_problem",
    ):
        assert forbidden not in opf_text
    for forbidden in (
        "class JCCLinearSolver",
        "class JCCLinearCVaRSolver",
        "class JCCLinearRobustScenarioSolver",
        "_solve_cvxpy_problem",
    ):
        assert forbidden not in linear_text


def test_jcc_solver_implementations_live_in_solver_package():
    solver_root = REPO_ROOT / "src" / "homopt" / "solvers"
    jcc_text = (solver_root / "jcc.py").read_text()
    jcc_linear_text = (solver_root / "jcc_linear.py").read_text()
    for snippet in (
        "class JCCDCOPFSolver",
        "class JCCDCOPFCVaRSolver",
        "class JCCDCOPFRobustScenarioSolver",
        "_JCCCVXPYModelBuilder",
        "_solve_cvxpy_problem",
    ):
        assert snippet in jcc_text
    for snippet in (
        "class JCCLinearSolver",
        "class JCCLinearCVaRSolver",
        "class JCCLinearRobustScenarioSolver",
        "_solve_cvxpy_problem",
    ):
        assert snippet in jcc_linear_text


def test_reference_context_helpers_live_under_experiments():
    references_impl = REPO_ROOT / "src" / "homopt" / "experiments" / "common" / "convex_reference.py"
    impl_text = references_impl.read_text()
    convex_benchmark_text = (
        REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "convex.py"
    ).read_text()

    assert "def prepare_convex_reference_context" in impl_text
    assert "def build_convex_problem_config" in impl_text
    assert "from homopt.experiments.common.convex_reference import" in convex_benchmark_text
    assert not (REPO_ROOT / "src" / "homopt" / "solvers" / "references.py").exists()


def test_common_reports_do_not_write_learning_plots_or_artifacts():
    reports_text = (REPO_ROOT / "src" / "homopt" / "experiments" / "common" / "reports.py").read_text()
    learning_reports_text = (
        REPO_ROOT / "src" / "homopt" / "experiments" / "common" / "learning_reports.py"
    ).read_text()
    assert "def make_parametric_learning_result_row" in reports_text
    assert "def save_parametric_learning_artifacts" not in reports_text
    assert "from homopt.viz" not in reports_text
    assert "def save_parametric_learning_artifacts" in learning_reports_text
    assert "def save_parametric_learning_plots" in learning_reports_text


def test_experiment_scripts_do_not_import_viz_or_artifact_internals():
    scripts_root = REPO_ROOT / "scripts"
    forbidden = (
        "homopt.viz",
        "homopt.records.artifacts",
    )
    offending = []
    for folder in ("hom_pgd", "hom_alm", "inn_pgd", "learning_postprocess"):
        for path in _python_files(scripts_root / folder):
            text = path.read_text()
            if any(pattern in text for pattern in forbidden):
                offending.append(str(path.relative_to(REPO_ROOT)))
    assert offending == []


def test_hom_alm_convex_eq_2d_script_is_thin():
    script_text = (REPO_ROOT / "scripts" / "hom_alm" / "run_convex_eq_2d.py").read_text()
    benchmark_text = (
        REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "convex_eq_2d.py"
    ).read_text()
    assert "TOY_LINEAR_A" not in script_text
    assert "def convex_eq_2d_benchmark" not in script_text
    assert "save_convex_2d_visualizations" not in script_text
    assert "TOY_LINEAR_A" in benchmark_text
    assert "def convex_eq_2d_benchmark" in benchmark_text
    assert "save_convex_2d_visualizations" in benchmark_text


def test_stiefel_problem_family_lives_in_dedicated_subpackage():
    problems_root = REPO_ROOT / "src" / "homopt" / "problems"
    assert not (problems_root / "_stiefel_impl.py").exists()

    problems_init = (problems_root / "__init__.py").read_text()
    stiefel_init = (problems_root / "stiefel" / "__init__.py").read_text()
    config_text = (problems_root / "stiefel" / "config.py").read_text()
    core_text = (problems_root / "stiefel" / "core.py").read_text()

    assert '"StiefelProblem": ("homopt.problems.stiefel.core", "StiefelProblem")' in problems_init
    assert (
        '"create_constrained_pca_stiefel_config": '
        '("homopt.problems.stiefel.config", "create_constrained_pca_stiefel_config")'
    ) in problems_init
    assert "from .config import create_constrained_pca_stiefel_config" in stiefel_init
    assert "from .core import StiefelProblem" in stiefel_init
    assert "def create_constrained_pca_stiefel_config" in config_text
    assert "class StiefelProblem" in core_text
    assert "class StiefelProblem" not in config_text


def test_gauge_map_lives_in_split_package():
    mappings_root = REPO_ROOT / "src" / "homopt" / "mappings"
    gauge_root = mappings_root / "gauge"
    assert not (mappings_root / "gauge.py").exists()
    assert not (gauge_root / "core.py").exists()

    map_text = (gauge_root / "map.py").read_text()
    maxcut_text = (gauge_root / "maxcut.py").read_text()
    init_text = (gauge_root / "__init__.py").read_text()
    math_text = (gauge_root / "math.py").read_text()
    state_text = (gauge_root / "state.py").read_text()
    gradients_text = (gauge_root / "gradients.py").read_text()
    constraints_text = (gauge_root / "constraints.py").read_text()
    autograd_text = (gauge_root / "autograd.py").read_text()

    assert "self.set = convex_set" not in map_text
    assert "class GaugeMap" in map_text
    assert "class GaugeMapMaxCut" not in map_text
    assert "class GaugeMapMaxCut" in maxcut_text
    assert "def explicit_forward_state" in state_text
    assert "def explicit_candidate_cache" in constraints_text
    assert "def explicit_vjp_from_state" in gradients_text
    assert "class GaugeForwardExplicitFunction" in autograd_text
    assert "_explicit_forward_state" not in init_text
    assert "_max_real_quadratic_root" not in init_text
    assert "def _max_real_quadratic_root" not in math_text
    for formula_detail in (
        "def explicit_candidate_cache",
        "def best_beta_gradient_u",
        "def explicit_forward_state",
        "class GaugeForwardExplicitFunction",
    ):
        assert formula_detail not in map_text


def test_gauge_and_problem_gradients_do_not_keep_debug_fallback_paths():
    checked_files = [
        REPO_ROOT / "src" / "homopt" / "mappings" / "gauge" / "map.py",
        REPO_ROOT / "src" / "homopt" / "problems" / "convex" / "standard.py",
        REPO_ROOT / "src" / "homopt" / "problems" / "convex" / "maxcut.py",
        REPO_ROOT / "src" / "homopt" / "problems" / "qcqp" / "deterministic.py",
    ]
    combined = "\n".join(path.read_text() for path in checked_files)

    assert "_bisection_point_to_boundary_distance" not in combined
    assert "zeroorder" not in combined
    assert "finite_diff" not in combined


def test_single_benchmarks_do_not_depend_on_parametric_common():
    experiments_root = REPO_ROOT / "src" / "homopt" / "experiments"
    assert not (experiments_root / "benchmarks_single.py").exists()
    for module_name in ("star.py", "convex.py", "maxcut.py"):
        text = (experiments_root / "benchmarks" / module_name).read_text()
        assert "homopt.experiments.common.parametric" not in text
        assert "homopt.experiments.common.inn_training" not in text


def test_common_has_no_parametric_family_bucket():
    common_root = REPO_ROOT / "src" / "homopt" / "experiments" / "common"
    qcqp_setup_text = (
        REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "qcqp" / "setup.py"
    ).read_text()
    qcqp_reports_text = (
        REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "qcqp" / "reports.py"
    ).read_text()
    jcc_setup_text = (
        REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "jcc" / "setup.py"
    ).read_text()
    assert not (common_root / "parametric.py").exists()
    assert not (common_root / "inn_training.py").exists()
    assert not (common_root / "learning_training.py").exists()
    assert "def normalize_qcqp_problem_config" in qcqp_setup_text
    assert "def normalize_qcqp_inn_benchmark_config" in qcqp_setup_text
    assert "def extract_qcqp_inn_instance_metrics" in qcqp_reports_text
    assert "def normalize_jcc_linear_problem_config" in jcc_setup_text


def test_inn_training_cache_helpers_live_in_learning_package():
    learning_root = REPO_ROOT / "src" / "homopt" / "learning"
    training_root = learning_root / "training"
    for retired_name in (
        "inn_training.py",
        "ipnn_training.py",
        "decision_predictor_training.py",
        "training_cache.py",
        "training_utils.py",
    ):
        assert not (learning_root / retired_name).exists()
    inn_text = (training_root / "inn.py").read_text()
    ipnn_text = (training_root / "ipnn.py").read_text()
    mapping_text = (training_root / "mapping.py").read_text()
    predictor_text = (training_root / "predictor.py").read_text()
    cache_text = (training_root / "cache.py").read_text()
    learning_init = (learning_root / "__init__.py").read_text()

    for snippet in (
        "def build_inn_runtime_args",
        "def load_or_train_inn_mapping",
    ):
        assert snippet in inn_text

    for snippet in (
        "def load_or_train_ipnn_mapping",
    ):
        assert snippet in ipnn_text
        assert snippet not in inn_text

    for snippet in (
        "def build_inn_training_payload",
        "def build_qcqp_inn_training_payload",
        "def build_qcqp_ipnn_training_payload",
    ):
        assert snippet not in inn_text
        assert snippet not in ipnn_text

    for snippet in (
        "def build_training_payload",
        "def load_or_train_mapping",
        "def training_time_from_record",
    ):
        assert snippet in cache_text
        assert snippet not in inn_text

    for snippet in (
        "def build_learning_mapping_payload",
        "def load_or_train_learning_mapping",
        "def prepare_learning_mapping",
    ):
        assert snippet in mapping_text
        assert snippet not in inn_text
        assert snippet not in ipnn_text

    for snippet in (
        "def build_predictor_training_payload",
        "def load_or_train_decision_predictor",
        "def prepare_decision_predictor",
        "def train_decision_predictor",
    ):
        assert snippet in predictor_text
        assert snippet not in inn_text
        assert snippet not in ipnn_text

    assert "build_learning_mapping_payload" in learning_init
    assert "build_predictor_training_payload" in learning_init
    assert "load_or_train_learning_mapping" in learning_init
    assert "load_or_train_decision_predictor" in learning_init
    assert "prepare_learning_mapping" in learning_init
    assert "prepare_decision_predictor" in learning_init
    assert "load_or_train_inn_mapping" in learning_init
    assert "load_or_train_ipnn_mapping" in learning_init
    assert "load_or_train_mapping" in learning_init
    assert "build_training_payload" in learning_init
    assert "build_inn_training_payload" not in learning_init
    assert "build_qcqp_inn_training_payload" not in learning_init
    assert "build_qcqp_ipnn_training_payload" not in learning_init


def test_convex_parametric_learning_stays_separate_from_inn_pgd():
    text = (REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "convex_parametric.py").read_text()
    forbidden = (
        "INNPGD",
        "load_or_train_inn_mapping",
        "load_or_train_ipnn_mapping",
        "HomeomorphicProjectionRefiner",
        "IPNNBisectionRefiner",
        "QCQPSolver",
    )
    for snippet in forbidden:
        assert snippet not in text


def test_learning_refiners_live_in_focused_subpackage():
    learning_root = REPO_ROOT / "src" / "homopt" / "learning"
    refiners_root = learning_root / "refiners"
    assert not (learning_root / "simple.py").exists()
    assert not (learning_root / "postprocess.py").exists()

    predictors_text = (learning_root / "predictors.py").read_text()
    basic_text = (refiners_root / "basic.py").read_text()
    gradient_text = (refiners_root / "gradient.py").read_text()
    solver_text = (refiners_root / "solver.py").read_text()
    homeomorphic_text = (refiners_root / "homeomorphic.py").read_text()

    assert "class ConstantDecisionPredictor" in predictors_text
    assert "class NeuralDecisionPredictor" in predictors_text
    assert "class IdentityRefiner" in basic_text
    assert "class RayBisectionRefiner" in basic_text
    assert "class DiffProjectionRefiner" in gradient_text
    assert "class ExactSolverProjectionRefiner" in solver_text
    assert "class InitializedOptSolverRefiner" in solver_text
    assert "class HomeomorphicProjectionRefiner" in homeomorphic_text
    assert "class IPNNBisectionRefiner" in homeomorphic_text


def test_learning_bisection_loop_has_single_owner():
    learning_root = REPO_ROOT / "src" / "homopt" / "learning"
    models_root = REPO_ROOT / "src" / "homopt" / "models"
    optim_root = REPO_ROOT / "src" / "homopt" / "optim"
    bisection_text = (learning_root / "bisection" / "core.py").read_text()
    assert "def bisect_segment" in bisection_text
    assert "alpha_lower" in bisection_text
    assert not (learning_root / "bisection.py").exists()
    assert not (learning_root / "bisection_adapters.py").exists()
    for path in (
        learning_root / "bisection" / "adapters.py",
        learning_root / "refiners" / "basic.py",
        models_root / "ipnn.py",
        optim_root / "inn" / "pgd.py",
    ):
        text = path.read_text()
        assert "alpha_lower" not in text
        assert "alpha_upper" not in text


def test_common_has_no_benchmark_facade():
    common_root = REPO_ROOT / "src" / "homopt" / "experiments" / "common"
    records_root = REPO_ROOT / "src" / "homopt" / "records"
    method_specs_root = REPO_ROOT / "src" / "homopt" / "experiments" / "method_specs"
    assert not (common_root / "benchmark.py").exists()
    artifacts_shim_text = (common_root / "artifacts.py").read_text()
    artifacts_text = (records_root / "artifacts.py").read_text()
    comparison_shim_text = (common_root / "comparison.py").read_text()
    comparison_text = (records_root / "comparison.py").read_text()
    runtime_text = (common_root / "runtime.py").read_text()
    methods_shim_text = (common_root / "methods.py").read_text()
    methods_text = (method_specs_root / "common.py").read_text()
    references_text = (common_root / "convex_reference.py").read_text()
    reports_text = (common_root / "reports.py").read_text()
    assert "from homopt.records.artifacts import *" in artifacts_shim_text
    assert "from homopt.records.comparison import *" in comparison_shim_text
    assert "from homopt.experiments.method_specs.common import *" in methods_shim_text
    moved_helpers = (
        "artifact_root",
        "artifact_ref",
        "artifact_mapping",
        "build_benchmark_payload",
        "build_visualize_only_benchmark_payload",
        "save_json",
        "save_numpy",
        "write_csv",
        "load_incremental_comparison_artifacts",
        "load_visualize_only_comparison_artifacts",
        "ordered_algorithm_subset",
        "save_incremental_comparison_artifacts",
        "save_table_artifacts",
    )
    for helper in moved_helpers:
        assert f"def {helper}" in artifacts_text
    for helper in (
        "build_comparison_views",
        "build_single_instance_comparison_views",
        "summarize_comparison_rows_by_method",
    ):
        assert f"def {helper}" in comparison_text
    for helper in (
        "final_or_nan",
        "trajectory_dim",
        "resolve_runtime",
        "as_builtin",
        "ensure_int_list",
    ):
        assert f"def {helper}" in runtime_text
    for helper in (
        "build_first_order_method_params",
        "build_penalty_method_params",
        "build_hom_penalty_method_params",
        "normalize_jcc_lagrangian_config",
    ):
        assert f"def {helper}" in methods_text
    assert "def prepare_convex_reference_context" in references_text
    assert "def convex_solution_diagnostics" in reports_text


def test_benchmark_visualize_only_state_uses_common_loader():
    benchmark_paths = [
        REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "convex.py",
        REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "maxcut.py",
        REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "star.py",
        REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "stiefel" / "compare.py",
    ]
    for path in benchmark_paths:
        text = path.read_text()
        assert "load_visualize_only_comparison_artifacts" in text
        assert "previous_result.get(\"artifacts\"" not in text
        assert "list(records.keys()) or list(manifest.get(\"algorithms\")" not in text


def test_benchmark_package_inits_export_public_entrypoints_only():
    benchmarks_root = REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks"
    for family in ("qcqp", "jcc", "stiefel"):
        text = (benchmarks_root / family / "__init__.py").read_text()
        exported = text.split("__all__ =", 1)[-1]
        assert '"_' not in exported
        assert "'_" not in exported


def test_common_uses_clear_module_names():
    common_root = REPO_ROOT / "src" / "homopt" / "experiments" / "common"
    assert not (common_root / "method_specs.py").exists()
    assert not (common_root / "problem_reports.py").exists()
    assert (common_root / "methods.py").exists()
    assert (common_root / "convex_reference.py").exists()
    assert not (common_root / "references.py").exists()
    assert (common_root / "reports.py").exists()
    assert (REPO_ROOT / "src" / "homopt" / "records" / "artifacts.py").exists()
    assert (REPO_ROOT / "src" / "homopt" / "experiments" / "method_specs" / "common.py").exists()


def test_stiefel_eq_script_is_parameter_entrypoint_only():
    script_text = (REPO_ROOT / "scripts" / "hom_alm" / "run_stiefel_eq_compare.py").read_text()
    facade_text = (REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "stiefel" / "__init__.py").read_text()
    compare_text = (REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "stiefel" / "compare.py").read_text()
    setup_text = (REPO_ROOT / "src" / "homopt" / "experiments" / "benchmarks" / "stiefel" / "setup.py").read_text()

    for snippet in (
        "def _build_alm_params",
        "def _outer_config",
        "def _run_stiefel_ipopt",
        "StiefelRetractionOptimizer",
        "StiefelPyomoIPOPTSolver",
        "run_algorithm(",
        "outer_lr_decay",
        "outer_stepsize_rule",
    ):
        assert snippet not in script_text
        assert snippet not in setup_text
    assert "stiefel_algorithm_comparison" in script_text
    assert "from .compare import" in facade_text
    assert "def _build_alm_params" in compare_text
    assert "def stiefel_algorithm_comparison" in compare_text
    assert "def build_stiefel_problem" in setup_text


def test_jcc_opf_benchmarks_live_in_dedicated_module():
    experiments_root = REPO_ROOT / "src" / "homopt" / "experiments"
    jcc_root = experiments_root / "benchmarks" / "jcc"
    compare_text = (jcc_root / "compare.py").read_text()
    methods_text = (jcc_root / "methods.py").read_text()
    setup_text = (jcc_root / "setup.py").read_text()

    assert not (experiments_root / "benchmarks_single.py").exists()
    assert not (experiments_root / "benchmarks" / "jcc.py").exists()
    assert not (jcc_root / "opf.py").exists()
    for module_name in ("star.py", "convex.py", "maxcut.py"):
        single_text = (experiments_root / "benchmarks" / module_name).read_text()
        single_forbidden = (
            "class _JCCINNPGDAdapter",
            "def _build_jcc_problem",
            "def _normalize_jcc_inn_configs",
            "def _run_jcc_inn_pgd_variant",
            "def _run_jcc_solver_suite",
            "def jcc_algorithm_comparison",
            "def jcc_baseline_solver_sweep",
            "_JCCINNPGDAdapter,",
            "_run_jcc_inn_pgd_variant,",
        )
        for snippet in single_forbidden:
            assert snippet not in single_text

    assert "class _JCCINNPGDAdapter" in methods_text
    assert "def _run_jcc_inn_pgd_variant" in methods_text
    assert "def _build_jcc_problem" in setup_text
    assert "def jcc_algorithm_comparison" in compare_text
    assert "def jcc_baseline_solver_sweep" in compare_text


def test_parametric_learning_postprocess_surface_is_not_qcqp_or_jcc():
    experiments_root = REPO_ROOT / "src" / "homopt" / "experiments"
    assert not (experiments_root / "benchmarks_parametric.py").exists()

    benchmarks_root = experiments_root / "benchmarks"
    jcc_text = (benchmarks_root / "jcc" / "linear.py").read_text()
    learning_postprocess_text = (experiments_root / "learning_postprocess.py").read_text()
    registry_text = (benchmarks_root / "parametric_registry.py").read_text()
    for old_module in (
        "jcc_parametric.py",
        "qcqp_learning.py",
        "qcqp_inn_common.py",
        "qcqp_inn.py",
        "qcqp_inn_sweep.py",
        "qcqp_inn_training.py",
        "qcqp_route.py",
    ):
        assert not (benchmarks_root / old_module).exists()
    assert not (benchmarks_root / "qcqp" / "problem.py").exists()
    for module_name in ("setup.py", "reports.py", "training.py", "inn_pgd.py", "sweep.py", "route.py"):
        text = (benchmarks_root / "qcqp" / module_name).read_text()
        assert "def jcc_problem_benchmark" not in text
        assert "def jcc_linear_solver_benchmark" not in text
        assert "JCCLinearSolver" not in text
        assert "def qcqp_learning_benchmark" not in text
        assert "def _run_qcqp_learning_baseline" not in text
        assert "def _prepare_qcqp_predictor_resource" not in text

    assert "def jcc_problem_benchmark" in jcc_text
    assert "def jcc_linear_solver_benchmark" in jcc_text
    assert "JCCLinearSolver" in jcc_text
    assert "qcqp_learning_benchmark" not in learning_postprocess_text
    assert "qcqp_route_comparison" not in learning_postprocess_text
    assert "qcqp_learning_benchmark" not in registry_text
    assert "qcqp_route_comparison" not in registry_text
    assert "acopf_parametric_learning_benchmark" in learning_postprocess_text
    assert "convex_parametric_learning_benchmark" in learning_postprocess_text


def test_qcqp_inn_parametric_benchmarks_live_in_dedicated_modules():
    experiments_root = REPO_ROOT / "src" / "homopt" / "experiments"
    assert not (experiments_root / "benchmarks_parametric.py").exists()

    registry_text = (experiments_root / "benchmarks" / "parametric_registry.py").read_text()
    qcqp_root = experiments_root / "benchmarks" / "qcqp"
    common_text = (qcqp_root / "training.py").read_text()
    comparison_text = (qcqp_root / "inn_pgd.py").read_text()
    sweep_text = (qcqp_root / "sweep.py").read_text()

    assert "PARAMETRIC_BENCHMARKS" in registry_text
    assert "def _build_qcqp_inn_case_context" in common_text
    assert "def _sample_qcqp_inn_test_instances" in common_text
    assert "def qcqp_inn_comparison" in comparison_text
    assert "def qcqp_inn_experiment" in sweep_text
    assert "def qcqp_inn_sensitivity_sweep" in sweep_text
    assert "def inn_training_benchmark" in common_text


def test_adversarial_attack_core_lives_in_dedicated_module():
    experiments_root = REPO_ROOT / "src" / "homopt" / "experiments"
    assert not (experiments_root / "adversarial.py").exists()

    adversarial_root = experiments_root / "adversarial"
    models_root = REPO_ROOT / "src" / "homopt" / "models"
    learning_root = REPO_ROOT / "src" / "homopt" / "learning"
    learning_adversarial_root = learning_root / "adversarial"
    viz_root = REPO_ROOT / "src" / "homopt" / "viz"
    attack_shim_text = (adversarial_root / "attack.py").read_text()
    attack_text = (learning_adversarial_root / "attack.py").read_text()
    data_shim_text = (adversarial_root / "data.py").read_text()
    data_text = (learning_adversarial_root / "data.py").read_text()
    common_shim_text = (adversarial_root / "common.py").read_text()
    config_shim_text = (adversarial_root / "config.py").read_text()
    selection_shim_text = (adversarial_root / "selection.py").read_text()
    model_shim_text = (adversarial_root / "model.py").read_text()
    model_text = (models_root / "adversarial.py").read_text()
    evaluation_text = (adversarial_root / "evaluation.py").read_text()
    training_shim_text = (adversarial_root / "training.py").read_text()
    training_text = (learning_root / "training" / "adversarial.py").read_text()
    visualization_shim_text = (adversarial_root / "visualization.py").read_text()
    visualization_text = (viz_root / "adversarial.py").read_text()
    workflow_text = (adversarial_root / "workflow.py").read_text()
    experiment_text = (adversarial_root / "experiment.py").read_text()

    assert "from homopt.learning.adversarial.attack import *" in attack_shim_text
    assert "class PGDAttack" in attack_text
    assert "torch.nn.functional" in attack_text
    assert "from homopt.learning.adversarial.data import *" in data_shim_text
    assert "def load_data" in data_text
    assert "from homopt.learning.adversarial.common import *" in common_shim_text
    assert "from homopt.learning.adversarial.config import *" in config_shim_text
    assert "from homopt.learning.adversarial.selection import *" in selection_shim_text
    assert "from homopt.models.adversarial import *" in model_shim_text
    assert "class AdaptiveCNN" in model_text
    assert "from homopt.learning.training.adversarial import *" in training_shim_text
    assert "def train_adversarial_model" in training_text
    assert "def evaluate_adversarial_attacks" in evaluation_text
    assert "from homopt.viz.adversarial import *" in visualization_shim_text
    assert "def visualize_adversarial_attacks" in visualization_text
    assert "from homopt.learning.training.adversarial import train_adversarial_model" in workflow_text
    assert "from homopt.viz.adversarial import visualize_adversarial_attacks" in workflow_text
    assert "def adversarial_attack_workflow" in workflow_text
    assert "def adversarial_attack_experiment" in experiment_text
    assert "class PGDAttack" not in workflow_text


def test_first_order_optimizers_use_shared_oracle_adapters():
    first_order_root = REPO_ROOT / "src" / "homopt" / "optim" / "first_order"
    core_oracle_text = (REPO_ROOT / "src" / "homopt" / "optim" / "core" / "oracles.py").read_text()
    assert "def _project_problem_point" in core_oracle_text
    assert "def _linear_oracle_point" in core_oracle_text
    assert "from homopt.problems import" in core_oracle_text
    assert "from homopt.solvers import solve_exact_result" in core_oracle_text

    for module_name in ("pgd.py", "frank_wolfe.py", "hom_pgd.py", "radial_dual.py"):
        text = (first_order_root / module_name).read_text()
        assert "from homopt.problems import" not in text
        assert "from homopt.solvers import" not in text
        assert "LagrangianOptimizer" not in text
    assert "_project_problem_point(" in (first_order_root / "pgd.py").read_text()
    assert "_linear_oracle_point(" in (first_order_root / "frank_wolfe.py").read_text()


def test_alm_optimizers_use_explicit_core_helper_imports():
    alm_root = REPO_ROOT / "src" / "homopt" / "optim" / "alm"
    for module_name in ("lagrangian.py", "hom_alm.py", "equality.py"):
        text = (alm_root / module_name).read_text()
        assert "from homopt.optim.core import *" not in text
    assert "from homopt.optim.core.constraints import" in (alm_root / "lagrangian.py").read_text()
    assert "from homopt.optim.core.penalty import" in (alm_root / "hom_alm.py").read_text()
    assert "from homopt.optim.core.solver_cache import" in (alm_root / "equality.py").read_text()


def test_experiments_top_level_has_no_family_private_modules():
    experiments_root = REPO_ROOT / "src" / "homopt" / "experiments"
    forbidden_prefixes = (
        "_adversarial_",
        "_jcc_",
        "_qcqp_",
        "_benchmark_",
        "_convex_",
        "_maxcut_",
        "_toy_",
        "_stiefel_",
    )
    offenders = [
        path.name
        for path in experiments_root.glob("*.py")
        if any(path.name.startswith(prefix) for prefix in forbidden_prefixes)
    ]
    assert offenders == []


def test_src_package_has_no_legacy_family_facades():
    retired_packages = {
        "gauge_eq",
        "gauge_ineq",
        "inn_parametric",
    }
    assert not [path.name for path in (REPO_ROOT / "src" / "homopt").iterdir() if path.name in retired_packages]


def test_viz_has_no_legacy_plotting_modules():
    viz_root = REPO_ROOT / "src" / "homopt" / "viz"
    assert not (viz_root / "legacy.py").exists()
    assert not (viz_root / "plots.py").exists()
    assert not (viz_root / "_coninn_impl.py").exists()
    assert not (viz_root / "_convex2d_impl.py").exists()
    assert not (viz_root / "_mdh_impl.py").exists()

    viz_init = viz_root / "__init__.py"
    text = viz_init.read_text()
    assert "from .legacy import" not in text
    assert "from .plots import" not in text
    assert '"evaluate_coninn_exactness": "coninn"' in text
    assert '"save_convex_2d_visualizations": "convex2d"' in text
    assert '"visualize_mdh_mapping_transformation": "mdh"' in text
    assert "def __getattr__(name):" in text


def test_optimizers_do_not_emit_algorithm_specific_iteration_messages():
    optim_root = REPO_ROOT / "src" / "homopt" / "optim"
    forbidden = (
        "Converged outer loop",
        "Iteration ",
        "Objective:",
    )
    offenders = {}
    for path in optim_root.rglob("*.py"):
        if path.name == "verbose.py":
            continue
        text = path.read_text()
        hits = [snippet for snippet in forbidden if snippet in text]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits

    assert offenders == {}


def test_solver_families_live_in_dedicated_modules():
    solvers_root = REPO_ROOT / "src" / "homopt" / "solvers"
    assert not (solvers_root / "core.py").exists()

    solvers_init = (solvers_root / "__init__.py").read_text()
    common_text = (solvers_root / "common.py").read_text()
    assert not (solvers_root / "cvxpy_backend.py").exists()
    assert not (solvers_root / "pyomo_backend.py").exists()
    cvxpy_text = (solvers_root / "cvxpy.py").read_text()
    pyomo_text = (solvers_root / "pyomo.py").read_text()
    convex_text = (solvers_root / "convex.py").read_text()
    maxcut_text = (solvers_root / "maxcut.py").read_text()
    chance_text = (solvers_root / "chance.py").read_text()
    qcqp_text = (solvers_root / "qcqp.py").read_text()

    assert '"ConvexSolver": ("homopt.solvers.convex", "ConvexSolver")' in solvers_init
    assert '"MaxCutSolver": ("homopt.solvers.maxcut", "MaxCutSolver")' in solvers_init
    assert '"ChanceConstraintSolver": ("homopt.solvers.chance", "ChanceConstraintSolver")' in solvers_init
    assert '"QCQPSolver": ("homopt.solvers.qcqp", "QCQPSolver")' in solvers_init
    assert '"solve_exact_result": ("homopt.solvers.common", "solve_exact_result")' in solvers_init

    assert "def solve_exact_result" in common_text
    assert "def _solve_cvxpy_problem" in cvxpy_text
    assert "PYOMO_NLP_SOLVER" in pyomo_text
    assert "class ConvexSolver" in convex_text
    assert "class MaxCutSolver" in maxcut_text
    assert "class ChanceConstraintSolver" in chance_text
    assert "class QCQPSolver" in qcqp_text


def test_optim_public_dispatch_is_explicit():
    optim_init = REPO_ROOT / "src" / "homopt" / "optim" / "__init__.py"
    text = optim_init.read_text()
    assert '"run_algorithm": ("homopt.optim.dispatch", "run_algorithm")' in text
    assert '"PGDOptimizer": ("homopt.optim.first_order", "PGDOptimizer")' in text
    assert '"HomALMOptimizer": ("homopt.optim.alm", "HomALMOptimizer")' in text
    assert '"LagrangianOptimizer": ("homopt.optim.alm", "LagrangianOptimizer")' in text
    assert '"INNPGDOptimizer": ("homopt.optim.inn", "INNPGDOptimizer")' in text
    assert '"StiefelRetractionOptimizer": ("homopt.optim.manifolds", "StiefelRetractionOptimizer")' in text

    optim_root = REPO_ROOT / "src" / "homopt" / "optim"
    for retired_name in (
        "common.py",
        "eq.py",
        "ineq.py",
        "shared.py",
        "first_order.py",
        "first_order_loop.py",
        "hom_alm.py",
        "lagrangian.py",
        "inn_pgd.py",
        "penalty.py",
        "recording.py",
        "stiefel.py",
        "updates.py",
        "verbose.py",
        "_core_impl.py",
        "_core_helpers.py",
        "_first_order_algorithms.py",
        "_hom_alm_algorithm.py",
        "_lagrangian_algorithms.py",
        "_stiefel_algorithms.py",
        "_defaults.py",
        "_penalty.py",
        "_recording.py",
        "_updates.py",
        "_verbose.py",
    ):
        assert not (optim_root / retired_name).exists()

    core_text = (optim_root / "registry.py").read_text()
    forbidden_core_classes = (
        "class PGDOptimizer",
        "class HomPGDOptimizer",
        "class HomALMOptimizer",
        "class LagrangianOptimizer",
        "class EqualityConstrainedALMOptimizer",
        "class FrankWolfeOptimizer",
        "class RadialDualOptimizer",
    )
    for snippet in forbidden_core_classes:
        assert snippet not in core_text

    assert "def run_algorithm" in core_text
    assert "def _algorithm_specs" in core_text

    assert "class PGDOptimizer" in (optim_root / "first_order" / "pgd.py").read_text()
    assert "class HomPGDOptimizer" in (optim_root / "first_order" / "hom_pgd.py").read_text()
    assert "class FrankWolfeOptimizer" in (optim_root / "first_order" / "frank_wolfe.py").read_text()
    assert "class RadialDualOptimizer" in (optim_root / "first_order" / "radial_dual.py").read_text()
    assert "class LagrangianOptimizer" in (optim_root / "alm" / "lagrangian.py").read_text()
    assert "class EqualityConstrainedALMOptimizer" in (optim_root / "alm" / "equality.py").read_text()
    assert "class HomALMOptimizer" in (optim_root / "alm" / "hom_alm.py").read_text()
    assert "class INNPGDOptimizer" in (optim_root / "inn" / "pgd.py").read_text()
    assert "class StiefelRetractionOptimizer" in (optim_root / "manifolds" / "stiefel.py").read_text()
