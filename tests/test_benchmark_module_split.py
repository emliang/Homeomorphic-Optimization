from homopt.experiments import hom_alm, hom_pgd, inn_pgd, learning_postprocess
from homopt.experiments.benchmarks import (
    acopf_parametric,
    convex,
    convex_parametric,
    jcc,
    maxcut,
    parametric_registry,
    qcqp,
    star,
)
from homopt.experiments.common import single_problem


def test_single_problem_benchmark_module_matches_family_exports():
    assert "convex_algorithm_comparison" in single_problem.SINGLE_PROBLEM_BENCHMARKS
    assert convex.convex_algorithm_comparison is hom_alm.convex_algorithm_comparison
    assert convex.socp_hompgd_benchmark is hom_pgd.socp_hompgd_benchmark
    assert star.poly_star_benchmark is hom_pgd.poly_star_benchmark
    assert maxcut.maxcut_algorithm_comparison is hom_pgd.maxcut_algorithm_comparison


def test_parametric_benchmark_module_matches_family_exports():
    assert "qcqp_inn_experiment" in parametric_registry.PARAMETRIC_BENCHMARKS
    assert qcqp.inn_training_benchmark is inn_pgd.inn_training_benchmark
    assert qcqp.qcqp_inn_experiment is inn_pgd.qcqp_inn_experiment
    assert not hasattr(learning_postprocess, "qcqp_learning_benchmark")
    assert not hasattr(learning_postprocess, "qcqp_route_comparison")
    assert "qcqp_learning_benchmark" not in parametric_registry.PARAMETRIC_BENCHMARKS
    assert "qcqp_route_comparison" not in parametric_registry.PARAMETRIC_BENCHMARKS
    assert acopf_parametric.acopf_parametric_learning_benchmark is learning_postprocess.acopf_parametric_learning_benchmark
    assert convex_parametric.convex_parametric_learning_benchmark is learning_postprocess.convex_parametric_learning_benchmark
    assert jcc.jcc_algorithm_comparison is inn_pgd.jcc_algorithm_comparison
    assert jcc.jcc_baseline_solver_sweep is inn_pgd.jcc_baseline_solver_sweep
    assert jcc.jcc_linear_solver_benchmark is inn_pgd.jcc_linear_solver_benchmark


def test_family_modules_keep_workload_surface():
    assert callable(hom_pgd.adversarial_attack_experiment)
    assert not hasattr(hom_pgd, "adversarial_attack_workflow")
    assert not hasattr(hom_alm, "jcc_algorithm_comparison")
    assert not hasattr(hom_alm, "jcc_baseline_solver_sweep")
    assert callable(inn_pgd.jcc_algorithm_comparison)
    assert callable(inn_pgd.jcc_baseline_solver_sweep)
    assert callable(inn_pgd.qcqp_inn_experiment)
    assert callable(inn_pgd.qcqp_inn_sensitivity_sweep)
    assert callable(learning_postprocess.convex_parametric_learning_benchmark)
    assert callable(learning_postprocess.acopf_parametric_learning_benchmark)
