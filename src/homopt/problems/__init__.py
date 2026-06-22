"""Problem definitions namespace."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_TARGET = {
    "BMSDP": ("homopt.problems.convex.maxcut", "BMSDP"),
    "BaseProblem": ("homopt.problems.base", "BaseProblem"),
    "ConvexOpt": ("homopt.problems.convex.standard", "ConvexOpt"),
    "ConvexOptEq": ("homopt.problems.convex.standard", "ConvexOptEq"),
    "FixedProblemParametricAdapter": ("homopt.problems.base", "FixedProblemParametricAdapter"),
    "JCCDCOPFProblem": ("homopt.problems.jcc.opf", "JCCDCOPFProblem"),
    "JCCIMProblem": ("homopt.problems.jcc.linear", "JCCIMProblem"),
    "JCCLinearProblem": ("homopt.problems.jcc.linear", "JCCLinearProblem"),
    "LinearProblem": ("homopt.problems.convex.wrappers", "LinearProblem"),
    "load_pglib_opf_case": ("homopt.problems.opf_cases", "load_pglib_opf_case"),
    "MaxCutSDP": ("homopt.problems.convex.maxcut", "MaxCutSDP"),
    "NonConvexQCProblem": ("homopt.problems.qcqp.parametric", "NonConvexQCProblem"),
    "ParametricACOPFProblem": ("homopt.problems.acopf", "ParametricACOPFProblem"),
    "ConvexParametricProblemBase": ("homopt.problems.convex_parametric", "ConvexParametricProblemBase"),
    "ParametricConvexProblem": ("homopt.problems.convex.standard", "ParametricConvexProblem"),
    "ParametricConvexQCQP": ("homopt.problems.convex_parametric", "ParametricConvexQCQP"),
    "ParametricQP": ("homopt.problems.convex_parametric", "ParametricQP"),
    "ParametricSDP": ("homopt.problems.convex_parametric", "ParametricSDP"),
    "ParametricSOCP": ("homopt.problems.convex_parametric", "ParametricSOCP"),
    "PolyStarOpt": ("homopt.problems.convex.star", "PolyStarOpt"),
    "ParametricProblemBase": ("homopt.problems.base", "ParametricProblemBase"),
    "ProblemInstance": ("homopt.problems.base", "ProblemInstance"),
    "ProblemInstanceBatch": ("homopt.problems.base", "ProblemInstanceBatch"),
    "ProjProblem": ("homopt.problems.convex.wrappers", "ProjProblem"),
    "QCOpt": ("homopt.problems.qcqp.deterministic", "QCOpt"),
    "StateBackedParametricProblemBase": ("homopt.problems.base", "StateBackedParametricProblemBase"),
    "StiefelProblem": ("homopt.problems.stiefel.core", "StiefelProblem"),
    "ToyStarOpt": ("homopt.problems.convex.star", "ToyStarOpt"),
    "create_constrained_pca_stiefel_config": ("homopt.problems.stiefel.config", "create_constrained_pca_stiefel_config"),
    "bind_problem_instance": ("homopt.problems.base", "bind_problem_instance"),
    "bind_singleton_problem_instance": ("homopt.problems.base", "bind_singleton_problem_instance"),
    "create_maxcut_problem": ("homopt.problems.convex.maxcut", "create_maxcut_problem"),
    "create_test_QC_problem": ("homopt.problems.qcqp.generators", "create_test_QC_problem"),
    "create_test_problem": ("homopt.problems.convex.generators", "create_test_problem"),
    "default_acopf_dataset_path": ("homopt.problems.acopf", "default_acopf_dataset_path"),
    "normalize_constraint_violation": ("homopt.problems.base", "normalize_constraint_violation"),
    "sample_problem_instance_batch": ("homopt.problems.base", "sample_problem_instance_batch"),
    "resolve_pglib_opf_case_path": ("homopt.problems.opf_cases", "resolve_pglib_opf_case_path"),
    "solve_jcc_dcopt": ("homopt.problems.jcc.opf", "solve_jcc_dcopt"),
}


def __getattr__(name):
    if name not in _SYMBOL_TO_TARGET:
        raise AttributeError(name)
    module_name, attr_name = _SYMBOL_TO_TARGET[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "BMSDP",
    "BaseProblem",
    "ConvexOpt",
    "ConvexOptEq",
    "FixedProblemParametricAdapter",
    "JCCDCOPFProblem",
    "JCCIMProblem",
    "JCCLinearProblem",
    "LinearProblem",
    "load_pglib_opf_case",
    "MaxCutSDP",
    "NonConvexQCProblem",
    "ParametricACOPFProblem",
    "ConvexParametricProblemBase",
    "ParametricConvexProblem",
    "ParametricConvexQCQP",
    "ParametricQP",
    "ParametricSDP",
    "ParametricSOCP",
    "PolyStarOpt",
    "ParametricProblemBase",
    "ProblemInstance",
    "ProblemInstanceBatch",
    "ProjProblem",
    "QCOpt",
    "StateBackedParametricProblemBase",
    "StiefelProblem",
    "ToyStarOpt",
    "create_constrained_pca_stiefel_config",
    "bind_problem_instance",
    "bind_singleton_problem_instance",
    "create_maxcut_problem",
    "create_test_QC_problem",
    "create_test_problem",
    "default_acopf_dataset_path",
    "normalize_constraint_violation",
    "sample_problem_instance_batch",
    "resolve_pglib_opf_case_path",
    "solve_jcc_dcopt",
]
