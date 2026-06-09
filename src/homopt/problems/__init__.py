"""Problem definitions namespace."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_TARGET = {
    "BMSDP": ("homopt.problems._convex_impl", "BMSDP"),
    "BaseProblem": ("homopt.problems.base", "BaseProblem"),
    "ChanceConstraint_Opt": ("homopt.problems._qcqp_impl", "ChanceConstraint_Opt"),
    "ConvexOpt": ("homopt.problems._convex_impl", "ConvexOpt"),
    "ConvexOptEq": ("homopt.problems._convex_impl", "ConvexOptEq"),
    "FixedProblemParametricAdapter": ("homopt.problems.base", "FixedProblemParametricAdapter"),
    "JCCDCOPFProblem": ("homopt.problems._jcc_opf_impl", "JCCDCOPFProblem"),
    "JCCIMProblem": ("homopt.problems._jcc_linear_impl", "JCCIMProblem"),
    "JCCLinearProblem": ("homopt.problems._jcc_linear_impl", "JCCLinearProblem"),
    "LearningProblemAdapter": ("homopt.problems.base", "LearningProblemAdapter"),
    "LinearProblem": ("homopt.problems._convex_impl", "LinearProblem"),
    "MaxCutSDP": ("homopt.problems._convex_impl", "MaxCutSDP"),
    "NonConvexQCLearningProblem": ("homopt.problems._qcqp_impl", "NonConvexQCLearningProblem"),
    "NonConvexQCProblem": ("homopt.problems._qcqp_impl", "NonConvexQCProblem"),
    "ParametricConvexProblem": ("homopt.problems._convex_impl", "ParametricConvexProblem"),
    "PolyStarOpt": ("homopt.problems._convex_impl", "PolyStarOpt"),
    "ParametricProblemBase": ("homopt.problems.base", "ParametricProblemBase"),
    "ProblemAdapter": ("homopt.problems.base", "ProblemAdapter"),
    "ProblemInstance": ("homopt.problems.base", "ProblemInstance"),
    "ProblemInstanceBatch": ("homopt.problems.base", "ProblemInstanceBatch"),
    "ProjProblem": ("homopt.problems._convex_impl", "ProjProblem"),
    "QCOpt": ("homopt.problems._qcqp_impl", "QCOpt"),
    "StateBackedParametricProblemBase": ("homopt.problems.base", "StateBackedParametricProblemBase"),
    "StiefelProblem": ("homopt.problems._stiefel_impl", "StiefelProblem"),
    "ToyStarOpt": ("homopt.problems._convex_impl", "ToyStarOpt"),
    "create_constrained_pca_stiefel_config": ("homopt.problems._stiefel_impl", "create_constrained_pca_stiefel_config"),
    "as_learning_problem": ("homopt.problems.base", "as_learning_problem"),
    "as_problem": ("homopt.problems.base", "as_problem"),
    "bind_problem_instance": ("homopt.problems.base", "bind_problem_instance"),
    "bind_singleton_problem_instance": ("homopt.problems.base", "bind_singleton_problem_instance"),
    "create_maxcut_problem": ("homopt.problems._convex_impl", "create_maxcut_problem"),
    "create_test_QC_problem": ("homopt.problems._qcqp_impl", "create_test_QC_problem"),
    "create_test_problem": ("homopt.problems._convex_impl", "create_test_problem"),
    "normalize_constraint_violation": ("homopt.problems.base", "normalize_constraint_violation"),
    "sample_problem_instance_batch": ("homopt.problems.base", "sample_problem_instance_batch"),
    "solve_jcc_dcopt": ("homopt.problems._jcc_opf_impl", "solve_jcc_dcopt"),
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
    "ChanceConstraint_Opt",
    "ConvexOpt",
    "ConvexOptEq",
    "FixedProblemParametricAdapter",
    "JCCDCOPFProblem",
    "JCCIMProblem",
    "JCCLinearProblem",
    "LearningProblemAdapter",
    "LinearProblem",
    "MaxCutSDP",
    "NonConvexQCLearningProblem",
    "NonConvexQCProblem",
    "ParametricConvexProblem",
    "PolyStarOpt",
    "ParametricProblemBase",
    "ProblemAdapter",
    "ProblemInstance",
    "ProblemInstanceBatch",
    "ProjProblem",
    "QCOpt",
    "StateBackedParametricProblemBase",
    "StiefelProblem",
    "ToyStarOpt",
    "create_constrained_pca_stiefel_config",
    "as_learning_problem",
    "as_problem",
    "bind_problem_instance",
    "bind_singleton_problem_instance",
    "create_maxcut_problem",
    "create_test_QC_problem",
    "create_test_problem",
    "normalize_constraint_violation",
    "sample_problem_instance_batch",
    "solve_jcc_dcopt",
]
