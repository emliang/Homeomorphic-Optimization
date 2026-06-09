"""Problem definitions namespace."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_TARGET = {
    "BMSDP": ("homopt.problems._convex_impl", "BMSDP"),
    "BaseProblem": ("homopt.problems.base", "BaseProblem"),
    "ConvexOpt": ("homopt.problems._convex_impl", "ConvexOpt"),
    "ConvexOptEq": ("homopt.problems._convex_impl", "ConvexOptEq"),
    "FixedProblemParametricAdapter": ("homopt.problems.base", "FixedProblemParametricAdapter"),
    "LearningProblemAdapter": ("homopt.problems.base", "LearningProblemAdapter"),
    "LinearProblem": ("homopt.problems._convex_impl", "LinearProblem"),
    "MaxCutSDP": ("homopt.problems._convex_impl", "MaxCutSDP"),
    "ParametricConvexProblem": ("homopt.problems._convex_impl", "ParametricConvexProblem"),
    "PolyStarOpt": ("homopt.problems._convex_impl", "PolyStarOpt"),
    "ParametricProblemBase": ("homopt.problems.base", "ParametricProblemBase"),
    "ProblemAdapter": ("homopt.problems.base", "ProblemAdapter"),
    "ProblemInstance": ("homopt.problems.base", "ProblemInstance"),
    "ProblemInstanceBatch": ("homopt.problems.base", "ProblemInstanceBatch"),
    "ProjProblem": ("homopt.problems._convex_impl", "ProjProblem"),
    "StateBackedParametricProblemBase": ("homopt.problems.base", "StateBackedParametricProblemBase"),
    "ToyStarOpt": ("homopt.problems._convex_impl", "ToyStarOpt"),
    "as_learning_problem": ("homopt.problems.base", "as_learning_problem"),
    "as_problem": ("homopt.problems.base", "as_problem"),
    "bind_problem_instance": ("homopt.problems.base", "bind_problem_instance"),
    "bind_singleton_problem_instance": ("homopt.problems.base", "bind_singleton_problem_instance"),
    "create_maxcut_problem": ("homopt.problems._convex_impl", "create_maxcut_problem"),
    "create_test_problem": ("homopt.problems._convex_impl", "create_test_problem"),
    "normalize_constraint_violation": ("homopt.problems.base", "normalize_constraint_violation"),
    "sample_problem_instance_batch": ("homopt.problems.base", "sample_problem_instance_batch"),
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
    "LearningProblemAdapter",
    "LinearProblem",
    "MaxCutSDP",
    "ParametricConvexProblem",
    "PolyStarOpt",
    "ParametricProblemBase",
    "ProblemAdapter",
    "ProblemInstance",
    "ProblemInstanceBatch",
    "ProjProblem",
    "StateBackedParametricProblemBase",
    "ToyStarOpt",
    "as_learning_problem",
    "as_problem",
    "bind_problem_instance",
    "bind_singleton_problem_instance",
    "create_maxcut_problem",
    "create_test_problem",
    "normalize_constraint_violation",
    "sample_problem_instance_batch",
]
