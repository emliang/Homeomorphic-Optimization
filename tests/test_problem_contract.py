import torch

from homopt.problems.base import (
    BaseProblem,
    FixedProblemParametricAdapter,
    ParametricProblemBase,
    ProblemInstanceBatch,
    StateBackedParametricProblemBase,
    TensorRuntimeMixin,
    bind_problem_instance,
    bind_singleton_problem_instance,
    normalize_constraint_violation,
    sample_problem_instance_batch,
)
from homopt.problems import ConvexOpt, JCCDCOPFProblem, NonConvexQCProblem, ParametricConvexProblem, create_test_problem


class _ModernProblem(BaseProblem):
    nvar = 2

    def objective_x(self, x):
        return x.sum(dim=-1, keepdim=True)

    def constraint_x(self, x, clip=True):
        return x


class _LearningProblem(ParametricProblemBase):
    nvar = 2

    def __init__(self):
        self.vec = torch.tensor([1.0, 2.0], dtype=torch.float32)

    def objective_xy(self, x, y):
        return (x + y).sum(dim=-1, keepdim=True)

    def constraint_residual_xy(self, x, y, clip=True):
        del clip
        return y - x

    def project_xy(self, x, y):
        del x
        return torch.clamp(y, 0.0, 1.0)


class _TensorRuntimeProblem(TensorRuntimeMixin):
    def __init__(self):
        self.vec = torch.tensor([1.0], dtype=torch.float32)


def test_base_problem_direct_contract():
    problem = _ModernProblem()
    x = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    assert problem.objective_x(x).shape == (1, 1)
    assert problem.constraint_x(x, clip=False).shape == (1, 2)


def test_normalize_constraint_violation_shapes():
    assert normalize_constraint_violation(torch.tensor(1.0)).shape == (1, 1)
    assert normalize_constraint_violation(torch.tensor([1.0, 2.0])).shape == (2, 1)


def test_parametric_problem_direct_contract():
    problem = _LearningProblem()
    x = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
    y = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    obj = problem.objective_xy(x, y)
    cons = problem.constraint_residual_xy(x, y, clip=False)
    proj = problem.project_xy(x, y)
    assert obj.shape == (1, 1)
    assert cons.shape == (1, 2)
    assert torch.allclose(proj, torch.tensor([[1.0, 0.0]], dtype=torch.float32))


def test_parametric_problem_runtime_and_feasibility():
    problem = _LearningProblem().with_runtime(dtype=torch.float64)
    assert problem.vec.dtype == torch.float64
    x = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    assert bool(problem.is_feasible_xy(x, y))


def test_tensor_runtime_mixin_normalizes_device_and_returns_self():
    problem = _TensorRuntimeProblem()
    returned = problem.to_device("cpu")
    assert returned is problem
    assert problem.device == torch.device("cpu")
    assert problem.vec.device == torch.device("cpu")


def _qcqp_problem_args():
    return {
        "n_var": 3,
        "n_qua_cons": 2,
        "n_linear_cons": 1,
        "obj": "norm",
        "nonconvex_ratio": 0.5,
        "x_lower": -1.0,
        "x_upper": 1.0,
        "seed": 7,
    }


def test_nonconvex_qc_problem_directly_supports_learning_adapter():
    problem = NonConvexQCProblem(_qcqp_problem_args())
    batch = problem.sample_instance_batch(n_instances=2, seed=11, sample_obj=False)
    input_samples = batch.inputs
    y = torch.zeros(2, problem.nvar, dtype=input_samples.dtype)
    obj = problem.objective_xy(input_samples, y)
    cons = problem.constraint_residual_xy(input_samples, y, clip=False)
    assert isinstance(problem, ParametricProblemBase)
    assert problem.is_parametric
    assert isinstance(batch, ProblemInstanceBatch)
    assert len(batch) == 2
    assert obj.shape == (2, 1)
    assert cons.shape[0] == 2


def test_parametric_convex_problem_wraps_fixed_convex_family_for_learning_route():
    config = create_test_problem({
        "seed": 5,
        "n_var": 2,
        "n_linear_cons": 1,
        "n_soc_cons": 0,
        "n_qua_cons": 0,
        "obj": "quad",
        "x_lower": -1,
        "x_upper": 1,
    })
    base_problem = ConvexOpt(config)
    wrapped = ParametricConvexProblem(base_problem)
    x = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
    y = torch.zeros(1, base_problem.nvar, dtype=torch.float32)
    assert wrapped.is_parametric
    assert torch.allclose(wrapped.objective_xy(x, y), base_problem.objective_x(y))
    assert torch.allclose(
        wrapped.constraint_residual_xy(x, y, clip=False),
        base_problem.constraint_x(y, clip=False),
    )
    assert isinstance(wrapped, FixedProblemParametricAdapter)


def test_parametric_convex_problem_exposes_singleton_instance_batch():
    config = create_test_problem({
        "seed": 5,
        "n_var": 2,
        "n_linear_cons": 1,
        "n_soc_cons": 0,
        "n_qua_cons": 0,
        "obj": "quad",
        "x_lower": -1,
        "x_upper": 1,
    })
    wrapped = ParametricConvexProblem(ConvexOpt(config))
    batch = wrapped.sample_instance_batch(n_instances=4, seed=17)
    assert isinstance(batch, ProblemInstanceBatch)
    assert len(batch) == 1
    assert batch.inputs is None
    assert batch.metadata["deterministic"] is True


def test_problem_instance_helpers_preserve_deterministic_binding_semantics():
    config = create_test_problem({
        "seed": 5,
        "n_var": 2,
        "n_linear_cons": 1,
        "n_soc_cons": 0,
        "n_qua_cons": 0,
        "obj": "quad",
        "x_lower": -1,
        "x_upper": 1,
    })
    wrapped = ParametricConvexProblem(ConvexOpt(config))
    batch = sample_problem_instance_batch(wrapped, n_instances=3, seed=17)
    bound = bind_singleton_problem_instance(wrapped, seed=17)
    assert isinstance(batch, ProblemInstanceBatch)
    assert len(batch) == 1
    assert batch.metadata["deterministic"] is True
    assert isinstance(bound, ConvexOpt)
    assert bind_problem_instance(wrapped, batch.get_instance(0)) is bound


def test_jcc_problem_now_exposes_parametric_problem_contract():
    problem = JCCDCOPFProblem(num_bus=30, config={"n_scenarios": 2, "seed": 9}).to_device(torch.device("cpu"))
    x = torch.zeros(1, problem.nvar, dtype=torch.float32)
    dummy_input = torch.zeros(1, 1, dtype=torch.float32)
    assert isinstance(problem, ParametricProblemBase)
    assert isinstance(problem, StateBackedParametricProblemBase)
    assert torch.allclose(problem.objective_xy(dummy_input, x), problem.objective_x(x))
    assert torch.allclose(
        problem.constraint_residual_xy(dummy_input, x, clip=False),
        problem.constraint_x(x, clip=False),
    )


def test_jcc_problem_exposes_scenario_instance_batch():
    problem = JCCDCOPFProblem(num_bus=30, config={"n_scenarios": 2, "seed": 9}).to_device(torch.device("cpu"))
    batch = problem.sample_instance_batch(n_instances=3, seed=17)
    assert isinstance(batch, ProblemInstanceBatch)
    assert len(batch) == 3
    assert tuple(batch.inputs.shape) == (3, problem.n_scenarios, problem.n_bus)
    assert batch.metadata["n_scenarios"] == problem.n_scenarios
    assert batch.metadata["num_bus"] == problem.n_bus


def test_jcc_problem_accepts_explicit_scenario_batch():
    problem = JCCDCOPFProblem(num_bus=30, config={"n_scenarios": 2, "seed": 9}).to_device(torch.device("cpu"))
    x = torch.zeros(1, problem.nvar, dtype=torch.float32)
    explicit = problem.demand_scenarios.unsqueeze(0)
    assert torch.allclose(problem.objective_xy(explicit, x), problem.objective_x(x))
    assert torch.allclose(
        problem.constraint_residual_xy(explicit, x, clip=False),
        problem.constraint_x(x, clip=False),
    )


def test_problem_instance_helpers_bind_state_backed_problem_without_mutating_family():
    problem = JCCDCOPFProblem(num_bus=30, config={"n_scenarios": 2, "seed": 9}).to_device(torch.device("cpu"))
    batch = sample_problem_instance_batch(problem, n_instances=2, seed=17)
    bound = bind_problem_instance(problem, batch.get_instance(0))
    singleton = bind_singleton_problem_instance(problem, seed=17)
    assert isinstance(batch, ProblemInstanceBatch)
    assert len(batch) == 2
    assert bound is not problem
    assert singleton is not problem
    assert tuple(bound.demand_scenarios.shape) == (problem.n_scenarios, problem.n_bus)
    assert tuple(singleton.demand_scenarios.shape) == (problem.n_scenarios, problem.n_bus)
