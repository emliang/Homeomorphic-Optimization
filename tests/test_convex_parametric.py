import torch

from homopt.problems import (
    ConvexParametricProblemBase,
    ParametricConvexQCQP,
    ParametricProblemBase,
    ParametricQP,
    ParametricSDP,
    ParametricSOCP,
    sample_problem_instance_batch,
)


def _assert_partial_completion_satisfies_equalities(problem):
    x = problem.sample_instances(3, seed=17)
    z = torch.zeros(3, problem.partial_dim, dtype=problem.dtype, device=problem.device)
    partial = problem.scale(x, z)
    y = problem.complete_partial(x, partial)
    assert y.shape == (3, problem.nvar)
    if problem.n_eq:
        assert torch.allclose(problem.eq_resid(x, y), torch.zeros_like(problem.eq_resid(x, y)), atol=1e-5)


def _assert_learning_contract(problem):
    assert isinstance(problem, ParametricProblemBase)
    assert isinstance(problem, ConvexParametricProblemBase)
    x = problem.sample_instances(4, seed=19)
    z = torch.zeros(4, problem.partial_dim, dtype=problem.dtype, device=problem.device)
    y = problem.complete_partial(x, problem.scale(x, z))
    obj = problem.objective_xy(x, y)
    cons_raw = problem.constraint_residual_xy(x, y, clip=False)
    cons_clip = problem.constraint_residual_xy(x, y, clip=True)
    assert obj.shape == (4, 1)
    assert cons_raw.shape == cons_clip.shape
    assert cons_raw.shape[0] == 4
    assert torch.all(cons_clip >= 0)
    batch = sample_problem_instance_batch(problem, n_instances=2, seed=23)
    assert len(batch) == 2
    assert tuple(batch.inputs.shape) == (2, problem.xdim)


def test_parametric_qp_contract_and_completion():
    problem = ParametricQP(n_var=6, n_eq=2, n_ineq=3, seed=5)
    _assert_partial_completion_satisfies_equalities(problem)
    _assert_learning_contract(problem)


def test_parametric_convex_qcqp_contract_and_completion():
    problem = ParametricConvexQCQP(n_var=5, n_eq=2, n_ineq=4, seed=7)
    _assert_partial_completion_satisfies_equalities(problem)
    _assert_learning_contract(problem)
    assert torch.all(torch.linalg.eigvalsh(problem.H) >= -1e-8)


def test_parametric_socp_contract_and_completion():
    problem = ParametricSOCP(n_var=5, n_eq=1, n_ineq=3, cone_dim=4, seed=11)
    _assert_partial_completion_satisfies_equalities(problem)
    _assert_learning_contract(problem)


def test_parametric_sdp_contract_completion_and_matrix_roundtrip():
    problem = ParametricSDP(matrix_dim=3, n_eq=2, seed=13)
    _assert_partial_completion_satisfies_equalities(problem)
    _assert_learning_contract(problem)
    matrix = torch.eye(3, dtype=problem.dtype).unsqueeze(0)
    lower = problem.matrix_to_lower_vector(matrix)
    recovered = problem.lower_vector_to_matrix(lower)
    assert torch.allclose(recovered, matrix)


def test_convex_parametric_scaling_roundtrip_and_dtype_runtime():
    problem = ParametricQP(n_var=4, n_eq=1, n_ineq=2, seed=17).with_runtime(dtype=torch.float64)
    x = problem.sample_instances(2, seed=29)
    z = torch.tensor([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.25]], dtype=torch.float64)
    scaled = problem.scale(x, z)
    recovered = problem.inverse_scale(x, scaled)
    assert scaled.dtype == torch.float64
    assert torch.allclose(recovered, z)
