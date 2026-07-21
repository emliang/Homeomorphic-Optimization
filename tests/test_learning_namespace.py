import pytest
import torch

from homopt.learning import (
    BasePredictor,
    BaseRefiner,
    BisectionConfig,
    BisectionResult,
    ConstantDecisionPredictor,
    ConstantPredictor,
    DiffProjectionRefiner,
    ExactSolverProjectionRefiner,
    HomeomorphicProjectionRefiner,
    IdentityRefiner,
    IPNNBisectionRefiner,
    NeuralDecisionPredictor,
    PredictionResult,
    ProjectionRefiner,
    RayBisectionRefiner,
    InitializedOptSolverRefiner,
    bisect_segment,
)
from homopt.learning.bisection import homeomorphic_bisection
from homopt.learning.training.predictor import _solver_targets


def test_learning_namespace_exports_basic_surface():
    assert BasePredictor is not None
    assert BaseRefiner is not None
    assert BisectionConfig is not None
    assert BisectionResult is not None
    assert ConstantDecisionPredictor is not None
    assert ConstantPredictor is not None
    assert DiffProjectionRefiner is not None
    assert ExactSolverProjectionRefiner is not None
    assert HomeomorphicProjectionRefiner is not None
    assert IdentityRefiner is not None
    assert IPNNBisectionRefiner is not None
    assert NeuralDecisionPredictor is not None
    assert PredictionResult is not None
    assert ProjectionRefiner is not None
    assert RayBisectionRefiner is not None
    assert InitializedOptSolverRefiner is not None
    assert bisect_segment is not None


def test_bisect_segment_supports_single_and_multiple_anchors():
    target = torch.tensor([[2.0], [3.0]])
    anchor = torch.zeros_like(target)

    result = bisect_segment(
        anchor=anchor,
        target=target,
        decode_to_y=lambda coord: coord,
        violation=lambda y: y - 1.0,
        config=BisectionConfig(max_steps=20, feasibility_tol=1e-6, convergence_tol=0.0),
    )
    assert torch.all(result.point <= 1.0 + 1e-5)
    assert torch.allclose(result.anchor, anchor)

    multi_anchor = torch.tensor([[[0.0], [0.5]], [[0.0], [0.25]]])
    result_multi = bisect_segment(
        anchor=multi_anchor,
        target=target,
        decode_to_y=lambda coord: coord,
        violation=lambda y: y - 1.0,
        config=BisectionConfig(max_steps=20, feasibility_tol=1e-6, convergence_tol=0.0),
    )
    assert torch.all(result_multi.point <= 1.0 + 1e-5)
    assert torch.allclose(result_multi.anchor, torch.tensor([[0.5], [0.25]]))


def test_bisect_segment_marks_an_infeasible_midpoint_as_infeasible():
    result = bisect_segment(
        anchor=torch.zeros(1, 1),
        target=torch.ones(1, 1),
        decode_to_y=lambda coord: coord,
        violation=lambda y: y - 0.5,
        config=BisectionConfig(
            max_steps=1,
            feasibility_tol=1e-6,
            convergence_tol=0.0,
            final_alpha="midpoint",
        ),
    )

    assert result.point.item() == pytest.approx(0.75)
    assert result.feasible_mask.item() is False


class _IdentityHomeomorphicModel(torch.nn.Module):
    def forward(self, z, input_params):
        del input_params
        return z


class _HalfspaceBisectionData:
    def scale(self, input_params, y):
        del input_params
        return y

    def complete_partial(self, input_params, y):
        del input_params
        return y

    def check_feasibility(self, input_params, y):
        del input_params
        return y - 0.5


def test_homeomorphic_bisection_returns_the_feasible_lower_endpoint():
    point, steps = homeomorphic_bisection(
        _IdentityHomeomorphicModel(),
        _HalfspaceBisectionData(),
        torch.ones(1, 1),
        torch.zeros(1, 1),
        {"proj_max_steps": 1, "proj_eps": 1e-6, "step_size": 0.5},
        eps_converge=0.0,
    )

    assert steps == 1
    assert point.item() == pytest.approx(0.5)
    assert _HalfspaceBisectionData().check_feasibility(torch.zeros(1, 1), point).item() <= 1e-6


def test_homeomorphic_bisection_rejects_an_infeasible_origin():
    class _InfeasibleOriginData(_HalfspaceBisectionData):
        def check_feasibility(self, input_params, y):
            del input_params
            return y + 0.5

    with pytest.raises(ValueError, match="latent origin to be feasible"):
        homeomorphic_bisection(
            _IdentityHomeomorphicModel(),
            _InfeasibleOriginData(),
            torch.ones(1, 1),
            torch.zeros(1, 1),
            {"proj_max_steps": 1, "proj_eps": 1e-6, "step_size": 0.5},
            eps_converge=0.0,
        )


def test_homeomorphic_bisection_accepts_an_explicit_feasible_anchor():
    class _OriginInfeasibleData(_HalfspaceBisectionData):
        def check_feasibility(self, input_params, y):
            del input_params
            return y + 0.5

    data = _OriginInfeasibleData()
    point, _ = homeomorphic_bisection(
        _IdentityHomeomorphicModel(),
        data,
        torch.ones(1, 1),
        torch.zeros(1, 1),
        {"proj_max_steps": 2, "proj_eps": 1e-6, "step_size": 0.5},
        eps_converge=0.0,
        feasible_anchor=-torch.ones(1, 1),
    )

    assert data.check_feasibility(torch.zeros(1, 1), point).item() <= 1e-6


class _DummyBoundProblem:
    prob_para = ("dummy",)


class _DummyFamily:
    def bind_instance(self, x):
        del x
        return _DummyBoundProblem()


def test_solver_target_fallback_only_handles_expected_solver_failures(monkeypatch):
    monkeypatch.setattr("homopt.solvers.QCQPSolver", lambda prob_para: object())

    def _raise_runtime_error(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("solver unavailable")

    monkeypatch.setattr("homopt.solvers.solve_exact_result", _raise_runtime_error)
    input_tensor = torch.randn(2, 3)
    fallback_target = torch.zeros(2, 4)

    targets, success = _solver_targets(
        _DummyFamily(),
        input_tensor,
        solver_name="ipopt",
        solver_options={},
        allow_fallback=True,
        fallback_target=fallback_target,
    )

    assert success == 0
    assert torch.equal(targets, fallback_target)


def test_solver_target_fallback_does_not_swallow_programming_errors(monkeypatch):
    monkeypatch.setattr("homopt.solvers.QCQPSolver", lambda prob_para: object())

    def _raise_key_error(*args, **kwargs):
        del args, kwargs
        raise KeyError("unexpected bug")

    monkeypatch.setattr("homopt.solvers.solve_exact_result", _raise_key_error)

    with pytest.raises(KeyError):
        _solver_targets(
            _DummyFamily(),
            torch.randn(1, 3),
            solver_name="ipopt",
            solver_options={},
            allow_fallback=True,
            fallback_target=torch.zeros(1, 4),
        )
