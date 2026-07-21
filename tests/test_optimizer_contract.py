from homopt.optim.base import BaseOptimizer
from homopt.optim.core import OptimizerRunResult, ParametricOptimizerRunResult, _normalize_optimizer_result

import torch


class _ModernOptimizer(BaseOptimizer):
    def optimize(self, initial_point=None, verbose=False, seed=2025):
        return {"x": initial_point, "seed": seed, "verbose": verbose}


def test_base_optimizer_subclass_runs_optimize_contract():
    optimizer = _ModernOptimizer()
    result = optimizer.optimize(initial_point=3, seed=7)
    assert result["x"] == 3
    assert result["seed"] == 7
    assert result["verbose"] is False


def _sample_loop_result_tuple():
    final_decision = torch.tensor([1.0, 2.0])
    decision_trajectory = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    objective_trajectory = torch.tensor([3.0, 2.0])
    violation_trajectory = torch.tensor([1.0, 0.0])
    per_iter_time = [0.1, 0.2]
    return (
        final_decision,
        decision_trajectory,
        objective_trajectory,
        violation_trajectory,
        per_iter_time,
    )


def test_normalize_optimizer_result_accepts_canonical_result():
    final_decision, decision_trajectory, objective_trajectory, violation_trajectory, per_iter_time = (
        _sample_loop_result_tuple()
    )
    run_result = OptimizerRunResult(
        final_decision=final_decision,
        decision_trajectory=decision_trajectory,
        objective_trajectory=objective_trajectory,
        violation_trajectory=violation_trajectory,
        per_iter_time=per_iter_time,
        latent_trajectory=[[1.0, 0.0]],
        extra_metrics={"inner_iter_time": [0.01]},
    )

    result = _normalize_optimizer_result(
        run_result,
        optimizer=object(),
        extra_metrics={"total_wall_time": 0.3},
    )

    assert result["z_traj"] == [[1.0, 0.0]]
    assert result["inner_iter_time"] == [0.01]
    assert result["total_wall_time"] == 0.3


def test_canonical_result_preserves_short_term_tuple_unpack_contract():
    final_decision, decision_trajectory, objective_trajectory, violation_trajectory, per_iter_time = (
        _sample_loop_result_tuple()
    )
    plain_result = OptimizerRunResult(
        final_decision=final_decision,
        decision_trajectory=decision_trajectory,
        objective_trajectory=objective_trajectory,
        violation_trajectory=violation_trajectory,
        per_iter_time=per_iter_time,
    )
    assert len(tuple(plain_result)) == 5

    transform_result = OptimizerRunResult(
        final_decision=final_decision,
        decision_trajectory=decision_trajectory,
        objective_trajectory=objective_trajectory,
        violation_trajectory=violation_trajectory,
        per_iter_time=per_iter_time,
        last_trans_time=0.25,
        include_transform_time=True,
    )
    assert tuple(transform_result)[-1] == 0.25
    assert len(tuple(transform_result)) == 6


def test_parametric_result_preserves_inn_pgd_unpack_contract():
    final_decision, decision_trajectory, objective_trajectory, violation_trajectory, per_iter_time = (
        _sample_loop_result_tuple()
    )
    latent_trajectory = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    result = ParametricOptimizerRunResult(
        final_decision=final_decision,
        decision_trajectory=decision_trajectory,
        latent_trajectory=latent_trajectory,
        objective_trajectory=objective_trajectory,
        violation_trajectory=violation_trajectory,
        per_iter_time=per_iter_time,
    )

    unpacked = tuple(result)
    assert unpacked[0] is final_decision
    assert unpacked[1] is decision_trajectory
    assert unpacked[2] is latent_trajectory
    assert unpacked[3] is objective_trajectory
    assert unpacked[4] is violation_trajectory
    assert unpacked[5] is per_iter_time


def test_normalize_optimizer_result_rejects_tuple_outputs():
    try:
        _normalize_optimizer_result((1, 2, 3), optimizer=object())
    except TypeError as exc:
        assert "OptimizerRunResult" in str(exc)
    else:
        raise AssertionError("optimizer tuple output should fail explicitly")
