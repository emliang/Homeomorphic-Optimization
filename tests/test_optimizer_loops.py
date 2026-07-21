import torch

from homopt.optim import INNPGDOptimizer
import pytest

import homopt.optim.core.oracles as oracle_helpers
import homopt.optim.alm.hom_alm as hom_alm_algorithm
import homopt.optim.alm.equality as equality_algorithms
import homopt.optim.alm.lagrangian as lagrangian_algorithms
import homopt.optim.alm.penalty_loop as penalty_loop
from homopt.optim.first_order.loop import run_first_order_loop
from homopt.optim import EqualityConstrainedALMOptimizer, FrankWolfeOptimizer, HomALMOptimizer, HomPGDOptimizer, LagrangianOptimizer, PGDOptimizer, run_algorithm
from homopt.problems import ConvexOpt
from homopt.optim.core.config import (
    _resolve_inner_stopping_config,
    _validate_gradient_method,
)
from homopt.optim.core.constraints import (
    _constraint_residual,
    _constraint_violation,
)
from homopt.optim.core.penalty import (
    _resolve_penalty_inner_solver_config,
    _should_increase_penalty,
    _update_penalty_outer_learning_rate,
)
from homopt.optim.core.recording import IterationRecorder
from homopt.optim.core.verbose import format_value


class _IdentityHomMap:
    p_norm = 2

    def forward(self, z, method="autograd", return_state=False):
        del method
        if return_state:
            return z, None
        return z

    def inverse(self, x):
        return x

    def vjp(self, z, grad_x, method="autograd", state=None):
        del z, method, state
        return grad_x


class _SimpleProblem:
    def __init__(self):
        self.nvar = 2
        self.device = torch.device("cpu")
        self.logged_hom_map_methods = []

    def objective_x(self, x):
        return (x**2).sum(dim=1)

    def constraint_x(self, x, eq_cons=False, clip=True):
        del x, eq_cons, clip
        return -torch.ones(1, 1)

    def gradient_objective_x(self, x):
        return 2 * x

    def gradient_objective_z(self, z, hom_map, hom_map_method="autograd"):
        del hom_map
        self.logged_hom_map_methods.append(hom_map_method)
        return 2 * z


class _StateAwareProblem(_SimpleProblem):
    def __init__(self):
        super().__init__()
        self.seen_x = None
        self.seen_hom_state = None

    def gradient_objective_z(self, z, hom_map, method="autograd", hom_map_method="autograd", x=None, hom_state=None):
        del hom_map, method
        self.logged_hom_map_methods.append(hom_map_method)
        self.seen_x = x
        self.seen_hom_state = hom_state
        return 2 * z


class _StateTrackingHomMap(_IdentityHomMap):
    def forward(self, z, method="autograd", return_state=False):
        del method
        state = {"marker": torch.ones_like(z)}
        if return_state:
            return z, state
        return z


def test_penalty_outer_loop_disables_progress_when_not_verbose(monkeypatch):
    captured = {}

    def fake_tqdm(iterable, **kwargs):
        captured["disable"] = kwargs.get("disable")
        return iterable

    monkeypatch.setattr(penalty_loop, "tqdm", fake_tqdm)
    initial_state = torch.zeros(1, 1)
    initial_eval = {
        "objective": torch.tensor([1.0]),
        "violation": torch.tensor([1.0]),
        "decision": initial_state,
    }
    initial_dual = torch.zeros(1, 1)

    penalty_loop.run_penalty_outer_loop(
        initial_state=initial_state,
        initial_eval=initial_eval,
        initial_dual_state=initial_dual,
        run_inner_loop=lambda state, dual_state, lr, outer_iter: (state, None),
        evaluate_state=lambda state: {
            "objective": torch.tensor([0.0]),
            "violation": torch.tensor([0.0]),
            "decision": state,
        },
        update_dual_state=lambda dual_state, state, outer_iter: dual_state,
        update_learning_rate=lambda current_lr, recorder, outer_iter: current_lr,
        should_stop=lambda eval_payload, error, outer_iter: True,
        outer_iterations=1,
        max_running_time=1.0,
        initial_learning_rate=1.0,
        outer_stepsize_rule="constant",
        verbose=False,
        progress_disable=False,
    )

    assert captured["disable"] is True


def _subproblem_params(**overrides):
    params = {
        "learning_rate": 1e-3,
        "outer_iterations": 1,
        "inner_iterations": 1,
        "max_running_time": 10,
        "convergence_threshold": 1e-6,
        "stepsize_rule": "constant",
        "outer_stepsize_rule": "constant",
        "inner_stepsize_rule": "constant",
        "lr_decay": 0.9,
        "min_lr": 1e-6,
        "dual_learning_rate": 1e-2,
        "penalty_coef": 10.0,
        "penalty_growth": 1.1,
        "proximal_coef": 0.1,
        "proximal_space": "x",
        "max_penalty": 1e2,
        "max_dual": 1e2,
        "use_lagrangian": True,
        "use_penalty": True,
        "use_proximal": False,
        "momentum": 0.0,
        "opt": "gd",
    }
    params.update(overrides)
    return params


class _FakeFWConvexProblem(ConvexOpt):
    def __init__(self):
        self.nvar = 2
        self.device = torch.device("cpu")

    def objective_x(self, x):
        return (x**2).sum(dim=1, keepdim=True)

    def gradient_objective_x(self, x):
        return 2 * x

    def constraint_x(self, x, eq_cons=False, clip=True):
        del x, eq_cons, clip
        return torch.zeros(1, 1)


class _IdentityModel(torch.nn.Module):
    def forward(self, z, input_params):
        del input_params
        return z


class _INNProblem:
    def __init__(self):
        self.nvar = 2
        self.device = torch.device("cpu")

    def scale(self, input_params, x):
        del input_params
        return x

    def complete_partial(self, input_params, x):
        del input_params
        return x

    def objective(self, x):
        return (x**2).sum(dim=1)

    def violations(self, input_params, x):
        del input_params, x
        return torch.zeros(1)


class _ViolatingINNProblem(_INNProblem):
    def violations(self, input_params, x):
        del input_params, x
        return torch.ones(1)


class _PenaltyProblem:
    def __init__(self):
        self.nvar = 2
        self.device = torch.device("cpu")
        self.ncon = 1
        self.n_eq = 0
        self.ineq_cons = None
        self.eq_cons = None

    def objective_x(self, x):
        return (x**2).sum(dim=1)

    def constraint_x(self, x, eq_cons=False, clip=True):
        del x, eq_cons, clip
        return torch.zeros(1, 1)

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        del dual_var, penalty_coef, proximal_coef, x_outer
        return 2 * x


class _DoubleMatrixPenaltyProblem:
    def __init__(self):
        self.nvar = 2
        self.device = torch.device("cpu")
        self.ncon = 1
        self.n_eq = 0
        self.ineq_cons = None
        self.eq_cons = None
        self.Q = torch.eye(2, dtype=torch.float64)
        self.p = torch.zeros(2, dtype=torch.float64)

    def objective_x(self, x):
        return torch.sum(0.5 * (x @ self.Q) * x + self.p * x, dim=-1, keepdim=True)

    def constraint_x(self, x, eq_cons=False, clip=True):
        del eq_cons, clip
        return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        del dual_var, penalty_coef, proximal_coef, x_outer
        return x @ self.Q + self.p


class _HomPenaltyProblem(_PenaltyProblem):
    def __init__(self):
        super().__init__()
        self.n_eq = 1
        self.ineq_cons = range(self.ncon)
        self.eq_cons = range(self.ncon, self.ncon + self.n_eq)
        self.logged_dual_vars = []
        self.logged_proximal_spaces = []
        self.logged_x_outer = []

    def eq_constraint_x(self, x):
        del x
        return torch.ones(1, 1)

    def gradient_lagrangian_z(
        self,
        z,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        hom_map=None,
        z_outer=None,
        x_outer=None,
        proximal_space="z",
        hom_map_method="autograd",
    ):
        del penalty_coef, proximal_coef, hom_map, z_outer, hom_map_method
        self.logged_dual_vars.append(dual_var.detach().clone())
        self.logged_proximal_spaces.append(proximal_space)
        self.logged_x_outer.append(None if x_outer is None else x_outer.detach().clone())
        return torch.zeros_like(z)


class _MixedConstraintProblem:
    def __init__(self):
        self.nvar = 2
        self.device = torch.device("cpu")
        self.ncon = 2
        self.n_eq = 1
        self.ineq_cons = range(0, 2)
        self.eq_cons = range(2, 3)

    def constraint_x(self, x, clip=False, eq_cons=True):
        del x, clip, eq_cons
        return torch.tensor([[-5.0, 0.25, -0.1]])

    def eq_constraint_x(self, x):
        del x
        return torch.tensor([[-0.1]])


class _AdaptiveInnerHomPenaltyProblem(_HomPenaltyProblem):
    def eq_constraint_x(self, x):
        return 3 * x[:, :1] + 1

    def gradient_lagrangian_z(
        self,
        z,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        hom_map=None,
        z_outer=None,
        x_outer=None,
        proximal_space="z",
        hom_map_method="autograd",
    ):
        del dual_var, penalty_coef, proximal_coef, hom_map, z_outer, x_outer, proximal_space, hom_map_method
        return torch.ones_like(z)


class _ConstantGradientHomPenaltyProblem(_HomPenaltyProblem):
    def eq_constraint_x(self, x):
        return torch.ones(1, 1, device=x.device)

    def gradient_lagrangian_z(
        self,
        z,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        hom_map=None,
        z_outer=None,
        x_outer=None,
        proximal_space="z",
        hom_map_method="autograd",
    ):
        del dual_var, penalty_coef, proximal_coef, hom_map, z_outer, x_outer, proximal_space, hom_map_method
        return torch.ones_like(z)


class _FeasibleNonstationaryHomPenaltyProblem(_HomPenaltyProblem):
    def eq_constraint_x(self, x):
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

    def gradient_lagrangian_z(
        self,
        z,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        hom_map=None,
        z_outer=None,
        x_outer=None,
        proximal_space="z",
        hom_map_method="autograd",
    ):
        del dual_var, penalty_coef, proximal_coef, hom_map, z_outer, x_outer, proximal_space, hom_map_method
        return torch.ones_like(z)


class _ConstantGradientPenaltyProblem(_PenaltyProblem):
    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        del dual_var, penalty_coef, proximal_coef, x_outer
        return torch.ones_like(x)


class _AdaptiveInnerPenaltyProblem(_PenaltyProblem):
    def constraint_x(self, x, eq_cons=False, clip=True):
        del eq_cons, clip
        return 4 * x[:, :1] + 2

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        del x, dual_var, penalty_coef, proximal_coef, x_outer
        return torch.ones(1, 2)


class _FeasibleAugmentedStationaryPenaltyProblem(_PenaltyProblem):
    def constraint_x(self, x, eq_cons=False, clip=True):
        del eq_cons, clip
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        del dual_var, proximal_coef, x_outer
        if penalty_coef is None:
            return torch.ones_like(x)
        return torch.zeros_like(x)


class _DecreasingEqHomPenaltyProblem(_HomPenaltyProblem):
    def eq_constraint_x(self, x):
        return x[:, :1]

    def gradient_lagrangian_z(
        self,
        z,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        hom_map=None,
        z_outer=None,
        x_outer=None,
        proximal_space="z",
        hom_map_method="autograd",
    ):
        del dual_var, penalty_coef, proximal_coef, hom_map, z_outer, x_outer, proximal_space, hom_map_method
        return z


class _EqOnlyPenaltyProblem:
    def __init__(self):
        self.nvar = 1
        self.device = torch.device("cpu")
        self.ncon = 0
        self.n_eq = 1
        self.ineq_cons = None
        self.eq_cons = range(1)

    def objective_x(self, x):
        return (x**2).sum(dim=1, keepdim=True)

    def eq_constraint_x(self, x):
        return x[:, :1]

    def constraint_x(self, x, clip=False, eq_cons=True):
        del clip, eq_cons
        return self.eq_constraint_x(x)


class _MixedEqPenaltyProblem(_EqOnlyPenaltyProblem):
    def __init__(self):
        super().__init__()
        self.ncon = 2
        self.n_eq = 1
        self.ineq_cons = range(2)
        self.eq_cons = range(2, 3)

    def constraint_x(self, x, clip=False, eq_cons=True):
        del clip, eq_cons
        return torch.cat([x[:, :1] - 1.0, -x[:, :1] - 1.0, self.eq_constraint_x(x)], dim=1)


class _ReturnBestPenaltyProblem(_PenaltyProblem):
    def constraint_x(self, x, eq_cons=False, clip=True):
        del eq_cons, clip
        return x[:, :1]

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        del x, dual_var, penalty_coef, proximal_coef, x_outer
        return -torch.ones(1, 2)


def test_constraint_violation_clips_inequalities_and_abs_equalities():
    problem = _MixedConstraintProblem()
    x = torch.zeros(1, 2)

    full_violation = _constraint_violation(problem, x, equality_only=False)
    equality_violation = _constraint_violation(problem, x, equality_only=True)

    assert full_violation.item() == pytest.approx(0.25)
    assert equality_violation.item() == pytest.approx(0.1)


def test_constraint_residual_does_not_swallow_internal_typeerror():
    class ProblemWithInternalTypeError(_MixedConstraintProblem):
        def constraint_x(self, x, clip=False, eq_cons=True):
            del x, clip, eq_cons
            raise TypeError("internal constraint failure")

    with pytest.raises(TypeError, match="internal constraint failure"):
        _constraint_residual(ProblemWithInternalTypeError(), torch.zeros(1, 2))


def test_pgd_optimizer_loop_records_consistent_lengths():
    optimizer = PGDOptimizer(
        _SimpleProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 3,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "acceleration_method": "none",
            "acceleration_space": "x",
            "stepsize_rule": "diminish",
            "opt": "gd",
            "momentum": 0.0,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "projection_subproblem": _subproblem_params(),
        },
    )
    x_opt, decision_traj, objective_traj, violation_traj, per_iter_time = optimizer.optimize(verbose=False, seed=0)
    assert x_opt.shape == (1, 2)
    assert decision_traj.shape[1] == 2
    assert objective_traj.numel() == len(per_iter_time) + 1
    assert violation_traj.shape[0] == len(per_iter_time) + 1


def test_pgd_initial_violation_clips_satisfied_inequalities():
    optimizer = PGDOptimizer(
        _SimpleProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 0,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "acceleration_method": "none",
            "acceleration_space": "x",
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "projection_subproblem": _subproblem_params(),
        },
    )
    _, _, _, violation_traj, _ = optimizer.optimize(initial_point=torch.zeros(1, 2), verbose=False, seed=0)
    assert violation_traj.item() == pytest.approx(0.0)


def test_pgd_optimizer_stops_on_objective_change_not_feasibility_only():
    optimizer = PGDOptimizer(
        _SimpleProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 5,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "objective_change_threshold": 1e-6,
            "min_iterations": 1,
            "acceleration_method": "none",
            "acceleration_space": "x",
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "projection_subproblem": _subproblem_params(),
        },
    )

    _, _, objective_traj, violation_traj, per_iter_time = optimizer.optimize(
        initial_point=torch.zeros(1, 2),
        verbose=False,
        seed=0,
    )

    assert len(per_iter_time) == 1
    assert objective_traj.numel() == 2
    assert torch.allclose(objective_traj, torch.zeros_like(objective_traj))
    assert violation_traj.max().item() == pytest.approx(0.0)


def test_pgd_nag_evaluates_gradient_at_standard_lookahead_point():
    optimizer = PGDOptimizer(
        _SimpleProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 2,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "acceleration_method": "nag",
            "acceleration_space": "x",
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.5,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "projection_subproblem": _subproblem_params(),
        },
    )
    x_opt, *_ = optimizer.optimize(initial_point=torch.tensor([[1.0, 0.0]]), verbose=False, seed=0)
    assert torch.allclose(x_opt, torch.tensor([[0.56, 0.0]]), atol=1e-6)


def test_hom_pgd_optimizer_loop_records_consistent_lengths():
    problem = _SimpleProblem()
    optimizer = HomPGDOptimizer(
        problem,
        {
            "learning_rate": 0.1,
            "max_iterations": 3,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "acceleration_method": "none",
            "acceleration_space": "z",
            "stepsize_rule": "diminish",
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "opt": "gd",
            "momentum": 0.0,
        },
        hom_map=_IdentityHomMap(),
    )
    x_opt, decision_traj, objective_traj, violation_traj, per_iter_time, last_trans_time = optimizer.optimize(
        verbose=False,
        seed=0,
    )
    assert x_opt.shape == (1, 2)
    assert decision_traj.shape[1] == 2
    assert objective_traj.numel() == len(per_iter_time) + 1
    assert violation_traj.shape[0] == len(per_iter_time) + 1
    assert last_trans_time >= 0.0
    assert problem.logged_hom_map_methods[0] == "explicit"


def test_hom_pgd_reuses_hom_map_state_for_problem_gradient():
    problem = _StateAwareProblem()
    optimizer = HomPGDOptimizer(
        problem,
        {
            "learning_rate": 0.1,
            "max_iterations": 1,
            "max_running_time": 10,
            "convergence_threshold": 0.0,
            "acceleration_method": "none",
            "acceleration_space": "z",
            "stepsize_rule": "constant",
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "opt": "gd",
            "momentum": 0.0,
        },
        hom_map=_StateTrackingHomMap(),
    )

    optimizer.optimize(initial_point=torch.tensor([[0.5, -0.25]]), verbose=False, seed=0)

    assert problem.seen_x is not None
    assert problem.seen_hom_state is not None
    assert torch.allclose(problem.seen_hom_state["marker"], torch.ones_like(problem.seen_x))


def test_hom_pgd_optimizer_stops_on_objective_change_by_default():
    optimizer = HomPGDOptimizer(
        _SimpleProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 5,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "objective_change_threshold": 1e-6,
            "min_iterations": 1,
            "acceleration_method": "none",
            "acceleration_space": "z",
            "stepsize_rule": "constant",
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "opt": "gd",
            "momentum": 0.0,
        },
        hom_map=_IdentityHomMap(),
    )

    _, _, objective_traj, violation_traj, per_iter_time, _ = optimizer.optimize(
        initial_point=torch.zeros(1, 2),
        verbose=False,
        seed=0,
    )

    assert len(per_iter_time) == 1
    assert torch.allclose(objective_traj, torch.zeros_like(objective_traj))
    assert violation_traj.max().item() <= 0.0


def test_hom_pgd_accepts_normalized_gd_backend():
    optimizer = HomPGDOptimizer(
        _SimpleProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 1,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "acceleration_method": "none",
            "acceleration_space": "z",
            "stepsize_rule": "constant",
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "opt": "normalized_gd",
            "momentum": 0.0,
        },
        hom_map=_IdentityHomMap(),
    )

    x_opt, *_ = optimizer.optimize(initial_point=torch.tensor([[0.6, 0.8]]), verbose=False, seed=0)

    assert torch.allclose(x_opt, torch.tensor([[0.54, 0.72]]), atol=1e-6)


def test_hom_pgd_proximal_mode_refreshes_z_center_by_stage():
    optimizer = HomPGDOptimizer(
        _SimpleProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 4,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "acceleration_method": "none",
            "acceleration_space": "z",
            "stepsize_rule": "constant",
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "opt": "gd",
            "momentum": 0.0,
            "use_proximal": True,
            "proximal_update_iterations": 2,
            "proximal_coef": 10.0,
            "proximal_space": "z",
        },
        hom_map=_IdentityHomMap(),
    )

    x_opt, *_ = optimizer.optimize(initial_point=torch.tensor([[1.0, 0.0]]), verbose=False, seed=0)

    assert torch.allclose(x_opt, torch.tensor([[0.7396, 0.0]]), atol=1e-6)
    assert optimizer.proximal_steps == 2


def test_hom_pgd_rejects_proximal_stage_count_larger_than_iteration_budget():
    with pytest.raises(ValueError, match="proximal_update_iterations"):
        HomPGDOptimizer(
            _SimpleProblem(),
            {
                "learning_rate": 0.1,
                "max_iterations": 2,
                "max_running_time": 10,
                "convergence_threshold": 1e-6,
                "acceleration_method": "none",
                "acceleration_space": "z",
                "stepsize_rule": "constant",
                "lr_decay": 0.9,
            "min_lr": 1e-6,
                "opt": "gd",
                "momentum": 0.0,
                "use_proximal": True,
                "proximal_update_iterations": 3,
                "proximal_coef": 1.0,
                "proximal_space": "z",
            },
            hom_map=_IdentityHomMap(),
        )


def test_hom_pgd_rejects_proximal_mode_with_adam_backend():
    with pytest.raises(ValueError, match="requires opt='gd'"):
        HomPGDOptimizer(
            _SimpleProblem(),
            {
                "learning_rate": 0.1,
                "max_iterations": 2,
                "max_running_time": 10,
                "convergence_threshold": 1e-6,
                "acceleration_method": "none",
                "acceleration_space": "z",
                "stepsize_rule": "constant",
                "lr_decay": 0.9,
            "min_lr": 1e-6,
                "opt": "adam",
                "momentum": 0.0,
                "use_proximal": True,
                "proximal_update_iterations": 1,
                "proximal_coef": 1.0,
                "proximal_space": "z",
            },
            hom_map=_IdentityHomMap(),
        )


def test_inn_pgd_optimizer_loop_records_consistent_lengths():
    optimizer = INNPGDOptimizer(
        _INNProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 3,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "feasibility_eps": 1e-6,
            "opt": "gd",
            "momentum": 0.0,
            "stepsize_rule": "diminish",
        },
        model=_IdentityModel(),
    )
    x_opt, decision_traj, latent_traj, objective_traj, violation_traj, per_iter_time = optimizer.optimize(
        input_params=torch.zeros(1, 1)
    )
    assert x_opt.shape == (1, 2)
    assert decision_traj.shape[1] == 2
    assert latent_traj.shape[1] == 2
    assert objective_traj.numel() == len(per_iter_time) + 1
    assert violation_traj.numel() == len(per_iter_time) + 1


def test_inn_pgd_verbose_uses_shared_iteration_table(capsys):
    optimizer = INNPGDOptimizer(
        _ViolatingINNProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 1,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "feasibility_eps": 1e-6,
            "opt": "gd",
            "momentum": 0.0,
            "stepsize_rule": "constant",
            "proj_eps": 2.0,
            "verbose_interval": 1,
        },
        model=_IdentityModel(),
    )

    optimizer.optimize(input_params=torch.zeros(1, 1), verbose=True)
    output = capsys.readouterr().out
    assert "outer" in output
    assert "obj" in output
    assert "eqvio" in output
    assert "ineqvio" in output
    assert "1.000e+00" in output


def test_lagrangian_optimizer_loop_records_consistent_lengths():
    optimizer = LagrangianOptimizer(
        _PenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 3,
            "inner_iterations": 2,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "use_lagrangian": False,
            "use_penalty": False,
        },
    )
    x_opt, decision_traj, objective_traj, violation_traj, per_iter_time = optimizer.optimize(verbose=False, seed=0)
    assert x_opt.shape == (1, 2)
    assert decision_traj.shape[1] == 2
    assert objective_traj.numel() == len(per_iter_time) + 1
    assert violation_traj.numel() == len(per_iter_time) + 1


def test_experiment_optimizer_gradient_config_rejects_unknown_methods():
    with pytest.raises(ValueError, match="Unsupported lagrangian_gradient"):
        _validate_gradient_method("finite_diff", name="lagrangian_gradient")


def test_lagrangian_optimizer_verbose_prints_outer_metrics(capsys):
    optimizer = LagrangianOptimizer(
        _PenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 1,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-12,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "use_lagrangian": False,
            "use_penalty": False,
        },
    )
    optimizer.optimize(initial_point=torch.ones(1, 2), verbose=True, seed=0)
    output = capsys.readouterr().out
    assert "outer" in output
    assert "obj" in output
    assert "eqvio" in output
    assert "ineqvio" in output
    assert "lr" in output
    assert "dual_norm" in output
    assert "penalty" in output
    assert "lag_gap" in output
    assert "iter_time" in output
    assert "inner_err" not in output
    assert "dual_lr" not in output
    assert "total_time" not in output
    assert "1" in output


def test_penalty_verbose_values_use_four_significant_digits_in_scientific_notation():
    assert format_value(12345.678) == "1.235e+04"
    assert format_value(0.0012345678) == "1.235e-03"
    assert format_value(1.0) == "1.000e+00"


def test_lagrangian_optimizer_verbose_interval_throttles_outer_metrics(capsys):
    optimizer = LagrangianOptimizer(
        _PenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 2,
            "inner_iterations": 1,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-12,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "use_lagrangian": False,
            "use_penalty": False,
            "verbose_interval": 50,
        },
    )
    optimizer.optimize(initial_point=torch.ones(1, 2), verbose=True, seed=0)
    output = capsys.readouterr().out
    assert "outer" in output
    data_rows = [line for line in output.splitlines() if line.strip().startswith("1")]
    assert len(data_rows) == 1


def test_first_order_loop_verbose_interval_throttles_metrics(capsys):
    class _IdentityUpdate:
        def step(self, grad):
            return grad

    def _evaluate_state(point):
        objective = (point**2).sum(dim=1, keepdim=True)
        violation = torch.clamp(point.max(dim=1, keepdim=True).values - 10.0, min=0.0)
        return {
            "objective": objective,
            "violation": violation,
            "eq_violation": torch.zeros_like(violation),
            "ineq_violation": violation,
        }

    initial_state = torch.ones(1, 2)
    initial_eval = _evaluate_state(initial_state)
    run_first_order_loop(
        initial_state=initial_state,
        initial_eval=initial_eval,
        update_backend=_IdentityUpdate(),
        compute_gradient=lambda state, _prev, _iter: state,
        apply_step=lambda state, update, lr, _prev, _iter: state - lr * update,
        evaluate_state=_evaluate_state,
        max_iterations=3,
        max_running_time=10,
        initial_learning_rate=0.1,
        stepsize_rule="constant",
        lr_decay=1.0,
        min_lr=1e-6,
        convergence_threshold=0.0,
        verbose=True,
        verbose_interval=2,
    )
    output = capsys.readouterr().out
    assert "outer" in output
    assert "eqvio" in output
    assert "ineqvio" in output
    data_rows = [line for line in output.splitlines() if line.strip().startswith(("1", "2", "3"))]
    assert len(data_rows) == 2
    assert data_rows[0].strip().startswith("1")
    assert data_rows[1].strip().startswith("2")


def test_lagrangian_outer_convergence_requires_first_order_lagrangian_gap():
    common_params = {
        "learning_rate": 0.1,
        "max_running_time": 10,
        "outer_iterations": 3,
        "inner_iterations": 1,
        "lr_decay": 0.9,
            "min_lr": 1e-6,
        "convergence_threshold": 1e-6,
        "stepsize_rule": "constant",
        "opt": "gd",
        "momentum": 0.0,
        "dual_learning_rate": 0.1,
        "penalty_coef": 1.0,
        "use_lagrangian": True,
        "use_penalty": True,
        "use_proximal": False,
    }
    optimizer = LagrangianOptimizer(
        _FeasibleAugmentedStationaryPenaltyProblem(),
        dict(common_params),
    )
    _, _, _, _, per_iter_time = optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)

    assert len(per_iter_time) == 3
    assert optimizer.last_final_first_order_lagrangian_gap == pytest.approx(1.0)

    optimizer_without_gap = LagrangianOptimizer(
        _FeasibleAugmentedStationaryPenaltyProblem(),
        {**common_params, "check_first_order_lagrangian_gap": False},
    )
    _, _, _, _, per_iter_time_without_gap = optimizer_without_gap.optimize(
        initial_point=[[0.0, 0.0]],
        verbose=False,
        seed=0,
    )

    assert len(per_iter_time_without_gap) == 1


def test_lagrangian_nag_inner_loop_uses_standard_lookahead_point():
    optimizer = LagrangianOptimizer(
        _PenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 2,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "inner_stepsize_rule": "constant",
            "opt": "gd",
            "acceleration_method": "nag",
            "acceleration_space": "x",
            "momentum": 0.5,
            "dual_learning_rate": 0.1,
            "use_lagrangian": False,
            "use_penalty": False,
        },
    )
    x_opt, *_ = optimizer.optimize(initial_point=torch.tensor([[1.0, 0.0]]), verbose=False, seed=0)
    assert torch.allclose(x_opt, torch.tensor([[0.56, 0.0]]), atol=1e-6)


def test_lagrangian_nag_preserves_inner_velocity_by_default_across_outer_iterations():
    params = {
        "learning_rate": 0.1,
        "max_running_time": 10,
        "outer_iterations": 2,
        "inner_iterations": 1,
        "lr_decay": 0.9,
            "min_lr": 1e-6,
        "convergence_threshold": 1e-12,
        "stepsize_rule": "constant",
        "inner_stepsize_rule": "constant",
        "opt": "gd",
        "acceleration_method": "nag",
        "acceleration_space": "x",
        "momentum": 0.5,
        "dual_learning_rate": 0.1,
        "use_lagrangian": False,
        "use_penalty": False,
    }
    optimizer = LagrangianOptimizer(_PenaltyProblem(), params)
    x_opt, *_ = optimizer.optimize(initial_point=torch.tensor([[1.0, 0.0]]), verbose=False, seed=0)
    assert torch.allclose(x_opt, torch.tensor([[0.56, 0.0]]), atol=1e-6)


def test_lagrangian_nag_inner_restart_resets_velocity_between_outer_iterations():
    params = {
        "learning_rate": 0.1,
        "max_running_time": 10,
        "outer_iterations": 2,
        "inner_iterations": 1,
        "lr_decay": 0.9,
            "min_lr": 1e-6,
        "convergence_threshold": 1e-12,
        "stepsize_rule": "constant",
        "inner_stepsize_rule": "constant",
        "opt": "gd",
        "acceleration_method": "nag",
        "acceleration_space": "x",
        "momentum": 0.5,
        "dual_learning_rate": 0.1,
        "use_lagrangian": False,
        "use_penalty": False,
        "inner_restart": True,
    }
    optimizer = LagrangianOptimizer(_PenaltyProblem(), params)
    x_opt, *_ = optimizer.optimize(initial_point=torch.tensor([[1.0, 0.0]]), verbose=False, seed=0)
    assert torch.allclose(x_opt, torch.tensor([[0.64, 0.0]]), atol=1e-6)


def test_lagrangian_prox_gd_refreshes_inner_proximal_center():
    optimizer = LagrangianOptimizer(
        _AdaptiveInnerPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 4,
            "inner_solver": "prox_gd",
            "inner_proximal_update_iterations": 2,
            "inner_proximal_coef": 10.0,
            "inner_proximal_space": "x",
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "use_lagrangian": False,
            "use_penalty": False,
            "use_proximal": False,
        },
    )
    x_opt, *_ = optimizer.optimize(initial_point=torch.zeros(1, 2), verbose=False, seed=0)
    assert torch.allclose(x_opt, torch.full((1, 2), -0.2), atol=1e-6)


def test_lagrangian_rejects_prox_gd_update_budget_larger_than_inner_budget():
    with pytest.raises(ValueError, match="inner_proximal_update_iterations"):
        LagrangianOptimizer(
            _ConstantGradientPenaltyProblem(),
            {
                "learning_rate": 0.1,
                "max_running_time": 10,
                "outer_iterations": 1,
                "inner_iterations": 2,
                "inner_solver": "prox_gd",
                "inner_proximal_update_iterations": 3,
                "inner_proximal_coef": 1.0,
                "inner_proximal_space": "x",
                "lr_decay": 0.9,
            "min_lr": 1e-6,
                "convergence_threshold": 1e-6,
                "stepsize_rule": "constant",
                "opt": "gd",
                "momentum": 0.0,
                "dual_learning_rate": 0.1,
                "use_lagrangian": False,
                "use_penalty": False,
                "use_proximal": False,
            },
        )


def test_lagrangian_rejects_removed_linearized_prox_gd_inner_solver():
    with pytest.raises(ValueError, match="Unsupported inner_solver"):
        LagrangianOptimizer(
            _ConstantGradientPenaltyProblem(),
            {
                "learning_rate": 0.1,
                "max_running_time": 10,
                "outer_iterations": 1,
                "inner_iterations": 2,
                "inner_solver": "linearized_prox_gd",
                "inner_proximal_update_iterations": 1,
                "inner_proximal_coef": 1.0,
                "inner_proximal_space": "x",
                "lr_decay": 0.9,
            "min_lr": 1e-6,
                "convergence_threshold": 1e-6,
                "stepsize_rule": "constant",
                "opt": "gd",
                "momentum": 0.0,
                "dual_learning_rate": 0.1,
                "use_lagrangian": False,
                "use_penalty": False,
                "use_proximal": False,
            },
        )


def test_hom_alm_optimizer_loop_records_consistent_lengths():
    problem = _HomPenaltyProblem()
    optimizer = HomALMOptimizer(
        problem,
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 3,
            "inner_iterations": 2,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "use_lagrangian": False,
            "use_penalty": False,
        },
        hom_map=_IdentityHomMap(),
    )
    x_opt, decision_traj, objective_traj, violation_traj, per_iter_time, last_trans_time = optimizer.optimize(
        verbose=False,
        seed=0,
    )
    assert x_opt.shape == (1, 2)
    assert decision_traj.shape[1] == 2
    assert objective_traj.numel() == len(per_iter_time) + 1
    assert violation_traj.numel() == len(per_iter_time) + 1
    assert last_trans_time >= 0.0


def test_hom_alm_dual_update_uses_penalty_scaled_dual_learning_rate_and_updates_penalty():
    problem = _HomPenaltyProblem()
    optimizer = HomALMOptimizer(
        problem,
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 2,
            "inner_iterations": 1,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.2,
            "penalty_coef": 2.0,
            "penalty_growth": 1.25,
            "max_penalty": 2.4,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
            "check_first_order_lagrangian_gap": False,
        },
        hom_map=_IdentityHomMap(),
    )
    optimizer.optimize(verbose=False, seed=0)
    assert len(problem.logged_dual_vars) >= 2
    assert problem.logged_dual_vars[0].shape == (1, 1)
    assert torch.allclose(problem.logged_dual_vars[0], torch.zeros(1, 1))
    assert torch.allclose(problem.logged_dual_vars[1], torch.full((1, 1), 0.4))
    assert optimizer.penalty_coef == pytest.approx(2.4)


def test_hom_alm_outer_convergence_requires_first_order_lagrangian_gap():
    common_params = {
        "learning_rate": 0.0,
        "max_running_time": 10,
        "outer_iterations": 3,
        "inner_iterations": 1,
        "lr_decay": 0.9,
            "min_lr": 1e-6,
        "convergence_threshold": 1e-6,
        "stepsize_rule": "constant",
        "opt": "gd",
        "momentum": 0.0,
        "dual_learning_rate": 0.1,
        "use_lagrangian": False,
        "use_penalty": False,
        "use_proximal": False,
    }
    optimizer = HomALMOptimizer(
        _FeasibleNonstationaryHomPenaltyProblem(),
        dict(common_params),
        hom_map=_IdentityHomMap(),
    )
    _, _, _, _, per_iter_time, _ = optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)

    assert len(per_iter_time) == 3
    assert optimizer.last_final_first_order_lagrangian_gap == pytest.approx(2**0.5)

    optimizer_without_gap = HomALMOptimizer(
        _FeasibleNonstationaryHomPenaltyProblem(),
        {**common_params, "check_first_order_lagrangian_gap": False},
        hom_map=_IdentityHomMap(),
    )
    _, _, _, _, per_iter_time_without_gap, _ = optimizer_without_gap.optimize(
        initial_point=[[0.0, 0.0]],
        verbose=False,
        seed=0,
    )

    assert len(per_iter_time_without_gap) == 1


def test_hom_alm_uses_z_proximal_by_default():
    problem = _HomPenaltyProblem()
    optimizer = HomALMOptimizer(
        problem,
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 1,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "proximal_coef": 0.5,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": True,
            "check_first_order_lagrangian_gap": False,
        },
        hom_map=_IdentityHomMap(),
    )
    optimizer.optimize(verbose=False, seed=0)
    assert problem.logged_proximal_spaces[0] == "z"
    assert problem.logged_x_outer[0] is None


def test_hom_alm_can_switch_to_x_proximal():
    problem = _HomPenaltyProblem()
    optimizer = HomALMOptimizer(
        problem,
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 1,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "proximal_coef": 0.5,
            "proximal_space": "x",
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": True,
            "check_first_order_lagrangian_gap": False,
        },
        hom_map=_IdentityHomMap(),
    )
    optimizer.optimize(verbose=False, seed=0)
    assert problem.logged_proximal_spaces[0] == "x"
    assert problem.logged_x_outer[0] is not None


def test_hom_alm_accepts_standard_nag_inner_acceleration():
    optimizer = HomALMOptimizer(
        _HomPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 2,
            "inner_iterations": 2,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "acceleration_method": "nag",
            "acceleration_space": "z",
            "momentum": 0.5,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
        },
        hom_map=_IdentityHomMap(),
    )
    x_opt, decision_traj, objective_traj, violation_traj, per_iter_time, last_trans_time = optimizer.optimize(
        verbose=False,
        seed=0,
    )
    assert x_opt.shape == (1, 2)
    assert decision_traj.shape[1] == 2
    assert objective_traj.numel() == len(per_iter_time) + 1
    assert violation_traj.numel() == len(per_iter_time) + 1
    assert last_trans_time >= 0.0


def test_hom_alm_rejects_unknown_update_backend():
    with pytest.raises(ValueError, match="Unsupported opt"):
        HomALMOptimizer(
            _HomPenaltyProblem(),
            {
                "learning_rate": 0.1,
                "max_running_time": 10,
                "outer_iterations": 1,
                "inner_iterations": 1,
                "lr_decay": 0.9,
                "min_lr": 1e-6,
                "convergence_threshold": 1e-6,
                "stepsize_rule": "constant",
                "opt": "unsupported",
                "momentum": 0.5,
                "dual_learning_rate": 0.1,
                "penalty_coef": 1.0,
                "use_lagrangian": True,
                "use_penalty": True,
                "use_proximal": False,
            },
            hom_map=_IdentityHomMap(),
        )


def test_hom_alm_prox_gd_refreshes_inner_proximal_center():
    optimizer = HomALMOptimizer(
        _ConstantGradientHomPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 4,
            "inner_solver": "prox_gd",
            "inner_proximal_update_iterations": 2,
            "inner_proximal_coef": 10.0,
            "inner_proximal_space": "z",
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "use_lagrangian": False,
            "use_penalty": False,
            "use_proximal": False,
        },
        hom_map=_IdentityHomMap(),
    )
    x_opt, *_ = optimizer.optimize(initial_point=torch.zeros(1, 2), verbose=False, seed=0)
    assert torch.allclose(x_opt, torch.full((1, 2), -0.2), atol=1e-6)


def test_hom_alm_rejects_prox_gd_update_budget_larger_than_inner_budget():
    with pytest.raises(ValueError, match="inner_proximal_update_iterations"):
        HomALMOptimizer(
            _ConstantGradientHomPenaltyProblem(),
            {
                "learning_rate": 0.1,
                "max_running_time": 10,
                "outer_iterations": 1,
                "inner_iterations": 2,
                "inner_solver": "prox_gd",
                "inner_proximal_update_iterations": 3,
                "inner_proximal_coef": 1.0,
                "inner_proximal_space": "z",
                "lr_decay": 0.9,
            "min_lr": 1e-6,
                "convergence_threshold": 1e-6,
                "stepsize_rule": "constant",
                "opt": "gd",
                "momentum": 0.0,
                "dual_learning_rate": 0.1,
                "penalty_coef": 1.0,
                "use_lagrangian": False,
                "use_penalty": False,
                "use_proximal": False,
            },
            hom_map=_IdentityHomMap(),
        )


def test_lagrangian_optimizer_penalty_growth_respects_max_penalty():
    optimizer = LagrangianOptimizer(
        _PenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 3,
            "inner_iterations": 1,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "penalty_growth": 1.3,
            "max_penalty": 1.4,
            "use_lagrangian": False,
            "use_penalty": True,
            "use_proximal": False,
        },
    )
    optimizer.optimize(verbose=False, seed=0)
    assert optimizer.penalty_coef == pytest.approx(1.4)


def test_penalty_optimizers_share_primal_stepsize_rule_resolution():
    params = {
        "learning_rate": 0.1,
        "max_running_time": 10,
        "outer_iterations": 1,
        "inner_iterations": 1,
        "lr_decay": 0.9,
            "min_lr": 1e-6,
        "outer_lr_decay": 0.8,
        "inner_lr_decay": 0.7,
        "inner_learning_rate": 0.05,
        "convergence_threshold": 1e-6,
        "stepsize_rule": "adaptive",
        "outer_stepsize_rule": "diminish",
        "inner_stepsize_rule": "constant",
        "opt": "gd",
        "momentum": 0.0,
        "dual_learning_rate": 0.1,
        "use_lagrangian": False,
        "use_penalty": False,
        "use_proximal": False,
    }
    alm = LagrangianOptimizer(_PenaltyProblem(), dict(params))
    hom_alm = HomALMOptimizer(_HomPenaltyProblem(), dict(params), hom_map=_IdentityHomMap())
    assert alm.outer_stepsize_rule == hom_alm.outer_stepsize_rule == "diminish"
    assert alm.inner_stepsize_rule == hom_alm.inner_stepsize_rule == "constant"
    assert alm.outer_lr_decay == hom_alm.outer_lr_decay == pytest.approx(0.8)
    assert alm.inner_lr_decay == hom_alm.inner_lr_decay == pytest.approx(0.7)
    assert alm.inner_learning_rate == hom_alm.inner_learning_rate == pytest.approx(0.05)


def test_penalty_outer_adaptive_lr_tracks_violation_before_objective():
    recorder = IterationRecorder()
    recorder.record_state(torch.tensor([2.0]), torch.tensor([0.1]))
    recorder.record_state(torch.tensor([1.0]), torch.tensor([0.2]))
    decayed = _update_penalty_outer_learning_rate(
        0.1,
        recorder,
        stepsize_rule="adaptive",
        lr_decay=0.5,
        min_lr=1e-6,
    )
    assert decayed == pytest.approx(0.05)

    recorder = IterationRecorder()
    recorder.record_state(torch.tensor([1.0]), torch.tensor([0.2]))
    recorder.record_state(torch.tensor([2.0]), torch.tensor([0.1]))
    unchanged = _update_penalty_outer_learning_rate(
        0.1,
        recorder,
        stepsize_rule="adaptive",
        lr_decay=0.5,
        min_lr=1e-6,
    )
    assert unchanged == pytest.approx(0.1)


def test_should_increase_penalty_only_when_violation_does_not_decrease():
    previous = torch.tensor([[1.0]])
    clearly_lower = torch.tensor([[0.8]])
    slightly_lower = torch.tensor([[0.9995]])
    same = torch.tensor([[1.0]])
    higher = torch.tensor([[1.2]])

    assert _should_increase_penalty(previous, clearly_lower) is False
    assert _should_increase_penalty(previous, slightly_lower) is True
    assert _should_increase_penalty(previous, same) is True
    assert _should_increase_penalty(previous, higher) is True


def test_hom_alm_penalty_growth_waits_for_nondecreasing_eq_violation():
    optimizer = HomALMOptimizer(
        _DecreasingEqHomPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 2,
            "inner_iterations": 1,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-12,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "penalty_growth": 1.5,
            "max_penalty": 10.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
        },
        hom_map=_IdentityHomMap(),
    )
    optimizer.optimize(initial_point=torch.tensor([[1.0, 0.0]]), verbose=False, seed=0)
    assert optimizer.penalty_coef == pytest.approx(1.0)


def test_iteration_recorder_keeps_best_decision_without_full_trajectory():
    recorder = IterationRecorder(track_decisions=False)
    recorder.record_state(torch.tensor([[2.0]]), torch.tensor([[0.5]]), decision=torch.tensor([[1.0, 1.0]]))
    recorder.record_state(torch.tensor([[3.0]]), torch.tensor([[0.1]]), decision=torch.tensor([[2.0, 2.0]]))
    recorder.record_state(torch.tensor([[1.0]]), torch.tensor([[0.2]]), decision=torch.tensor([[3.0, 3.0]]))

    payload = recorder.finalize()

    assert payload["decision_trajectory"] == []
    assert payload["best_iteration"] == 1
    assert torch.allclose(payload["best_violation"], torch.tensor(0.1))
    assert torch.allclose(payload["best_decision"], torch.tensor([[2.0, 2.0]]))


def test_lagrangian_optimizer_can_return_best_violation_iterate():
    base_params = {
        "learning_rate": 1.0,
        "max_running_time": 10,
        "outer_iterations": 1,
        "inner_iterations": 1,
        "lr_decay": 0.9,
            "min_lr": 1e-6,
        "convergence_threshold": 1e-6,
        "stepsize_rule": "constant",
        "opt": "gd",
        "momentum": 0.0,
        "dual_learning_rate": 0.1,
        "use_lagrangian": False,
        "use_penalty": False,
    }
    final_optimizer = LagrangianOptimizer(_ReturnBestPenaltyProblem(), dict(base_params))
    final_x, *_ = final_optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)

    best_params = dict(base_params)
    best_params["return_best_violation"] = True
    best_optimizer = LagrangianOptimizer(_ReturnBestPenaltyProblem(), best_params)
    best_x, *_ = best_optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)

    assert torch.allclose(final_x, torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(best_x, torch.tensor([[0.0, 0.0]]))


def test_lagrangian_optimizer_aligns_initial_state_dtype_with_problem_tensors():
    optimizer = LagrangianOptimizer(
        _DoubleMatrixPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 1,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "use_lagrangian": False,
            "use_penalty": False,
        },
    )

    x_opt, *_ = optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)

    assert x_opt.dtype == torch.float64
    assert optimizer.dual_var.dtype == torch.float64


def test_hom_alm_inner_adaptive_lr_tracks_equality_violation(monkeypatch):
    captured = {}

    def _capture_inner_lr(base_lr, current_lr, **kwargs):
        del base_lr, current_lr
        captured["current_error"] = kwargs["current_error"]
        captured["best_error"] = kwargs["best_error"]
        return 0.1

    monkeypatch.setattr(hom_alm_algorithm, "_update_penalty_inner_learning_rate", _capture_inner_lr)

    optimizer = HomALMOptimizer(
        _AdaptiveInnerHomPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 1,
            "lr_decay": 0.5,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "inner_stepsize_rule": "adaptive",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
        },
        hom_map=_IdentityHomMap(),
    )
    optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)
    assert captured["current_error"] == pytest.approx(0.7)
    assert captured["best_error"] == pytest.approx(float("inf"))


def test_hom_alm_constant_inner_stepsize_skips_adaptive_lr_update(monkeypatch):
    def _should_not_run(*args, **kwargs):
        raise AssertionError("adaptive inner learning-rate update should not run for constant inner stepsizes")

    monkeypatch.setattr(hom_alm_algorithm, "_update_penalty_inner_learning_rate", _should_not_run)

    optimizer = HomALMOptimizer(
        _AdaptiveInnerHomPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 1,
            "lr_decay": 0.5,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "inner_stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
        },
        hom_map=_IdentityHomMap(),
    )
    optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)


def test_lagrangian_optimizer_adaptive_inner_stopping_respects_min_iterations():
    optimizer = LagrangianOptimizer(
        _AdaptiveInnerPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 5,
            "inner_stopping_rule": "adaptive",
            "inner_iterations_min": 2,
            "inner_iterations_max": 5,
            "inner_tol_factor": 10.0,
            "inner_tol_min": 1e-6,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-12,
            "stepsize_rule": "constant",
            "inner_stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "use_lagrangian": False,
            "use_penalty": False,
        },
    )
    x_opt, *_ = optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)
    assert torch.allclose(x_opt, torch.tensor([[-0.2, -0.2]]), atol=1e-7)


def test_hom_alm_adaptive_inner_stopping_respects_min_iterations():
    optimizer = HomALMOptimizer(
        _ConstantGradientHomPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 5,
            "inner_stopping_rule": "adaptive",
            "inner_iterations_min": 2,
            "inner_iterations_max": 5,
            "inner_tol_factor": 10.0,
            "inner_tol_min": 1e-6,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-12,
            "stepsize_rule": "constant",
            "inner_stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
        },
        hom_map=_IdentityHomMap(),
    )
    x_opt, *_ = optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)
    assert torch.allclose(x_opt, torch.tensor([[-0.2, -0.2]]), atol=1e-7)


def test_alm_inner_adaptive_lr_tracks_constraint_violation(monkeypatch):
    captured = {}

    def _capture_inner_lr(base_lr, current_lr, **kwargs):
        del base_lr, current_lr
        captured["current_error"] = kwargs["current_error"]
        captured["best_error"] = kwargs["best_error"]
        return 0.1

    monkeypatch.setattr(lagrangian_algorithms, "_update_penalty_inner_learning_rate", _capture_inner_lr)

    optimizer = LagrangianOptimizer(
        _AdaptiveInnerPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 1,
            "lr_decay": 0.5,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "inner_stepsize_rule": "adaptive",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
        },
    )
    optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)
    assert captured["current_error"] == pytest.approx(1.6)
    assert captured["best_error"] == pytest.approx(float("inf"))


def test_alm_constant_inner_stepsize_skips_adaptive_lr_update(monkeypatch):
    def _should_not_run(*args, **kwargs):
        raise AssertionError("adaptive inner learning-rate update should not run for constant inner stepsizes")

    monkeypatch.setattr(lagrangian_algorithms, "_update_penalty_inner_learning_rate", _should_not_run)

    optimizer = LagrangianOptimizer(
        _AdaptiveInnerPenaltyProblem(),
        {
            "learning_rate": 0.1,
            "max_running_time": 10,
            "outer_iterations": 1,
            "inner_iterations": 1,
            "lr_decay": 0.5,
            "min_lr": 1e-6,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "inner_stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
        },
    )
    optimizer.optimize(initial_point=[[0.0, 0.0]], verbose=False, seed=0)


def test_adaptive_inner_stopping_max_iterations_can_exceed_nominal_iterations():
    config = _resolve_inner_stopping_config(
        {
            "inner_stopping_rule": "adaptive",
            "inner_iterations_min": 10,
            "inner_iterations_max": 1000,
        },
        inner_iterations=200,
    )

    assert config["inner_iterations_min"] == 10
    assert config["inner_iterations_max"] == 1000


def test_adaptive_prox_inner_solver_uses_active_inner_budget_for_stage_steps():
    config = _resolve_penalty_inner_solver_config(
        {
            "inner_solver": "prox_gd",
            "inner_proximal_update_iterations": 10,
        },
        inner_iterations=200,
        inner_iteration_budget=1000,
        default_proximal_space="z",
    )

    assert config["inner_proximal_steps"] == 100


def test_alm_eq_penalty_growth_waits_for_nondecreasing_eq_violation(monkeypatch):
    solutions = iter(
        [
            {"solution": [1.0], "runtime_total": 0.0, "status": "optimal"},
            {"solution": [0.8], "runtime_total": 0.0, "status": "optimal"},
        ]
    )

    def _fake_solve_exact_result(*args, **kwargs):
        del args, kwargs
        return next(solutions)

    monkeypatch.setattr(equality_algorithms, "_cached_exact_solver", lambda *args, **kwargs: object())
    monkeypatch.setattr(equality_algorithms, "solve_exact_result", _fake_solve_exact_result)

    optimizer = EqualityConstrainedALMOptimizer(
        _EqOnlyPenaltyProblem(),
        {
            "max_running_time": 10,
            "outer_iterations": 2,
            "convergence_threshold": 1e-12,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "penalty_growth": 1.5,
            "max_penalty": 10.0,
            "max_dual": 10.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
            "opt_type": "ALM-EQ",
        },
    )
    optimizer.optimize(initial_point=[[1.2]], verbose=False, seed=0)
    assert optimizer.penalty_coef == pytest.approx(1.0)


def test_alm_eq_outer_convergence_requires_objective_change(monkeypatch):
    calls = []

    def _fake_solve_exact_result(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        return {"solution": [0.0], "runtime_total": 0.0, "status": "optimal"}

    monkeypatch.setattr(equality_algorithms, "_cached_exact_solver", lambda *args, **kwargs: object())
    monkeypatch.setattr(equality_algorithms, "solve_exact_result", _fake_solve_exact_result)

    common_params = {
        "max_running_time": 10,
        "outer_iterations": 3,
        "convergence_threshold": 1e-6,
        "dual_learning_rate": 0.1,
        "penalty_coef": 1.0,
        "penalty_growth": 1.5,
        "max_penalty": 10.0,
        "max_dual": 10.0,
        "use_lagrangian": True,
        "use_penalty": True,
        "use_proximal": False,
        "opt_type": "ALM-EQ",
    }
    optimizer = EqualityConstrainedALMOptimizer(_EqOnlyPenaltyProblem(), dict(common_params))
    optimizer.optimize(initial_point=[[1.2]], verbose=False, seed=0)

    assert len(calls) == 2

    calls.clear()
    optimizer_without_objective_gate = EqualityConstrainedALMOptimizer(
        _EqOnlyPenaltyProblem(),
        {**common_params, "check_outer_objective_change": False},
    )
    optimizer_without_objective_gate.optimize(initial_point=[[1.2]], verbose=False, seed=0)

    assert len(calls) == 1


def test_alm_eq_exact_subproblem_uses_only_equality_duals(monkeypatch):
    seen = {}

    def _fake_solve_exact_result(*args, **kwargs):
        del args
        solve_config = kwargs["solve_config"]
        seen["dual_var_shape"] = tuple(solve_config["dual_var"].shape)
        return {"solution": [0.0], "runtime_total": 0.0, "status": "optimal"}

    monkeypatch.setattr(equality_algorithms, "_cached_exact_solver", lambda *args, **kwargs: object())
    monkeypatch.setattr(equality_algorithms, "solve_exact_result", _fake_solve_exact_result)

    optimizer = EqualityConstrainedALMOptimizer(
        _MixedEqPenaltyProblem(),
        {
            "max_running_time": 10,
            "outer_iterations": 1,
            "convergence_threshold": 1e-6,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "penalty_growth": 1.5,
            "max_penalty": 10.0,
            "max_dual": 10.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
            "opt_type": "ALM-EQ",
            "check_outer_objective_change": False,
        },
    )
    optimizer.optimize(initial_point=[[1.2]], verbose=False, seed=0)

    assert seen["dual_var_shape"] == (1,)


def test_alm_eq_verbose_uses_shared_iteration_table(monkeypatch, capsys):
    solutions = iter(
        [
            {"solution": [1.0], "runtime_total": 0.0, "status": "optimal"},
            {"solution": [0.8], "runtime_total": 0.0, "status": "optimal"},
        ]
    )

    def _fake_solve_exact_result(*args, **kwargs):
        del args, kwargs
        return next(solutions)

    monkeypatch.setattr(equality_algorithms, "_cached_exact_solver", lambda *args, **kwargs: object())
    monkeypatch.setattr(equality_algorithms, "solve_exact_result", _fake_solve_exact_result)

    optimizer = EqualityConstrainedALMOptimizer(
        _EqOnlyPenaltyProblem(),
        {
            "max_running_time": 10,
            "outer_iterations": 2,
            "convergence_threshold": 1e-12,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "penalty_growth": 1.5,
            "max_penalty": 10.0,
            "max_dual": 10.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
            "opt_type": "ALM-EQ",
            "verbose_interval": 2,
        },
    )

    optimizer.optimize(initial_point=[[1.2]], verbose=True, seed=0)
    output = capsys.readouterr().out
    assert "outer" in output
    assert "obj" in output
    assert "eqvio" in output
    assert "ineqvio" in output
    data_rows = [line for line in output.splitlines() if line.strip().startswith(("1", "2"))]
    assert len(data_rows) == 2


@pytest.mark.parametrize("algorithm", ["Prox-Penalty-EQ", "Prox-ALM-EQ"])
def test_run_algorithm_rejects_eq_prox_algorithm_with_disabled_proximal(algorithm):
    params = {
        "common": {"seed": 0},
        algorithm: {
            "max_running_time": 10,
            "outer_iterations": 1,
            "convergence_threshold": 1e-6,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "penalty_growth": 1.5,
            "proximal_coef": 0.1,
            "max_penalty": 10.0,
            "max_dual": 10.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "use_proximal": False,
            "opt_type": algorithm,
        },
    }
    with pytest.raises(ValueError, match="requires use_proximal=True"):
        run_algorithm(algorithm, _EqOnlyPenaltyProblem(), params)


@pytest.mark.parametrize("algorithm", ["Prox-Penalty-EQ", "Prox-ALM-EQ"])
def test_run_algorithm_rejects_eq_prox_algorithm_without_proximal_coef(algorithm):
    params = {
        "common": {"seed": 0},
        algorithm: {
            "max_running_time": 10,
            "outer_iterations": 1,
            "convergence_threshold": 1e-6,
            "dual_learning_rate": 0.1,
            "penalty_coef": 1.0,
            "penalty_growth": 1.5,
            "max_penalty": 10.0,
            "max_dual": 10.0,
            "use_lagrangian": True,
            "use_penalty": True,
            "opt_type": algorithm,
        },
    }
    with pytest.raises(KeyError, match="proximal_coef"):
        run_algorithm(algorithm, _EqOnlyPenaltyProblem(), params)


def test_pgd_optimizer_accepts_explicit_projection_subproblem_config():
    optimizer = PGDOptimizer(
        _SimpleProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 2,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "acceleration_method": "none",
            "acceleration_space": "x",
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "projection_subproblem": _subproblem_params(
                learning_rate=2e-3,
                outer_iterations=7,
                inner_iterations=8,
                dual_learning_rate=3e-2,
            ),
        },
    )
    assert optimizer.projection_subproblem_params["learning_rate"] == pytest.approx(2e-3)
    assert optimizer.projection_subproblem_params["outer_iterations"] == 7
    assert optimizer.projection_subproblem_params["dual_learning_rate"] == pytest.approx(3e-2)


def test_pgd_optimizer_requires_explicit_projection_subproblem_config():
    with pytest.raises(KeyError, match="projection_subproblem"):
        PGDOptimizer(
            _SimpleProblem(),
            {
                "learning_rate": 0.1,
                "max_iterations": 2,
                "max_running_time": 10,
                "convergence_threshold": 1e-6,
                "acceleration_method": "nag",
                "acceleration_space": "x",
                "stepsize_rule": "constant",
                "opt": "gd",
                "momentum": 0.5,
                "lr_decay": 0.9,
                "min_lr": 1e-6,
            },
        )


def test_fw_optimizer_accepts_explicit_linearization_subproblem_config():
    optimizer = FrankWolfeOptimizer(
        _SimpleProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 2,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "opt_type": "FW",
            "linearization_subproblem": _subproblem_params(
                learning_rate=4e-3,
                outer_iterations=9,
                inner_iterations=10,
                dual_learning_rate=5e-2,
            ),
        },
    )
    assert optimizer.linearization_subproblem_params["learning_rate"] == pytest.approx(4e-3)
    assert optimizer.linearization_subproblem_params["outer_iterations"] == 9
    assert optimizer.linearization_subproblem_params["dual_learning_rate"] == pytest.approx(5e-2)


def test_fw_optimizer_requires_explicit_linearization_subproblem_config():
    with pytest.raises(KeyError, match="linearization_subproblem"):
        FrankWolfeOptimizer(
            _SimpleProblem(),
            {
                "learning_rate": 0.1,
                "max_iterations": 2,
                "max_running_time": 10,
                "convergence_threshold": 1e-6,
                "stepsize_rule": "constant",
                "opt": "gd",
                "momentum": 0.5,
                "lr_decay": 0.9,
                "min_lr": 1e-6,
                "opt_type": "FW",
            },
        )


def test_fw_optimizer_stops_on_zero_frank_wolfe_gap(monkeypatch):
    monkeypatch.setattr(oracle_helpers, "_cached_exact_solver", lambda *args, **kwargs: object())

    def _fake_solve_exact_result(*args, **kwargs):
        del kwargs
        x_init = args[2]
        return {"solution": x_init, "runtime_total": 0.0, "status": "optimal"}

    monkeypatch.setattr(oracle_helpers, "solve_exact_result", _fake_solve_exact_result)
    optimizer = FrankWolfeOptimizer(
        _FakeFWConvexProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 5,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "opt_type": "FW",
            "linearization_subproblem": _subproblem_params(),
        },
    )

    _, _, objective_traj, violation_traj, per_iter_time = optimizer.optimize(
        initial_point=torch.tensor([[1.0, 0.0]]),
        verbose=False,
        seed=0,
    )

    assert len(per_iter_time) == 0
    assert objective_traj.numel() == 1
    assert violation_traj.max().item() == pytest.approx(0.0)


def test_fw_optimizer_verbose_formats_tensor_objective(monkeypatch, capsys):
    monkeypatch.setattr(oracle_helpers, "_cached_exact_solver", lambda *args, **kwargs: object())

    def _fake_solve_exact_result(*args, **kwargs):
        del args, kwargs
        return {"solution": [0.0, 0.0], "runtime_total": 0.0, "status": "optimal"}

    monkeypatch.setattr(oracle_helpers, "solve_exact_result", _fake_solve_exact_result)
    optimizer = FrankWolfeOptimizer(
        _FakeFWConvexProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 1,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "opt_type": "FW",
            "linearization_subproblem": _subproblem_params(),
        },
    )

    optimizer.optimize(
        initial_point=torch.tensor([[1.0, 0.0]]),
        verbose=True,
        seed=0,
    )

    output = capsys.readouterr().out
    assert "outer" in output
    assert "obj" in output
    assert "eqvio" in output
    assert "ineqvio" in output


def test_fw_optimizer_uses_canonical_linearization_subproblem_config():
    optimizer = FrankWolfeOptimizer(
        _SimpleProblem(),
        {
            "learning_rate": 0.1,
            "max_iterations": 2,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "stepsize_rule": "constant",
            "opt": "gd",
            "momentum": 0.0,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "opt_type": "FW",
            "linearization_subproblem": _subproblem_params(
                learning_rate=6e-3,
                outer_iterations=11,
                inner_iterations=12,
                dual_learning_rate=7e-2,
            ),
        },
    )
    assert optimizer.linearization_subproblem_params["learning_rate"] == pytest.approx(6e-3)
    assert optimizer.linearization_subproblem_params["outer_iterations"] == 11
    assert optimizer.linearization_subproblem_params["dual_learning_rate"] == pytest.approx(7e-2)


def test_run_algorithm_returns_unified_payload_for_pgd():
    params = {
        "common": {"seed": 0},
        "PGD": {
            "learning_rate": 0.1,
            "max_iterations": 2,
            "max_running_time": 10,
            "convergence_threshold": 1e-6,
            "acceleration_method": "none",
            "acceleration_space": "x",
            "stepsize_rule": "diminish",
            "opt": "gd",
            "momentum": 0.0,
            "lr_decay": 0.9,
            "min_lr": 1e-6,
            "projection_subproblem": _subproblem_params(),
        },
    }
    result = run_algorithm("PGD", _SimpleProblem(), params)
    assert set(result) == {
        "x_traj",
        "obj_traj",
        "cons_traj",
        "iter_time",
        "last_trans_time",
        "x_solved",
        "total_wall_time",
        "eq_violation_traj",
        "ineq_violation_traj",
        "full_violation_traj",
    }
    assert result["x_solved"].shape == (1, 2)
    assert result["obj_traj"].shape[0] == len(result["iter_time"]) + 1
    assert result["cons_traj"].shape[0] == len(result["iter_time"]) + 1
    assert result["eq_violation_traj"].shape[0] == len(result["iter_time"]) + 1
    assert result["ineq_violation_traj"].shape[0] == len(result["iter_time"]) + 1
    assert result["full_violation_traj"].shape[0] == len(result["iter_time"]) + 1
    assert result["total_wall_time"] >= 0.0


def test_run_algorithm_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        run_algorithm("UNKNOWN", _SimpleProblem(), {"common": {"seed": 0}})


def test_pgd_optimizer_rejects_unknown_stepsize_rule():
    with pytest.raises(ValueError, match="Unsupported stepsize_rule"):
        PGDOptimizer(
            _SimpleProblem(),
            {
                "learning_rate": 0.1,
                "max_iterations": 2,
                "max_running_time": 10,
                "convergence_threshold": 1e-6,
                "acceleration_method": "none",
                "acceleration_space": "x",
                "stepsize_rule": "fixed",
                "opt": "gd",
                "momentum": 0.0,
                "lr_decay": 0.9,
                "min_lr": 1e-6,
                "projection_subproblem": _subproblem_params(),
            },
        )
