import torch

import homopt.learning.training.inn as inn_training
import homopt.models.inn as inn_impl
from homopt.learning.training.inn import unsupervised_training_mdh
from homopt.models import CvxINN, INN, MLP, PINN
from homopt.optim import INNPGDOptimizer, pgd_transformed_space
from homopt.mappings.gauge import GaugeMap
from homopt.models.flows.coupling import CombinedActNormLU


class IdentityModel(torch.nn.Module):
    def forward(self, z, input_params):
        return z


class IdentityInvertible(torch.nn.Module):
    def forward(self, z, c=None):
        del c
        return z

    def inverse(self, x, c=None):
        del c
        return x


class DummyBoxSet:
    def __init__(self):
        self.nvar = 2
        self.G = None
        self.h = None
        self.C = None
        self.d = None
        self.Qq = None
        self.pq = None
        self.bq = None
        self.A = None
        self.b = None
        self.L = torch.tensor([-1.0, -1.0], dtype=torch.float32)
        self.U = torch.tensor([1.0, 1.0], dtype=torch.float32)

    def constraint_x(self, x):
        upper = x - self.U.view(1, -1)
        lower = self.L.view(1, -1) - x
        return torch.cat([upper, lower], dim=1)


class DummyProblem:
    def __init__(self):
        self.nvar = 2
        self.device = torch.device('cpu')

    def scale(self, input_params, x):
        return x

    def complete_partial(self, input_params, x):
        return x

    def objective(self, x):
        return (x ** 2).sum(dim=1)

    def violations(self, input_params, x):
        return torch.zeros(x.shape[0], device=x.device)

    def check_feasibility(self, input_params, x):
        return torch.zeros(x.shape[0], 1, device=x.device)


class DummyProblemNeedsProjection(DummyProblem):
    def objective(self, x):
        return ((x - 1.0) ** 2).sum(dim=1)

    def violations(self, input_params, x):
        del input_params
        return torch.clamp(x[:, :1] - 0.05, min=0.0).view(-1)

    def check_feasibility(self, input_params, x):
        del input_params
        return torch.clamp(x[:, :1] - 0.05, min=0.0)


class ShiftedHalfspaceProblem(DummyProblem):
    """A feasible region that excludes the INN latent origin."""

    def objective(self, x):
        return ((x + 1.0) ** 2).sum(dim=1)

    def violations(self, input_params, x):
        del input_params
        return torch.clamp(0.5 - x[:, :1], min=0.0).view(-1)

    def check_feasibility(self, input_params, x):
        del input_params
        return torch.clamp(0.5 - x[:, :1], min=0.0)


class ShiftedQuadraticHalfspaceProblem(ShiftedHalfspaceProblem):
    def __init__(self):
        super().__init__()
        self.fixed_Q = torch.eye(self.nvar)
        self.fixed_p = torch.ones(self.nvar)


class DummyTrainingData(DummyProblem):
    def ineq_resid(self, input_params, x):
        del input_params
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

    def generate_problem_samples_torch(self, n_samples, sample_obj=False, device=None, dtype=None):
        del sample_obj
        return torch.zeros(n_samples, 1, device=device, dtype=dtype), None


class DummyMDH(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x, input_params):
        del input_params
        batch_size = x.shape[0]
        values = self.weight.expand(batch_size)
        return x * self.weight, values, values


def test_inn_public_exports_available():
    assert INN is not None
    assert CvxINN is not None
    assert MLP is not None
    assert PINN is not None


def test_combined_actnorm_lu_roundtrip_shape():
    layer = CombinedActNormLU(3)
    x = torch.randn(4, 3)
    y, _ = layer(x)
    x_back, _ = layer(y, mode='inverse')
    assert y.shape == x.shape
    assert x_back.shape == x.shape
    assert torch.isfinite(x_back).all()


def test_inn_forward_and_inverse_smoke():
    model = INN(nin=2, nhid=8, cin=3, nl=1, inv='coupling', Con_type='MLP')
    c = torch.randn(5, 3)
    x = torch.randn(5, 2)

    model.train()
    y, logdet, logdis = model(x, c)
    assert y.shape == x.shape
    assert logdet.shape == torch.Size([5])
    assert logdis.shape == torch.Size([5])

    model.eval()
    y_eval = model(x, c)
    x_back = model.inverse(y_eval, c)
    assert y_eval.shape == x.shape
    assert x_back.shape == x.shape
    assert torch.isfinite(x_back).all()


def test_mdh_training_records_lipschitz_max_term(tmp_path):
    model = DummyMDH()
    x_tensor = torch.zeros(4, 2)
    input_tensor = torch.zeros(4, 1)
    args = {
        "batch_size": 2,
        "total_iteration": 1,
        "w_penalty": 0.0,
        "w_distortion": 0.0,
        "w_lipschitz": 1.0,
        "center": 0.0,
        "lr": 1e-3,
        "lr_decay_step": 10,
        "lr_decay": 1.0,
        "ema_decay": 0.0,
        "resultsSaveFreq": 10,
        "record_every": 1,
        "device": "cpu",
        "dtype": torch.float32,
    }

    _, record = unsupervised_training_mdh(model, DummyTrainingData(), x_tensor, input_tensor, tmp_path, args)

    assert record["w_lipschitz"] == 1.0
    assert len(record["lipschitz_list"]) == 1
    assert record["lipschitz_list"][0] == 1.0


def test_mdh_training_progress_is_opt_in(monkeypatch, tmp_path):
    calls = []

    def _fake_tqdm(iterable, **kwargs):
        calls.append(kwargs.get("disable"))
        return iterable

    monkeypatch.setattr(inn_training, "tqdm", _fake_tqdm)

    base_args = {
        "batch_size": 2,
        "total_iteration": 1,
        "w_penalty": 0.0,
        "w_distortion": 0.0,
        "w_lipschitz": 1.0,
        "center": 0.0,
        "lr": 1e-3,
        "lr_decay_step": 10,
        "lr_decay": 1.0,
        "ema_decay": 0.0,
        "resultsSaveFreq": 10,
        "record_every": 1,
        "device": "cpu",
        "dtype": torch.float32,
    }
    x_tensor = torch.zeros(4, 2)
    input_tensor = torch.zeros(4, 1)

    unsupervised_training_mdh(DummyMDH(), DummyTrainingData(), x_tensor, input_tensor, tmp_path, dict(base_args))
    unsupervised_training_mdh(
        DummyMDH(),
        DummyTrainingData(),
        x_tensor,
        input_tensor,
        tmp_path,
        {**base_args, "show_progress": True},
    )

    assert calls == [True, False]


def test_coninn_wraps_unconstrained_model_into_exact_convex_homeomorphism():
    model = CvxINN(IdentityInvertible(), GaugeMap(DummyBoxSet(), p_norm=2))
    z = torch.tensor([[0.2, -0.3], [0.5, 0.1]], dtype=torch.float32)

    x = model(z)
    z_back = model.inverse(x)

    cons = DummyBoxSet().constraint_x(x)
    assert x.shape == z.shape
    assert torch.all(cons <= 1e-5)
    assert torch.allclose(z_back, z, atol=1e-5)


def test_pinn_output_shape():
    model = PINN(input_dim=2, hidden_dims=8, output_dim=4, num_layer=1)
    x = torch.randn(6, 10)
    y = model(x)
    assert y.shape == (6, 4)


def test_inn_pgd_optimizer_smoke():
    problem = DummyProblem()
    model = IdentityModel()
    optimizer = INNPGDOptimizer(
        problem=problem,
        paras={
            'learning_rate': 0.1,
            'max_iterations': 3,
            'max_running_time': 10,
            'convergence_threshold': 1e-6,
            'lr_decay': 0.9,
            'min_lr': 1e-6,
            'feasibility_eps': 1e-6,
            'opt': 'gd',
            'momentum': 0.0,
            'stepsize_rule': 'diminish',
        },
        model=model,
    )
    input_params = torch.zeros(1, 1)
    x_opt, decision_traj, latent_traj, objective_traj, violation_traj, per_iter_time = optimizer.optimize(input_params=input_params)
    assert x_opt.shape == (1, 2)
    assert decision_traj.shape[1] == 2
    assert latent_traj.shape[1] == 2
    assert objective_traj.numel() >= 1
    assert violation_traj.numel() >= 1
    assert len(per_iter_time) >= 1
    assert len(per_iter_time) <= 3
    assert objective_traj.numel() == len(per_iter_time) + 1
    assert violation_traj.numel() == len(per_iter_time) + 1


def test_inn_pgd_optimizer_projection_path_runs_multiple_steps():
    problem = DummyProblemNeedsProjection()
    model = IdentityModel()
    optimizer = INNPGDOptimizer(
        problem=problem,
        paras={
            'learning_rate': 0.1,
            'max_iterations': 2,
            'max_running_time': 10,
            'convergence_threshold': 1e-6,
            'lr_decay': 0.9,
            'min_lr': 1e-6,
            'feasibility_eps': 1e-6,
            'opt': 'gd',
            'momentum': 0.0,
            'stepsize_rule': 'constant',
            'proj_max_steps': 2,
            'proj_eps': 1e-6,
            'step_size': 0.5,
        },
        model=model,
    )
    input_params = torch.zeros(1, 1)
    x_opt, _, _, objective_traj, violation_traj, per_iter_time = optimizer.optimize(input_params=input_params)
    assert x_opt.shape == (1, 2)
    assert len(per_iter_time) == 2
    assert objective_traj.numel() == 3
    assert violation_traj.numel() == 3


def test_inn_pgd_projects_from_the_current_feasible_state_when_origin_is_infeasible():
    problem = ShiftedHalfspaceProblem()
    optimizer = INNPGDOptimizer(
        problem=problem,
        paras={
            'learning_rate': 0.5,
            'max_iterations': 1,
            'max_running_time': 10,
            'convergence_threshold': 1e-6,
            'lr_decay': 0.9,
            'min_lr': 1e-6,
            'feasibility_eps': 1e-6,
            'opt': 'gd',
            'momentum': 0.0,
            'stepsize_rule': 'constant',
            'proj_max_steps': 20,
            'proj_eps': 1e-6,
            'step_size': 0.5,
        },
        model=IdentityModel(),
    )

    x_opt, *_ = optimizer.optimize(
        initial_point=torch.ones(1, 2),
        input_params=torch.zeros(1, 1),
    )

    assert problem.check_feasibility(torch.zeros(1, 1), x_opt).max().item() <= 1e-6


def test_inn_pgd_without_a_feasible_anchor_retains_the_lower_violation_state():
    problem = ShiftedHalfspaceProblem()
    optimizer = INNPGDOptimizer(
        problem=problem,
        paras={
            'learning_rate': 0.5,
            'max_iterations': 1,
            'max_running_time': 10,
            'convergence_threshold': 1e-6,
            'lr_decay': 0.9,
            'min_lr': 1e-6,
            'feasibility_eps': 1e-6,
            'opt': 'gd',
            'momentum': 0.0,
            'stepsize_rule': 'constant',
            'proj_max_steps': 20,
            'proj_eps': 1e-6,
            'step_size': 0.5,
        },
        model=IdentityModel(),
    )

    x_opt, *_ = optimizer.optimize(
        initial_point=torch.zeros(1, 2),
        input_params=torch.zeros(1, 1),
    )

    assert torch.allclose(x_opt, torch.zeros_like(x_opt))


def test_transformed_pgd_uses_the_same_safe_anchor_and_fallback_rules():
    problem = ShiftedQuadraticHalfspaceProblem()
    args = {
        'pgd_max_iter': 1,
        'pgd_lr': 1.0,
        'lr_decay': 0.9,
        'min_lr': 1e-6,
        'proj_max_steps': 20,
        'proj_eps': 1e-6,
        'step_size': 0.5,
    }
    input_params = torch.zeros(1, 1)

    projected = pgd_transformed_space(
        IdentityModel(),
        problem,
        input_params,
        args,
        initial_z=torch.ones(1, 2),
    )
    retained = pgd_transformed_space(
        IdentityModel(),
        problem,
        input_params,
        args,
        initial_z=torch.zeros(1, 2),
    )

    assert problem.check_feasibility(input_params, projected['z_final']).max().item() <= 1e-6
    assert torch.allclose(retained['z_final'], torch.zeros_like(retained['z_final']))
