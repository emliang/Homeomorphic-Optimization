import torch

from homopt.problems import NonConvexQCProblem, QCOpt, create_test_QC_problem


class _IdentityMap:
    def forward(self, z, method="autograd", return_state=False):
        del method
        if return_state:
            return z, None
        return z

    def vjp(self, z, grad_x, method="autograd", state=None):
        del z, method, state
        return grad_x


def test_create_test_qc_problem_smoke():
    config = create_test_QC_problem({
        'seed': 0,
        'n_var': 3,
        'n_linear_cons': 1,
        'n_qua_cons': 2,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
        'nonconvex_ratio': 0.5,
        'R': 2,
    })
    assert 'Qq' in config and config['Qq'].shape[0] == 2


def test_qcopt_objective_and_constraints_smoke():
    config = create_test_QC_problem({
        'seed': 1,
        'n_var': 3,
        'n_linear_cons': 1,
        'n_qua_cons': 1,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
        'nonconvex_ratio': 0.5,
        'R': 2,
    })
    problem = QCOpt(config)
    x = torch.zeros(1, problem.nvar)
    assert problem.objective_x(x).shape == (1, 1)
    assert problem.constraint_x(x).ndim == 2


def test_qcopt_explicit_lagrangian_gradient_matches_autograd():
    config = create_test_QC_problem({
        'seed': 2,
        'n_var': 3,
        'n_linear_cons': 1,
        'n_qua_cons': 2,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
        'nonconvex_ratio': 0.5,
        'R': 2,
    })
    problem = QCOpt(config)
    x = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float32)
    dual = torch.linspace(0.1, 0.9, problem.ncon, dtype=torch.float32).view(1, -1)
    x_outer = torch.zeros_like(x)
    explicit = problem.gradient_lagrangian_x(
        x,
        dual,
        penalty_coef=1.7,
        proximal_coef=0.3,
        x_outer=x_outer,
        method="explicit",
    )
    autograd = problem.gradient_lagrangian_x(
        x,
        dual,
        penalty_coef=1.7,
        proximal_coef=0.3,
        x_outer=x_outer,
        method="autograd",
    )
    assert torch.allclose(explicit, autograd, atol=1e-5, rtol=1e-5)


def test_qcopt_explicit_lagrangian_gradient_matches_autograd_for_nonsymmetric_matrices():
    config = create_test_QC_problem({
        'seed': 3,
        'n_var': 3,
        'n_linear_cons': 1,
        'n_qua_cons': 2,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
        'nonconvex_ratio': 0.5,
        'R': 2,
    })
    config['Q'] = config['Q'].copy()
    config['Qq'] = config['Qq'].copy()
    config['Q'][0, 1] += 0.7
    config['Qq'][0, 1, 2] -= 0.4
    problem = QCOpt(config)
    x = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float32)
    dual = torch.linspace(0.1, 0.9, problem.ncon, dtype=torch.float32).view(1, -1)
    explicit = problem.gradient_lagrangian_x(
        x,
        dual,
        penalty_coef=1.7,
        method="explicit",
    )
    autograd = problem.gradient_lagrangian_x(
        x,
        dual,
        penalty_coef=1.7,
        method="autograd",
    )
    assert torch.allclose(explicit, autograd, atol=1e-5, rtol=1e-5)


def test_qcopt_explicit_objective_z_gradient_matches_autograd():
    config = create_test_QC_problem({
        'seed': 4,
        'n_var': 3,
        'n_linear_cons': 1,
        'n_qua_cons': 2,
        'obj': 'quad',
        'x_lower': -1,
        'x_upper': 1,
        'nonconvex_ratio': 0.5,
        'R': 2,
    })
    problem = QCOpt(config)
    mapping = _IdentityMap()
    z = torch.tensor([[0.2, -0.1, 0.25]], dtype=torch.float32)
    x, state = mapping.forward(z, method="explicit", return_state=True)

    explicit = problem.gradient_objective_z(
        z,
        mapping,
        method="explicit",
        hom_map_method="explicit",
        x=x,
        hom_state=state,
    )
    autograd = problem.gradient_objective_z(z, mapping, method="autograd", hom_map_method="autograd")

    assert torch.allclose(explicit, autograd, atol=1e-5, rtol=1e-5)


def test_nonconvex_problem_generates_samples():
    generator = NonConvexQCProblem({
        'n_var': 3,
        'n_qua_cons': 2,
        'n_linear_cons': 1,
        'rank': 2,
        'constraint_convexity': False,
        'objective_convexity': False,
        'obj': 'quad',
        'nonconvex_ratio': 0.5,
        'x_lower': -1,
        'x_upper': 1,
        'R': 2,
        'seed': 0,
    })
    inp, obj = generator.generate_problem_samples(n_samples=4, seed=0, sample_obj=True)
    assert inp.shape[0] == 4
    assert obj.shape[0] == 4
