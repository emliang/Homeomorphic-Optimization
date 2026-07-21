import torch

from homopt.learning.refiners.common import inverse_scale_decision
from homopt.problems.acopf import DifferentiablePowerFlowSolver
from homopt.problems import ParametricACOPFProblem, default_acopf_dataset_path
from homopt.problems.opf_cases import load_pglib_opf_case, resolve_pglib_opf_case_path


def _write_tiny_pglib_case(path):
    path.write_text(
        "\n".join(
            [
                "function mpc = pglib_opf_case30_ieee",
                "mpc.version = '2';",
                "mpc.baseMVA = 100;",
                "mpc.bus = [",
                "1 3 0 0 0 0 1 1.0 0 230 1 1.1 0.9;",
                "2 1 10 3 0 0 1 1.0 0 230 1 1.1 0.9;",
                "];",
                "mpc.gen = [",
                "1 20 0 30 -30 1 100 1 100 0;",
                "];",
                "mpc.branch = [",
                "1 2 0.01 0.1 0 100 100 100 0 0 1 -360 360;",
                "];",
                "mpc.gencost = [",
                "2 0 0 3 0.01 1 0;",
                "];",
            ]
        )
    )


def test_pglib_opf_case_loader_reads_cached_matpower_case(tmp_path):
    case_path = tmp_path / "pglib_opf_case30_ieee.m"
    _write_tiny_pglib_case(case_path)

    path = resolve_pglib_opf_case_path(30, data_dir=tmp_path, download_if_missing=False)
    case = load_pglib_opf_case(30, data_dir=tmp_path, download_if_missing=False)

    assert path.name == "pglib_opf_case30_ieee.m"
    assert case["bus"].shape[0] == 2
    assert case["gen"].shape[1] >= 10
    assert case["branch"].shape[1] >= 13


def test_acopf_problem_builds_parametric_samples_from_pglib_case(tmp_path):
    _write_tiny_pglib_case(tmp_path / "pglib_opf_case30_ieee.m")
    problem = ParametricACOPFProblem(
        case=30,
        dataset_samples=5,
        pglib_data_dir=tmp_path,
        download_if_missing=False,
    )

    assert problem.nb == 2
    assert problem.input_samples.shape == (5, 2 * problem.nb)
    assert problem.reference_y is None


def test_acopf_power_balance_jacobian_matches_autograd():
    problem = ParametricACOPFProblem(default_acopf_dataset_path(case=30), dtype=torch.float64)
    input_params = problem.input_samples[:1]
    y = problem.reference_y[:1].detach().clone().requires_grad_(True)

    jacobian = problem.equality_jacobian_y(y)

    def residual_fn(flat_y):
        return problem.eq_resid(input_params, flat_y.view(1, -1)).view(-1)

    autograd_jacobian = torch.autograd.functional.jacobian(residual_fn, y.view(-1))

    assert jacobian.shape == (1, problem.neq, problem.nvar)
    assert torch.max(torch.abs(jacobian[0] - autograd_jacobian)).item() < 1e-9


def test_acopf_equality_penalty_gradient_matches_autograd():
    problem = ParametricACOPFProblem(default_acopf_dataset_path(case=30), dtype=torch.float64)
    input_params = problem.input_samples[:1]
    y = problem.reference_y[:1].detach().clone().requires_grad_(True)

    explicit_grad = problem.equality_penalty_gradient_y(input_params, y)
    loss = problem.eq_resid(input_params, y).pow(2).sum()
    autograd_grad = torch.autograd.grad(loss, y)[0]

    assert explicit_grad.shape == (1, problem.nvar)
    assert torch.max(torch.abs(explicit_grad - autograd_grad)).item() < 1e-9


def test_acopf_generator_incidence_supports_repeated_generator_buses():
    problem = ParametricACOPFProblem(default_acopf_dataset_path(case=30), dtype=torch.float64)
    problem.gen_bus_idx = torch.tensor([0, 0], device=problem.device, dtype=torch.long)
    problem.ng = 2
    problem.nvar = 2 * problem.ng + 2 * problem.nb
    problem.ydim = problem.nvar
    problem.qg_start_yidx = problem.ng
    problem.vm_start_yidx = 2 * problem.ng
    problem.va_start_yidx = 2 * problem.ng + problem.nb
    problem.generator_bus_incidence = torch.zeros(problem.nb, problem.ng, device=problem.device, dtype=problem.dtype)
    problem.generator_bus_incidence.index_put_(
        (problem.gen_bus_idx, torch.arange(problem.ng, device=problem.device)),
        torch.ones(problem.ng, device=problem.device, dtype=problem.dtype),
    )

    y = torch.cat(
        [
            torch.tensor([[0.2, 0.3, 0.1, 0.4]], device=problem.device, dtype=problem.dtype),
            problem.vm_init.view(1, -1),
            problem.va_init.view(1, -1),
        ],
        dim=1,
    )
    jacobian = problem.power_balance_jacobian_y(y)

    assert jacobian[0, 0, 0].item() == 1.0
    assert jacobian[0, 0, 1].item() == 1.0
    assert jacobian[0, problem.nb, 2].item() == 1.0
    assert jacobian[0, problem.nb, 3].item() == 1.0


def test_acopf_differentiable_pf_completion_reconstructs_power_balance():
    problem = ParametricACOPFProblem(default_acopf_dataset_path(case=30), dtype=torch.float64)
    solver = DifferentiablePowerFlowSolver(problem, {"max_iters": 12, "tol": 1e-10, "damping": 1e-8})
    input_params = problem.input_samples[:2]
    partial = solver.extract_partial(problem.reference_y[:2])

    completed = solver.complete(input_params, partial)
    residual = problem.eq_resid(input_params, completed)

    assert completed.shape == (2, problem.nvar)
    assert residual.abs().max().item() < 1e-5


def test_acopf_pf_completion_gradient_modes_backpropagate_to_partial():
    problem = ParametricACOPFProblem(default_acopf_dataset_path(case=30), dtype=torch.float64)
    input_params = problem.input_samples[:1]

    for gradient_mode in ("unrolled", "last_step", "implicit"):
        solver = DifferentiablePowerFlowSolver(
            problem,
            {
                "max_iters": 2,
                "tol": 0.0,
                "damping": 1e-5,
                "gradient_mode": gradient_mode,
            },
        )
        partial = solver.extract_partial(problem.reference_y[:1]).detach().clone().requires_grad_(True)
        completed = solver.complete(input_params, partial)
        loss = problem.objective_xy(input_params, completed).sum() + problem.eq_resid(input_params, completed).pow(2).sum()
        loss.backward()

        assert partial.grad is not None
        assert torch.isfinite(partial.grad).all()


def test_acopf_pf_completion_implicit_mode_reconstructs_power_balance():
    problem = ParametricACOPFProblem(default_acopf_dataset_path(case=57), dtype=torch.float64)
    solver = DifferentiablePowerFlowSolver(
        problem,
        {
            "max_iters": 12,
            "tol": 1e-10,
            "damping": 1e-8,
            "gradient_mode": "implicit",
        },
    )
    input_params = problem.input_samples[:1]
    partial = solver.extract_partial(problem.reference_y[:1])

    completed = solver.complete(input_params, partial)
    residual = problem.eq_resid(input_params, completed)

    assert residual.abs().max().item() < 1e-5


def test_acopf_problem_pf_partial_prediction_space_completes_partial_decisions():
    problem = ParametricACOPFProblem(
        default_acopf_dataset_path(case=57),
        dtype=torch.float64,
        prediction_space="pf_partial",
        completion_config={
            "gradient_mode": "implicit",
            "max_iters": 12,
            "tol": 1e-10,
            "damping": 1e-8,
        },
    )
    input_params = problem.input_samples[:1]
    partial = problem.power_flow_completion.extract_partial(problem.reference_y[:1])
    normalized = problem.inverse_scale(input_params, partial)
    scaled = problem.scale(input_params, normalized)
    completed = problem.complete_partial(input_params, scaled)

    assert problem.partial_dim == problem.power_flow_completion.partial_dim
    assert normalized.shape[-1] == problem.partial_dim
    assert completed.shape == (1, problem.nvar)
    assert problem.eq_resid(input_params, completed).abs().max().item() < 1e-5


def test_acopf_prediction_spaces_expose_normalized_bounds():
    full_problem = ParametricACOPFProblem(default_acopf_dataset_path(case=30), dtype=torch.float64)
    full_input = full_problem.input_samples[:1]
    full_lower = full_problem.scale(full_input, -torch.ones(1, full_problem.nvar, dtype=full_problem.dtype))
    full_upper = full_problem.scale(full_input, torch.ones(1, full_problem.nvar, dtype=full_problem.dtype))

    assert torch.allclose(full_lower, full_problem.L.view(1, -1))
    assert torch.allclose(full_upper, full_problem.U.view(1, -1))
    free_full = ((full_problem.U - full_problem.L) > torch.finfo(full_problem.dtype).eps).view(1, -1)
    full_lower_normalized = full_problem.inverse_scale(full_input, full_lower)
    full_upper_normalized = full_problem.inverse_scale(full_input, full_upper)
    assert torch.isfinite(full_lower_normalized).all()
    assert torch.isfinite(full_upper_normalized).all()
    assert torch.allclose(full_lower_normalized[free_full], -torch.ones_like(full_lower_normalized[free_full]))
    assert torch.allclose(full_upper_normalized[free_full], torch.ones_like(full_upper_normalized[free_full]))

    partial_problem = ParametricACOPFProblem(
        default_acopf_dataset_path(case=57),
        dtype=torch.float64,
        prediction_space="pf_partial",
        completion_config={
            "gradient_mode": "implicit",
            "max_iters": 2,
            "tol": 0.0,
            "damping": 1e-5,
        },
    )
    partial_input = partial_problem.input_samples[:1]
    partial_lower, partial_upper = partial_problem.power_flow_completion.partial_bounds()
    scaled_lower = partial_problem.scale(
        partial_input,
        -torch.ones(1, partial_problem.partial_dim, dtype=partial_problem.dtype),
    )
    scaled_upper = partial_problem.scale(
        partial_input,
        torch.ones(1, partial_problem.partial_dim, dtype=partial_problem.dtype),
    )

    assert torch.allclose(scaled_lower, partial_lower.view(1, -1))
    assert torch.allclose(scaled_upper, partial_upper.view(1, -1))
    free_partial = ((partial_upper - partial_lower) > torch.finfo(partial_problem.dtype).eps).view(1, -1)
    partial_lower_normalized = partial_problem.inverse_scale(partial_input, scaled_lower)
    partial_upper_normalized = partial_problem.inverse_scale(partial_input, scaled_upper)
    assert torch.isfinite(partial_lower_normalized).all()
    assert torch.isfinite(partial_upper_normalized).all()
    assert torch.allclose(
        partial_lower_normalized[free_partial],
        -torch.ones_like(partial_lower_normalized[free_partial]),
    )
    assert torch.allclose(
        partial_upper_normalized[free_partial],
        torch.ones_like(partial_upper_normalized[free_partial]),
    )
    assert torch.allclose(
        inverse_scale_decision(partial_problem, partial_input, scaled_lower)[free_partial],
        -torch.ones_like(partial_lower_normalized[free_partial]),
    )
