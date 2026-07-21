import numpy as np
import torch

import homopt.experiments.benchmarks.jcc.setup as jcc_setup
from homopt.problems import JCCDCOPFProblem, JCCIMProblem, JCCLinearProblem


def test_jcc_problem_generates_reproducible_scenarios():
    config = {"n_scenarios": 3, "seed": 123}
    problem_a = JCCDCOPFProblem(num_bus=30, config=config)
    problem_b = JCCDCOPFProblem(num_bus=30, config=config)
    assert np.allclose(problem_a.demand_scenarios, problem_b.demand_scenarios)


def test_jcc_problem_uses_generation_dimension_as_nvar():
    problem = JCCDCOPFProblem(num_bus=30, config={"n_scenarios": 2, "seed": 7}).to_device(torch.device("cpu"))
    assert problem.nvar == problem.n_gen_vars
    x = torch.zeros(1, problem.nvar, dtype=torch.float32)
    obj = problem.objective_x(x)
    cons = problem.constraint_x(x)
    feas = problem.compute_scenario_feasibility(x)
    assert obj.shape == (1, 1)
    assert cons.shape == (1, 1)
    assert feas.shape == (1, problem.n_scenarios)
    assert torch.isfinite(obj).all()
    assert torch.isfinite(cons).all()


def test_jcc_builder_forwards_normalized_pglib_loader_controls(monkeypatch, tmp_path):
    captured = {}

    class _FakeJCCProblem:
        def __init__(self, num_bus, config):
            captured["num_bus"] = num_bus
            captured["config"] = dict(config)

        def to_device(self, device):
            self.device = device
            return self

    monkeypatch.setattr(jcc_setup, "JCCDCOPFProblem", _FakeJCCProblem)

    problem = jcc_setup._build_jcc_problem(
        num_bus=30,
        n_scenarios=2,
        epsilon=0.1,
        demand_std=0.05,
        seed=7,
        runtime_device=torch.device("cpu"),
        runtime_dtype=torch.float64,
        problem_config={
            "pglib_case_name": "pglib_opf_case30_ieee",
            "pglib_data_dir": tmp_path,
            "download_if_missing": False,
        },
    )

    assert problem.dtype is torch.float64
    assert captured["num_bus"] == 30
    assert captured["config"]["pglib_case_name"] == "pglib_opf_case30_ieee"
    assert captured["config"]["pglib_data_dir"] == str(tmp_path)
    assert captured["config"]["download_if_missing"] is False


def test_jcc_linear_problem_generates_reproducible_instance_samples():
    config = {"n_var": 5, "n_input_dim": 3, "n_ineq": 2, "n_scenarios": 4, "seed": 123}
    problem_a = JCCLinearProblem(config=config)
    problem_b = JCCLinearProblem(config=config)
    batch_a = problem_a.sample_instance_batch(n_instances=3, seed=17)
    batch_b = problem_b.sample_instance_batch(n_instances=3, seed=17)
    assert np.allclose(batch_a.inputs.detach().cpu().numpy(), batch_b.inputs.detach().cpu().numpy())


def test_jcc_linear_problem_supports_parametric_learning_contract():
    problem = JCCLinearProblem(
        config={"n_var": 6, "n_input_dim": 3, "n_ineq": 2, "n_scenarios": 5, "epsilon": 0.2, "seed": 7}
    ).to_device(torch.device("cpu"))
    batch = problem.sample_instance_batch(n_instances=2, seed=11)
    x = batch.inputs
    y = problem.fixed_x0.view(1, -1).expand(2, -1).clone()
    obj = problem.objective_xy(x, y)
    cons = problem.constraint_residual_xy(x, y, clip=False)
    feas = problem.compute_scenario_feasibility(x, y)
    bound = problem.build_instance(x[0])
    bound_cons = bound.constraint_x(y[:1], clip=True)

    assert obj.shape == (2, 1)
    assert cons.shape[0] == 2
    assert cons.shape[1] == problem.n_ineq + 2 * problem.nvar + 1
    assert feas.shape == (2, problem.n_scenarios)
    assert bound_cons.shape == (1, problem.n_ineq + 2 * problem.nvar + 1)
    assert torch.isfinite(obj).all()
    assert torch.isfinite(cons).all()


def test_jccim_problem_supports_parametric_learning_contract():
    problem = JCCIMProblem(
        config={"n_var": 6, "n_eq": 3, "n_ineq": 2, "n_scenarios": 5, "epsilon": 0.1, "seed": 9}
    ).to_device(torch.device("cpu"))
    batch = problem.sample_instance_batch(n_instances=2, seed=13)
    x = batch.inputs
    y = problem.fixed_x0.view(1, -1).expand(2, -1).clone()
    obj = problem.objective_xy(x, y)
    cons = problem.constraint_residual_xy(x, y, clip=False)
    feas = problem.compute_scenario_feasibility(x, y)
    bound = problem.build_instance(x[0])
    bound_cons = bound.constraint_x(y[:1], clip=True)

    assert problem.n_eq == 3
    assert obj.shape == (2, 1)
    assert cons.shape[0] == 2
    assert cons.shape[1] == problem.n_ineq + 2 * problem.nvar + 1
    assert feas.shape == (2, problem.n_scenarios)
    assert bound_cons.shape == (1, problem.n_ineq + 2 * problem.nvar + 1)
    assert torch.isfinite(obj).all()
    assert torch.isfinite(cons).all()
