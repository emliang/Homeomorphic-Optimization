"""Joint chance-constrained DC-OPF implementation."""

from __future__ import annotations

import copy
import numpy as np
import torch

from homopt.problems.base import ProblemInstanceBatch, StateBackedParametricProblemBase
from homopt.solvers.core import CVXPY_INTEGER_SOLVER_PRIORITY, CVXPY_SOLVER_PRIORITY, _solve_cvxpy_problem

try:
    import scipy.io
except ImportError as exc:  # pragma: no cover - dependency availability varies by env
    scipy = None
    _SCIPY_IMPORT_ERROR = exc
else:  # pragma: no cover - simple import branch
    _SCIPY_IMPORT_ERROR = None

try:
    from pypower.idx_brch import BR_X, RATE_A
    from pypower.idx_bus import BUS_TYPE, PD, PQ, REF
    from pypower.idx_gen import GEN_BUS, PMAX, PMIN
except ImportError as exc:  # pragma: no cover - dependency availability varies by env
    _PYPOWER_IMPORT_ERROR = exc
else:  # pragma: no cover - simple import branch
    _PYPOWER_IMPORT_ERROR = None


torch.set_default_dtype(torch.float32)
EPSILON = 1e-4
BASEMVA = 100.0


def _require_powerflow_dependencies():
    if _SCIPY_IMPORT_ERROR is not None:
        raise ImportError(
            "JCC-DC-OPF requires scipy. Install with: pip install -e .[research]"
        ) from _SCIPY_IMPORT_ERROR
    if _PYPOWER_IMPORT_ERROR is not None:
        raise ImportError(
            "JCC-DC-OPF requires pypower. Install with: pip install pypower"
        ) from _PYPOWER_IMPORT_ERROR


def _require_cvxpy():
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - dependency availability varies by env
        raise ImportError(
            "JCC-DC-OPF solver requires cvxpy. Install with: pip install -e .[solvers]"
        ) from exc
    return cp


def _solve_jcc_cvxpy_problem(prob, cp_mod, config, *, solver_priority=CVXPY_SOLVER_PRIORITY):
    """Solve package CVXPY models with the project-wide solver priority."""
    _solve_cvxpy_problem(
        prob,
        cp_mod,
        warm_start=True,
        verbose=bool(config.get("verbose", False)),
        solver_options=dict(config.get("solver_options") or {}),
        time_limit_sec=config.get("time_limit_sec", None),
        solver_priority=solver_priority,
    )


def _as_tensor(value, *, device=None):
    if torch.is_tensor(value):
        return value.to(device=device) if device is not None else value
    return torch.tensor(value, dtype=torch.float32, device=device)


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _build_gen_to_bus_matrix(problem):
    gen_to_bus_matrix = np.zeros((problem.n_bus - 1, problem.n_gen_vars))
    gen_col = 0
    for gen_idx, bus_idx in enumerate(problem.gen_bus):
        if gen_idx == problem.slack_gen or not problem.active_gen_mask[gen_idx - (1 if gen_idx > problem.slack_gen else 0)]:
            continue
        if bus_idx != problem.ref_bus:
            reduced_bus_idx = bus_idx if bus_idx < problem.ref_bus else bus_idx - 1
            gen_to_bus_matrix[reduced_bus_idx, gen_col] = 1.0
        gen_col += 1
    return gen_to_bus_matrix


def _build_reduced_demand(problem, demand_scenario_k):
    d_reduced = np.zeros(problem.n_bus - 1)
    for bus_idx, demand_val in enumerate(demand_scenario_k):
        if bus_idx != problem.ref_bus:
            reduced_idx = bus_idx if bus_idx < problem.ref_bus else bus_idx - 1
            d_reduced[reduced_idx] = demand_val
    return d_reduced


def _evaluate_scenario_feasibility(problem, x_opt):
    device = getattr(problem, "device", None)
    x_tensor = _as_tensor(x_opt, device=device).view(1, -1)
    scenario_feasibility = problem.compute_scenario_feasibility(x_tensor)
    return float(scenario_feasibility.mean().item())


class _JCCCVXPYModelBuilder:
    """Shared CVXPY model helper to reduce duplication across JCC baselines."""

    def __init__(self, problem, cp):
        self.problem = problem
        self.cp = cp
        self.P_min = _as_numpy(problem.P_min)
        self.P_max = _as_numpy(problem.P_max)
        self.P_slack_min = _as_numpy(problem.P_slack_min)
        self.P_slack_max = _as_numpy(problem.P_slack_max)
        self.theta_min = _as_numpy(problem.theta_min)
        self.theta_max = _as_numpy(problem.theta_max)
        self.S_max = _as_numpy(problem.S_max)
        self.B_line = _as_numpy(problem.B_line)
        self.P_demand_nominal = _as_numpy(problem.P_demand_nominal)
        self.demand_scenarios = _as_numpy(problem.demand_scenarios)
        self.B_bus_reduced_inv = problem.B_bus_reduced_inv
        self.gen_to_bus_matrix = _build_gen_to_bus_matrix(problem)

    def add_scenario_balance_theta_constraints(self, constraints, P_g, p_slack, theta, scenario_idx):
        total_demand_k = np.sum(self.P_demand_nominal * self.demand_scenarios[scenario_idx])
        constraints.append(p_slack[scenario_idx] == total_demand_k - self.cp.sum(P_g))

        demand_scenario_k = self.P_demand_nominal * self.demand_scenarios[scenario_idx]
        d_reduced = _build_reduced_demand(self.problem, demand_scenario_k)
        p_reduced = self.gen_to_bus_matrix @ P_g
        net_injection = p_reduced - d_reduced
        constraints.append(theta[scenario_idx, :] == self.B_bus_reduced_inv @ net_injection)

    def add_theta_full_and_line_flow(self, constraints, theta, scenario_idx, name_prefix):
        theta_full = self.cp.Variable(self.problem.n_bus, name=f"{name_prefix}_{scenario_idx}")
        constraints.append(theta_full[self.problem.ref_bus] == 0)
        for bus_idx, angle_idx in self.problem.bus_to_angle_idx:
            constraints.append(theta_full[bus_idx] == theta[scenario_idx, angle_idx])
        return self.B_line @ theta_full

    def build_objective(self, cp, P_g, p_slack):
        cost_c2 = _as_numpy(self.problem.cost_c2)
        cost_c1 = _as_numpy(self.problem.cost_c1)
        cost_c2_slack = _as_numpy(self.problem.cost_c2_slack)
        cost_c1_slack = _as_numpy(self.problem.cost_c1_slack)
        total_cost = 0
        for scenario_idx in range(self.problem.n_scenarios):
            non_slack_cost = cp.sum(cp.multiply(cost_c2, cp.square(P_g)) + cp.multiply(cost_c1, P_g))
            slack_cost = cost_c2_slack * cp.square(p_slack[scenario_idx]) + cost_c1_slack * p_slack[scenario_idx]
            total_cost += non_slack_cost + slack_cost
        return cp.Minimize(total_cost / self.problem.n_scenarios / self.problem.n_bus)


class JCCDCOPFProblem(StateBackedParametricProblemBase):
    """Joint chance-constrained DC optimal power flow problem."""

    def __init__(self, num_bus=30, config=None):
        _require_powerflow_dependencies()
        self.ppc = self._load_case_data(num_bus)
        self.case_name = f"{num_bus}_bus_case"
        self.config = {
            "n_scenarios": 20,
            "epsilon": 0.05,
            "demand_std": 0.1,
            "seed": 2025,
            **(config or {}),
        }
        self._extract_case_data()
        self._setup_matrices()
        self._setup_constraints()
        self._generate_scenarios()
        self.is_parametric = True

    @staticmethod
    def _load_case_data(num_bus):
        ppc_mat = scipy.io.loadmat(f"data/acopf/{num_bus}bus_casefile.mat")
        ppc = ppc_mat.get("mpc")
        return {
            "version": int(np.asarray(ppc["version"][0, 0]).item()),
            "baseMVA": float(np.asarray(ppc["baseMVA"][0, 0]).item()),
            "bus": ppc["bus"][0, 0],
            "gen": ppc["gen"][0, 0],
            "branch": ppc["branch"][0, 0],
            "gencost": ppc["gencost"][0, 0],
        }

    def _extract_case_data(self):
        self.bus = self.ppc["bus"].copy()
        self.gen = self.ppc["gen"].copy()
        self.branch = self.ppc["branch"].copy()
        self.gencost = self.ppc["gencost"].copy()

        self.n_bus = self.bus.shape[0]
        self.n_gen = self.gen.shape[0]
        self.n_branch = self.branch.shape[0]
        self.n_scenarios = self.config["n_scenarios"]
        self.n_angle_vars = self.n_bus - 1
        self.ncon = 1
        self.n_eq = 0
        self.ineq_cons = range(self.ncon)
        self.eq_cons = None

        ref_buses = np.where(self.bus[:, BUS_TYPE] == REF)[0]
        self.ref_bus = ref_buses[0] if len(ref_buses) > 0 else 0
        self.non_slack_idx = np.where(self.bus[:, BUS_TYPE] != REF)[0]
        self.slack_gen = np.where(self.gen[:, GEN_BUS] == (self.ref_bus + 1))[0][0]
        self.PQ_bus_flag = (self.bus[:, BUS_TYPE] == PQ).astype(int)
        self.gen_bus = self.gen[:, GEN_BUS].astype(int) - 1

        bus_to_angle_idx = []
        angle_idx = 0
        for bus_idx in range(self.n_bus):
            if bus_idx != self.ref_bus:
                bus_to_angle_idx.append([bus_idx, angle_idx])
                angle_idx += 1
        self.bus_to_angle_idx = np.array(bus_to_angle_idx)

    def _setup_matrices(self):
        self.B_bus = np.zeros((self.n_bus, self.n_bus))
        self.B_line = np.zeros((self.n_branch, self.n_bus))

        for branch_idx, branch in enumerate(self.branch):
            from_bus = int(branch[0]) - 1
            to_bus = int(branch[1]) - 1
            reactance = branch[BR_X]
            susceptance = 1.0 / reactance if reactance != 0 else 1000.0
            self.B_bus[from_bus, from_bus] += susceptance
            self.B_bus[to_bus, to_bus] += susceptance
            self.B_bus[from_bus, to_bus] -= susceptance
            self.B_bus[to_bus, from_bus] -= susceptance
            self.B_line[branch_idx, from_bus] = susceptance
            self.B_line[branch_idx, to_bus] = -susceptance

        self.B_bus_reduced = np.delete(np.delete(self.B_bus, self.ref_bus, axis=0), self.ref_bus, axis=1)
        self.B_line_reduced = np.delete(self.B_line, self.ref_bus, axis=1)
        self.B_bus_tensor = torch.tensor(self.B_bus, dtype=torch.float32)
        self.B_line_tensor = torch.tensor(self.B_line, dtype=torch.float32)
        try:
            self.B_bus_reduced_inv = np.linalg.inv(self.B_bus_reduced)
        except np.linalg.LinAlgError:
            self.B_bus_reduced_inv = np.linalg.pinv(self.B_bus_reduced)
        self.B_bus_reduced_inv_tensor = torch.tensor(self.B_bus_reduced_inv, dtype=torch.float32)

    def _setup_constraints(self):
        non_slack_gen_pmin = np.delete(self.gen[:, PMIN], self.slack_gen) / BASEMVA
        non_slack_gen_pmax = np.delete(self.gen[:, PMAX], self.slack_gen) / BASEMVA
        non_slack_gen_cost = np.delete(self.gencost, self.slack_gen, axis=0)
        self.non_slack_gen_indices = np.delete(np.arange(self.n_gen), self.slack_gen)

        active_gens = non_slack_gen_pmax > 1e-6
        if not np.any(active_gens):
            active_gens[0] = True

        self.active_gen_mask = active_gens
        self.P_min = non_slack_gen_pmin[active_gens]
        self.P_max = non_slack_gen_pmax[active_gens]
        self.n_gen_vars = len(self.P_min)
        self.nvar = self.n_gen_vars

        self.P_slack_min = self.gen[self.slack_gen, PMIN] / BASEMVA
        self.P_slack_max = self.gen[self.slack_gen, PMAX] / BASEMVA
        self.P_demand_nominal = self.bus[:, PD] / BASEMVA
        self.S_max = self.branch[:, RATE_A] / BASEMVA
        self.S_max[self.S_max == 0] = 2.0
        self.theta_min = np.full(self.n_angle_vars, -np.pi / 3)
        self.theta_max = np.full(self.n_angle_vars, np.pi / 3)

        self.cost_c2 = np.maximum(non_slack_gen_cost[active_gens, 4], 0.1)
        self.cost_c1 = non_slack_gen_cost[active_gens, 5]
        self.cost_c2_slack = np.maximum(self.gencost[self.slack_gen, 4], 0.1)
        self.cost_c1_slack = self.gencost[self.slack_gen, 5]

    def _draw_scenarios(self, *, n_instances=1, seed=None):
        rng = np.random.RandomState(self.config["seed"] if seed is None else seed)
        return rng.normal(
            1,
            self.config["demand_std"],
            (int(n_instances), self.n_scenarios, self.n_bus),
        ).astype(np.float32)

    def _generate_scenarios(self):
        self.demand_scenarios = self._draw_scenarios(n_instances=1, seed=self.config["seed"])[0]
        return self.demand_scenarios

    def sample_instances(self, n_samples, seed=2025, **kwargs):
        del kwargs
        return _as_tensor(self._draw_scenarios(n_instances=n_samples, seed=seed), device=getattr(self, "device", None))

    def sample_instance_batch(self, n_instances=1, seed=2025, **kwargs):
        scenarios = self.sample_instances(n_instances, seed=seed, **kwargs)
        return ProblemInstanceBatch(
            family=type(self).__name__,
            inputs=scenarios,
            objectives=None,
            n_instances=int(scenarios.shape[0]),
            metadata={
                "seed": int(seed),
                "n_scenarios": int(self.n_scenarios),
                "num_bus": int(self.n_bus),
                "demand_std": float(self.config["demand_std"]),
            },
        )

    def build_instance(self, input_data, objective_data=None):
        del objective_data
        bound = copy.copy(self)
        scenarios = _as_tensor(input_data, device=getattr(self, "device", None))
        bound.demand_scenarios = scenarios
        bound.n_scenarios = int(scenarios.shape[0])
        bound.config = {**self.config, "n_scenarios": bound.n_scenarios}
        return bound

    def to_device(self, device):
        tensor_attrs = [
            "cost_c2",
            "cost_c1",
            "P_min",
            "P_max",
            "P_demand_nominal",
            "S_max",
            "theta_min",
            "theta_max",
            "B_bus_reduced",
            "B_line_reduced",
            "demand_scenarios",
        ]
        for attr_name in tensor_attrs:
            if hasattr(self, attr_name):
                setattr(self, attr_name, _as_tensor(getattr(self, attr_name), device=device))
        self.B_bus_reduced_inv_tensor = self.B_bus_reduced_inv_tensor.to(device)
        self.B_bus_tensor = self.B_bus_tensor.to(device)
        self.B_line_tensor = self.B_line_tensor.to(device)
        self.cost_c2_slack = _as_tensor(self.cost_c2_slack, device=device)
        self.cost_c1_slack = _as_tensor(self.cost_c1_slack, device=device)
        self.P_slack_min = _as_tensor(self.P_slack_min, device=device)
        self.P_slack_max = _as_tensor(self.P_slack_max, device=device)
        self.device = device
        if hasattr(self, "gen_to_bus_reduced_matrix"):
            self.gen_to_bus_reduced_matrix = self.gen_to_bus_reduced_matrix.to(device)
        if hasattr(self, "demand_reduced_base"):
            self.demand_reduced_base = self.demand_reduced_base.to(device)
        return self

    def _precompute_mapping_matrices(self):
        matrix = np.zeros((self.n_bus - 1, self.n_gen_vars))
        gen_col = 0
        for non_slack_idx, gen_idx in enumerate(self.non_slack_gen_indices):
            if not self.active_gen_mask[non_slack_idx]:
                continue
            bus_idx = self.gen_bus[gen_idx]
            if bus_idx != self.ref_bus:
                reduced_bus_idx = bus_idx if bus_idx < self.ref_bus else bus_idx - 1
                matrix[reduced_bus_idx, gen_col] = 1.0
            gen_col += 1
        self.gen_to_bus_reduced_matrix = torch.tensor(matrix, dtype=torch.float32, device=getattr(self, "device", "cpu"))

        demand_reduced = torch.zeros(self.n_bus - 1, dtype=torch.float32, device=getattr(self, "device", "cpu"))
        P_demand_tensor = _as_tensor(self.P_demand_nominal, device=getattr(self, "device", "cpu"))
        for bus_idx, demand_val in enumerate(P_demand_tensor):
            if bus_idx != self.ref_bus:
                reduced_idx = bus_idx if bus_idx < self.ref_bus else bus_idx - 1
                demand_reduced[reduced_idx] = demand_val
        self.demand_reduced_base = demand_reduced

    def _scenario_tensor(self, scenario_k, device):
        if isinstance(scenario_k, int):
            scenario_tensor = self.demand_scenarios[scenario_k]
        else:
            scenario_tensor = scenario_k
        return _as_tensor(scenario_tensor, device=device)

    def _coerce_scenario_batch(self, input_batch, device, batch_size=None):
        if input_batch is None:
            return None
        scenario_batch = _as_tensor(input_batch, device=device)
        if scenario_batch.ndim == 2 and tuple(scenario_batch.shape) == (self.n_scenarios, self.n_bus):
            scenario_batch = scenario_batch.unsqueeze(0)
        elif scenario_batch.ndim != 3 or tuple(scenario_batch.shape[1:]) != (self.n_scenarios, self.n_bus):
            return None
        if batch_size is None or scenario_batch.shape[0] == batch_size:
            return scenario_batch
        if scenario_batch.shape[0] == 1:
            return scenario_batch.expand(batch_size, -1, -1)
        raise ValueError(
            f"Scenario batch size {scenario_batch.shape[0]} does not match decision batch size {batch_size}."
        )

    def _compute_slack_generation(self, P_g, scenario_k):
        scenario_tensor = self._scenario_tensor(scenario_k, P_g.device)
        total_demand = (self.P_demand_nominal.unsqueeze(0) * scenario_tensor).sum(dim=1, keepdim=True)
        total_non_slack_gen = P_g.sum(dim=1, keepdim=True)
        return total_demand - total_non_slack_gen

    def _compute_theta_from_power_balance(self, P_g, scenario_k):
        scenario_tensor = self._scenario_tensor(scenario_k, P_g.device)
        if not hasattr(self, "gen_to_bus_reduced_matrix"):
            self._precompute_mapping_matrices()
        p_reduced = torch.matmul(P_g, self.gen_to_bus_reduced_matrix.T)
        demand_scenario_k = self.P_demand_nominal.unsqueeze(0) * scenario_tensor
        d_reduced = demand_scenario_k[:, self.non_slack_idx]
        net_injection = p_reduced - d_reduced
        return torch.matmul(net_injection, self.B_bus_reduced_inv_tensor.T)

    def _create_full_theta(self, theta_reduced):
        theta_full = torch.zeros(theta_reduced.shape[0], self.n_bus, device=theta_reduced.device)
        theta_full[:, self.bus_to_angle_idx[:, 0]] = theta_reduced[:, self.bus_to_angle_idx[:, 1]]
        return theta_full

    def objective_x(self, x):
        P_g = x
        total_cost = torch.zeros(x.shape[0], 1, device=x.device)
        non_slack_cost = torch.sum(self.cost_c2 * P_g * P_g, dim=-1, keepdim=True) + torch.sum(
            self.cost_c1 * P_g, dim=-1, keepdim=True
        )
        for scenario_idx in range(self.n_scenarios):
            P_slack = self._compute_slack_generation(P_g, scenario_idx)
            slack_cost = self.cost_c2_slack * P_slack * P_slack + self.cost_c1_slack * P_slack
            total_cost += slack_cost / self.n_scenarios
        total_cost += non_slack_cost
        return total_cost / self.n_bus

    def objective_xy(self, input_batch, x):
        scenario_batch = self._coerce_scenario_batch(input_batch, x.device, batch_size=x.shape[0])
        if scenario_batch is None:
            return self.objective_x(x)
        P_g = x
        total_cost = torch.zeros(x.shape[0], 1, device=x.device)
        non_slack_cost = torch.sum(self.cost_c2 * P_g * P_g, dim=-1, keepdim=True) + torch.sum(
            self.cost_c1 * P_g, dim=-1, keepdim=True
        )
        for scenario_idx in range(self.n_scenarios):
            P_slack = self._compute_slack_generation(P_g, scenario_batch[:, scenario_idx, :])
            slack_cost = self.cost_c2_slack * P_slack * P_slack + self.cost_c1_slack * P_slack
            total_cost += slack_cost / self.n_scenarios
        total_cost += non_slack_cost
        return total_cost / self.n_bus

    def gradient_objective_x(self, x):
        x.requires_grad = True
        return torch.autograd.grad(self.objective_x(x), x)[0]

    def constraint_x(self, x, clip=True):
        z_indicators = self.compute_scenario_feasibility(x)
        prob_satisfied = z_indicators.mean(dim=1, keepdim=True)
        chance_violation = 1 - prob_satisfied
        return torch.clamp(chance_violation, min=0) if clip else chance_violation

    def constraint_residual_xy(self, input_batch, x, clip=True):
        scenario_batch = self._coerce_scenario_batch(input_batch, x.device, batch_size=x.shape[0])
        if scenario_batch is None:
            return self.constraint_x(x, clip=clip)
        z_indicators = self.compute_scenario_feasibility(x, scenario_batch=scenario_batch)
        prob_satisfied = z_indicators.mean(dim=1, keepdim=True)
        chance_violation = 1 - prob_satisfied
        return torch.clamp(chance_violation, min=0) if clip else chance_violation

    def gradient_penalty_x(self, x):
        resid = 0.5 * self.robust_constraint_x(x, clip=True) ** 2
        return torch.autograd.grad(resid, x)[0]

    def gradient_constraint_x(self, x, clip=True, dual_var=None):
        resid = self.robust_constraint_x(x, clip)
        grad = torch.autograd.grad(resid, x)[0]
        return grad * dual_var

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None, method="autograd"):
        if method == "explicit":
            raise NotImplementedError("JCCDCOPFProblem does not provide an explicit lagrangian gradient.")
        if method != "autograd":
            raise NotImplementedError(f"Unsupported JCC lagrangian gradient method: {method}")
        x_detached = x.detach().requires_grad_(True)
        objective = self.objective_x(x_detached).view(-1)
        residual = self.robust_constraint_x(x_detached, clip=False)
        dual = (dual_var * residual).sum(-1)
        if penalty_coef is not None:
            penalty = 0.5 * float(penalty_coef) * torch.clamp(residual, min=0).square().sum(-1)
        else:
            penalty = 0
        if proximal_coef is not None and x_outer is not None:
            proximal = 0.5 * float(proximal_coef) * (x_detached - x_outer).square().sum(-1)
        else:
            proximal = 0
        return torch.autograd.grad((objective + dual + penalty + proximal).sum(), x_detached, create_graph=False)[0]

    def compute_scenario_k_residual(self, scenario, x, clip=True):
        P_slack = self._compute_slack_generation(x, scenario)
        theta_reduced = self._compute_theta_from_power_balance(x, scenario)
        theta_full = self._create_full_theta(theta_reduced)
        residual = torch.cat(
            [
                self.P_min.unsqueeze(0) - x,
                x - self.P_max.unsqueeze(0),
                self.P_slack_min - P_slack,
                P_slack - self.P_slack_max,
                self.theta_min.unsqueeze(0) - theta_reduced,
                theta_reduced - self.theta_max.unsqueeze(0),
                torch.abs(torch.matmul(theta_full, self.B_line_tensor.T)) - self.S_max.unsqueeze(0),
            ],
            dim=1,
        )
        return torch.clamp(residual, min=0) if clip else residual

    def compute_scenario_feasibility(self, x, scenario_batch=None):
        scenario_batch = self._coerce_scenario_batch(scenario_batch, x.device, batch_size=x.shape[0])
        z_indicators = torch.zeros(x.shape[0], self.n_scenarios, device=x.device)
        for scenario_idx in range(self.n_scenarios):
            scenario_input = self.demand_scenarios[scenario_idx] if scenario_batch is None else scenario_batch[:, scenario_idx, :]
            scenario_residual = self.compute_scenario_k_residual(scenario_input, x)
            feasible = scenario_residual.max(dim=1)[0] <= EPSILON
            z_indicators[:, scenario_idx] = feasible.float()
        return z_indicators

    def robust_constraint_x(self, x, clip=True):
        robust_constraint = []
        for scenario_idx in range(self.n_scenarios):
            scenario_residual = self.compute_scenario_k_residual(self.demand_scenarios[scenario_idx], x, clip=clip)
            robust_constraint.append(scenario_residual)
        robust_constraint = torch.cat(robust_constraint, dim=1)
        if clip:
            robust_constraint = robust_constraint.clamp(min=0)
        max_violation = robust_constraint.max(1, keepdim=True)[0]
        return torch.clamp(max_violation, min=0) if clip else max_violation


class JCCDCOPFSolver:
    """CVXPY-based solver using a big-M formulation with binary variables."""

    def __init__(self, problem, solver_config=None):
        self.cp = _require_cvxpy()
        self.problem = problem
        self.config = {
            "M": 100.0,
            "solver": self.cp.GUROBI,
            "solver_options": {},
            "verbose": False,
            "time_limit_sec": 600.0,
            **(solver_config or {}),
        }
        self.M = self.config["M"]
        self.n_vars = problem.n_gen_vars + problem.n_angle_vars
        self.n_scenarios = problem.n_scenarios
        self.model = _JCCCVXPYModelBuilder(problem, self.cp)
        self._create_variables()
        self._create_objective()
        self._create_constraints()

    def _create_variables(self):
        cp = self.cp
        self.P_g = cp.Variable(self.problem.n_gen_vars, name="P_g")
        self.z = cp.Variable(self.n_scenarios, boolean=True, name="z")
        self.p_slack = cp.Variable(self.n_scenarios, name="p_slack")
        self.theta = cp.Variable((self.n_scenarios, self.problem.n_bus - 1), name="theta")

    def _create_objective(self):
        self.objective = self.model.build_objective(self.cp, self.P_g, self.p_slack)

    def _create_constraints(self):
        cp = self.cp
        self.constraints = []
        self.constraints.append(cp.sum(self.z) >= self.n_scenarios * (1 - self.problem.config["epsilon"]))

        for scenario_idx in range(self.n_scenarios):
            self.model.add_scenario_balance_theta_constraints(
                self.constraints, self.P_g, self.p_slack, self.theta, scenario_idx
            )
            self.constraints.extend(
                [
                    self.P_g >= self.model.P_min - self.M * (1 - self.z[scenario_idx]),
                    self.P_g <= self.model.P_max + self.M * (1 - self.z[scenario_idx]),
                    self.p_slack[scenario_idx] >= self.model.P_slack_min - self.M * (1 - self.z[scenario_idx]),
                    self.p_slack[scenario_idx] <= self.model.P_slack_max + self.M * (1 - self.z[scenario_idx]),
                    self.theta[scenario_idx, :] >= self.model.theta_min - self.M * (1 - self.z[scenario_idx]),
                    self.theta[scenario_idx, :] <= self.model.theta_max + self.M * (1 - self.z[scenario_idx]),
                ]
            )
            line_flows = self.model.add_theta_full_and_line_flow(
                self.constraints, self.theta, scenario_idx, name_prefix="theta_full"
            )
            self.constraints.extend(
                [
                    line_flows <= self.model.S_max + self.M * (1 - self.z[scenario_idx]),
                    line_flows >= -self.model.S_max - self.M * (1 - self.z[scenario_idx]),
                ]
            )

    def solve(self, warm_start=None):
        prob = self.cp.Problem(self.objective, self.constraints)
        if warm_start is not None:
            if "x" in warm_start:
                self.P_g.value = warm_start["x"]
            if "z" in warm_start:
                self.z.value = warm_start["z"]
            if "p_slack" in warm_start:
                self.p_slack.value = warm_start["p_slack"]
        try:
            _solve_jcc_cvxpy_problem(prob, self.cp, self.config, solver_priority=CVXPY_INTEGER_SOLVER_PRIORITY)
        except Exception:
            return self._create_failed_result()
        return self._extract_results(prob)

    def _extract_results(self, prob):
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return self._create_failed_result()
        z_opt = self.z.value
        feasibility_rate = np.mean(z_opt) if z_opt is not None else 0.0
        return {
            "status": prob.status,
            "optimal": prob.status == "optimal",
            "objective_value": prob.value,
            "x_optimal": self.P_g.value,
            "P_g_optimal": self.P_g.value,
            "theta_optimal": self.theta.value,
            "p_slack_optimal": self.p_slack.value,
            "z_optimal": z_opt,
            "feasibility_rate": feasibility_rate,
            "chance_constraint_satisfied": feasibility_rate >= (1 - self.problem.config["epsilon"]),
            "n_feasible_scenarios": int(np.sum(z_opt)) if z_opt is not None else 0,
            "required_feasible_scenarios": int(self.n_scenarios * (1 - self.problem.config["epsilon"])),
        }

    def _create_failed_result(self):
        return {
            "status": "failed",
            "optimal": False,
            "objective_value": None,
            "x_optimal": None,
            "P_g_optimal": None,
            "theta_optimal": None,
            "p_slack_optimal": None,
            "z_optimal": None,
            "feasibility_rate": 0.0,
            "chance_constraint_satisfied": False,
            "n_feasible_scenarios": 0,
            "required_feasible_scenarios": int(self.n_scenarios * (1 - self.problem.config["epsilon"])),
        }


class JCCDCOPFCVaRSolver:
    """CVXPY CVaR convex approximation baseline for JCC-DC-OPF."""

    def __init__(self, problem, solver_config=None):
        self.cp = _require_cvxpy()
        self.problem = problem
        self.config = {
            "solver": self.cp.MOSEK,
            "solver_options": {},
            "verbose": False,
            "time_limit_sec": 600.0,
            "epsilon": float(problem.config["epsilon"]),
            **(solver_config or {}),
        }
        self.n_scenarios = problem.n_scenarios
        self.model = _JCCCVXPYModelBuilder(problem, self.cp)
        self._create_variables()
        self._create_objective()
        self._create_constraints()

    def _create_variables(self):
        cp = self.cp
        self.P_g = cp.Variable(self.problem.n_gen_vars, name="P_g")
        self.p_slack = cp.Variable(self.n_scenarios, name="p_slack")
        self.theta = cp.Variable((self.n_scenarios, self.problem.n_bus - 1), name="theta")
        # scenario risk variable: max raw residual per scenario (can be negative)
        self.xi = cp.Variable(self.n_scenarios, name="xi")
        self.eta = cp.Variable(name="eta")
        self.u = cp.Variable(self.n_scenarios, nonneg=True, name="u")

    def _create_objective(self):
        self.objective = self.model.build_objective(self.cp, self.P_g, self.p_slack)

    def _create_constraints(self):
        cp = self.cp
        self.constraints = []
        epsilon = float(self.config["epsilon"])

        for scenario_idx in range(self.n_scenarios):
            self.model.add_scenario_balance_theta_constraints(
                self.constraints, self.P_g, self.p_slack, self.theta, scenario_idx
            )
            line_flows = self.model.add_theta_full_and_line_flow(
                self.constraints, self.theta, scenario_idx, name_prefix="cvar_theta_full"
            )

            raw_residual = cp.hstack(
                [
                    self.model.P_min - self.P_g,
                    self.P_g - self.model.P_max,
                    cp.reshape(self.model.P_slack_min - self.p_slack[scenario_idx], (1,)),
                    cp.reshape(self.p_slack[scenario_idx] - self.model.P_slack_max, (1,)),
                    self.model.theta_min - self.theta[scenario_idx, :],
                    self.theta[scenario_idx, :] - self.model.theta_max,
                    cp.abs(line_flows) - self.model.S_max,
                ]
            )
            # xi_s is the max raw residual over this scenario.
            self.constraints.append(self.xi[scenario_idx] >= raw_residual)

        # CVaR_{1-epsilon}(xi) <= 0
        self.constraints.extend(
            [
                self.u >= self.xi - self.eta,
                self.eta + (1.0 / (epsilon * self.n_scenarios)) * cp.sum(self.u) <= 0,
            ]
        )

    def solve(self, warm_start=None):
        prob = self.cp.Problem(self.objective, self.constraints)
        if warm_start is not None and "x" in warm_start:
            self.P_g.value = warm_start["x"]
        try:
            _solve_jcc_cvxpy_problem(prob, self.cp, self.config)
        except Exception:
            return self._create_failed_result()
        return self._extract_results(prob)

    def _extract_results(self, prob):
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return self._create_failed_result()
        x_opt = self.P_g.value
        if x_opt is None:
            return self._create_failed_result()
        feasibility_rate = _evaluate_scenario_feasibility(self.problem, x_opt)
        return {
            "status": prob.status,
            "optimal": prob.status == "optimal",
            "objective_value": prob.value,
            "x_optimal": x_opt,
            "P_g_optimal": x_opt,
            "theta_optimal": self.theta.value,
            "p_slack_optimal": self.p_slack.value,
            "xi_optimal": self.xi.value,
            "eta_optimal": self.eta.value,
            "u_optimal": self.u.value,
            "feasibility_rate": feasibility_rate,
            "chance_constraint_satisfied": feasibility_rate >= (1 - self.problem.config["epsilon"]),
            "n_feasible_scenarios": int(round(feasibility_rate * self.n_scenarios)),
            "required_feasible_scenarios": int(self.n_scenarios * (1 - self.problem.config["epsilon"])),
        }

    def _create_failed_result(self):
        return {
            "status": "failed",
            "optimal": False,
            "objective_value": None,
            "x_optimal": None,
            "P_g_optimal": None,
            "theta_optimal": None,
            "p_slack_optimal": None,
            "xi_optimal": None,
            "eta_optimal": None,
            "u_optimal": None,
            "feasibility_rate": 0.0,
            "chance_constraint_satisfied": False,
            "n_feasible_scenarios": 0,
            "required_feasible_scenarios": int(self.n_scenarios * (1 - self.problem.config["epsilon"])),
        }


class JCCDCOPFRobustScenarioSolver:
    """CVXPY robust-scenario baseline enforcing all sampled scenarios."""

    def __init__(self, problem, solver_config=None):
        self.cp = _require_cvxpy()
        self.problem = problem
        self.config = {
            "solver": self.cp.MOSEK,
            "solver_options": {},
            "verbose": False,
            "time_limit_sec": 600.0,
            **(solver_config or {}),
        }
        self.n_scenarios = problem.n_scenarios
        self.model = _JCCCVXPYModelBuilder(problem, self.cp)
        self._create_variables()
        self._create_objective()
        self._create_constraints()

    def _create_variables(self):
        cp = self.cp
        self.P_g = cp.Variable(self.problem.n_gen_vars, name="P_g")
        self.p_slack = cp.Variable(self.n_scenarios, name="p_slack")
        self.theta = cp.Variable((self.n_scenarios, self.problem.n_bus - 1), name="theta")

    def _create_objective(self):
        self.objective = self.model.build_objective(self.cp, self.P_g, self.p_slack)

    def _create_constraints(self):
        self.constraints = []
        for scenario_idx in range(self.n_scenarios):
            self.model.add_scenario_balance_theta_constraints(
                self.constraints, self.P_g, self.p_slack, self.theta, scenario_idx
            )
            line_flows = self.model.add_theta_full_and_line_flow(
                self.constraints, self.theta, scenario_idx, name_prefix="robust_theta_full"
            )
            self.constraints.extend(
                [
                    self.P_g >= self.model.P_min,
                    self.P_g <= self.model.P_max,
                    self.p_slack[scenario_idx] >= self.model.P_slack_min,
                    self.p_slack[scenario_idx] <= self.model.P_slack_max,
                    self.theta[scenario_idx, :] >= self.model.theta_min,
                    self.theta[scenario_idx, :] <= self.model.theta_max,
                    line_flows <= self.model.S_max,
                    line_flows >= -self.model.S_max,
                ]
            )

    def solve(self, warm_start=None):
        prob = self.cp.Problem(self.objective, self.constraints)
        if warm_start is not None and "x" in warm_start:
            self.P_g.value = warm_start["x"]
        try:
            _solve_jcc_cvxpy_problem(prob, self.cp, self.config)
        except Exception:
            return self._create_failed_result()
        return self._extract_results(prob)

    def _extract_results(self, prob):
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return self._create_failed_result()
        x_opt = self.P_g.value
        if x_opt is None:
            return self._create_failed_result()
        feasibility_rate = _evaluate_scenario_feasibility(self.problem, x_opt)
        return {
            "status": prob.status,
            "optimal": prob.status == "optimal",
            "objective_value": prob.value,
            "x_optimal": x_opt,
            "P_g_optimal": x_opt,
            "theta_optimal": self.theta.value,
            "p_slack_optimal": self.p_slack.value,
            "feasibility_rate": feasibility_rate,
            "chance_constraint_satisfied": feasibility_rate >= (1 - self.problem.config["epsilon"]),
            "n_feasible_scenarios": int(round(feasibility_rate * self.n_scenarios)),
            "required_feasible_scenarios": int(self.n_scenarios * (1 - self.problem.config["epsilon"])),
        }

    def _create_failed_result(self):
        return {
            "status": "failed",
            "optimal": False,
            "objective_value": None,
            "x_optimal": None,
            "P_g_optimal": None,
            "theta_optimal": None,
            "p_slack_optimal": None,
            "feasibility_rate": 0.0,
            "chance_constraint_satisfied": False,
            "n_feasible_scenarios": 0,
            "required_feasible_scenarios": int(self.n_scenarios * (1 - self.problem.config["epsilon"])),
        }


def solve_jcc_dcopt(num_bus=30, problem_config=None, solver_config=None):
    from homopt.solvers import JCCDCOPFSolver

    problem = JCCDCOPFProblem(num_bus=num_bus, config=problem_config)
    solver = JCCDCOPFSolver(problem, solver_config=solver_config)
    results = solver.solve()
    return problem, solver, results


__all__ = [
    "EPSILON",
    "BASEMVA",
    "JCCDCOPFProblem",
    "solve_jcc_dcopt",
]
