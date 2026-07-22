"""Joint chance-constrained DC-OPF implementation."""

from __future__ import annotations

import copy
import numpy as np
import torch

from homopt.problems.base import ProblemInstanceBatch, StateBackedParametricProblemBase
from homopt.problems.opf_cases import load_pglib_opf_case
from homopt.utils import resolve_torch_dtype

try:
    from pypower.idx_brch import BR_X, F_BUS, RATE_A, T_BUS
    from pypower.idx_bus import BUS_I, BUS_TYPE, PD, PQ, REF
    from pypower.idx_gen import GEN_BUS, PMAX, PMIN
except ImportError as exc:  # pragma: no cover - dependency availability varies by env
    _PYPOWER_IMPORT_ERROR = exc
else:  # pragma: no cover - simple import branch
    _PYPOWER_IMPORT_ERROR = None


torch.set_default_dtype(torch.float32)
EPSILON = 1e-4
BASEMVA = 100.0


def _require_powerflow_dependencies():
    if _PYPOWER_IMPORT_ERROR is not None:
        raise ImportError(
            "JCC-DC-OPF requires pypower. Install with: pip install pypower"
        ) from _PYPOWER_IMPORT_ERROR


def _as_tensor(value, *, device=None, dtype=None):
    if torch.is_tensor(value):
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return value.to(**kwargs) if kwargs else value
    return torch.tensor(value, dtype=dtype or torch.float32, device=device)


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class JCCDCOPFProblem(StateBackedParametricProblemBase):
    """Joint chance-constrained DC optimal power flow problem."""

    _RUNTIME_TENSOR_ATTRS = (
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
        "B_bus_reduced_inv_tensor",
        "B_bus_tensor",
        "B_line_tensor",
        "cost_c2_slack",
        "cost_c1_slack",
        "P_slack_min",
        "P_slack_max",
        "gen_to_bus_reduced_matrix",
        "demand_reduced_base",
    )

    def __init__(self, num_bus=30, config=None):
        _require_powerflow_dependencies()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.config = {
            "n_scenarios": 20,
            "epsilon": 0.05,
            "demand_std": 0.1,
            "seed": 2025,
            "pglib_case_name": None,
            "pglib_data_dir": None,
            "download_if_missing": True,
            **(config or {}),
        }
        self.ppc = self._load_case_data(num_bus, self.config)
        self.case_name = f"pglib_opf_case{int(num_bus)}"
        self._extract_case_data()
        self._setup_matrices()
        self._setup_constraints()
        self._generate_scenarios()
        # Keep the public problem contract usable immediately after construction:
        # objective/constraint methods operate on tensors, not the raw NumPy
        # arrays used only while parsing the MATPOWER case.
        self.to_device(self.device)
        self.is_parametric = True

    @staticmethod
    def _load_case_data(num_bus, config):
        return load_pglib_opf_case(
            int(num_bus),
            case_name=config.get("pglib_case_name"),
            data_dir=config.get("pglib_data_dir"),
            download_if_missing=bool(config.get("download_if_missing", True)),
        )

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

        self.bus_ids = np.asarray(self.bus[:, BUS_I], dtype=int)
        self.bus_id_to_idx = {int(bus_id): idx for idx, bus_id in enumerate(self.bus_ids)}
        if len(self.bus_id_to_idx) != self.n_bus:
            raise ValueError("MATPOWER case contains duplicate bus IDs.")
        ref_buses = np.where(self.bus[:, BUS_TYPE] == REF)[0]
        self.ref_bus = ref_buses[0] if len(ref_buses) > 0 else 0
        self.ref_bus_id = int(self.bus_ids[self.ref_bus])
        self.non_slack_idx = np.where(self.bus[:, BUS_TYPE] != REF)[0]
        slack_generators = np.where(np.asarray(self.gen[:, GEN_BUS], dtype=int) == self.ref_bus_id)[0]
        if len(slack_generators) == 0:
            raise ValueError(
                f"MATPOWER reference bus {self.ref_bus_id} has no associated generator."
            )
        self.slack_gen = int(slack_generators[0])
        self.PQ_bus_flag = (self.bus[:, BUS_TYPE] == PQ).astype(int)
        try:
            self.gen_bus = np.asarray(
                [self.bus_id_to_idx[int(bus_id)] for bus_id in self.gen[:, GEN_BUS]],
                dtype=int,
            )
        except KeyError as exc:
            raise ValueError(f"Generator references unknown MATPOWER bus ID {exc.args[0]}.") from exc

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

        self.branch_bus_indices = np.empty((self.n_branch, 2), dtype=int)
        for branch_idx, branch in enumerate(self.branch):
            try:
                from_bus = self.bus_id_to_idx[int(branch[F_BUS])]
                to_bus = self.bus_id_to_idx[int(branch[T_BUS])]
            except KeyError as exc:
                raise ValueError(f"Branch references unknown MATPOWER bus ID {exc.args[0]}.") from exc
            self.branch_bus_indices[branch_idx] = (from_bus, to_bus)
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
        self.B_bus_tensor = torch.tensor(self.B_bus, dtype=self.dtype, device=self.device)
        self.B_line_tensor = torch.tensor(self.B_line, dtype=self.dtype, device=self.device)
        try:
            self.B_bus_reduced_inv = np.linalg.inv(self.B_bus_reduced)
        except np.linalg.LinAlgError:
            self.B_bus_reduced_inv = np.linalg.pinv(self.B_bus_reduced)
        self.B_bus_reduced_inv_tensor = torch.tensor(
            self.B_bus_reduced_inv,
            dtype=self.dtype,
            device=self.device,
        )

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
        return _as_tensor(
            self._draw_scenarios(n_instances=n_samples, seed=seed),
            device=self.device,
            dtype=self.dtype,
        )

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
        scenarios = _as_tensor(input_data, device=self.device, dtype=self.dtype)
        bound.demand_scenarios = scenarios
        bound.n_scenarios = int(scenarios.shape[0])
        bound.config = {**self.config, "n_scenarios": bound.n_scenarios}
        return bound

    def _set_runtime(self, *, device, dtype):
        for attr_name in self._RUNTIME_TENSOR_ATTRS:
            if hasattr(self, attr_name):
                setattr(
                    self,
                    attr_name,
                    _as_tensor(getattr(self, attr_name), device=device, dtype=dtype),
                )
        self.device = torch.device(device)
        self.dtype = dtype
        return self

    def to_device(self, device):
        return self._set_runtime(device=torch.device(device), dtype=self.dtype)

    def to_dtype(self, dtype):
        return self._set_runtime(
            device=self.device,
            dtype=resolve_torch_dtype(dtype, default=torch.float32),
        )

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
        self.gen_to_bus_reduced_matrix = torch.tensor(matrix, dtype=self.dtype, device=self.device)

        demand_reduced = torch.zeros(self.n_bus - 1, dtype=self.dtype, device=self.device)
        P_demand_tensor = _as_tensor(self.P_demand_nominal, device=self.device, dtype=self.dtype)
        for bus_idx, demand_val in enumerate(P_demand_tensor):
            if bus_idx != self.ref_bus:
                reduced_idx = bus_idx if bus_idx < self.ref_bus else bus_idx - 1
                demand_reduced[reduced_idx] = demand_val
        self.demand_reduced_base = demand_reduced

    def _scenario_tensor(self, scenario_k, device, dtype=None):
        if isinstance(scenario_k, int):
            scenario_tensor = self.demand_scenarios[scenario_k]
        else:
            scenario_tensor = scenario_k
        return _as_tensor(scenario_tensor, device=device, dtype=dtype)

    def _coerce_scenario_batch(self, input_batch, device, batch_size=None, dtype=None):
        if input_batch is None:
            return None
        scenario_batch = _as_tensor(input_batch, device=device, dtype=dtype)
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
        scenario_tensor = self._scenario_tensor(scenario_k, P_g.device, dtype=P_g.dtype)
        demand_nominal = self.P_demand_nominal.to(device=P_g.device, dtype=P_g.dtype)
        total_demand = (demand_nominal.unsqueeze(0) * scenario_tensor).sum(dim=1, keepdim=True)
        total_non_slack_gen = P_g.sum(dim=1, keepdim=True)
        return total_demand - total_non_slack_gen

    def _compute_theta_from_power_balance(self, P_g, scenario_k):
        scenario_tensor = self._scenario_tensor(scenario_k, P_g.device, dtype=P_g.dtype)
        if not hasattr(self, "gen_to_bus_reduced_matrix"):
            self._precompute_mapping_matrices()
        generator_matrix = self.gen_to_bus_reduced_matrix.to(device=P_g.device, dtype=P_g.dtype)
        bus_inverse = self.B_bus_reduced_inv_tensor.to(device=P_g.device, dtype=P_g.dtype)
        demand_nominal = self.P_demand_nominal.to(device=P_g.device, dtype=P_g.dtype)
        p_reduced = torch.matmul(P_g, generator_matrix.T)
        demand_scenario_k = demand_nominal.unsqueeze(0) * scenario_tensor
        d_reduced = demand_scenario_k[:, self.non_slack_idx]
        net_injection = p_reduced - d_reduced
        return torch.matmul(net_injection, bus_inverse.T)

    def _create_full_theta(self, theta_reduced):
        theta_full = torch.zeros(
            theta_reduced.shape[0],
            self.n_bus,
            device=theta_reduced.device,
            dtype=theta_reduced.dtype,
        )
        theta_full[:, self.bus_to_angle_idx[:, 0]] = theta_reduced[:, self.bus_to_angle_idx[:, 1]]
        return theta_full

    def objective_x(self, x):
        P_g = x
        total_cost = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        cost_c2 = self.cost_c2.to(device=x.device, dtype=x.dtype)
        cost_c1 = self.cost_c1.to(device=x.device, dtype=x.dtype)
        cost_c2_slack = self.cost_c2_slack.to(device=x.device, dtype=x.dtype)
        cost_c1_slack = self.cost_c1_slack.to(device=x.device, dtype=x.dtype)
        non_slack_cost = torch.sum(cost_c2 * P_g * P_g, dim=-1, keepdim=True) + torch.sum(
            cost_c1 * P_g, dim=-1, keepdim=True
        )
        for scenario_idx in range(self.n_scenarios):
            P_slack = self._compute_slack_generation(P_g, scenario_idx)
            slack_cost = cost_c2_slack * P_slack * P_slack + cost_c1_slack * P_slack
            total_cost += slack_cost / self.n_scenarios
        total_cost += non_slack_cost
        return total_cost / self.n_bus

    def objective_xy(self, input_params, y):
        scenario_batch = self._coerce_scenario_batch(
            input_params,
            y.device,
            batch_size=y.shape[0],
            dtype=y.dtype,
        )
        if scenario_batch is None:
            return self.objective_x(y)
        P_g = y
        total_cost = torch.zeros(y.shape[0], 1, device=y.device, dtype=y.dtype)
        cost_c2 = self.cost_c2.to(device=y.device, dtype=y.dtype)
        cost_c1 = self.cost_c1.to(device=y.device, dtype=y.dtype)
        cost_c2_slack = self.cost_c2_slack.to(device=y.device, dtype=y.dtype)
        cost_c1_slack = self.cost_c1_slack.to(device=y.device, dtype=y.dtype)
        non_slack_cost = torch.sum(cost_c2 * P_g * P_g, dim=-1, keepdim=True) + torch.sum(
            cost_c1 * P_g, dim=-1, keepdim=True
        )
        for scenario_idx in range(self.n_scenarios):
            P_slack = self._compute_slack_generation(P_g, scenario_batch[:, scenario_idx, :])
            slack_cost = cost_c2_slack * P_slack * P_slack + cost_c1_slack * P_slack
            total_cost += slack_cost / self.n_scenarios
        total_cost += non_slack_cost
        return total_cost / self.n_bus

    def gradient_objective_x(self, x):
        x.requires_grad = True
        return torch.autograd.grad(self.objective_x(x), x)[0]

    def constraint_x(self, x, clip=True):
        scenario_feasible = self.compute_scenario_feasibility(x)
        constraint = 1.0 - scenario_feasible.mean(dim=1, keepdim=True)
        return torch.clamp(constraint, min=0) if clip else constraint

    def constraint_residual_xy(self, input_params, y, clip=True):
        scenario_batch = self._coerce_scenario_batch(
            input_params,
            y.device,
            batch_size=y.shape[0],
            dtype=y.dtype,
        )
        if scenario_batch is None:
            return self.constraint_x(y, clip=clip)
        scenario_feasible = self.compute_scenario_feasibility(y, scenario_batch=scenario_batch)
        constraint = 1.0 - scenario_feasible.mean(dim=1, keepdim=True)
        return torch.clamp(constraint, min=0) if clip else constraint

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
        p_min = self.P_min.to(device=x.device, dtype=x.dtype)
        p_max = self.P_max.to(device=x.device, dtype=x.dtype)
        p_slack_min = self.P_slack_min.to(device=x.device, dtype=x.dtype)
        p_slack_max = self.P_slack_max.to(device=x.device, dtype=x.dtype)
        theta_min = self.theta_min.to(device=x.device, dtype=x.dtype)
        theta_max = self.theta_max.to(device=x.device, dtype=x.dtype)
        line_matrix = self.B_line_tensor.to(device=x.device, dtype=x.dtype)
        s_max = self.S_max.to(device=x.device, dtype=x.dtype)
        residual = torch.cat(
            [
                p_min.unsqueeze(0) - x,
                x - p_max.unsqueeze(0),
                p_slack_min - P_slack,
                P_slack - p_slack_max,
                theta_min.unsqueeze(0) - theta_reduced,
                theta_reduced - theta_max.unsqueeze(0),
                torch.abs(torch.matmul(theta_full, line_matrix.T)) - s_max.unsqueeze(0),
            ],
            dim=1,
        )
        return torch.clamp(residual, min=0) if clip else residual

    def compute_scenario_feasibility(self, x, scenario_batch=None):
        scenario_batch = self._coerce_scenario_batch(
            scenario_batch,
            x.device,
            batch_size=x.shape[0],
            dtype=x.dtype,
        )
        scenario_feasible = torch.zeros(x.shape[0], self.n_scenarios, device=x.device, dtype=x.dtype)
        for scenario_idx in range(self.n_scenarios):
            scenario_input = self.demand_scenarios[scenario_idx] if scenario_batch is None else scenario_batch[:, scenario_idx, :]
            scenario_residual = self.compute_scenario_k_residual(scenario_input, x)
            feasible = scenario_residual.max(dim=1)[0] <= EPSILON
            scenario_feasible[:, scenario_idx] = feasible.to(dtype=x.dtype)
        return scenario_feasible

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
