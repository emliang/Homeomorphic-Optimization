"""JCC DC-OPF exact solver implementations."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from homopt.problems.jcc.opf import _as_numpy, _as_tensor
from homopt.solvers.common import _normalized_exact_solver_result
from homopt.solvers.cvxpy import CVXPY_INTEGER_SOLVER_PRIORITY, CVXPY_SOLVER_PRIORITY, _solve_cvxpy_problem


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
        preferred_solver=config.get("solver"),
    )


def _evaluate_scenario_feasibility(problem, x_opt):
    device = getattr(problem, "device", None)
    x_tensor = _as_tensor(x_opt, device=device).view(1, -1)
    scenario_feasibility = problem.compute_scenario_feasibility(x_tensor)
    return float(scenario_feasibility.mean().item())


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




class _BaseJCCSolverMixin:
    def solve_result(self, solve_type="opt", x_init=None, solver=None, options=None):
        if solve_type not in {"opt", "initialized_opt"}:
            raise ValueError(f"Unsupported JCC solve_type: {solve_type}")
        warm_start_payload = {}
        if x_init is not None:
            warm_start_payload.setdefault("x", x_init)
        if not warm_start_payload:
            warm_start_payload = None
        original_config = dict(self.config)
        if solver is not None:
            resolved_solver = getattr(self.cp, solver) if isinstance(solver, str) and hasattr(self.cp, solver) else solver
            self.config["solver"] = resolved_solver
        if options:
            self.config.update(dict(options))
        start = perf_counter()
        try:
            raw = self.solve(warm_start=warm_start_payload)
            runtime_total = perf_counter() - start
        finally:
            self.config = original_config
        return _normalized_exact_solver_result(
            solution=raw.get("x_optimal"),
            status=raw.get("status", "unknown"),
            objective=raw.get("objective_value"),
            runtime_total=runtime_total,
            violation=None
            if raw.get("feasibility_rate") is None
            else max(0.0, (1.0 - float(raw["feasibility_rate"])) - float(self.problem.config["epsilon"])),
            feasible=raw.get("chance_constraint_satisfied"),
            extras={k: v for k, v in raw.items() if k not in {"x_optimal", "status", "objective_value"}},
        )


class JCCDCOPFSolver(_BaseJCCSolverMixin):
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


class JCCDCOPFCVaRSolver(_BaseJCCSolverMixin):
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
                    cp.reshape(self.model.P_slack_min - self.p_slack[scenario_idx], (1,), order="F"),
                    cp.reshape(self.p_slack[scenario_idx] - self.model.P_slack_max, (1,), order="F"),
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


class JCCDCOPFRobustScenarioSolver(_BaseJCCSolverMixin):
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




__all__ = [
    "JCCDCOPFCVaRSolver",
    "JCCDCOPFRobustScenarioSolver",
    "JCCDCOPFSolver",
]
