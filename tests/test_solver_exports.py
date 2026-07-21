import inspect
import sys

import numpy as np
import pytest

import homopt.problems.jcc.opf as jcc_opf_impl
import homopt.optim.core.solver_cache as solver_cache
import homopt.solvers.jcc as jcc_solver_impl
import homopt.solvers.common as solver_common
from homopt.solvers import (
    ChanceConstraintSolver,
    ConvexSolver,
    JCCDCOPFCVaRSolver,
    JCCDCOPFRobustScenarioSolver,
    JCCDCOPFSolver,
    JCCLinearCVaRSolver,
    JCCLinearRobustScenarioSolver,
    JCCLinearSolver,
    MaxCutSolver,
    normalize_exact_solver_call_kwargs,
    QCQPSolver,
    solve_exact_result,
)
from homopt.solvers.chance import ChanceConstraintSolver as ChanceConstraintSolverImpl
from homopt.solvers.common import normalize_exact_solver_call_kwargs as CommonNormalizeExactSolverCallKwargs
from homopt.solvers.convex import ConvexSolver as ConvexSolverImpl
from homopt.solvers.cvxpy import (
    CVXPY_INTEGER_SOLVER_PRIORITY,
    CVXPY_SOLVER_PRIORITY,
    _candidate_cvxpy_solvers,
    _solve_cvxpy_problem,
)
from homopt.solvers.maxcut import MaxCutSolver as MaxCutSolverImpl
from homopt.solvers.pyomo import PYOMO_NLP_SOLVER
from homopt.solvers.qcqp import QCQPSolver as QCQPSolverImpl
from homopt.solvers import (
    JCCDCOPFCVaRSolver as LegacyJCCDCOPFCVaRSolver,
    JCCDCOPFRobustScenarioSolver as LegacyJCCDCOPFRobustScenarioSolver,
    JCCDCOPFSolver as LegacyJCCDCOPFSolver,
    JCCLinearCVaRSolver as LegacyJCCLinearCVaRSolver,
    JCCLinearRobustScenarioSolver as LegacyJCCLinearRobustScenarioSolver,
    JCCLinearSolver as LegacyJCCLinearSolver,
)
from homopt.solvers.jcc_linear import (
    JCCLinearCVaRSolver as LinearCVaRSolverImpl,
    JCCLinearRobustScenarioSolver as LinearRobustScenarioSolverImpl,
    JCCLinearSolver as LinearSolverImpl,
)


def test_solver_namespace_matches_package_exports():
    assert ChanceConstraintSolverImpl is ChanceConstraintSolver
    assert ConvexSolverImpl is ConvexSolver
    assert LegacyJCCDCOPFCVaRSolver is JCCDCOPFCVaRSolver
    assert LegacyJCCDCOPFRobustScenarioSolver is JCCDCOPFRobustScenarioSolver
    assert LegacyJCCDCOPFSolver is JCCDCOPFSolver
    assert LegacyJCCLinearCVaRSolver is JCCLinearCVaRSolver
    assert LegacyJCCLinearRobustScenarioSolver is JCCLinearRobustScenarioSolver
    assert LegacyJCCLinearSolver is JCCLinearSolver
    assert LinearCVaRSolverImpl is JCCLinearCVaRSolver
    assert LinearRobustScenarioSolverImpl is JCCLinearRobustScenarioSolver
    assert LinearSolverImpl is JCCLinearSolver
    assert MaxCutSolverImpl is MaxCutSolver
    assert CommonNormalizeExactSolverCallKwargs is normalize_exact_solver_call_kwargs
    assert QCQPSolverImpl is QCQPSolver
    with pytest.raises(AttributeError):
        getattr(sys.modules["homopt.solvers"], "StiefelRetractionOptimizer")


def test_normalize_exact_solver_call_kwargs_maps_aliases_and_filters_unknown_keys():
    class _MockSolver:
        def solve(self, solve_type="opt", solver="ipopt", options=None):
            return {
                "solution": [0.0],
                "status": "optimal",
                "objective_value": 1.0,
                "solver_seen": solver,
                "options_seen": options,
            }

    solver = _MockSolver()
    normalized = normalize_exact_solver_call_kwargs(
        solver,
        solve_config={
            "solve_type": "initialized_opt",
            "solver_name": "mock-solver",
            "solver_options": {"tol": 1e-5},
            "ignored_flag": True,
        },
    )
    assert normalized == {
        "solve_type": "initialized_opt",
        "solver": "mock-solver",
        "options": {"tol": 1e-5},
    }

    result = solve_exact_result(
        solver,
        solve_config={
            "solve_type": "initialized_opt",
            "solver_name": "mock-solver",
            "solver_options": {"tol": 1e-5},
            "ignored_flag": True,
        },
    )
    assert result["status"] == "optimal"
    assert result["extras"]["solver_seen"] == "mock-solver"
    assert result["extras"]["options_seen"] == {"tol": 1e-5}


def test_normalize_exact_solver_call_kwargs_preserves_solver_options_when_supported():
    class _MockSolver:
        def solve_result(self, solve_type="opt", solver_options=None, time_limit_sec=None):
            return {
                "solve_type": solve_type,
                "solver_options": solver_options,
                "time_limit_sec": time_limit_sec,
            }

    solver = _MockSolver()
    normalized = normalize_exact_solver_call_kwargs(
        solver,
        solve_config={
            "solve_type": "eq_alm",
            "solver_options": {"max_iters": 50},
            "time_limit_sec": 1.5,
        },
    )
    assert normalized == {
        "solve_type": "eq_alm",
        "solver_options": {"max_iters": 50},
        "time_limit_sec": 1.5,
    }


def test_normalize_exact_solver_call_kwargs_rejects_unsupported_explicit_solver_name():
    class _MockSolver:
        def solve_result(self, solve_type="opt", solver_options=None):
            return {"solve_type": solve_type, "solver_options": solver_options}

    with pytest.raises(TypeError, match="explicit solver_name"):
        normalize_exact_solver_call_kwargs(
            _MockSolver(),
            solve_config={
                "solve_type": "eq_alm",
                "solver_name": "MOSEK",
            },
        )


def test_convex_solver_accepts_explicit_solver_name():
    problem_params = {
        "Q": np.eye(1),
        "p": np.zeros(1),
        "A": None,
        "b": None,
        "A_eq": None,
        "b_eq": None,
        "Qq": None,
        "pq": None,
        "bq": None,
        "G": None,
        "h": None,
        "C": None,
        "d": None,
        "L": np.array([-1.0]),
        "U": np.array([1.0]),
    }
    normalized = normalize_exact_solver_call_kwargs(
        ConvexSolverImpl(problem_params),
        solve_config={
            "solve_type": "opt",
            "solver_name": "SCS",
            "solver_options": {"eps": 1e-4},
        },
    )
    assert normalized["solver_name"] == "SCS"
    assert normalized["solver_options"] == {"eps": 1e-4}


def test_normalize_exact_solver_call_kwargs_caches_solver_signature(monkeypatch):
    class _SignatureCacheSolver:
        def solve(self, solve_type="opt"):
            return {"solution": [0.0], "status": "optimal", "solve_type": solve_type}

    calls = {"count": 0}
    original_signature = solver_common.inspect.signature

    def counting_signature(fn):
        calls["count"] += 1
        return original_signature(fn)

    monkeypatch.setattr(solver_common.inspect, "signature", counting_signature)
    solver = _SignatureCacheSolver()
    normalize_exact_solver_call_kwargs(solver, solve_config={"solve_type": "opt", "ignored": True})
    normalize_exact_solver_call_kwargs(solver, solve_config={"solve_type": "initialized_opt", "ignored": True})
    assert calls["count"] == 1


def test_extract_exact_solution_reports_solver_error_detail():
    result = {
        "solution": None,
        "status": "error",
        "extras": {"error": "dual_var has size 3, expected 1."},
    }

    with pytest.raises(RuntimeError, match="dual_var has size 3, expected 1"):
        solver_cache._extract_exact_solution(result)


def test_chance_constraint_solver_accepts_robust_alias():
    solver = ChanceConstraintSolver(
        (
            np.eye(1),
            np.zeros(1),
            np.ones((1, 1)),
            np.ones(1),
            np.array([-1.0]),
            np.array([1.0]),
            [np.zeros(1)],
            [np.zeros((1, 1))],
            0.1,
        )
    )
    assert solver._normalize_approach("robust") == "robust-scenario"
    assert solver._normalize_approach("robust-scenario") == "robust-scenario"
    assert solver._normalize_approach("mixed-integer") == "mixed-integer"
    with pytest.raises(ValueError, match="Unknown approach"):
        solver._normalize_approach("bad")


def test_qcqp_solver_validates_solve_type_and_ball_violation():
    pytest.importorskip("pyomo.environ")
    solver = QCQPSolver(
        (
            np.eye(2),
            np.zeros(2),
            None,
            None,
            None,
            None,
            None,
            np.array([-10.0, -10.0]),
            np.array([10.0, 10.0]),
            1.0,
        )
    )
    with pytest.raises(ValueError, match="Unknown solve_type"):
        solver.solve("bad")
    feasibility = solver.check_feasibility(np.array([0.8, 0.8]))
    assert feasibility["max_violation"] == pytest.approx(0.28)
    assert feasibility["violations"]["r_ball"] == pytest.approx(0.28)


def test_jcc_solver_facades_share_the_same_solve_result_surface():
    jcc_params = tuple(inspect.signature(JCCDCOPFSolver.solve_result).parameters)
    linear_params = tuple(inspect.signature(JCCLinearSolver.solve_result).parameters)
    assert jcc_params == linear_params


def test_cvxpy_solver_priority_is_mosek_ecos_scs_only():
    class _FakeCP:
        MOSEK = "MOSEK"
        ECOS = "ECOS"
        SCS = "SCS"
        CVXOPT = "CVXOPT"

        @staticmethod
        def installed_solvers():
            return ["SCS", "CVXOPT", "MOSEK", "ECOS"]

    assert CVXPY_SOLVER_PRIORITY == ("MOSEK", "ECOS", "SCS")
    assert _candidate_cvxpy_solvers(_FakeCP) == ["MOSEK", "ECOS", "SCS"]


def test_convex_solver_eq_alm_keeps_inequality_constraints_hard():
    params = {
        "Q": np.array([[1.0]]),
        "p": np.array([0.0]),
        "A": np.array([[1.0]]),
        "b": np.array([0.0]),
        "A_eq": np.array([[1.0]]),
        "b_eq": np.array([1.0]),
        "Qq": None,
        "pq": None,
        "bq": None,
        "G": None,
        "h": None,
        "C": None,
        "d": None,
        "L": np.array([-10.0]),
        "U": np.array([10.0]),
    }
    result = ConvexSolver(params).solve_result(
        "eq_alm",
        x_init=np.array([0.0]),
        dual_var=np.array([0.0]),
        penalty_coef=1.0,
    )
    assert result["solution"] is not None
    assert result["solution"][0] <= 1e-5
    assert result["violation"] == pytest.approx(1.0, abs=1e-5)
    assert result["extras"]["equality"] is False
    assert result["extras"]["hard_equalities"] is False


def test_convex_solver_eq_alm_objective_uses_half_quadratic_coefficients():
    cp = pytest.importorskip("cvxpy")
    params = {
        "Q": np.eye(2),
        "p": np.zeros(2),
        "A": None,
        "b": None,
        "A_eq": np.array([[1.0, 2.0]]),
        "b_eq": np.array([0.5]),
        "Qq": None,
        "pq": None,
        "bq": None,
        "G": None,
        "h": None,
        "C": None,
        "d": None,
        "L": np.array([-10.0, -10.0]),
        "U": np.array([10.0, 10.0]),
    }
    solver = ConvexSolver(params)
    x = cp.Variable(2)
    dual_var = np.array([0.25])
    penalty_coef = 3.0
    proximal_coef = 4.0
    x_outer = np.array([0.1, -0.2])
    expr = solver._eq_alm_objective_expr(
        cp,
        x,
        dual_var=dual_var,
        penalty_coef=penalty_coef,
        proximal_coef=proximal_coef,
        x_outer=x_outer,
    )

    x_value = np.array([0.7, -0.3])
    x.value = x_value
    eq_residual = params["A_eq"] @ x_value - params["b_eq"]
    expected = (
        0.5 * x_value @ params["Q"] @ x_value
        + params["p"] @ x_value
        + dual_var @ eq_residual
        + 0.5 * penalty_coef * np.sum(eq_residual**2)
        + 0.5 * proximal_coef * np.sum((x_value - x_outer) ** 2)
    )
    assert expr.value == pytest.approx(expected)


def test_convex_solver_projection_and_linear_modes_keep_equalities_hard():
    params = {
        "Q": np.array([[1.0]]),
        "p": np.array([0.0]),
        "A": None,
        "b": None,
        "A_eq": np.array([[1.0]]),
        "b_eq": np.array([0.0]),
        "Qq": None,
        "pq": None,
        "bq": None,
        "G": None,
        "h": None,
        "C": None,
        "d": None,
        "L": np.array([-10.0]),
        "U": np.array([10.0]),
    }
    solver = ConvexSolver(params)

    projection = solver.solve_result("proj", x_init=np.array([2.0]))
    linear = solver.solve_result("linear", x_init=np.array([0.0]), grad=np.array([1.0]))

    assert projection["solution"] is not None
    assert projection["solution"][0] == pytest.approx(0.0, abs=1e-5)
    assert projection["violation"] == pytest.approx(0.0, abs=1e-5)
    assert linear["solution"] is not None
    assert linear["solution"][0] == pytest.approx(0.0, abs=1e-5)
    assert linear["violation"] == pytest.approx(0.0, abs=1e-5)


def test_convex_solver_ip_eps_shrinks_feasible_box():
    params = {
        "Q": np.array([[1.0]]),
        "p": np.array([0.0]),
        "A": None,
        "b": None,
        "A_eq": None,
        "b_eq": None,
        "Qq": None,
        "pq": None,
        "bq": None,
        "G": None,
        "h": None,
        "C": None,
        "d": None,
        "L": np.array([0.9]),
        "U": np.array([1.0]),
    }
    result = ConvexSolver(params).solve_result("ip", eps=0.05)
    assert result["solution"] is not None
    assert 0.95 <= result["solution"][0] <= 0.95 + 1e-5


def test_convex_solver_geometric_central_ip_scales_linear_margin():
    params = {
        "Q": np.array([[1.0]]),
        "p": np.array([0.0]),
        "A": np.array([[10.0]]),
        "b": np.array([10.0]),
        "A_eq": None,
        "b_eq": None,
        "Qq": None,
        "pq": None,
        "bq": None,
        "G": None,
        "h": None,
        "C": None,
        "d": None,
        "L": np.array([0.0]),
        "U": np.array([10.0]),
    }
    solver = ConvexSolver(params)

    raw_center = solver.solve_result("central_ip", equality=False)
    geometric_center = solver.solve_result("geometric_central_ip", equality=False)

    assert raw_center["solution"] is not None
    assert geometric_center["solution"] is not None
    assert raw_center["solution"][0] == pytest.approx(10.0 / 11.0, abs=1e-4)
    assert geometric_center["solution"][0] == pytest.approx(0.5, abs=1e-4)


def test_candidate_cvxpy_solvers_caches_installed_solver_lookup():
    class _FakeCP:
        MOSEK = "MOSEK"
        ECOS = "ECOS"
        SCS = "SCS"
        calls = 0

        @classmethod
        def installed_solvers(cls):
            cls.calls += 1
            return ["SCS", "MOSEK"]

    assert _candidate_cvxpy_solvers(_FakeCP) == ["MOSEK", "SCS"]
    assert _candidate_cvxpy_solvers(_FakeCP) == ["MOSEK", "SCS"]
    assert _FakeCP.calls == 1


def test_integer_and_nonlinear_solver_defaults_are_explicit():
    class _FakeCP:
        GUROBI = "GUROBI"
        MOSEK = "MOSEK"
        ECOS = "ECOS"
        SCS = "SCS"

        @staticmethod
        def installed_solvers():
            return ["SCS", "GUROBI", "MOSEK", "ECOS"]

    assert CVXPY_INTEGER_SOLVER_PRIORITY == ("GUROBI",)
    assert _candidate_cvxpy_solvers(_FakeCP, CVXPY_INTEGER_SOLVER_PRIORITY) == ["GUROBI"]
    assert PYOMO_NLP_SOLVER == "ipopt"


def test_cvxpy_solve_falls_back_in_project_priority_order():
    class _FakeCP:
        MOSEK = "MOSEK"
        ECOS = "ECOS"
        SCS = "SCS"

        @staticmethod
        def installed_solvers():
            return ["SCS", "ECOS", "MOSEK"]

    class _FakeProblem:
        def __init__(self):
            self.seen = []

        def solve(self, **kwargs):
            self.seen.append(kwargs["solver"])
            if kwargs["solver"] in {"MOSEK", "ECOS"}:
                raise RuntimeError(f"{kwargs['solver']} failed")

    problem = _FakeProblem()
    _solve_cvxpy_problem(problem, _FakeCP, warm_start=False)
    assert problem.seen == ["MOSEK", "ECOS", "SCS"]


def test_cvxpy_solve_honors_preferred_solver_without_fallback():
    class _FakeCP:
        MOSEK = "MOSEK"
        SCS = "SCS"

        @staticmethod
        def installed_solvers():
            return ["MOSEK", "SCS"]

    class _FakeProblem:
        def __init__(self):
            self.seen = []

        def solve(self, **kwargs):
            self.seen.append(kwargs["solver"])

    problem = _FakeProblem()
    _solve_cvxpy_problem(problem, _FakeCP, preferred_solver="SCS")
    assert problem.seen == ["SCS"]


def test_cvxpy_gurobi_uses_owned_env(monkeypatch):
    class _FakeEnv:
        instances = []

        def __init__(self, *, empty):
            self.empty = empty
            self.params = {}
            self.started = False
            self.closed = False
            type(self).instances.append(self)

        def setParam(self, key, value):
            self.params[key] = value

        def start(self):
            self.started = True

        def close(self):
            self.closed = True

    class _FakeGurobi:
        Env = _FakeEnv

    class _FakeCP:
        GUROBI = "GUROBI"

        @staticmethod
        def installed_solvers():
            return ["GUROBI"]

    class _FakeProblem:
        def solve(self, **kwargs):
            env = kwargs["env"]
            assert env.empty is True
            assert env.started is True
            assert env.closed is False
            assert env.params["OutputFlag"] == 0
            assert env.params["WLSAccessID"] == "id"

    monkeypatch.setitem(sys.modules, "gurobipy", _FakeGurobi)
    _solve_cvxpy_problem(
        _FakeProblem(),
        _FakeCP,
        preferred_solver="GUROBI",
        solver_options={"gurobi_env_params": {"WLSAccessID": "id"}},
    )

    assert len(_FakeEnv.instances) == 1
    assert _FakeEnv.instances[0].closed is True


def test_cvxpy_gurobi_preserves_user_owned_env(monkeypatch):
    class _FakeEnvFactory:
        def __init__(self, *, empty):
            raise AssertionError("helper should not create an env when one is provided")

    class _FakeGurobi:
        Env = _FakeEnvFactory

    class _FakeCP:
        GUROBI = "GUROBI"

        @staticmethod
        def installed_solvers():
            return ["GUROBI"]

    class _UserEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _FakeProblem:
        def __init__(self, expected_env):
            self.expected_env = expected_env

        def solve(self, **kwargs):
            assert kwargs["env"] is self.expected_env
            assert "gurobi_env_params" not in kwargs

    env = _UserEnv()
    monkeypatch.setitem(sys.modules, "gurobipy", _FakeGurobi)
    _solve_cvxpy_problem(
        _FakeProblem(env),
        _FakeCP,
        preferred_solver="GUROBI",
        solver_options={"env": env, "gurobi_env_params": {"WLSAccessID": "ignored"}},
    )

    assert env.closed is False


def test_jcc_opf_cvxpy_wrapper_uses_configured_solver(monkeypatch):
    calls = []

    class _FakeCP:
        GUROBI = "GUROBI"

    def _fake_solve(problem, cp_mod, **kwargs):
        calls.append((problem, cp_mod, kwargs))

    monkeypatch.setattr(jcc_solver_impl, "_solve_cvxpy_problem", _fake_solve)
    problem = object()
    jcc_solver_impl._solve_jcc_cvxpy_problem(
        problem,
        _FakeCP,
        {
            "solver": "GUROBI",
            "solver_options": {"MIPGap": 1e-4},
            "time_limit_sec": 12.0,
            "verbose": True,
        },
        solver_priority=("MOSEK", "SCS"),
    )

    assert calls == [
        (
            problem,
            _FakeCP,
            {
                "warm_start": True,
                "verbose": True,
                "solver_options": {"MIPGap": 1e-4},
                "time_limit_sec": 12.0,
                "solver_priority": ("MOSEK", "SCS"),
                "preferred_solver": "GUROBI",
            },
        )
    ]


def test_maxcut_solver_edge_index_uses_set_membership_and_reversed_edges():
    solver = MaxCutSolver(
        {
            "node": [0, 1, 2],
            "edge": [(2, 0), (0, 1)],
            "weights": np.array([1.0, 2.0]),
        }
    )
    assert solver.edge_index.tolist() == [1, 0]
