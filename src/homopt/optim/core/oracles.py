"""Projection and linear-oracle adapters for first-order optimizers."""

from __future__ import annotations

from homopt.optim.alm.lagrangian import LagrangianOptimizer
from homopt.problems import ConvexOpt, LinearProblem, MaxCutSDP, PolyStarOpt, ProjProblem, ToyStarOpt
from homopt.solvers import solve_exact_result

from .solver_cache import _cached_exact_solver, _extract_exact_solution
from .tensors import _as_numpy_vector, _as_torch_row


def _project_problem_point(owner, x_init, *, subproblem_params, device, dtype=None):
    """Project ``x_init`` onto ``owner.problem`` using the problem's registered oracle."""

    problem = owner.problem
    if isinstance(problem, ConvexOpt):
        solver = _cached_exact_solver(owner, "convex")
        x_proj = _extract_exact_solution(
            solve_exact_result(solver, "proj", _as_numpy_vector(x_init))
        )
    elif isinstance(problem, MaxCutSDP):
        solver = _cached_exact_solver(owner, "maxcut")
        x_proj = _extract_exact_solution(
            solve_exact_result(solver, "proj", _as_numpy_vector(x_init))
        )
    elif isinstance(problem, (ToyStarOpt, PolyStarOpt)):
        proj_problem = ProjProblem(x_init, problem).to_device(x_init.device)
        lag_optimizer = LagrangianOptimizer(proj_problem, subproblem_params)
        x_proj, _, _, _, _ = lag_optimizer.optimize(initial_point=x_init, verbose=False)
    else:
        raise ValueError(f"Projection is not implemented for {type(problem).__name__}")
    return _as_torch_row(x_proj, device=device, dtype=dtype or x_init.dtype)


def _linear_oracle_point(owner, x, grad, *, subproblem_params):
    """Return the linear minimization oracle point for ``owner.problem``."""

    problem = owner.problem
    if isinstance(problem, ConvexOpt):
        solver = _cached_exact_solver(owner, "convex")
        s = _extract_exact_solution(
            solve_exact_result(
                solver,
                "linear",
                _as_numpy_vector(x),
                _as_numpy_vector(grad),
            )
        )
        return _as_torch_row(s, device=x.device, dtype=x.dtype)
    if isinstance(problem, ToyStarOpt):
        linearization_problem = LinearProblem(x, problem).to_device(x.device)
        lag_optimizer = LagrangianOptimizer(linearization_problem, subproblem_params)
        s, _, _, _, _ = lag_optimizer.optimize(initial_point=x, verbose=False)
        return s
    raise ValueError(f"Linear oracle is not implemented for {type(problem).__name__}")


__all__ = ["_linear_oracle_point", "_project_problem_point"]

