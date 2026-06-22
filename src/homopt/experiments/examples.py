"""Small package-owned experiment examples used by docs and tests."""

from __future__ import annotations

import torch

from homopt.learning import ConstantPredictor, ProjectionRefiner, summarize_prediction_route
from homopt.problems import ConvexOpt, ParametricProblemBase, ToyStarOpt, create_test_problem
from homopt.utils import set_global_seed


def convex_problem_summary(seed=0, n_var=2, n_linear_cons=1, x_lower=-1.0, x_upper=1.0):
    set_global_seed(seed)
    config = create_test_problem({
        "seed": seed,
        "n_var": n_var,
        "n_linear_cons": n_linear_cons,
        "n_soc_cons": 0,
        "n_qua_cons": 0,
        "obj": "quad",
        "x_lower": x_lower,
        "x_upper": x_upper,
    })
    problem = ConvexOpt(config)
    x0 = torch.zeros(1, problem.nvar)
    objective = float(problem.objective_x(x0).item())
    max_violation = float(problem.constraint_x(x0, clip=True).max().item())
    return {
        "objective": objective,
        "feasible": max_violation <= 1e-8,
        "metrics": {
            "nvar": int(problem.nvar),
            "ncon": int(problem.ncon),
            "max_violation": max_violation,
            "seed": seed,
        },
    }


def toy_star_summary(alpha=1.0, num_star=4):
    problem = ToyStarOpt(alpha=alpha, num_star=num_star)
    x0 = torch.zeros(1, problem.nvar)
    objective = float(problem.objective_x(x0).item())
    max_violation = float(problem.constraint_x(x0, clip=True).max().item())
    return {
        "objective": objective,
        "feasible": max_violation <= 1e-8,
        "metrics": {
            "alpha": float(alpha),
            "num_star": int(num_star),
            "max_violation": max_violation,
        },
    }


class _ToyLearningProblem(ParametricProblemBase):
    def __init__(self, n_var=2):
        self.nvar = int(n_var)
        self.is_parametric = True

    def objective_xy(self, x, y):
        target = 0.5 * x
        return ((y - target) ** 2).sum(dim=-1, keepdim=True)

    def constraint_residual_xy(self, x, y, clip=True):
        lower = torch.maximum(x - 0.25, torch.full_like(x, -1.0))
        upper = torch.minimum(x + 0.25, torch.full_like(x, 1.0))
        residual = torch.cat([lower - y, y - upper], dim=1)
        return torch.clamp(residual, min=0.0) if clip else residual

    def project_xy(self, x, y):
        lower = torch.maximum(x - 0.25, torch.full_like(x, -1.0))
        upper = torch.minimum(x + 0.25, torch.full_like(x, 1.0))
        return torch.minimum(torch.maximum(y, lower), upper)


def learning_route_summary(seed=0, n_var=2, n_samples=8, prediction_value=0.0):
    set_global_seed(seed)
    grid = torch.linspace(-0.75, 0.75, steps=n_samples).view(-1, 1)
    x = grid.repeat(1, n_var)
    problem = _ToyLearningProblem(n_var=n_var)
    predictor = ConstantPredictor(prediction_value)
    refiner = ProjectionRefiner()
    summary = summarize_prediction_route(problem, x, predictor, refiner)
    summary.update({
        "objective": float(summary["refined_objective_mean"]),
        "feasible": summary["refined_feasibility_rate"] >= 1.0 - 1e-12,
    })
    metrics = dict(summary)
    objective = metrics.pop("objective")
    feasible = metrics.pop("feasible")
    metrics.update({
        "seed": int(seed),
        "n_var": int(n_var),
        "n_samples": int(n_samples),
        "prediction_value": float(prediction_value),
    })
    return {"objective": objective, "feasible": feasible, "metrics": metrics}
