"""Optimization routines for gauge-based inequality-constrained methods."""

from homopt.optim.ineq import FrankWolfeOptimizer, HomPGDOptimizer, PGDOptimizer, RadialDualOptimizer, run_gauge_ineq_algorithm

run_algorithm = run_gauge_ineq_algorithm

__all__ = ['FrankWolfeOptimizer', 'HomPGDOptimizer', 'PGDOptimizer', 'RadialDualOptimizer', 'run_algorithm', 'run_gauge_ineq_algorithm']
