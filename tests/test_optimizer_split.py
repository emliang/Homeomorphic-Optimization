import pytest

from homopt.optim.alm import EqualityConstrainedALMOptimizer, HomALMOptimizer, LagrangianOptimizer
from homopt.optim.core import AdamOptimizer, GDOptimizer
from homopt.optim.first_order import FrankWolfeOptimizer, HomPGDOptimizer, PGDOptimizer, RadialDualOptimizer
from homopt.optim.registry import _algorithm_specs


def test_shared_optimizer_layer_exports_primitives():
    assert GDOptimizer is not None
    assert AdamOptimizer is not None


def test_ineq_optimizer_layer_exports_track_specific_optimizers():
    assert PGDOptimizer is not None
    assert HomPGDOptimizer is not None
    assert FrankWolfeOptimizer is not None
    assert RadialDualOptimizer is not None


def test_eq_optimizer_layer_exports_track_specific_optimizers():
    assert EqualityConstrainedALMOptimizer is not None
    assert LagrangianOptimizer is not None
    assert HomALMOptimizer is not None


def test_retired_ineq_facade_is_removed():
    with pytest.raises(ModuleNotFoundError):
        __import__("homopt.optim.ineq")


def test_retired_eq_facade_is_removed():
    with pytest.raises(ModuleNotFoundError):
        __import__("homopt.optim.eq")


def test_registry_remains_single_algorithm_dispatch_surface():
    specs = _algorithm_specs()
    assert "PGD" in specs
    assert "Hom-ALM" in specs
