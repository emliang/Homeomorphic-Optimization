from homopt.experiments import run_and_record
from homopt.mappings import GaugeMap, StarMap
from homopt.models import CvxINN, INN, local_log_stretch_loss
from homopt.optim import (
    HomALMOptimizer,
    HomPGDOptimizer,
    StiefelALMEQPGDOptimizer,
    StiefelRetractionALMOptimizer,
    StiefelRetractionOptimizer,
)
from homopt.problems import ConvexOpt, ConvexOptEq, FixedProblemParametricAdapter, JCCDCOPFProblem, ParametricConvexProblem, ParametricProblemBase, StateBackedParametricProblemBase, StiefelProblem, create_constrained_pca_stiefel_config
from homopt.utils import optional_import


def test_cross_package_exports_are_available():
    assert GaugeMap is not None
    assert StarMap is not None
    assert HomPGDOptimizer is not None
    assert HomALMOptimizer is not None
    assert ConvexOpt is not None
    assert ConvexOptEq is not None
    assert FixedProblemParametricAdapter is not None
    assert ParametricConvexProblem is not None
    assert ParametricProblemBase is not None
    assert StateBackedParametricProblemBase is not None
    assert StiefelProblem is not None
    assert create_constrained_pca_stiefel_config is not None
    assert JCCDCOPFProblem is not None
    assert CvxINN is not None
    assert INN is not None
    assert local_log_stretch_loss is not None
    assert StiefelALMEQPGDOptimizer is not None
    assert StiefelRetractionALMOptimizer is not None
    assert StiefelRetractionOptimizer is not None
    assert run_and_record is not None


def test_optional_import_returns_none_for_missing_symbol():
    missing = optional_import('homopt.problems', 'DefinitelyMissingSymbol')
    assert missing is None
