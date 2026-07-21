from homopt.experiments import run_and_record
from homopt.learning import BasePredictor as LearningBasePredictor, BaseRefiner as LearningBaseRefiner
from homopt.mappings import GaugeMap, StarMap
from homopt.mappings.base import BaseMap
from homopt.models import INN
from homopt.optim import BaseOptimizer, INNPGDOptimizer
from homopt.optim.alm import HomALMOptimizer
from homopt.optim.first_order import HomPGDOptimizer
from homopt.problems import BaseProblem, ConvexOpt, ConvexOptEq, JCCDCOPFProblem, NonConvexQCProblem, ToyStarOpt


def test_canonical_namespaces_export_shared_components():
    assert BaseProblem is not None
    assert BaseMap is not None
    assert BaseOptimizer is not None
    assert LearningBasePredictor is not None
    assert LearningBaseRefiner is not None
    assert run_and_record is not None


def test_canonical_namespaces_export_hom_pgd_components():
    assert ConvexOpt is not None
    assert ToyStarOpt is not None
    assert GaugeMap is not None
    assert StarMap is not None
    assert HomPGDOptimizer is not None


def test_canonical_namespaces_export_hom_alm_components():
    assert ConvexOptEq is not None
    assert GaugeMap is not None
    assert HomALMOptimizer is not None


def test_canonical_namespaces_export_inn_pgd_components():
    assert INN is not None
    assert INNPGDOptimizer is not None
    assert JCCDCOPFProblem is not None
    assert NonConvexQCProblem is not None
