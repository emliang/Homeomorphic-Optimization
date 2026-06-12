"""Learning-based prediction and refinement namespace."""

from .base import BasePredictor, BaseRefiner, PredictionResult
from .bisection import BisectionConfig, BisectionResult, bisect_segment
from .decision_predictor_training import train_decision_predictor
from .eval import summarize_prediction_route
from .inn_training import train_mdh_mapping, unsupervised_training_mdh
from .ipnn_training import train_ipnn_mapping
from .postprocess import (
    DiffProjectionRefiner,
    ExactSolverProjectionRefiner,
    HomeomorphicProjectionRefiner,
    IPNNBisectionRefiner,
    InitializedOptSolverRefiner,
)
from .predictors import NeuralDecisionPredictor
from .simple import ConstantDecisionPredictor, ConstantPredictor, IdentityRefiner, ProjectionRefiner, RayBisectionRefiner

__all__ = [
    "BasePredictor",
    "BaseRefiner",
    "BisectionConfig",
    "BisectionResult",
    "ConstantDecisionPredictor",
    "ConstantPredictor",
    "DiffProjectionRefiner",
    "ExactSolverProjectionRefiner",
    "HomeomorphicProjectionRefiner",
    "IPNNBisectionRefiner",
    "InitializedOptSolverRefiner",
    "IdentityRefiner",
    "NeuralDecisionPredictor",
    "PredictionResult",
    "ProjectionRefiner",
    "RayBisectionRefiner",
    "bisect_segment",
    "summarize_prediction_route",
    "train_decision_predictor",
    "train_ipnn_mapping",
    "train_mdh_mapping",
    "unsupervised_training_mdh",
]
