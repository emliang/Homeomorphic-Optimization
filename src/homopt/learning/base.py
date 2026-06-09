"""Base interfaces for learning-based solution routes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PredictionResult:
    """Container for prediction/refinement outputs inside learning routes."""

    prediction: Any
    refined: Optional[Any] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class BasePredictor(ABC):
    """Unified interface for learning-based predictors."""

    name: str = "predictor"

    @abstractmethod
    def predict(self, x, **kwargs):
        """Produce a candidate decision for input instances `x`."""


class PredictorAdapter(BasePredictor):
    """Wrap legacy predictor-like objects without changing their implementation."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.name = type(predictor).__name__

    def predict(self, x, **kwargs):
        if hasattr(self.predictor, "predict"):
            return self.predictor.predict(x, **kwargs)
        if hasattr(self.predictor, "__call__"):
            return self.predictor(x, **kwargs)
        raise AttributeError(f"{type(self.predictor).__name__} has neither predict nor __call__.")

    def __getattr__(self, name):
        return getattr(self.predictor, name)


class BaseRefiner(ABC):
    """Unified interface for post-prediction refinement policies."""

    name: str = "refiner"

    @abstractmethod
    def refine(self, problem, x, y, **kwargs):
        """Refine a predicted candidate `y` for problem inputs `x`."""


class RefinerAdapter(BaseRefiner):
    """Wrap legacy refiner-like objects without changing their implementation."""

    def __init__(self, refiner):
        self.refiner = refiner
        self.name = type(refiner).__name__

    def refine(self, problem, x, y, **kwargs):
        if hasattr(self.refiner, "refine"):
            return self.refiner.refine(problem=problem, x=x, y=y, **kwargs)
        if hasattr(self.refiner, "__call__"):
            return self.refiner(problem=problem, x=x, y=y, **kwargs)
        raise AttributeError(f"{type(self.refiner).__name__} has neither refine nor __call__.")

    def __getattr__(self, name):
        return getattr(self.refiner, name)


def as_predictor(predictor):
    """Return `predictor` as a BasePredictor without breaking legacy code."""
    if isinstance(predictor, BasePredictor):
        return predictor
    return PredictorAdapter(predictor)


def as_refiner(refiner):
    """Return `refiner` as a BaseRefiner without breaking legacy code."""
    if isinstance(refiner, BaseRefiner):
        return refiner
    return RefinerAdapter(refiner)
