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


class BaseRefiner(ABC):
    """Unified interface for post-prediction refinement policies."""

    name: str = "refiner"

    @abstractmethod
    def refine(self, problem, x, y, **kwargs):
        """Refine a predicted candidate `y` for problem inputs `x`."""

