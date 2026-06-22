"""Predictor implementations for learning routes."""

from __future__ import annotations

import torch

from .base import BasePredictor


class ConstantPredictor(BasePredictor):
    """Predict a constant candidate with the same shape as the input batch."""

    name = "constant"

    def __init__(self, value=0.0):
        self.value = float(value)

    def predict(self, input_like, **kwargs):
        del kwargs
        return torch.full_like(input_like, self.value)


class ConstantDecisionPredictor(BasePredictor):
    """Predict a constant decision vector with a fixed decision dimension."""

    name = "constant_decision"

    def __init__(self, decision_dim, value=0.0):
        self.decision_dim = int(decision_dim)
        self.value = float(value)

    def predict(self, input_params, **kwargs):
        del kwargs
        batch_size = int(input_params.shape[0])
        return torch.full(
            (batch_size, self.decision_dim),
            self.value,
            dtype=input_params.dtype,
            device=input_params.device,
        )


def _flatten_context_input(input_tensor):
    if input_tensor.ndim == 1:
        return input_tensor.view(1, -1)
    return input_tensor.view(input_tensor.shape[0], -1)


class NeuralDecisionPredictor(BasePredictor):
    """Neural predictor that emits decisions in the physical decision space."""

    name = "nn_decision"

    def __init__(self, model, problem):
        self.model = model
        self.problem = problem

    def predict(self, input_params, **kwargs):
        del kwargs
        self.model.eval()
        with torch.inference_mode():
            u = self.model(_flatten_context_input(input_params))
            y_partial = self.problem.scale(input_params, u)
            return self.problem.complete_partial(input_params, y_partial)


__all__ = ["ConstantDecisionPredictor", "ConstantPredictor", "NeuralDecisionPredictor"]
