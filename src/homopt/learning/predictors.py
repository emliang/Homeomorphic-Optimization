"""Predictor implementations beyond the constant smoke baselines."""

from __future__ import annotations

import torch

from .base import BasePredictor


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


__all__ = ["NeuralDecisionPredictor"]
