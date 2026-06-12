"""Reusable neural-network model definitions."""

from __future__ import annotations

import torch
import torch.nn as nn

from .io import load_torch_model, save_torch_model

PREDICTOR_MODEL_FILENAME = "decision_predictor.pt"


class ResidualMLPBlock(nn.Module):
    """Minimal residual block reused by small MLP model families."""

    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(width), int(width)),
            nn.ReLU(),
            nn.Linear(int(width), int(width)),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualMLP(nn.Module):
    """Residual MLP backbone with a shared width across blocks."""

    def __init__(self, nin, nout, nhid, nl, *, dropout=0.1):
        super().__init__()
        width = int(nhid)
        layers = [nn.Linear(int(nin), width)]
        for _ in range(int(nl)):
            layers += [ResidualMLPBlock(width), nn.Dropout(float(dropout))]
        layers.append(nn.Linear(width, int(nout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def apply_bounded_output(x, outact, *, symmetric_sigmoid=False):
    outact = None if outact is None else str(outact).strip().lower()
    if outact is None:
        return x
    if outact == "tanh":
        return torch.tanh(x)
    if outact == "sigmoid":
        x = torch.sigmoid(x)
        return 2 * x - 1 if symmetric_sigmoid else x
    raise ValueError(f"Unsupported output activation: {outact}")


class DecisionPredictorNet(nn.Module):
    """Residual MLP that predicts normalized decisions from instance parameters."""

    def __init__(self, nin, nout, nhid, nl, *, outact="tanh", dropout=0.1):
        super().__init__()
        self.net = ResidualMLP(nin, nout, nhid, nl, dropout=dropout)
        self.outact = outact

    def forward(self, x):
        return apply_bounded_output(self.net(x), self.outact, symmetric_sigmoid=True)


def save_decision_predictor(model, save_dir, filename=PREDICTOR_MODEL_FILENAME):
    return save_torch_model(model, save_dir, filename)


def load_decision_predictor(path_or_dir, map_location=None):
    return load_torch_model(path_or_dir, PREDICTOR_MODEL_FILENAME, map_location=map_location)


__all__ = [
    "DecisionPredictorNet",
    "PREDICTOR_MODEL_FILENAME",
    "ResidualMLP",
    "ResidualMLPBlock",
    "apply_bounded_output",
    "load_decision_predictor",
    "save_decision_predictor",
]
