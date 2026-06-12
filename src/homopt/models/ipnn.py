"""Interior-point predictor models."""

from __future__ import annotations

import torch
import torch.nn as nn

from .io import load_torch_model, save_torch_model
from .nn import (
    ResidualMLP,
    apply_bounded_output,
)

IPNN_MODEL_FILENAME = "ipnn_mapping.pt"


class NoiseModule(nn.Module):
    def __init__(self, fixed_margin=False, gamma=1e-3, noise_type="add"):
        super().__init__()
        gamma = max(float(gamma), 1e-8)
        log_gamma = torch.log(torch.tensor(gamma, dtype=torch.float32))
        self.log_gamma = log_gamma if fixed_margin else nn.Parameter(log_gamma)
        self.noise_type = str(noise_type).strip().lower()

    def forward(self, x):
        if not self.training:
            return x
        noise = torch.randn_like(x)
        gamma = torch.exp(self.log_gamma).to(device=x.device, dtype=x.dtype)
        if self.noise_type == "mul":
            return x * (1 + noise.detach() * gamma)
        return x + noise.detach() * gamma


class IPNN(nn.Module):
    """Residual MLP that predicts a feasible interior point in normalized coordinates."""

    def __init__(
        self,
        nin,
        nout,
        nhid,
        nl,
        *,
        fixed_margin=False,
        gamma=1e-3,
        noise_type="add",
        outact="tanh",
    ):
        super().__init__()
        self.backbone = ResidualMLP(nin, nout, nhid, nl, dropout=0.1)
        self.noise = None
        if float(gamma) > 0:
            self.noise = NoiseModule(fixed_margin=fixed_margin, gamma=gamma, noise_type=noise_type)
        self.outact = outact

    def forward(self, x):
        out = self.backbone(x)
        if self.noise is not None:
            out = self.noise(out)
        return apply_bounded_output(out, self.outact, symmetric_sigmoid=False)


def save_ipnn_mapping(model, save_dir, filename=IPNN_MODEL_FILENAME):
    return save_torch_model(model, save_dir, filename)


def load_ipnn_mapping(path_or_dir, map_location=None):
    return load_torch_model(path_or_dir, IPNN_MODEL_FILENAME, map_location=map_location)


__all__ = [
    "IPNN",
    "NoiseModule",
    "load_ipnn_mapping",
    "save_ipnn_mapping",
]
