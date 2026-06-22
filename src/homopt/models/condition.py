"""Condition encoder modules used by INN flows."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hid // 2), nn.ReLU(), nn.Linear(n_hid // 2, n_in))

    def forward(self, x):
        return x + self.net(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_layer=1, activation=nn.ReLU):
        super().__init__()
        hid_dim = hidden_dims
        layers = [nn.Linear(input_dim, hid_dim), activation()]
        for _ in range(num_layer):
            layers.append(nn.LayerNorm(hid_dim))
            layers.append(ResBlock(hid_dim, hid_dim))
        layers.append(nn.LayerNorm(hid_dim))
        layers.append(nn.Linear(hid_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PINN(nn.Module):
    """Permutation invariant condition encoder."""

    def __init__(self, input_dim, hidden_dims, output_dim, num_layer=1, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        hid_dim = hidden_dims
        self.phi = MLP(input_dim, hid_dim, hid_dim, num_layer=num_layer, activation=activation)
        self.rho = MLP(hid_dim, hid_dim, output_dim, num_layer=num_layer, activation=activation)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.input_dim)
        return self.rho(self.phi(x).max(1)[0])


class Mixer(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_layer=1, activation=nn.ReLU):
        super().__init__()
        hid_dim = hidden_dims
        self.row_mlp = MLP(input_dim, hid_dim, hid_dim, num_layer=num_layer, activation=activation)
        self.col_mlp = MLP(input_dim, hid_dim, hid_dim, num_layer=num_layer, activation=activation)

    def forward(self, x):
        row_emb = self.row_mlp(x)
        col_emb = self.col_mlp(row_emb.permute(0, 1, 3, 2))
        return col_emb.mean(2)


class QuadMixer(nn.Module):
    def __init__(self, n_var, input_dim, hidden_dims, output_dim, num_layer=1, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        hid_dim = hidden_dims
        self.n_var = n_var
        self.mixer = Mixer(n_var, hid_dim, hid_dim, num_layer=1, activation=activation)
        self.emb = MLP(n_var + 1, hid_dim, hid_dim, num_layer=1, activation=activation)
        self.phi = MLP(hid_dim, hid_dim, hid_dim, num_layer=num_layer, activation=activation)
        self.rho = MLP(hid_dim, hid_dim, output_dim, num_layer=num_layer, activation=activation)

    def forward(self, x):
        batch_size = x.shape[0]
        quad_input = x[:, :, : self.n_var ** 2]
        lin_input = x[:, :, self.n_var ** 2 :]
        quad_emb = self.mixer(quad_input.view(batch_size, -1, self.n_var, self.n_var))
        lin_emb = self.emb(lin_input)
        emb = self.phi(lin_emb + quad_emb).max(1)[0]
        return self.rho(emb)


def normalize_condition_encoder_type(value):
    key = "none" if value is None else str(value).strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "identity": "none",
        "raw": "none",
        "pi": "pi",
        "pinn": "pi",
        "permutation_invariant": "pi",
        "mix": "mix",
        "mixer": "mix",
        "quad_mixer": "mix",
        "mlp": "mlp",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported condition encoder type: {value}. Use one of: none, PI, Mix, MLP.")
    return aliases[key]


def build_condition_encoder(encoder_type, *, n_var, input_dim, hidden_dim, output_dim, num_layer=2):
    encoder_type = normalize_condition_encoder_type(encoder_type)
    if encoder_type == "none":
        return None
    if encoder_type == "pi":
        return PINN(input_dim, hidden_dim, output_dim, num_layer=num_layer)
    if encoder_type == "mix":
        return QuadMixer(n_var, input_dim, hidden_dim, output_dim, num_layer=num_layer)
    if encoder_type == "mlp":
        return MLP(input_dim, hidden_dim, output_dim, num_layer=num_layer)
    raise AssertionError(f"Unhandled condition encoder type: {encoder_type}")


__all__ = [
    "MLP",
    "Mixer",
    "PINN",
    "QuadMixer",
    "ResBlock",
    "build_condition_encoder",
    "normalize_condition_encoder_type",
]
