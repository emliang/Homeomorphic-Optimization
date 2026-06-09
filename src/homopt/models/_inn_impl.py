"""Structured shared INN/model/training implementation."""

import time
import pickle
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from homopt.models.flows.coupling import CombinedActNormLU, CouplingLayer, MADE, MaskedLinear
from homopt.models.flows.iresblock import iResidualLayer
from homopt.models.flows.lipschitz import SpectralNormLinear
from homopt.mappings.gauge import GaugeMap
from homopt.optim._first_order import run_first_order_loop
from homopt.optim._core_impl import _validate_stepsize_rule
from homopt.optim import AdamOptimizer, GDOptimizer
from homopt.sampling import sampling_sphere
from homopt.utils import cast_tensors_to_dtype, resolve_torch_device, resolve_torch_dtype

DEVICE = resolve_torch_device(default="auto")
INN_MODEL_FILENAME = "inn_mapping.pt"
INN_MODEL_CHECKPOINT_FILENAME = "inn_mapping_checkpoint.pt"


class INN(nn.Module):
    """A sequence of invertible layers."""

    def __init__(
        self,
        nin,
        nhid,
        cin,
        nl,
        inv='made',
        outact='sigmoid',
        bilip=False,
        lip=2,
        Con_type='PI',
        cond_embed_mode='shared',
    ):
        super().__init__()
        del outact  # accepted by the public constructor, not used by current layers
        self.con_type = Con_type
        self.cond_embed_mode = str(cond_embed_mode).strip().lower()
        if self.cond_embed_mode not in {'shared', 'per_layer', 'per_layer_independent'}:
            raise ValueError(
                f"Unsupported cond_embed_mode: {self.cond_embed_mode}. Use 'shared', 'per_layer', or 'per_layer_independent'."
            )
        self._num_conditioned_flows = int(nl)
        self._cond_embed_dim = int(nhid)
        self.track_training_stats = True
        flows = []

        def _build_cond_embed(out_dim):
            if Con_type == 'PI':
                return PINN(cin, nhid, out_dim, num_layer=2)
            if Con_type == 'Mix':
                return QuadMixer(nin, cin, nhid, out_dim, num_layer=2)
            if Con_type == 'MLP':
                return MLP(cin, nhid, out_dim, num_layer=2)
            return None

        self.con_emb = None
        self.con_emb_layers = None
        if self.cond_embed_mode == 'per_layer_independent':
            one_layer_embed = _build_cond_embed(self._cond_embed_dim)
            if one_layer_embed is not None:
                self.con_emb_layers = nn.ModuleList(
                    [one_layer_embed]
                    + [_build_cond_embed(self._cond_embed_dim) for _ in range(self._num_conditioned_flows - 1)]
                )
        elif self.cond_embed_mode == 'per_layer':
            # Fast path: compute all layer condition embeddings in one forward.
            self.con_emb = _build_cond_embed(self._num_conditioned_flows * self._cond_embed_dim)
        else:
            self.con_emb = _build_cond_embed(self._cond_embed_dim)

        if (self.con_emb is not None) or (self.con_emb_layers is not None):
            cin = nhid
        for _ in range(nl):
            # flows += [CombinedActNormLU(nin)]
            if inv == 'made':
                flows.append(MADE(nin, nhid, cin, bilip=bilip, lip=lip))
            elif inv == 'coupling':
                flows.append(CouplingLayer(nin, nhid, cin, bilip=bilip, lip=lip))
            elif inv == 'residual':
                flows.append(iResidualLayer(nin, nhid, cin))
            else:
                raise ValueError('Unknown invertible layer type: {}'.format(inv))
            flows += [CombinedActNormLU(nin)]
        self.flows = nn.ModuleList(flows)
        self._is_conditioned_flow = tuple(not isinstance(flow, CombinedActNormLU) for flow in self.flows)

    def _prepare_condition_input(self, c, batch_size):
        if self.con_emb is None and self.con_emb_layers is None:
            return c.reshape(batch_size, -1)
        if self.con_type == 'MLP':
            return c.reshape(batch_size, -1)
        return c

    def _normalize_shared_condition_embedding(self, c_emb, batch_size):
        if c_emb.dim() > 2:
            c_emb = c_emb.reshape(batch_size, -1, c_emb.shape[-1]).mean(1)
        return c_emb

    def _flow_condition(self, c_emb, cond_ptr):
        if c_emb is None:
            return None
        return c_emb[cond_ptr] if c_emb.dim() == 3 else c_emb

    def embed_condition(self, c, batch_size=None):
        if c is None:
            return None
        if batch_size is None:
            batch_size = c.shape[0]
        cond_in = self._prepare_condition_input(c, batch_size)
        if self.con_emb_layers is not None:
            embs = []
            for emb_layer in self.con_emb_layers:
                embs.append(self._normalize_shared_condition_embedding(emb_layer(cond_in), batch_size))
            return torch.stack(embs, dim=0)
        if self.con_emb is not None:
            c_emb = self.con_emb(cond_in)
            if self.cond_embed_mode == 'per_layer':
                if c_emb.dim() > 2:
                    c_emb = c_emb.reshape(batch_size, -1)
                c_emb = c_emb.reshape(batch_size, self._num_conditioned_flows, self._cond_embed_dim)
                return c_emb.permute(1, 0, 2).contiguous()
            return self._normalize_shared_condition_embedding(c_emb, batch_size)
        return cond_in

    def forward_embedded(self, x, c_emb=None):
        b = x.shape[0]
        cond_ptr = 0
        if self.training and self.track_training_stats:
            log_det = torch.zeros(b, device=x.device, dtype=x.dtype)
            log_dis = torch.zeros(b, device=x.device, dtype=x.dtype)
            for flow, needs_cond in zip(self.flows, self._is_conditioned_flow):
                flow_c = self._flow_condition(c_emb, cond_ptr) if needs_cond else None
                cond_ptr += int(needs_cond)
                x, ls = flow.forward(x, flow_c)
                ld = ls.sum(-1)
                dis = torch.amax(ls, dim=-1)
                log_det += ld.view(-1)
                log_dis += dis.view(-1)
            return x, log_det, log_dis
        for flow, needs_cond in zip(self.flows, self._is_conditioned_flow):
            flow_c = self._flow_condition(c_emb, cond_ptr) if needs_cond else None
            cond_ptr += int(needs_cond)
            if isinstance(flow, iResidualLayer):
                x, _ = flow.forward(x, flow_c, compute_logdet=False)
            else:
                x, _ = flow.forward(x, flow_c)
        return x

    def forward(self, x, c=None):
        c_emb = self.embed_condition(c, x.shape[0])
        return self.forward_embedded(x, c_emb)

    def inverse_embedded(self, z, c_emb=None):
        cond_ptr = self._num_conditioned_flows - 1
        for flow, needs_cond in zip(reversed(self.flows), reversed(self._is_conditioned_flow)):
            flow_c = self._flow_condition(c_emb, cond_ptr) if needs_cond else None
            cond_ptr -= int(needs_cond)
            z, _ = flow.forward(z, flow_c, mode='inverse')
        if self.training:
            return z, None, None
        return z

    def inverse(self, z, c):
        c_emb = self.embed_condition(c, z.shape[0])
        return self.inverse_embedded(z, c_emb)


def radial_compactify(x, eps=1e-12):
    norm = torch.norm(x, dim=1, p=2, keepdim=True)
    safe_norm = norm.clamp_min(eps)
    scale = norm / (1.0 + norm)
    return torch.where(norm > eps, x / safe_norm * scale, torch.zeros_like(x))


def radial_decompactify(z, eps=1e-12):
    norm = torch.norm(z, dim=1, p=2, keepdim=True)
    safe_norm = norm.clamp_min(eps)
    clipped_norm = norm.clamp_max(1.0 - eps)
    scale = clipped_norm / (1.0 - clipped_norm)
    return torch.where(norm > eps, z / safe_norm * scale, torch.zeros_like(z))


def cube_compactify(x):
    return torch.tanh(x)


def cube_decompactify(z, eps=1e-12):
    clipped = torch.clamp(z, min=-1.0 + float(eps), max=1.0 - float(eps))
    return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))


class CvxINN(nn.Module):
    """Composite convex-constrained INN for exact convex-set homeomorphisms.

    This wrapper keeps an unconstrained invertible model in Euclidean space and
    composes it with:

    1. a radial decompactification from the open unit ball to `R^n`,
    2. the unconstrained INN in `R^n`,
    3. a radial compactification back to the open unit ball,
    4. an exact `GaugeMap` from the unit ball to the target convex set.

    The composite map is therefore structurally:

        ball -> R^n -> unconstrained INN -> ball -> convex set

    This gives a clean constrained route without modifying the inner INN
    architecture. The constrained output is exact at the level of the outer
    homeomorphism. Note that the returned `logdet/logdis` in training mode still
    come from the inner INN only; the current MDH training objective is not yet
    augmented with the Jacobian terms of the compactification/gauge layers.
    """

    def __init__(self, base_model, constraint_map, *, gauge_forward_method="autograd", compactify_eps=1e-12):
        super().__init__()
        self.base_model = base_model
        self.constraint_map = constraint_map
        self.gauge_forward_method = str(gauge_forward_method)
        self.compactify_eps = float(compactify_eps)

    @classmethod
    def from_convex_set(
        cls,
        base_model,
        convex_set,
        *,
        p_norm=2,
        x_origin=None,
        smooth=False,
        gauge_forward_method="autograd",
        compactify_eps=1e-12,
    ):
        return cls(
            base_model,
            GaugeMap(convex_set, p_norm=p_norm, x_origin=x_origin, smooth=smooth),
            gauge_forward_method=gauge_forward_method,
            compactify_eps=compactify_eps,
        )

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if len(args) >= 1:
            first = args[0]
            if isinstance(first, (torch.device, str)):
                device = first
            elif isinstance(first, torch.dtype):
                dtype = first
        if len(args) >= 2 and isinstance(args[1], torch.dtype):
            dtype = args[1]
        if device is not None and hasattr(self.constraint_map, "to_device"):
            self.constraint_map.to_device(device)
        if dtype is not None:
            cast_tensors_to_dtype(self.constraint_map, dtype)
        return module

    def embed_condition(self, c, batch_size=None):
        if hasattr(self.base_model, "embed_condition"):
            return self.base_model.embed_condition(c, batch_size)
        return None

    def forward_unconstrained_embedded(self, z, c_emb=None):
        z_euclid = radial_decompactify(z, eps=self.compactify_eps)
        if hasattr(self.base_model, "forward_embedded"):
            return self.base_model.forward_embedded(z_euclid, c_emb)
        return self.base_model(z_euclid)

    def forward_unconstrained(self, z, c=None):
        c_emb = self.embed_condition(c, z.shape[0])
        return self.forward_unconstrained_embedded(z, c_emb)

    def _constrain_output(self, x_unconstrained):
        z_ball = radial_compactify(x_unconstrained, eps=self.compactify_eps)
        return self.constraint_map.forward(z_ball, method=self.gauge_forward_method)

    def forward_embedded(self, z, c_emb=None):
        out = self.forward_unconstrained_embedded(z, c_emb)
        if self.training and isinstance(out, tuple) and len(out) == 3:
            x_unconstrained, logdet, logdis = out
            return self._constrain_output(x_unconstrained), logdet, logdis
        if isinstance(out, tuple):
            out = out[0]
        return self._constrain_output(out)

    def forward(self, z, c=None):
        c_emb = self.embed_condition(c, z.shape[0])
        return self.forward_embedded(z, c_emb)

    def inverse_embedded(self, x, c_emb=None):
        z_ball = self.constraint_map.inverse(x)
        z_euclid = radial_decompactify(z_ball, eps=self.compactify_eps)
        if hasattr(self.base_model, "inverse_embedded"):
            out = self.base_model.inverse_embedded(z_euclid, c_emb)
        else:
            out = self.base_model.inverse(z_euclid)
        if isinstance(out, tuple):
            z_unconstrained = out[0]
            z_ball_out = radial_compactify(z_unconstrained, eps=self.compactify_eps)
            if self.training and len(out) == 3:
                return z_ball_out, out[1], out[2]
            return z_ball_out
        return radial_compactify(out, eps=self.compactify_eps)

    def inverse(self, x, c=None):
        c_emb = self.embed_condition(c, x.shape[0])
        return self.inverse_embedded(x, c_emb)


# Backward-compatible alias while the public name migrates to CvxINN.
ConINN = CvxINN


class TanhGaugeCvxINN(nn.Module):
    """Composite constrained INN using `tanh` to the open cube before GaugeMap.

    Structurally:

        ball -> R^n -> unconstrained INN -> tanh -> open cube -> convex set

    The final GaugeMap uses `p_norm = inf`, so the source set for the outer
    exact map is the open `l_infty` unit ball (the cube).
    """

    def __init__(self, base_model, constraint_map, *, gauge_forward_method="autograd", compactify_eps=1e-12):
        super().__init__()
        self.base_model = base_model
        self.constraint_map = constraint_map
        self.gauge_forward_method = str(gauge_forward_method)
        self.compactify_eps = float(compactify_eps)

    @classmethod
    def from_convex_set(
        cls,
        base_model,
        convex_set,
        *,
        x_origin=None,
        smooth=False,
        gauge_forward_method="autograd",
        compactify_eps=1e-12,
    ):
        return cls(
            base_model,
            GaugeMap(convex_set, p_norm=np.inf, x_origin=x_origin, smooth=smooth),
            gauge_forward_method=gauge_forward_method,
            compactify_eps=compactify_eps,
        )

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if len(args) >= 1:
            first = args[0]
            if isinstance(first, (torch.device, str)):
                device = first
            elif isinstance(first, torch.dtype):
                dtype = first
        if len(args) >= 2 and isinstance(args[1], torch.dtype):
            dtype = args[1]
        if device is not None and hasattr(self.constraint_map, "to_device"):
            self.constraint_map.to_device(device)
        if dtype is not None:
            cast_tensors_to_dtype(self.constraint_map, dtype)
        return module

    def embed_condition(self, c, batch_size=None):
        if hasattr(self.base_model, "embed_condition"):
            return self.base_model.embed_condition(c, batch_size)
        return None

    def forward_unconstrained_embedded(self, z, c_emb=None):
        z_euclid = radial_decompactify(z, eps=self.compactify_eps)
        if hasattr(self.base_model, "forward_embedded"):
            return self.base_model.forward_embedded(z_euclid, c_emb)
        return self.base_model(z_euclid)

    def forward_unconstrained(self, z, c=None):
        c_emb = self.embed_condition(c, z.shape[0])
        return self.forward_unconstrained_embedded(z, c_emb)

    def _constrain_output(self, x_unconstrained):
        z_cube = cube_compactify(x_unconstrained)
        return self.constraint_map.forward(z_cube, method=self.gauge_forward_method)

    def forward_embedded(self, z, c_emb=None):
        out = self.forward_unconstrained_embedded(z, c_emb)
        if self.training and isinstance(out, tuple) and len(out) == 3:
            x_unconstrained, logdet, logdis = out
            return self._constrain_output(x_unconstrained), logdet, logdis
        if isinstance(out, tuple):
            out = out[0]
        return self._constrain_output(out)

    def forward(self, z, c=None):
        c_emb = self.embed_condition(c, z.shape[0])
        return self.forward_embedded(z, c_emb)

    def inverse_embedded(self, x, c_emb=None):
        z_cube = self.constraint_map.inverse(x)
        z_euclid = cube_decompactify(z_cube, eps=self.compactify_eps)
        if hasattr(self.base_model, "inverse_embedded"):
            out = self.base_model.inverse_embedded(z_euclid, c_emb)
        else:
            out = self.base_model.inverse(z_euclid)
        if isinstance(out, tuple):
            z_unconstrained = out[0]
            z_ball_out = radial_compactify(z_unconstrained, eps=self.compactify_eps)
            if self.training and len(out) == 3:
                return z_ball_out, out[1], out[2]
            return z_ball_out
        return radial_compactify(out, eps=self.compactify_eps)

    def inverse(self, x, c=None):
        c_emb = self.embed_condition(c, x.shape[0])
        return self.inverse_embedded(x, c_emb)


class CubeGaugeCvxINN(nn.Module):
    """Composite constrained INN using a cube-to-cube homeomorphism.

    Structurally:

        cube -> R^n -> unconstrained INN -> cube -> convex set

    The inner unconstrained INN is conjugated by `artanh/tanh`, so the learned
    reparameterization stays native to the open `l_infty` unit ball. The final
    `GaugeMap` also uses `p_norm = inf`, making this a cube-to-cube-to-convex
    constrained route rather than a ball-to-ball route.
    """

    def __init__(self, base_model, constraint_map, *, gauge_forward_method="autograd", compactify_eps=1e-12):
        super().__init__()
        self.base_model = base_model
        self.constraint_map = constraint_map
        self.gauge_forward_method = str(gauge_forward_method)
        self.compactify_eps = float(compactify_eps)

    @classmethod
    def from_convex_set(
        cls,
        base_model,
        convex_set,
        *,
        x_origin=None,
        smooth=False,
        gauge_forward_method="autograd",
        compactify_eps=1e-12,
    ):
        return cls(
            base_model,
            GaugeMap(convex_set, p_norm=np.inf, x_origin=x_origin, smooth=smooth),
            gauge_forward_method=gauge_forward_method,
            compactify_eps=compactify_eps,
        )

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if len(args) >= 1:
            first = args[0]
            if isinstance(first, (torch.device, str)):
                device = first
            elif isinstance(first, torch.dtype):
                dtype = first
        if len(args) >= 2 and isinstance(args[1], torch.dtype):
            dtype = args[1]
        if device is not None and hasattr(self.constraint_map, "to_device"):
            self.constraint_map.to_device(device)
        if dtype is not None:
            cast_tensors_to_dtype(self.constraint_map, dtype)
        return module

    def embed_condition(self, c, batch_size=None):
        if hasattr(self.base_model, "embed_condition"):
            return self.base_model.embed_condition(c, batch_size)
        return None

    def forward_unconstrained_embedded(self, z, c_emb=None):
        z_euclid = cube_decompactify(z, eps=self.compactify_eps)
        if hasattr(self.base_model, "forward_embedded"):
            return self.base_model.forward_embedded(z_euclid, c_emb)
        return self.base_model(z_euclid)

    def forward_unconstrained(self, z, c=None):
        c_emb = self.embed_condition(c, z.shape[0])
        return self.forward_unconstrained_embedded(z, c_emb)

    def _constrain_output(self, x_unconstrained):
        z_cube = cube_compactify(x_unconstrained)
        return self.constraint_map.forward(z_cube, method=self.gauge_forward_method)

    def forward_embedded(self, z, c_emb=None):
        out = self.forward_unconstrained_embedded(z, c_emb)
        if self.training and isinstance(out, tuple) and len(out) == 3:
            x_unconstrained, logdet, logdis = out
            return self._constrain_output(x_unconstrained), logdet, logdis
        if isinstance(out, tuple):
            out = out[0]
        return self._constrain_output(out)

    def forward(self, z, c=None):
        c_emb = self.embed_condition(c, z.shape[0])
        return self.forward_embedded(z, c_emb)

    def inverse_embedded(self, x, c_emb=None):
        z_cube = self.constraint_map.inverse(x)
        z_euclid = cube_decompactify(z_cube, eps=self.compactify_eps)
        if hasattr(self.base_model, "inverse_embedded"):
            out = self.base_model.inverse_embedded(z_euclid, c_emb)
        else:
            out = self.base_model.inverse(z_euclid)
        if isinstance(out, tuple):
            z_unconstrained = out[0]
            z_cube_out = cube_compactify(z_unconstrained)
            if self.training and len(out) == 3:
                return z_cube_out, out[1], out[2]
            return z_cube_out
        return cube_compactify(out)

    def inverse(self, x, c=None):
        c_emb = self.embed_condition(c, x.shape[0])
        return self.inverse_embedded(x, c_emb)


class SphereReparam(nn.Module):
    """Exact ball-to-ball homeomorphism via angular reparameterization.

    For `z = r u` with `r = ||z||_2` and `u` on the sphere, this model keeps
    the radius fixed and applies an invertible projective map on directions:

        u -> A u / ||A u||_2

    where `A = exp(M)` is always invertible. The inverse uses `A^{-1} = exp(-M)`,
    so this gives an exact homeomorphism of the open unit ball without leaving
    the ball domain.
    """

    def __init__(self, n_dim, init_scale=0.0):
        super().__init__()
        self.n_dim = int(n_dim)
        init = torch.eye(self.n_dim, dtype=torch.float32) * float(init_scale)
        self.raw_matrix = nn.Parameter(init)

    def _matrix(self):
        return torch.matrix_exp(self.raw_matrix)

    def _apply_direction_map(self, z, matrix):
        norm = torch.norm(z, dim=1, p=2, keepdim=True)
        safe_norm = norm.clamp_min(1e-12)
        direction = z / safe_norm
        mapped = torch.matmul(direction, matrix.T)
        mapped = mapped / torch.norm(mapped, dim=1, p=2, keepdim=True).clamp_min(1e-12)
        return torch.where(norm > 1e-12, mapped * norm, torch.zeros_like(z))

    def embed_condition(self, c, batch_size=None):
        del c, batch_size
        return None

    def forward_embedded(self, z, c_emb=None):
        del c_emb
        return self._apply_direction_map(z, self._matrix())

    def forward(self, z, c=None):
        del c
        return self.forward_embedded(z)

    def inverse_embedded(self, z, c_emb=None):
        del c_emb
        return self._apply_direction_map(z, torch.matrix_exp(-self.raw_matrix))

    def inverse(self, z, c=None):
        del c
        return self.inverse_embedded(z)


class SphereReparamGaugeCvxINN(nn.Module):
    """Constrained route using an exact sphere-direction reparameterization.

    Structurally:

        ball -> SphereReparam(ball) -> GaugeMap(ball -> convex set)

    Unlike `CvxINN`, this wrapper never leaves the ball domain. It only learns
    an angular reparameterization of the source ball before the final exact
    convex-set map.
    """

    def __init__(self, base_model, constraint_map, *, gauge_forward_method="autograd"):
        super().__init__()
        self.base_model = base_model
        self.constraint_map = constraint_map
        self.gauge_forward_method = str(gauge_forward_method)

    @classmethod
    def from_convex_set(
        cls,
        base_model,
        convex_set,
        *,
        p_norm=2,
        x_origin=None,
        smooth=False,
        gauge_forward_method="autograd",
    ):
        return cls(
            base_model,
            GaugeMap(convex_set, p_norm=p_norm, x_origin=x_origin, smooth=smooth),
            gauge_forward_method=gauge_forward_method,
        )

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if len(args) >= 1:
            first = args[0]
            if isinstance(first, (torch.device, str)):
                device = first
            elif isinstance(first, torch.dtype):
                dtype = first
        if len(args) >= 2 and isinstance(args[1], torch.dtype):
            dtype = args[1]
        if device is not None and hasattr(self.constraint_map, "to_device"):
            self.constraint_map.to_device(device)
        if dtype is not None:
            cast_tensors_to_dtype(self.constraint_map, dtype)
        return module

    def embed_condition(self, c, batch_size=None):
        if hasattr(self.base_model, "embed_condition"):
            return self.base_model.embed_condition(c, batch_size)
        return None

    def forward_reparam_embedded(self, z, c_emb=None):
        if hasattr(self.base_model, "forward_embedded"):
            return self.base_model.forward_embedded(z, c_emb)
        return self.base_model(z)

    def _constrain_output(self, z_ball):
        return self.constraint_map.forward(z_ball, method=self.gauge_forward_method)

    def forward_embedded(self, z, c_emb=None):
        out = self.forward_reparam_embedded(z, c_emb)
        if self.training and isinstance(out, tuple) and len(out) == 3:
            z_ball, logdet, logdis = out
            return self._constrain_output(z_ball), logdet, logdis
        if isinstance(out, tuple):
            out = out[0]
        return self._constrain_output(out)

    def forward(self, z, c=None):
        c_emb = self.embed_condition(c, z.shape[0])
        return self.forward_embedded(z, c_emb)

    def inverse_embedded(self, x, c_emb=None):
        z_ball = self.constraint_map.inverse(x)
        if hasattr(self.base_model, "inverse_embedded"):
            out = self.base_model.inverse_embedded(z_ball, c_emb)
        else:
            out = self.base_model.inverse(z_ball)
        return out

    def inverse(self, x, c=None):
        c_emb = self.embed_condition(c, x.shape[0])
        return self.inverse_embedded(x, c_emb)


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
            # layers += [nn.Linear(hid_dim, hid_dim), activation()]
            layers.append(ResBlock(hid_dim, hid_dim))
        layers.append(nn.LayerNorm(hid_dim))
        layers.append(nn.Linear(hid_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PINN(nn.Module):
    """Permutation Invariant Neural Network."""

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


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])

    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.backup[name])
        self.backup = {}


def initialize_inn_as_identity(model):
    """Initialize an INN close to the identity map.

    This is especially useful for constrained wrappers such as ``CvxINN``:
    with an identity inner INN, the end-to-end map reduces to the outer
    `GaugeMap` route, providing a strong low-distortion baseline before
    learning a nontrivial ball-to-ball homeomorphism.
    """

    with torch.no_grad():
        if hasattr(model, "flows"):
            for flow in model.flows:
                if isinstance(flow, CombinedActNormLU):
                    flow.weight.zero_()
                    flow.bias.zero_()
                    flow.initialized = True
                    flow.LU.zero_()
                    flow.log_S.zero_()
                    flow.sign_S.fill_(1.0)
                    flow.P.copy_(torch.eye(flow.P.shape[0], device=flow.P.device, dtype=flow.P.dtype))
                    continue
                if isinstance(flow, CouplingLayer):
                    for module in flow.net:
                        if isinstance(module, nn.Linear):
                            module.weight.zero_()
                            if module.bias is not None:
                                module.bias.zero_()
                    continue
                if isinstance(flow, MADE):
                    joiner = getattr(flow, "joiner", None)
                    if joiner is not None:
                        joiner.linear.weight.zero_()
                        if joiner.linear.bias is not None:
                            joiner.linear.bias.zero_()
                        if hasattr(joiner, "cond_linear"):
                            for module in joiner.cond_linear:
                                if isinstance(module, nn.Linear):
                                    module.weight.zero_()
                                    if module.bias is not None:
                                        module.bias.zero_()
                    for module in flow.trunk:
                        if isinstance(module, MaskedLinear):
                            module.linear.weight.zero_()
                            if module.linear.bias is not None:
                                module.linear.bias.zero_()
                            if hasattr(module, "cond_linear"):
                                for submodule in module.cond_linear:
                                    if isinstance(submodule, nn.Linear):
                                        submodule.weight.zero_()
                                        if submodule.bias is not None:
                                            submodule.bias.zero_()
                    continue
                if isinstance(flow, iResidualLayer):
                    lipnet = getattr(flow, "lipnet", None)
                    if lipnet is None:
                        continue
                    for module in lipnet.modules():
                        if isinstance(module, nn.Linear):
                            module.weight.zero_()
                            if module.bias is not None:
                                module.bias.zero_()
                        elif isinstance(module, SpectralNormLinear):
                            module.weight.zero_()
                            if module.bias is not None:
                                module.bias.zero_()
        return model


def _build_quadratic_terms(data, device, dtype):
    fixed_Q = torch.as_tensor(data.fixed_Q, dtype=dtype, device=device)
    fixed_p = torch.as_tensor(data.fixed_p, dtype=dtype, device=device)
    return fixed_Q, fixed_p


def _quadratic_objective(x_full, fixed_Q, fixed_p):
    quad_term = 0.5 * torch.sum((x_full @ fixed_Q) * x_full, dim=1)
    linear_term = torch.sum(fixed_p * x_full, dim=1)
    return quad_term + linear_term


def save_inn_mapping(model, save_dir, filename=INN_MODEL_FILENAME):
    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / filename
    torch.save(model, model_path)
    return model_path


def load_inn_mapping(path_or_dir, map_location=None):
    def _load_checkpoint(path_obj):
        try:
            return torch.load(path_obj, map_location=map_location)
        except pickle.UnpicklingError as exc:
            # PyTorch >= 2.6 defaults to weights_only=True. Our checkpoints store
            # full model objects, so retry with weights_only=False for trusted files.
            if "Weights only load failed" not in str(exc):
                raise
            return torch.load(path_obj, map_location=map_location, weights_only=False)

    path = Path(path_or_dir)
    if path.is_dir():
        candidate = path / INN_MODEL_FILENAME
        if candidate.exists():
            return _load_checkpoint(candidate)
        raise FileNotFoundError(f"No INN mapping checkpoint found in {path}")
    return _load_checkpoint(path)


def _embed_condition_if_available(model, input_params, batch_size, *, detach=False):
    if not hasattr(model, "embed_condition"):
        return None
    # Use no_grad rather than inference_mode so the returned tensor can still be
    # consumed in autograd-enabled forwards without becoming an inference tensor.
    with torch.no_grad():
        condition_emb = model.embed_condition(input_params, batch_size)
    return condition_emb.detach() if detach else condition_emb


def _forward_with_condition(model, z_tensor, input_params, condition_emb=None):
    if hasattr(model, "forward_embedded"):
        return model.forward_embedded(z_tensor, condition_emb)
    return model(z_tensor, input_params)


def _extract_forward_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _project_to_latent_domain(z_tensor, domain_shape="ball", boundary_eps=1e-6):
    shape = str(domain_shape).strip().lower()
    if shape in {"ball", "sphere", "l2_ball"}:
        norm = torch.norm(z_tensor, dim=-1, p=2, keepdim=True).clamp_min(1e-12)
        max_norm = 1.0 - float(boundary_eps)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return z_tensor * scale
    if shape in {"cube", "square", "linf_ball"}:
        limit = 1.0 - float(boundary_eps)
        return torch.clamp(z_tensor, min=-limit, max=limit)
    raise ValueError(f"Unsupported latent domain shape: {domain_shape}")


def local_log_stretch_loss(
    forward_fn,
    z_tensor,
    *,
    num_directions=8,
    fd_eps=1e-3,
    domain_shape="ball",
    boundary_eps=1e-6,
    log_eps=1e-8,
    aggregate="max",
):
    """Estimate end-to-end local stretch with a zeroth-order log-stabilized loss.

    For each sample ``z`` and random unit direction ``v``, this uses the
    symmetric finite-difference quotient

        ||T(z + eps v) - T(z - eps v)|| / ||(z + eps v) - (z - eps v)||

    as a directional local stretch estimate. The returned loss applies
    ``log(stretch + log_eps)`` for numerical stability, then aggregates across
    directions by ``max``, ``mean``, or ``spread`` (max-min).
    """

    if int(num_directions) <= 0:
        raise ValueError("num_directions must be positive.")
    batch_size, n_dim = z_tensor.shape
    dirs = torch.randn(batch_size, int(num_directions), n_dim, device=z_tensor.device, dtype=z_tensor.dtype)
    dirs = dirs / torch.norm(dirs, dim=-1, p=2, keepdim=True).clamp_min(1e-12)

    z_base = z_tensor.unsqueeze(1)
    z_plus = _project_to_latent_domain(z_base + float(fd_eps) * dirs, domain_shape=domain_shape, boundary_eps=boundary_eps)
    z_minus = _project_to_latent_domain(z_base - float(fd_eps) * dirs, domain_shape=domain_shape, boundary_eps=boundary_eps)

    flat_plus = z_plus.reshape(-1, n_dim)
    flat_minus = z_minus.reshape(-1, n_dim)
    out_plus = forward_fn(flat_plus)
    out_minus = forward_fn(flat_minus)
    out_plus = _extract_forward_tensor(out_plus).reshape(batch_size, int(num_directions), -1)
    out_minus = _extract_forward_tensor(out_minus).reshape(batch_size, int(num_directions), -1)

    output_diff = torch.norm(out_plus - out_minus, dim=-1, p=2)
    input_diff = torch.norm(z_plus - z_minus, dim=-1, p=2).clamp_min(float(log_eps))
    stretch = output_diff / input_diff
    log_stretch = torch.log(stretch.clamp_min(float(log_eps)))

    aggregate_mode = str(aggregate).strip().lower()
    if aggregate_mode == "max":
        per_sample = log_stretch.max(dim=1)[0]
    elif aggregate_mode == "mean":
        per_sample = log_stretch.mean(dim=1)
    elif aggregate_mode == "spread":
        per_sample = log_stretch.max(dim=1)[0] - log_stretch.min(dim=1)[0]
    else:
        raise ValueError(f"Unsupported local stretch aggregate mode: {aggregate}")

    loss = per_sample.mean()
    stats = {
        "loss": loss,
        "per_sample": per_sample,
        "stretch": stretch,
        "log_stretch": log_stretch,
    }
    return loss, stats


def _robust_log_stretch_spread(log_stretch, trim_k=2):
    k = max(1, min(int(trim_k), max(log_stretch.shape[1] // 2, 1)))
    top = torch.topk(log_stretch, k=k, dim=1, largest=True, sorted=False)[0].mean(dim=1)
    bottom = torch.topk(log_stretch, k=k, dim=1, largest=False, sorted=False)[0].mean(dim=1)
    return top - bottom


def _soft_log_stretch_spread(log_stretch, temperature=0.1):
    tau = max(float(temperature), 1e-6)
    soft_max = tau * torch.logsumexp(log_stretch / tau, dim=1)
    soft_min = -tau * torch.logsumexp(-log_stretch / tau, dim=1)
    return soft_max - soft_min


def combine_local_stretch_objective(
    loss_spread,
    stats,
    objective="spread",
    soft_temperature=0.1,
    spread_weight=0.25,
):
    """Build a training objective from local-stretch summary statistics."""

    mode = str(objective).strip().lower()
    if mode == "spread":
        return loss_spread
    log_stretch = stats["log_stretch"]
    centered = log_stretch - log_stretch.mean(dim=1, keepdim=True)
    variance_loss = centered.pow(2).mean()
    weight = float(spread_weight)
    if mode == "variance":
        return variance_loss
    if mode == "variance_plus_spread":
        return variance_loss + weight * loss_spread
    if mode == "variance_plus_robust_spread":
        robust_spread = _robust_log_stretch_spread(log_stretch, trim_k=2).mean()
        return variance_loss + weight * robust_spread
    if mode == "soft_spread":
        return _soft_log_stretch_spread(log_stretch, temperature=soft_temperature).mean()
    if mode == "variance_plus_soft_spread":
        soft_spread = _soft_log_stretch_spread(log_stretch, temperature=soft_temperature).mean()
        return variance_loss + weight * soft_spread
    raise ValueError(f"Unsupported local stretch objective: {objective}")


def _evaluate_local_stretch_objective(
    forward_fn,
    z_tensor,
    *,
    num_directions,
    fd_eps,
    domain_shape,
    boundary_eps,
    log_eps,
    aggregate,
    objective,
    soft_temperature,
    spread_weight,
):
    loss_spread, stats = local_log_stretch_loss(
        forward_fn,
        z_tensor,
        num_directions=num_directions,
        fd_eps=fd_eps,
        domain_shape=domain_shape,
        boundary_eps=boundary_eps,
        log_eps=log_eps,
        aggregate=aggregate,
    )
    objective_value = combine_local_stretch_objective(
        loss_spread,
        stats,
        objective=objective,
        soft_temperature=soft_temperature,
        spread_weight=spread_weight,
    )
    return objective_value, loss_spread, stats


def _sample_latent_points(
    n_samples,
    n_dim,
    *,
    domain_shape,
    boundary_eps=0.0,
    boundary_focus_ratio=0.0,
    boundary_shell_radius=0.8,
    device,
    dtype,
):
    shape = str(domain_shape).strip().lower()
    max_radius = max(1.0 - float(boundary_eps), 1e-6)
    if shape in {"ball", "sphere", "l2_ball"}:
        points = sampling_sphere(int(n_samples), int(n_dim))
        points = np.asarray(points, dtype=np.float32) * float(max_radius)
        focus_ratio = float(boundary_focus_ratio)
        if focus_ratio > 0:
            n_focus = min(int(round(int(n_samples) * focus_ratio)), int(n_samples))
            if n_focus > 0:
                shell_radius = float(boundary_shell_radius) * float(max_radius)
                shell_radius = max(0.0, min(shell_radius, float(max_radius)))
                focus_points = sampling_sphere(int(n_focus), int(n_dim))
                focus_points = np.asarray(focus_points, dtype=np.float32)
                focus_norm = np.linalg.norm(focus_points, axis=1, keepdims=True)
                focus_norm = np.maximum(focus_norm, 1e-12)
                focus_dir = focus_points / focus_norm
                focus_r = np.random.uniform(low=shell_radius, high=float(max_radius), size=(n_focus, 1)).astype(np.float32)
                points[:n_focus] = focus_dir * focus_r
    elif shape in {"cube", "square", "linf_ball"}:
        points = np.random.uniform(low=-max_radius, high=max_radius, size=(int(n_samples), int(n_dim))).astype(np.float32)
    else:
        raise ValueError(f"Unsupported latent domain shape: {domain_shape}")
    return torch.tensor(points, device=device, dtype=dtype)


def train_cvxinn_min_distortion(
    convex_set,
    *,
    base_model=None,
    wrapper_type="radial_gauge",
    n_dim=None,
    h_dim=32,
    num_layer=2,
    inv_type="coupling",
    total_iteration=500,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    p_norm=2,
    gauge_forward_method="autograd",
    compactify_eps=1e-12,
    track_logdet_stats=False,
    stretch_num_directions=8,
    stretch_fd_eps=1e-3,
    stretch_log_eps=1e-8,
    stretch_boundary_eps=1e-6,
    stretch_aggregate="spread",
    stretch_objective="spread",
    stretch_soft_temperature=0.1,
    stretch_spread_weight=0.25,
    identity_init_inner=False,
    boundary_focus_ratio=0.0,
    boundary_shell_radius=0.8,
    grad_clip_norm=None,
    lr_decay_step=None,
    lr_decay_gamma=1.0,
    latent_domain_shape="ball",
    sphere_reparam_init_scale=0.0,
    record_every=10,
    eval_every=None,
    eval_batch_size=512,
    eval_z_tensor=None,
    early_stop_patience=None,
    early_stop_min_delta=0.0,
    early_stop_min_iterations=0,
    seed=2025,
    device=None,
    dtype=None,
):
    """Train a CvxINN on a fixed convex set using only end-to-end distortion loss."""

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    runtime_device = resolve_torch_device(device, default="cpu")
    runtime_dtype = resolve_torch_dtype(dtype, default=torch.float32)
    if n_dim is None:
        n_dim = int(convex_set.nvar)

    wrapper_mode = str(wrapper_type).strip().lower()

    if base_model is None and wrapper_mode in {"sphere_reparam_gauge", "sphere_reparam", "sphere"}:
        base_model = SphereReparam(n_dim=n_dim, init_scale=float(sphere_reparam_init_scale))
    elif base_model is None:
        base_model = INN(
            nin=n_dim,
            nhid=max(int(h_dim), n_dim + 1),
            cin=1,
            nl=int(num_layer),
            inv=inv_type,
            outact=None,
            bilip=False,
            lip=2,
            Con_type="MLP",
            cond_embed_mode="shared",
        )
    if identity_init_inner:
        initialize_inn_as_identity(base_model)

    base_model = base_model.to(device=runtime_device, dtype=runtime_dtype)
    if hasattr(base_model, "track_training_stats"):
        base_model.track_training_stats = bool(track_logdet_stats)
    if wrapper_mode in {"radial_gauge", "cvxinn", "ball_gauge"}:
        model = CvxINN.from_convex_set(
            base_model,
            convex_set,
            p_norm=p_norm,
            gauge_forward_method=gauge_forward_method,
            compactify_eps=compactify_eps,
        ).to(device=runtime_device, dtype=runtime_dtype)
    elif wrapper_mode in {"tanh_gauge", "tanh"}:
        model = TanhGaugeCvxINN.from_convex_set(
            base_model,
            convex_set,
            gauge_forward_method=gauge_forward_method,
            compactify_eps=compactify_eps,
        ).to(device=runtime_device, dtype=runtime_dtype)
    elif wrapper_mode in {"cube_gauge", "cube_reparam_gauge", "cube"}:
        model = CubeGaugeCvxINN.from_convex_set(
            base_model,
            convex_set,
            gauge_forward_method=gauge_forward_method,
            compactify_eps=compactify_eps,
        ).to(device=runtime_device, dtype=runtime_dtype)
        if str(latent_domain_shape).strip().lower() == "ball":
            latent_domain_shape = "cube"
    elif wrapper_mode in {"sphere_reparam_gauge", "sphere_reparam", "sphere"}:
        model = SphereReparamGaugeCvxINN.from_convex_set(
            base_model,
            convex_set,
            p_norm=p_norm,
            gauge_forward_method=gauge_forward_method,
        ).to(device=runtime_device, dtype=runtime_dtype)
    else:
        raise ValueError(f"Unsupported wrapper_type: {wrapper_type}")
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = None
    if lr_decay_step is not None and float(lr_decay_gamma) < 1.0:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(int(lr_decay_step), 1),
            gamma=float(lr_decay_gamma),
        )
    model.train()

    loss_list = []
    eval_interval = max(int(eval_every if eval_every is not None else record_every), 1)
    if eval_z_tensor is None:
        eval_z = _sample_latent_points(
            eval_batch_size,
            n_dim,
            domain_shape=latent_domain_shape,
            boundary_eps=stretch_boundary_eps,
            boundary_focus_ratio=boundary_focus_ratio,
            boundary_shell_radius=boundary_shell_radius,
            device=runtime_device,
            dtype=runtime_dtype,
        )
    else:
        eval_z = eval_z_tensor.to(device=runtime_device, dtype=runtime_dtype)

    def _forward(flat_z):
        repeated_condition = torch.zeros((flat_z.shape[0], 1), device=runtime_device, dtype=runtime_dtype)
        return model(flat_z, repeated_condition)

    model.eval()
    with torch.no_grad():
        best_eval_objective, best_eval_spread, _ = _evaluate_local_stretch_objective(
            _forward,
            eval_z,
            num_directions=stretch_num_directions,
            fd_eps=stretch_fd_eps,
            domain_shape=latent_domain_shape,
            boundary_eps=stretch_boundary_eps,
            log_eps=stretch_log_eps,
            aggregate=stretch_aggregate,
            objective=stretch_objective,
            soft_temperature=stretch_soft_temperature,
            spread_weight=stretch_spread_weight,
        )
    best_eval_value = float(best_eval_objective.item())
    best_eval_spread_value = float(best_eval_spread.item())
    best_iteration = 0
    best_state = deepcopy(model.state_dict())
    bad_eval_count = 0
    stop_reason = "max_iteration"
    eval_history = [
        {
            "iteration": 0,
            "objective": best_eval_value,
            "spread_loss": best_eval_spread_value,
            "is_best": True,
        }
    ]
    model.train()

    final_iteration = 0
    for iteration in range(1, int(total_iteration) + 1):
        optimizer.zero_grad(set_to_none=True)
        z_batch = _sample_latent_points(
            batch_size,
            n_dim,
            domain_shape=latent_domain_shape,
            boundary_eps=stretch_boundary_eps,
            device=runtime_device,
            dtype=runtime_dtype,
        )
        loss, loss_spread, stretch_stats = _evaluate_local_stretch_objective(
            _forward,
            z_batch,
            num_directions=stretch_num_directions,
            fd_eps=stretch_fd_eps,
            domain_shape=latent_domain_shape,
            boundary_eps=stretch_boundary_eps,
            log_eps=stretch_log_eps,
            aggregate=stretch_aggregate,
            objective=stretch_objective,
            soft_temperature=stretch_soft_temperature,
            spread_weight=stretch_spread_weight,
        )
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        final_iteration = iteration

        if (iteration % max(int(record_every), 1)) == 0:
            loss_list.append(float(loss.detach().item()))

        if (iteration % eval_interval) != 0 and iteration != int(total_iteration):
            continue

        model.eval()
        with torch.no_grad():
            eval_objective, eval_spread, _ = _evaluate_local_stretch_objective(
                _forward,
                eval_z,
                num_directions=stretch_num_directions,
                fd_eps=stretch_fd_eps,
                domain_shape=latent_domain_shape,
                boundary_eps=stretch_boundary_eps,
                log_eps=stretch_log_eps,
                aggregate=stretch_aggregate,
                objective=stretch_objective,
                soft_temperature=stretch_soft_temperature,
                spread_weight=stretch_spread_weight,
            )
        eval_value = float(eval_objective.item())
        eval_spread_value = float(eval_spread.item())
        improved = eval_value < (best_eval_value - float(early_stop_min_delta))
        if improved:
            best_eval_value = eval_value
            best_eval_spread_value = eval_spread_value
            best_iteration = iteration
            best_state = deepcopy(model.state_dict())
            bad_eval_count = 0
        else:
            bad_eval_count += 1
        eval_history.append(
            {
                "iteration": iteration,
                "objective": eval_value,
                "spread_loss": eval_spread_value,
                "is_best": bool(improved),
            }
        )
        model.train()

        patience = None if early_stop_patience is None else int(early_stop_patience)
        if (
            patience is not None
            and iteration >= int(early_stop_min_iterations)
            and bad_eval_count >= patience
        ):
            stop_reason = "early_stopping"
            break

    model.load_state_dict(best_state)
    model.eval()

    training_record = {
        "loss_list": loss_list,
        "total_iteration": int(total_iteration),
        "final_iteration": int(final_iteration),
        "record_every": max(int(record_every), 1),
        "eval_every": int(eval_interval),
        "eval_batch_size": int(eval_z.shape[0]),
        "latent_domain_shape": str(latent_domain_shape),
        "sphere_reparam_init_scale": float(sphere_reparam_init_scale),
        "stretch_aggregate": str(stretch_aggregate),
        "stretch_objective": str(stretch_objective),
        "stretch_soft_temperature": float(stretch_soft_temperature),
        "stretch_spread_weight": float(stretch_spread_weight),
        "identity_init_inner": bool(identity_init_inner),
        "wrapper_type": str(wrapper_type),
        "track_logdet_stats": bool(track_logdet_stats),
        "boundary_focus_ratio": float(boundary_focus_ratio),
        "boundary_shell_radius": float(boundary_shell_radius),
        "grad_clip_norm": None if grad_clip_norm is None else float(grad_clip_norm),
        "lr_decay_step": None if lr_decay_step is None else int(lr_decay_step),
        "lr_decay_gamma": float(lr_decay_gamma),
        "best_eval_objective": float(best_eval_value),
        "best_eval_spread_loss": float(best_eval_spread_value),
        "best_iteration": int(best_iteration),
        "stop_reason": str(stop_reason),
        "early_stop_patience": None if early_stop_patience is None else int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stop_min_iterations": int(early_stop_min_iterations),
        "eval_history": eval_history,
    }
    return model, training_record


def unsupervised_training_mdh(MDH, data, x_tensor, input_tensor, save_dir, args):
    batch_size = args['batch_size']
    total_iteration = args['total_iteration']
    w_penalty = args['w_penalty']
    w_distortion = args['w_distortion']
    w_lipschitz = float(args.get('w_lipschitz', args.get('w_lipschitz_reg', 0.0)))
    w_local_stretch = float(args.get('w_local_stretch', 0.0))
    stretch_num_directions = int(args.get('stretch_num_directions', 8))
    stretch_fd_eps = float(args.get('stretch_fd_eps', 1e-3))
    stretch_log_eps = float(args.get('stretch_log_eps', 1e-8))
    stretch_boundary_eps = float(args.get('stretch_boundary_eps', 1e-6))
    stretch_aggregate = args.get('stretch_aggregate', 'max')
    latent_domain_shape = args.get('latent_domain_shape', 'ball')
    results_save_freq = max(int(args.get('resultsSaveFreq', total_iteration + 1)), 1)
    # GPU training can stall if we call .item() every step; default to periodic logging.
    record_every = max(int(args.get('record_every', 10)), 1)
    grad_clip_norm = args.get('grad_clip_norm', None)
    runtime_device = resolve_torch_device(args.get('device'), default=str(x_tensor.device))
    runtime_dtype = resolve_torch_dtype(args.get('dtype'), default=x_tensor.dtype)
    use_amp = bool(args.get('use_amp', False)) and torch.cuda.is_available() and str(x_tensor.device).startswith("cuda")
    amp_dtype_name = str(args.get('amp_dtype', 'float16')).strip().lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {'bf16', 'bfloat16'} else torch.float16
    use_new_amp_api = hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and hasattr(torch.amp, "autocast")
    if use_amp:
        if use_new_amp_api:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    volume_list, penalty_list, dist_list, lipschitz_list, stretch_list = [], [], [], [], []
    x_tensor = x_tensor.contiguous()
    if input_tensor is not None:
        # Keep condition pool on its original device (GPU or CPU). For CPU pools and CUDA
        # training, pin memory once so each mini-batch transfer can use non_blocking copy.
        input_tensor = input_tensor.contiguous()
        if input_tensor.device.type == "cpu" and x_tensor.device.type == "cuda" and not input_tensor.is_pinned():
            input_tensor = input_tensor.pin_memory()
    n_dim = x_tensor.shape[1]
    bias_tensor = torch.full((batch_size, n_dim), float(args['center']), device=x_tensor.device, dtype=x_tensor.dtype)
    optimizer_kwargs = {"lr": args['lr'], "weight_decay": 1e-5}
    if x_tensor.is_cuda:
        optimizer_kwargs["foreach"] = True
    optimizer = optim.AdamW(MDH.parameters(), **optimizer_kwargs)
    ema_decay = args.get('ema_decay', 0.99)
    ema = EMA(MDH, decay=ema_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['lr_decay_step'], gamma=args['lr_decay'])
    MDH.train()

    for n in tqdm(range(total_iteration)):
        optimizer.zero_grad(set_to_none=True)
        batch_index = torch.randint(0, x_tensor.shape[0], (batch_size,), device=x_tensor.device)
        x_input = x_tensor[batch_index]
        if input_tensor is not None:
            batch_index = torch.randint(0, input_tensor.shape[0], (batch_size,), device=input_tensor.device)
            input_batch = input_tensor[batch_index]
        else:
            input_batch, _ = data.generate_problem_samples_torch(
                n_samples=batch_size,
                sample_obj=False,
                device=runtime_device,
                dtype=runtime_dtype,
            )
        if input_batch.device != x_input.device or input_batch.dtype != x_input.dtype:
            input_batch = input_batch.to(
                device=x_input.device,
                dtype=x_input.dtype,
                non_blocking=(input_batch.device.type == "cpu" and x_input.device.type == "cuda"),
            )
        if use_amp:
            if use_new_amp_api:
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True, dtype=amp_dtype)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            xt, logdet, logdis = MDH(x_input, input_batch)
            xt_scale = data.scale(input_batch, xt)
            xt_full = data.complete_partial(input_batch, xt_scale)
            ineq_vio = torch.clamp(data.ineq_resid(input_batch, xt_full), min=0)
            penalty = ineq_vio.sum(-1)
            logdet_mean = logdet.mean()
            penalty_mean = penalty.mean()
            logdis_mean = logdis.mean()
            logdis_max = torch.amax(logdis)
            if w_local_stretch > 0:
                repeated_input = input_batch.unsqueeze(1).expand(-1, stretch_num_directions, *input_batch.shape[1:])

                def _end_to_end_forward(flat_z):
                    batch_repeat = repeated_input.reshape(flat_z.shape[0], *input_batch.shape[1:])
                    output = MDH(flat_z, batch_repeat)
                    output = _extract_forward_tensor(output)
                    return data.complete_partial(batch_repeat, data.scale(batch_repeat, output))

                local_stretch_loss, _ = local_log_stretch_loss(
                    _end_to_end_forward,
                    x_input,
                    num_directions=stretch_num_directions,
                    fd_eps=stretch_fd_eps,
                    domain_shape=latent_domain_shape,
                    boundary_eps=stretch_boundary_eps,
                    log_eps=stretch_log_eps,
                    aggregate=stretch_aggregate,
                )
            else:
                local_stretch_loss = torch.zeros((), device=x_input.device, dtype=x_input.dtype)
            loss = (
                w_distortion * logdis_mean
                + w_lipschitz * logdis_max
                + w_penalty * penalty_mean
                + w_local_stretch * local_stretch_loss
                - logdet_mean / n_dim
            )
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if grad_clip_norm is not None:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(MDH.parameters(), float(grad_clip_norm))
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        ema.update()

        if (n % record_every) == 0:
            volume_list.append(logdet_mean.detach().item() / n_dim)
            penalty_list.append(penalty_mean.detach().item())
            dist_list.append(logdis_mean.detach().item() / n_dim)
            lipschitz_list.append(logdis_max.detach().item() / n_dim)
            stretch_list.append(local_stretch_loss.detach().item())

        if n > 0 and n % results_save_freq == 0:
            save_inn_mapping(MDH, save_dir, filename=INN_MODEL_CHECKPOINT_FILENAME)
            MDH.eval()
            with torch.inference_mode():
                x0 = MDH(bias_tensor, input_batch)
                x0_scale = data.scale(input_batch, x0)
                x0_full = data.complete_partial(input_batch, x0_scale)
                violation_0 = data.check_feasibility(input_batch, x0_full)
                penalty_0 = torch.abs(violation_0).detach().cpu()
            print(
                'Iteration: {}/{}, Volume: {:.3f}, Penalty: {:.3f}, Distortion: {:.3f}, Lipschitz: {:.3f}, LocalStretch: {:.3f}, Center Penalty: {:.4f}, Valid rate: {:.2f}'.format(
                    n,
                    total_iteration,
                    logdet_mean.detach().item() / n_dim,
                    penalty_mean.detach().item(),
                    logdis_mean.detach().item() / n_dim,
                    logdis_max.detach().item() / n_dim,
                    local_stretch_loss.detach().item(),
                    penalty_0.sum(-1).max().numpy(),
                    (penalty_0.max(-1)[0] < 1e-5).numpy().mean(),
                )
            )
            MDH.train()
    training_record = {
        'volume_list': volume_list,
        'penalty_list': penalty_list,
        'dist_list': dist_list,
        'lipschitz_list': lipschitz_list,
        'stretch_list': stretch_list,
        'trans_list': [],
        'w_lipschitz': float(w_lipschitz),
    }
    ema.apply_shadow()
    return MDH, training_record


def train_mdh_mapping(data, args, save_dir):
    paras = args["model_config"]
    runtime_device = resolve_torch_device(paras.get('device'), default=str(DEVICE))
    runtime_dtype = resolve_torch_dtype(paras.get('dtype'), default=torch.float32)
    con_type = paras['Con_type']
    if con_type == 'MLP':
        # MLP consumes flattened conditioning vector.
        c_dim = int(data.n_qua * data.npara)
    else:
        c_dim = data.npara
    n_dim = data.nvar
    num_layer = paras['num_layer']
    inv_type = paras['inv_type']
    h_dim = max(n_dim + 1, paras['h_dim'])
    model = INN(
        n_dim,
        h_dim,
        c_dim,
        num_layer,
        inv=inv_type,
        outact=None,
        bilip=paras.get('bilip', False),
        lip=paras.get('L', 2),
        Con_type=con_type,
        cond_embed_mode=paras.get('cond_embed_mode', 'shared'),
    ).to(device=runtime_device, dtype=runtime_dtype)
    latent_init_shape = str(paras.get('latent_init_shape', 'auto')).strip().lower()
    if latent_init_shape not in {'auto', 'sphere', 'cube'}:
        raise ValueError(
            f"Unsupported latent_init_shape: {latent_init_shape}. Use 'auto', 'sphere', or 'cube'."
        )
    use_sphere = (n_dim == 2) if latent_init_shape == 'auto' else (latent_init_shape == 'sphere')
    if use_sphere:
        ball_sample = sampling_sphere(paras['n_samples'], n_dim)
        ball_sample = torch.tensor(ball_sample, dtype=runtime_dtype, device=runtime_device)
    else:
        ball_sample = (torch.rand(size=[paras['n_samples'], n_dim], device=runtime_device, dtype=runtime_dtype) - 0.5) * 2
    # Always sample conditions on-the-fly per iteration to avoid caching large pools.
    input_samples = None
    training_args = dict(paras)
    training_args.setdefault('device', str(runtime_device))
    training_args.setdefault('dtype', runtime_dtype)
    training_args.setdefault('latent_domain_shape', 'ball' if use_sphere else 'cube')
    model, training_record = unsupervised_training_mdh(model, data, ball_sample, input_samples, save_dir, training_args)
    return model, training_record


def homeo_bisection(model, data, z_infeasible, c_params, args, eps_converge=1e-4, condition_emb=None):
    model.eval()
    batch_size = z_infeasible.shape[0]
    device = z_infeasible.device
    max_steps = args.get('proj_max_steps', 20)
    feasibility_eps = args.get('proj_eps', 1e-5)
    bis_step = float(args.get('step_size', 0.5))
    bis_step = min(max(bis_step, 1e-6), 1.0 - 1e-6)
    alpha_upper = torch.ones(batch_size, 1, device=device, dtype=z_infeasible.dtype)
    alpha_lower = torch.zeros(batch_size, 1, device=device, dtype=z_infeasible.dtype)
    c_emb = condition_emb if condition_emb is not None else _embed_condition_if_available(model, c_params, batch_size)
    with torch.inference_mode():
        for k in range(max_steps):
            alpha = (1 - bis_step) * alpha_lower + bis_step * alpha_upper
            z_candidate = alpha * z_infeasible
            x_mapped = _forward_with_condition(model, z_candidate, c_params, c_emb)
            x_scaled = data.scale(c_params, x_mapped)
            x_full = data.complete_partial(c_params, x_scaled)
            violations = data.check_feasibility(c_params, x_full)
            max_violations = torch.amax(violations, dim=1, keepdim=True)
            feasible_mask = max_violations < feasibility_eps
            alpha_lower = torch.where(feasible_mask, alpha, alpha_lower)
            alpha_upper = torch.where(feasible_mask, alpha_upper, alpha)
            if (alpha_upper - alpha_lower).max() < eps_converge:
                break
        alpha = (alpha_lower + alpha_upper) / 2
        z_feasible = alpha * z_infeasible
    return z_feasible, k + 1


def pgd_transformed_space(model, data, input_params, args, initial_z=None):
    model.eval()
    batch_size = input_params.shape[0]
    device = input_params.device
    dtype = input_params.dtype
    max_iter = args.get('pgd_max_iter', 200)
    lr = args.get('pgd_lr', 0.001)
    z = (
        torch.zeros(batch_size, data.nvar, device=device, dtype=dtype)
        if initial_z is None
        else initial_z.clone().detach().to(device=device, dtype=dtype)
    )
    z.requires_grad_(True)
    optimizer = torch.optim.SGD([z], lr=lr, momentum=0.0)
    lr_decay = 0.95
    min_lr = 1e-5
    current_lr = lr
    proj_eps = float(args.get('proj_eps', 1e-5))
    best_objective = float('inf')
    patience = 0
    lr_patience = 1
    trace_every = max(int(args.get('trace_every', 1)), 1)
    objective_history, violation_history, trajectory_data = [], [], []
    fixed_Q, fixed_p = _build_quadratic_terms(data, device, dtype)
    condition_emb = _embed_condition_if_available(model, input_params, batch_size, detach=True)
    for iteration in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        x_mapped = _forward_with_condition(model, z, input_params, condition_emb)
        x_scaled = data.scale(input_params, x_mapped)
        x_full = data.complete_partial(input_params, x_scaled)
        objective = _quadratic_objective(x_full, fixed_Q, fixed_p)
        with torch.inference_mode():
            violations = data.check_feasibility(input_params, x_full)
            max_violations = torch.amax(violations, dim=1)
        if (iteration % trace_every) == 0:
            trajectory_data.append({
                'iteration': iteration,
                'z': z.detach().cpu().numpy().copy(),
                'x': x_full.detach().cpu().numpy().copy(),
                'objective': objective.detach().cpu().numpy().copy(),
                'violation': max_violations.detach().cpu().numpy().copy(),
            })
        loss = objective.sum()
        grad_z = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
        z.grad = grad_z
        optimizer.step()
        with torch.inference_mode():
            x_update = _forward_with_condition(model, z, input_params, condition_emb)
            x_update = data.complete_partial(input_params, data.scale(input_params, x_update))
            violations_update = data.check_feasibility(input_params, x_update)
            max_violations_update = torch.amax(violations_update, dim=1)
            if torch.any(max_violations_update > proj_eps):
                z_proj, _ = homeo_bisection(
                    model,
                    data,
                    z.detach(),
                    input_params,
                    args,
                    condition_emb=condition_emb,
                )
                z.copy_(z_proj)
        objective_history.append(objective.detach().cpu().numpy())
        violation_history.append(max_violations.detach().cpu().numpy())
        current_obj = objective.mean().item()
        if current_obj < best_objective:
            best_objective = current_obj
            patience = 0
        else:
            patience += 1
        if patience >= lr_patience and current_lr > min_lr:
            current_lr = max(current_lr * lr_decay, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            patience = 0
    with torch.inference_mode():
        x_final = _forward_with_condition(model, z, input_params, condition_emb)
        x_scaled = data.scale(input_params, x_final)
        x_full = data.complete_partial(input_params, x_scaled)
        final_violations = data.check_feasibility(input_params, x_full)
        final_max_violations = torch.amax(final_violations, dim=1)
        if hasattr(data, 'fixed_Q') and hasattr(data, 'fixed_p'):
            final_objective = _quadratic_objective(x_full, fixed_Q, fixed_p)
        else:
            final_objective = torch.sum(x_full ** 2, dim=1)
    return {
        'z_final': z.detach(),
        'x_final': x_full,
        'final_objective': final_objective,
        'final_violations': final_max_violations,
        'objective_history': objective_history,
        'violation_history': violation_history,
        'iterations': max_iter,
        'feasible_solutions': (final_max_violations < 1e-5).sum().item(),
        'trajectory': trajectory_data,
    }


class INNPGDOptimizer:
    def __init__(self, problem, paras, model=None):
        self.problem = problem
        self.model = model
        self._init_optimization_params(paras)
        model_param = next(self.model.parameters(), None) if self.model is not None else None
        self.device = self.problem.device if hasattr(self.problem, 'device') else (
            model_param.device if model_param is not None else torch.device('cpu')
        )
        self.dtype = getattr(self.problem, 'dtype', None)
        if self.dtype is None:
            self.dtype = model_param.dtype if model_param is not None else torch.get_default_dtype()

    def _init_optimization_params(self, paras):
        self.learning_rate = paras.get('learning_rate', 0.001)
        self.max_iterations = paras.get('max_iterations', 200)
        self.max_running_time = paras.get('max_running_time', 300)
        self.convergence_threshold = paras.get('convergence_threshold', 1e-6)
        self.lr_decay = paras.get('lr_decay', 0.9)
        self.eps_converge = paras.get('feasibility_eps', 1e-5)
        self.proj_max_steps = paras.get('proj_max_steps', 20)
        self.proj_step_size = paras.get('step_size', 0.5)
        self.proj_eps = paras.get('proj_eps', 1e-5)
        self.opt_type = paras.get('opt', 'gd')
        self.momentum = paras.get('momentum', 0.0)
        self.stepsize_rule = _validate_stepsize_rule(paras.get('stepsize_rule', 'constant'))

    def optimize(self, initial_point=None, verbose=False, seed=2025, input_params=None, objective_params=None):
        del seed
        if self.model is None:
            raise ValueError('INN model must be provided for INNPGDOptimizer')
        if input_params is None:
            raise ValueError('input_params must be provided for INN optimization')
        self.model.eval()
        input_params = input_params.to(device=self.device, dtype=self.dtype)
        if objective_params is not None:
            objective_params = objective_params.to(device=self.device, dtype=self.dtype)
        if self.opt_type == 'gd':
            opt = GDOptimizer(beta1=self.momentum)
        else:
            opt = AdamOptimizer()
        learning_rate = self.learning_rate
        batch_size = input_params.shape[0]
        should_track_traj = self.problem.nvar == 2
        model_forward = self.model
        scale = self.problem.scale
        complete_partial = self.problem.complete_partial
        violations_fn = self.problem.violations
        condition_emb = _embed_condition_if_available(model_forward, input_params, batch_size, detach=True)

        def _forward_to_full(z_tensor):
            x_tensor = _forward_with_condition(model_forward, z_tensor, input_params, condition_emb)
            x_scaled_tensor = scale(input_params, x_tensor)
            return complete_partial(input_params, x_scaled_tensor)

        def _objective_value(x_full):
            if objective_params is not None and hasattr(self.problem, "objective_xy"):
                return self.problem.objective_xy(input_params, x_full, objective_params).reshape(-1)
            return self.problem.objective(x_full).reshape(-1)

        if initial_point is None:
            z = torch.zeros(batch_size, self.problem.nvar, device=self.device, dtype=self.dtype)
        else:
            z = torch.as_tensor(initial_point, dtype=self.dtype, device=self.device).view(batch_size, -1)
        z.requires_grad_(True)
        with torch.inference_mode():
            x_full_current = _forward_to_full(z)
            initial_eval = {
                "objective": _objective_value(x_full_current),
                "violation": violations_fn(input_params, x_full_current).reshape(-1),
                "decision": x_full_current,
                "latent_trajectory": z,
            }

        def _compute_gradient(z_state, prev_state, iteration):
            del prev_state, iteration
            z_state = z_state.detach().requires_grad_(True)
            x_mapped = _forward_with_condition(model_forward, z_state, input_params, condition_emb)
            x_scaled = scale(input_params, x_mapped)
            x_full = complete_partial(input_params, x_scaled)
            objective = _objective_value(x_full)
            grad = torch.autograd.grad(objective.sum(), z_state)[0]
            return grad

        def _apply_step(z_state, update, lr, prev_state, iteration):
            del prev_state, iteration
            z_update = z_state - lr * update
            with torch.inference_mode():
                x_update = _forward_to_full(z_update)
                violation_update = violations_fn(input_params, x_update)
                if torch.any(violation_update > self.proj_eps):
                    z_proj, _ = self._homeo_bisection(z_update, input_params, condition_emb=condition_emb)
                else:
                    z_proj = z_update
            return z_proj.detach().clone()

        def _evaluate_state(z_state):
            with torch.inference_mode():
                x_eval = _forward_to_full(z_state)
                return {
                    "objective": _objective_value(x_eval),
                    "violation": violations_fn(input_params, x_eval).reshape(-1),
                    "decision": x_eval,
                    "latent_trajectory": z_state,
                }

        z, payload, learning_rate = run_first_order_loop(
            initial_state=z,
            initial_eval=initial_eval,
            update_backend=opt,
            compute_gradient=_compute_gradient,
            apply_step=_apply_step,
            evaluate_state=_evaluate_state,
            max_iterations=self.max_iterations,
            max_running_time=self.max_running_time,
            initial_learning_rate=learning_rate,
            stepsize_rule=self.stepsize_rule,
            lr_decay=self.lr_decay,
            convergence_threshold=self.convergence_threshold,
            verbose=verbose,
            progress_disable=not verbose,
            track_decisions=should_track_traj,
            extra_track_names=("latent_trajectory",),
        )
        with torch.inference_mode():
            x_opt = _forward_to_full(z)
        return (
            x_opt,
            payload["decision_trajectory"],
            payload["latent_trajectory"],
            payload["objective_traj"],
            payload["violation_traj"],
            payload["per_iter_time"],
        )

    def _homeo_bisection(self, z_infeasible, input_params, condition_emb=None):
        args = {
            'proj_max_steps': self.proj_max_steps,
            'proj_eps': self.proj_eps,
            'step_size': self.proj_step_size,
        }
        return homeo_bisection(
            self.model,
            self.problem,
            z_infeasible,
            input_params,
            args,
            self.eps_converge,
            condition_emb=condition_emb,
        )
