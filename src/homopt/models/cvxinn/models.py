"""CvxINN model wrappers for exact convex-set parameterizations."""

import numpy as np
import torch
import torch.nn as nn

from homopt.mappings.gauge import GaugeMap
from homopt.utils import cast_tensors_to_dtype

from .geometry import cube_compactify, cube_decompactify, radial_compactify, radial_decompactify


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

__all__ = [
    "CubeGaugeCvxINN",
    "CvxINN",
    "SphereReparam",
    "SphereReparamGaugeCvxINN",
    "TanhGaugeCvxINN",
]
