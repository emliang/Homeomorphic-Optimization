"""Custom autograd functions for explicit gauge maps."""

from __future__ import annotations

import torch

from .gradients import explicit_vjp_from_state


class GaugeForwardExplicitFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z,
        x,
        u,
        r,
        beta,
        scaling,
        best_grad_u,
        eps,
    ):
        ctx.save_for_backward(z, u, r, beta, scaling, best_grad_u)
        ctx.eps = float(eps)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        z, u, r, beta, scaling, best_grad_u = ctx.saved_tensors
        grad_z = explicit_vjp_from_state(
            z=z,
            u=u,
            r=r,
            beta=beta,
            scaling=scaling,
            best_grad_u=best_grad_u,
            grad_x=grad_output,
            eps=ctx.eps,
        )
        return (
            grad_z,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

