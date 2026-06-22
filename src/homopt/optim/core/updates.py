"""Shared first-order update backends."""

from __future__ import annotations

import torch


class GDOptimizer:
    """Momentum gradient update backend."""

    def __init__(self, beta1=0.0):
        self.beta1 = beta1
        self.m = 0

    def reset(self):
        self.m = 0

    def step(self, grad):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        return self.m


class NormalizedGDOptimizer:
    """Gradient update backend that keeps only the per-sample gradient direction."""

    def __init__(self, epsilon=1e-12):
        self.epsilon = float(epsilon)

    def reset(self):
        pass

    def step(self, grad):
        norm = torch.linalg.vector_norm(grad, ord=2, dim=-1, keepdim=True)
        return grad / torch.clamp(norm, min=self.epsilon)


class AdamOptimizer:
    """Adam-style normalized update backend."""

    def __init__(self, beta1=0.9, beta2=0.99, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def reset(self):
        self.m = 0
        self.v = 0
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return m_hat / (torch.sqrt(v_hat) + self.epsilon)


__all__ = ["AdamOptimizer", "GDOptimizer", "NormalizedGDOptimizer"]
