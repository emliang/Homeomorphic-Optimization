"""Spectral helpers used by MaxCut gauge maps."""

from __future__ import annotations

import torch


def max_eigenvalue_with_gradient(A):
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    max_eigenvalue = eigenvalues[-1]
    max_eigenvector = eigenvectors[:, -1]

    class MaxEigenvalueFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_matrix, eigenvalue, eigenvector):
            ctx.save_for_backward(eigenvector)
            return eigenvalue

        @staticmethod
        def backward(ctx, grad_output):
            (eigenvector,) = ctx.saved_tensors
            grad_matrix = grad_output * torch.outer(eigenvector, eigenvector)
            return grad_matrix, None, None

    return MaxEigenvalueFunction.apply(A, max_eigenvalue, max_eigenvector)

