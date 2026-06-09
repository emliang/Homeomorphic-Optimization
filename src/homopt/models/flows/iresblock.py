import logging
import numpy as np
import torch
import torch.nn as nn
from .lipschitz import LipNet

logger = logging.getLogger()

__all__ = ['iResidualLayer', 'iGraphResidualLayer']


class iResidualLayer(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_hidden,
        num_cond_inputs=None,
        residual_scale=0.05,
        lip_activation='gelu',
        geom_p=0.5,
        lamb=2.0,
        n_power_series=None,
        exact_trace=False,
        brute_force=False,
        n_samples=1,
        n_exact_terms=2,
        n_dist='geometric',
        neumann_grad=True,
        grad_in_forward=True,
    ):
        super(iResidualLayer, self).__init__()
        self.lipnet = LipNet(num_inputs, num_hidden, num_cond_inputs, activation=lip_activation)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.n_dist = n_dist
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1.0 - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_exact_terms = n_exact_terms
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad

    def _residual_update(self, inputs, cond_inputs=None):
        return self.residual_scale * self.lipnet(inputs, cond_inputs)

    def forward(self, inputs, cond_inputs=None, mode='direct', compute_logdet=True):
        if mode == 'direct':
            if self.training:
                if compute_logdet:
                    gx, logdet = self.logdet_estimator(inputs, cond_inputs)
                    logdet = logdet.view(inputs.shape[0], -1)
                else:
                    gx = self._residual_update(inputs, cond_inputs)
                    logdet = None
            else:
                gx = self._residual_update(inputs, cond_inputs)
                logdet = None
            return inputs + gx, logdet
        return self.inverse_fixed_point(inputs, cond_inputs), None

    def inverse_fixed_point(self, y, cond_inputs=None, atol=1e-5, rtol=1e-5):
        with torch.no_grad():
            x, x_prev = y - self._residual_update(y, cond_inputs), y
            i = 0
            tol = atol + y.abs() * rtol
            while not torch.all((x - x_prev) ** 2 / tol < 1):
                x, x_prev = y - self._residual_update(x, cond_inputs), x
                i += 1
                if i > 1000:
                    logger.info('Iterations exceeded 1000 for inverse.')
                    break
        return x

    def logdet_estimator(self, x, c):
        with torch.enable_grad():
            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            else:
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)
            if self.training:
                if self.n_power_series is None:
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.0
            else:
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = lambda k: 1 / rcdf_fn(k, 20) * sum(n_samples >= k - 20) / len(n_samples)
            vareps = torch.randn_like(x)
            estimator_fn = neumann_logdet_estimator if self.training and self.neumann_grad else basic_logdet_estimator
            if self.training and self.grad_in_forward:
                g, logdetgrad = mem_eff_wrapper(
                    estimator_fn,
                    self._residual_update,
                    x,
                    c,
                    n_power_series,
                    vareps,
                    coeff_fn,
                    self.training,
                )
            else:
                x = x.requires_grad_(True)
                g = self._residual_update(x, c)
                logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, self.training)
            return g, logdetgrad.view(-1, 1)


class iGraphResidualLayer(iResidualLayer):
    pass


def batch_jacobian(g, x):
    jac = []
    for d in range(g.shape[1]):
        jac.append(torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=True)[0].view(x.shape[0], 1, x.shape[1]))
    return torch.cat(jac, 1)


def batch_trace(M):
    return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)


def geometric_sample(p, n_samples):
    return np.random.geometric(p, size=n_samples)


def geometric_1mcdf(p, k, offset):
    return (1 - p) ** max(k - offset, 0)


def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, size=n_samples)


def poisson_1mcdf(lamb, k, offset):
    start = max(k - offset, 0)
    probs = [np.exp(-lamb) * lamb ** i / np.math.factorial(i) for i in range(start)]
    return max(1.0 - sum(probs), 1e-12)


def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.zeros(x.shape[0], device=x.device)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, retain_graph=True, create_graph=training)[0]
        logdetgrad = logdetgrad + ((-1) ** (k + 1) / k) * coeff_fn(k) * torch.sum(vjp * vareps, dim=1)
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, retain_graph=True, create_graph=training)[0]
        neumann_vjp = neumann_vjp + ((-1) ** k) * coeff_fn(k) * vjp
    return torch.sum(neumann_vjp * vareps, dim=1)


def mem_eff_wrapper(estimator_fn, gnet, x, c, n_power_series, vareps, coeff_fn, training):
    x = x.requires_grad_(True)
    g = gnet(x, c)
    logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)
    return g, logdetgrad
