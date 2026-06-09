import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _lu_factor(matrix):
    if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'lu'):
        return torch.linalg.lu(matrix)
    return torch.lu_unpack(*torch.lu(matrix))


def _solve_triangular(matrix, rhs, *, upper):
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve_triangular"):
        return torch.linalg.solve_triangular(matrix, rhs, upper=upper)
    # torch.triangular_solve returns (solution, cloned_coefficient)
    return torch.triangular_solve(rhs, matrix, upper=upper)[0]


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)
    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)
    return out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, cond_in_features=None, bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if cond_in_features is not None:
            self.cond_linear = nn.Sequential(nn.Linear(cond_in_features, 2 * out_features))
        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            w, b = self.cond_linear(cond_inputs).chunk(2, 1)
            output = output * w + b
        return output

class MADE(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_cond_inputs=None, act='relu', bilip=False, lip=1.5):
        super(MADE, self).__init__()
        gelu = getattr(nn, 'GELU', nn.ReLU)
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'gelu': gelu}
        act_func = activations[act]
        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')
        self.joiner = MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.trunk = nn.Sequential(
            act_func(), MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(), MaskedLinear(num_hidden, num_inputs * 2, output_mask),
        )
        self.LogL = np.log(lip)
        self.bilip = bilip

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            if self.bilip:
                a = torch.tanh(a) * self.LogL
            u = (inputs - m) * torch.exp(-a)
            return u, -a
        x = torch.zeros_like(inputs)
        for i_col in range(inputs.shape[1]):
            h = self.joiner(x, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            if self.bilip:
                a = torch.tanh(a) * self.LogL
            x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
        return x, -a


class CouplingLayer(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_cond_inputs=None, act='relu', bilip=False, lip=1.5):
        super(CouplingLayer, self).__init__()
        gelu = getattr(nn, 'GELU', nn.ReLU)
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'gelu': gelu}
        act_func = activations[act]
        mask = torch.zeros(size=[1, num_inputs])
        mask[:, :num_inputs // 2] = 1
        self.register_buffer('mask', mask)
        total_inputs = num_inputs + num_cond_inputs if num_cond_inputs is not None else num_inputs
        self.net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), act_func(),
            nn.Linear(num_hidden, num_hidden), act_func(),
            nn.Linear(num_hidden, num_inputs * 2),
        )
        self.LogL = np.log(lip)
        self.bilip = bilip

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask if self.mask.device == inputs.device else self.mask.to(inputs.device)
        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)
        h = self.net(masked_inputs)
        log_s, t = h.chunk(2, 1)
        log_s = log_s * (1 - mask)
        t = t * (1 - mask)
        if self.bilip:
            log_s = torch.tanh(log_s) * self.LogL
        if mode == 'direct':
            s = torch.exp(log_s)
            return inputs * s + t, log_s
        s = torch.exp(-log_s)
        return (inputs - t) * s, -log_s


class LUInvertibleMM(nn.Module):
    def __init__(self, num_inputs, bilip=False, lip=1.5):
        super(LUInvertibleMM, self).__init__()
        W = torch.empty(num_inputs, num_inputs)
        nn.init.orthogonal_(W)
        self.register_buffer('L_mask', torch.tril(torch.ones(W.size()), -1))
        self.register_buffer('U_mask', torch.tril(torch.ones(W.size()), -1).t().clone())
        self.register_buffer('I', torch.eye(W.size(0)))
        P, L, U = _lu_factor(W)
        LU = L * self.L_mask + U * self.U_mask
        self.register_buffer('P', P)
        self.LU = nn.Parameter(LU)
        S = torch.diag(U)
        self.register_buffer('sign_S', torch.sign(S))
        self.log_S = nn.Parameter(torch.log(torch.abs(S)))
        self.bilip = bilip
        self.LogL = np.log(lip)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        log_s = torch.tanh(self.log_S) * self.LogL if self.bilip else self.log_S
        L = self.LU * self.L_mask + self.I
        U = self.LU * self.U_mask + torch.diag(self.sign_S * torch.exp(log_s))
        if mode == 'direct':
            W = self.P @ L @ U
            return inputs @ W, log_s.unsqueeze(0).repeat(inputs.size(0), 1)
        # For row vectors y = x @ (P L U), the inverse is
        # x^T = P @ L^{-T} @ U^{-T} @ y^T.
        rhs = inputs.T
        rhs = _solve_triangular(U.T, rhs, upper=False)
        rhs = _solve_triangular(L.T, rhs, upper=True)
        rhs = self.P @ rhs
        return rhs.T, -log_s.unsqueeze(0).repeat(inputs.size(0), 1)


class ActNorm(nn.Module):
    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized is False:
            with torch.no_grad():
                self.weight.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
                self.bias.copy_(inputs.mean(0))
            self.initialized = True
        if mode == 'direct':
            return (inputs - self.bias) * torch.exp(self.weight), self.weight.unsqueeze(0).repeat(inputs.size(0), 1)
        return inputs * torch.exp(-self.weight) + self.bias, -self.weight.unsqueeze(0).repeat(inputs.size(0), 1)


class Con_ActNorm(nn.Module):
    def __init__(self, num_inputs, num_cond_inputs):
        super(Con_ActNorm, self).__init__()
        n_hid = (num_inputs + num_cond_inputs) // 2
        self.weight = nn.Sequential(nn.Linear(num_cond_inputs, n_hid), nn.ReLU(), nn.Linear(n_hid, num_inputs))
        self.bias = nn.Sequential(nn.Linear(num_cond_inputs, n_hid), nn.ReLU(), nn.Linear(n_hid, num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        weight = self.weight(cond_inputs)
        bias = self.bias(cond_inputs)
        if mode == 'direct':
            return (inputs - bias) * torch.exp(weight), weight
        return inputs * torch.exp(-weight) + bias, -weight


class CombinedActNormLU(nn.Module):
    def __init__(self, num_inputs, bilip=False, lip=1.5):
        super(CombinedActNormLU, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False
        W = torch.empty(num_inputs, num_inputs)
        nn.init.orthogonal_(W)
        self.register_buffer('L_mask', torch.tril(torch.ones(W.size()), -1))
        self.register_buffer('U_mask', torch.tril(torch.ones(W.size()), -1).t().clone())
        self.register_buffer('I', torch.eye(W.size(0)))
        P, L, U = _lu_factor(W)
        self.register_buffer('P', P)
        self.LU = nn.Parameter(L * self.L_mask + U * self.U_mask)
        S = torch.diag(U)
        self.register_buffer('sign_S', torch.sign(S))
        self.log_S = nn.Parameter(torch.log(torch.abs(S)))
        self.bilip = bilip
        self.LogL = np.log(lip)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized is False:
            with torch.no_grad():
                self.weight.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
                self.bias.copy_(inputs.mean(0))
                self.initialized = True
        log_s = torch.tanh(self.log_S) * self.LogL if self.bilip else self.log_S
        L = self.LU * self.L_mask + self.I
        U = self.LU * self.U_mask + torch.diag(self.sign_S * torch.exp(log_s))
        act_log_s = self.weight
        total_log_s = act_log_s + log_s
        if mode == 'direct':
            x = (inputs - self.bias) * torch.exp(self.weight)
            x = x @ self.P @ L @ U
            return x, total_log_s.unsqueeze(0).repeat(inputs.size(0), 1)
        # For row vectors y = x @ (P L U), the inverse is
        # x^T = P @ L^{-T} @ U^{-T} @ y^T.
        rhs = inputs.T
        rhs = _solve_triangular(U.T, rhs, upper=False)
        rhs = _solve_triangular(L.T, rhs, upper=True)
        rhs = self.P @ rhs
        x = rhs.T
        x = x * torch.exp(-self.weight) + self.bias
        return x, -total_log_s.unsqueeze(0).repeat(inputs.size(0), 1)
