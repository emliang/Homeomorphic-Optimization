import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .activation import Swish
import torch.nn.init as init


class GraphLipNet(nn.Module):
    def __init__(self, adj, num_inputs, num_hidden, num_cond_inputs=None, lip=0.99):
        super(GraphLipNet, self).__init__()
        num_node = adj.shape[1]
        self.num_node = num_node
        self.adj = adj + torch.eye(self.num_node, device=adj.device)
        self.deg = self.adj.sum(1).view(1, self.num_node, 1)
        self.lip = lip
        if num_cond_inputs is not None:
            self.w = nn.Sequential(nn.Linear(num_cond_inputs, num_hidden), nn.Tanh())
            self.b = nn.Sequential(nn.Linear(num_cond_inputs, num_hidden), nn.ReLU())
        self.emb = nn.Sequential(LinearNormalized(num_inputs, num_hidden), Swish())
        self.cat = nn.Sequential(LinearNormalized(num_hidden, num_hidden), Swish(), LinearNormalized(num_hidden, num_inputs))

    def forward(self, inputs, cond_inputs=None):
        ndim = inputs.shape[-1]
        emb = inputs.view(-1, self.num_node, ndim)
        emb = torch.matmul(emb.permute(0, 2, 1), self.adj)
        emb = emb.permute(0, 2, 1).contiguous() / self.deg
        emb = emb.view(-1, ndim)
        emb = self.emb(emb)
        if cond_inputs is not None:
            w = self.w(cond_inputs)
            b = self.b(cond_inputs)
            emb = w * emb + b
        gx = self.cat(emb)
        return gx * self.lip


class LipNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_cond_inputs=None, lip=1.0, activation='gelu'):
        super(LipNet, self).__init__()
        self.lip = lip
        act_name = str(activation).strip().lower()
        if act_name == 'gelu':
            act = nn.GELU
        elif act_name in {'silu', 'swish'}:
            act = nn.SiLU
        elif act_name == 'relu':
            act = nn.ReLU
        else:
            raise ValueError(f'Unsupported LipNet activation: {activation}')
        if num_cond_inputs is not None:
            self.b = nn.Sequential(nn.Linear(num_cond_inputs, num_hidden))
        self.emb = nn.Sequential(nn.Linear(num_inputs, num_hidden))
        self.cat = nn.Sequential(
            SpectralNormLinear(num_hidden, num_hidden), act(),
            SpectralNormLinear(num_hidden, num_hidden), act(),
            SpectralNormLinear(num_hidden, num_inputs),
        )

    def forward(self, inputs, cond_inputs=None):
        emb = self.emb(inputs)
        if cond_inputs is not None:
            emb = emb + self.b(cond_inputs)
        gx = self.cat(emb)
        return gx * self.lip


class SpectralNormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, coeff=0.99, n_iterations=1, atol=None, rtol=None, **unused_kwargs):
        del unused_kwargs
        super(SpectralNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.0))
        self.register_buffer('u', F.normalize(self.weight.new_empty(h).normal_(0, 1), dim=0))
        self.register_buffer('v', F.normalize(self.weight.new_empty(w).normal_(0, 1), dim=0))
        self.compute_weight(True, 1)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol
        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')
        if n_iterations is None:
            n_iterations = 20000
        u = self.u
        v = self.v
        weight = self.weight
        if update:
            with torch.no_grad():
                for _ in range(n_iterations):
                    old_v = v.clone()
                    old_u = u.clone()
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(weight, v), dim=0, out=u)
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement() ** 0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement() ** 0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                u = u.clone()
                v = v.clone()
            sigma = torch.dot(u, torch.mv(weight, v))
            with torch.no_grad():
                self.scale.copy_(sigma)
            factor = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
        else:
            factor = torch.max(torch.ones(1, device=weight.device), self.scale / self.coeff)
        return weight / factor

    def forward(self, input):
        weight = self.compute_weight(update=self.training)
        return F.linear(input, weight, self.bias)


class LinearNormalized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearNormalized, self).__init__(in_features, out_features, bias)
        self.linear = spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)
