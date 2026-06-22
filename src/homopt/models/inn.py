"""Structured shared INN/model/training implementation."""

import pickle
from pathlib import Path

import torch
import torch.nn as nn
from homopt.models.condition import MLP, Mixer, PINN, QuadMixer, ResBlock, build_condition_encoder, normalize_condition_encoder_type
from homopt.models.flows.coupling import CombinedActNormLU, CouplingLayer, MADE, MaskedLinear
from homopt.models.flows.iresblock import iResidualLayer
from homopt.models.flows.lipschitz import SpectralNormLinear
from homopt.utils import resolve_torch_device, resolve_torch_dtype

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
        self.con_type = normalize_condition_encoder_type(Con_type)
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
            return build_condition_encoder(
                self.con_type,
                n_var=nin,
                input_dim=cin,
                hidden_dim=nhid,
                output_dim=out_dim,
                num_layer=2,
            )

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
        if self.con_type == 'mlp':
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


def _quadratic_objective(y_full, fixed_Q, fixed_p):
    quad_term = 0.5 * torch.sum((y_full @ fixed_Q) * y_full, dim=1)
    linear_term = torch.sum(fixed_p * y_full, dim=1)
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
