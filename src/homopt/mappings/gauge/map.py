"""Gauge mapping public classes."""

from __future__ import annotations

import torch

from homopt.utils import move_tensors_to_device

from .autograd import GaugeForwardExplicitFunction
from .constraints import explicit_candidate_cache
from .constants import (
    FAMILY_BOX_LOWER as _FAMILY_BOX_LOWER,
    FAMILY_BOX_UPPER as _FAMILY_BOX_UPPER,
    FAMILY_GENERAL as _FAMILY_GENERAL,
    FAMILY_LINEAR as _FAMILY_LINEAR,
    FAMILY_QUAD as _FAMILY_QUAD,
    FAMILY_SOC as _FAMILY_SOC,
)
from .gradients import explicit_vjp_from_state
from .general import general_boundary_candidates
from .math import _pairwise_max, _safe_divide
from .state import explicit_forward_state


class GaugeMap:
    def __init__(
        self,
        convex_set,
        p_norm=2,
        x_origin=None,
        smooth=False,
        explicit_gradient_rule="polynomial",
        smooth_tie_tol=1e-7,
        smooth_temperature=1e-4,
        general_bisection_iterations=60,
        general_bisection_expand_steps=40,
        general_bisection_initial_radius=1.0,
        general_bisection_max_radius=1e6,
    ):
        self.constraint_set = convex_set
        self.p_norm = p_norm
        self.smooth = smooth
        if explicit_gradient_rule not in {"implicit", "polynomial", "hybrid"}:
            raise ValueError("explicit_gradient_rule must be 'implicit', 'polynomial', or 'hybrid'.")
        self.explicit_gradient_rule = explicit_gradient_rule
        self.smooth_tie_tol = float(smooth_tie_tol)
        if self.smooth_tie_tol < 0:
            raise ValueError("smooth_tie_tol must be nonnegative.")
        self.smooth_temperature = float(smooth_temperature)
        if self.smooth_temperature <= 0:
            raise ValueError("smooth_temperature must be positive.")
        self.general_bisection_iterations = int(general_bisection_iterations)
        self.general_bisection_expand_steps = int(general_bisection_expand_steps)
        self.general_bisection_initial_radius = float(general_bisection_initial_radius)
        self.general_bisection_max_radius = float(general_bisection_max_radius)
        self._eps = 1e-12
        self._static_cache = {}
        self.last_forward_mode = "uninitialized"
        self.forward_mode_counts = {"autograd": 0, "explicit": 0}
        if x_origin is None:
            self.center = torch.zeros(1, convex_set.nvar)
        else:
            self.center = torch.as_tensor(x_origin, dtype=torch.float32).view(1, -1)

    def to_device(self, device):
        self._static_cache.clear()
        return move_tensors_to_device(self, device)

    def reset_forward_stats(self):
        self.last_forward_mode = "uninitialized"
        self.forward_mode_counts = {"autograd": 0, "explicit": 0}

    def forward(self, z, method="autograd", tie_tol=1e-7, return_state=False):
        if method == "explicit":
            if not self._supports_explicit():
                raise NotImplementedError("GaugeMap explicit forward requires p_norm=2 and supported constraints.")
            terms = self._get_static_terms(z.device, z.dtype)
            raw_r = torch.norm(z, dim=1, p=2, keepdim=True)
            u = z / raw_r.clamp_min(self._eps)
            center_mask = raw_r <= self._eps
            if center_mask.any():
                fallback_u = torch.zeros_like(u)
                fallback_u[:, 0] = 1
                u = torch.where(center_mask, fallback_u, u)
            general_beta, general_grad_u = self._general_boundary_candidates(
                u,
                terms["x0"],
                need_gradient=True,
            )
            smooth_family_codes = terms.get("smooth_family_codes")
            smooth_candidate_index = terms.get("smooth_candidate_index")
            if general_beta is not None:
                smooth_family_codes = None
                smooth_candidate_index = None
            state = explicit_forward_state(
                z,
                center=terms["x0"],
                eps=float(self._eps),
                tie_tol=float(tie_tol),
                smooth_tie_tol=self.smooth_tie_tol if self.smooth else None,
                smooth_temperature=self.smooth_temperature,
                A=terms.get("A"),
                inv_slack=terms.get("inv_slack"),
                inv_lower_denom=terms.get("inv_lower_denom"),
                inv_upper_denom=terms.get("inv_upper_denom"),
                G=terms.get("G"),
                C=terms.get("C"),
                Gx0=terms.get("Gx0"),
                Cx0=terms.get("Cx0"),
                soc_gradB_const=terms.get("soc_gradB_const"),
                soc_Ccoef=terms.get("soc_Ccoef"),
                Qsym=terms.get("Qsym"),
                pq=terms.get("pq"),
                quad_gradB_const=terms.get("quad_gradB_const"),
                quad_Ccoef=terms.get("Cq0"),
                general_beta=general_beta,
                general_grad_u=general_grad_u,
                smooth_family_codes=smooth_family_codes,
                smooth_candidate_index=smooth_candidate_index,
                gradient_rule=self.explicit_gradient_rule,
            )
            self.last_forward_mode = "explicit"
            self.forward_mode_counts["explicit"] += 1
            output = GaugeForwardExplicitFunction.apply(
                z,
                state["x"],
                state["u"],
                state["r"],
                state["beta"],
                state["scaling"],
                state["best_grad_u"],
                float(self._eps),
            )
            if return_state:
                return output, state
            return output
        if method != "autograd":
            raise ValueError(f"Unsupported GaugeMap forward method: {method}.")
        if self._has_general_convex_constraints():
            if self.p_norm != 2:
                raise NotImplementedError(
                    "GaugeMap general-convex differentiation requires p_norm=2. "
                    "Autograd through the bisection boundary search is not a valid derivative."
                )
            # Bisection selects branches with torch.where, whose traced derivative
            # treats the selected radius as locally constant.  Route the default
            # path through the implicit VJP instead of exposing that wrong gradient.
            return self.forward(z, method="explicit", tie_tol=tie_tol, return_state=return_state)
        scaling = self._compute_scaling(z, forward=True)
        self.last_forward_mode = "autograd"
        self.forward_mode_counts["autograd"] += 1
        x = scaling * z + self.center
        if return_state:
            return x, None
        return x

    def inverse(self, x, method="autograd"):
        if method not in {"autograd", "explicit"}:
            raise ValueError(f"Unsupported GaugeMap inverse method: {method}.")
        x_shifted = x - self.center
        scaling = self._compute_scaling(x_shifted, forward=False)
        return scaling * x_shifted

    def vjp(self, z, grad_x, method="autograd", state=None):
        if method == "explicit":
            explicit_state = state
            if explicit_state is None:
                _, explicit_state = self.forward(z, method="explicit", return_state=True)
            return explicit_vjp_from_state(
                z=z,
                u=explicit_state["u"],
                r=explicit_state["r"],
                beta=explicit_state["beta"],
                scaling=explicit_state["scaling"],
                best_grad_u=explicit_state["best_grad_u"],
                grad_x=grad_x,
                eps=self._eps,
            )
        if method != "autograd":
            raise ValueError(f"Unsupported GaugeMap vjp method: {method}.")
        with torch.enable_grad():
            z_detached = z.detach().requires_grad_(True)
            x = self.forward(z_detached, method=method)
            grad_z = torch.autograd.grad(x, z_detached, grad_outputs=grad_x, create_graph=False)[0]
        return grad_z

    def _direction_and_norm(self, x):
        x_norm = torch.norm(x, dim=1, p=2, keepdim=True)
        safe_x_norm = x_norm.clamp_min(self._eps)
        u_dir = x / safe_x_norm
        return x_norm, safe_x_norm, u_dir

    def _compute_scaling(self, x, forward=True):
        x_norm, safe_x_norm, u_dir = self._direction_and_norm(x)
        nonzero_mask = (x_norm > self._eps).view(-1)
        if not torch.any(nonzero_mask):
            return torch.ones_like(x_norm)
        p_norm = x_norm if self.p_norm == 2 else torch.norm(x, dim=1, p=self.p_norm, keepdim=True)
        safe_p_norm = p_norm.clamp_min(self._eps)
        boundary_distance = torch.ones_like(x_norm)
        boundary_distance[nonzero_mask] = self._closed_form_point_to_boundary_distance(
            u_dir[nonzero_mask]
        ).clamp_min(self._eps)
        if forward:
            scaling = p_norm / safe_x_norm / boundary_distance
        else:
            scaling = x_norm / safe_p_norm * boundary_distance
        return torch.where(x_norm > self._eps, scaling, torch.ones_like(scaling))

    def gauge(self, x, method="autograd"):
        if method not in {"autograd", "explicit"}:
            raise ValueError(f"Unsupported GaugeMap gauge method: {method}.")
        x_norm, _, u_dir = self._direction_and_norm(x)
        nonzero_mask = (x_norm > self._eps).view(-1)
        if not torch.any(nonzero_mask):
            return torch.zeros_like(x_norm)
        boundary_distance = torch.ones_like(x_norm)
        boundary_distance[nonzero_mask] = self._closed_form_point_to_boundary_distance(u_dir[nonzero_mask])
        scaling = x_norm * boundary_distance
        return torch.where(x_norm > self._eps, scaling, torch.zeros_like(scaling))

    def _get_static_terms(self, device, dtype):
        key = (device.type, device.index, dtype)
        cached = self._static_cache.get(key)
        if cached is not None:
            return cached

        terms = {}
        x0 = self.center.to(device=device, dtype=dtype)
        terms["x0"] = x0
        constraint_set = self.constraint_set

        if constraint_set.G is not None:
            G = constraint_set.G.to(device=device, dtype=dtype)
            h = constraint_set.h.to(device=device, dtype=dtype)
            C = constraint_set.C.to(device=device, dtype=dtype)
            d = constraint_set.d.to(device=device, dtype=dtype)
            Gx0 = torch.matmul(G, x0.T).permute(2, 0, 1) + h.unsqueeze(0)
            Cx0 = torch.matmul(x0, C.T) + d
            terms.update(
                {
                    "G": G,
                    "C": C,
                    "Gx0": Gx0,
                    "Cx0": Cx0,
                    "soc_gradB_const": 2
                    * (torch.einsum("mkn,mk->mn", G, Gx0.squeeze(0)) - Cx0.squeeze(0).unsqueeze(-1) * C),
                    "soc_Ccoef": torch.sum(Gx0.squeeze(0) * Gx0.squeeze(0), dim=-1, keepdim=True).T
                    - Cx0 * Cx0,
                }
            )

        if constraint_set.Qq is not None:
            Qq = constraint_set.Qq.to(device=device, dtype=dtype)
            pq = constraint_set.pq.to(device=device, dtype=dtype)
            bq = constraint_set.bq.to(device=device, dtype=dtype)
            Qsym = 0.5 * (Qq + torch.transpose(Qq, 1, 2))
            Qq_x0 = torch.matmul(Qq, x0.T).permute(2, 0, 1)
            Qsym_x0 = torch.matmul(Qsym, x0.T).permute(2, 0, 1)
            Cq0 = (
                0.5 * torch.sum(x0.unsqueeze(1) * Qq_x0, dim=-1)
                + torch.matmul(x0, pq.T)
                - bq.view(1, -1)
            )
            terms.update(
                {
                    "Qsym": Qsym,
                    "pq": pq,
                    "Cq0": Cq0,
                    "quad_gradB_const": (Qsym_x0.squeeze(0) + pq),
                }
            )

        if constraint_set.A is not None:
            A = constraint_set.A.to(device=device, dtype=dtype)
            b = constraint_set.b.to(device=device, dtype=dtype)
            Ax0 = torch.matmul(x0, A.T)
            slack = b - Ax0
            terms.update(
                {
                    "A": A,
                    "inv_slack": _safe_divide(torch.ones_like(slack), slack, eps=self._eps),
                }
            )

        if constraint_set.L is not None:
            L = constraint_set.L.to(device=device, dtype=dtype)
            U = constraint_set.U.to(device=device, dtype=dtype)
            lower_denom = L.view(1, -1) - x0
            upper_denom = U.view(1, -1) - x0
            terms.update(
                {
                    "inv_lower_denom": _safe_divide(torch.ones_like(lower_denom), lower_denom, eps=self._eps),
                    "inv_upper_denom": _safe_divide(torch.ones_like(upper_denom), upper_denom, eps=self._eps),
                }
            )

        family_parts = []
        index_parts = []
        if terms.get("A") is not None:
            count = int(terms["A"].shape[0])
            family_parts.append(torch.full((count,), _FAMILY_LINEAR, device=device, dtype=torch.long))
            index_parts.append(torch.arange(count, device=device, dtype=torch.long))
        if terms.get("inv_lower_denom") is not None:
            count = int(self.constraint_set.nvar)
            arange = torch.arange(count, device=device, dtype=torch.long)
            family_parts.append(torch.full((count,), _FAMILY_BOX_LOWER, device=device, dtype=torch.long))
            index_parts.append(arange)
            family_parts.append(torch.full((count,), _FAMILY_BOX_UPPER, device=device, dtype=torch.long))
            index_parts.append(arange)
        if terms.get("G") is not None:
            count = int(terms["G"].shape[0])
            family_parts.append(torch.full((count,), _FAMILY_SOC, device=device, dtype=torch.long))
            index_parts.append(torch.arange(count, device=device, dtype=torch.long))
        if terms.get("Qsym") is not None:
            count = int(terms["Qsym"].shape[0])
            family_parts.append(torch.full((count,), _FAMILY_QUAD, device=device, dtype=torch.long))
            index_parts.append(torch.arange(count, device=device, dtype=torch.long))
        general_count = self._general_constraint_count(x0)
        if general_count:
            count = int(general_count)
            family_parts.append(torch.full((count,), _FAMILY_GENERAL, device=device, dtype=torch.long))
            index_parts.append(torch.arange(count, device=device, dtype=torch.long))
        if family_parts:
            terms["smooth_family_codes"] = torch.cat(family_parts)
            terms["smooth_candidate_index"] = torch.cat(index_parts)

        self._static_cache[key] = terms
        return terms

    def _supports_explicit(self):
        has_supported_constraints = any(
            (
                getattr(self.constraint_set, "A", None) is not None,
                getattr(self.constraint_set, "L", None) is not None,
                getattr(self.constraint_set, "G", None) is not None,
                getattr(self.constraint_set, "Qq", None) is not None,
                self._has_general_convex_constraints(),
            )
        )
        return (self.p_norm == 2) and has_supported_constraints

    def _has_general_convex_constraints(self):
        return callable(getattr(self.constraint_set, "general_convex_constraint_x", None)) and callable(
            getattr(self.constraint_set, "general_convex_constraint_gradient_x", None)
        )

    def _general_constraint_count(self, center):
        if not self._has_general_convex_constraints():
            return 0
        values = self.constraint_set.general_convex_constraint_x(center)
        if values.ndim == 1:
            return 1
        if values.ndim != 2:
            raise ValueError("General convex constraint values must have shape (batch, n_constraints).")
        return int(values.shape[1])

    def _general_boundary_candidates(self, u, center, *, need_gradient):
        if not self._has_general_convex_constraints():
            return None, None
        return general_boundary_candidates(
            u,
            center=center,
            constraint_fn=self.constraint_set.general_convex_constraint_x,
            gradient_fn=self.constraint_set.general_convex_constraint_gradient_x,
            eps=self._eps,
            initial_radius=self.general_bisection_initial_radius,
            max_radius=self.general_bisection_max_radius,
            expand_steps=self.general_bisection_expand_steps,
            iterations=self.general_bisection_iterations,
            need_gradient=need_gradient,
        )

    def _closed_form_point_to_boundary_distance(self, u):
        terms = self._get_static_terms(u.device, u.dtype)
        batch_size = u.shape[0]
        smooth_candidates = []
        max_alpha = None
        neg_inf = torch.tensor(float("-inf"), device=u.device, dtype=u.dtype)
        reduce_mode = "full" if self.smooth else "winner"
        candidate_cache = explicit_candidate_cache(
            u,
            eps=self._eps,
            A=terms.get("A"),
            inv_slack=terms.get("inv_slack"),
            inv_lower_denom=terms.get("inv_lower_denom"),
            inv_upper_denom=terms.get("inv_upper_denom"),
            G=terms.get("G"),
            C=terms.get("C"),
            Gx0=terms.get("Gx0"),
            Cx0=terms.get("Cx0"),
            soc_Ccoef=terms.get("soc_Ccoef"),
            Qsym=terms.get("Qsym"),
            quad_gradB_const=terms.get("quad_gradB_const"),
            quad_Ccoef=terms.get("Cq0"),
            need_grad_aux=False,
            reduce=reduce_mode,
        )
        general_beta, _ = self._general_boundary_candidates(u, terms["x0"], need_gradient=False)

        def _consume(candidate):
            nonlocal max_alpha
            candidate = candidate.reshape(batch_size, -1)
            candidate = torch.where(candidate > self._eps, candidate, neg_inf)
            if self.smooth:
                smooth_candidates.append(candidate)
                return
            candidate_max = torch.max(candidate, dim=-1, keepdim=True)[0]
            max_alpha = candidate_max if max_alpha is None else _pairwise_max(max_alpha, candidate_max)

        candidate_keys = (
            ("alpha_soc", "alpha_quad", "alpha_lin", "alpha_box_lower", "alpha_box_upper")
            if self.smooth
            else ("soc_beta", "quad_beta", "linear_beta", "box_lower_beta", "box_upper_beta")
        )
        for key in candidate_keys:
            candidate = candidate_cache.get(key)
            if candidate is not None:
                _consume(candidate)
        if general_beta is not None:
            _consume(general_beta)
        if self.smooth:
            if not smooth_candidates:
                raise ValueError("GaugeMap requires at least one supported constraint family.")
            dist_list = torch.cat(smooth_candidates, dim=-1)
            valid_candidate = torch.isfinite(dist_list)
            if not valid_candidate.any(dim=-1, keepdim=True).all():
                raise ValueError("No valid boundary candidate found for at least one sample.")
            finite_dist = torch.where(valid_candidate, dist_list, torch.full_like(dist_list, float("-inf")))
            hard_max = torch.max(finite_dist, dim=-1, keepdim=True)[0]
            active_candidate = valid_candidate & (dist_list >= hard_max - self.smooth_tie_tol)
            active_shifted = torch.where(
                active_candidate,
                (dist_list - hard_max) / self.smooth_temperature,
                torch.full_like(dist_list, float("-inf")),
            )
            max_alpha = hard_max + self.smooth_temperature * torch.logsumexp(active_shifted, dim=-1, keepdim=True)
            return max_alpha.view(-1, 1)
        if max_alpha is None:
            raise ValueError("GaugeMap requires at least one supported constraint family.")
        if (not torch.isfinite(max_alpha).all()) or (max_alpha <= self._eps).any():
            raise ValueError("No valid positive boundary candidate found.")
        return max_alpha
