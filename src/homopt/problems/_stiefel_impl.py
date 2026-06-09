"""Stiefel-manifold problem families."""

from __future__ import annotations

import torch

from homopt.problems.base import TensorRuntimeMixin


def _square(value):
    return value * value


def _as_float_tensor(value, *, name):
    if value is None:
        raise ValueError(f"Missing required StiefelProblem config field: {name}")
    return torch.as_tensor(value, dtype=torch.float32)


def _optional_tensor(value):
    return torch.as_tensor(value, dtype=torch.float32) if value is not None else None


def _expand_bound(value, nvar, *, name):
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.repeat(nvar)
    tensor = tensor.reshape(-1)
    if tensor.numel() != nvar:
        raise ValueError(f"{name} must be scalar or have length {nvar}; got {tensor.numel()}.")
    return tensor


def _selection_matrix(nvar, indices):
    selector = torch.zeros(nvar, nvar, dtype=torch.float32)
    if indices:
        idx = torch.as_tensor(indices, dtype=torch.long)
        selector[idx, idx] = 1.0
    return selector


def _flatten_group_indices(group, n_rows, n_cols):
    if "flat_indices" in group:
        flat_indices = [int(index) for index in group["flat_indices"]]
    elif "rows" in group:
        flat_indices = [
            int(row) * n_cols + col
            for row in group["rows"]
            for col in range(n_cols)
        ]
    else:
        raise ValueError("Each group norm constraint needs either 'rows' or 'flat_indices'.")
    nvar = n_rows * n_cols
    if not flat_indices:
        raise ValueError("Group norm constraints cannot be empty.")
    if min(flat_indices) < 0 or max(flat_indices) >= nvar:
        raise ValueError(f"Group norm flat indices must be in [0, {nvar}).")
    return sorted(set(flat_indices))


def _normalize_group_norm_constraints(config, n_rows, n_cols):
    raw_constraints = config.get("group_norm_constraints", config.get("feature_group_norms", []))
    normalized = []
    for index, group in enumerate(raw_constraints):
        budget = float(group["budget"])
        if budget <= 0:
            raise ValueError("Group norm constraint budget must be positive.")
        flat_indices = _flatten_group_indices(group, n_rows, n_cols)
        normalized.append(
            {
                "name": str(group.get("name", f"group_{index}")),
                "rows": None if "rows" not in group else [int(row) for row in group["rows"]],
                "flat_indices": flat_indices,
                "budget": budget,
            }
        )
    return normalized


def _build_soc_constraint_tensors(nvar, *, norm_radius=None, group_norm_constraints=()):
    matrices = []
    radii = []
    if norm_radius is not None:
        norm_radius = float(norm_radius)
        if norm_radius <= 0:
            raise ValueError("norm_radius must be positive.")
        matrices.append(torch.eye(nvar, dtype=torch.float32))
        radii.append(norm_radius)
    for group in group_norm_constraints:
        matrices.append(_selection_matrix(nvar, group["flat_indices"]))
        radii.append(float(group["budget"]))
    if not matrices:
        return None, None, None, None
    G = torch.stack(matrices, dim=0)
    h = torch.zeros(len(matrices), nvar, dtype=torch.float32)
    C = torch.zeros(len(matrices), nvar, dtype=torch.float32)
    d = torch.tensor(radii, dtype=torch.float32)
    return G, h, C, d


def _normalize_pca_covariance(covariance_tensor, *, n_cols, objective_normalization=None, objective_scale=None):
    mode = "none" if objective_normalization is None else str(objective_normalization).lower()
    aliases = {
        "": "none",
        "none": "none",
        "false": "none",
        "no": "none",
        "off": "none",
        "top-k-sum": "top_sum",
        "top_k_sum": "top_sum",
        "top_sum": "top_sum",
        "trace": "trace",
        "spectral": "spectral",
        "top": "spectral",
        "fro": "frobenius",
        "frobenius": "frobenius",
    }
    if mode not in aliases:
        raise ValueError(
            "objective_normalization must be one of: None, 'top_sum', "
            "'trace', 'spectral', or 'frobenius'."
        )
    mode = aliases[mode]

    if objective_scale is not None:
        scale = float(objective_scale)
        if scale <= 0:
            raise ValueError("objective_scale must be positive.")
        return covariance_tensor / scale, scale, "custom"

    if mode == "none":
        return covariance_tensor, 1.0, mode
    if mode == "top_sum":
        eigvals = torch.linalg.eigvalsh(covariance_tensor)
        top_count = min(max(int(n_cols), 1), int(eigvals.numel()))
        scale = float(torch.topk(eigvals, top_count).values.sum().detach().cpu().item())
    elif mode == "trace":
        scale = float(torch.trace(covariance_tensor).detach().cpu().item())
    elif mode == "spectral":
        scale = float(torch.linalg.eigvalsh(covariance_tensor).abs().max().detach().cpu().item())
    else:
        scale = float(torch.linalg.norm(covariance_tensor, ord="fro").detach().cpu().item())

    if scale <= 1e-12:
        raise ValueError(f"Cannot normalize Stiefel PCA objective with nonpositive scale {scale}.")
    return covariance_tensor / scale, scale, mode


def create_constrained_pca_stiefel_config(
    *,
    data=None,
    covariance=None,
    n_cols=1,
    restricted_rows=None,
    group_budget=None,
    group_norm_constraints=None,
    L=-1.5,
    U=1.5,
    norm_radius=None,
    objective_normalization=None,
    objective_scale=None,
):
    """Build a Stiefel config for constrained PCA/subspace learning."""

    if covariance is None:
        if data is None:
            raise ValueError("Either covariance or data must be provided.")
        data_tensor = torch.as_tensor(data, dtype=torch.float32)
        if data_tensor.ndim != 2:
            raise ValueError("data must have shape (n_samples, n_features).")
        centered = data_tensor - data_tensor.mean(dim=0, keepdim=True)
        covariance_tensor = centered.T @ centered / max(int(data_tensor.shape[0]) - 1, 1)
    else:
        covariance_tensor = torch.as_tensor(covariance, dtype=torch.float32)
    if covariance_tensor.ndim != 2 or covariance_tensor.shape[0] != covariance_tensor.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    covariance_tensor, normalization_scale, normalization_mode = _normalize_pca_covariance(
        covariance_tensor,
        n_cols=n_cols,
        objective_normalization=objective_normalization,
        objective_scale=objective_scale,
    )

    constraints = list(group_norm_constraints or [])
    if restricted_rows is not None or group_budget is not None:
        if restricted_rows is None or group_budget is None:
            raise ValueError("restricted_rows and group_budget must be provided together.")
        constraints.append(
            {
                "name": "restricted_rows",
                "rows": [int(row) for row in restricted_rows],
                "budget": float(group_budget),
            }
        )

    config = {
        "A_obj": covariance_tensor,
        "n_cols": int(n_cols),
        "L": L,
        "U": U,
        "group_norm_constraints": constraints,
        "objective_normalization": normalization_mode,
        "objective_scale": normalization_scale,
    }
    if norm_radius is not None:
        config["norm_radius"] = float(norm_radius)
    return config


class StiefelProblem(TensorRuntimeMixin):
    """Orthogonality-constrained quadratic objective with convex side constraints.

    The deterministic form is:

        minimize    -tr(X.T A X)
        subject to  X.T B X = I
                    A_ineq vec(X) <= b_ineq
                    L <= vec(X) <= U
                    ||vec(X)||_2 <= norm_radius
                    ||X[rows, :]||_F <= budget

    `B` defaults to identity. The convex inequalities are optional, but Hom-ALM
    needs at least one bounded convex inequality family for the gauge map.
    """

    name = "StiefelProblem"
    has_eq = True
    is_parametric = False

    def __init__(self, config):
        self.prob_para = dict(config)
        self.A_obj = _as_float_tensor(
            config.get("A_obj", config.get("objective_matrix", config.get("A"))),
            name="A_obj",
        )
        if self.A_obj.ndim != 2 or self.A_obj.shape[0] != self.A_obj.shape[1]:
            raise ValueError("A_obj must be a square matrix.")
        self.n_rows = int(self.A_obj.shape[0])
        self.n_cols = int(config.get("n_cols", config.get("rank", 1)))
        if self.n_cols < 1 or self.n_cols > self.n_rows:
            raise ValueError("n_cols must satisfy 1 <= n_cols <= A_obj.shape[0].")
        self.nvar = self.n_rows * self.n_cols

        self.B_eq = _optional_tensor(config.get("B_eq", config.get("B")))
        if self.B_eq is not None and self.B_eq.shape != self.A_obj.shape:
            raise ValueError("B_eq must have the same shape as A_obj.")

        self.A = _optional_tensor(config.get("A_ineq"))
        self.b = _optional_tensor(config.get("b_ineq"))
        if (self.A is None) != (self.b is None):
            raise ValueError("A_ineq and b_ineq must be provided together.")
        if self.A is not None:
            if self.A.ndim != 2 or self.A.shape[1] != self.nvar:
                raise ValueError(f"A_ineq must have shape (m, {self.nvar}).")
            self.b = self.b.reshape(-1)
            if self.b.numel() != self.A.shape[0]:
                raise ValueError("b_ineq length must match A_ineq rows.")

        self.L = _expand_bound(config.get("L"), self.nvar, name="L")
        self.U = _expand_bound(config.get("U"), self.nvar, name="U")
        if (self.L is None) != (self.U is None):
            raise ValueError("L and U must be provided together.")
        if self.L is not None and torch.any(self.L >= self.U):
            raise ValueError("All lower bounds must be strictly smaller than upper bounds.")

        self.norm_radius = config.get("norm_radius", None)
        self.group_norm_constraints = _normalize_group_norm_constraints(config, self.n_rows, self.n_cols)
        self.G, self.h, self.C, self.d = _build_soc_constraint_tensors(
            self.nvar,
            norm_radius=self.norm_radius,
            group_norm_constraints=self.group_norm_constraints,
        )
        self.norm_radius = None if self.norm_radius is None else float(self.norm_radius)

        self.Qq = None
        self.pq = None
        self.bq = None
        self.n_lin = int(self.A.shape[0]) if self.A is not None else 0
        self.n_soc = int(self.G.shape[0]) if self.G is not None else 0
        self.n_qua = 0
        self.n_box = 2 * self.nvar if self.L is not None else 0
        self.ncon = self.n_lin + self.n_soc + self.n_box
        self.n_eq = self.n_cols * self.n_cols
        self.ineq_cons = range(self.ncon)
        self.eq_cons = range(self.ncon, self.ncon + self.n_eq)
        self.n_constraints = self.ncon + self.n_eq
        self.device = torch.device("cpu")

    def __str__(self):
        return self.name

    def to_dtype(self, dtype):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor) and attr.is_floating_point():
                setattr(self, attr_name, attr.to(dtype=dtype))
        return self

    def _matrix_view(self, x):
        if x.ndim == 1:
            x = x.view(1, -1)
        return x.reshape(x.shape[0], self.n_rows, self.n_cols)

    def _b_times_x(self, X):
        if self.B_eq is None:
            return X
        return torch.einsum("ij,bjp->bip", self.B_eq, X)

    def objective_x(self, x):
        X = self._matrix_view(x)
        AX = torch.einsum("ij,bjp->bip", self.A_obj, X)
        return -torch.sum(X * AX, dim=(1, 2), keepdim=False).view(-1, 1)

    def gradient_objective_x(self, x):
        X = self._matrix_view(x)
        sym_A = self.A_obj + self.A_obj.T
        grad = -torch.einsum("ij,bjp->bip", sym_A, X)
        return grad.reshape_as(x)

    def eq_constraint_x(self, x):
        X = self._matrix_view(x)
        BX = self._b_times_x(X)
        gram = torch.matmul(X.transpose(1, 2), BX)
        identity = torch.eye(self.n_cols, device=x.device, dtype=x.dtype).unsqueeze(0)
        return (gram - identity).reshape(x.shape[0], -1)

    def inequality_constraint_terms_x(self, x):
        resids = []
        if self.A is not None:
            resids.append(torch.matmul(x, self.A.T) - self.b)
        if self.G is not None:
            gx_h = torch.einsum("mkn,bn->bmk", self.G, x) + self.h.unsqueeze(0)
            rhs = torch.matmul(x, self.C.T) + self.d.view(1, -1)
            resids.append(torch.norm(gx_h, dim=-1, p=2) - rhs)
        if self.L is not None:
            resids.append(self.L.view(1, -1) - x)
            resids.append(x - self.U.view(1, -1))
        return resids

    def constraint_x(self, x, clip=True, eq_cons=True):
        resids = self.inequality_constraint_terms_x(x)
        if resids:
            ineq_resid = torch.cat(resids, dim=1)
        else:
            ineq_resid = torch.empty(x.shape[0], 0, device=x.device, dtype=x.dtype)
        if clip:
            ineq_resid = torch.clamp(ineq_resid, min=0)
        if not eq_cons:
            return ineq_resid
        return torch.cat([ineq_resid, self.eq_constraint_x(x)], dim=1)

    def lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        objective = self.objective_x(x).view(-1)
        ineq_resid = self.constraint_x(x, clip=False, eq_cons=False)
        eq_resid = self.eq_constraint_x(x)
        constraint_resid = torch.cat([ineq_resid, eq_resid], dim=1)
        if penalty_coef is not None:
            ineq_penalty = _square(torch.clamp(ineq_resid, min=0)).sum(-1)
            eq_penalty = _square(eq_resid).sum(-1)
            penalty = 0.5 * penalty_coef * (ineq_penalty + eq_penalty)
        else:
            penalty = 0
        if proximal_coef is not None and x_outer is not None:
            proximal = 0.5 * proximal_coef * _square(x - x_outer).sum(-1)
        else:
            proximal = 0
        dual = (dual_var * constraint_resid).sum(-1)
        return objective + penalty + dual + proximal

    def _explicit_inequality_weighted_gradient_x(self, x, weights):
        grad = torch.zeros_like(x)
        cursor = 0
        if self.A is not None:
            next_cursor = cursor + self.n_lin
            grad = grad + torch.matmul(weights[:, cursor:next_cursor], self.A)
            cursor = next_cursor
        if self.G is not None:
            next_cursor = cursor + self.n_soc
            gx_h = torch.einsum("mkn,bn->bmk", self.G, x) + self.h.unsqueeze(0)
            gx_norm = torch.norm(gx_h, dim=-1, keepdim=True).clamp_min(1e-12)
            soc_grads = torch.einsum("bmk,mkn->bmn", gx_h / gx_norm, self.G) - self.C.unsqueeze(0)
            grad = grad + torch.einsum("bm,bmn->bn", weights[:, cursor:next_cursor], soc_grads)
            cursor = next_cursor
        if self.L is not None:
            lower_weights = weights[:, cursor:cursor + self.nvar]
            cursor += self.nvar
            upper_weights = weights[:, cursor:cursor + self.nvar]
            grad = grad - lower_weights + upper_weights
        return grad

    def _explicit_equality_weighted_gradient_x(self, x, weights):
        X = self._matrix_view(x)
        W = weights.reshape(x.shape[0], self.n_cols, self.n_cols)
        if self.B_eq is None:
            grad = torch.matmul(X, W.transpose(1, 2)) + torch.matmul(X, W)
        else:
            BX = torch.einsum("ij,bjp->bip", self.B_eq, X)
            BTX = torch.einsum("ij,bjp->bip", self.B_eq.T, X)
            grad = torch.matmul(BX, W.transpose(1, 2)) + torch.matmul(BTX, W)
        return grad.reshape_as(x)

    def _explicit_lagrangian_x_gradient(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None):
        grad = self.gradient_objective_x(x)
        if self.ncon > 0:
            ineq_resid = self.constraint_x(x, clip=False, eq_cons=False)
            ineq_weights = dual_var[:, self.ineq_cons]
            if penalty_coef is not None:
                ineq_weights = ineq_weights + penalty_coef * torch.clamp(ineq_resid, min=0)
            grad = grad + self._explicit_inequality_weighted_gradient_x(x, ineq_weights)
        eq_weights = dual_var[:, self.eq_cons]
        if penalty_coef is not None:
            eq_weights = eq_weights + penalty_coef * self.eq_constraint_x(x)
        grad = grad + self._explicit_equality_weighted_gradient_x(x, eq_weights)
        if proximal_coef is not None and x_outer is not None:
            grad = grad + proximal_coef * (x - x_outer)
        return grad

    def gradient_lagrangian_x(self, x, dual_var, penalty_coef=None, proximal_coef=None, x_outer=None, method="autograd"):
        if method == "autograd":
            x_detached = x.detach().requires_grad_(True)
            lagrangian = self.lagrangian_x(x_detached, dual_var, penalty_coef, proximal_coef, x_outer)
            return torch.autograd.grad(lagrangian.sum(), x_detached, create_graph=False)[0]
        if method == "explicit":
            return self._explicit_lagrangian_x_gradient(x, dual_var, penalty_coef, proximal_coef, x_outer)
        raise NotImplementedError(f"Unsupported Stiefel lagrangian gradient method: {method}")

    def objective_z(self, z, hom_map=None, hom_map_method="autograd"):
        x = hom_map.forward(z, method=hom_map_method)
        return self.objective_x(x)

    def gradient_objective_z(self, z, hom_map=None, method="autograd", hom_map_method="autograd"):
        if method != "autograd":
            raise NotImplementedError("StiefelProblem only supports autograd objective_z gradients.")
        z_detached = z.detach().requires_grad_(True)
        obj = self.objective_z(z_detached, hom_map, hom_map_method=hom_map_method)
        return torch.autograd.grad(obj.sum(), z_detached, create_graph=False)[0]

    def lagrangian_z(
        self,
        z,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        hom_map=None,
        z_outer=None,
        x_outer=None,
        proximal_space="z",
        hom_map_method="autograd",
    ):
        x = hom_map.forward(z, method=hom_map_method)
        objective = self.objective_x(x).view(-1)
        eq_resid = self.eq_constraint_x(x)
        if penalty_coef is not None:
            penalty = 0.5 * penalty_coef * _square(eq_resid).sum(-1)
        else:
            penalty = 0
        if proximal_coef is not None:
            if proximal_space == "z":
                proximal = 0.5 * proximal_coef * _square(z - z_outer).sum(-1)
            elif proximal_space == "x":
                proximal = 0.5 * proximal_coef * _square(x - x_outer).sum(-1)
            else:
                raise ValueError(f"Unsupported proximal_space: {proximal_space}")
        else:
            proximal = 0
        dual = (dual_var * eq_resid).sum(-1)
        return objective + penalty + dual + proximal

    def gradient_lagrangian_z(
        self,
        z,
        dual_var,
        penalty_coef=None,
        proximal_coef=None,
        hom_map=None,
        z_outer=None,
        x_outer=None,
        proximal_space="z",
        hom_map_method="autograd",
        method=None,
        x=None,
        hom_state=None,
        outer_cache=None,
    ):
        del outer_cache
        gradient_method = hom_map_method if method is None else method
        if gradient_method == "explicit":
            if hom_map is None:
                raise ValueError("hom_map is required for explicit Stiefel z-gradient.")
            if x is None:
                x, hom_state = hom_map.forward(z, method="explicit", return_state=True)
            grad_x = self.gradient_objective_x(x)
            eq_weights = dual_var
            if penalty_coef is not None:
                eq_weights = eq_weights + penalty_coef * self.eq_constraint_x(x)
            grad_x = grad_x + self._explicit_equality_weighted_gradient_x(x, eq_weights)
            if proximal_coef is not None and proximal_space == "x":
                grad_x = grad_x + proximal_coef * (x - x_outer)
            grad_z = hom_map.vjp(z, grad_x, method="explicit", state=hom_state)
            if proximal_coef is not None and proximal_space == "z":
                grad_z = grad_z + proximal_coef * (z - z_outer)
            elif proximal_coef is not None and proximal_space != "x":
                raise ValueError(f"Unsupported proximal_space: {proximal_space}")
            return grad_z
        if gradient_method != "autograd":
            raise NotImplementedError(f"Unsupported Stiefel z-gradient method: {gradient_method}")
        z_detached = z.detach().requires_grad_(True)
        lagrangian = self.lagrangian_z(
            z_detached,
            dual_var,
            penalty_coef,
            proximal_coef,
            hom_map,
            z_outer,
            x_outer,
            proximal_space,
            hom_map_method,
        )
        return torch.autograd.grad(lagrangian.sum(), z_detached, create_graph=False)[0]

    def retract(self, x):
        if self.B_eq is not None:
            raise NotImplementedError("QR retraction is currently implemented only for B_eq=None.")
        X = self._matrix_view(x)
        projected = []
        for sample in X:
            q, r = torch.linalg.qr(sample, mode="reduced")
            signs = torch.sign(torch.diag(r))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            q = q * signs.view(1, -1)
            projected.append(q[:, : self.n_cols])
        return torch.stack(projected, dim=0).reshape_as(x)

    def project(self, x):
        if self.B_eq is not None:
            return x
        return self.retract(x)

    def riemannian_gradient_x(self, x):
        if self.B_eq is not None:
            raise NotImplementedError("Riemannian gradient is currently implemented only for B_eq=None.")
        X = self._matrix_view(x)
        euclidean_grad = self._matrix_view(self.gradient_objective_x(x))
        xtg = torch.matmul(X.transpose(1, 2), euclidean_grad)
        sym_xtg = 0.5 * (xtg + xtg.transpose(1, 2))
        riemannian_grad = euclidean_grad - torch.matmul(X, sym_xtg)
        return riemannian_grad.reshape_as(x)
