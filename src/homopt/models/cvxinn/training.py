"""Training routines for CvxINN convex-set wrappers."""

from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim

from homopt.models.inn import INN, initialize_inn_as_identity
from homopt.models.local_stretch import combine_local_stretch_objective, local_log_stretch_loss
from homopt.models.sampling import sampling_sphere
from homopt.utils import resolve_torch_device, resolve_torch_dtype

from .models import CubeGaugeCvxINN, CvxINN, SphereReparam, SphereReparamGaugeCvxINN, TanhGaugeCvxINN


def _evaluate_local_stretch_objective(
    forward_fn,
    z_tensor,
    *,
    num_directions,
    fd_eps,
    domain_shape,
    boundary_eps,
    log_eps,
    aggregate,
    objective,
    soft_temperature,
    spread_weight,
):
    loss_spread, stats = local_log_stretch_loss(
        forward_fn,
        z_tensor,
        num_directions=num_directions,
        fd_eps=fd_eps,
        domain_shape=domain_shape,
        boundary_eps=boundary_eps,
        log_eps=log_eps,
        aggregate=aggregate,
    )
    objective_value = combine_local_stretch_objective(
        loss_spread,
        stats,
        objective=objective,
        soft_temperature=soft_temperature,
        spread_weight=spread_weight,
    )
    return objective_value, loss_spread, stats


def _sample_latent_points(
    n_samples,
    n_dim,
    *,
    domain_shape,
    boundary_eps=0.0,
    boundary_focus_ratio=0.0,
    boundary_shell_radius=0.8,
    device,
    dtype,
):
    shape = str(domain_shape).strip().lower()
    max_radius = max(1.0 - float(boundary_eps), 1e-6)
    if shape in {"ball", "sphere", "l2_ball"}:
        points = sampling_sphere(int(n_samples), int(n_dim))
        points = np.asarray(points, dtype=np.float32) * float(max_radius)
        focus_ratio = float(boundary_focus_ratio)
        if focus_ratio > 0:
            n_focus = min(int(round(int(n_samples) * focus_ratio)), int(n_samples))
            if n_focus > 0:
                shell_radius = float(boundary_shell_radius) * float(max_radius)
                shell_radius = max(0.0, min(shell_radius, float(max_radius)))
                focus_points = sampling_sphere(int(n_focus), int(n_dim))
                focus_points = np.asarray(focus_points, dtype=np.float32)
                focus_norm = np.linalg.norm(focus_points, axis=1, keepdims=True)
                focus_norm = np.maximum(focus_norm, 1e-12)
                focus_dir = focus_points / focus_norm
                focus_r = np.random.uniform(low=shell_radius, high=float(max_radius), size=(n_focus, 1)).astype(np.float32)
                points[:n_focus] = focus_dir * focus_r
    elif shape in {"cube", "square", "linf_ball"}:
        points = np.random.uniform(low=-max_radius, high=max_radius, size=(int(n_samples), int(n_dim))).astype(np.float32)
    else:
        raise ValueError(f"Unsupported latent domain shape: {domain_shape}")
    return torch.tensor(points, device=device, dtype=dtype)


def train_cvxinn_min_distortion(
    convex_set,
    *,
    base_model=None,
    wrapper_type="radial_gauge",
    n_dim=None,
    h_dim=32,
    num_layer=2,
    inv_type="coupling",
    total_iteration=500,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    p_norm=2,
    gauge_forward_method="autograd",
    compactify_eps=1e-12,
    track_logdet_stats=False,
    stretch_num_directions=8,
    stretch_fd_eps=1e-3,
    stretch_log_eps=1e-8,
    stretch_boundary_eps=1e-6,
    stretch_aggregate="spread",
    stretch_objective="spread",
    stretch_soft_temperature=0.1,
    stretch_spread_weight=0.25,
    identity_init_inner=False,
    boundary_focus_ratio=0.0,
    boundary_shell_radius=0.8,
    grad_clip_norm=None,
    lr_decay_step=None,
    lr_decay_gamma=1.0,
    latent_domain_shape="ball",
    sphere_reparam_init_scale=0.0,
    record_every=10,
    eval_every=None,
    eval_batch_size=512,
    eval_z_tensor=None,
    early_stop_patience=None,
    early_stop_min_delta=0.0,
    early_stop_min_iterations=0,
    seed=2025,
    device=None,
    dtype=None,
):
    """Train a CvxINN on a fixed convex set using only end-to-end distortion loss."""

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    runtime_device = resolve_torch_device(device, default="cpu")
    runtime_dtype = resolve_torch_dtype(dtype, default=torch.float32)
    if n_dim is None:
        n_dim = int(convex_set.nvar)

    wrapper_mode = str(wrapper_type).strip().lower()

    if base_model is None and wrapper_mode in {"sphere_reparam_gauge", "sphere_reparam", "sphere"}:
        base_model = SphereReparam(n_dim=n_dim, init_scale=float(sphere_reparam_init_scale))
    elif base_model is None:
        base_model = INN(
            nin=n_dim,
            nhid=max(int(h_dim), n_dim + 1),
            cin=1,
            nl=int(num_layer),
            inv=inv_type,
            outact=None,
            bilip=False,
            lip=2,
            Con_type="MLP",
            cond_embed_mode="shared",
        )
    if identity_init_inner:
        initialize_inn_as_identity(base_model)

    base_model = base_model.to(device=runtime_device, dtype=runtime_dtype)
    if hasattr(base_model, "track_training_stats"):
        base_model.track_training_stats = bool(track_logdet_stats)
    if wrapper_mode in {"radial_gauge", "cvxinn", "ball_gauge"}:
        model = CvxINN.from_convex_set(
            base_model,
            convex_set,
            p_norm=p_norm,
            gauge_forward_method=gauge_forward_method,
            compactify_eps=compactify_eps,
        ).to(device=runtime_device, dtype=runtime_dtype)
    elif wrapper_mode in {"tanh_gauge", "tanh"}:
        model = TanhGaugeCvxINN.from_convex_set(
            base_model,
            convex_set,
            gauge_forward_method=gauge_forward_method,
            compactify_eps=compactify_eps,
        ).to(device=runtime_device, dtype=runtime_dtype)
    elif wrapper_mode in {"cube_gauge", "cube_reparam_gauge", "cube"}:
        model = CubeGaugeCvxINN.from_convex_set(
            base_model,
            convex_set,
            gauge_forward_method=gauge_forward_method,
            compactify_eps=compactify_eps,
        ).to(device=runtime_device, dtype=runtime_dtype)
        if str(latent_domain_shape).strip().lower() == "ball":
            latent_domain_shape = "cube"
    elif wrapper_mode in {"sphere_reparam_gauge", "sphere_reparam", "sphere"}:
        model = SphereReparamGaugeCvxINN.from_convex_set(
            base_model,
            convex_set,
            p_norm=p_norm,
            gauge_forward_method=gauge_forward_method,
        ).to(device=runtime_device, dtype=runtime_dtype)
    else:
        raise ValueError(f"Unsupported wrapper_type: {wrapper_type}")
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = None
    if lr_decay_step is not None and float(lr_decay_gamma) < 1.0:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(int(lr_decay_step), 1),
            gamma=float(lr_decay_gamma),
        )
    model.train()

    loss_list = []
    eval_interval = max(int(eval_every if eval_every is not None else record_every), 1)
    if eval_z_tensor is None:
        eval_z = _sample_latent_points(
            eval_batch_size,
            n_dim,
            domain_shape=latent_domain_shape,
            boundary_eps=stretch_boundary_eps,
            boundary_focus_ratio=boundary_focus_ratio,
            boundary_shell_radius=boundary_shell_radius,
            device=runtime_device,
            dtype=runtime_dtype,
        )
    else:
        eval_z = eval_z_tensor.to(device=runtime_device, dtype=runtime_dtype)

    def _forward(flat_z):
        repeated_condition = torch.zeros((flat_z.shape[0], 1), device=runtime_device, dtype=runtime_dtype)
        return model(flat_z, repeated_condition)

    model.eval()
    with torch.no_grad():
        best_eval_objective, best_eval_spread, _ = _evaluate_local_stretch_objective(
            _forward,
            eval_z,
            num_directions=stretch_num_directions,
            fd_eps=stretch_fd_eps,
            domain_shape=latent_domain_shape,
            boundary_eps=stretch_boundary_eps,
            log_eps=stretch_log_eps,
            aggregate=stretch_aggregate,
            objective=stretch_objective,
            soft_temperature=stretch_soft_temperature,
            spread_weight=stretch_spread_weight,
        )
    best_eval_value = float(best_eval_objective.item())
    best_eval_spread_value = float(best_eval_spread.item())
    best_iteration = 0
    best_state = deepcopy(model.state_dict())
    bad_eval_count = 0
    stop_reason = "max_iteration"
    eval_history = [
        {
            "iteration": 0,
            "objective": best_eval_value,
            "spread_loss": best_eval_spread_value,
            "is_best": True,
        }
    ]
    model.train()

    final_iteration = 0
    for iteration in range(1, int(total_iteration) + 1):
        optimizer.zero_grad(set_to_none=True)
        z_batch = _sample_latent_points(
            batch_size,
            n_dim,
            domain_shape=latent_domain_shape,
            boundary_eps=stretch_boundary_eps,
            device=runtime_device,
            dtype=runtime_dtype,
        )
        loss, loss_spread, stretch_stats = _evaluate_local_stretch_objective(
            _forward,
            z_batch,
            num_directions=stretch_num_directions,
            fd_eps=stretch_fd_eps,
            domain_shape=latent_domain_shape,
            boundary_eps=stretch_boundary_eps,
            log_eps=stretch_log_eps,
            aggregate=stretch_aggregate,
            objective=stretch_objective,
            soft_temperature=stretch_soft_temperature,
            spread_weight=stretch_spread_weight,
        )
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        final_iteration = iteration

        if (iteration % max(int(record_every), 1)) == 0:
            loss_list.append(float(loss.detach().item()))

        if (iteration % eval_interval) != 0 and iteration != int(total_iteration):
            continue

        model.eval()
        with torch.no_grad():
            eval_objective, eval_spread, _ = _evaluate_local_stretch_objective(
                _forward,
                eval_z,
                num_directions=stretch_num_directions,
                fd_eps=stretch_fd_eps,
                domain_shape=latent_domain_shape,
                boundary_eps=stretch_boundary_eps,
                log_eps=stretch_log_eps,
                aggregate=stretch_aggregate,
                objective=stretch_objective,
                soft_temperature=stretch_soft_temperature,
                spread_weight=stretch_spread_weight,
            )
        eval_value = float(eval_objective.item())
        eval_spread_value = float(eval_spread.item())
        improved = eval_value < (best_eval_value - float(early_stop_min_delta))
        if improved:
            best_eval_value = eval_value
            best_eval_spread_value = eval_spread_value
            best_iteration = iteration
            best_state = deepcopy(model.state_dict())
            bad_eval_count = 0
        else:
            bad_eval_count += 1
        eval_history.append(
            {
                "iteration": iteration,
                "objective": eval_value,
                "spread_loss": eval_spread_value,
                "is_best": bool(improved),
            }
        )
        model.train()

        patience = None if early_stop_patience is None else int(early_stop_patience)
        if (
            patience is not None
            and iteration >= int(early_stop_min_iterations)
            and bad_eval_count >= patience
        ):
            stop_reason = "early_stopping"
            break

    model.load_state_dict(best_state)
    model.eval()

    training_record = {
        "loss_list": loss_list,
        "total_iteration": int(total_iteration),
        "final_iteration": int(final_iteration),
        "record_every": max(int(record_every), 1),
        "eval_every": int(eval_interval),
        "eval_batch_size": int(eval_z.shape[0]),
        "latent_domain_shape": str(latent_domain_shape),
        "sphere_reparam_init_scale": float(sphere_reparam_init_scale),
        "stretch_aggregate": str(stretch_aggregate),
        "stretch_objective": str(stretch_objective),
        "stretch_soft_temperature": float(stretch_soft_temperature),
        "stretch_spread_weight": float(stretch_spread_weight),
        "identity_init_inner": bool(identity_init_inner),
        "wrapper_type": str(wrapper_type),
        "track_logdet_stats": bool(track_logdet_stats),
        "boundary_focus_ratio": float(boundary_focus_ratio),
        "boundary_shell_radius": float(boundary_shell_radius),
        "grad_clip_norm": None if grad_clip_norm is None else float(grad_clip_norm),
        "lr_decay_step": None if lr_decay_step is None else int(lr_decay_step),
        "lr_decay_gamma": float(lr_decay_gamma),
        "best_eval_objective": float(best_eval_value),
        "best_eval_spread_loss": float(best_eval_spread_value),
        "best_iteration": int(best_iteration),
        "stop_reason": str(stop_reason),
        "early_stop_patience": None if early_stop_patience is None else int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stop_min_iterations": int(early_stop_min_iterations),
        "eval_history": eval_history,
    }
    return model, training_record

__all__ = ["train_cvxinn_min_distortion"]
