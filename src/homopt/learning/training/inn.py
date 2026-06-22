"""INN mapping training routines."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from tqdm import tqdm

from homopt.models.inn import DEVICE, EMA, INN, INN_MODEL_CHECKPOINT_FILENAME, load_inn_mapping, save_inn_mapping
from homopt.models.local_stretch import _extract_forward_tensor, local_log_stretch_loss
from homopt.models.sampling import sampling_sphere
from homopt.utils import resolve_torch_device, resolve_torch_dtype

from .cache import load_or_train_mapping
from .utils import build_adamw_with_step_scheduler


def unsupervised_training_mdh(MDH, data, x_tensor, input_tensor, save_dir, args):
    batch_size = args["batch_size"]
    total_iteration = args["total_iteration"]
    w_penalty = args["w_penalty"]
    w_distortion = args["w_distortion"]
    w_lipschitz = float(args.get("w_lipschitz", args.get("w_lipschitz_reg", 0.0)))
    w_local_stretch = float(args.get("w_local_stretch", 0.0))
    stretch_num_directions = int(args.get("stretch_num_directions", 8))
    stretch_fd_eps = float(args.get("stretch_fd_eps", 1e-3))
    stretch_log_eps = float(args.get("stretch_log_eps", 1e-8))
    stretch_boundary_eps = float(args.get("stretch_boundary_eps", 1e-6))
    stretch_aggregate = args.get("stretch_aggregate", "max")
    latent_domain_shape = args.get("latent_domain_shape", "ball")
    results_save_freq = max(int(args.get("resultsSaveFreq", total_iteration + 1)), 1)
    record_every = max(int(args.get("record_every", 10)), 1)
    show_progress = bool(args.get("show_progress", args.get("verbose", False)))
    grad_clip_norm = args.get("grad_clip_norm", None)
    runtime_device = resolve_torch_device(args.get("device"), default=str(x_tensor.device))
    runtime_dtype = resolve_torch_dtype(args.get("dtype"), default=x_tensor.dtype)
    use_amp = bool(args.get("use_amp", False)) and torch.cuda.is_available() and str(x_tensor.device).startswith("cuda")
    amp_dtype_name = str(args.get("amp_dtype", "float16")).strip().lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    use_new_amp_api = hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and hasattr(torch.amp, "autocast")
    if use_amp:
        if use_new_amp_api:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    volume_list, penalty_list, dist_list, lipschitz_list, stretch_list = [], [], [], [], []
    x_tensor = x_tensor.contiguous()
    if input_tensor is not None:
        input_tensor = input_tensor.contiguous()
        if input_tensor.device.type == "cpu" and x_tensor.device.type == "cuda" and not input_tensor.is_pinned():
            input_tensor = input_tensor.pin_memory()
    n_dim = x_tensor.shape[1]
    bias_tensor = torch.full((batch_size, n_dim), float(args["center"]), device=x_tensor.device, dtype=x_tensor.dtype)
    optimizer, scheduler = build_adamw_with_step_scheduler(
        args,
        MDH,
        total_iteration=total_iteration,
        default_lr=1e-4,
        weight_decay=1e-5,
    )
    ema_decay = args.get("ema_decay", 0.99)
    ema = EMA(MDH, decay=ema_decay)
    MDH.train()

    for n in tqdm(range(total_iteration), disable=not show_progress):
        optimizer.zero_grad(set_to_none=True)
        batch_index = torch.randint(0, x_tensor.shape[0], (batch_size,), device=x_tensor.device)
        x_input = x_tensor[batch_index]
        if input_tensor is not None:
            batch_index = torch.randint(0, input_tensor.shape[0], (batch_size,), device=input_tensor.device)
            input_batch = input_tensor[batch_index]
        else:
            input_batch, _ = data.generate_problem_samples_torch(
                n_samples=batch_size,
                sample_obj=False,
                device=runtime_device,
                dtype=runtime_dtype,
            )
        if input_batch.device != x_input.device or input_batch.dtype != x_input.dtype:
            input_batch = input_batch.to(
                device=x_input.device,
                dtype=x_input.dtype,
                non_blocking=(input_batch.device.type == "cpu" and x_input.device.type == "cuda"),
            )
        if use_amp:
            if use_new_amp_api:
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True, dtype=amp_dtype)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            u_mapped, logdet, logdis = MDH(x_input, input_batch)
            y_partial = data.scale(input_batch, u_mapped)
            y_full = data.complete_partial(input_batch, y_partial)
            ineq_vio = torch.clamp(data.ineq_resid(input_batch, y_full), min=0)
            penalty = ineq_vio.sum(-1)
            logdet_mean = logdet.mean()
            penalty_mean = penalty.mean()
            logdis_mean = logdis.mean()
            logdis_max = torch.amax(logdis)
            if w_local_stretch > 0:
                repeated_input = input_batch.unsqueeze(1).expand(-1, stretch_num_directions, *input_batch.shape[1:])

                def _end_to_end_forward(flat_z):
                    batch_repeat = repeated_input.reshape(flat_z.shape[0], *input_batch.shape[1:])
                    output = MDH(flat_z, batch_repeat)
                    output = _extract_forward_tensor(output)
                    return data.complete_partial(batch_repeat, data.scale(batch_repeat, output))

                local_stretch_loss, _ = local_log_stretch_loss(
                    _end_to_end_forward,
                    x_input,
                    num_directions=stretch_num_directions,
                    fd_eps=stretch_fd_eps,
                    domain_shape=latent_domain_shape,
                    boundary_eps=stretch_boundary_eps,
                    log_eps=stretch_log_eps,
                    aggregate=stretch_aggregate,
                )
            else:
                local_stretch_loss = torch.zeros((), device=x_input.device, dtype=x_input.dtype)
            loss = (
                w_distortion * logdis_mean
                + w_lipschitz * logdis_max
                + w_penalty * penalty_mean
                + w_local_stretch * local_stretch_loss
                - logdet_mean / n_dim
            )
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if grad_clip_norm is not None:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(MDH.parameters(), float(grad_clip_norm))
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        ema.update()

        if (n % record_every) == 0:
            volume_list.append(logdet_mean.detach().item() / n_dim)
            penalty_list.append(penalty_mean.detach().item())
            dist_list.append(logdis_mean.detach().item() / n_dim)
            lipschitz_list.append(logdis_max.detach().item() / n_dim)
            stretch_list.append(local_stretch_loss.detach().item())

        if n > 0 and n % results_save_freq == 0:
            save_inn_mapping(MDH, save_dir, filename=INN_MODEL_CHECKPOINT_FILENAME)
            MDH.eval()
            with torch.inference_mode():
                u0 = MDH(bias_tensor, input_batch)
                y0_partial = data.scale(input_batch, u0)
                y0_full = data.complete_partial(input_batch, y0_partial)
                violation_0 = data.check_feasibility(input_batch, y0_full)
                penalty_0 = torch.abs(violation_0).detach().cpu()
            print(
                "Iteration: {}/{}, Volume: {:.3f}, Penalty: {:.3f}, Distortion: {:.3f}, Lipschitz: {:.3f}, LocalStretch: {:.3f}, Center Penalty: {:.4f}, Valid rate: {:.2f}".format(
                    n,
                    total_iteration,
                    logdet_mean.detach().item() / n_dim,
                    penalty_mean.detach().item(),
                    logdis_mean.detach().item() / n_dim,
                    logdis_max.detach().item() / n_dim,
                    local_stretch_loss.detach().item(),
                    penalty_0.sum(-1).max().numpy(),
                    (penalty_0.max(-1)[0] < 1e-5).numpy().mean(),
                )
            )
            MDH.train()
    training_record = {
        "volume_list": volume_list,
        "penalty_list": penalty_list,
        "dist_list": dist_list,
        "lipschitz_list": lipschitz_list,
        "stretch_list": stretch_list,
        "trans_list": [],
        "w_lipschitz": float(w_lipschitz),
    }
    if hasattr(data, "training_record_metadata"):
        metadata = data.training_record_metadata()
        if metadata:
            training_record.update(dict(metadata))
    ema.apply_shadow()
    return MDH, training_record


def train_mdh_mapping(data, args, save_dir):
    paras = args["model_config"]
    runtime_device = resolve_torch_device(paras.get("device"), default=str(DEVICE))
    runtime_dtype = resolve_torch_dtype(paras.get("dtype"), default=torch.float32)
    con_type = paras["Con_type"]
    if con_type == "MLP":
        c_dim = int(data.n_qua * data.npara)
    else:
        c_dim = data.npara
    n_dim = data.nvar
    num_layer = paras["num_layer"]
    inv_type = paras["inv_type"]
    h_dim = max(n_dim + 1, paras["h_dim"])
    model = INN(
        n_dim,
        h_dim,
        c_dim,
        num_layer,
        inv=inv_type,
        outact=None,
        bilip=paras.get("bilip", False),
        lip=paras.get("L", 2),
        Con_type=con_type,
        cond_embed_mode=paras.get("cond_embed_mode", "shared"),
    ).to(device=runtime_device, dtype=runtime_dtype)
    latent_init_shape = str(paras.get("latent_init_shape", "auto")).strip().lower()
    if latent_init_shape not in {"auto", "sphere", "cube"}:
        raise ValueError(
            f"Unsupported latent_init_shape: {latent_init_shape}. Use 'auto', 'sphere', or 'cube'."
        )
    use_sphere = (n_dim == 2) if latent_init_shape == "auto" else (latent_init_shape == "sphere")
    if use_sphere:
        ball_sample = sampling_sphere(paras["n_samples"], n_dim)
        ball_sample = torch.tensor(ball_sample, dtype=runtime_dtype, device=runtime_device)
    else:
        ball_sample = (torch.rand(size=[paras["n_samples"], n_dim], device=runtime_device, dtype=runtime_dtype) - 0.5) * 2
    input_samples = None
    training_args = dict(paras)
    training_args.setdefault("device", str(runtime_device))
    training_args.setdefault("dtype", runtime_dtype)
    training_args.setdefault("latent_domain_shape", "ball" if use_sphere else "cube")
    model, training_record = unsupervised_training_mdh(model, data, ball_sample, input_samples, save_dir, training_args)
    return model, training_record


def build_inn_runtime_args(
    *,
    runtime_device,
    runtime_dtype,
    base_inn_args,
    base_optimizer_config,
    model_config=None,
    optimizer_config=None,
):
    model_args = {**dict(base_inn_args or {}), **dict(model_config or {})}
    model_args["device"] = str(runtime_device)
    model_args["dtype"] = runtime_dtype
    optimizer_args = {**dict(base_optimizer_config or {}), **dict(optimizer_config or {})}
    return model_args, optimizer_args


def load_or_train_inn_mapping(data, args, save_dir, *, retrain=False):
    return load_or_train_mapping(
        data=data,
        args=args,
        save_dir=save_dir,
        retrain=retrain,
        model_filename="inn_mapping.pt",
        record_filename="inn_training_record.npy",
        load_fn=load_inn_mapping,
        save_fn=save_inn_mapping,
        train_fn=train_mdh_mapping,
    )


__all__ = [
    "build_inn_runtime_args",
    "load_or_train_inn_mapping",
    "train_mdh_mapping",
    "unsupervised_training_mdh",
]
