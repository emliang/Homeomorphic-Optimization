"""High-level adversarial attack experiment entrypoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import _merged, _norm_label, _rebase_artifacts, _save_json
from .config import _apply_auto_eps
from .data import get_model_checkpoint
from .workflow import adversarial_attack_workflow
from homopt.experiments.common.io import ensure_dir


def _print_attack_config(attack_cfg, auto_eps_applied):
    print(
        "[adv-run] effective attack config:",
        {
            "eps": attack_cfg.get("eps"),
            "alpha": attack_cfg.get("alpha"),
            "iters": attack_cfg.get("iters"),
            "norm": attack_cfg.get("norm"),
            "optimizer": attack_cfg.get("optimizer"),
            "weighted_norm": attack_cfg.get("weighted_norm"),
            "auto_eps_applied": auto_eps_applied,
        },
    )


def adversarial_attack_experiment(
    dataset_name="mnist",
    train=False,
    attack=False,
    visualize=False,
    run_all=False,
    train_if_missing=False,
    data_root="data",
    checkpoint_dir="data",
    training_epochs=10,
    training_lr=None,
    batch_size=None,
    attack_selection_size=1000,
    attack_eval_indices=None,
    visualize_examples=5,
    visualize_eval_indices=None,
    attack_overrides=None,
    norm_eps_sweep=None,
    auto_eps=None,
    single_case_subdir_by_norm=True,
    seed=42,
    device=None,
    dtype=None,
    download=True,
    output_dir=None,
    verbose=False,
    train_log_every=1,
):
    """High-level adversarial experiment wrapper for single runs and norm sweeps."""

    base_params = {
        "dataset_name": dataset_name,
        "train": train,
        "attack": attack,
        "visualize": visualize,
        "run_all": run_all,
        "train_if_missing": train_if_missing,
        "data_root": data_root,
        "checkpoint_dir": checkpoint_dir,
        "training_epochs": training_epochs,
        "training_lr": training_lr,
        "batch_size": batch_size,
        "attack_selection_size": attack_selection_size,
        "attack_eval_indices": attack_eval_indices,
        "visualize_examples": visualize_examples,
        "visualize_eval_indices": visualize_eval_indices,
        "attack_overrides": attack_overrides,
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "download": download,
        "verbose": verbose,
        "train_log_every": train_log_every,
    }

    def _run_case(case_params, case_output_dir, *, root_output_dir=None):
        run_params, auto_eps_value = _apply_auto_eps(case_params, auto_eps)
        attack_cfg = dict(run_params.get("attack_overrides", {}))
        _print_attack_config(attack_cfg, auto_eps_applied=auto_eps_value is not None)
        payload = adversarial_attack_workflow(output_dir=case_output_dir, **run_params)
        payload["artifacts"] = _rebase_artifacts(
            payload.get("artifacts", {}),
            root_output_dir=root_output_dir,
            case_output_dir=case_output_dir,
        )
        return payload, run_params, auto_eps_value

    def _ensure_base_model_once():
        checkpoint = get_model_checkpoint(dataset_name, checkpoint_dir=checkpoint_dir)
        if checkpoint.exists():
            return
        warmup = _merged(
            base_params,
            {
                "train": True,
                "attack": False,
                "visualize": False,
                "train_if_missing": False,
            },
        )
        warmup_out = None if output_dir is None else Path(output_dir) / "base_model_warmup"
        adversarial_attack_workflow(output_dir=warmup_out, **warmup)

    if norm_eps_sweep:
        _ensure_base_model_once()
        metrics = {"norm_results": {}, "auto_eps": dict(auto_eps or {})}
        shared_indices = None
        for norm, eps in list(norm_eps_sweep):
            case_params = _merged(
                base_params,
                {
                    "attack_overrides": {"norm": norm, "eps": eps},
                    "attack_eval_indices": shared_indices,
                    "train": False,
                    "train_if_missing": False,
                },
            )
            case_out = None if output_dir is None else Path(output_dir) / f"norm_{_norm_label(norm)}"
            payload, run_params, auto_eps_value = _run_case(case_params, case_out, root_output_dir=output_dir)
            summary = payload.get("attack_summary", {})
            if shared_indices is None:
                eval_idx_art = payload.get("artifacts", {}).get("attack_eval_indices")
                if output_dir is not None and eval_idx_art is not None:
                    eval_idx_path = Path(output_dir) / eval_idx_art
                    if eval_idx_path.exists():
                        shared_indices = np.load(eval_idx_path).astype(int).tolist()
            pgd = summary.get("results", {}).get("pgd", {})
            metrics["norm_results"][_norm_label(norm)] = {
                "norm": norm,
                "eps": auto_eps_value if auto_eps_value is not None else run_params.get("attack_overrides", {}).get("eps"),
                "hom_pgd_success_rate": payload.get("objective"),
                "pgd_success_rate": pgd.get("success_rate"),
                "selection_mode": summary.get("selection_mode"),
                "selection_size": summary.get("selection_size"),
                "output_dir": None if case_out is None else str(case_out),
            }

        artifacts = {}
        if output_dir is not None:
            artifact_root = ensure_dir(Path(output_dir) / "artifacts")
            summary_path = artifact_root / "adversarial_norm_sweep_summary.json"
            _save_json(summary_path, metrics)
            artifacts["norm_sweep_summary"] = str(summary_path.relative_to(Path(output_dir)))
        hom_vals = [
            value["hom_pgd_success_rate"]
            for value in metrics["norm_results"].values()
            if value["hom_pgd_success_rate"] is not None
        ]
        return {
            "objective": (sum(hom_vals) / len(hom_vals)) if hom_vals else None,
            "feasible": None,
            "metrics": metrics,
            "artifacts": artifacts,
        }

    single_out = Path(output_dir) / f"norm_{_norm_label(dict(attack_overrides or {}).get('norm', 'unknown'))}" if (output_dir is not None and single_case_subdir_by_norm) else output_dir
    payload, run_params, auto_eps_value = _run_case(base_params, single_out, root_output_dir=output_dir)
    payload.setdefault("metrics", {})
    payload["metrics"]["effective_attack_overrides"] = dict(run_params.get("attack_overrides", {}))
    payload["metrics"]["effective_auto_eps"] = {
        "enabled": bool(dict(auto_eps or {}).get("enabled", False)),
        "force_override": bool(dict(auto_eps or {}).get("force_override", False)),
        "applied": auto_eps_value is not None,
        "eps": auto_eps_value,
    }
    payload["metrics"]["single_run_output_dir"] = None if single_out is None else str(single_out)
    if auto_eps_value is not None:
        payload["metrics"]["auto_eps"] = {
            "enabled": True,
            "eps": auto_eps_value,
            "norm": run_params.get("attack_overrides", {}).get("norm"),
        }
    return payload


__all__ = ["adversarial_attack_experiment"]
