"""Evaluation entrypoint for adversarial attacks."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .attack import PGDAttack
from .common import _load_model_for_dataset, _save_checkpoint_artifact, _save_json, _select_device
from .config import _build_attack_weight_matrix, _normalized_attack_config
from .data import get_model_checkpoint, load_data
from .selection import _filter_correctly_classified_indices, find_accurately_classified_samples
from homopt.experiments.common.io import ensure_dir
from homopt.utils import set_global_seed


def evaluate_adversarial_attacks(
    dataset_name="mnist",
    data_root="data",
    checkpoint_dir="data",
    attack_overrides=None,
    selection_size=1000,
    eval_indices=None,
    batch_size=None,
    seed=42,
    device=None,
    download=True,
    output_dir=None,
):
    set_global_seed(seed)
    device = _select_device(device)
    _, test_dataset, dataset_config = load_data(dataset_name, data_root=data_root, download=download)
    checkpoint_path = get_model_checkpoint(dataset_name, checkpoint_dir=checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")

    model = _load_model_for_dataset(dataset_name, dataset_config, checkpoint_path, device)
    if eval_indices is not None:
        selected = _filter_correctly_classified_indices(
            model,
            test_dataset,
            eval_indices,
            device=device,
            batch_size=int(batch_size or dataset_config.get("batch_size", 256)),
        )
        if len(selected) == 0:
            raise RuntimeError(
                "Provided eval_indices has no correctly classified samples in dataset range."
            )
        chosen_indices = np.asarray(selected, dtype=int)
        chosen_count = int(len(chosen_indices))
        selection_mode = "fixed_indices_correct_only"
    else:
        accurate_indices = find_accurately_classified_samples(model, test_dataset)
        chosen_count = min(int(selection_size), len(accurate_indices))
        if chosen_count == 0:
            raise RuntimeError("No accurately classified samples found for adversarial evaluation.")
        chosen_indices = np.random.choice(accurate_indices, chosen_count, replace=False)
        selection_mode = "random_correct_subset"
    subset = Subset(test_dataset, indices=chosen_indices.tolist())
    eval_batch_size = min(chosen_count, int(batch_size or dataset_config.get("batch_size", 256)))
    test_loader = DataLoader(subset, batch_size=eval_batch_size, shuffle=False)

    attack_cfg = _normalized_attack_config(dataset_config, attack_overrides=attack_overrides)
    feature_dim = dataset_config["input_channels"] * (dataset_config["image_size"] ** 2)
    weight_matrix = _build_attack_weight_matrix(
        feature_dim,
        attack_cfg,
        device,
        input_channels=dataset_config["input_channels"],
        image_size=dataset_config["image_size"],
    )

    results = {}
    for name, hom in (("pgd", False), ("hom_pgd", True)):
        attacker = PGDAttack(model, device, hom=hom)
        start = perf_counter()
        clean_accuracy, adv_accuracy, success_rate, perturb_detail = attacker.evaluate_attack_success(
            model,
            test_loader,
            eps=attack_cfg["eps"],
            alpha=attack_cfg["alpha"],
            iters=int(attack_cfg["iters"]),
            norm_type=attack_cfg["norm"],
            W=weight_matrix,
            optimizer_name=attack_cfg["optimizer"],
            return_details=True,
        )
        elapsed = perf_counter() - start
        results[name] = {
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "success_rate": success_rate,
            "per_iter_sec": elapsed / max(int(attack_cfg["iters"]), 1),
            "perturbation": perturb_detail,
        }

    summary = {
        "dataset": dataset_name,
        "selection_size": chosen_count,
        "selection_mode": selection_mode,
        "attack_config": {
            "eps": float(attack_cfg["eps"]),
            "alpha": float(attack_cfg["alpha"]),
            "iters": attack_cfg["iters"],
            "norm": "inf" if attack_cfg["norm"] == np.inf else float(attack_cfg["norm"]),
            "weighted_norm": bool(attack_cfg.get("weighted_norm", False)),
            "weight_mode": attack_cfg.get("weight_mode", "random_rank1"),
            "weight_scale": float(attack_cfg.get("weight_scale", 0.01)),
            "optimizer": attack_cfg.get("optimizer", "gd"),
        },
        "results": results,
    }
    artifacts = {}
    if output_dir is not None:
        artifact_root = ensure_dir(Path(output_dir) / "artifacts")
        eval_indices_path = artifact_root / f"{dataset_name}_attack_eval_indices.npy"
        np.save(eval_indices_path, chosen_indices.astype(np.int64))
        artifacts["attack_eval_indices"] = str(eval_indices_path.relative_to(Path(output_dir)))
        summary["eval_indices_artifact"] = artifacts["attack_eval_indices"]
        summary_path = artifact_root / f"{dataset_name}_attack_summary.json"
        _save_json(summary_path, summary)
        artifacts["attack_summary"] = str(summary_path.relative_to(Path(output_dir)))
    artifacts.update(_save_checkpoint_artifact(checkpoint_path, output_dir, dataset_name))
    return summary, artifacts


__all__ = ["evaluate_adversarial_attacks"]
