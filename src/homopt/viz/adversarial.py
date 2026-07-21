"""Visualization entrypoint for adversarial attack examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from homopt.learning.adversarial.attack import PGDAttack
from homopt.learning.adversarial.common import _load_model_for_dataset, _save_checkpoint_artifact, _select_device
from homopt.learning.adversarial.config import _build_attack_weight_matrix, _normalized_attack_config
from homopt.learning.adversarial.data import get_model_checkpoint, load_data
from homopt.learning.adversarial.selection import _filter_correctly_classified_indices
from homopt.utils import set_global_seed
from homopt.utils.io import ensure_dir


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib. Install with: pip install -r requirements.txt"
        ) from exc
    return plt


def _render_attack_figure(
    images,
    labels,
    clean_pred,
    adv_images,
    adv_pred,
    title_prefix: str,
    rows: int = 3,
    class_names=None,
):
    plt = _require_matplotlib()
    # Paper-friendly typography (larger labels for readability after scaling in manuscripts).
    title_fs = 15
    label_fs = 14
    row_fs = 13
    cbar_label_fs = 12
    cbar_tick_fs = 11
    rows = max(int(rows), 1)
    sample_count = int(images.shape[0])
    cmap = "gray" if images.shape[1] == 1 else None
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(9.2, 2.8 * rows),
        gridspec_kw={"wspace": 0.03, "hspace": 0.12},
    )
    if rows == 1:
        axes = axes.reshape(1, -1)
    col_titles = ["Original", "Attack", "Perturbation"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=title_fs, fontweight="bold")
    def _name(v):
        idx = int(v)
        if class_names is None:
            return str(idx)
        if 0 <= idx < len(class_names):
            return str(class_names[idx])
        return str(idx)

    for idx in range(rows):
        if idx >= sample_count:
            for col in range(3):
                axes[idx, col].axis("off")
            continue
        image = images[idx].detach().cpu()
        adv_image = adv_images[idx].detach().cpu()
        image = image[0] if images.shape[1] == 1 else image.permute(1, 2, 0)
        adv_image = adv_image[0] if images.shape[1] == 1 else adv_image.permute(1, 2, 0)
        perturbation = (adv_images[idx].detach().cpu() - images[idx].detach().cpu()).abs()
        perturbation = perturbation[0] if images.shape[1] == 1 else perturbation.mean(dim=0)
        axes[idx, 0].imshow(image, cmap=cmap)
        axes[idx, 1].imshow(adv_image, cmap=cmap)
        pert_img = axes[idx, 2].imshow(perturbation, cmap="hot")
        true_name = _name(labels[idx])
        clean_name = _name(clean_pred[idx])
        adv_name = _name(adv_pred[idx])
        axes[idx, 0].set_xlabel(f"true: {true_name}", fontsize=label_fs)
        axes[idx, 1].set_xlabel(f"pred: {adv_name}", fontsize=label_fs)
        axes[idx, 2].set_xlabel("|delta|", fontsize=label_fs)
        axes[idx, 0].set_ylabel(f"#{idx + 1}", rotation=0, labelpad=10, fontsize=row_fs)
        cbar = fig.colorbar(pert_img, ax=axes[idx, 2], fraction=0.046, pad=0.01)
        cbar.ax.tick_params(labelsize=cbar_tick_fs, length=2)
        cbar.set_label("intensity", fontsize=cbar_label_fs)
        for col in range(3):
            axes[idx, col].set_xticks([])
            axes[idx, col].set_yticks([])
            for spine in axes[idx, col].spines.values():
                spine.set_visible(False)
    fig.subplots_adjust(top=0.98, bottom=0.06)
    return fig, plt


def visualize_adversarial_attacks(
    dataset_name="mnist",
    data_root="data",
    checkpoint_dir="data",
    attack_overrides=None,
    eval_indices=None,
    num_examples=5,
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
    display_rows = max(int(num_examples), 1)
    target_examples = display_rows
    candidate_count = min(max(target_examples * 20, 64), len(test_dataset))
    if candidate_count == 0:
        raise RuntimeError("No samples available for visualization.")

    if eval_indices is not None:
        selected = _filter_correctly_classified_indices(
            model,
            test_dataset,
            eval_indices,
            device=device,
            batch_size=dataset_config.get("batch_size", 256),
        )
        if len(selected) == 0:
            raise RuntimeError(
                "Provided eval_indices has no correctly classified samples in dataset range."
            )
        indices = np.asarray(selected[:candidate_count], dtype=int)
    else:
        candidate_indices = np.random.choice(len(test_dataset), candidate_count, replace=False)
        correct_indices = _filter_correctly_classified_indices(
            model,
            test_dataset,
            candidate_indices,
            device=device,
            batch_size=dataset_config.get("batch_size", 256),
        )
        if len(correct_indices) == 0:
            raise RuntimeError("No correctly classified samples available for visualization.")
        indices = np.asarray(correct_indices[:candidate_count], dtype=int)
    images = torch.stack([test_dataset[i][0] for i in indices])
    labels = torch.tensor([test_dataset[i][1] for i in indices])

    attack_cfg = _normalized_attack_config(dataset_config, attack_overrides=attack_overrides)
    feature_dim = dataset_config["input_channels"] * (dataset_config["image_size"] ** 2)
    weight_matrix = _build_attack_weight_matrix(
        feature_dim,
        attack_cfg,
        device,
        input_channels=dataset_config["input_channels"],
        image_size=dataset_config["image_size"],
    )

    iterations = min(attack_cfg["iters"], 200)
    with torch.no_grad():
        clean_pred = torch.argmax(model(images.to(device)), dim=1).cpu()
    visualizations = {}
    artifacts = {}
    for attack_name, hom in (("pgd", False), ("hom_pgd", True)):
        attacker = PGDAttack(model, device, hom=hom)
        adv_images = attacker.generate(
            images.to(device),
            labels.to(device),
            eps=attack_cfg["eps"],
            alpha=attack_cfg["alpha"],
            iters=iterations,
            norm=attack_cfg["norm"],
            W=weight_matrix,
            optimizer_name=attack_cfg["optimizer"],
        ).cpu()
        with torch.no_grad():
            adv_pred = torch.argmax(model(adv_images.to(device)), dim=1).cpu()
        success_mask = (clean_pred == labels) & (adv_pred != labels)
        success_indices = torch.where(success_mask)[0][:display_rows]

        success_images = images[success_indices]
        success_labels = labels[success_indices]
        success_clean_pred = clean_pred[success_indices]
        success_adv_images = adv_images[success_indices]
        success_adv_pred = adv_pred[success_indices]

        fig, plt = _render_attack_figure(
            success_images,
            success_labels,
            success_clean_pred,
            success_adv_images,
            success_adv_pred,
            attack_name.upper(),
            rows=display_rows,
            class_names=dataset_config.get("class_names"),
        )
        visualizations[attack_name] = {
            "successful_examples": int(success_indices.numel()),
            "clean_predictions": [int(value) for value in success_clean_pred.tolist()],
            "adversarial_predictions": [int(value) for value in success_adv_pred.tolist()],
        }
        if output_dir is not None:
            artifact_root = ensure_dir(Path(output_dir) / "artifacts")
            fig_path = artifact_root / f"{dataset_name}_{attack_name}_examples.pdf"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            artifacts[f"{attack_name}_visualization"] = str(fig_path.relative_to(Path(output_dir)))
        plt.close(fig)

    artifacts.update(_save_checkpoint_artifact(checkpoint_path, output_dir, dataset_name))
    return {
        "dataset": dataset_name,
        "num_examples": display_rows,
        "candidate_pool": candidate_count,
        "visualizations": visualizations,
    }, artifacts


__all__ = ["visualize_adversarial_attacks"]
