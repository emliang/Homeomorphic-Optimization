"""Workflow orchestration for adversarial attack experiments."""

from __future__ import annotations

from .data import get_model_checkpoint
from .evaluation import evaluate_adversarial_attacks
from .training import train_adversarial_model
from .visualization import visualize_adversarial_attacks


def adversarial_attack_workflow(
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
    seed=42,
    device=None,
    dtype=None,
    download=True,
    output_dir=None,
    verbose=False,
    train_log_every=1,
):
    """Run the package-native adversarial workflow for the old 1.3 family."""
    del dtype

    if run_all:
        train = attack = visualize = True
    if not any([train, attack, visualize]):
        raise ValueError("At least one of train, attack, visualize, or run_all must be enabled.")

    dataset_name = dataset_name.lower()
    checkpoint_path = get_model_checkpoint(dataset_name, checkpoint_dir=checkpoint_dir)
    artifacts = {}
    metrics = {
        "dataset": dataset_name,
        "train": bool(train),
        "attack": bool(attack),
        "visualize": bool(visualize),
        "checkpoint": str(checkpoint_path),
    }

    def _run_training():
        return train_adversarial_model(
            dataset_name=dataset_name,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            training_epochs=training_epochs,
            training_lr=training_lr,
            batch_size=batch_size,
            seed=seed,
            device=device,
            download=download,
            output_dir=output_dir,
            verbose=verbose,
            log_every=train_log_every,
        )

    if train:
        train_summary, train_artifacts = _run_training()
        metrics["training"] = train_summary
        artifacts.update(train_artifacts)

    checkpoint_exists = checkpoint_path.exists()
    if (attack or visualize) and not checkpoint_exists:
        if train_if_missing:
            train_summary, train_artifacts = _run_training()
            metrics["training"] = train_summary
            artifacts.update(train_artifacts)
        else:
            raise FileNotFoundError(f"Checkpoint not found for attack workflow: {checkpoint_path}")

    if attack:
        attack_summary, attack_artifacts = evaluate_adversarial_attacks(
            dataset_name=dataset_name,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            attack_overrides=attack_overrides,
            selection_size=attack_selection_size,
            eval_indices=attack_eval_indices,
            batch_size=batch_size,
            seed=seed,
            device=device,
            download=download,
            output_dir=output_dir,
        )
        metrics["attack_summary"] = attack_summary
        artifacts.update(attack_artifacts)

    if visualize:
        viz_summary, viz_artifacts = visualize_adversarial_attacks(
            dataset_name=dataset_name,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            attack_overrides=attack_overrides,
            eval_indices=visualize_eval_indices,
            num_examples=visualize_examples,
            seed=seed,
            device=device,
            download=download,
            output_dir=output_dir,
        )
        metrics["visualization"] = viz_summary
        artifacts.update(viz_artifacts)

    objective = None
    feasible = None
    if "attack_summary" in metrics:
        objective = metrics["attack_summary"]["results"]["hom_pgd"]["success_rate"]

    return {
        "objective": objective,
        "feasible": feasible,
        "artifacts": artifacts,
        **metrics,
    }


__all__ = ["adversarial_attack_workflow"]
