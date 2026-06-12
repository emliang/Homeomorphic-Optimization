"""Training entrypoint for adversarial experiment models."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from torch.utils.data import DataLoader

from .common import _save_checkpoint_artifact, _save_json, _select_device
from .data import get_model_checkpoint, load_data
from .model import ModelTrainer, build_model
from homopt.utils import ensure_dir, set_global_seed


def train_adversarial_model(
    dataset_name="mnist",
    data_root="data",
    checkpoint_dir="data",
    training_epochs=10,
    training_lr=None,
    batch_size=None,
    seed=42,
    device=None,
    download=True,
    output_dir=None,
    verbose=False,
    log_every=1,
):
    set_global_seed(seed)
    device = _select_device(device)
    train_dataset, test_dataset, dataset_config = load_data(dataset_name, data_root=data_root, download=download)
    train_defaults = dict(dataset_config.get("training_defaults", {}))
    batch_size = int(batch_size or train_defaults.get("batch_size", dataset_config.get("batch_size", 256)))
    epochs = int(training_epochs or train_defaults.get("epochs", 10))
    learning_rate = float(training_lr or train_defaults.get("learning_rate", 1e-3))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    checkpoint_path = get_model_checkpoint(dataset_name, checkpoint_dir=checkpoint_dir)
    model = build_model(dataset_config).to(device)
    trainer = ModelTrainer(model, device, checkpoint_path, learning_rate=learning_rate)

    start = perf_counter()
    trainer.train(train_loader, epochs=epochs, verbose=bool(verbose), log_every=log_every)
    test_accuracy = trainer.evaluate(test_loader)
    elapsed = perf_counter() - start

    summary = {
        "dataset": dataset_name,
        "test_accuracy": test_accuracy,
        "training_epochs": epochs,
        "training_lr": learning_rate,
        "batch_size": batch_size,
        "checkpoint": str(checkpoint_path),
        "elapsed_sec": elapsed,
        "verbose": bool(verbose),
        "log_every": int(max(int(log_every), 1)),
    }
    artifact_root = ensure_dir(Path(output_dir) / "artifacts") if output_dir is not None else None
    artifacts = {}
    if artifact_root is not None:
        summary_path = artifact_root / f"{dataset_name}_training_summary.json"
        _save_json(summary_path, summary)
        artifacts["training_summary"] = str(summary_path.relative_to(Path(output_dir)))
    artifacts.update(_save_checkpoint_artifact(checkpoint_path, output_dir, dataset_name))
    return summary, artifacts


__all__ = ["train_adversarial_model"]
