from pathlib import Path

import torch
from torch.utils.data import Dataset

import homopt.experiments.adversarial.evaluation as adversarial_evaluation
import homopt.experiments.adversarial.workflow as adversarial_workflow
import homopt.learning.training.adversarial as adversarial_training


class _ConstantImageDataset(Dataset):
    def __init__(self, size, label=0):
        self.images = torch.zeros(size, 1, 28, 28)
        self.labels = torch.full((size,), label, dtype=torch.long)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx].item())


def test_adversarial_attack_workflow_trains_and_writes_summary(tmp_path, monkeypatch):
    train_dataset = _ConstantImageDataset(8)
    test_dataset = _ConstantImageDataset(8)

    monkeypatch.setattr(
        adversarial_training,
        "load_data",
        lambda dataset_name, data_root="data", download=True: (
            train_dataset,
            test_dataset,
            {
                "input_channels": 1,
                "image_size": 28,
                "num_classes": 10,
                "batch_size": 4,
                "class_names": [str(i) for i in range(10)],
                "attack_defaults": {"eps": 0.1, "alpha": 0.05, "iters": 2, "norm": 2},
            },
        ),
    )
    monkeypatch.setattr(
        adversarial_evaluation,
        "load_data",
        lambda dataset_name, data_root="data", download=True: (
            train_dataset,
            test_dataset,
            {
                "input_channels": 1,
                "image_size": 28,
                "num_classes": 10,
                "batch_size": 4,
                "class_names": [str(i) for i in range(10)],
                "attack_defaults": {"eps": 0.1, "alpha": 0.05, "iters": 2, "norm": 2},
            },
        ),
    )
    monkeypatch.setattr(
        adversarial_workflow,
        "get_model_checkpoint",
        lambda dataset_name, checkpoint_dir="data": Path(tmp_path) / f"{dataset_name}_model.pth",
    )
    monkeypatch.setattr(
        adversarial_training,
        "get_model_checkpoint",
        lambda dataset_name, checkpoint_dir="data": Path(tmp_path) / f"{dataset_name}_model.pth",
    )
    monkeypatch.setattr(
        adversarial_evaluation,
        "get_model_checkpoint",
        lambda dataset_name, checkpoint_dir="data": Path(tmp_path) / f"{dataset_name}_model.pth",
    )
    monkeypatch.setattr(
        adversarial_evaluation,
        "find_accurately_classified_samples",
        lambda model, dataset, max_samples=None: list(range(len(dataset))),
    )

    result = adversarial_workflow.adversarial_attack_workflow(
        dataset_name="mnist",
        train=True,
        attack=True,
        visualize=False,
        training_epochs=1,
        batch_size=4,
        attack_selection_size=4,
        attack_overrides={"eps": 0.1, "alpha": 0.05, "iters": 2, "norm": 2},
        seed=1,
        device="cpu",
        download=False,
        output_dir=tmp_path,
    )

    assert result["dataset"] == "mnist"
    assert "training" in result
    assert "attack_summary" in result
    assert "checkpoint" in result["artifacts"]
    assert result["artifacts"]["checkpoint"] == "artifacts/mnist_model.pth"
    assert (tmp_path / result["artifacts"]["checkpoint"]).exists()
    assert (tmp_path / result["artifacts"]["attack_summary"]).exists()


def test_adversarial_entry_defaults_are_self_starting():
    script_text = Path("scripts/hom_pgd/run_adversarial_attack.py").read_text()
    assert '"train_if_missing": True' in script_text
