"""Sample-selection helpers for adversarial attack experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def find_lowest_confidence_accurate_samples(model, test_dataset, num_samples=100):
    device = next(model.parameters()).device
    model.eval()
    accurate_samples = []
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, true_label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            if predicted_label == true_label:
                accurate_samples.append((idx, float(torch.max(probabilities).item()), true_label))
    accurate_samples.sort(key=lambda item: item[1])
    return [idx for idx, _, _ in accurate_samples[:num_samples]]


def find_accurately_classified_samples(model, test_dataset):
    device = next(model.parameters()).device
    model.eval()
    accurate_indices = []
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, true_label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            predicted_label = torch.argmax(output, dim=1).item()
            if predicted_label == true_label:
                accurate_indices.append(int(idx))
    return accurate_indices


def _filter_correctly_classified_indices(
    model: nn.Module,
    dataset,
    indices,
    device: torch.device,
    batch_size: int = 256,
):
    selected = [int(i) for i in list(indices)]
    selected = [i for i in selected if 0 <= i < len(dataset)]
    if len(selected) == 0:
        return []
    subset = Subset(dataset, selected)
    loader = DataLoader(subset, batch_size=max(int(batch_size), 1), shuffle=False)
    model.eval()
    correct = []
    cursor = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predicted = torch.argmax(model(inputs), dim=1)
            mask = predicted.eq(labels).detach().cpu().tolist()
            batch_indices = selected[cursor: cursor + len(mask)]
            correct.extend(idx for idx, keep in zip(batch_indices, mask) if bool(keep))
            cursor += len(mask)
    return correct


__all__ = [
    "_filter_correctly_classified_indices",
    "find_accurately_classified_samples",
    "find_lowest_confidence_accurate_samples",
]
