from __future__ import annotations

import torch


def calc_accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    return correct / len(labels)


__all__ = ["calc_accuracy"]
