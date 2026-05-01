from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


JointPolicy = Literal["physical", "fdi", "drop"]


@dataclass(frozen=True)
class DatasetPaths:
    data_dir: Path
    voltage_file: str = "VphasorFDI_boundary_stress.npy"
    label_file: str = "AttackLabelFDI_boundary_stress.npy"
    gso_file: str = "Y_norm_sparse.npy"

    @property
    def voltage_path(self) -> Path:
        return self.data_dir / self.voltage_file

    @property
    def label_path(self) -> Path:
        return self.data_dir / self.label_file

    @property
    def gso_path(self) -> Path:
        return self.data_dir / self.gso_file


def labels_to_three_class(
    labels: np.ndarray,
    num_physical: int = 30,
    joint_policy: JointPolicy = "physical",
) -> Tuple[np.ndarray, np.ndarray]:
    """Map multi-label physical/FDI targets to normal/FDI/cyber classes.

    The generated label matrix has physical inverter labels first and FDI sensor
    labels second.  The paper is a three-class task, so joint windows need a
    deterministic mapping.  By default, physical/cyber labels take priority
    because the mitigation action is physical isolation.
    """
    physical = labels[:, :num_physical].any(axis=1)
    fdi = labels[:, num_physical:].any(axis=1) if labels.shape[1] > num_physical else np.zeros_like(physical)
    keep = np.ones(labels.shape[0], dtype=bool)
    y = np.zeros(labels.shape[0], dtype=np.int64)

    if joint_policy == "drop":
        keep = ~(physical & fdi)
        y[fdi] = 1
        y[physical] = 2
    elif joint_policy == "fdi":
        y[physical] = 2
        y[fdi] = 1
    elif joint_policy == "physical":
        y[fdi] = 1
        y[physical] = 2
    else:
        raise ValueError(f"Unknown joint policy: {joint_policy}")

    return y, keep


class PhasorAttackDataset(Dataset):
    """Memory-mapped dataset for complex voltage windows and 3-class labels."""

    def __init__(
        self,
        voltage_path: Path,
        label_path: Path,
        indices: np.ndarray,
        num_physical: int = 30,
        joint_policy: JointPolicy = "physical",
        normalize: bool = True,
        mean: complex | None = None,
        scale: float | None = None,
    ) -> None:
        self.voltage = np.load(voltage_path, mmap_mode="r")
        raw_labels = np.load(label_path, mmap_mode="r")
        y, keep = labels_to_three_class(raw_labels, num_physical, joint_policy)
        filtered_indices = np.asarray(indices, dtype=np.int64)
        filtered_indices = filtered_indices[keep[filtered_indices]]
        self.indices = filtered_indices
        self.labels = y
        self.normalize = normalize
        self.mean = mean if mean is not None else 0.0 + 0.0j
        self.scale = scale if scale is not None else 1.0

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        x = np.asarray(self.voltage[idx], dtype=np.complex64)
        if self.normalize:
            x = (x - self.mean) / max(self.scale, 1e-8)
        return torch.from_numpy(x), torch.tensor(self.labels[idx], dtype=torch.long)


def compute_complex_normalization(voltage_path: Path, indices: np.ndarray, max_samples: int = 2048) -> Tuple[complex, float]:
    voltage = np.load(voltage_path, mmap_mode="r")
    sample_indices = np.asarray(indices[: min(len(indices), max_samples)], dtype=np.int64)
    chunk = np.asarray(voltage[sample_indices], dtype=np.complex64)
    mean = complex(chunk.mean())
    scale = float(np.abs(chunk - mean).std())
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return mean, scale


def split_indices(
    total: int,
    train_size: int = 25_000,
    val_size: int = 4_040,
    test_size: int = 6_000,
    seed: int = 100,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_size + val_size + test_size > total:
        raise ValueError(
            f"Requested split {train_size + val_size + test_size} exceeds dataset length {total}."
        )
    indices = np.arange(total, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    train = indices[:train_size]
    val = indices[train_size:train_size + val_size]
    test = indices[train_size + val_size:train_size + val_size + test_size]
    return train, val, test


def load_gso(gso_path: Path, device: torch.device) -> torch.Tensor:
    loaded = np.load(gso_path, allow_pickle=True)
    gso_obj = loaded.item() if loaded.shape == () else loaded
    if sp.issparse(gso_obj):
        dense = gso_obj.toarray()
    else:
        dense = np.asarray(gso_obj)
    dense = dense.astype(np.complex64, copy=False)
    return torch.from_numpy(dense).to(device=device, dtype=torch.complex64)


def class_counts(label_path: Path, indices: np.ndarray, joint_policy: JointPolicy = "physical") -> np.ndarray:
    labels = np.load(label_path, mmap_mode="r")
    y, keep = labels_to_three_class(labels, joint_policy=joint_policy)
    usable = np.asarray(indices, dtype=np.int64)
    usable = usable[keep[usable]]
    return np.bincount(y[usable], minlength=3)


def class_weights(counts: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = counts.astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)
