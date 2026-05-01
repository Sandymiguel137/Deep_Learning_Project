#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Course_Project.Model.data_utils import (  # type: ignore
        DatasetPaths,
        PhasorAttackDataset,
        class_counts,
        class_weights,
        compute_complex_normalization,
        load_gso,
        split_indices,
    )
    from Course_Project.Model.gated_attention_gcn import GatedAttentionGCN  # type: ignore
else:
    from .data_utils import (
        DatasetPaths,
        PhasorAttackDataset,
        class_counts,
        class_weights,
        compute_complex_normalization,
        load_gso,
        split_indices,
    )
    from .gated_attention_gcn import GatedAttentionGCN


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train the gated-attention GCN on boundary-stress data.")
    parser.add_argument("--data-dir", type=Path, default=root / "Results_Course")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "gated_attention")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-channels", type=int, default=10)
    parser.add_argument("--cheb-order", type=int, default=6)
    parser.add_argument("--temporal-order", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--fc-width", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=0.001)
    parser.add_argument("--train-size", type=int, default=25_000)
    parser.add_argument("--val-size", type=int, default=4_040)
    parser.add_argument("--test-size", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--joint-policy", choices=["physical", "fdi", "drop"], default="physical")
    parser.add_argument("--no-attention", action="store_true", help="Train the un-gated GCN baseline.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_complex_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    noise_real = torch.randn_like(x.real) * (std / np.sqrt(2.0))
    noise_imag = torch.randn_like(x.imag) * (std / np.sqrt(2.0))
    return x + torch.complex(noise_real, noise_imag)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    gso: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    noise_std: float = 0.0,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device=device, dtype=torch.complex64, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        if training:
            x = add_complex_noise(x, noise_std)
            optimizer.zero_grad(set_to_none=True)

        logits = model(x, gso)
        loss = criterion(logits, y)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch = y.numel()
        total_loss += float(loss.detach().item()) * batch
        total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
        total_count += batch

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = DatasetPaths(args.data_dir)
    total = int(np.load(paths.voltage_path, mmap_mode="r").shape[0])
    train_idx, val_idx, test_idx = split_indices(
        total,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    mean, scale = compute_complex_normalization(paths.voltage_path, train_idx)
    train_ds = PhasorAttackDataset(
        paths.voltage_path,
        paths.label_path,
        train_idx,
        joint_policy=args.joint_policy,
        mean=mean,
        scale=scale,
    )
    val_ds = PhasorAttackDataset(
        paths.voltage_path,
        paths.label_path,
        val_idx,
        joint_policy=args.joint_policy,
        mean=mean,
        scale=scale,
    )
    test_ds = PhasorAttackDataset(
        paths.voltage_path,
        paths.label_path,
        test_idx,
        joint_policy=args.joint_policy,
        mean=mean,
        scale=scale,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    gso = load_gso(paths.gso_path, device)
    model = GatedAttentionGCN(
        num_nodes=156,
        num_timesteps=20,
        hidden_channels=args.hidden_channels,
        cheb_order=args.cheb_order,
        temporal_order=args.temporal_order,
        num_layers=args.num_layers,
        fc_width=args.fc_width,
        dropout=args.dropout,
        use_attention=not args.no_attention,
    ).to(device)

    counts = class_counts(paths.label_path, train_idx, joint_policy=args.joint_policy)
    weights = class_weights(counts, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config.update(
        {
            "class_counts": counts.tolist(),
            "class_weights": weights.detach().cpu().tolist(),
            "normalization_mean_real": float(np.real(mean)),
            "normalization_mean_imag": float(np.imag(mean)),
            "normalization_scale": scale,
            "train_samples_after_policy": len(train_ds),
            "val_samples_after_policy": len(val_ds),
            "test_samples_after_policy": len(test_ds),
        }
    )
    with (args.output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, default=str)

    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            gso,
            criterion,
            device,
            optimizer=optimizer,
            noise_std=args.noise_std,
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, gso, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={record['train_loss']:.4f} train_acc={record['train_accuracy']:.4f} "
            f"val_loss={record['val_loss']:.4f} val_acc={record['val_accuracy']:.4f}"
        )
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_accuracy": best_val,
                },
                args.output_dir / "best_model.pt",
            )

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, gso, criterion, device)

    with (args.output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    with (args.output_dir / "test_metrics.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)

    print(
        f"best_val_acc={best_val:.4f} "
        f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f}"
    )
    print(f"saved best model to {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
