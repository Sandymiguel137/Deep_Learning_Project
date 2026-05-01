# Gated-Attention GCN for SMART-DS Anomaly Classification

This repository contains a reproducible PyTorch implementation for the paper:

**Gated-Attention Graph Convolutional Networks with Boundary-Stress Hard-Sample Mining for Three-Class Anomaly Classification on Spatio-Temporal Graph Signals**

The code trains graph, recurrent, and fully-connected baselines on the SMART-DS/OpenDSS data already present in this directory. It uses the large NumPy arrays in place, so the repository does not duplicate the dataset.

## What Is Included

- Complex-valued Chebyshev spatio-temporal GCN backbone.
- Residual node/time/channel gated-attention module.
- FCNN, RNN, LSTM, un-gated GCN, and gated GCN baselines.
- Dataset utilities for converting node-level physical and FDI labels into the paper's classification labels.
- Training, evaluation, ablation, and figure-generation scripts.
- LaTeX paper source under `paper/`.

## Data Expected In This Directory

The default commands use these existing files:

- `Vphasor_FDI_Physical_WithVoltageNoise.npy`: complex voltage windows, shape `(35040, 156, 20)`.
- `AttackLabel_FDI_Physical_WithVoltageNoise.npy`: combined labels, shape `(35040, 150)`.
- `Y_norm_sparse.npy`: sparse complex graph shift operator.
- `metadata_run_config.npy`: bus/sensor metadata.

The first 30 label columns are physical/control anomaly indicators. The remaining 120 columns are FDI/measurement anomaly indicators.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For GPU training, install a CUDA-enabled PyTorch build before `pip install -e .` if needed.

## Quick Smoke Test

```bash
python -m gagcn.cli.inspect_data
python -m gagcn.cli.train --model gated_gcn --epochs 1 --max-samples 512 --batch-size 64 --output-dir runs/smoke
python -m gagcn.cli.evaluate --checkpoint runs/smoke/best.pt --output-dir runs/smoke_eval
```

## Reproduce Main Experiments

Train the proposed model:

```bash
python -m gagcn.cli.train --model gated_gcn --epochs 50 --batch-size 200 --output-dir runs/gated_gcn
```

Train baselines:

```bash
for model in fcnn rnn lstm gcn; do
  python -m gagcn.cli.train --model "$model" --epochs 50 --batch-size 200 --output-dir "runs/$model"
done
```

Evaluate a checkpoint:

```bash
python -m gagcn.cli.evaluate --checkpoint runs/gated_gcn/best.pt --output-dir runs/gated_gcn_eval
```

Run gate-axis ablations:

```bash
for ablation in none no_spatial no_temporal no_channel; do
  python -m gagcn.cli.train --model gated_gcn --gate-ablation "$ablation" --epochs 50 --batch-size 200 --output-dir "runs/gated_${ablation}"
done
```

Generate paper figures from a completed run:

```bash
python -m gagcn.cli.make_figures --run-dir runs/gated_gcn --output-dir figures
```

## Notes

- By default, training uses `four_class` label mode, which keeps joint FDI+physical examples as their own class for reporting while still exposing the anomaly regimes used in the paper.
- Use `--label-mode three_class` if you want strict three-class training. In that mode, joint samples are assigned to Type-B/physical by default to avoid dropping ambiguous cases.
- Results depend on random seed, package versions, and whether the full dataset is used. The paper tables can be regenerated from `metrics.json` files emitted by the training and evaluation scripts.
