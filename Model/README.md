# Gated-Attention GCN

This folder contains the course-project model for the boundary-stress dataset in
`Results_Course`.

The model accepts complex voltage windows with shape `(batch, 156, 20)` and
predicts three classes:

- `0`: normal
- `1`: FDI attack
- `2`: cyber-physical inverter attack

The generated labels are multi-label (`30` physical inverter labels followed by
`120` FDI sensor labels).  The training script maps them to the three-class task.
When both physical and FDI attacks are active, the default policy is
`--joint-policy physical`, meaning the cyber-physical class takes priority.

## Train

```bash
python Model/train.py \
  --data-dir /home/sa165267/Desktop/AttackJuly14_Paper/Results_Course
```

Defaults match the report setup: `25,000` train samples, `4,040` validation
samples, `6,000` test samples, batch size `200`, `50` epochs, Chebyshev order
`6`, two graph-convolution layers, hidden channel count `10`, and a `512`-wide
fully connected layer.

Outputs are saved by default to:

```text
Model/runs/gated_attention/
```

To train the un-gated baseline with the same backbone:

```bash
python /Model/train.py --no-attention
```
