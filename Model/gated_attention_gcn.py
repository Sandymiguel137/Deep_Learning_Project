from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def complex_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.relu(x.real), torch.relu(x.imag))


class ComplexChebTemporalConv(nn.Module):
    """Complex Chebyshev graph convolution with a short temporal filter.

    Input shape is ``(batch, nodes, time, in_channels)`` and output shape is
    ``(batch, nodes, time, out_channels)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cheb_order: int = 6,
        temporal_order: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if cheb_order < 1:
            raise ValueError("cheb_order must be >= 1")
        if temporal_order < 1:
            raise ValueError("temporal_order must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cheb_order = cheb_order
        self.temporal_order = temporal_order
        weight_shape = (cheb_order, temporal_order, in_channels, out_channels)
        scale = 1.0 / math.sqrt(max(1, cheb_order * temporal_order * in_channels))
        self.weight_real = nn.Parameter(torch.empty(weight_shape).uniform_(-scale, scale))
        self.weight_imag = nn.Parameter(torch.empty(weight_shape).uniform_(-scale, scale))
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    @property
    def weight(self) -> torch.Tensor:
        return torch.complex(self.weight_real, self.weight_imag)

    @property
    def bias(self) -> torch.Tensor | None:
        if self.bias_real is None or self.bias_imag is None:
            return None
        return torch.complex(self.bias_real, self.bias_imag)

    def _cheb_polynomials(self, x: torch.Tensor, gso: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        polynomials = [x]
        if self.cheb_order == 1:
            return tuple(polynomials)
        polynomials.append(torch.einsum("nm,bmtc->bntc", gso, x))
        for order in range(2, self.cheb_order):
            next_poly = 2 * torch.einsum("nm,bmtc->bntc", gso, polynomials[-1]) - polynomials[-2]
            polynomials.append(next_poly)
        return tuple(polynomials)

    @staticmethod
    def _lag(x: torch.Tensor, steps: int) -> torch.Tensor:
        if steps == 0:
            return x
        padded = torch.zeros_like(x)
        padded[:, :, steps:, :] = x[:, :, :-steps, :]
        return padded

    def forward(self, x: torch.Tensor, gso: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (batch,nodes,time,channels), got {tuple(x.shape)}")
        if not torch.is_complex(x):
            x = x.to(torch.complex64)
        if not torch.is_complex(gso):
            gso = gso.to(torch.complex64)

        output = None
        for tau in range(self.temporal_order):
            x_tau = self._lag(x, tau)
            for k, poly in enumerate(self._cheb_polynomials(x_tau, gso)):
                term = torch.einsum("bnti,io->bnto", poly, self.weight[k, tau])
                output = term if output is None else output + term

        assert output is not None
        if self.bias is not None:
            output = output + self.bias.view(1, 1, 1, -1)
        return output


class NodeTimeChannelGate(nn.Module):
    """Outer-product gated attention over node, time, and channel axes."""

    def __init__(self, num_nodes: int, num_timesteps: int, num_channels: int) -> None:
        super().__init__()
        self.node_gate = nn.Linear(num_channels, num_nodes)
        self.time_gate = nn.Linear(num_channels, num_timesteps)
        self.channel_gate = nn.Linear(num_channels, num_channels)

    def forward(self, h: torch.Tensor, return_gates: bool = False):
        if h.ndim != 4:
            raise ValueError(f"Expected h with shape (batch,nodes,time,channels), got {tuple(h.shape)}")
        context = h.abs().mean(dim=(1, 2))
        node = torch.sigmoid(self.node_gate(context))
        time = torch.sigmoid(self.time_gate(context))
        channel = torch.sigmoid(self.channel_gate(context))
        scale = 1.0 + (
            node[:, :, None, None]
            * time[:, None, :, None]
            * channel[:, None, None, :]
        )
        gated = h * scale.to(dtype=h.real.dtype)
        if return_gates:
            return gated, {"node": node, "time": time, "channel": channel, "scale": scale}
        return gated


class GatedAttentionGCN(nn.Module):
    """Complex spatio-temporal ChebNet with the paper's gated-attention module."""

    def __init__(
        self,
        num_nodes: int = 156,
        num_timesteps: int = 20,
        in_channels: int = 1,
        hidden_channels: int = 10,
        cheb_order: int = 6,
        temporal_order: int = 3,
        num_layers: int = 2,
        fc_width: int = 512,
        num_classes: int = 3,
        dropout: float = 0.2,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.use_attention = use_attention
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(
                ComplexChebTemporalConv(
                    current_channels,
                    hidden_channels,
                    cheb_order=cheb_order,
                    temporal_order=temporal_order,
                )
            )
            current_channels = hidden_channels
        self.backbone = nn.ModuleList(layers)
        self.attention = NodeTimeChannelGate(num_nodes, num_timesteps, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        flat_dim = 2 * num_nodes * num_timesteps * hidden_channels
        self.fc = nn.Linear(flat_dim, fc_width)
        self.classifier = nn.Linear(fc_width, num_classes)

    def forward(self, x: torch.Tensor, gso: torch.Tensor, return_attention: bool = False):
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.shape[1] != self.num_nodes or x.shape[2] != self.num_timesteps:
            raise ValueError(
                f"Expected input shape (batch,{self.num_nodes},{self.num_timesteps}), got {tuple(x.shape)}"
            )
        if not torch.is_complex(x):
            x = x.to(torch.complex64)

        h = x
        for layer in self.backbone:
            h = complex_relu(layer(h, gso))
            real = self.dropout(h.real)
            imag = self.dropout(h.imag)
            h = torch.complex(real, imag)

        attention_info = None
        if self.use_attention:
            if return_attention:
                h, attention_info = self.attention(h, return_gates=True)
            else:
                h = self.attention(h)

        features = torch.cat([h.real.flatten(start_dim=1), h.imag.flatten(start_dim=1)], dim=1)
        features = self.dropout(torch.relu(self.fc(features)))
        logits = self.classifier(features)
        if return_attention:
            return logits, attention_info
        return logits
