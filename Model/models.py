from __future__ import annotations

from .gated_attention_gcn import GatedAttentionGCN


class GatedAttentionChebyNet(GatedAttentionGCN):
    """Compatibility name for the course-project gated Chebyshev GCN."""


__all__ = ["GatedAttentionChebyNet", "GatedAttentionGCN"]
