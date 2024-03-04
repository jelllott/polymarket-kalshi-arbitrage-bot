"""Streaming anchor evictor.

Incremental cache eviction during autoregressive decode. Anchors are
tokens at sentence boundaries on the *text* side and at VAD-detected
voice onsets on the *audio* side. We never evict anchors.
"""
from __future__ import annotations
import torch
from .base import BasePruner, PruneMeta


class StreamingAnchor(BasePruner):
    name = "streaming_anchor"

    def __init__(self, budget: int, anchor_keep: int = 32,
                 window: int = 256, **kw):
        super().__init__(budget, **kw)
        self.anchor_keep = int(anchor_keep)
        self.window = int(window)
        self._anchors: set[int] = set()

    def mark_anchor(self, idx: int):
        self._anchors.add(int(idx))

    def score(self, k, v, attn_weights, meta: PruneMeta):
        seq = k.shape[2]
        device = k.device
        recency = torch.linspace(0.0, 1.0, seq, device=device)
        # anchors get max score
        for a in self._anchors:
            if 0 <= a < seq:
                recency[a] = 2.0
        # sliding window: anything within last `self.window` tokens kept
        recency[-self.window:] = torch.maximum(
            recency[-self.window:],
            torch.full((min(self.window, seq),), 1.5, device=device),
        )
        return recency
