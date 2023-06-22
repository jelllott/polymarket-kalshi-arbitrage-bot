"""Base pruner interface.

All pruners take a per-layer (K, V) pair and produce a kept-index tensor.
The CLI / harness handles batching and dispatching to backbones.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import torch


@dataclass
class PruneMeta:
    layer_idx: int
    step: int
    audio_len: int          # tokens of audio prefix already in cache
    text_len: int           # tokens of text prefix
    is_audio_mask: Optional[torch.Tensor] = None  # bool [seq]
    extra: dict = field(default_factory=dict)


class BasePruner:
    name: str = "base"

    def __init__(self, budget: int, **kw):
        self.budget = int(budget)
        self.kw = kw

    def score(self, k: torch.Tensor, v: torch.Tensor,
              attn_weights: Optional[torch.Tensor],
              meta: PruneMeta) -> torch.Tensor:
        raise NotImplementedError

    def select(self, scores: torch.Tensor, budget: int) -> torch.Tensor:
        # scores: [seq] higher = keep
        if scores.numel() <= budget:
            return torch.arange(scores.numel(), device=scores.device)
        # top-k, then re-sort to preserve position order for cache layout
        _, idx = torch.topk(scores, budget)
        idx, _ = torch.sort(idx)
        return idx

    def apply(self, k: torch.Tensor, v: torch.Tensor,
              kept_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # k, v: [batch, heads, seq, head_dim]
        return k.index_select(2, kept_idx), v.index_select(2, kept_idx)

    def __call__(self, k, v, attn_weights, meta):
        scores = self.score(k, v, attn_weights, meta)
        idx = self.select(scores, self.budget)
        return self.apply(k, v, idx)
