"""Token-level pruner: weighted mix of mean received attention + recency."""
from __future__ import annotations
import torch
from .base import BasePruner, PruneMeta


class TokenAttnRecency(BasePruner):
    name = "token_attn_recency"

    def __init__(self, budget: int, alpha: float = 0.7, **kw):
        super().__init__(budget, **kw)
        # alpha = weight on attention vs recency. tuned on librispeech-long dev
        self.alpha = float(alpha)

    def score(self, k, v, attn_weights, meta: PruneMeta):
        seq = k.shape[2]
        device = k.device
        if attn_weights is None:
            # fall back to magnitude (works ok-ish)
            attn = k.float().pow(2).sum(dim=-1).mean(dim=(0, 1))
        else:
            # attn_weights: [batch, heads, q, k] — average over q & heads & batch
            attn = attn_weights.float().mean(dim=(0, 1, 2))
        # normalize
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        # recency: linear from 0 (oldest) to 1 (newest)
        recency = torch.linspace(0.0, 1.0, seq, device=device)
        score = self.alpha * attn + (1 - self.alpha) * recency
        # protect the first few "anchor" tokens (BOS + system) and last N
        score[:4] = 1.0
        score[-8:] = 1.0
        return score
