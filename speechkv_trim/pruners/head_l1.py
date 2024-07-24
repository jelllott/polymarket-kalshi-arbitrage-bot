"""Head-level pruner. Drops whole attention heads per layer by L1 norm."""
from __future__ import annotations
import torch
from .base import BasePruner, PruneMeta


class HeadL1(BasePruner):
    """Score heads by sum of |K| + |V| on the current cache and drop the
    lowest-norm ones. The 'budget' here is # heads to keep, not # tokens.
    """
    name = "head_l1"

    def __init__(self, budget: int, **kw):
        super().__init__(budget, **kw)

    def score(self, k, v, attn_weights, meta: PruneMeta):
        # k, v: [b, h, s, d]
        norms = k.abs().sum(dim=(0, 2, 3)) + v.abs().sum(dim=(0, 2, 3))
        return norms  # [heads]

    def select(self, scores, budget):
        if scores.numel() <= budget:
            return torch.arange(scores.numel(), device=scores.device)
        _, idx = torch.topk(scores, budget)
        idx, _ = torch.sort(idx)
        return idx

    def apply(self, k, v, kept_idx):
        # index along head dim
        return k.index_select(1, kept_idx), v.index_select(1, kept_idx)
