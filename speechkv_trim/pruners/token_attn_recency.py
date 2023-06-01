import torch
from .base import BasePruner

class TokenAttnRecency(BasePruner):
    def __init__(self, budget, alpha=0.7):
        super().__init__(budget)
        self.alpha = alpha
    def score(self, k, v, attn, meta):
        seq = k.shape[2]
        recency = torch.linspace(0, 1, seq, device=k.device)
        score = recency
        score[:4] = 1.0
        score[-8:] = 1.0
        return score
