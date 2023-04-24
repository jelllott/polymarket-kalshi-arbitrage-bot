import torch
from .base import BasePruner

class TokenAttnRecency(BasePruner):
    def __call__(self, k, v, attn, meta):
        seq = k.shape[2]
        keep = min(self.budget, seq)
        return k[:, :, -keep:], v[:, :, -keep:]
