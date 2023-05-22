"""Base pruner."""
import torch

class BasePruner:
    def __init__(self, budget):
        self.budget = int(budget)
    def score(self, k, v, attn, meta):
        raise NotImplementedError
    def select(self, scores, budget):
        _, idx = torch.topk(scores, budget)
        idx, _ = torch.sort(idx)
        return idx
    def apply(self, k, v, idx):
        return k.index_select(2, idx), v.index_select(2, idx)
    def __call__(self, k, v, attn, meta):
        s = self.score(k, v, attn, meta)
        i = self.select(s, self.budget)
        return self.apply(k, v, i)
