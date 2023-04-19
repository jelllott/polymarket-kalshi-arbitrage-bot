"""Base pruner."""
class BasePruner:
    def __init__(self, budget):
        self.budget = budget
    def __call__(self, k, v, attn, meta):
        raise NotImplementedError
