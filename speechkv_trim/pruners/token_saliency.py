"""Token-level pruner with a tiny acoustic-saliency head on top of recency."""
from __future__ import annotations
import torch
import torch.nn as nn
from .base import BasePruner, PruneMeta


class _SaliencyHead(nn.Module):
    """Linear -> GELU -> Linear, scores each KV slot.

    Trained offline on forced-aligned silence/voice labels — see
    scripts/train_saliency_head.py.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, k):
        # k: [batch, heads, seq, head_dim] -> pool over heads
        x = k.mean(dim=1)  # [batch, seq, dim]
        return self.net(x).squeeze(-1)  # [batch, seq]


class TokenSaliency(BasePruner):
    name = "token_saliency"

    def __init__(self, budget: int, head_path: str = None,
                 alpha: float = 0.5, beta: float = 0.3, **kw):
        super().__init__(budget, **kw)
        self.alpha = float(alpha)  # attention weight
        self.beta = float(beta)    # saliency weight
        # remaining (1-alpha-beta) goes to recency
        self.head_path = head_path
        self._head = None  # lazy

    def _maybe_load(self, dim: int, device):
        if self._head is None:
            self._head = _SaliencyHead(dim).to(device)
            if self.head_path:
                sd = torch.load(self.head_path, map_location=device)
                self._head.load_state_dict(sd)
            self._head.eval()

    @torch.no_grad()
    def score(self, k, v, attn_weights, meta: PruneMeta):
        seq = k.shape[2]
        dim = k.shape[-1]
        device = k.device
        self._maybe_load(dim, device)

        if attn_weights is None:
            attn = k.float().pow(2).sum(dim=-1).mean(dim=(0, 1))
        else:
            attn = attn_weights.float().mean(dim=(0, 1, 2))
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        sal = self._head(k).mean(dim=0)  # [seq]
        sal = torch.sigmoid(sal)

        recency = torch.linspace(0.0, 1.0, seq, device=device)
        gamma = max(0.0, 1.0 - self.alpha - self.beta)
        score = self.alpha * attn + self.beta * sal + gamma * recency

        # always keep audio prefix anchors + last few text tokens
        score[:4] = 1.0
        score[-8:] = 1.0
        # TODO: also protect tokens around detected sentence boundaries
        return score
