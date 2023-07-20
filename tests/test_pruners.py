import torch
from speechkv_trim.pruners.token_attn_recency import TokenAttnRecency
from speechkv_trim.pruners.base import PruneMeta

def test_recency_basic():
    k = torch.randn(1, 4, 32, 8)
    v = torch.randn(1, 4, 32, 8)
    pr = TokenAttnRecency(budget=8)
    meta = PruneMeta(layer_idx=0, step=0, audio_len=20, text_len=12)
    nk, nv = pr(k, v, None, meta)
    assert nk.shape[2] == 8
