import torch
import pytest
from speechkv_trim.pruners.base import BasePruner, PruneMeta
from speechkv_trim.pruners.token_attn_recency import TokenAttnRecency
from speechkv_trim.pruners.head_l1 import HeadL1


def _fake_kv(seq=64, heads=4, d=16):
    k = torch.randn(1, heads, seq, d)
    v = torch.randn(1, heads, seq, d)
    return k, v


def test_recency_keeps_anchors_and_tail():
    k, v = _fake_kv(seq=64)
    pr = TokenAttnRecency(budget=16)
    meta = PruneMeta(layer_idx=0, step=0, audio_len=40, text_len=24)
    nk, nv = pr(k, v, None, meta)
    assert nk.shape[2] == 16
    assert nv.shape[2] == 16


def test_recency_no_op_when_budget_exceeds_seq():
    k, v = _fake_kv(seq=8)
    pr = TokenAttnRecency(budget=64)
    meta = PruneMeta(layer_idx=0, step=0, audio_len=4, text_len=4)
    nk, nv = pr(k, v, None, meta)
    assert nk.shape[2] == 8


def test_head_l1_keeps_top_heads():
    k, v = _fake_kv(seq=10, heads=8)
    pr = HeadL1(budget=4)
    meta = PruneMeta(layer_idx=0, step=0, audio_len=6, text_len=4)
    nk, nv = pr(k, v, None, meta)
    assert nk.shape[1] == 4
    assert nv.shape[1] == 4
