"""KV-cache surgery primitives.

We don't subclass HF's `DynamicCache` because the layouts differ between
backbones (Qwen2-Audio uses `past_key_values` tuples, SALMONN wraps in
its own container). Instead we expose two helpers that accept a list of
per-layer (k, v) tensors and a per-layer kept-index tensor.
"""
from __future__ import annotations
import torch


def gather_cache(past_key_values, kept_idx_per_layer):
    """Apply per-layer kept indices to a HF-style past_key_values tuple."""
    out = []
    for layer_idx, (k, v) in enumerate(past_key_values):
        idx = kept_idx_per_layer[layer_idx]
        out.append((k.index_select(2, idx), v.index_select(2, idx)))
    return tuple(out)


def cache_size_bytes(past_key_values) -> int:
    total = 0
    for k, v in past_key_values:
        total += k.element_size() * k.numel()
        total += v.element_size() * v.numel()
    return total


def cache_summary(past_key_values) -> dict:
    """Quick stats dict for logging."""
    seqs = [k.shape[2] for k, _ in past_key_values]
    return {
        "layers": len(past_key_values),
        "min_seq": min(seqs),
        "max_seq": max(seqs),
        "mean_seq": sum(seqs) / len(seqs),
        "bytes": cache_size_bytes(past_key_values),
    }
