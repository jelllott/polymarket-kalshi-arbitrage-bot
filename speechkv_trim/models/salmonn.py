"""SALMONN backbone wrapper.

We rely on the official SALMONN repo being importable as `salmonn`.
"""
from __future__ import annotations
import torch


def load_salmonn(ckpt_path: str, dtype: torch.dtype = torch.bfloat16,
                 device: str = "cuda"):
    try:
        from salmonn.models import SALMONN
    except ImportError as e:
        raise RuntimeError(
            "salmonn package not installed. clone the upstream repo and "
            "`pip install -e .` it before using this backbone."
        ) from e
    m = SALMONN.from_pretrained(ckpt_path)
    m = m.to(device=device, dtype=dtype)
    m.eval()
    return m
