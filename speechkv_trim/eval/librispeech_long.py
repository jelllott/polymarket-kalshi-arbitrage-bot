"""LibriSpeech-long eval harness.

We concatenate multi-utterance segments from the same speaker to get
inputs in the 30s-180s range. Metric is WER on the final segment.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import torch
from .metrics import wer


def load_manifest(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def evaluate(manifest_path: str, infer_fn, max_items: int | None = None):
    items = load_manifest(manifest_path)
    if max_items is not None:
        items = items[:max_items]
    refs, hyps = [], []
    for it in items:
        hyp = infer_fn(it["audio"], it["prompt"])
        refs.append(it["ref"])
        hyps.append(hyp)
    return {"wer": wer(refs, hyps), "n": len(items)}
