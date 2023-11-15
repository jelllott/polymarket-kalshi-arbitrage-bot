"""GigaSpeech test eval (subset M & L)."""
from __future__ import annotations
from .librispeech_long import load_manifest
from .metrics import wer


def evaluate(manifest_path: str, infer_fn, max_items=None):
    items = load_manifest(manifest_path)
    if max_items:
        items = items[:max_items]
    refs, hyps = [], []
    for it in items:
        hyp = infer_fn(it["audio"], it["prompt"])
        refs.append(it["ref"])
        hyps.append(hyp)
    return {"wer": wer(refs, hyps), "n": len(items)}
