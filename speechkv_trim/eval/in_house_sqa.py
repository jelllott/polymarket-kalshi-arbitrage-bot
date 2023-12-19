"""In-house spoken QA split."""
from __future__ import annotations
from .librispeech_long import load_manifest
from .metrics import exact_match, f1_token


def evaluate(manifest_path: str, infer_fn, max_items=None):
    items = load_manifest(manifest_path)
    if max_items:
        items = items[:max_items]
    em, f1 = 0.0, 0.0
    for it in items:
        hyp = infer_fn(it["audio"], it["question"])
        em += exact_match(hyp, it["answer"])
        f1 += f1_token(hyp, it["answer"])
    n = max(len(items), 1)
    return {"exact_match": em / n, "f1": f1 / n, "n": len(items)}
