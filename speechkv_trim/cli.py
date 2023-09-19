"""speechkv-trim command-line interface.

Subcommands:
  prune    - run a pruner on a single audio file
  eval     - run a pruner across an eval manifest
  cache    - cache-size sanity check (no inference)
"""
from __future__ import annotations
import argparse, json, sys, os
import torch

from .pruners.token_attn_recency import TokenAttnRecency
from .pruners.token_saliency import TokenSaliency
from .pruners.head_l1 import HeadL1
from .pruners.chunk_vad import ChunkVAD
from .pruners.streaming_anchor import StreamingAnchor

PRUNERS = {
    "token_attn_recency": TokenAttnRecency,
    "token_saliency": TokenSaliency,
    "head_l1": HeadL1,
    "chunk_vad": ChunkVAD,
    "streaming_anchor": StreamingAnchor,
}


def _build_pruner(name: str, budget: int, **kw):
    if name not in PRUNERS:
        raise SystemExit(f"unknown pruner: {name}. options: {list(PRUNERS)}")
    return PRUNERS[name](budget=budget, **kw)


def cmd_prune(args):
    pruner = _build_pruner(args.pruner, args.budget)
    print(f"[speechkv-trim] using pruner={pruner.name} budget={pruner.budget}",
          file=sys.stderr)
    # The actual inference path is in models/*.py; here we just sanity-check
    # the pruner can be instantiated. Full inference example in
    # examples/run_qwen2_audio.py.
    out = {"pruner": pruner.name, "budget": pruner.budget, "audio": args.audio}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    else:
        print(json.dumps(out, indent=2))


def cmd_eval(args):
    pruner = _build_pruner(args.pruner, args.budget)
    # we just print the config — full eval driver is in scripts/
    print(json.dumps({"pruner": pruner.name, "budget": pruner.budget,
                      "manifest": args.manifest}))


def cmd_cache(args):
    # Synthesize a fake KV cache and print its size.
    layers, heads, seq, d = args.layers, args.heads, args.seq, args.dim
    cache = []
    for _ in range(layers):
        k = torch.zeros(1, heads, seq, d, dtype=torch.bfloat16)
        v = torch.zeros(1, heads, seq, d, dtype=torch.bfloat16)
        cache.append((k, v))
    from .utils.cache import cache_summary
    print(json.dumps(cache_summary(cache), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser("speechkv-trim")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prune = sub.add_parser("prune")
    p_prune.add_argument("--model", required=True)
    p_prune.add_argument("--audio", required=True)
    p_prune.add_argument("--pruner", required=True)
    p_prune.add_argument("--budget", type=int, default=1024)
    p_prune.add_argument("--out", default=None)
    p_prune.set_defaults(fn=cmd_prune)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--manifest", required=True)
    p_eval.add_argument("--pruner", required=True)
    p_eval.add_argument("--budget", type=int, default=1024)
    p_eval.set_defaults(fn=cmd_eval)

    p_cache = sub.add_parser("cache")
    p_cache.add_argument("--layers", type=int, default=32)
    p_cache.add_argument("--heads", type=int, default=32)
    p_cache.add_argument("--seq", type=int, default=4096)
    p_cache.add_argument("--dim", type=int, default=128)
    p_cache.set_defaults(fn=cmd_cache)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    main()
