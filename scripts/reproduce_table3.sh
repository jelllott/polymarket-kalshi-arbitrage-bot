#!/usr/bin/env bash
# Reproduce paper Table 3 (GigaSpeech subset M).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2-Audio-7B-Instruct}"
MANIFEST="${MANIFEST:-data/manifests/gigaspeech_test_m.jsonl}"
OUTDIR="${OUTDIR:-out/table3}"
mkdir -p "$OUTDIR"

for pruner in token_attn_recency token_saliency streaming_anchor; do
  for budget in 1024 2048; do
    python -m speechkv_trim.cli eval --manifest "$MANIFEST" \
      --pruner "$pruner" --budget "$budget" > "$OUTDIR/${pruner}_b${budget}.json"
  done
done
