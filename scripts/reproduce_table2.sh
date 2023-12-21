#!/usr/bin/env bash
# Reproduce paper Table 2 (LibriSpeech-long) for all pruners.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2-Audio-7B-Instruct}"
MANIFEST="${MANIFEST:-data/manifests/librispeech_long.jsonl}"
OUTDIR="${OUTDIR:-out/table2}"

mkdir -p "$OUTDIR"

for pruner in token_attn_recency token_saliency head_l1 chunk_vad; do
  for budget in 512 1024 2048; do
    out="$OUTDIR/${pruner}_b${budget}.json"
    if [[ -f "$out" ]]; then
      echo "[skip] $out"
      continue
    fi
    echo "[run]  $pruner budget=$budget"
    python -m speechkv_trim.cli eval \
      --manifest "$MANIFEST" \
      --pruner "$pruner" \
      --budget "$budget" > "$out"
  done
done

python scripts/aggregate_table2.py "$OUTDIR"
