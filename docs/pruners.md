# Pruners

## token_attn_recency

Cheap baseline. Score is `alpha * normalized_mean_attn + (1-alpha) * recency`.
Defaults to `alpha=0.7`. Anchors (first 4 + last 8 tokens) are pinned.

## token_saliency

Adds a small two-layer MLP head that scores each K slot by predicted
acoustic saliency. The head is trained offline against forced-aligned
silence/voice labels — see `scripts/train_saliency_head.py`. Score is
`alpha * attn + beta * sigmoid(head(k)) + (1-alpha-beta) * recency`.

## head_l1

Operates on whole heads, not tokens. Useful in combination with
token-level pruning: run `head_l1` first to drop low-utility heads per
layer, then run a token pruner on what's left.

## chunk_vad

Chunks the audio prefix by silero-VAD boundaries (or uniform chunks if
silero is unavailable) and drops low-attention chunks whole. Better
than uniform chunking when there's heavy silence.

## streaming_anchor

Incremental evictor for streaming decode. Tracks a small set of anchor
indices (sentence boundaries, voice onsets) that are never evicted.
Sliding window of size `window` is also pinned. Use when running with
`--stream`.
