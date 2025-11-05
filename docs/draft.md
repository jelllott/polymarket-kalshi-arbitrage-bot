# Paper draft

(this is a working document, not the published paper)

## Title

Hush KV: Speech-Aware KV Cache Pruning for Long-Form Speech LLMs

## Abstract (current)

Decoder-only speech LLMs concatenate dense audio tokens with text and
quickly run out of KV cache budget on inputs beyond 30 seconds. We
study what survives in the cache during long-form ASR and spoken QA
and propose a family of speech-aware pruners that combine attention
recency with a small acoustic-saliency head. On LibriSpeech-long
(30-180s) and GigaSpeech-test, our token+head combo retains 95% of
the unpruned WER while shrinking the cache by 4x. We release code and
manifests at https://github.com/jelllott/speechkv-trim.

## Sketch of method

* Per-layer token-level scoring: attention-received + recency + saliency
* Per-layer head pruning by L1 norm
* Chunk-VAD pruning as an optional pre-filter
* Streaming anchor evictor for autoregressive decode

## Open questions

* Does the saliency head transfer across backbones? early evidence: no.
* Calibration of alpha/beta across audio durations.
* Interaction with speculative decoding.

## Bibtex (stub)

```
@article{chen2024hushkv,
  title={Hush KV: Speech-Aware KV Cache Pruning for Long-Form Speech LLMs},
  author={Chen, Zhuoran and others},
  year={2024},
  note={in preparation}
}
```

## Streaming results (preliminary)

LibriSpeech-long, streaming_anchor budget=1024:
- WER 6.8 (vs 6.2 unpruned)
- 3.4x cache shrink
