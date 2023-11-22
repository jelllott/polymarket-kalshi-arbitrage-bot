# Hush KV: Speech-Aware KV Cache Pruning

> Companion code for our group's ongoing work on long-context speech LLMs.
> Most decoder-only speech LLMs blow up the KV cache once you push past
> ~30s of audio prefix. We look at *what* lives in those caches and how to
> drop it without hurting downstream ASR / SQA performance.

This repo contains the reference implementation for the "Hush KV" pruning
family (token-level + head-level + chunk-level) and the small evaluation
harness we use on LibriSpeech-long, GigaSpeech-test, and an in-house
spoken QA split. The code is organized so each pruner is a single file
under `speechkv_trim/pruners/` and is selectable by name from the CLI.

## TL;DR

Speech tokenization is *dense*. With a typical 50 Hz audio tokenizer, 60s
of audio is 3000 tokens before the text prompt even starts. Naive sliding
window cache eviction drops too many "anchor" frames (silence + sentence
boundaries) and hurts disfluency / repetition handling. We score keys
using a weighted mix of attention recency and a small acoustic-saliency
head, then evict per-layer with a budget.

You should be able to reproduce the LibriSpeech-long numbers in the paper
draft (`docs/draft.md`) on a single A100-40G. The GigaSpeech runs need
2x A100s for the larger backbones.

## Why "Hush KV"?

Because most of the speech KV cache is silence, breathing, and "uhm".
We hush it.

## Status

| component               | state         | notes                                |
|-------------------------|---------------|---------------------------------------|
| token-level pruner      | stable        | default in CLI                        |
| head-level pruner       | stable        |                                       |
| chunk-level pruner      | beta          | needs better boundary detector        |
| streaming evictor       | experimental  | works for Qwen2-Audio, breaks on SALMONN sometimes |
| eval harness            | stable        | LibriSpeech-long, GigaSpeech, in-house |
| paper                   | drafting      | see `docs/draft.md`                   |

## Install

```bash
git clone https://github.com/jelllott/speechkv-trim
cd speechkv-trim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Tested with python 3.10 / 3.11. Requires `torch>=2.1`, `transformers>=4.40`,
`torchaudio>=2.1`, `soundfile`, and `librosa` for the eval scripts. The
streaming evictor uses `flash-attn>=2.5` if it's installed and falls back
to SDPA otherwise (slower, slightly different numbers).

## Quickstart

Run token-level pruning on a single audio file with a Qwen2-Audio
backbone:

```bash
python -m speechkv_trim.cli prune \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --audio examples/long_lecture.wav \
    --pruner token_attn_recency \
    --budget 1024 \
    --out out/transcript.json
```

Reproduce the paper Table 2 (LibriSpeech-long, all pruners):

```bash
bash scripts/reproduce_table2.sh
```

This will download the prepared metadata (~6 MB) and stream audio from a
local LibriSpeech mirror. See `docs/data.md` for the expected layout.

## Pruners

All pruners implement the `BasePruner` interface in
`speechkv_trim/pruners/base.py`:

```python
class BasePruner:
    def score(self, k, v, attn_weights, meta): ...
    def select(self, scores, budget): ...
    def apply(self, k, v, kept_idx): ...
```

Currently shipped:

- `token_attn_recency` — weighted mix of mean attention received and
  position recency. Cheap, surprisingly strong baseline.
- `token_saliency` — adds a small acoustic-saliency head (trained on
  forced-aligned silence/voice labels) to the recency score.
- `head_l1` — drops *whole heads* per layer based on attention L1 norm.
  Combines well with token-level pruning.
- `chunk_vad` — chunk-level pruning gated by Silero-VAD boundaries.
- `streaming_anchor` — incremental, anchored on sentence-start tokens
  from the text side. Use with `--stream`.

You can register your own pruner via the `speechkv_trim.pruners` entry
point group (see `pyproject.toml`).

## Repo layout

```
speechkv_trim/
  cli.py                 # argparse front end
  models/                # backbone wrappers
    qwen2_audio.py
    salmonn.py
    whisper_llm.py       # ASR-only baseline
  pruners/
    base.py
    token_attn_recency.py
    token_saliency.py
    head_l1.py
    chunk_vad.py
    streaming_anchor.py
  eval/
    librispeech_long.py
    gigaspeech.py
    in_house_sqa.py
    metrics.py
  utils/
    audio.py
    cache.py             # KV-cache surgery primitives
    profile.py
docs/
  draft.md
  data.md
  pruners.md
scripts/
  reproduce_table2.sh
  reproduce_table3.sh
tests/
```

## Reproducibility

We seed `torch`, `numpy`, and `random` from the `--seed` flag; results
should be bit-identical with `flash-attn` off. With `flash-attn` on you
get small ULP-level deltas — the paper numbers are reported with
`flash-attn` on because that's what we actually use.

## Citing

If this code is useful for your work, please cite the (in-progress)
preprint — see `docs/draft.md` for the current bibtex stub. We'll update
this section once the arXiv id is live.

## License

Apache-2.0. See `LICENSE`.

## Acknowledgements

Thanks to the WUT MIIT group for compute, and to the SALMONN authors
for releasing checkpoints under a permissive license.
