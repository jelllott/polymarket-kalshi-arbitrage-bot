# Data layout

We don't ship audio. The eval scripts expect a JSONL manifest with the
following fields per line:

```json
{"audio": "/path/to/file.wav",
 "prompt": "Transcribe the audio.",
 "ref": "the ground truth transcript",
 "id": "ls-1234-0001-0042"}
```

For the spoken QA split, swap `prompt` for `question` and `ref` for
`answer`.

## LibriSpeech-long

Generated from LibriSpeech `test-clean` / `test-other` by concatenating
consecutive utterances from the same speaker until total duration is in
[30, 180] seconds. The script is `scripts/make_librispeech_long.py`. We
include the resulting manifest under `data/manifests/` (audio paths are
relative to `$LIBRISPEECH_ROOT`).

## GigaSpeech

Use the official subset M / L test split. We use `gigaspeech` from the
HuggingFace datasets hub; the manifest builder dumps a local copy so we
don't re-stream during eval.

## In-house spoken QA

Internal split, not redistributable. The questions cover entity,
yes/no, and short-answer types over 60-120s news clips. Roughly 1.2k
items.
