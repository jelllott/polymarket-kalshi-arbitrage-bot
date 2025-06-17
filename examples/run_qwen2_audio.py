"""Minimal example: prune Qwen2-Audio KV cache during long-form ASR.

This is a stub; real wiring lives in scripts/reproduce_table2.sh.
TODO: turn this into a proper end-to-end runnable example.
"""
from speechkv_trim.pruners.token_attn_recency import TokenAttnRecency
from speechkv_trim.models.qwen2_audio import load_qwen2_audio

def main():
    proc, model = load_qwen2_audio()
    pr = TokenAttnRecency(budget=1024)
    print("loaded:", model.__class__.__name__, "pruner:", pr.name)

if __name__ == "__main__":
    main()
