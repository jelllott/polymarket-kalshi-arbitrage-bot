"""Chunk-level pruner gated by Silero-VAD boundaries.

We chunk the audio prefix into VAD-detected segments, score each chunk
by its mean received attention, and drop low-score chunks whole.
"""
from __future__ import annotations
import torch
from .base import BasePruner, PruneMeta


class ChunkVAD(BasePruner):
    name = "chunk_vad"

    def __init__(self, budget: int, chunk_size: int = 50,
                 frame_hz: int = 50, **kw):
        super().__init__(budget, **kw)
        # default 50hz tokenizer -> 1s chunks at chunk_size=50
        self.chunk_size = int(chunk_size)
        self.frame_hz = int(frame_hz)

    def _vad_chunks(self, seq_len, audio_len):
        # FIXME: this stubs out actual silero-vad; the real impl in
        # utils.audio.run_silero_vad() returns frame indices. For now we
        # just use uniform chunks of self.chunk_size frames.
        chunks = []
        i = 0
        while i < audio_len:
            j = min(i + self.chunk_size, audio_len)
            chunks.append((i, j))
            i = j
        # text-side: each text token is its own chunk
        for t in range(audio_len, seq_len):
            chunks.append((t, t + 1))
        return chunks

    def score(self, k, v, attn_weights, meta: PruneMeta):
        seq = k.shape[2]
        device = k.device
        audio_len = meta.audio_len
        chunks = self._vad_chunks(seq, audio_len)

        if attn_weights is None:
            tok = k.float().pow(2).sum(dim=-1).mean(dim=(0, 1))
        else:
            tok = attn_weights.float().mean(dim=(0, 1, 2))

        scores = torch.zeros(seq, device=device)
        for (lo, hi) in chunks:
            mean_attn = tok[lo:hi].mean()
            scores[lo:hi] = mean_attn
        return scores
