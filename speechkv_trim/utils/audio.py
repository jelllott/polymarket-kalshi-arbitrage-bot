"""Audio loading + simple VAD wrapper."""
from __future__ import annotations
import os
import numpy as np
import torch


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load mono float32 audio resampled to target_sr."""
    try:
        import soundfile as sf
        wav, sr = sf.read(path, always_2d=False)
    except Exception:
        import librosa
        wav, sr = librosa.load(path, sr=None, mono=True)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def run_silero_vad(wav: np.ndarray, sr: int = 16000,
                   threshold: float = 0.5) -> list[tuple[float, float]]:
    """Return list of (start_sec, end_sec) speech regions."""
    # NOTE: requires torch hub silero-vad to be cached. We don't fail
    # hard if it's missing, just return a single full-audio segment.
    try:
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad",
            trust_repo=True, verbose=False,
        )
        get_speech_timestamps = utils[0]
        tens = torch.from_numpy(wav)
        ts = get_speech_timestamps(tens, model, sampling_rate=sr,
                                   threshold=threshold)
        return [(t["start"] / sr, t["end"] / sr) for t in ts]
    except Exception as e:
        # silently fall back; the eval scripts log this once per run
        return [(0.0, len(wav) / sr)]
