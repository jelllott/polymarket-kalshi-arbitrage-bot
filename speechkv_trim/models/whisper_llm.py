"""ASR-only baseline wrapper around openai/whisper-large-v3."""
from __future__ import annotations
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration


def load_whisper(name: str = "openai/whisper-large-v3",
                 dtype: torch.dtype = torch.float16,
                 device: str = "cuda"):
    proc = AutoProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(
        name, torch_dtype=dtype,
    ).to(device)
    model.eval()
    return proc, model
