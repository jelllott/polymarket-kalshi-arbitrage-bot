"""Wrapper around Qwen2-Audio for pruned-cache inference."""
from __future__ import annotations
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def load_qwen2_audio(name: str = "Qwen/Qwen2-Audio-7B-Instruct",
                    dtype: torch.dtype = torch.bfloat16,
                    device: str = "cuda"):
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map=device,
    )
    model.eval()
    return proc, model


@torch.no_grad()
def encode_prompt(proc, model, audio, text: str, sr: int = 16000):
    """Build inputs for Qwen2-Audio with audio + text prompt."""
    msgs = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "in-memory"},
            {"type": "text", "text": text},
        ]},
    ]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                      tokenize=False)
    inputs = proc(text=prompt, audios=[audio], sampling_rate=sr,
                  return_tensors="pt").to(model.device)
    return inputs
