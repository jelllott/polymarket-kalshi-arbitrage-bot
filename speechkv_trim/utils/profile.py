"""Tiny profiler — wraps torch.cuda events for per-step timing."""
from __future__ import annotations
import time, contextlib
import torch


@contextlib.contextmanager
def cuda_timer(label: str, log_to: list | None = None):
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        yield
        ms = (time.perf_counter() - t0) * 1000
    if log_to is not None:
        log_to.append((label, ms))
