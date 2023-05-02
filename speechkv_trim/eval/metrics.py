"""Eval metrics: WER, exact-match, token F1."""
from __future__ import annotations
import re
from collections import Counter


_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 \u4e00-\u9fff]+", " ", s)
    s = _WS.sub(" ", s)
    return s


def wer(refs: list[str], hyps: list[str]) -> float:
    total_words, total_errs = 0, 0
    for r, h in zip(refs, hyps):
        rw = _normalize(r).split()
        hw = _normalize(h).split()
        total_words += len(rw)
        total_errs += _edit_distance(rw, hw)
    return total_errs / max(total_words, 1)


def _edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[m]


def exact_match(hyp: str, ref: str) -> float:
    return 1.0 if _normalize(hyp) == _normalize(ref) else 0.0


def f1_token(hyp: str, ref: str) -> float:
    h = _normalize(hyp).split()
    r = _normalize(ref).split()
    if not h or not r:
        return 0.0
    common = Counter(h) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p = same / len(h)
    rc = same / len(r)
    return 2 * p * rc / (p + rc)
