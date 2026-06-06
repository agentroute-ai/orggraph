"""Shared text-overlap metrics (SQuAD-style F1) used by RQ2/RQ3 evaluators."""

from __future__ import annotations

import re
from collections import Counter

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    """SQuAD-style token-overlap F1 between two strings."""
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    p = n_same / len(pred_toks)
    r = n_same / len(gold_toks)
    return 2 * p * r / (p + r)


def best_f1(pred: str, gold: str, alternates: list[str] | None = None) -> float:
    """Max F1 between *pred* and *gold* or any of *alternates*."""
    options = [gold] + list(alternates or [])
    return max(f1_score(pred, o) for o in options)
