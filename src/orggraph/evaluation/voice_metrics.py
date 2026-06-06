"""Stylometric features for Tier 3 persona voice evaluation.

Pure functions: no LLM calls, no file I/O. Each takes an email body
string and returns a comparable feature value.
"""

from __future__ import annotations

import re
from collections import Counter

_SENTENCE_END = re.compile(r"[.!?]+")
_BLACKBERRY = re.compile(r"sent\s+from\s+my\s+blackberry", re.IGNORECASE)
_FORMAL_SALUTATION_HEADS = ("dear ", "good morning", "good afternoon", "good evening")
_FIRST_NAME_SALUTATION = re.compile(r"^([A-Z][a-z]+),\s*$", re.MULTILINE)


def mean_sentence_length(body: str) -> float:
    """Mean word count per sentence. 0.0 for empty/whitespace input."""
    sentences = [s.strip() for s in _SENTENCE_END.split(body) if s.strip()]
    if not sentences:
        return 0.0
    word_counts = [len(s.split()) for s in sentences]
    return sum(word_counts) / len(word_counts)


def salutation_pattern(body: str) -> str:
    """One of: 'none', 'first_name', 'formal'."""
    head = body[:120].lstrip().lower()
    if any(head.startswith(s) for s in _FORMAL_SALUTATION_HEADS):
        return "formal"
    if _FIRST_NAME_SALUTATION.search(body[:120]):
        return "first_name"
    return "none"


def signoff_pattern(body: str) -> str:
    """One of: 'none', 'dash', 'initials', 'full_name', 'blackberry'."""
    if _BLACKBERRY.search(body):
        return "blackberry"
    lines = [ln.strip() for ln in body.rstrip().split("\n") if ln.strip()]
    if not lines:
        return "none"
    tail = lines[-1]
    if tail in {"-", "--"}:
        return "dash"
    if re.fullmatch(r"[A-Z]{2,4}", tail):
        return "initials"
    if re.fullmatch(r"[A-Z][a-z]+(\s+[A-Z][a-z]+){0,2}", tail):
        return "full_name"
    return "none"


def blackberry_tag_rate(body: str) -> float:
    """1.0 if body contains BlackBerry tag, else 0.0."""
    return 1.0 if _BLACKBERRY.search(body) else 0.0


def top_char_bigrams(body: str, n: int = 10) -> set[str]:
    """Top-n character bigrams by frequency. Lowercased; whitespace folded."""
    s = re.sub(r"\s+", " ", body.lower())
    bigrams = Counter(s[i : i + 2] for i in range(len(s) - 1))
    return {bg for bg, _ in bigrams.most_common(n)}


def char_bigram_jaccard(a: str, b: str, n: int = 10) -> float:
    """Jaccard similarity over top-n character bigrams. 1.0 if both empty."""
    set_a = top_char_bigrams(a, n=n)
    set_b = top_char_bigrams(b, n=n)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
