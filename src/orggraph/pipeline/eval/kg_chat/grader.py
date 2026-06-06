"""Deterministic graders for the KG-Chat structural eval.

Each function returns ``(pass: bool, reason: str)``. The ``judge`` kind
returns ``(None, "needs_llm_judge")`` so the runner can dispatch to
``judge.py`` for that question.
"""

from __future__ import annotations


def grade(answer: str, grading: dict) -> tuple[bool | None, str]:
    """Dispatch on grading.kind. Returns (pass, reason)."""
    kind = grading.get("kind")
    if kind == "contains":
        return _grade_contains(answer, grading)
    if kind == "integer":
        return _grade_integer(answer, grading)
    if kind == "numeric":
        return _grade_numeric(answer, grading)
    if kind == "name_in_set":
        return _grade_name_in_set(answer, grading)
    if kind == "judge":
        return None, "needs_llm_judge"
    return False, f"unknown grading kind: {kind!r}"


def _grade_contains(answer: str, g: dict) -> tuple[bool, str]:
    expected = str(g["value"]).strip().lower()
    found = expected in (answer or "").lower()
    return found, f"contains({g['value']!r})" if found else f"missing({g['value']!r})"


# Recognise ASCII hyphen, Unicode minus (U+2212), en-dash, em-dash, and
# the Arabic minus as leading-negative markers — LLM outputs frequently
# substitute these for ASCII hyphen, especially on small floats.
_MINUS_CHARS = "-−–—➖"
_NUM_RE = None  # lazy-compiled


def _normalise_minus(text: str) -> str:
    """Map any unicode minus glyph onto ASCII hyphen so regex sees a sign."""
    if not text:
        return text
    return text.translate({ord(ch): "-" for ch in "−–—➖"})


def _grade_integer(answer: str, g: dict) -> tuple[bool, str]:
    import re
    global _NUM_RE
    if _NUM_RE is None:
        _NUM_RE = re.compile(r"-?\d[\d,]*")
    target = int(g["value"])
    tolerance = int(g.get("tolerance", 0))
    text = _normalise_minus(answer or "")
    found_ints = [int(m.replace(",", "")) for m in _NUM_RE.findall(text)]
    for v in found_ints:
        if abs(v - target) <= tolerance:
            return True, f"integer {v} within ±{tolerance} of {target}"
    return False, f"no integer within ±{tolerance} of {target} in {found_ints[:5]}"


def _grade_numeric(answer: str, g: dict) -> tuple[bool, str]:
    """Match a numeric target with tolerance, considering percentage form.

    When the target is in [0, 1] (fraction), any number followed by '%' in
    the answer is also tried as that number divided by 100 (e.g., '68.2%'
    becomes 0.682). LLM outputs frequently substitute Unicode minus glyphs
    (U+2212, en-dash, em-dash, Arabic minus) for ASCII hyphen on negative
    values; those are normalised to '-' before extraction so the sign is
    preserved.
    """
    import re
    target = float(g["value"])
    tolerance = float(g["tolerance"])
    text = _normalise_minus(answer or "")
    # First pass: raw numbers as written.
    raw_matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    raw_nums = [float(m) for m in raw_matches]
    for v in raw_nums:
        if abs(v - target) <= tolerance:
            return True, f"numeric {v} within ±{tolerance} of {target}"
    # Second pass: if target is a fraction in [0, 1], any "N%" counts as N/100.
    if 0.0 <= target <= 1.0:
        pct_matches = re.findall(r"(-?\d+(?:\.\d+)?)\s*%", text)
        pct_nums = [float(m) / 100.0 for m in pct_matches]
        for v in pct_nums:
            if abs(v - target) <= tolerance:
                return True, f"percentage {v * 100:.1f}% (={v}) within ±{tolerance} of {target}"
    return False, f"no numeric within ±{tolerance} of {target} in {raw_nums[:5]}"


def _grade_name_in_set(answer: str, g: dict) -> tuple[bool, str]:
    """Pass if at least ``k`` names from the set appear in the answer.

    Names are matched case-insensitively. Last-name-only matches count
    (e.g., 'Lavorato' matches 'John Lavorato').
    """
    names: list[str] = list(g["names"])
    k = int(g["k"])
    ans_lower = (answer or "").lower()
    hits = []
    for name in names:
        full = name.lower()
        # try full name, then last word (handles single-name mentions)
        if full in ans_lower:
            hits.append(name)
            continue
        last = name.split()[-1].lower()
        if len(last) >= 4 and last in ans_lower:
            hits.append(name)
    return len(hits) >= k, f"matched {len(hits)}/{k} required from {hits}"
