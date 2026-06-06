"""Robust JSON extraction from LLM responses.

LLMs vary in how they wrap JSON output: bare object, fenced markdown,
trailing prose, multiple objects. This module extracts the *first*
syntactically-complete top-level object and parses it.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.MULTILINE)


def _find_first_object(text: str) -> str | None:
    """Return the substring of text that is the first balanced {...} block.

    Walks the string with a brace-depth counter, ignoring braces inside strings.
    Returns None if no balanced object is found.
    """
    depth = 0
    start: int | None = None
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def extract_json(text: str | None) -> dict[str, Any] | None:
    """Extract and parse the first JSON object from an LLM response.

    Tries fenced code blocks first, then falls back to scanning the raw text.
    Returns the parsed dict, or None if nothing parses.
    """
    if not text:
        return None

    # 1) Look inside markdown fences first
    fence = _FENCE_RE.search(text)
    candidate = fence.group(1) if fence else text

    obj = _find_first_object(candidate) or _find_first_object(text)
    if obj is None:
        return None

    try:
        parsed = json.loads(obj)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed
