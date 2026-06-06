"""Compare a generated email against a real ground-truth email.

Two complementary metrics:

* :func:`textual_overlap` — a cheap deterministic Jaccard-on-tokens
  baseline. Doesn't need an LLM. Useful as a sanity floor (a totally
  unrelated reply still gets some shared stopwords, so very low
  overlap is a strong negative signal).

* :func:`judge_continuation_match` — the LLM judge scores the generated
  email against the actual one on three dimensions: ``content_match``
  (does it cover what the real reply covered?), ``style_match`` (does
  it sound like the same author?), and ``intent_match`` (does it move
  the thread forward in the same direction?). Each is 1–5 with a short
  justification. The judge sees the prefix, the actual reply, and the
  generated reply.

The idea is to triangulate: textual_overlap catches gross divergences
cheaply; the LLM judge catches stylistic / intentional differences a
token-overlap metric can't see (and which is exactly the "natural
organisational dialogue" signal RQ2 cares about).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from orggraph.agents.agent import TextChatClient
from orggraph.simulation.transcript import Message

CONTINUATION_DIMENSIONS = ("content_match", "style_match", "intent_match")


# ---------------------------------------------------------------------------
# Cheap textual baseline
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def textual_overlap(generated: str, actual: str) -> dict[str, float]:
    """Return Jaccard token overlap, length ratio, and shared-token count.

    Jaccard is on the set of lowercased alphanumeric tokens. It's a
    coarse measure but stable: identical content scores 1.0, totally
    disjoint scores 0.0, and noise from stopwords washes out at the
    scale we report (3 sig figs).
    """
    g_tokens = _tokens(generated)
    a_tokens = _tokens(actual)
    union = g_tokens | a_tokens
    inter = g_tokens & a_tokens
    jaccard = len(inter) / len(union) if union else 0.0

    # Length comparisons are useful diagnostically — generations that
    # are wildly shorter or longer than the real reply usually fail
    # the style match too.
    ga_chars = max(len(generated or ""), 1)
    ac_chars = max(len(actual or ""), 1)
    length_ratio = ga_chars / ac_chars

    return {
        "jaccard": round(jaccard, 4),
        "shared_tokens": len(inter),
        "generated_tokens": len(g_tokens),
        "actual_tokens": len(a_tokens),
        "length_ratio": round(length_ratio, 3),
        "generated_chars": ga_chars,
        "actual_chars": ac_chars,
    }


# ---------------------------------------------------------------------------
# LLM-judge similarity scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuationScore:
    score: int
    justification: str


@dataclass
class ContinuationVerdict:
    """LLM-judge verdict for one (generated, actual) pair."""

    condition: str
    scores: dict[str, ContinuationScore] = field(default_factory=dict)
    overall_summary: str = ""
    raw_response: str = ""

    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores.values()) / len(self.scores)

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "mean_score": self.mean_score(),
            "scores": {
                k: {"score": v.score, "justification": v.justification}
                for k, v in self.scores.items()
            },
            "overall_summary": self.overall_summary,
        }


_SYSTEM_PROMPT = """You are evaluating whether a model-generated email
plausibly stands in for a real email written by the same person in the
same thread. Score the generated email against the real one on three
dimensions (1–5 each, integers only).

Be conservative. Reward concrete matches; penalise generic plausibility.
Return STRICT JSON. No prose outside the JSON object."""


_RUBRIC = """
Dimensions (1–5 each):

1. content_match: Does the generated email address the same substantive
   points the real reply addressed (decisions, requests, acknowledgements)?
   5: covers the same content with comparable specificity.
   3: covers the topic generically; misses key specifics from the real reply.
   1: addresses something different or contradicts the real reply.

2. style_match: Does the generated reply sound like it was written by the
   same author as the real one — vocabulary, sentence rhythm, formality,
   sign-off conventions?
   5: indistinguishable register; could plausibly come from the same person.
   3: same broad register but noticeable stylistic differences.
   1: clearly a different voice (more formal, more verbose, etc.).

3. intent_match: Does the generated email move the thread in the same
   direction the real reply moved it (advance vs. block vs. ack)?
   5: same forward move and same emotional valence.
   3: same forward move but different emotional/transactional tone.
   1: different intent entirely (e.g. real reply pushed back, generated agreed)."""


_OUTPUT_SCHEMA = """
Return JSON of this exact shape:
{
  "scores": {
    "content_match": {"score": <int 1-5>, "justification": "<one sentence>"},
    "style_match":   {"score": <int 1-5>, "justification": "<one sentence>"},
    "intent_match":  {"score": <int 1-5>, "justification": "<one sentence>"}
  },
  "overall_summary": "<2-3 sentence verdict>"
}"""


def _render_prefix(prefix: list[Message]) -> str:
    if not prefix:
        return "(no prior turns)"
    lines = []
    for m in prefix:
        body = m.body.replace("\n", "\n    ")
        lines.append(f"[turn {m.turn_id}] {m.sender} → {', '.join(m.recipients)}:\n    {body}")
    return "\n\n".join(lines)


def build_continuation_prompt(
    prefix: list[Message],
    actual: Message,
    generated: Message,
    condition_label: str,
) -> tuple[str, str]:
    """Build (system, user) for one (real, generated) comparison."""
    user = (
        f"Prior turns in this real Enron thread:\n\n"
        f"{_render_prefix(prefix)}\n\n"
        f"--- The REAL next email ({actual.sender} → {', '.join(actual.recipients)}) ---\n"
        f"{actual.body}\n\n"
        f"--- The GENERATED next email (condition: {condition_label}) ---\n"
        f"{generated.body}\n\n"
        f"{_RUBRIC}\n\n"
        f"{_OUTPUT_SCHEMA}"
    )
    return _SYSTEM_PROMPT, user


class ContinuationJudgeError(RuntimeError):
    pass


def parse_continuation_response(raw: str, condition: str) -> ContinuationVerdict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            head, rest = text.split("\n", 1)
            if head.lower().strip() in {"json", ""}:
                text = rest

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContinuationJudgeError(
            f"continuation judge response is not valid JSON: {exc}\n--- raw ---\n{raw[:400]}"
        ) from exc

    if not isinstance(payload, dict):
        raise ContinuationJudgeError(
            f"continuation judge response is not a JSON object: {type(payload).__name__}"
        )

    raw_scores = payload.get("scores", {})
    scores: dict[str, ContinuationScore] = {}
    for dim in CONTINUATION_DIMENSIONS:
        entry = raw_scores.get(dim)
        if not isinstance(entry, dict) or "score" not in entry:
            raise ContinuationJudgeError(f"missing dimension {dim!r}")
        try:
            s_int = int(entry["score"])
        except (TypeError, ValueError):
            raise ContinuationJudgeError(f"non-integer score for {dim!r}: {entry['score']!r}")
        if not 1 <= s_int <= 5:
            raise ContinuationJudgeError(f"score for {dim!r} out of [1,5]: {s_int}")
        scores[dim] = ContinuationScore(
            score=s_int,
            justification=str(entry.get("justification", "")).strip(),
        )

    return ContinuationVerdict(
        condition=condition,
        scores=scores,
        overall_summary=str(payload.get("overall_summary", "")).strip(),
        raw_response=raw,
    )


def judge_continuation_match(
    prefix: list[Message],
    actual: Message,
    generated: Message,
    client: TextChatClient,
    *,
    condition: str,
    model: str,
    temperature: float = 0.0,
) -> ContinuationVerdict:
    """Score one (generated, actual) pair against the thread prefix."""
    system, user = build_continuation_prompt(prefix, actual, generated, condition)
    raw = client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        model=model,
        temperature=temperature,
    )
    return parse_continuation_response(raw, condition)
