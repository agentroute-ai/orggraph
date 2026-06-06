"""LLM-as-judge for RQ2 dialogue transcripts.

Scores a saved Transcript on five dimensions and returns per-turn
issue flags so the prompt-iteration loop has actionable feedback,
not just numbers.

The five dimensions are picked to surface the failure modes we
already observed in early pilot runs:

* **speaker_consistency** — does each turn read as the named persona?
  (The 24/04 single-LLM run had Louise writing to "Louise" and
  signing as "Louise"; this dimension catches that.)
* **persona_fidelity** — does the writing match the persona's known
  style attributes (formality, register, signature habits)?
* **scenario_grounding** — does the conversation stay on the scenario
  topic with the *listed* participants only? (Catches the post-
  few-shot "invented Mark" hallucination in single-LLM.)
* **conversational_coherence** — do turns logically reference each
  other; do later turns honour earlier commitments?
* **naturalness** — would a human reader plausibly mistake this for
  real Enron correspondence?

Each dimension gets a 1–5 score with a one-sentence justification.
Per-turn flags (``speaker_confusion``, ``character_break``,
``hallucination``, ``disfluency``) carry a short detail string so
prompt regressions surface the exact turn that introduced them.

The judge is *reference-grounded*: it sees the scenario brief, the
list of valid participants, and short persona summaries. It does
NOT see which condition produced the dialogue; the calling script
shuffles transcripts before scoring to remove ordering bias.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from orggraph.agents.agent import TextChatClient
from orggraph.agents.persona import Persona
from orggraph.simulation.transcript import Message, Transcript

DIMENSIONS = (
    "speaker_consistency",
    "persona_fidelity",
    "scenario_grounding",
    "conversational_coherence",
    "naturalness",
)

ISSUE_TYPES = (
    "speaker_confusion",
    "character_break",
    "hallucination",
    "disfluency",
)


@dataclass(frozen=True)
class DimensionScore:
    """One dimension's score (1–5) plus a short justification."""

    score: int
    justification: str


@dataclass(frozen=True)
class TurnFlag:
    """A specific issue observed at a single turn."""

    turn_id: int
    issue: str
    detail: str


@dataclass
class JudgeResult:
    """Full per-transcript verdict."""

    scenario_name: str
    condition: str
    scores: dict[str, DimensionScore] = field(default_factory=dict)
    turn_flags: list[TurnFlag] = field(default_factory=list)
    overall_summary: str = ""
    raw_response: str = ""

    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores.values()) / len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "condition": self.condition,
            "mean_score": self.mean_score(),
            "scores": {
                k: {"score": v.score, "justification": v.justification}
                for k, v in self.scores.items()
            },
            "turn_flags": [
                {"turn_id": t.turn_id, "issue": t.issue, "detail": t.detail}
                for t in self.turn_flags
            ],
            "overall_summary": self.overall_summary,
        }


class JudgeError(RuntimeError):
    """Raised when the judge response can't be parsed into a JudgeResult."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """You are an evaluator of simulated business email dialogues.
Your job is to score the dialogue on five dimensions and flag specific
issues turn-by-turn. Be precise, conservative, and reference-grounded —
do not invent context not present in the brief or the dialogue itself.

Scoring rubric (1–5 each, integer scores only):
- 1: severe failure on this dimension; multiple unambiguous violations
- 2: substantive issues that a careful reader would notice
- 3: noticeable issues but the dialogue is recognisably plausible
- 4: minor issues only
- 5: no issues observable from the dialogue alone

Return STRICT JSON. No prose outside the JSON object."""


_RUBRIC_TEXT = """
Dimensions (score each 1–5 with a one-sentence justification):

1. speaker_consistency: Does each turn read as authored by the named
   persona? Penalise turns that address themselves by name, sign with
   the wrong name, or otherwise confuse who is speaking.

2. persona_fidelity: Does the writing match the persona's described
   style — formality, role, expertise vocabulary, sign-off conventions?

3. scenario_grounding: Does the conversation stay on the scenario topic
   and reference ONLY the listed participants? Penalise hallucinated
   attendees, off-topic digressions, and references to events not in
   the brief.

4. conversational_coherence: Do later turns logically follow earlier
   ones? Are commitments / decisions referenced consistently across
   turns? Penalise contradictions and non-sequiturs.

5. naturalness: Would a reader plausibly mistake this for real
   organisational correspondence? Penalise robotic templating,
   exaggerated formality, repetitive sentence structures.

Per-turn issue flags. List ONLY turns with observable issues. Each
flag has issue ∈ {speaker_confusion, character_break, hallucination,
disfluency} and a detail string ≤ 120 chars."""


_OUTPUT_SCHEMA = """
Return JSON of this exact shape:
{
  "scores": {
    "speaker_consistency": {"score": <int 1-5>, "justification": "<one sentence>"},
    "persona_fidelity":    {"score": <int 1-5>, "justification": "<one sentence>"},
    "scenario_grounding":  {"score": <int 1-5>, "justification": "<one sentence>"},
    "conversational_coherence": {"score": <int 1-5>, "justification": "<one sentence>"},
    "naturalness":         {"score": <int 1-5>, "justification": "<one sentence>"}
  },
  "turn_flags": [
    {"turn_id": <int>, "issue": "<one of the four issue types>", "detail": "<short detail>"}
  ],
  "overall_summary": "<2-3 sentence prose summary identifying the strongest signal across the dialogue>"
}"""


def _render_dialogue(messages: list[Message]) -> str:
    if not messages:
        return "(empty dialogue)"
    lines: list[str] = []
    for m in messages:
        addressing = "[broadcast]" if m.is_broadcast() else f"[to {', '.join(m.recipients)}]"
        body = m.body.replace("\n", "\n    ")
        lines.append(f"[turn {m.turn_id}] {m.sender} {addressing}:\n    {body}")
    return "\n\n".join(lines)


def _short_persona_block(persona: Persona) -> str:
    """A compact persona summary the judge can use as a reference."""
    topics = ", ".join(persona.expertise_topics) if persona.expertise_topics else "—"
    return (
        f"- **{persona.name}**: {persona.role_summary[:160]} "
        f"(formality {persona.formality}/5; "
        f"style: {persona.communication_style[:80]}; "
        f"topics: {topics})"
    )


def build_judge_prompt(
    transcript: Transcript,
    scenario_brief: str,
    participants: list[str],
    personas: dict[str, Persona],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) ready for the judge LLM."""
    persona_blocks = "\n".join(
        _short_persona_block(personas[name])
        for name in participants
        if name in personas
    )
    user = (
        f"Scenario brief: {scenario_brief}\n\n"
        f"Listed participants: {', '.join(participants)}\n\n"
        f"Persona summaries:\n{persona_blocks}\n\n"
        f"{_RUBRIC_TEXT}\n\n"
        f"{_OUTPUT_SCHEMA}\n\n"
        f"Dialogue to score:\n{_render_dialogue(transcript.messages)}"
    )
    return _SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_judge_response(raw: str, transcript: Transcript) -> JudgeResult:
    """Parse the judge's JSON response into a :class:`JudgeResult`.

    Tolerates a few minor format mismatches (extra whitespace, code
    fences) but raises :class:`JudgeError` on missing required keys.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # strip optional language tag on the first line
        if "\n" in text:
            first, rest = text.split("\n", 1)
            if first.lower().strip() in {"json", "javascript", ""}:
                text = rest

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge response is not valid JSON: {exc}\n--- raw ---\n{raw[:400]}") from exc

    if not isinstance(payload, dict):
        raise JudgeError(f"judge response is not a JSON object: {type(payload).__name__}")

    raw_scores = payload.get("scores", {})
    scores: dict[str, DimensionScore] = {}
    for dim in DIMENSIONS:
        entry = raw_scores.get(dim)
        if not isinstance(entry, dict) or "score" not in entry:
            raise JudgeError(f"judge response missing dimension {dim!r}")
        s = entry["score"]
        try:
            s_int = int(s)
        except (TypeError, ValueError):
            raise JudgeError(f"non-integer score for {dim!r}: {s!r}")
        if not 1 <= s_int <= 5:
            raise JudgeError(f"score for {dim!r} out of [1,5] range: {s_int}")
        scores[dim] = DimensionScore(
            score=s_int,
            justification=str(entry.get("justification", "")).strip(),
        )

    flags: list[TurnFlag] = []
    for raw_flag in payload.get("turn_flags", []) or []:
        if not isinstance(raw_flag, dict):
            continue
        try:
            tid = int(raw_flag.get("turn_id", -1))
        except (TypeError, ValueError):
            continue
        issue = str(raw_flag.get("issue", "")).strip()
        if issue and issue not in ISSUE_TYPES:
            # Unknown issue type — keep but don't crash
            issue = f"other:{issue}"
        flags.append(
            TurnFlag(
                turn_id=tid,
                issue=issue,
                detail=str(raw_flag.get("detail", "")).strip()[:240],
            )
        )

    return JudgeResult(
        scenario_name=transcript.scenario_name,
        condition=transcript.condition,
        scores=scores,
        turn_flags=flags,
        overall_summary=str(payload.get("overall_summary", "")).strip(),
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def judge_transcript(
    transcript: Transcript,
    scenario_brief: str,
    participants: list[str],
    personas: dict[str, Persona],
    client: TextChatClient,
    *,
    model: str,
    temperature: float = 0.0,
    max_attempts: int = 3,
) -> JudgeResult:
    """Score one transcript with the LLM judge.

    ``temperature=0`` by default for reproducibility — the judge is the
    one place we explicitly do NOT want sample diversity, since the
    score is the measurement.

    Retries up to ``max_attempts`` times if the endpoint returns an
    empty body or unparseable JSON. The first retry stays at the
    requested temperature; subsequent ones bump it slightly to break
    the model out of a stuck state.
    """
    system, user = build_judge_prompt(transcript, scenario_brief, participants, personas)
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        # Bump temperature only on later retries; first retry uses the
        # requested temperature so a transient empty response still
        # reproduces if it was just network noise.
        attempt_temp = temperature if attempt < 2 else max(temperature, 0.2)
        try:
            raw = client.chat(
                system=system,
                messages=[{"role": "user", "content": user}],
                model=model,
                temperature=attempt_temp,
            )
            if not raw or not raw.strip():
                raise JudgeError("judge endpoint returned an empty response")
            return parse_judge_response(raw, transcript)
        except (JudgeError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt + 1 < max_attempts:
                continue
            raise
    # Unreachable, but keep mypy happy
    raise JudgeError(f"judge_transcript exhausted retries: {last_exc}")
