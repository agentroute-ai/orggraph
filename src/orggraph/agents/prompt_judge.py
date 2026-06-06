"""LLM-as-judge for persona-prompt quality (Tier 1 of the quality gate).

Scores a candidate persona prompt on five dimensions and returns a
RubricScores. The full design is in docs/plans/2026-05-07-persona-
prompt-quality-design.md §3.1.

Splits cleanly from prompt_builder.py: builder generates, judge
evaluates. Different concerns, different test surfaces.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

DIMENSIONS = (
    "specificity",
    "source_grounding",
    "voice_integration",
    "completeness",
    "clean_form",
)


@dataclass(frozen=True)
class RubricScores:
    """Per-dimension 1-5 integer scores for one candidate prompt."""

    specificity: int
    source_grounding: int
    voice_integration: int
    completeness: int
    clean_form: int

    def all_pass(self, threshold: int = 3) -> bool:
        return all(getattr(self, d) >= threshold for d in DIMENSIONS)

    def min_score(self) -> int:
        return min(getattr(self, d) for d in DIMENSIONS)

    def total(self) -> int:
        return sum(getattr(self, d) for d in DIMENSIONS)

    def to_dict(self) -> dict[str, int]:
        return {d: getattr(self, d) for d in DIMENSIONS}


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def parse_rubric_response(text: str) -> RubricScores:
    """Parse a judge response into RubricScores.

    Accepts bare JSON, JSON inside a ``` fence, and ``{dim: {"score": N}}``
    nested form.
    """
    body = text.strip()
    fence_match = _JSON_FENCE.search(body)
    if fence_match:
        body = fence_match.group(1).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge response is not JSON: {e!s}") from e
    if not isinstance(data, dict):
        raise ValueError(
            f"Judge response must be a JSON object, got {type(data).__name__}"
        )

    parsed: dict[str, int] = {}
    for dim in DIMENSIONS:
        if dim not in data:
            raise ValueError(f"Missing dimension {dim!r} in judge response")
        score = data[dim]
        if isinstance(score, dict) and "score" in score:
            score = score["score"]
        if isinstance(score, bool) or not isinstance(score, int) or not (1 <= score <= 5):
            raise ValueError(
                f"Dimension {dim!r} must be int in [1, 5], got {score!r}"
            )
        parsed[dim] = score
    return RubricScores(**parsed)


from orggraph.agents.agent import TextChatClient
from orggraph.agents.persona import Persona

# Note: _format_signals / _format_samples are imported lazily inside
# format_judge_request to avoid a circular import — prompt_builder.py
# imports score_prompt from this module.

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI agent system prompts.

You will be given:
1. The structured signals about a real Enron employee.
2. Writing samples by that employee.
3. A candidate system prompt that an AI agent will use to role-play this employee.

Score the prompt on five dimensions, each integer 1-5:

- specificity: 5 = unique to this person; 1 = could apply to any Enron employee.
- source_grounding: 5 = every claim traceable to signals or samples; 1 = invented biography.
- voice_integration: 5 = describes observable patterns from the samples (sign-offs, salutations, register shifts); 1 = surface paraphrase of structured signals only.
- completeness: 5 = covers role, expertise, communication style, organisational position; 1 = missing categories.
- clean_form: 5 = 250-450 words, no leaked code fences, no JSON wrappers, no meta-commentary; 1 = clear LLM artefacts.

Return ONLY a JSON object with the five integer scores. No prose, no markdown fences:
{"specificity": N, "source_grounding": N, "voice_integration": N, "completeness": N, "clean_form": N}
"""


def format_judge_request(
    prompt: str,
    persona: Persona,
    samples: tuple[str, ...],
) -> str:
    """Compose the user message the judge sees."""
    # Lazy import: prompt_builder imports score_prompt from this module,
    # so a top-level import here would be circular.
    from orggraph.agents.prompt_builder import _format_samples, _format_signals

    return (
        f"## Persona signals\n\n{_format_signals(persona)}\n\n"
        f"## Writing samples\n\n{_format_samples(samples)}\n\n"
        f"## Candidate system prompt\n\n{prompt}\n\n"
        "Score it now."
    )


def score_prompt(
    prompt: str,
    persona: Persona,
    samples: tuple[str, ...],
    *,
    client: TextChatClient,
    model: str,
    temperature: float = 0.0,
) -> RubricScores:
    """Score one candidate prompt via a single judge LLM call."""
    user = format_judge_request(prompt, persona, samples)
    raw = client.chat(
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        model=model,
        temperature=temperature,
    )
    return parse_rubric_response(raw)
