"""Build a per-persona system prompt with an LLM, once.

Splits the agent prompt into two layers:

* **Prebuilt** — a thoughtful, narrative system prompt the LLM
  assembles from the Stage 4a enrichment fields, writing samples, and
  optional KG context. Captures who the persona is, how they
  communicate, and what they care about. Stable across simulations
  for that persona; can be inspected, versioned, and edited by hand.

* **Runtime** — context that depends on what's happening *now*:
  pair-specific samples for the current recipient, the scenario brief,
  conversation-handling instructions, termination signals. Appended to
  the prebuilt prompt when an agent is constructed for a specific run.

This module covers the prebuilt half. The runtime half is composed
inside :func:`orggraph.agents.persona.serialize_persona` and the
runners.

Why an LLM-curated prebuilt prompt instead of a fixed template?
Templates produce uniform-feeling prompts: every persona reads like a
form-filled card. An LLM given the underlying signals can write a
prompt that *integrates* them — noting that a persona's verbosity
profile shows up as long, structured emails with numbered lists, or
that a high-formality authority style manifests as third-person
references to themselves in legal contexts. That kind of integrated
writing is exactly what the agent needs to imitate.

Important: the prebuilt prompt is generated from corpus-wide data and
is **not temporally hygienic**. Use it for the dialogue-naturalness
experiment freely, but for strict-temporal continuation experiments
either rebuild it per cutoff or disable prebuilt prompts and fall
back to the runtime template (the latter is the current default).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from orggraph.agents.agent import TextChatClient
from orggraph.agents.kg_context import KGContext, format_kg_context
from orggraph.agents.persona import Persona


_BUILDER_SYSTEM_PROMPT_V1 = """You are an expert prompt engineer.
Your task is to write a system prompt for an AI agent that will
role-play a specific real Enron employee in business email
simulations. The agent must produce emails that pass for that
person's actual writing.

You will be given the person's identity, expertise signals, behavioural
profile, calibrated communication style, organisational position, and
several real writing samples from the corpus.

Synthesise a system prompt that:

1. Opens with "You are {name}, ..." and continues in second person.
2. Describes their role and position concretely, with reference to
   their actual responsibilities and the topics they deal with.
3. Captures their voice using *observations from the samples*:
   typical sentence length, sign-off habits, salutation conventions,
   punctuation patterns, characteristic phrases, register shifts
   between formal and informal contexts.
4. Notes any distinctive verbal tics, recurring phrases, or
   structural habits visible in the samples (e.g. always starts
   with the recipient's first name; favours bulleted lists; signs
   only with initials in informal threads).
5. Is 250–400 words. Concrete and specific to this person.

Do NOT include conversation-handling rules — those are appended at
runtime by the simulation harness.
Do NOT quote the writing samples verbatim in your output — the build
harness will append them automatically as concrete reference material
after your prose section. Your prose should describe and integrate
the patterns; the verbatim samples follow.
Do NOT add JSON wrappers, markdown headers, code fences, or
meta-commentary like "Here is the prompt:".

Output ONLY the system prompt text, ready to use as-is."""


_BUILDER_SYSTEM_PROMPT_V2 = """You are an expert prompt engineer.
Your task is to write a system prompt for an AI agent that will
role-play a specific real Enron employee in business email
simulations. The agent must produce emails that pass for that
person's actual writing.

You will be given the person's identity, expertise signals, behavioural
profile, calibrated communication style, organisational position, and
several real writing samples from the corpus.

Synthesise a system prompt with EXACTLY these four markdown sections:

## Identity
2-3 sentences. Open with "You are {name}." State their role, primary
expertise, and what kinds of email they handle.

## Voice & Style
3-5 sentences describing HOW they write — sentence rhythm, register,
formality calibration, characteristic phrases — using observations from
the samples.

## How You Write
3-5 sentences on observable patterns: salutation conventions, sign-off
habits, punctuation, structural habits (bullets, BlackBerry tags,
brevity vs elaboration). Be concrete with patterns visible in the
samples.

## Organisational Position
2-3 sentences on hierarchy, who they report to, who reports to them,
who they collaborate with. Use the KG context if available.

Total target: 250-400 words across all four sections.

Do NOT include conversation-handling rules — those are appended at runtime.
Do NOT quote samples verbatim — the harness will append them.
Do NOT add JSON wrappers, code fences, or meta-commentary.

Output ONLY the four-section system prompt, ready to use as-is."""


_BUILDER_SYSTEM_PROMPT_V3 = """You are an expert prompt engineer.
Your task is to write a system prompt for an AI agent that will
role-play a specific real Enron employee in business email
simulations. The agent must produce emails that pass for that
person's actual writing.

You will be given the person's identity, expertise signals, behavioural
profile, calibrated communication style, organisational position, and
several real writing samples from the corpus.

Synthesise a system prompt in TWO parts:

## Identity card
A bulleted list of facts:
- **Name:** {name}
- **Role:** ...
- **Tier:** ... within the organisation, with reporting relationships
  if known
- **Formality:** ... /5
- **Primary expertise:** 2-3 specific topics
- **Communication signature:** one phrase capturing a recurring voice
  pattern observed in the samples

## Voice
2-3 short paragraphs (target 150-200 words combined) describing how this
person writes. Reference observable patterns from the samples — sign-
offs, salutations, sentence rhythm, register — without quoting them.
Open with "You are {name}, ..." and continue in second person.

Total target: 200-280 words combined.

Do NOT include conversation-handling rules.
Do NOT quote samples verbatim — the harness will append them.
Do NOT add JSON wrappers, code fences, or meta-commentary.

Output ONLY the identity card plus voice paragraphs, ready to use as-is."""


@dataclass(frozen=True)
class BuilderVariant:
    """A named builder-system-prompt configuration for the structure search."""

    name: str
    system_prompt: str
    description: str


BUILDER_VARIANTS: dict[str, BuilderVariant] = {
    "v1_freeform": BuilderVariant(
        name="v1_freeform",
        system_prompt=_BUILDER_SYSTEM_PROMPT_V1,
        description="Free-form 250-400 word prose narrative. Existing baseline.",
    ),
    "v2_sections": BuilderVariant(
        name="v2_sections",
        system_prompt=_BUILDER_SYSTEM_PROMPT_V2,
        description="Four section-headed parts: Identity, Voice & Style, How You Write, Organisational Position.",
    ),
    "v3_idcard": BuilderVariant(
        name="v3_idcard",
        system_prompt=_BUILDER_SYSTEM_PROMPT_V3,
        description="Bulleted identity card + 2-3 voice paragraphs. Compact, 200-280 words.",
    ),
}


def _slug(name: str) -> str:
    """Turn 'Sara Shackleton' into 'sara_shackleton' for filenames."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return s or "unnamed"


def _format_signals(p: Persona) -> str:
    """The structured signal block the builder sees as input."""
    lines = [
        f"- name: {p.name}",
        f"- role_summary: {p.role_summary}",
    ]
    if p.expertise:
        lines.append(f"- specific_expertise: {'; '.join(p.expertise[:8])}")
    if p.expertise_topics:
        lines.append(f"- top_topics: {', '.join(p.expertise_topics[:5])}")
    if p.seniority_narrative:
        lines.append(f"- seniority_narrative: {p.seniority_narrative}")
    lines.append(
        f"- behavioural_profile: directiveness={p.directiveness:.2f}, "
        f"agenda_setting={p.agenda_setting:.2f}, "
        f"verbosity={p.verbosity:.2f}"
    )
    lines.append(f"- formality: {p.formality}/5")
    if p.authority_style:
        lines.append(f"- authority_style: {p.authority_style}")
    if p.communication_style:
        lines.append(f"- communication_style: {p.communication_style}")
    if p.confidence_self_report is not None:
        lines.append(f"- persona_inference_confidence: {p.confidence_self_report}/5")
    return "\n".join(lines)


def _format_samples(samples: tuple[str, ...]) -> str:
    if not samples:
        return "(no writing samples available)"
    return "\n\n".join(
        f"--- sample {i + 1} ---\n{s}" for i, s in enumerate(samples)
    )


def _format_samples_for_prompt(samples: tuple[str, ...], n_keep: int) -> str:
    """Render the verbatim-samples section appended to every prebuilt prompt.

    Two or three samples are ideal: enough to anchor the model on the
    persona's actual register, few enough to keep the system prompt
    manageable when stitched with the runtime additions.

    ``n_keep <= 0`` disables embedding entirely (returns "").
    """
    if not samples or n_keep <= 0:
        return ""
    keep = samples[:n_keep]
    blocks = "\n\n".join(
        f"--- example {i + 1} ---\n{s}" for i, s in enumerate(keep)
    )
    return (
        "\n\n## Examples of how you actually write\n"
        "Match this register — vocabulary, sentence rhythm, salutations, "
        "sign-offs, and overall length. Do NOT copy these verbatim; they "
        "are stylistic anchors, not content to reuse.\n\n"
        + blocks
    )


def build_persona_prompt(
    persona: Persona,
    samples: tuple[str, ...],
    kg: KGContext | None,
    client: TextChatClient,
    *,
    model: str,
    temperature: float = 0.4,
    embed_samples: int = 3,
    system_prompt: str | None = None,
) -> str:
    """Synthesise a per-persona system prompt via one LLM call,
    then append a verbatim samples section.

    Parameters
    ----------
    persona:
        The Stage 4a-enriched Persona (samples + kg can be empty).
    samples:
        Writing samples — separate from ``persona.sample_messages`` so
        callers can pass a curated set (e.g. forwarded-content
        filtered, length-stratified, date-bounded) without mutating
        the Persona. The first ``embed_samples`` of these are appended
        verbatim to the LLM-written prose so the final prompt carries
        both narrative voice description AND concrete reference
        examples.
    kg:
        Optional KGContext block. Passed to the builder as additional
        signal but the prebuilt prompt should NOT depend on it for
        identity — KG can change, identity shouldn't.
    client, model, temperature:
        LLM call parameters. Default ``temperature=0.4`` — low enough
        for stability, high enough for the model to write coherent
        prose rather than templated output.
    embed_samples:
        Number of verbatim samples to embed at the end of the prompt
        (default 3). 0 disables embedding.
    system_prompt:
        Override the default V1 builder system prompt. Used by the
        structure search (Task 14) to A/B-test alternative output
        structures. Defaults to V1 (free-form prose).
    """
    signals = _format_signals(persona)
    samples_block = _format_samples(samples)
    kg_block = format_kg_context(kg) if kg else ""

    user = (
        f"## Persona signals\n\n{signals}\n\n"
        f"## Writing samples\n\n{samples_block}\n\n"
    )
    if kg_block:
        user += f"## Organisational position{kg_block}\n\n"
    user += "Now write the system prompt as instructed."

    raw = client.chat(
        system=system_prompt or _BUILDER_SYSTEM_PROMPT_V1,
        messages=[{"role": "user", "content": user}],
        model=model,
        temperature=temperature,
    )
    text = raw.strip()
    # Strip code fences if the model added them despite instructions
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            head, rest = text.split("\n", 1)
            if head.lower().strip() in {"text", "markdown", ""}:
                text = rest
    text = text.strip()

    # Append verbatim samples as a guaranteed section. The LLM was told
    # not to embed them in its own output, so this avoids duplication.
    text += _format_samples_for_prompt(samples, n_keep=embed_samples)
    return text


from orggraph.agents.prompt_judge import RubricScores, score_prompt


@dataclass(frozen=True)
class CandidateRound:
    """One (text, scores) pair in an N-best generation round."""

    text: str
    scores: RubricScores


def build_persona_prompt_n_best(
    persona: Persona,
    samples: tuple[str, ...],
    kg: KGContext | None,
    builder_client: TextChatClient,
    judge_client: TextChatClient,
    *,
    model: str,
    judge_model: str,
    n_best: int = 3,
    builder_temperature: float = 0.6,
    judge_temperature: float = 0.0,
    embed_samples: int = 3,
    variant_name: str = "v1_freeform",
) -> tuple[CandidateRound, list[CandidateRound]]:
    """Generate N candidates, judge each, return (winner, all candidates).

    Tie-breaks: total score, then min_score (more balanced wins),
    then earliest index.

    Note on judge input: the judge sees the full prompt as the agent
    will consume it — including the verbatim ``embed_samples`` block
    appended by :func:`build_persona_prompt`. This is intentional
    (we score the delivered artefact, not the prose alone) but means
    the rubric's ``voice_integration`` dimension can be inflated when
    the appended samples themselves carry strong voice signal. If this
    inflation distorts calibration in Task 8, strip the samples block
    before scoring.

    variant_name:
        Which BuilderVariant to use. Defaults to v1_freeform (the
        existing free-form prose builder). Other choices: v2_sections,
        v3_idcard. Raises KeyError for unknown names.

    Failure semantics: any builder or judge call that raises propagates
    out; the caller (Stage 4b run loop, Task 6) catches and logs the
    persona as errored. There is no per-candidate degradation.
    """
    if n_best < 1:
        raise ValueError(f"n_best must be >= 1, got {n_best}")

    variant = BUILDER_VARIANTS[variant_name]

    candidates: list[CandidateRound] = []
    for _ in range(n_best):
        text = build_persona_prompt(
            persona, samples=samples, kg=kg,
            client=builder_client, model=model,
            temperature=builder_temperature,
            embed_samples=embed_samples,
            system_prompt=variant.system_prompt,
        )
        scores = score_prompt(
            text, persona, samples,
            client=judge_client, model=judge_model,
            temperature=judge_temperature,
        )
        candidates.append(CandidateRound(text=text, scores=scores))

    winner_index, winner = max(
        enumerate(candidates),
        key=lambda iv: (iv[1].scores.total(), iv[1].scores.min_score(), -iv[0]),
    )
    return winner, candidates


# Re-exported under the package's public names for convenience
__all__ = [
    "build_persona_prompt", "build_persona_prompt_n_best",
    "BuilderVariant", "BUILDER_VARIANTS",
    "CandidateRound", "_slug",
]
