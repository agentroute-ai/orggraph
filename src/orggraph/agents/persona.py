"""Persona dataclass + serialisation for RQ2 agent grounding.

Reads Stage 4a output (``person_enrichment.csv``) and produces the
persona system prompt described in thesis Ch.4 §4.6. The serialised
form is the agent grounding for the multi-agent condition (each agent
gets its own persona) and a stitched component of the single-LLM
condition (all personas concatenated).

Example
-------
>>> persona = Persona(
...     name="Sara Shackleton",
...     role_summary="VP and Associate General Counsel at Enron",
...     directiveness=0.71, agenda_setting=0.63, verbosity=0.42,
...     formality=4, authority_style="authoritative but collegial",
...     communication_style="closes decisions rather than initiates them",
...     expertise_topics=["derivatives confirmations", "regulatory disclosure"],
... )
>>> "Sara Shackleton" in serialize_persona(persona)
True
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from orggraph.agents.kg_context import KGContext, format_kg_context

# Maximum chars to include from role_summary when stitching personas
# in the single-LLM baseline. Keeps the combined prompt under typical
# 8K context limits even with 6 participants.
ROLE_SUMMARY_TRUNC = 200


@dataclass(frozen=True)
class Persona:
    """Agent grounding extracted from Stage 4a person_enrichment.csv.

    The optional ``sample_messages`` carry few-shot writing samples used
    to ground the agent's style in concrete examples. Each sample is a
    short formatted block (e.g. ``"Subject: …\\n  Body…"``); the
    serialiser appends them under an "Examples of how you write" header
    when present. Empty by default so existing call sites that don't
    care about style grounding keep working unchanged.
    """

    name: str
    role_summary: str
    directiveness: float
    agenda_setting: float
    verbosity: float
    formality: int
    authority_style: str
    communication_style: str
    expertise_topics: tuple[str, ...] = ()
    expertise: tuple[str, ...] = ()
    seniority_narrative: str = ""
    confidence_self_report: int | None = None
    sample_messages: tuple[str, ...] = ()
    kg_context: KGContext | None = None
    # Prebuilt system prompt produced by orggraph.agents.prompt_builder.
    # When set, serialize_persona uses this as the base instead of the
    # mechanical template, and only appends runtime context (samples
    # for the current recipient, KG, conversation rules).
    prebuilt_prompt: str | None = None

    @classmethod
    def from_row(cls, row: pd.Series) -> "Persona":
        """Build from a row of person_enrichment.csv. Loads every Stage 4a
        narrative field; ``serialize_persona`` decides what makes it into
        the prompt."""
        confidence_raw = row.get("confidence_self_report")
        confidence: int | None
        try:
            confidence = int(float(confidence_raw)) if confidence_raw is not None and not (
                isinstance(confidence_raw, float) and math.isnan(confidence_raw)
            ) else None
        except (TypeError, ValueError):
            confidence = None

        return cls(
            name=str(row["name"]).strip(),
            role_summary=_safe_str(row.get("role_summary")),
            directiveness=_safe_float(row.get("directiveness_signal")),
            agenda_setting=_safe_float(row.get("agenda_setting_signal")),
            verbosity=_safe_float(row.get("verbosity_signal")),
            formality=_safe_int(row.get("formality"), default=3),
            authority_style=_safe_str(row.get("authority_style")),
            communication_style=_safe_str(row.get("communication_style")),
            expertise_topics=_parse_topics(row.get("expertise_topics")),
            expertise=_parse_topics(row.get("expertise")),
            seniority_narrative=_safe_str(row.get("seniority_narrative")),
            confidence_self_report=confidence,
        )

    def with_samples(self, samples: tuple[str, ...]) -> "Persona":
        """Return a new Persona with ``sample_messages`` replaced.

        Used to attach few-shot samples loaded from a separate corpus
        file (e.g. ``clean_emails.parquet``) without re-parsing the
        persona CSV.
        """
        from dataclasses import replace
        return replace(self, sample_messages=tuple(samples))

    def with_kg_context(self, kg: KGContext | None) -> "Persona":
        """Return a new Persona with ``kg_context`` replaced.

        Used to attach Neo4j-derived organisational context loaded
        separately via :func:`orggraph.agents.kg_context.load_kg_context`.
        """
        from dataclasses import replace
        return replace(self, kg_context=kg)

    def with_prebuilt_prompt(self, prompt: str | None) -> "Persona":
        """Return a new Persona with ``prebuilt_prompt`` replaced.

        Used to attach an LLM-generated system prompt loaded from disk
        (see :mod:`orggraph.agents.prompt_builder`). When set,
        :func:`serialize_persona` uses it as the base instead of the
        mechanical template and only appends runtime context.
        """
        from dataclasses import replace
        return replace(self, prebuilt_prompt=prompt)


def serialize_persona(
    persona: Persona,
    *,
    truncate_role: bool = False,
    max_samples: int | None = None,
) -> str:
    """Render a persona as the system-prompt narrative paragraph.

    The narrative format follows thesis Ch.4 §4.6 and is deliberately
    deterministic so identical inputs produce identical prompts (caching
    + reproducibility).

    Set ``truncate_role=True`` when stitching multiple personas into a
    single-LLM baseline prompt; this trims ``role_summary`` to
    ``ROLE_SUMMARY_TRUNC`` chars so the combined prompt stays within
    typical 8K context windows.

    If ``persona.sample_messages`` is non-empty, the samples are
    appended under an "Examples of how you write" section so the model
    has concrete style references in addition to the abstract
    descriptors. ``max_samples`` caps how many are emitted (useful when
    stitching multiple personas into one prompt — fewer samples per
    persona keeps the total prompt budget under control).
    """
    if persona.prebuilt_prompt:
        # Prebuilt path: use the LLM-generated system prompt as base.
        # Skip the mechanical template entirely; the prebuilt prompt
        # already integrates role + voice + expertise into a
        # narrative-quality system prompt.
        base = persona.prebuilt_prompt.strip()
    else:
        # Legacy template path: build mechanically from individual fields.
        role = persona.role_summary
        if truncate_role and len(role) > ROLE_SUMMARY_TRUNC:
            role = role[: ROLE_SUMMARY_TRUNC - 1].rstrip() + "…"

        topics = ", ".join(persona.expertise_topics) if persona.expertise_topics else "general business operations"
        formality_label = _formality_label(persona.formality)

        # communication_style was stored as a sentence (capitalized first letter,
        # often with a trailing period) — fold it into a clause that flows after
        # "your communication style is" without producing "you Sends..." or
        # double-period artefacts.
        comm_style = persona.communication_style.strip().rstrip(".")
        if comm_style:
            comm_style = comm_style[0].lower() + comm_style[1:]

        base = (
            f"You are {persona.name}, {role}. "
            f"Behavioural profile: directiveness {persona.directiveness:.2f}, "
            f"agenda-setting {persona.agenda_setting:.2f}, "
            f"verbosity {persona.verbosity:.2f}. "
            f"Your top expertise topics are {topics}. "
            f"Calibrated style: {formality_label} ({persona.formality}/5), "
            f"{persona.authority_style}; "
            f"your communication style is one in which {comm_style}."
        )

        # Stage 4a narrative fields that previously weren't rendered.
        if persona.seniority_narrative:
            base += f"\n\n## Seniority and role context\n{persona.seniority_narrative}"

        if persona.expertise:
            expertise_str = "; ".join(persona.expertise[:5])
            base += f"\n\n## Specific expertise\n{expertise_str}"

        if persona.confidence_self_report is not None:
            # Confidence is a 1–5 self-report from the LLM about how confident
            # it is in its persona inference. Useful for the agent to know it's
            # operating on shakier ground (low confidence) and should hedge.
            base += (
                f"\n\n## Persona-inference confidence\nThe persona was inferred "
                f"from this corpus with confidence {persona.confidence_self_report}/5; "
            )
            if persona.confidence_self_report <= 2:
                base += (
                    "you have only a thin signal of how this person actually writes. "
                    "Lean conservative — short, neutral, professional registers."
                )
            elif persona.confidence_self_report >= 4:
                base += "you have strong signal; commit to the calibrated style above."
            else:
                base += "the signal is moderate; trust the calibrated style but stay flexible."

    # ---- Runtime sections (apply in BOTH prebuilt and template paths) ----

    # KG context: hierarchy / collaborators / topics from Neo4j
    if persona.kg_context is not None:
        kg_block = format_kg_context(persona.kg_context)
        if kg_block:
            base += kg_block

    if persona.sample_messages:
        samples = persona.sample_messages
        if max_samples is not None and max_samples >= 0:
            samples = samples[:max_samples]
        if samples:
            blocks = "\n\n".join(
                f"--- example {i + 1} ---\n{s}" for i, s in enumerate(samples)
            )
            # Use a distinct header when the prebuilt prompt is in use, so
            # we don't duplicate the "## Examples of how you actually write"
            # heading that's already baked into the prebuilt by Stage 4b.
            # In that case the runtime samples are typically pair-specific
            # (sender → current recipient), giving the agent BOTH the
            # general voice (prebuilt) and conversation-specific register
            # (here).
            if persona.prebuilt_prompt:
                header = (
                    "\n\n## Recent and recipient-specific samples\n"
                    "These supplement the general examples above with samples "
                    "more relevant to the current conversation. Match the "
                    "register; do not quote verbatim.\n\n"
                )
            else:
                header = (
                    "\n\n## Examples of how you actually write\n"
                    "Match this style. Do not quote verbatim — these are "
                    "stylistic references for vocabulary, sentence rhythm, "
                    "and sign-off conventions, not content to reuse.\n\n"
                )
            base += header + blocks

    # Conversation-handling instructions. These specifically target the
    # "subject-line lock" failure mode observed in the early continuation
    # experiment, where the agent fixated on the literal email subject
    # and missed that the conversation had moved on.
    base += (
        "\n\n## How to engage with the conversation\n\n"
        "1. Read the *most recent* turns carefully — the conversation may have "
        "moved past the original subject line. Match the topical and emotional "
        "register of where the discussion currently is, not where it started.\n"
        "2. If the prior turns are short and informal, do not respond with a "
        "long formal email. If they are panicked or stressed, do not respond "
        "with bureaucratic templates.\n"
        "3. Speak only as yourself. Do not narrate, do not summarise the "
        "conversation, do not address yourself by name in the body, do not "
        "sign with the recipient's name.\n"
        "4. Reply with only the email body — no quotation marks, no "
        "meta-commentary, no preamble like \"Here is my reply:\"."
    )

    return base


def load_personas_from_csv(
    path: Path,
    *,
    prompts_dir: Path | None = None,
) -> dict[str, Persona]:
    """Load every persona from a person_enrichment.csv file.

    Returns a name → Persona dict for fast lookup by canonical name.
    Rows missing the ``name`` column or with empty names are skipped.

    If ``prompts_dir`` is provided, look for a per-persona prebuilt
    prompt at ``<prompts_dir>/<slug>.txt`` (slug per
    :func:`orggraph.agents.prompt_builder._slug`) and attach it via
    :meth:`Persona.with_prebuilt_prompt`. Personas without a prompt
    file fall back to the runtime template.
    """
    df = pd.read_csv(path)
    out: dict[str, Persona] = {}
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name or name.lower() == "nan":
            continue
        out[name] = Persona.from_row(row)

    if prompts_dir is not None:
        # Imported here to avoid a circular import at module load time.
        from orggraph.agents.prompt_builder import _slug

        prompts_dir = Path(prompts_dir)
        if prompts_dir.exists():
            for name, persona in list(out.items()):
                prompt_path = prompts_dir / f"{_slug(name)}.txt"
                if prompt_path.exists():
                    text = prompt_path.read_text().strip()
                    if text:
                        out[name] = persona.with_prebuilt_prompt(text)

    return out


# --- private helpers ---------------------------------------------------


def _safe_str(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v).strip()


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
        if math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_int(v: object, default: int = 0) -> int:
    try:
        return int(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _parse_topics(v: object) -> tuple[str, ...]:
    """``expertise_topics`` is stored as a JSON-ish list literal in CSV."""
    s = _safe_str(v)
    if not s:
        return ()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return tuple(str(x).strip() for x in parsed if str(x).strip())
    except (ValueError, SyntaxError):
        pass
    # Fallback: comma-separated bare string
    return tuple(p.strip() for p in s.split(",") if p.strip())


def _formality_label(level: int) -> str:
    return {
        1: "very informal",
        2: "casual",
        3: "neutral",
        4: "formal",
        5: "very formal",
    }.get(level, "neutral")
