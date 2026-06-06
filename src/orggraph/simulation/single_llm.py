"""Single-LLM baseline — the comparison condition for RQ2.

In this condition one model receives the union of all agent personas
(stitched into a single system prompt) plus the full conversation
history, and is asked to produce the next message in the voice of
whichever participant is scheduled to speak. This is the single-LLM
long-context baseline against which the multi-agent A2A condition is
compared.

To keep both conditions comparable:

* **Same scenario brief** is prepended to every turn.
* **Same model + temperature** must be passed in by the caller.
* **Same Transcript shape** is emitted (one Message per turn,
  ``condition="single_llm"``), so the two transcripts diff cleanly.
* **Same termination conditions** apply (``max_turns`` or an exact
  ``END`` body from the model).
"""

from __future__ import annotations

from orggraph.agents.agent import TextChatClient
from orggraph.agents.persona import Persona, serialize_persona
from orggraph.simulation.runner import END_TOKEN
from orggraph.simulation.scenario import Scenario
from orggraph.simulation.transcript import (
    BROADCAST,
    Message,
    Transcript,
    now_isoformat,
)


def stitch_personas(
    personas: list[Persona],
    *,
    truncate_role: bool = True,
    max_samples_per_persona: int = 2,
) -> str:
    """Serialise multiple personas into one system prompt.

    Each persona block is delimited by a heading so the model can keep
    them apart. ``role_summary`` is truncated by default to keep the
    combined prompt under typical 8K context windows even with 6
    participants — see :data:`orggraph.agents.persona.ROLE_SUMMARY_TRUNC`.

    ``max_samples_per_persona`` caps how many writing samples each
    persona contributes to the stitched prompt. Default 2 (vs the
    typical 3 in single-agent mode) keeps total prompt size under
    control when 6 personas are present. Set to 0 to omit samples
    entirely from the stitched form.
    """
    if not personas:
        raise ValueError("stitch_personas requires at least one persona")

    blocks = ["You are role-playing the following participants. Stay in character "
              "for whichever participant the user asks you to speak as next. "
              "Speak ONLY as the participant the user names; do not address "
              "your own name or sign with your own name in messages you send.\n"]
    for p in personas:
        block = serialize_persona(
            p, truncate_role=truncate_role, max_samples=max_samples_per_persona,
        )
        blocks.append(f"### {p.name}\n{block}")
    return "\n\n".join(blocks)


def render_history(messages: list[Message]) -> str:
    """Render the full conversation transcript for the model's user
    context. Both senders and recipients are surfaced so the model
    can see who was being addressed at each turn."""
    if not messages:
        return "(no messages yet — this is the opening turn)"
    lines = []
    for m in messages:
        tag = "[broadcast]" if m.is_broadcast() else f"[to {', '.join(m.recipients)}]"
        lines.append(f"{m.sender} {tag}: {m.body}")
    return "\n".join(lines)


def run_single_llm(
    scenario: Scenario,
    personas: dict[str, Persona],
    client: TextChatClient,
    *,
    model: str,
    temperature: float = 0.7,
) -> Transcript:
    """Run the single-LLM-with-long-context baseline for a scenario.

    Parameters
    ----------
    scenario:
        Same scenario object the multi-agent runner consumes — enables
        running both conditions on identical input.
    personas:
        ``name → Persona`` registry; must contain every
        ``scenario.participants`` entry.
    client:
        Anything implementing :class:`orggraph.agents.agent.TextChatClient`.
    model, temperature:
        Must match the values used in the multi-agent run for fair
        comparison.

    Returns
    -------
    Transcript with ``condition="single_llm"`` and the same shape the
    multi-agent runner produces.
    """
    missing = [n for n in scenario.participants if n not in personas]
    if missing:
        raise KeyError(
            f"Scenario {scenario.name!r} requires personas not in registry: {missing}"
        )

    participating = [personas[n] for n in scenario.participants]
    system_prompt = stitch_personas(participating)

    transcript = Transcript(
        scenario_name=scenario.name,
        condition="single_llm",
        metadata={
            "flow": scenario.flow,
            "participants": list(scenario.participants),
            "max_turns": scenario.max_turns,
            "model": model,
            "temperature": temperature,
        },
    )

    history: list[Message] = []
    turn_id = 0
    last_msg_id: int | None = None

    if scenario.seed_message:
        seed = Message(
            sender=scenario.starter,
            recipients=(BROADCAST,),
            body=scenario.seed_message,
            turn_id=turn_id,
            timestamp=now_isoformat(),
            in_reply_to=None,
        )
        history.append(seed)
        transcript.append(seed)
        last_msg_id = turn_id
        turn_id += 1

    starter_idx = scenario.participants.index(scenario.starter)
    n = len(scenario.participants)
    speaker_idx = starter_idx if not scenario.seed_message else (starter_idx + 1) % n

    while turn_id < scenario.max_turns:
        speaker_name = scenario.participants[speaker_idx]
        prior = render_history(history)
        user_content = (
            f"Scenario brief: {scenario.brief}\n\n"
            f"Conversation so far:\n{prior}\n\n"
            f"It is now the turn of: {speaker_name}. "
            f"Reply in character as {speaker_name} with one message addressed to "
            f"the other participants. Do not narrate; speak directly. "
            f"If the conversation has reached its natural conclusion, "
            f"reply with the single word END."
        )

        body = client.chat(
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            model=model,
            temperature=temperature,
        ).strip()

        msg = Message(
            sender=speaker_name,
            recipients=(BROADCAST,),
            body=body,
            turn_id=turn_id,
            timestamp=now_isoformat(),
            in_reply_to=last_msg_id,
        )
        history.append(msg)
        transcript.append(msg)

        if msg.body.strip() == END_TOKEN:
            break

        last_msg_id = turn_id
        turn_id += 1
        speaker_idx = (speaker_idx + 1) % n

    return transcript
