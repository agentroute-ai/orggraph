"""Multi-agent runner — the A2A condition's orchestration loop.

Given a scenario and a registry of ``PersonaAgent`` instances, run a
round-robin dialogue: each turn one agent emits a message, the Bus
delivers it to the appropriate recipients (other participants or
broadcast), agents update their local memory, and the runner appends
the message to the Transcript. The loop terminates when ``max_turns``
is reached or any agent emits the literal body ``END``.

Termination sentinel: an exact-match ``"END"`` body. We don't try to
detect natural conversation ends with a classifier — that's future
work and would muddy the pilot's reproducibility story.

This is the multi-agent (A2A) condition. The single-LLM baseline
lives in :mod:`orggraph.simulation.single_llm` and emits a Transcript
of the same shape so the two can be compared directly.
"""

from __future__ import annotations

from orggraph.agents.agent import PersonaAgent
from orggraph.simulation.a2a import Bus
from orggraph.simulation.scenario import Scenario
from orggraph.simulation.transcript import (
    BROADCAST,
    Message,
    Transcript,
    now_isoformat,
)

END_TOKEN = "END"


def run_multi_agent(
    scenario: Scenario,
    agents: dict[str, PersonaAgent],
    *,
    condition: str = "multi_agent",
) -> Transcript:
    """Run the A2A condition for a scenario.

    Parameters
    ----------
    scenario:
        Configured scenario (participants, starter, brief, max_turns).
    agents:
        Mapping ``canonical_name → PersonaAgent``. Must include every
        ``scenario.participants`` entry. Agents not in
        ``scenario.participants`` are ignored (lets a single
        registry serve multiple scenarios).
    condition:
        Label written into the transcript header. Defaults to
        ``"multi_agent"``; the pilot orchestrator overrides this to
        ``"multi_agent_tools"`` when the agents have the KG tool
        registry attached, so downstream judges and aggregators can
        keep the two variants apart on disk.

    Returns
    -------
    Transcript tagged with ``condition`` (default ``"multi_agent"``)
    and one Message per turn (or fewer if an agent emitted END).

    Raises
    ------
    KeyError: a scenario participant has no agent in the registry.
    """
    _check_participants(scenario, agents)

    bus = Bus()
    for name in scenario.participants:
        agent = agents[name]
        agent.reset()
        bus.register(name, agent.receive)

    transcript = Transcript(
        scenario_name=scenario.name,
        condition=condition,
        metadata={
            "flow": scenario.flow,
            "participants": list(scenario.participants),
            "max_turns": scenario.max_turns,
        },
    )

    turn_id = 0
    last_msg_id: int | None = None

    # Optional seed message: dispatched as a broadcast from the
    # starter so every participant sees it identically. If empty,
    # the starter generates its own opener via the LLM.
    if scenario.seed_message:
        seed = Message(
            sender=scenario.starter,
            recipients=(BROADCAST,),
            body=scenario.seed_message,
            turn_id=turn_id,
            timestamp=now_isoformat(),
            in_reply_to=None,
        )
        bus.dispatch(seed)
        transcript.append(seed)
        last_msg_id = turn_id
        turn_id += 1

    starter_idx = scenario.participants.index(scenario.starter)
    n = len(scenario.participants)

    # Round-robin from the starter (skipping a turn if seed already taken).
    speaker_idx = starter_idx if not scenario.seed_message else (starter_idx + 1) % n

    while turn_id < scenario.max_turns:
        speaker_name = scenario.participants[speaker_idx]
        speaker = agents[speaker_name]

        # Default routing: broadcast to the rest of the participants.
        # In the pilot we keep this fixed; future work can let agents
        # pick recipients dynamically.
        msg = speaker.respond(
            scenario_brief=scenario.brief,
            recipients=(BROADCAST,),
            turn_id=turn_id,
            in_reply_to=last_msg_id,
        )
        bus.dispatch(msg)
        transcript.append(msg)

        if msg.body.strip() == END_TOKEN:
            break

        last_msg_id = turn_id
        turn_id += 1
        speaker_idx = (speaker_idx + 1) % n

    return transcript


# --- helpers -----------------------------------------------------------


def _check_participants(scenario: Scenario, agents: dict[str, PersonaAgent]) -> None:
    missing = [n for n in scenario.participants if n not in agents]
    if missing:
        raise KeyError(
            f"Scenario {scenario.name!r} requires agents not in registry: {missing}"
        )
