"""Message + Transcript dataclasses for the RQ2 simulation pilot.

A ``Message`` is the unit of inter-agent communication in the A2A
protocol. ``Transcript`` is the append-only log written by both
conditions (multi-agent and single-LLM); it serialises to JSONL so
that two runs of the same scenario produce comparable artifacts that
can be diffed and fed into downstream evaluation.

The wildcard recipient ``"*"`` denotes a broadcast to every registered
agent on the Bus. This mirrors email semantics where "all" is a
distinct addressing concept rather than an enumeration.

Both Message and Transcript are JSON-safe (no Python-only types in
their public fields) so transcripts can be inspected with `jq`,
loaded by other tools, or attached to a thesis appendix.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

BROADCAST = "*"


@dataclass(frozen=True)
class Message:
    """One turn of inter-agent communication.

    Attributes
    ----------
    sender:
        Canonical name of the agent producing the message. Must match a
        ``Persona.name`` in the simulation.
    recipients:
        Tuple of canonical recipient names, or the single-element
        ``("*",)`` for broadcast. Tuple (not list) so the message is
        hashable and frozen-friendly.
    body:
        The textual content. Empty string is allowed (e.g. an
        agent passing its turn).
    turn_id:
        Monotonic integer assigned by the runner; turn_id 0 is the seed
        message if any.
    timestamp:
        Wall-clock time the message was created. ISO-8601 string for
        JSON safety.
    in_reply_to:
        Optional turn_id of the message this one responds to. ``None``
        for opener / broadcast messages with no specific predecessor.
    """

    sender: str
    recipients: tuple[str, ...]
    body: str
    turn_id: int
    timestamp: str
    in_reply_to: int | None = None

    def is_broadcast(self) -> bool:
        return tuple(self.recipients) == (BROADCAST,)

    def is_visible_to(self, agent_name: str) -> bool:
        """Per the A2A protocol: an agent sees a message iff it is
        the sender, an explicit recipient, or the message is a
        broadcast. Self-messages are visible (lets an agent see its
        own outgoing turns when reconstructing context)."""
        if self.is_broadcast():
            return True
        return agent_name == self.sender or agent_name in self.recipients

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "recipients": list(self.recipients),
            "body": self.body,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "in_reply_to": self.in_reply_to,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            sender=d["sender"],
            recipients=tuple(d["recipients"]),
            body=d["body"],
            turn_id=int(d["turn_id"]),
            timestamp=d["timestamp"],
            in_reply_to=d["in_reply_to"],
        )


def now_isoformat() -> str:
    """Wall-clock timestamp in ISO 8601, UTC, second precision."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Transcript:
    """Append-only log of every message produced during a simulation run.

    Both runners (multi-agent and single-LLM) emit the same Transcript
    shape, which is the precondition for direct comparison.

    The ``condition`` and ``scenario_name`` fields tag each transcript
    with enough provenance to identify it after the fact without
    needing the surrounding directory structure.
    """

    scenario_name: str
    condition: str  # "multi_agent" | "single_llm"
    messages: list[Message] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def __len__(self) -> int:
        return len(self.messages)

    def to_jsonl(self, path: Path | str) -> None:
        """Write one JSON object per line. The first line is the
        header (scenario_name, condition, metadata); subsequent lines
        are messages in order."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "_header": True,
                        "scenario_name": self.scenario_name,
                        "condition": self.condition,
                        "metadata": self.metadata,
                    }
                )
                + "\n"
            )
            for m in self.messages:
                f.write(json.dumps(m.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "Transcript":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            header = json.loads(f.readline())
            if not header.get("_header"):
                raise ValueError(f"{path}: first line is not a transcript header")
            t = cls(
                scenario_name=header["scenario_name"],
                condition=header["condition"],
                metadata=header.get("metadata", {}),
            )
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t.messages.append(Message.from_dict(json.loads(line)))
        return t

    def visible_to(self, agent_name: str) -> Iterable[Message]:
        """Yield messages an agent's local memory would have seen,
        in turn order. Used by ``PersonaAgent`` to reconstruct
        per-agent context after restoring a Transcript from disk."""
        for m in self.messages:
            if m.is_visible_to(agent_name):
                yield m
