"""JSONL log of tool invocations during a simulation run.

Sits *alongside* the dialogue Transcript: the Transcript captures
what agents said to each other; the ToolCallLog captures how they
queried the knowledge graph to inform what they said. The two
together are the full evidence of how a simulation ran.

One entry per tool_call. Entries are append-only and JSONL-serialised
so they can be inspected with jq, loaded by analysis notebooks, or
attached to a thesis appendix. Errors propagate (no silent failures).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ToolCallEntry:
    """One observation of an agent invoking a tool."""

    scenario_name: str
    condition: str
    agent: str
    turn_id: int
    tool: str
    args: dict[str, Any]
    result_summary: str
    result_chars: int
    latency_ms: int
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "condition": self.condition,
            "agent": self.agent,
            "turn_id": self.turn_id,
            "tool": self.tool,
            "args": self.args,
            "result_summary": self.result_summary,
            "result_chars": self.result_chars,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCallEntry":
        return cls(
            scenario_name=d["scenario_name"],
            condition=d["condition"],
            agent=d["agent"],
            turn_id=int(d["turn_id"]),
            tool=d["tool"],
            args=d.get("args", {}),
            result_summary=d.get("result_summary", ""),
            result_chars=int(d.get("result_chars", 0)),
            latency_ms=int(d.get("latency_ms", 0)),
            timestamp=d.get("timestamp", _now_iso()),
        )


@dataclass
class ToolCallLog:
    """Append-only log of tool calls observed during a simulation run."""

    entries: list[ToolCallEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def append(self, entry: ToolCallEntry) -> None:
        self.entries.append(entry)

    def to_jsonl(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "ToolCallLog":
        log = cls()
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                log.entries.append(ToolCallEntry.from_dict(json.loads(line)))
        return log

    @staticmethod
    def summarise_result(result: Any, max_chars: int = 240) -> str:
        """Compact one-line summary of a tool result, capped at max_chars."""
        try:
            s = json.dumps(result, default=str, separators=(",", ":"))
        except (TypeError, ValueError):
            s = str(result)
        if len(s) > max_chars:
            s = s[: max_chars - 1] + "…"
        return s
