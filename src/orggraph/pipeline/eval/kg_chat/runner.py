"""Per-question runner for the KG-Chat structural eval.

Given a Question, runs the chat agent's tool-calling loop with
temperature=0 and a fixed tool budget. Captures every tool call,
the final assistant body, latency, and any tool errors. Does NOT
grade — that's the orchestrator's job after the runner returns.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[5] / "dashboard" / "pages" / "4_KG_Chat.py"
SYSTEM_PROMPT_EVAL_SUFFIX = "\n\nAnswer concisely. Do not hedge if the tools give a clear answer."


def load_system_prompt() -> str:
    """Read dashboard SYSTEM_PROMPT and append the eval suffix."""
    text = SYSTEM_PROMPT_PATH.read_text()
    marker = 'SYSTEM_PROMPT = """'
    if marker not in text:
        return (
            "You are an organisational knowledge assistant for Enron Corporation. "
            "Use tools when you need a fact you don't already have."
        ) + SYSTEM_PROMPT_EVAL_SUFFIX
    start = text.index(marker) + len(marker)
    end = text.index('"""', start)
    return text[start:end].rstrip() + SYSTEM_PROMPT_EVAL_SUFFIX


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result_summary: str
    latency_ms: int
    error: str | None


@dataclass
class RunResult:
    question_id: str
    category: str
    question: str
    tool_calls: list[ToolCallRecord]
    final_body: str
    wall_time_ms: int


def run_one(
    question,        # orggraph.pipeline.eval.kg_chat.dataset.Question
    *,
    registry,        # orggraph.pipeline.agents.tools.ToolRegistry
    client,          # orggraph.agents.agent.OpenAIChatClient
    model: str,
    system_prompt: str,
    max_tools: int = 8,
    temperature: float = 0.0,
    on_tool_call=None,  # Callable[[ToolCallRecord], None] | None
) -> RunResult:
    t0 = time.time()
    messages: list[dict] = [{"role": "user", "content": question.question}]
    tools_list = registry.to_openai_tools()
    tool_calls: list[ToolCallRecord] = []
    final_body = ""
    n_calls = 0

    while True:
        tools_arg = tools_list if n_calls < max_tools else None
        try:
            resp = client.chat_with_tools(
                system=system_prompt,
                messages=messages,
                model=model,
                temperature=temperature,
                tools=tools_arg,
            )
        except Exception as exc:  # noqa: BLE001
            # Network / model failure — surface as a runner error.
            final_body = f"[runner-error] chat_with_tools failed: {exc}"
            break

        if resp.is_final or tools_arg is None:
            final_body = (resp.body or "").strip()
            break

        messages.append({
            "role": "assistant",
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in resp.tool_calls
            ],
        })
        for tc in resp.tool_calls:
            args = json.loads(tc["arguments"]) if tc.get("arguments") else {}
            t_call = time.time()
            try:
                result = registry.call(tc["name"], args)
                err = None
                if isinstance(result, dict) and result.get("error"):
                    err = result["error"]
            except Exception as exc:  # noqa: BLE001
                result = {"error": "ToolError", "details": str(exc)}
                err = str(exc)
            latency = int((time.time() - t_call) * 1000)
            summary = _summarise(result)
            record = ToolCallRecord(
                name=tc["name"], args=args, result_summary=summary,
                latency_ms=latency, error=err,
            )
            tool_calls.append(record)
            if on_tool_call is not None:
                try:
                    on_tool_call(record)
                except Exception:  # noqa: BLE001 — callback errors must not break the loop
                    pass
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["name"],
                "content": json.dumps(result, default=str),
            })
            n_calls += 1

    return RunResult(
        question_id=question.id,
        category=question.category,
        question=question.question,
        tool_calls=tool_calls,
        final_body=final_body,
        wall_time_ms=int((time.time() - t0) * 1000),
    )


def _summarise(result: Any, max_chars: int = 240) -> str:
    s = json.dumps(result, default=str)
    return s[: max_chars - 1] + "…" if len(s) > max_chars else s
