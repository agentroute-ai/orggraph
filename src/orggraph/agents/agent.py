"""PersonaAgent — the multi-agent condition's primitive.

Each ``PersonaAgent`` wraps one ``Persona`` plus an LLM client. It
maintains its own conversation history (only messages it sent or
received), and ``respond`` makes a single LLM call to produce the
next message in the dialogue.

The LLM dependency is expressed as the ``TextChatClient`` Protocol so
tests can substitute a deterministic mock. A thin production
implementation that wraps ``openai.OpenAI`` lives in
``OpenAIChatClient`` below.
"""

from __future__ import annotations

import json as _json_mod
import os
import re as _re
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from orggraph.agents.persona import Persona, serialize_persona
from orggraph.simulation.transcript import Message, now_isoformat


# Tunable: max tool_calls per turn before the runner forces a final response.
# Capped at 5: vLLM-served MiniMax was hitting a crash on long tool-call
# sequences (likely context overrun from accumulated tool results). The
# catalog has 21 tools but the model has to compose a final answer from
# what it can fit, so 5 is the safe ceiling for this serving stack.
MAX_TOOLS_PER_TURN = 5

# MiniMax (and a few other models served via vLLM) sometimes emit tool calls
# as XML inside the response body instead of populating the structured
# `tool_calls` field. This happens when the vLLM server isn't started with
# a model-specific `--tool-call-parser`. Pattern is:
#   <invoke name="tool_name">
#   <parameter name="arg1">value1</parameter>
#   <parameter name="arg2">value2</parameter>
#   </invoke>
# (Often wrapped in <minimax:tool_call>...</minimax:tool_call> but the
# wrapper isn't load-bearing — we look for <invoke> directly.)
_INVOKE_RE = _re.compile(
    r"<invoke\s+name=\"([^\"]+)\">(.*?)</invoke>",
    _re.DOTALL,
)
_PARAM_RE = _re.compile(
    r"<parameter\s+name=\"([^\"]+)\">(.*?)</parameter>",
    _re.DOTALL,
)


_TOOL_CALL_BLOCK_RE = _re.compile(
    r"<(?:minimax:)?tool_call>.*?</(?:minimax:)?tool_call>",
    _re.DOTALL,
)
_INVOKE_BLOCK_RE = _re.compile(
    r"<invoke\s+name=\"[^\"]+\">.*?</invoke>",
    _re.DOTALL,
)


def _strip_xml_tool_calls(body: str) -> str:
    """Remove leftover <invoke>/<tool_call> markup from a final-answer body.

    Used when the caller asked for a plain text answer (tools=None) but the
    model still tried to emit tool-call XML. Anything outside the markup is
    returned as-is so we don't lose actual prose the model wrote alongside.
    """
    if not body:
        return body
    cleaned = _TOOL_CALL_BLOCK_RE.sub("", body)
    cleaned = _INVOKE_BLOCK_RE.sub("", cleaned)
    return cleaned.strip()


def _parse_xml_tool_calls(body: str) -> list[dict] | None:
    """Parse MiniMax-style ``<invoke>`` tool calls out of a response body.

    Returns the same shape as ``OpenAIChatClient.chat_with_tools`` would
    produce from a properly-structured ``tool_calls`` field — a list of
    ``{"id", "name", "arguments"}`` dicts. Returns ``None`` if no
    invocation markup is present so callers can treat the body as a
    final answer.

    Parameter values get a tiny type-coercion pass: pure-int strings
    become ints, pure-float strings become floats, everything else stays
    a string. This matches what the underlying tool callables expect
    (``n: int`` vs ``name: str``) without requiring a schema lookup.
    """
    invokes = _INVOKE_RE.findall(body or "")
    if not invokes:
        return None

    out: list[dict] = []
    for tool_name, inner in invokes:
        args: dict[str, Any] = {}
        for param_name, raw_value in _PARAM_RE.findall(inner):
            value = raw_value.strip()
            # Try int, then float, otherwise keep as string.
            try:
                args[param_name] = int(value)
                continue
            except ValueError:
                pass
            try:
                args[param_name] = float(value)
                continue
            except ValueError:
                pass
            args[param_name] = value

        out.append({
            "id": f"call_{_uuid.uuid4().hex[:12]}",
            "name": tool_name.strip(),
            "arguments": _json_mod.dumps(args),
        })
    return out


@dataclass(frozen=True)
class ToolCallResponse:
    """One assistant response from chat_with_tools.

    Either a list of tool_calls (the runner dispatches each, then loops
    back into chat_with_tools with the tool results) or a final body
    (the loop terminates and the body becomes the agent's turn).
    """

    tool_calls: list[dict] | None  # [{"id", "name", "arguments": json_str}, ...]
    body: str

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


class TextChatClient(Protocol):
    """Anything that takes a system prompt + a list of (role, content)
    chat messages and returns either a final assistant body (chat) or
    a tool-using exchange (chat_with_tools)."""

    def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str: ...

    def chat_with_tools(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        temperature: float,
        tools: list[dict] | None,
    ) -> ToolCallResponse: ...


@dataclass
class PersonaAgent:
    """A persona-grounded LLM agent for the multi-agent condition.

    Attributes
    ----------
    persona:
        The persona used to build the system prompt.
    client:
        Anything implementing ``TextChatClient``.
    model:
        Model identifier passed to the client.
    temperature:
        Sampling temperature; the runner is responsible for keeping
        this matched across both conditions.
    history:
        Messages this agent has seen (sent or received). Maintained by
        ``receive``.
    """

    persona: Persona
    client: TextChatClient
    model: str
    temperature: float = 0.7
    history: list[Message] = field(default_factory=list)
    # When set, respond() runs the multi-step function-calling loop instead
    # of plain chat(). Tools is ToolRegistry-like (avoids import cycle in
    # type), tool_call_log is ToolCallLog-like.
    tools: Any = None
    tool_call_log: Any = None

    @property
    def name(self) -> str:
        return self.persona.name

    @property
    def system_prompt(self) -> str:
        base = serialize_persona(self.persona)
        if self.tools is None:
            return base
        return base + self._tools_help_section()

    def _tools_help_section(self) -> str:
        """Append a brief description of available tools to the persona prompt.

        Iterates self.tools (a ToolRegistry) and emits one bullet per tool.
        The 'Don't fabricate' line directly targets the failure modes seen
        in earlier RQ2 pilot runs (imaginary participants; made-up emails).
        """
        bullets = "\n".join(f"- {t.name}: {t.description}" for t in self.tools)
        return (
            "\n\n## Tools you can use mid-response\n"
            "Before composing your reply, you may call any of these to retrieve "
            "grounded facts from the organisational graph:\n\n"
            f"{bullets}\n\n"
            "Use them when you need a fact you don't already have. "
            "Don't fabricate."
        )

    # --- inbound ---------------------------------------------------

    def receive(self, message: Message) -> None:
        """Append a message to the agent's local memory.

        Called by the Bus when a message is delivered. The runner is
        responsible for ensuring delivery order matches turn order.
        """
        self.history.append(message)

    def reset(self) -> None:
        """Clear local memory (used between scenarios in batch runs)."""
        self.history.clear()

    # --- outbound --------------------------------------------------

    def respond(
        self,
        scenario_brief: str,
        *,
        recipients: tuple[str, ...],
        turn_id: int,
        in_reply_to: int | None = None,
    ) -> Message:
        """Produce the next message by invoking the LLM.

        Parameters
        ----------
        scenario_brief:
            Short scene-setting paragraph injected at the top of the
            user-side context. Same brief is used by the single-LLM
            baseline for fair comparison.
        recipients:
            Who the produced message is addressed to. The agent does
            not pick recipients itself in the pilot — the runner
            decides routing per scenario configuration. (Letting
            agents pick recipients is a future-work item; locking it
            here keeps the pilot reproducible.)
        turn_id, in_reply_to:
            Set on the returned ``Message``.
        """
        prior = self._render_history()
        user_content = (
            f"Scenario brief: {scenario_brief}\n\n"
            f"Conversation so far:\n{prior}\n\n"
            f"It is now your turn ({self.persona.name}). "
            f"Reply in character with one message addressed to "
            f"{', '.join(recipients) if recipients != ('*',) else 'everyone'}. "
            f"Do not narrate; speak directly. If the conversation has reached "
            f"its natural conclusion, reply with the single word END."
        )
        if self.tools is not None:
            user_content += (
                "\n\nIMPORTANT: You have function-calling tools that query the "
                "organisation's knowledge graph (Person records, threads, "
                "topics, recent activity, mentions, semantic search over the "
                "email corpus). Before composing your reply, decide whether "
                "calling any tool would let you ground your message in concrete "
                "facts from the organisation's history rather than generic "
                "professional pleasantries. Realistic anchors include: the "
                "recipient's recent threads with you, their current projects, "
                "who has been mentioned alongside them on the topic at hand, "
                "or specific past emails that bear on the decision. When in "
                "doubt about whether to call a tool, call it — grounded "
                "responses are preferred. If you have already gathered enough "
                "context from prior turns or the tools you just called, "
                "proceed to write the email."
            )

        if self.tools is None:
            body = self.client.chat(
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_content}],
                model=self.model,
                temperature=self.temperature,
            ).strip()
        else:
            body = self._run_tool_loop(user_content=user_content, turn_id=turn_id).strip()
            # Safety net: if the tool loop returned an empty body (the
            # model exhausted MAX_TOOLS_PER_TURN and then emitted only
            # XML tool-call markup that got stripped), retry once with a
            # plain chat call and no tools attached. This costs at most
            # one extra LLM call per failing turn but prevents
            # zero-content turns from leaking into the transcript.
            if not body:
                body = self.client.chat(
                    system=self.system_prompt,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"{user_content}\n\n"
                            "Write your email reply now as plain prose. Do "
                            "not call any tools, do not emit tool_call markup."
                        ),
                    }],
                    model=self.model,
                    temperature=self.temperature,
                ).strip()

        return Message(
            sender=self.persona.name,
            recipients=tuple(recipients),
            body=body,
            turn_id=turn_id,
            timestamp=now_isoformat(),
            in_reply_to=in_reply_to,
        )

    def _run_tool_loop(self, *, user_content: str, turn_id: int) -> str:
        """Multi-step function-calling loop.

        The model can emit tool_calls; the runner dispatches them via
        ``self.tools.call`` and feeds results back as tool messages.
        Loop terminates on a final response (no tool_calls) or when
        MAX_TOOLS_PER_TURN is hit (next call passes tools=None to
        force a final response).
        """
        # Imports are inside the function to avoid a circular import:
        # tools_logging is in orggraph.pipeline.agents which depends on
        # nothing in orggraph.agents — but we still keep them local for
        # parity with the design and to keep the module's top imports
        # focused on type definitions.
        import json as _json
        import time as _time

        from orggraph.pipeline.agents.tools_logging import ToolCallEntry, ToolCallLog

        tools_list = self.tools.to_openai_tools()
        messages: list[dict] = [{"role": "user", "content": user_content}]
        n_tool_calls = 0

        budget_nudge_sent = False
        while True:
            tools_arg = tools_list if n_tool_calls < MAX_TOOLS_PER_TURN else None
            # When the per-turn tool budget is exhausted we force a final
            # response by passing tools=None. The model is heavily prompted
            # to use tools (by the per-turn nudge added to user_content), so
            # without an explicit instruction it tends to keep emitting XML
            # tool_calls in the content field, which `_strip_xml_tool_calls`
            # then strips to an empty string and the agent writes an empty
            # turn. Inject a one-shot user-role nudge once, immediately
            # before the forced-final call, to break the loop cleanly.
            if tools_arg is None and not budget_nudge_sent:
                messages.append({
                    "role": "user",
                    "content": (
                        "STOP. You have used the tool budget for this turn. "
                        "Do not call any more tools. Write your email reply "
                        "now as plain prose, using the information you have "
                        "already retrieved. Do not emit XML-style "
                        "tool_call markup. Begin the reply directly."
                    ),
                })
                budget_nudge_sent = True
            resp = self.client.chat_with_tools(
                system=self.system_prompt,
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                tools=tools_arg,
            )
            if resp.is_final or tools_arg is None:
                return resp.body

            # Append the assistant's tool_calls turn before the tool results.
            # The OpenAI message schema requires tool_calls in the wrapped
            # format {id, type: "function", function: {name, arguments}}; the
            # flat shape returned by chat_with_tools is converted here.
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in resp.tool_calls
                ],
            })

            for tc in resp.tool_calls:
                args = _json.loads(tc["arguments"]) if tc["arguments"] else {}
                t0 = _time.time()
                try:
                    result = self.tools.call(tc["name"], args)
                except Exception as e:  # noqa: BLE001
                    result = {"error": str(e)}
                latency_ms = int((_time.time() - t0) * 1000)

                if self.tool_call_log is not None:
                    self.tool_call_log.append(ToolCallEntry(
                        scenario_name="",
                        condition="",
                        agent=self.persona.name,
                        turn_id=turn_id,
                        tool=tc["name"],
                        args=args,
                        result_summary=ToolCallLog.summarise_result(result),
                        result_chars=len(_json.dumps(result, default=str)),
                        latency_ms=latency_ms,
                    ))
                n_tool_calls += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": _json.dumps(result, default=str),
                })

    def _render_history(self) -> str:
        if not self.history:
            return "(no messages yet — you are opening the conversation)"
        lines = []
        for m in self.history:
            tag = "[broadcast]" if m.is_broadcast() else f"[to {', '.join(m.recipients)}]"
            lines.append(f"{m.sender} {tag}: {m.body}")
        return "\n".join(lines)


# --- production OpenAI-compatible client ---------------------------------


class OpenAIChatClient:
    """Thin TextChatClient implementation over ``openai.OpenAI``.

    Used in the pilot's live runs (against Ollama or vLLM). Tests use
    a mock instead.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 600.0) -> None:
        from openai import OpenAI  # imported here so the test path doesn't need openai

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.timeout = timeout

    def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int = 16384,
    ) -> str:
        """Call the chat API and return the message content.

        ``max_tokens`` defaults to 16384 to accommodate reasoning models
        (MiniMax-M2.7, Qwen-QwQ, etc.) that consume a large slice of the
        completion budget on hidden reasoning before producing visible
        content. With vLLM's small default cap, reasoning models silently
        return empty content; 4096 is enough for ~400-word persona
        prompts plus reasoning.
        """
        full = [{"role": "system", "content": system}, *messages]
        resp = self._client.chat.completions.create(
            model=model,
            messages=full,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content or ""

    def chat_with_tools(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        temperature: float,
        tools: list[dict] | None,
        max_tokens: int = 8192,
    ) -> ToolCallResponse:
        """Tool-aware variant that uses the OpenAI function-calling protocol.

        When ``tools`` is non-empty, sets ``tool_choice="auto"`` so the
        model decides whether to call any. When ``tools`` is None or
        empty, this is equivalent to a plain chat call returning a final
        body — used by the runner to force a final response when the
        per-turn tool budget is exhausted.

        ``max_tokens`` defaults to 8192 because reasoning models (MiniMax,
        Qwen-QwQ, etc.) spend thousands of tokens on hidden ``<think>``
        blocks before producing visible content. With vLLM's small default
        the visible body silently truncates to empty.
        """
        full = [{"role": "system", "content": system}, *messages]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": full,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        # Debug trace: when RQ2_TRACE_DIR is set, dump the request + response
        # to per-call JSON files so we can inspect whether tools were offered,
        # whether the model emitted tool_calls, and what content came back.
        _trace_dir = os.environ.get("RQ2_TRACE_DIR")
        if _trace_dir:
            from pathlib import Path as _Path
            import time as _time
            import json as _json
            import uuid as _uuid
            _Path(_trace_dir).mkdir(parents=True, exist_ok=True)
            _trace_id = f"{int(_time.time() * 1000)}_{_uuid.uuid4().hex[:6]}"
            _trace_path = _Path(_trace_dir) / f"call_{_trace_id}.json"
            _req_dump = {
                "model": kwargs.get("model"),
                "temperature": kwargs.get("temperature"),
                "tools_offered": [t["function"]["name"] for t in (tools or [])],
                "n_tools_offered": len(tools or []),
                "tool_choice": kwargs.get("tool_choice"),
                "system_chars": len(system or ""),
                "messages": [
                    {"role": m.get("role"), "content_chars": len(str(m.get("content") or "")), "content_preview": str(m.get("content") or "")[:500]}
                    for m in full
                ],
            }
            _trace_path.write_text(_json.dumps({"request": _req_dump}, indent=2))
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        body = msg.content or ""
        if _trace_dir:
            _resp_dump = {
                "finish_reason": resp.choices[0].finish_reason,
                "role": msg.role,
                "content": body[:2000] if body else None,
                "tool_calls": [
                    {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in (getattr(msg, "tool_calls", None) or [])
                ],
            }
            try:
                _existing = _json.loads(_trace_path.read_text())
            except Exception:
                _existing = {}
            _existing["response"] = _resp_dump
            _trace_path.write_text(_json.dumps(_existing, indent=2))

        # When the caller passed tools=None they want a final answer, not more
        # tool calls. Some models (e.g. MiniMax under vLLM without the right
        # tool-call parser) still emit tool_calls or XML invocations even
        # though no tools are available. Treat that as a final body and strip
        # any leftover invocation markup so the user sees readable text.
        if not tools:
            return ToolCallResponse(
                tool_calls=None, body=_strip_xml_tool_calls(body),
            )

        tcs = getattr(msg, "tool_calls", None) or None
        if tcs:
            return ToolCallResponse(
                tool_calls=[
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in tcs
                ],
                body="",
            )
        # Fallback: vLLM serving MiniMax (or similar) without a model-specific
        # `--tool-call-parser` flag emits tool calls as XML inside the content
        # body instead of the structured tool_calls field. Catch that here so
        # the loop can still dispatch them.
        parsed = _parse_xml_tool_calls(body)
        if parsed:
            return ToolCallResponse(tool_calls=parsed, body="")
        return ToolCallResponse(tool_calls=None, body=body)
