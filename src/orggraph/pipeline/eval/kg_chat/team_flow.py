"""Headless team-discussion flow for the KG-Chat eval.

Mirrors the Streamlit Team Chat page behaviour (router → multi-round
persona discussion → moderator synthesis) but as a single callable
returning a ``RunResult`` shaped the same as the single-agent
``runner.run_one`` output. That lets the existing grader / report /
metrics infrastructure consume team runs without modification.

Why a sys.path shim
-------------------
The team-chat helpers live in ``dashboard/lib/persona_chat.py`` because
that's where the Streamlit page imports from. We pull them in by
adding ``<repo-root>/dashboard/lib`` to ``sys.path``. Cleaner option
would be to move the helpers into the package proper, but that's a
larger refactor — this flag lets us measure first and refactor later
if the team flow proves valuable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from orggraph.pipeline.eval.kg_chat.runner import RunResult, ToolCallRecord


# Add dashboard/lib to sys.path so we can import persona_chat without
# turning the dashboard into an installed package. parents[5] is the
# repo root (src/orggraph/pipeline/eval/kg_chat/team_flow.py).
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DASHBOARD_LIB = _REPO_ROOT / "dashboard" / "lib"
if str(_DASHBOARD_LIB) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_LIB))

from persona_chat import (  # noqa: E402  (path shim above)
    DEFAULT_ROSTER_SLUGS,
    build_discussion_prompt,
    build_roster,
    route,
    stream_persona_turn,
    synthesise_team_answer,
)


def _drain_persona_stream(*, persona_slug: str, user_prompt: str,
                          openai_client, model: str, registry,
                          max_tools: int, temperature: float) -> dict:
    """Consume ``stream_persona_turn`` for one persona and return the
    final ``done`` event payload (final_body, tool_calls, wall_time_ms).

    The generator yields content / reasoning / tool_call events along
    the way; we discard those and just take the last one. This is how
    we re-use the streaming function in a non-streaming context.
    """
    final_event: dict | None = None
    for event in stream_persona_turn(
        persona_slug=persona_slug,
        user_prompt=user_prompt,
        openai_client=openai_client,
        model=model,
        registry=registry,
        max_tools=max_tools,
        temperature=temperature,
    ):
        if event.get("kind") == "done":
            final_event = event
    return final_event or {"final_body": "", "tool_calls": [], "wall_time_ms": 0}


def run_team_turn(
    question,
    *,
    registry,
    client,                 # OpenAIChatClient wrapper
    model: str,
    system_prompt: str,     # unused for team mode but kept for runner-compat
    max_tools: int = 8,
    temperature: float = 0.0,
    n_rounds: int = 2,
    k_responders: int = 3,
    roster_slugs: list[str] | None = None,
) -> RunResult:
    """Run one question through the team-discussion flow.

    Returns a ``RunResult`` with:
      - ``final_body`` set to the moderator's synthesised answer
      - ``tool_calls`` aggregated across router + every persona turn
      - ``wall_time_ms`` covering the entire turn

    The synthesised answer is what the grader judges, so team-mode
    results are directly comparable to single-agent results.
    """
    t0 = time.time()
    openai_client = client._client  # raw OpenAI client for streaming
    roster = build_roster(roster_slugs or DEFAULT_ROSTER_SLUGS)
    all_tool_calls: list[ToolCallRecord] = []

    # 1. Router — pick K personas
    slugs, _reasoning, router_trace = route(
        client=client,
        model=model,
        registry=registry,
        user_message=question.question,
        conversation_recap="",  # eval is one-shot, no prior thread
        roster=roster,
        temperature=temperature,
        max_tools=max(5, max_tools),
    )
    for tc in router_trace:
        all_tool_calls.append(ToolCallRecord(
            name=f"router::{tc['name']}",
            args=tc.get("args") or {},
            result_summary=tc.get("result_summary") or "",
            latency_ms=tc.get("latency_ms") or 0,
            error=tc.get("error"),
        ))
    slugs = slugs[:k_responders]

    # Routing failure — fall back so the question still gets answered
    if not slugs:
        return RunResult(
            question_id=question.id,
            category=question.category,
            question=question.question,
            tool_calls=all_tool_calls,
            final_body="[team-flow] router returned no responders",
            wall_time_ms=int((time.time() - t0) * 1000),
        )

    # 2. Multi-round discussion — each picked persona speaks per round
    discussion: list[dict] = []
    for round_idx in range(n_rounds):
        is_last = (round_idx == n_rounds - 1)
        for slug in slugs:
            prompt_for_persona = build_discussion_prompt(
                user_question=question.question,
                discussion_so_far=discussion,
                current_speaker_slug=slug,
                is_last_round=is_last,
            )
            turn = _drain_persona_stream(
                persona_slug=slug,
                user_prompt=prompt_for_persona,
                openai_client=openai_client,
                model=model,
                registry=registry,
                max_tools=max_tools,
                temperature=temperature,
            )
            # Aggregate persona tool calls, namespaced by slug
            for tc in turn.get("tool_calls") or []:
                all_tool_calls.append(ToolCallRecord(
                    name=f"{slug}::{tc.get('name','?')}",
                    args=tc.get("args") or {},
                    result_summary=tc.get("result_summary") or "",
                    latency_ms=tc.get("latency_ms") or 0,
                    error=tc.get("error"),
                ))
            discussion.append({
                "slug": slug,
                "content": turn.get("final_body") or "",
            })

    # 3. Synthesis — moderator collapses discussion into a single answer
    final_body = synthesise_team_answer(
        client=client,
        model=model,
        user_question=question.question,
        discussion=discussion,
        temperature=0.0,
    )

    return RunResult(
        question_id=question.id,
        category=question.category,
        question=question.question,
        tool_calls=all_tool_calls,
        final_body=final_body or "",
        wall_time_ms=int((time.time() - t0) * 1000),
    )
