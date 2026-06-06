"""Headless A2A cascade runner for the team-chat eval.

Wraps ``run_a2a_session_cascade`` (the same dispatcher the Live Demo
Streamlit page uses) in a non-UI callable. Each call produces a
``RunResult`` shaped exactly like the single-agent runner's output, so
the existing grader / report / metrics pipeline consumes A2A runs
without modification — plus an ``a2a_cascade`` dict with routing +
dispatch traces for the team-specific metrics scorer.

The output dict embedded inside ``RunResult`` (via the runner's
``final_body`` text — we put a JSON header on it? No — we put the
cascade metadata on a side channel) is left simple: ``final_body`` is
the synthesized team answer, ``tool_calls`` is the aggregated list
from every agent that fired. The cascade metadata is written to a
sibling JSON file by the runner if it needs it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from orggraph.pipeline.eval.kg_chat.runner import RunResult, ToolCallRecord


# Pull persona_agent (where run_a2a_session_cascade lives) from the
# dashboard/lib path, the same shim team_flow.py uses.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DASHBOARD_LIB = _REPO_ROOT / "dashboard" / "lib"
if str(_DASHBOARD_LIB) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_LIB))

from persona_agent import (  # noqa: E402 — path shim above
    PersonaAgent,
    make_address_colleague_tool,
    run_a2a_session_cascade,
)
from persona_chat import (  # noqa: E402
    DEFAULT_ROSTER_SLUGS,
    build_roster,
    load_persona,
    route,
)


def run_a2a_turn(
    question,
    *,
    registry,                  # full ToolRegistry with all 15 tools
    client,                    # OpenAIChatClient wrapper
    model: str,
    system_prompt: str,        # unused for A2A but kept for runner-compat
    max_tools: int = 10,
    temperature: float = 0.3,
    roster_slugs: list[str] | None = None,
    max_hops: int = 20,
    on_router_tool=None,       # optional callback for live router tool captions
) -> RunResult:
    """Run one question through the A2A cascade headlessly.

    Returns a ``RunResult`` with:
      - ``final_body`` = the originating agent's last reply (the team's
        integrated answer)
      - ``tool_calls`` = aggregated across router + every agent turn,
        namespaced by source agent (e.g. ``router::find_topic_collaborators``,
        ``jeff_dasovich::get_person``)
      - ``wall_time_ms`` covering the full cascade

    The ``a2a_cascade`` attribute (monkey-patched onto the returned
    RunResult) captures the cascade structure for routing/dispatch
    metrics:
      - ``initial_pick``: slug picked by the router
      - ``dispatches``: list of (source_slug, target_slug) edges
      - ``hops``: total agent turns
      - ``final_agent``: slug of the agent whose last bubble holds
        the team answer
    """
    t0 = time.time()
    roster = build_roster(roster_slugs or DEFAULT_ROSTER_SLUGS)
    all_tool_calls: list[ToolCallRecord] = []
    cascade_edges: list[tuple[str, str]] = []

    # 1. Router — pick the initial agent
    slugs, _reasoning, router_trace = route(
        client=client,
        model=model,
        registry=registry,
        user_message=question.question,
        conversation_recap="",
        roster=roster,
        temperature=0.0,
        max_tools=5,
        on_tool_call=on_router_tool,
    )
    for tc in router_trace:
        all_tool_calls.append(ToolCallRecord(
            name=f"router::{tc['name']}",
            args=tc.get("args") or {},
            result_summary=tc.get("result_summary") or "",
            latency_ms=tc.get("latency_ms") or 0,
            error=tc.get("error"),
        ))

    if not slugs:
        result = RunResult(
            question_id=question.id,
            category=question.category,
            question=question.question,
            tool_calls=all_tool_calls,
            final_body="[a2a-flow] router returned no responders",
            wall_time_ms=int((time.time() - t0) * 1000),
        )
        result.a2a_cascade = {
            "initial_pick": None,
            "dispatches": [],
            "hops": 0,
            "final_agent": None,
        }
        return result

    initial_slug = slugs[0]
    cascade_edges.append(("user", initial_slug))

    # 2. Build a PersonaAgent for every roster member so any cascade
    # target works (not just the router's pick).
    # The registry is shared across questions in a single eval run, so
    # only register address_colleague the first time we see it.
    a2a_registry = registry
    if "address_colleague" not in a2a_registry:
        a2a_registry.register(make_address_colleague_tool())
    roster_table = "\n".join(
        f"- `{p.slug}` — {p.display_name} — {p.one_line_role}" for p in roster
    )
    agents: dict[str, PersonaAgent] = {
        p.slug: PersonaAgent(
            slug=p.slug,
            display_name=p.display_name,
            system_prompt=load_persona(p.slug),
            registry=a2a_registry,
            openai_client=client._client,
            model=model,
            roster_table=roster_table,
            max_tools_per_turn=max_tools,
            temperature=temperature,
        )
        for p in roster
    }

    # 3. Drive the cascade. Capture all events for tool aggregation +
    # the final answer (= the originating agent's last reply).
    final_replies_per_slug: dict[str, str] = {}
    hops = {"n": 0}

    def _on_event(slug: str, event: dict, from_slug: str) -> None:
        kind = event.get("kind")
        if kind == "tool_call":
            if event.get("name") == "address_colleague":
                return  # handled via dispatch event
            all_tool_calls.append(ToolCallRecord(
                name=f"{slug}::{event['name']}",
                args=event.get("args") or {},
                result_summary=event.get("result_summary") or "",
                latency_ms=event.get("latency_ms") or 0,
                error=event.get("error"),
            ))
        elif kind == "dispatch":
            for ts in event.get("targets") or []:
                cascade_edges.append((slug, ts))
        elif kind == "done":
            reply = (event.get("reply") or "").strip()
            if reply:
                final_replies_per_slug[slug] = reply
            hops["n"] += 1

    def _on_new_bubble(slug: str, from_slug: str) -> None:
        # We don't need to do anything; on_event captures everything.
        pass

    run_a2a_session_cascade(
        agents=agents,
        initial_targets=[initial_slug],
        user_question=question.question,
        on_event=_on_event,
        on_new_bubble=_on_new_bubble,
        max_hops=max_hops,
        attach_ctx=None,   # headless — no Streamlit context to attach
    )

    # 4. The team's final answer is the initial agent's last reply
    # (their post-cascade synthesis). If they never produced one, fall
    # back to any other agent's reply, then to a placeholder.
    final_body = final_replies_per_slug.get(initial_slug) or ""
    final_agent = initial_slug
    if not final_body:
        for slug, reply in final_replies_per_slug.items():
            if reply:
                final_body = reply
                final_agent = slug
                break
    if not final_body:
        final_body = "[a2a-flow] cascade produced no visible reply"

    result = RunResult(
        question_id=question.id,
        category=question.category,
        question=question.question,
        tool_calls=all_tool_calls,
        final_body=final_body,
        wall_time_ms=int((time.time() - t0) * 1000),
    )
    # Side channel for the team metrics scorer.
    result.a2a_cascade = {
        "initial_pick": initial_slug,
        "dispatches": cascade_edges,           # list of (src, dst)
        "hops": hops["n"],
        "final_agent": final_agent,
        "n_tool_calls": len(all_tool_calls),
    }
    return result
