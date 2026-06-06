"""Agent demo - replay of recorded multi-agent dialogues (no live model).

This page replays dialogues that were generated offline by the OrgGraph
simulation runtime, including the knowledge-graph tool calls the agents made.
Nothing here calls an LLM or a database at runtime - it reads committed
transcripts, so it runs anywhere the static dashboard does.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from lib.header import render_header

st.set_page_config(page_title="Agent demo - OrgGraph", layout="wide")
render_header(
    title="Agent demo (replay)",
    subtitle="Recorded multi-agent dialogues with their knowledge-graph tool calls.",
)

TRANSCRIPTS = Path(__file__).resolve().parents[2] / "data" / "rq2" / "transcripts"

COND_LABEL = {
    "multi_agent_tools": "Multi-agent + KG tools",
    "multi_agent": "Multi-agent (personas only)",
    "single_llm": "Single LLM (long context)",
}
AGENT_AVATAR = "🧑‍💼"


@st.cache_data(show_spinner=False)
def _scenarios() -> list[str]:
    if not TRANSCRIPTS.is_dir():
        return []
    names = set()
    for f in TRANSCRIPTS.glob("*.jsonl"):
        if "tool_calls" in f.name:
            continue
        names.add(f.name.split("__")[0])
    return sorted(names)


@st.cache_data(show_spinner=False)
def _load(scenario: str, condition: str):
    tpath = TRANSCRIPTS / f"{scenario}__{condition}.jsonl"
    if not tpath.is_file():
        return None, [], {}
    lines = [json.loads(ln) for ln in tpath.read_text().splitlines() if ln.strip()]
    header = next((ln for ln in lines if ln.get("_header")), {})
    turns = [ln for ln in lines if not ln.get("_header")]
    # tool calls grouped by turn_id
    calls: dict[int, list] = {}
    cpath = TRANSCRIPTS / f"{scenario}__{condition}__tool_calls.jsonl"
    if cpath.is_file():
        for ln in cpath.read_text().splitlines():
            if not ln.strip():
                continue
            tc = json.loads(ln)
            calls.setdefault(tc.get("turn_id"), []).append(tc)
    return header, turns, calls


scenarios = _scenarios()
if not scenarios:
    st.warning("No transcripts shipped under `data/rq2/transcripts/`.")
    st.stop()

st.caption(
    "Replay only - these dialogues were generated offline. The live agent chat runs "
    "against a local model + Neo4j and is kept for the defence demo."
)

c = st.columns([2, 2, 1])
scenario = c[0].selectbox("Scenario", scenarios)
condition = c[1].selectbox("Condition", list(COND_LABEL), format_func=lambda x: COND_LABEL[x])
animate = c[2].toggle("Animate", value=False, help="Reveal the dialogue turn by turn")

header, turns, calls = _load(scenario, condition)
meta = header.get("metadata", {}) if header else {}
parts = meta.get("participants", [])
if parts:
    st.markdown(
        f"**Participants:** {', '.join(parts)}  ·  **flow:** {meta.get('flow', '?')}  ·  "
        f"**condition:** {COND_LABEL.get(condition, condition)}"
    )
st.divider()


def _render_turn(turn: dict) -> None:
    body = (turn.get("body") or "").strip()
    if not body or body.upper() == "END":
        return
    sender = turn.get("sender", "agent")
    tcs = calls.get(turn.get("turn_id"), [])
    if tcs:
        with st.container(border=True):
            st.caption(f"🔧 {sender} queried the knowledge graph — {len(tcs)} call(s)")
            for tc in tcs:
                args = ", ".join(f"{k}={v!r}" for k, v in (tc.get("args") or {}).items())
                summary = (tc.get("result_summary") or "")[:90]
                st.markdown(
                    f"&nbsp;&nbsp;`{tc.get('tool')}({args})` → `{summary}` "
                    f"· {tc.get('latency_ms', 0)} ms",
                    unsafe_allow_html=True,
                )
    with st.chat_message(sender, avatar=AGENT_AVATAR):
        st.markdown(f"**{sender}**")
        st.write(body)


if animate:
    for t in turns:
        _render_turn(t)
        time.sleep(0.8)
else:
    for t in turns:
        _render_turn(t)

n_calls = sum(len(v) for v in calls.values())
if condition == "multi_agent_tools":
    st.divider()
    st.caption(
        f"This run made **{n_calls}** knowledge-graph tool calls across the dialogue. "
        "Switch the condition to compare against personas-only and single-LLM runs."
    )
