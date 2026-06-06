"""RQ2 - do multi-agent systems produce more natural dialogue than a single LLM."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.header import render_header

st.set_page_config(page_title="RQ2 - Dialogue naturalness - OrgGraph", layout="wide")
render_header(
    title="RQ2 - Dialogue naturalness",
    subtitle="Multi-agent vs single-LLM dialogue, scored by an LLM judge (pilot).",
)

RQ2 = Path(__file__).resolve().parents[2] / "data" / "rq2"
VERDICTS = RQ2 / "judge"
TRANSCRIPTS = RQ2 / "transcripts"

COND_LABEL = {
    "multi_agent": "Multi-agent (personas)",
    "multi_agent_tools": "Multi-agent + KG tools",
    "single_llm": "Single LLM (long context)",
}


@st.cache_data(show_spinner=False)
def _load_verdicts() -> pd.DataFrame:
    rows = []
    if not VERDICTS.is_dir():
        return pd.DataFrame()
    for f in sorted(VERDICTS.glob("*.json")):
        d = json.loads(f.read_text())
        for dim, v in (d.get("scores") or {}).items():
            rows.append({
                "scenario": d.get("scenario_name"),
                "condition": d.get("condition"),
                "dimension": dim,
                "score": v.get("score"),
                "justification": v.get("justification", ""),
            })
    return pd.DataFrame(rows)


df = _load_verdicts()
if df.empty:
    st.warning("No RQ2 verdicts shipped under `data/rq2/judge/`.")
    st.stop()

df["condition_label"] = df["condition"].map(COND_LABEL).fillna(df["condition"])
n_scen = df["scenario"].nunique()
st.caption(
    f"Pilot: {n_scen} scenarios x {df['condition'].nunique()} conditions, scored 1-5 "
    f"by an LLM judge on 5 naturalness dimensions. Small sample - directional, not powered."
)

# --- mean score per condition ----------------------------------------------
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("Overall")
    overall = (df.groupby("condition_label")["score"].mean()
               .reset_index().sort_values("score", ascending=False))
    overall["score"] = overall["score"].round(2)
    fig = px.bar(overall, x="score", y="condition_label", orientation="h",
                 text="score", range_x=[0, 5], color="condition_label")
    fig.update_layout(showlegend=False, height=320, yaxis_title="", xaxis_title="mean score",
                      margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("By dimension")
    piv = (df.groupby(["dimension", "condition_label"])["score"].mean()
           .reset_index())
    piv["score"] = piv["score"].round(2)
    fig2 = px.bar(piv, x="dimension", y="score", color="condition_label",
                  barmode="group", text="score", range_y=[0, 5])
    fig2.update_layout(height=320, xaxis_title="", yaxis_title="mean score",
                       legend_title="", margin=dict(t=10, b=10),
                       xaxis_tickangle=-20)
    st.plotly_chart(fig2, use_container_width=True)

# --- per-scenario heatmap ---------------------------------------------------
st.subheader("Per-scenario mean (across dimensions)")
heat = (df.groupby(["scenario", "condition_label"])["score"].mean()
        .reset_index().pivot(index="scenario", columns="condition_label", values="score")
        .round(2))
fig3 = px.imshow(heat, text_auto=True, color_continuous_scale="Greens",
                 zmin=1, zmax=5, aspect="auto")
fig3.update_layout(height=320, xaxis_title="", yaxis_title="", margin=dict(t=10, b=10))
st.plotly_chart(fig3, use_container_width=True)

# --- transcript reader ------------------------------------------------------
st.divider()
st.subheader("Read a dialogue")
scenarios = sorted(df["scenario"].unique())
colA, colB = st.columns(2)
scen = colA.selectbox("Scenario", scenarios)
cond = colB.selectbox("Condition", list(COND_LABEL), format_func=lambda c: COND_LABEL[c])

tpath = TRANSCRIPTS / f"{scen}__{cond}.jsonl"
if not tpath.is_file():
    st.caption("No transcript for this combination.")
else:
    lines = [json.loads(ln) for ln in tpath.read_text().splitlines() if ln.strip()]
    turns = [ln for ln in lines if not ln.get("_header")]
    left, right = st.columns([3, 2])
    with left:
        for t in turns:
            body = (t.get("body") or "").strip()
            if not body or body.upper() == "END":
                continue
            with st.chat_message(t.get("sender", "agent")):
                st.markdown(f"**{t.get('sender', '?')}**")
                st.write(body)
    with right:
        vfile = VERDICTS / f"{scen}__{cond}__verdict.json"
        if vfile.is_file():
            v = json.loads(vfile.read_text())
            st.metric("Judge mean score", f"{v.get('mean_score', float('nan')):.2f} / 5")
            st.caption(v.get("overall_summary", ""))
            for dim, sc in (v.get("scores") or {}).items():
                with st.expander(f"{dim.replace('_', ' ')} — {sc.get('score')}/5"):
                    st.write(sc.get("justification", ""))
