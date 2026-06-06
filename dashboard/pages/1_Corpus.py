"""Corpus overview - the Enron email dataset at a glance (precomputed stats)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.header import render_header

st.set_page_config(page_title="Corpus - OrgGraph", layout="wide")
render_header(
    title="The Enron corpus",
    subtitle="The email dataset OrgGraph is built from.",
)

STATS = Path(__file__).resolve().parents[2] / "data" / "corpus_stats.json"
if not STATS.is_file():
    st.warning("Missing `data/corpus_stats.json`.")
    st.stop()
s = json.loads(STATS.read_text())

st.caption(
    "Stats are precomputed from the filtered corpus, so this page stays fast and needs "
    "no live data. The raw corpus comes from the Enron email dataset on HuggingFace."
)

c = st.columns(4)
c[0].metric("Emails (filtered)", f"{s['total_emails']:,}")
c[1].metric("Internal (enron.com)", f"{s['internal_emails']:,}")
c[2].metric("External", f"{s['external_emails']:,}")
c[3].metric("Sender domains", f"{s['unique_sender_domains']:,}")
c2 = st.columns(4)
c2[0].metric("Resolved senders", f"{s['unique_senders_resolved']:,}")
c2[1].metric("Date range", f"{s.get('date_min','?')} → {s.get('date_max','?')}")
c2[2].metric("Empty subjects", f"{s['empty_subjects']:,}")
if s.get("mean_body_chars") is not None:
    c2[3].metric("Mean body length", f"{s['mean_body_chars']:.0f} chars")

st.divider()

# --- volume timeline --------------------------------------------------------
st.subheader("Email volume over time")
mv = s.get("monthly_volume", {})
if mv:
    tdf = pd.DataFrame({"month": pd.to_datetime(list(mv.keys())), "emails": list(mv.values())})
    fig = px.area(tdf, x="month", y="emails")
    fig.update_traces(line_color="#16a34a", fillcolor="rgba(22,163,74,0.18)")
    fig.update_layout(height=320, xaxis_title="", yaxis_title="emails / month",
                      margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Internal vs external")
    split = pd.DataFrame({
        "kind": ["Internal (enron.com)", "External"],
        "emails": [s["internal_emails"], s["external_emails"]],
    })
    fig2 = px.pie(split, names="kind", values="emails", hole=0.5,
                  color="kind",
                  color_discrete_map={"Internal (enron.com)": "#16a34a", "External": "#2b6cb0"})
    fig2.update_layout(height=320, margin=dict(t=10, b=10), legend_title="")
    st.plotly_chart(fig2, use_container_width=True)
with right:
    st.subheader("Top external domains")
    ed = s.get("top_external_domains", {})
    if ed:
        edf = pd.DataFrame({"domain": list(ed.keys()), "emails": list(ed.values())})
        fig3 = px.bar(edf.sort_values("emails"), x="emails", y="domain", orientation="h",
                      text="emails")
        fig3.update_layout(height=320, yaxis_title="", xaxis_title="emails",
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

st.subheader("Most active senders")
ts = s.get("top_senders", {})
if ts:
    sdf = pd.DataFrame({"sender": list(ts.keys()), "emails": list(ts.values())})
    fig4 = px.bar(sdf.sort_values("emails"), x="emails", y="sender", orientation="h", text="emails")
    fig4.update_layout(height=460, yaxis_title="", xaxis_title="emails sent",
                       margin=dict(t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)
