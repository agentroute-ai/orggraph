"""All topics discovered by the cluster pipeline — searchable list + top chart."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.data import load_topics
from lib.header import render_header

st.set_page_config(page_title="Topics - OrgGraph", layout="wide")
render_header(
    title="Topics",
    subtitle=(
        "Fine-grained topic clusters discovered by UMAP + HDBSCAN, named by "
        "an LLM. Each topic groups emails about the same narrow subject."
    ),
)


df = load_topics()
if df.empty:
    st.warning("`topics.csv` not found. Run the cluster discovery stage.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Topics discovered", f"{len(df):,}")
if "n_emails" in df.columns:
    c2.metric("Emails clustered", f"{int(df['n_emails'].sum()):,}")
    c3.metric("Median emails per topic", f"{int(df['n_emails'].median()):,}")

st.divider()

st.subheader("Top 15 topics by email volume")
if "n_emails" in df.columns and "name" in df.columns:
    top = df.sort_values("n_emails", ascending=False).head(15)
    hover = {"topic_id": True}
    if "description" in df.columns:
        hover["description"] = True
    fig = px.bar(
        top.iloc[::-1],
        x="n_emails",
        y="name",
        orientation="h",
        title=None,
        hover_data=hover,
    )
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("All topics")
query = st.text_input(
    "Search by name or description",
    placeholder="Type to filter…",
)
view = df
if query:
    q = query.lower()
    cond = df["name"].str.lower().str.contains(q, na=False)
    if "description" in df.columns:
        cond = cond | df["description"].str.lower().str.contains(q, na=False)
    view = df[cond]

st.caption(f"{len(view):,} of {len(df):,} topics")
display_cols = [c for c in ["topic_id", "name", "n_emails", "description"] if c in view.columns]
st.dataframe(
    view[display_cols].sort_values(
        "n_emails" if "n_emails" in display_cols else display_cols[0],
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
    column_config={
        "topic_id": st.column_config.TextColumn("ID", width="small"),
        "name": "Topic",
        "n_emails": st.column_config.NumberColumn("Emails", format="%d"),
        "description": "Description",
    },
    height=560,
)
