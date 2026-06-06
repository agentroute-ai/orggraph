"""All projects discovered by the cluster pipeline — searchable list + top chart."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.data import PROCESSED
from lib.header import render_header

st.set_page_config(page_title="Projects - OrgGraph", layout="wide")
render_header(
    title="Projects",
    subtitle=(
        "Project-level clusters discovered by UMAP + HDBSCAN over email "
        "embeddings, then named by an LLM."
    ),
)


@st.cache_data(show_spinner=False)
def _load_projects() -> pd.DataFrame:
    path = PROCESSED / "projects.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


df = _load_projects()
if df.empty:
    st.warning("`projects.csv` not found. Run the cluster discovery stage.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Projects discovered", f"{len(df):,}")
c2.metric("Emails clustered", f"{int(df['n_emails'].sum()):,}")
c3.metric("Median emails per project", f"{int(df['n_emails'].median()):,}")

st.divider()

st.subheader("Top 15 projects by email volume")
top = df.sort_values("n_emails", ascending=False).head(15)
fig = px.bar(
    top.iloc[::-1],
    x="n_emails",
    y="name",
    orientation="h",
    title=None,
    hover_data={"project_id": True, "description": True},
)
fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("All projects")
query = st.text_input(
    "Search by name or description",
    placeholder="Type to filter…",
)
view = df
if query:
    q = query.lower()
    view = df[
        df["name"].str.lower().str.contains(q, na=False)
        | df["description"].str.lower().str.contains(q, na=False)
    ]

st.caption(f"{len(view):,} of {len(df):,} projects")
st.dataframe(
    view[["project_id", "name", "n_emails", "description"]].sort_values(
        "n_emails", ascending=False
    ),
    use_container_width=True,
    hide_index=True,
    column_config={
        "project_id": st.column_config.TextColumn("ID", width="small"),
        "name": "Project",
        "n_emails": st.column_config.NumberColumn("Emails", format="%d"),
        "description": "Description",
    },
    height=560,
)
