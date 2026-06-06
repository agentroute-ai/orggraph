"""External organizations - the outside companies Enron communicated with."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.data import DATA_DIR
from lib.header import render_header

st.set_page_config(page_title="External organizations - OrgGraph", layout="wide")
render_header(
    title="External organizations",
    subtitle="Outside companies Enron exchanged email with, by domain and category.",
)


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    path = DATA_DIR / "clients_suppliers.json"
    if not path.is_file():
        return pd.DataFrame()
    raw = json.loads(path.read_text())
    orgs = raw.get("organizations", raw) if isinstance(raw, dict) else raw
    return pd.DataFrame(orgs)


df = _load()
if df.empty:
    st.warning("Missing or empty `clients_suppliers.json`.")
    st.stop()

st.caption(
    "Derived from sender/recipient domains in the corpus: each external domain is "
    "labeled with a category and relationship type, with email volume to and from Enron."
)

c = st.columns(4)
c[0].metric("External organizations", f"{len(df):,}")
c[1].metric("Categories", f"{df['category'].nunique():,}" if "category" in df else "n/a")
c[2].metric("Total external emails", f"{int(df['total_emails'].sum()):,}" if "total_emails" in df else "n/a")
if "relationship_type" in df:
    c[3].metric("Relationship types", f"{df['relationship_type'].nunique():,}")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Organizations by category")
    if "category" in df:
        by_cat = df["category"].value_counts().reset_index()
        by_cat.columns = ["category", "orgs"]
        fig = px.bar(by_cat.sort_values("orgs"), x="orgs", y="category", orientation="h", text="orgs")
        fig.update_layout(height=360, yaxis_title="", xaxis_title="organizations",
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Top organizations by email volume")
    if "total_emails" in df and "company_name" in df:
        top = df.nlargest(15, "total_emails")[["company_name", "total_emails"]]
        fig2 = px.bar(top.sort_values("total_emails"), x="total_emails", y="company_name",
                      orientation="h", text="total_emails")
        fig2.update_layout(height=360, yaxis_title="", xaxis_title="total emails",
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("All external organizations")
cats = sorted(df["category"].dropna().unique()) if "category" in df else []
sel = st.multiselect("Filter by category", cats, default=cats)
view = df[df["category"].isin(sel)] if (sel and "category" in df) else df
show_cols = [c for c in ["company_name", "domain", "category", "relationship_type",
                         "emails_from_enron", "emails_to_enron", "total_emails", "direction_ratio"]
             if c in view.columns]
sort_col = "total_emails" if "total_emails" in view.columns else show_cols[0]
st.caption(f"{len(view):,} of {len(df):,} organizations")
st.dataframe(
    view[show_cols].sort_values(sort_col, ascending=False),
    use_container_width=True, hide_index=True, height=420,
)
