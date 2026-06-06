"""OrgGraph thesis demo dashboard - home page.

Run from repo root:
    streamlit run dashboard/Home.py
"""
from __future__ import annotations

import streamlit as st

from lib.data import (
    PROCESSED,
    load_employees,
    load_extracted_hierarchy,
    load_topics,
)
from lib.header import render_header
from lib.network_viz import build_top_n_network_figure
from orggraph.config import REPO_ROOT

st.set_page_config(
    page_title="OrgGraph",
    page_icon=str(REPO_ROOT / "assets" / "logo-mark.svg"),
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_header(
    subtitle="Unsupervised organizational knowledge from the Enron email corpus.",
)


@st.cache_data(show_spinner=False)
def _emails_count() -> str:
    """Approximate full corpus size. Hardcoded fallback if file absent."""
    parquet = PROCESSED / "clean_emails.parquet"
    if parquet.is_file():
        try:
            import pyarrow.parquet as pq

            n = pq.ParquetFile(parquet).metadata.num_rows
            return f"{n:,}"
        except Exception:  # noqa: BLE001
            pass
    return "517K"


@st.cache_data(show_spinner=False)
def _employees_count() -> int:
    df = load_employees()
    return len(df)


@st.cache_data(show_spinner=False)
def _personas_count() -> int:
    p = PROCESSED / "persona_prompts"
    if not p.is_dir():
        return 0
    return sum(1 for f in p.iterdir() if f.suffix == ".txt")


@st.cache_data(show_spinner=False)
def _tiers_count() -> int:
    hier = load_extracted_hierarchy()
    if hier.empty or "tier" not in hier.columns:
        return 6
    return int(hier["tier"].nunique())


@st.cache_data(show_spinner=False)
def _topics_count() -> int:
    df = load_topics()
    return len(df) if not df.empty else 0


@st.cache_data(show_spinner=False)
def _kg_tools_count() -> int:
    """Count V1 tools without opening a Neo4j connection (lazy import)."""
    try:
        from orggraph.pipeline.agents.tools import build_default_registry

        reg = build_default_registry(driver=None)
        return len(reg)
    except Exception:  # noqa: BLE001
        return 0


row1 = st.columns(3)
row1[0].metric("Emails analyzed", _emails_count())
row1[1].metric("Employees mapped", f"{_employees_count():,}")
row1[2].metric("AI personas built", f"{_personas_count():,}")

row2 = st.columns(3)
row2[0].metric("Org tiers", _tiers_count())
row2[1].metric("Topics discovered", f"{_topics_count():,}")
row2[2].metric("KG tools available", _kg_tools_count())

st.divider()

st.subheader("Communication network: top 50 by volume")
network_fig = build_top_n_network_figure(n=50, label_top=15)
if network_fig is None:
    st.info("Need `communication_graph.gpickle`. Run the pipeline first.")
else:
    st.plotly_chart(network_fig, use_container_width=True)
    st.caption(
        "Hover any node for details. Drag to pan, scroll to zoom. Node size "
        "scales with total email volume; color shades from senior (dark) "
        "to junior (light) tier."
    )

st.divider()

st.markdown(
    """
**Pages**

- **The Graph**: communication network, topic clusters, hierarchy
- **Personas**: per-employee profile + the system prompt that drives the agent voice
- **KG Chat**: ask questions of the knowledge graph through the V1 tool catalog
"""
)
