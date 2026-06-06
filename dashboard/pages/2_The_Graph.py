"""The Graph: supporting visualizations - network, topics, hierarchy."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from lib.data import (
    PROCESSED,
    load_communication_graph,
    load_extracted_hierarchy,
    load_projects,
    load_topics,
)
from lib.header import render_header
from lib.network_viz import build_top_n_network_figure

st.set_page_config(page_title="The Graph - OrgGraph", layout="wide")
render_header(
    title="The Graph",
    subtitle="Communication network, topic clusters, and org hierarchy.",
)


@st.cache_resource(show_spinner=False)
def _load_kg_graph(path: str):
    return nx.read_graphml(path)


tab_net, tab_topics, tab_hier, tab_kg = st.tabs(
    ["Network", "Topics & Projects", "Hierarchy", "Knowledge graph"]
)


with tab_net:
    g = load_communication_graph()
    if g is None:
        st.warning("Need `communication_graph.gpickle`. Run the pipeline first.")
    else:
        deg = pd.DataFrame(
            [(n, g.degree(n)) for n in g.nodes()],
            columns=["employee", "degree"],
        ).sort_values("degree", ascending=False).head(25)

        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("Top 25 by communication volume")
            fig = px.bar(
                deg.iloc[::-1],
                x="degree", y="employee",
                orientation="h",
                title=None,
            )
            fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Top 50 subgraph (interactive)")
            net_fig = build_top_n_network_figure(n=50, label_top=15)
            if net_fig is None:
                st.info("Need `communication_graph.gpickle` for the subgraph.")
            else:
                st.plotly_chart(net_fig, use_container_width=True)

with tab_topics:
    umap_path = PROCESSED / "umap_embeddings.npy"
    labels_path = PROCESSED / "cluster_labels.parquet"

    if not umap_path.is_file() or not labels_path.is_file():
        st.warning(
            "Need `umap_embeddings.npy` + `cluster_labels.parquet`. "
            "Run the cluster discovery stage."
        )
    else:
        view_mode = st.radio(
            "Cluster granularity",
            options=["Projects", "Topics"],
            horizontal=True,
            help=(
                "HDBSCAN runs at two granularities. Projects are the larger, "
                "coarser clusters (~450). Topics are finer (~2,800)."
            ),
        )

        coords = np.load(umap_path)
        labels = pd.read_parquet(labels_path)

        if view_mode == "Topics":
            label_col, prefix = "topic_label", "T"
            cluster_df = load_topics()
            id_col = "topic_id"
        else:
            label_col, prefix = "project_label", "P"
            cluster_df = load_projects()
            id_col = "project_id"

        raw_labels = (
            labels[label_col].values
            if label_col in labels.columns
            else np.full(len(coords), -1)
        )

        # Drop HDBSCAN noise (label = -1) from the scatter, otherwise it
        # dominates the plot — 39% of points are topic-noise, 49% project-noise.
        mask = raw_labels >= 0
        n_noise = int((~mask).sum())

        cluster_ids = [
            f"{prefix}{int(lab):03d}" for lab in raw_labels[mask]
        ]

        scatter_df = pd.DataFrame({
            "x": coords[mask, 0],
            "y": coords[mask, 1],
            "cluster_id": cluster_ids,
        })

        if (
            not cluster_df.empty
            and id_col in cluster_df.columns
            and "name" in cluster_df.columns
        ):
            name_map = dict(zip(cluster_df[id_col], cluster_df["name"]))
            scatter_df["cluster_name"] = (
                scatter_df["cluster_id"].map(name_map).fillna(scatter_df["cluster_id"])
            )
        else:
            scatter_df["cluster_name"] = scatter_df["cluster_id"]

        n_clusters = scatter_df["cluster_name"].nunique()
        st.caption(
            f"Showing {len(scatter_df):,} clustered emails across {n_clusters} "
            f"{view_mode.lower()}. {n_noise:,} unclustered (noise) points hidden."
        )

        fig = px.scatter(
            scatter_df,
            x="x", y="y",
            color="cluster_name",
            opacity=0.55,
            height=600,
            title=None,
            labels={"x": "UMAP-1", "y": "UMAP-2"},
        )
        fig.update_traces(marker=dict(size=4))
        if n_clusters > 25:
            fig.update_layout(showlegend=False)
            st.caption(
                f"Legend hidden ({n_clusters} clusters); hover any point for the name."
            )
        st.plotly_chart(fig, use_container_width=True)

        if not cluster_df.empty and "n_emails" in cluster_df.columns:
            st.subheader(f"Top 10 {view_mode.lower()} by email volume")
            top10 = cluster_df.sort_values("n_emails", ascending=False).head(10)
            fig2 = px.bar(
                top10.iloc[::-1],
                x="n_emails", y="name",
                orientation="h",
                title=None,
            )
            fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

with tab_hier:
    hier = load_extracted_hierarchy()
    if hier.empty:
        st.warning("Need `extracted_hierarchy.csv`. Run the score stage.")
    else:
        if "tier" in hier.columns:
            tier_counts = (
                hier["tier"].value_counts().sort_index().reset_index()
            )
            tier_counts.columns = ["tier", "count"]
            fig = px.bar(
                tier_counts,
                x="tier", y="count",
                text="count",
                title="People per inferred tier",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        score_col = (
            "composite_score_v3" if "composite_score_v3" in hier.columns
            else "composite_score" if "composite_score" in hier.columns
            else None
        )
        if score_col is None:
            st.info("No composite score column found in `extracted_hierarchy.csv`.")
        else:
            top10 = hier.sort_values(score_col, ascending=False).head(10)
            st.subheader("Top 10 most senior (by composite score)")
            for _, row in top10.iterrows():
                st.markdown(
                    f"- **{row.get('node', '?')}** : "
                    f"tier {int(row.get('tier', 0))}, "
                    f"score {row.get(score_col, 0):.3f}"
                )

with tab_kg:
    st.caption(
        "Interactive preview of the organizational knowledge graph - a static export of "
        "the Neo4j graph, no live database. Email and topic nodes are left out to keep it "
        "readable. Drag nodes to explore; hover for details."
    )
    kg_path = Path(__file__).resolve().parents[2] / "data" / "kg_backbone.graphml"
    if not kg_path.is_file():
        st.warning("Need `data/kg_backbone.graphml`.")
    else:
        G = _load_kg_graph(str(kg_path))
        NTYPES = ["Person", "Team", "ExternalEntity", "Function"]
        ETYPES = ["REPORTS_TO", "MEMBER_OF", "COMMUNICATES_WITH"]
        COLOR = {"Person": "#16a34a", "Team": "#9333ea",
                 "ExternalEntity": "#2b6cb0", "Function": "#f59e0b"}

        cc = st.columns([2, 2, 1])
        sel_n = cc[0].multiselect("Node types", NTYPES, default=["Person", "Team", "Function"])
        sel_e = cc[1].multiselect("Edge types", ETYPES, default=["REPORTS_TO", "MEMBER_OF"])
        cap = cc[2].slider("Max nodes", 40, G.number_of_nodes(),
                           min(160, G.number_of_nodes()), step=20)

        H = nx.DiGraph()
        for n, d in G.nodes(data=True):
            if d.get("ntype") in sel_n:
                H.add_node(n, **d)
        for u, v, d in G.edges(data=True):
            if d.get("etype") in sel_e and u in H and v in H:
                H.add_edge(u, v, **d)
        H.remove_nodes_from([n for n in list(H.nodes()) if H.degree(n) == 0])
        if H.number_of_nodes() > cap:
            keep = sorted(H.nodes(), key=lambda x: H.degree(x), reverse=True)[:cap]
            H = H.subgraph(keep).copy()

        if H.number_of_nodes() == 0:
            st.info("No nodes match the current filters.")
        else:
            legend = "  ".join(
                f"<span style='color:{COLOR[t]};font-size:18px'>●</span> {t}"
                for t in sel_n if t in COLOR
            )
            st.markdown(
                f"**{H.number_of_nodes()}** nodes · **{H.number_of_edges()}** "
                f"relationships &nbsp;&nbsp; {legend}", unsafe_allow_html=True,
            )
            try:
                import streamlit.components.v1 as components
                from pyvis.network import Network

                net = Network(height="600px", width="100%", directed=True,
                              bgcolor="#ffffff", font_color="#0b1220",
                              cdn_resources="in_line")
                net.barnes_hut(spring_length=120)
                for n, d in H.nodes(data=True):
                    nt = d.get("ntype", "?")
                    tip = f"{d.get('name', '?')} ({nt})"
                    if d.get("title"):
                        tip += f" - {d['title']}"
                    net.add_node(n, label=d.get("name", "?"), title=tip,
                                 color=COLOR.get(nt, "#9ca3af"),
                                 size=10 + 1.4 * H.degree(n), shape="dot")
                for u, v, d in H.edges(data=True):
                    net.add_edge(u, v, title=d.get("etype", ""), color="#cbd5e1")
                components.html(net.generate_html(notebook=False), height=620, scrolling=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not render the graph: {exc}")
