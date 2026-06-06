"""The Graph: supporting visualizations - network, topics, hierarchy."""
from __future__ import annotations

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


tab_net, tab_topics, tab_hier = st.tabs(["Network", "Topics & Projects", "Hierarchy"])


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
