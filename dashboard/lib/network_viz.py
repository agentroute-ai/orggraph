"""Interactive plotly network visualizations for the demo dashboard."""
from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from lib.data import load_communication_graph, load_extracted_hierarchy


@st.cache_data(show_spinner="Laying out network...")
def build_top_n_network_figure(n: int = 50, label_top: int = 15) -> go.Figure | None:
    """Build a plotly figure of the top-N communication subgraph.

    Nodes are sized by degree and colored by inferred tier (1 = senior, 6 = junior).
    Hover shows the employee's name, tier, and connection count. The top `label_top`
    most-connected nodes are labeled directly on the chart.
    """
    g = load_communication_graph()
    if g is None:
        return None

    degrees = dict(g.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:n]
    sub = g.subgraph(top_nodes).copy()

    pos = nx.spring_layout(sub, seed=42, k=0.7, iterations=120)

    hier = load_extracted_hierarchy()
    tier_map = dict(zip(hier["node"], hier["tier"])) if not hier.empty else {}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in sub.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="#cbd5e1"),
        hoverinfo="none",
        showlegend=False,
    )

    nodes = list(sub.nodes())
    nodes_sorted_by_degree = sorted(nodes, key=lambda x: sub.degree(x), reverse=True)
    label_set = set(nodes_sorted_by_degree[:label_top])

    node_x = [pos[node][0] for node in nodes]
    node_y = [pos[node][1] for node in nodes]
    node_sizes = [12 + 0.7 * sub.degree(node) for node in nodes]
    node_colors = [float(tier_map.get(node, 6)) for node in nodes]
    node_labels = [node if node in label_set else "" for node in nodes]
    node_hover = [
        f"<b>{node}</b><br>"
        f"Tier: {tier_map.get(node, '?')}<br>"
        f"Connections: {sub.degree(node)}"
        for node in nodes
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=10, color="#0b1220"),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Viridis",
            reversescale=True,
            cmin=1,
            cmax=6,
            line=dict(width=1.5, color="white"),
            colorbar=dict(
                title=dict(text="Tier", side="right"),
                thickness=12,
                len=0.6,
                tickvals=[1, 2, 3, 4, 5, 6],
            ),
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=620,
        plot_bgcolor="white",
        hovermode="closest",
    )
    return fig
