"""RQ3 - does graph-structured retrieval improve agent coherence vs vector-only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.header import render_header

st.set_page_config(page_title="RQ3 - Retrieval - OrgGraph", layout="wide")
render_header(
    title="RQ3 - GraphRAG vs vector retrieval",
    subtitle="Does graph-structured retrieval beat vector-only? (exploratory)",
)

RQ3 = Path(__file__).resolve().parents[2] / "data" / "rq3"

st.info(
    "This is an **exploratory, unpowered** comparison, reported as found. In the "
    "task-completion run, graph retrieval (R2) matched vector (R1) and fell back to "
    "vector on almost every query; in the retrieval-quality pivot, vector outranked "
    "graph. Read it as evidence about *when* graph structure helps, not a win for "
    "either side."
)

# --- task completion: R1 (vector) vs R2 (GraphRAG) --------------------------
summ_path = RQ3 / "task_summary.csv"
if summ_path.is_file():
    summ = pd.read_csv(summ_path)
    label = {"R1": "Vector (R1)", "R2": "GraphRAG (R2)"}
    summ["system"] = summ["condition"].map(label).fillna(summ["condition"])
    keep = ["overall", "factual", "relational", "temporal", "relational_temporal"]
    view = summ[summ["subset"].isin(keep) & (summ["n"] > 0)].copy()

    st.subheader("Task completion (enron_qa, n=149)")
    c = st.columns(3)
    ov = view[view["subset"] == "overall"]
    if not ov.empty:
        r1 = ov[ov["condition"] == "R1"].iloc[0]
        r2 = ov[ov["condition"] == "R2"].iloc[0]
        c[0].metric("Vector F1", f"{r1['mean_f1']:.3f}")
        c[1].metric("GraphRAG F1", f"{r2['mean_f1']:.3f}", f"{r2['mean_f1'] - r1['mean_f1']:+.3f}")
        c[2].metric("GraphRAG fallback rate", f"{r2['fallback_rate'] * 100:.0f}%",
                    help="share of queries where graph retrieval fell back to vector")

    left, right = st.columns(2)
    with left:
        fig = px.bar(view, x="subset", y="mean_f1", color="system", barmode="group",
                     text_auto=".3f", range_y=[0, 0.6],
                     color_discrete_map={"Vector (R1)": "#2b6cb0", "GraphRAG (R2)": "#16a34a"})
        fig.update_layout(height=340, xaxis_title="", yaxis_title="mean F1", legend_title="",
                          margin=dict(t=10, b=10), xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig2 = px.bar(view, x="subset", y="mean_recall_at_10", color="system", barmode="group",
                      text_auto=".3f", range_y=[0, 0.6],
                      color_discrete_map={"Vector (R1)": "#2b6cb0", "GraphRAG (R2)": "#16a34a"})
        fig2.update_layout(height=340, xaxis_title="", yaxis_title="recall@10", legend_title="",
                           margin=dict(t=10, b=10), xaxis_tickangle=-20)
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.caption("No `data/rq3/task_summary.csv`.")

# --- retrieval-quality pivot: V / G / H -------------------------------------
pivot_path = RQ3 / "retrieval_pivot.json"
if pivot_path.is_file():
    pv = json.loads(pivot_path.read_text())
    st.subheader(f"Retrieval quality pivot (n={pv.get('n_queries', '?')} queries)")
    name = {"V": "Vector", "G": "Graph", "H": "Hybrid"}
    rows = []
    for sys_key, metrics in (pv.get("metrics") or {}).items():
        for metric, val in metrics.items():
            rows.append({"system": name.get(sys_key, sys_key), "metric": metric, "value": round(val, 3)})
    mdf = pd.DataFrame(rows)
    fig3 = px.bar(mdf, x="metric", y="value", color="system", barmode="group", text_auto=".3f",
                  color_discrete_map={"Vector": "#2b6cb0", "Graph": "#16a34a", "Hybrid": "#9333ea"})
    fig3.update_layout(height=360, xaxis_title="", yaxis_title="", legend_title="",
                       margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

# --- per-query explorer -----------------------------------------------------
rows_path = RQ3 / "task_rows.jsonl"
if rows_path.is_file():
    st.divider()
    st.subheader("Per-query explorer")

    @st.cache_data(show_spinner=False)
    def _rows() -> pd.DataFrame:
        recs = [json.loads(ln) for ln in rows_path.read_text().splitlines() if ln.strip()]
        return pd.DataFrame(recs)

    rows = _rows()
    label = {"R1": "Vector (R1)", "R2": "GraphRAG (R2)"}
    rows["system"] = rows["condition"].map(label).fillna(rows["condition"])
    cols = st.columns(2)
    qtypes = sorted(rows["qtype"].dropna().unique())
    qsel = cols[0].multiselect("Question type", qtypes, default=qtypes)
    csel = cols[1].multiselect("System", sorted(rows["system"].unique()),
                               default=sorted(rows["system"].unique()))
    view = rows[rows["qtype"].isin(qsel) & rows["system"].isin(csel)]
    st.dataframe(
        view[["qid", "system", "qtype", "f1", "recall_at_10", "used_fallback", "question"]]
        .sort_values("f1", ascending=False),
        use_container_width=True, hide_index=True, height=320,
    )
    if not view.empty:
        qid = st.selectbox("Inspect query", sorted(view["qid"].unique()))
        row = view[view["qid"] == qid].iloc[0]
        st.markdown(f"**Q:** {row['question']}")
        st.markdown(f"**Gold:** {row['gold']}")
        st.markdown(f"**Answer ({row['system']}, F1={row['f1']:.2f}):** {row['answer']}")
