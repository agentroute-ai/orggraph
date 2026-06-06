"""RQ1 - how much organizational structure can be recovered unsupervised."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.data import PROCESSED, load_extracted_hierarchy, load_rq1_results
from lib.header import render_header

st.set_page_config(page_title="RQ1 - Structure recovery - OrgGraph", layout="wide")
render_header(
    title="RQ1 - Organizational structure recovery",
    subtitle="Unsupervised dominance ranking vs the Enron ground truth.",
)


def _pair_predictions(pairs: pd.DataFrame, score: dict[str, float]) -> pd.DataFrame:
    """Score each dominance pair as correct / wrong-order / missing using the
    composite ranking (superior should outrank subordinate)."""
    rows = []
    for _, r in pairs.iterrows():
        sup, sub = r["superior"], r["subordinate"]
        ss, bs = score.get(sup), score.get(sub)
        if ss is None or bs is None:
            status = "missing"
        elif ss > bs:
            status = "correct"
        else:
            status = "wrong_order"
        rows.append({
            "superior": sup, "subordinate": sub,
            "superior_level": r.get("superior_level"),
            "subordinate_level": r.get("subordinate_level"),
            "superior_score": ss, "subordinate_score": bs,
            "status": status,
        })
    return pd.DataFrame(rows)


hier = load_extracted_hierarchy()
if hier.empty:
    st.warning("Need `extracted_hierarchy.csv`.")
    st.stop()
score = dict(zip(hier["node"], hier["composite_score"]))

st.info(
    "Evaluation is **pair-classification accuracy** on dominance pairs: for each "
    "manager/subordinate pair, does the unsupervised composite score rank the "
    "superior above the subordinate? Agarwal et al. (2012) report ~83.9% accuracy "
    "on **their own** pair set, which is a reference point, not a head-to-head "
    "baseline (different pairs and employee subset)."
)

# --- pair set selector ------------------------------------------------------
sets = {
    "Senior-executive subset (30 people, 349 pairs)": "dominance_pairs.csv",
    "Ground truth v2 (103 people, 3,814 pairs)": "dominance_pairs_v2.csv",
}
choice = st.radio("Ground-truth pair set", list(sets), index=1, horizontal=True)
pairs_path = PROCESSED / sets[choice]
if not pairs_path.is_file():
    st.warning(f"Missing `{sets[choice]}`.")
    st.stop()
pairs = pd.read_csv(pairs_path)
preds = _pair_predictions(pairs, score)

counts = preds["status"].value_counts().to_dict()
correct = counts.get("correct", 0)
wrong = counts.get("wrong_order", 0)
missing = counts.get("missing", 0)
covered = correct + wrong
total = len(preds)

c = st.columns(4)
c[0].metric("Accuracy (covered pairs)", f"{(correct / covered * 100) if covered else 0:.1f}%",
            help="correct / (correct + wrong-order)")
c[1].metric("Coverage", f"{(covered / total * 100) if total else 0:.1f}%",
            help="pairs where both people were extracted")
c[2].metric("Accuracy (all pairs)", f"{(correct / total * 100) if total else 0:.1f}%",
            help="missing pairs counted as misses")
res = load_rq1_results()
rho = (res.get("spearman_rho") if isinstance(res, dict) else None)
c[3].metric("Spearman ρ", f"{rho:.2f}" if rho is not None else "n/a",
            help="rank correlation vs ground-truth level (precomputed, legacy subset)")

# --- outcome breakdown ------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Pair outcomes")
    bar = pd.DataFrame({
        "outcome": ["correct", "wrong-order", "missing"],
        "pairs": [correct, wrong, missing],
    })
    fig = px.bar(bar, x="outcome", y="pairs", color="outcome", text="pairs",
                 color_discrete_map={"correct": "#16a34a", "wrong-order": "#dc2626",
                                     "missing": "#9ca3af"})
    fig.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Accuracy by superior level")
    cov = preds[preds["status"] != "missing"].copy()
    if not cov.empty and cov["superior_level"].notna().any():
        cov["is_correct"] = (cov["status"] == "correct").astype(int)
        by_lvl = (cov.groupby("superior_level")
                  .agg(accuracy=("is_correct", "mean"), pairs=("is_correct", "size"))
                  .reset_index())
        by_lvl["accuracy"] = (by_lvl["accuracy"] * 100).round(1)
        fig2 = px.bar(by_lvl, x="superior_level", y="accuracy", text="accuracy",
                      hover_data=["pairs"])
        fig2.update_layout(height=340, margin=dict(t=10, b=10),
                           yaxis_title="accuracy (%)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("No per-level data for this pair set.")

# --- per-pair table ---------------------------------------------------------
st.subheader("Dominance-pair predictions")
flt = st.multiselect("Outcome filter", ["correct", "wrong_order", "missing"],
                     default=["wrong_order"])
view = preds[preds["status"].isin(flt)] if flt else preds
st.caption(f"{len(view):,} of {total:,} pairs")
st.dataframe(
    view[["superior", "subordinate", "superior_level", "subordinate_level",
          "superior_score", "subordinate_score", "status"]],
    use_container_width=True, hide_index=True, height=360,
)
