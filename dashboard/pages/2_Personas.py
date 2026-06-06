"""Per-employee persona profile + the system prompt that drives the agent voice."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.data import (
    PROCESSED,
    load_communication_graph,
    load_extracted_hierarchy,
    load_ground_truth,
    load_persona_prompt,
    load_topics,
    persona_for,
)
from lib.header import render_header


# ---------------------------------------------------------------------------
# Per-person data slices (cached so tab-switching is instant)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _emails_for_person(name: str) -> pd.DataFrame:
    """Return all emails where `name` appears as sender or recipient.

    Needs the full ``clean_emails.parquet`` (Stage 0 output), which is not
    shipped with the repo. When it is absent (e.g. the public/static
    deployment), return an empty frame so the per-person email / project /
    topic drill-downs degrade gracefully. Regenerate the parquet with
    ``make corpus`` to enable them.
    """
    cols = ["date", "direction", "from", "recipients", "subject", "email_id"]
    parquet = PROCESSED / "clean_emails.parquet"
    if not parquet.is_file():
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(
        parquet,
        columns=[
            "email_id", "date", "subject",
            "sender_resolved", "recipients_resolved",
        ],
    )
    is_sender = df["sender_resolved"] == name
    is_recipient = df["recipients_resolved"].apply(
        lambda r: name in r if r is not None and len(r) > 0 else False
    )
    mask = is_sender | is_recipient
    sub = df.loc[mask].copy()
    sub["direction"] = is_sender[mask].map({True: "sent", False: "received"})
    sub["recipients"] = sub["recipients_resolved"].apply(
        lambda r: ", ".join(r) if r is not None else ""
    )
    out = sub[[
        "date", "direction", "sender_resolved", "recipients", "subject", "email_id",
    ]].rename(columns={"sender_resolved": "from"})
    return out.sort_values("date", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _projects_for_person(name: str) -> pd.DataFrame:
    """Aggregate projects this person's emails fall into, with counts."""
    emails = _emails_for_person(name)
    if emails.empty:
        return pd.DataFrame()
    cluster = pd.read_parquet(PROCESSED / "cluster_labels.parquet")
    merged = emails.merge(cluster, on="email_id", how="left")
    valid = merged[merged["project_label"] >= 0]
    if valid.empty:
        return pd.DataFrame()
    counts = (
        valid.groupby("project_label").size().reset_index(name="emails")
    )
    counts["project_id"] = counts["project_label"].apply(lambda x: f"P{int(x):03d}")

    projects_path = PROCESSED / "projects.csv"
    if projects_path.is_file():
        projects = pd.read_csv(projects_path)[["project_id", "name", "description"]]
        out = counts.merge(projects, on="project_id", how="left")
    else:
        out = counts.assign(name="(unknown)", description="")
    return out.sort_values("emails", ascending=False)[
        ["project_id", "name", "emails", "description"]
    ].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _topics_for_person(name: str) -> pd.DataFrame:
    """Aggregate topics this person's emails fall into, with counts."""
    emails = _emails_for_person(name)
    if emails.empty:
        return pd.DataFrame()
    cluster = pd.read_parquet(PROCESSED / "cluster_labels.parquet")
    merged = emails.merge(cluster, on="email_id", how="left")
    valid = merged[merged["topic_label"] >= 0]
    if valid.empty:
        return pd.DataFrame()
    counts = valid.groupby("topic_label").size().reset_index(name="emails")
    counts["topic_id"] = counts["topic_label"].apply(lambda x: f"T{int(x):03d}")

    topics = load_topics()
    if not topics.empty and "topic_id" in topics.columns and "name" in topics.columns:
        out = counts.merge(
            topics[["topic_id", "name"]], on="topic_id", how="left",
        )
    else:
        out = counts.assign(name="(unknown)")
    return out.sort_values("emails", ascending=False)[
        ["topic_id", "name", "emails"]
    ].reset_index(drop=True)

st.set_page_config(page_title="Personas - OrgGraph", layout="wide")
render_header(
    title="Personas",
    subtitle="Profile + voice-grounded system prompt for the RQ2 agents.",
)

g = load_communication_graph()
hier = load_extracted_hierarchy()
gt = load_ground_truth()

if g is None or hier.empty:
    st.warning(
        "Need both `communication_graph.gpickle` and `extracted_hierarchy.csv`."
    )
    st.stop()

default_pool = hier.sort_values("composite_score", ascending=False)["node"].tolist()
st.sidebar.subheader("Filter")
restrict = st.sidebar.radio(
    "Show personas for",
    ["Ground-truth labeled", "All extracted nodes"],
    index=0,
)
if restrict == "Ground-truth labeled" and not gt.empty:
    pool = [n for n in default_pool if n in set(gt["name"])]
else:
    pool = default_pool

if not pool:
    st.warning("No employees available with the chosen filter.")
    st.stop()

choice = st.selectbox("Employee", pool, index=0)
profile = persona_for(choice, top_k=10)

top_left, top_right = st.columns([2, 1])
with top_left:
    st.subheader(profile.get("node", "-"))
    sub = []
    if profile.get("gt_title"):
        sub.append(profile["gt_title"])
    if profile.get("gt_level"):
        sub.append(
            f"GT level: **{profile['gt_level']}** "
            f"({profile.get('gt_level_numeric', '?')}/6)"
        )
    if profile.get("email"):
        sub.append(f"`{profile['email']}`")
    st.markdown("  ·  ".join(sub) or "_no ground-truth metadata_")
with top_right:
    if profile.get("composite_score") is not None:
        st.metric("Composite score", f"{profile['composite_score']:.3f}")
    if profile.get("tier") is not None:
        st.metric("Inferred tier", profile["tier"])

st.divider()

m_cols = st.columns(4)
m_cols[0].metric("PageRank", f"{profile.get('pagerank', 0):.4f}")
m_cols[1].metric("Betweenness", f"{profile.get('betweenness', 0):.4f}")
m_cols[2].metric("In-degree (log)", f"{profile.get('in_degree', 0):.2f}")
m_cols[3].metric("Out-degree", f"{int(profile.get('out_degree', 0)):,}")

st.divider()

t_prompt, t_emails, t_projects, t_topics, t_contacts, t_llm = st.tabs(
    ["System prompt", "Emails", "Projects", "Topics", "Top contacts", "LLM seniority"]
)

with t_prompt:
    text = load_persona_prompt(profile.get("node", ""))
    if text:
        st.caption(
            "Generated from this person's actual sent emails. "
            "Used as the system prompt for the RQ2 multi-agent simulations."
        )
        st.code(text, language=None, wrap_lines=True)
    else:
        st.info(
            "No persona prompt has been generated for this employee yet. "
            "Generated prompts live in `datasets/enron/processed/persona_prompts/`."
        )

with t_emails:
    name = profile.get("node", "")
    emails_df = _emails_for_person(name) if name else pd.DataFrame()
    if emails_df.empty:
        st.info("No emails found for this employee.")
    else:
        query = st.text_input(
            "Search subject or sender",
            key=f"email_search_{name}",
            placeholder="Type to filter…",
        )
        view = emails_df
        if query:
            q = query.lower()
            view = emails_df[
                emails_df["subject"].str.lower().str.contains(q, na=False)
                | emails_df["from"].str.lower().str.contains(q, na=False)
                | emails_df["recipients"].str.lower().str.contains(q, na=False)
            ]
        st.caption(f"{len(view):,} of {len(emails_df):,} emails")
        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn(
                    "Date", format="YYYY-MM-DD HH:mm",
                ),
                "direction": st.column_config.TextColumn("Dir", width="small"),
                "from": "From",
                "recipients": "To",
                "subject": "Subject",
                "email_id": st.column_config.TextColumn("ID", width="small"),
            },
            height=500,
        )

with t_projects:
    name = profile.get("node", "")
    projects_df = _projects_for_person(name) if name else pd.DataFrame()
    if projects_df.empty:
        st.info("No projects associated with this employee.")
    else:
        st.caption(f"{len(projects_df):,} projects")
        st.dataframe(
            projects_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "project_id": st.column_config.TextColumn("ID", width="small"),
                "name": "Project",
                "emails": st.column_config.NumberColumn("Emails", format="%d"),
                "description": "Description",
            },
            height=500,
        )

with t_topics:
    name = profile.get("node", "")
    topics_df = _topics_for_person(name) if name else pd.DataFrame()
    if topics_df.empty:
        st.info("No topics associated with this employee.")
    else:
        st.caption(f"{len(topics_df):,} topics")
        st.dataframe(
            topics_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "topic_id": st.column_config.TextColumn("ID", width="small"),
                "name": "Topic",
                "emails": st.column_config.NumberColumn("Emails", format="%d"),
            },
            height=500,
        )

with t_contacts:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top recipients (out-edges)**")
        recip = pd.DataFrame(
            profile.get("top_recipients") or [],
            columns=["recipient", "messages"],
        )
        if recip.empty:
            st.caption("No outgoing edges.")
        else:
            fig = px.bar(
                recip.iloc[::-1],
                x="messages", y="recipient",
                orientation="h",
                title=None,
            )
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Top senders (in-edges)**")
        sndr = pd.DataFrame(
            profile.get("top_senders") or [],
            columns=["sender", "messages"],
        )
        if sndr.empty:
            st.caption("No incoming edges.")
        else:
            fig = px.bar(
                sndr.iloc[::-1],
                x="messages", y="sender",
                orientation="h",
                title=None,
            )
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

with t_llm:
    if profile.get("llm_reasoning"):
        st.metric("LLM seniority (1-10)", profile.get("llm_seniority", "-"))
        st.caption(f"Model: `{profile.get('llm_model', '?')}`")
        st.markdown(profile["llm_reasoning"])
    else:
        st.info("This employee was not in the tier 1 LLM seniority sample.")
