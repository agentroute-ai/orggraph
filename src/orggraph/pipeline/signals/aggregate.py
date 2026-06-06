"""Stage 3 — aggregate_email_signals_to_kg.py

Reads embeddings_email (Stage 1 embedding + Stage 1.5 metadata + Stage 2a signals),
then writes:

  1. pair_signals table in Postgres — dyadic (sender, recipient) rollups.
  2. Person aggregate properties to Neo4j.
  3. Email nodes (decision_carrying = true only) to Neo4j.
  4. Project and Topic nodes from CSVs if present, else skipped.

Usage:
    source .env && python scripts/aggregate_email_signals_to_kg.py
    python scripts/aggregate_email_signals_to_kg.py --limit 500   # dev
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from pgvector.psycopg import register_vector

from orggraph.config import OUTPUT_DIR

PROCESSED_DIR = OUTPUT_DIR

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "orggraph2026")

PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://{os.environ.get('POSTGRES_USER', 'orggraph')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'orggraph2026')}@localhost:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'orggraph')}",
)



# ---------------------------------------------------------------------------
# Public API: compute_pair_signals
# ---------------------------------------------------------------------------

def compute_pair_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dyadic rollups — one row per directed (sender_resolved, recipient_resolved) pair.

    Args:
        df: DataFrame with columns:
            email_id, sender_resolved, recipients_resolved (list), speech_acts (list),
            action_required, commitment_made, decision_carrying, sentiment,
            body_word_count, reply_latency_hours, date.

    Returns:
        DataFrame with columns: sender_id, recipient_id, n_emails, n_to, n_cc,
        n_request_sent, n_commit_sent, n_deliver_sent, n_propose_sent,
        n_decision, n_action_required, mean_sentiment, mean_body_words,
        mean_reply_latency_h, length_asymmetry, request_commit_ratio,
        first_email_date, last_email_date.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "sender_id", "recipient_id", "n_emails", "n_to", "n_cc",
            "n_request_sent", "n_commit_sent", "n_deliver_sent", "n_propose_sent",
            "n_decision", "n_action_required", "mean_sentiment", "mean_body_words",
            "mean_reply_latency_h", "length_asymmetry", "request_commit_ratio",
            "first_email_date", "last_email_date",
        ])

    # Explode recipients — one row per (email, sender, recipient)
    _BAD_NAMES = {"", "nan", "none", "null"}
    rows = []
    for _, row in df.iterrows():
        sender = row.get("sender_resolved") or ""
        recips = row.get("recipients_resolved") or []
        if not sender or not recips:
            continue
        if str(sender).strip().lower() in _BAD_NAMES:
            continue  # filter literal "NaN"/"None" strings from upstream resolution
        speech_acts = row.get("speech_acts") or []
        for recip in recips:
            if not recip or recip == sender:
                continue  # skip empty recipients and self-loops
            if str(recip).strip().lower() in _BAD_NAMES:
                continue  # filter literal "NaN"/"None" strings
            rows.append({
                "sender_id": sender,
                "recipient_id": recip,
                "is_request": "request" in speech_acts,
                "is_commit": "commit" in speech_acts,
                "is_deliver": "deliver" in speech_acts,
                "is_propose": "propose" in speech_acts,
                "decision_carrying": bool(row.get("decision_carrying", False)),
                "action_required": bool(row.get("action_required", False)),
                "sentiment": row.get("sentiment") if row.get("sentiment") is not None else float("nan"),
                "body_word_count": row.get("body_word_count") if row.get("body_word_count") is not None else float("nan"),
                "reply_latency_hours": row.get("reply_latency_hours"),
                "date": row.get("date"),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "sender_id", "recipient_id", "n_emails", "n_to", "n_cc",
            "n_request_sent", "n_commit_sent", "n_deliver_sent", "n_propose_sent",
            "n_decision", "n_action_required", "mean_sentiment", "mean_body_words",
            "mean_reply_latency_h", "length_asymmetry", "request_commit_ratio",
            "first_email_date", "last_email_date",
        ])

    exploded = pd.DataFrame(rows)

    # Aggregate per (sender_id, recipient_id)
    grp = exploded.groupby(["sender_id", "recipient_id"], sort=False)

    agg = grp.agg(
        n_emails=("is_request", "count"),
        n_request_sent=("is_request", "sum"),
        n_commit_sent=("is_commit", "sum"),
        n_deliver_sent=("is_deliver", "sum"),
        n_propose_sent=("is_propose", "sum"),
        n_decision=("decision_carrying", "sum"),
        n_action_required=("action_required", "sum"),
        mean_sentiment=("sentiment", "mean"),
        mean_body_words=("body_word_count", "mean"),
        mean_reply_latency_h=("reply_latency_hours", "mean"),
        first_email_date=("date", "min"),
        last_email_date=("date", "max"),
    ).reset_index()

    # n_to = n_emails (all recipients treated as TO; n_cc always 0)
    agg["n_to"] = agg["n_emails"]
    agg["n_cc"] = 0

    # Cast int columns
    for col in ["n_emails", "n_to", "n_cc", "n_request_sent", "n_commit_sent",
                "n_deliver_sent", "n_propose_sent", "n_decision", "n_action_required"]:
        agg[col] = agg[col].astype(int)

    # Build reverse-pair lookup for length_asymmetry and request_commit_ratio
    pair_mean_words = agg.set_index(["sender_id", "recipient_id"])["mean_body_words"]
    pair_n_commit = agg.set_index(["sender_id", "recipient_id"])["n_commit_sent"]

    def _length_asym(row):
        rev_key = (row["recipient_id"], row["sender_id"])
        if rev_key in pair_mean_words.index:
            return row["mean_body_words"] - pair_mean_words[rev_key]
        return float("nan")

    def _req_commit_ratio(row):
        rev_key = (row["recipient_id"], row["sender_id"])
        if rev_key not in pair_n_commit.index:
            return float("nan")
        rev_commits = pair_n_commit[rev_key]
        if rev_commits == 0:
            return float("nan")
        return float(row["n_request_sent"]) / float(rev_commits)

    agg["length_asymmetry"] = agg.apply(_length_asym, axis=1)
    agg["request_commit_ratio"] = agg.apply(_req_commit_ratio, axis=1)

    # Final column order
    cols = [
        "sender_id", "recipient_id", "n_emails", "n_to", "n_cc",
        "n_request_sent", "n_commit_sent", "n_deliver_sent", "n_propose_sent",
        "n_decision", "n_action_required", "mean_sentiment", "mean_body_words",
        "mean_reply_latency_h", "length_asymmetry", "request_commit_ratio",
        "first_email_date", "last_email_date",
    ]
    return agg[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API: compute_person_aggregates
# ---------------------------------------------------------------------------

def compute_person_aggregates(df: pd.DataFrame) -> dict[str, dict]:
    """Compute per-Person rollups keyed by sender_resolved name.

    Args:
        df: DataFrame with per-email rows. Skips rows where sender_resolved is empty.

    Returns:
        Dict mapping sender_resolved -> dict of aggregate properties.
    """
    if df.empty:
        return {}

    result: dict[str, dict] = {}

    # Filter out empty senders AND literal "NaN"/"None" strings from upstream resolution
    valid = df[
        df["sender_resolved"].notna()
        & (df["sender_resolved"] != "")
        & ~df["sender_resolved"].astype(str).str.strip().str.lower().isin(
            ["nan", "none", "null"]
        )
    ]
    if valid.empty:
        return {}

    for sender, grp in valid.groupby("sender_resolved", sort=False):
        n = len(grp)
        speech_acts_flat = [
            act
            for acts in grp.get("speech_acts", pd.Series([[] for _ in range(n)]))
            for act in (acts or [])
        ]

        # Basic counts
        n_decision = int(grp.get("decision_carrying", pd.Series([False] * n)).sum())
        n_action = int(grp.get("action_required", pd.Series([False] * n)).sum())
        n_commit = int(grp.get("commitment_made", pd.Series([False] * n)).sum())

        # Speech-act fractions
        n_request = sum(1 for a in speech_acts_flat if a == "request")
        n_commit_sa = sum(1 for a in speech_acts_flat if a == "commit")
        n_deliver = sum(1 for a in speech_acts_flat if a == "deliver")

        pct_request = n_request / n if n else 0.0
        pct_commit = n_commit_sa / n if n else 0.0
        pct_deliver = n_deliver / n if n else 0.0
        pct_decision = n_decision / n if n else 0.0

        # Numeric means — use .get() with safe defaults
        sentiment_col = grp.get("sentiment", pd.Series([float("nan")] * n))
        body_col = grp.get("body_word_count", pd.Series([float("nan")] * n))
        to_col = grp.get("to_count", pd.Series([float("nan")] * n))
        latency_col = grp.get("reply_latency_hours", pd.Series([float("nan")] * n))

        mean_sentiment = float(pd.to_numeric(sentiment_col, errors="coerce").mean())
        mean_body_words = float(pd.to_numeric(body_col, errors="coerce").mean())
        mean_to_count = float(pd.to_numeric(to_col, errors="coerce").mean())
        mean_latency = float(pd.to_numeric(latency_col, errors="coerce").mean())

        # Thread + off-hours
        initiator_col = grp.get("is_thread_initiator", pd.Series([False] * n))
        off_col = grp.get("is_off_hours", pd.Series([False] * n))
        pct_initiator = float(initiator_col.astype(bool).mean()) if n else 0.0
        pct_off_hours = float(off_col.astype(bool).mean()) if n else 0.0

        # Politeness / hedge
        pol_col = grp.get("politeness_score", pd.Series([float("nan")] * n))
        hedge_col = grp.get("hedge_count", pd.Series([float("nan")] * n))
        politeness_baseline = float(pd.to_numeric(pol_col, errors="coerce").mean())
        hedge_count_sum = pd.to_numeric(hedge_col, errors="coerce").sum()
        hedge_rate = float(hedge_count_sum / n) if n else 0.0

        # Topics and entities — top 3 by frequency
        topics_flat: list[str] = [
            t
            for ts in grp.get("topics", pd.Series([[] for _ in range(n)]))
            for t in (ts or [])
        ]
        entities_flat: list[str] = [
            e
            for es in grp.get("entities_mentioned", pd.Series([[] for _ in range(n)]))
            for e in (es or [])
        ]
        top3_topics = [t for t, _ in Counter(topics_flat).most_common(3)]
        top3_entities = [e for e, _ in Counter(entities_flat).most_common(3)]

        # Most active year
        date_col = grp.get("date", pd.Series([pd.NaT] * n))
        dates = pd.to_datetime(date_col, utc=True, errors="coerce").dropna()
        if len(dates) > 0:
            most_active_year = int(dates.dt.year.value_counts().idxmax())
        else:
            most_active_year = None

        result[sender] = {
            "n_emails_sent": n,
            "n_decision_carrying_sent": n_decision,
            "n_action_required_sent": n_action,
            "n_commitment_made_sent": n_commit,
            "pct_request_sent": pct_request,
            "pct_commit_sent": pct_commit,
            "pct_deliver_sent": pct_deliver,
            "pct_decision_carrying": pct_decision,
            "mean_sentiment_sent": mean_sentiment,
            "mean_body_words": mean_body_words,
            "mean_to_count": mean_to_count,
            "pct_thread_initiator": pct_initiator,
            "mean_reply_latency_received_hours": mean_latency,
            "pct_off_hours": pct_off_hours,
            "politeness_baseline": politeness_baseline,
            "hedge_rate": hedge_rate,
            "top3_topics": top3_topics,
            "top3_entities": top3_entities,
            "most_active_year": most_active_year,
        }

    return result


# ---------------------------------------------------------------------------
# Public API: upsert_pair_signals
# ---------------------------------------------------------------------------

def upsert_pair_signals(cur, pair_row: dict) -> None:
    """INSERT-ON-CONFLICT upsert into pair_signals table.

    Args:
        cur: psycopg cursor.
        pair_row: dict with keys matching pair_signals columns.
    """

    def _f(val, default=None):
        """Return None for NaN floats so Postgres receives NULL."""
        if val is None:
            return default
        try:
            if isinstance(val, float) and (val != val):  # NaN check
                return None
        except TypeError:
            pass
        return val

    cur.execute(
        """
        INSERT INTO pair_signals (
            sender_id, recipient_id,
            n_emails, n_to, n_cc,
            n_request_sent, n_commit_sent, n_deliver_sent, n_propose_sent,
            n_decision, n_action_required,
            mean_sentiment, mean_body_words, mean_reply_latency_h,
            length_asymmetry, request_commit_ratio,
            first_email_date, last_email_date,
            aggregated_at
        ) VALUES (
            %(sender_id)s, %(recipient_id)s,
            %(n_emails)s, %(n_to)s, %(n_cc)s,
            %(n_request_sent)s, %(n_commit_sent)s, %(n_deliver_sent)s, %(n_propose_sent)s,
            %(n_decision)s, %(n_action_required)s,
            %(mean_sentiment)s, %(mean_body_words)s, %(mean_reply_latency_h)s,
            %(length_asymmetry)s, %(request_commit_ratio)s,
            %(first_email_date)s, %(last_email_date)s,
            NOW()
        )
        ON CONFLICT (sender_id, recipient_id) DO UPDATE SET
            n_emails             = EXCLUDED.n_emails,
            n_to                 = EXCLUDED.n_to,
            n_cc                 = EXCLUDED.n_cc,
            n_request_sent       = EXCLUDED.n_request_sent,
            n_commit_sent        = EXCLUDED.n_commit_sent,
            n_deliver_sent       = EXCLUDED.n_deliver_sent,
            n_propose_sent       = EXCLUDED.n_propose_sent,
            n_decision           = EXCLUDED.n_decision,
            n_action_required    = EXCLUDED.n_action_required,
            mean_sentiment       = EXCLUDED.mean_sentiment,
            mean_body_words      = EXCLUDED.mean_body_words,
            mean_reply_latency_h = EXCLUDED.mean_reply_latency_h,
            length_asymmetry     = EXCLUDED.length_asymmetry,
            request_commit_ratio = EXCLUDED.request_commit_ratio,
            first_email_date     = EXCLUDED.first_email_date,
            last_email_date      = EXCLUDED.last_email_date,
            aggregated_at        = NOW()
        """,
        {
            "sender_id": pair_row["sender_id"],
            "recipient_id": pair_row["recipient_id"],
            "n_emails": int(pair_row.get("n_emails", 0)),
            "n_to": int(pair_row.get("n_to", 0)),
            "n_cc": int(pair_row.get("n_cc", 0)),
            "n_request_sent": int(pair_row.get("n_request_sent", 0)),
            "n_commit_sent": int(pair_row.get("n_commit_sent", 0)),
            "n_deliver_sent": int(pair_row.get("n_deliver_sent", 0)),
            "n_propose_sent": int(pair_row.get("n_propose_sent", 0)),
            "n_decision": int(pair_row.get("n_decision", 0)),
            "n_action_required": int(pair_row.get("n_action_required", 0)),
            "mean_sentiment": _f(pair_row.get("mean_sentiment")),
            "mean_body_words": _f(pair_row.get("mean_body_words")),
            "mean_reply_latency_h": _f(pair_row.get("mean_reply_latency_h")),
            "length_asymmetry": _f(pair_row.get("length_asymmetry")),
            "request_commit_ratio": _f(pair_row.get("request_commit_ratio")),
            "first_email_date": pair_row.get("first_email_date"),
            "last_email_date": pair_row.get("last_email_date"),
        },
    )


# ---------------------------------------------------------------------------
# Public API: write_person_aggregates_to_neo4j
# ---------------------------------------------------------------------------

def write_person_aggregates_to_neo4j(driver, aggregates: dict[str, dict]) -> int:
    """MERGE Person nodes and set aggregate properties.

    Args:
        driver: Neo4j GraphDatabase driver.
        aggregates: Dict mapping person name -> properties dict.

    Returns:
        Count of Person nodes updated.
    """
    if not aggregates:
        return 0

    rows = []
    for name, props in aggregates.items():
        row: dict[str, Any] = {"name": name}
        row.update({
            k: v for k, v in props.items()
            if k not in ("top3_topics", "top3_entities")  # lists handled separately
        })
        row["top3_topics"] = props.get("top3_topics") or []
        row["top3_entities"] = props.get("top3_entities") or []
        # Sanitise NaN to None for Neo4j
        cleaned = {}
        for k, v in row.items():
            if isinstance(v, float) and v != v:  # NaN
                cleaned[k] = None
            else:
                cleaned[k] = v
        rows.append(cleaned)

    query = """
    UNWIND $rows AS row
    MERGE (p:Person {name: row.name})
    SET p.n_emails_sent                    = row.n_emails_sent,
        p.n_decision_carrying_sent         = row.n_decision_carrying_sent,
        p.n_action_required_sent           = row.n_action_required_sent,
        p.n_commitment_made_sent           = row.n_commitment_made_sent,
        p.pct_request_sent                 = row.pct_request_sent,
        p.pct_commit_sent                  = row.pct_commit_sent,
        p.pct_deliver_sent                 = row.pct_deliver_sent,
        p.pct_decision_carrying            = row.pct_decision_carrying,
        p.mean_sentiment_sent              = row.mean_sentiment_sent,
        p.mean_body_words                  = row.mean_body_words,
        p.mean_to_count                    = row.mean_to_count,
        p.pct_thread_initiator             = row.pct_thread_initiator,
        p.mean_reply_latency_received_hours = row.mean_reply_latency_received_hours,
        p.pct_off_hours                    = row.pct_off_hours,
        p.politeness_baseline              = row.politeness_baseline,
        p.hedge_rate                       = row.hedge_rate,
        p.top3_topics                      = row.top3_topics,
        p.top3_entities                    = row.top3_entities,
        p.most_active_year                 = row.most_active_year,
        p.signal_aggregates_at             = datetime()
    """

    with driver.session() as session:
        session.run(query, rows=rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Public API: write_decision_email_nodes
# ---------------------------------------------------------------------------

def write_decision_email_nodes(
    driver,
    df: pd.DataFrame,
    threshold_property: str = "decision_carrying",
) -> int:
    """MERGE Email nodes for rows where df[threshold_property] is True.

    Args:
        driver: Neo4j GraphDatabase driver.
        df: DataFrame with per-email rows.
        threshold_property: Column name to filter on (default: "decision_carrying").

    Returns:
        Count of Email nodes upserted.
    """
    if df.empty:
        return 0

    decision_df = df[df[threshold_property].astype(bool)]
    if decision_df.empty:
        return 0

    rows = []
    for _, row in decision_df.iterrows():
        recipients = row.get("recipients_resolved") or []
        if not isinstance(recipients, list):
            recipients = []
        rows.append({
            "email_id": row.get("email_id") or "",
            "subject": str(row.get("subject") or ""),
            "date": row.get("date").isoformat() if pd.notna(row.get("date")) else None,
            "sender_resolved": row.get("sender_resolved") or "",
            "recipients_resolved": [r for r in recipients if r],
            "decision_carrying": True,
        })

    if not rows:
        return 0

    # Single Cypher pass:
    #   1. MERGE Email node
    #   2. MERGE (Email)-[:SENT_BY]->(Person) for the resolved sender
    #   3. MERGE (Email)-[:SENT_TO]->(Person) for each resolved recipient
    # Persons are MATCHed (not MERGEd) — Stage 6a already created them, and we
    # don't want to silently spawn ghost Person nodes from a typo.
    query = """
    UNWIND $rows AS row
    MERGE (e:Email {email_id: row.email_id})
    SET e.subject          = row.subject,
        e.date             = row.date,
        e.sender_resolved  = row.sender_resolved,
        e.decision_carrying = true
    WITH e, row
    OPTIONAL MATCH (sender:Person {name: row.sender_resolved})
    FOREACH (_ IN CASE WHEN sender IS NULL THEN [] ELSE [1] END |
        MERGE (e)-[:SENT_BY]->(sender)
    )
    WITH e, row
    UNWIND row.recipients_resolved AS recipient_name
    OPTIONAL MATCH (recipient:Person {name: recipient_name})
    FOREACH (_ IN CASE WHEN recipient IS NULL THEN [] ELSE [1] END |
        MERGE (e)-[:SENT_TO]->(recipient)
    )
    """

    chunk_size = 500
    n_written = 0
    with driver.session() as session:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i: i + chunk_size]
            session.run(query, rows=chunk)
            n_written += len(chunk)

    return n_written


def write_email_cluster_edges(
    driver,
    df: pd.DataFrame,
    cluster_labels_path: Path,
) -> tuple[int, int]:
    """MERGE (Email)-[:ABOUT_PROJECT]->(Project) and (Email)-[:ABOUT_TOPIC]->(Topic).

    Reads ``cluster_labels.parquet`` (written by Stage 2b) and joins the
    integer ``project_label`` / ``topic_label`` columns to the canonical
    ``project_id`` / ``topic_id`` properties on the Project / Topic nodes
    (formatted as ``P{label:03d}`` / ``T{label:03d}`` to match Stage 2b's
    naming convention).

    Only emits edges from emails that already exist as Email nodes in Neo4j
    (i.e. ``decision_carrying = true`` rows from the same DataFrame).
    Noise labels (``label == -1``) are skipped.

    Args:
        driver: Neo4j GraphDatabase driver.
        df: DataFrame holding the per-email rows from this run (used to filter
            cluster_labels to only the decision-carrying email_ids).
        cluster_labels_path: Path to the Stage 2b cluster_labels.parquet.

    Returns:
        ``(n_project_edges, n_topic_edges)`` MERGE counts (best-effort,
        equals the number of label rows passed in regardless of whether the
        target Project/Topic node existed).
    """
    if not cluster_labels_path.exists():
        print(f"      {cluster_labels_path.name} not found — skipping Email→cluster edges")
        return (0, 0)

    if df.empty:
        return (0, 0)

    decision_ids = set(df.loc[df["decision_carrying"].astype(bool), "email_id"].astype(str))
    if not decision_ids:
        return (0, 0)

    labels_df = pd.read_parquet(cluster_labels_path)
    labels_df = labels_df[labels_df["email_id"].astype(str).isin(decision_ids)].copy()

    project_rows = [
        {"email_id": str(r["email_id"]), "project_id": f"P{int(r['project_label']):03d}"}
        for _, r in labels_df.iterrows()
        if int(r["project_label"]) >= 0
    ]
    topic_rows = [
        {"email_id": str(r["email_id"]), "topic_id": f"T{int(r['topic_label']):03d}"}
        for _, r in labels_df.iterrows()
        if int(r["topic_label"]) >= 0
    ]

    n_proj = 0
    n_top = 0
    chunk_size = 500
    with driver.session() as session:
        for i in range(0, len(project_rows), chunk_size):
            chunk = project_rows[i: i + chunk_size]
            session.run(
                """
                UNWIND $rows AS row
                MATCH (e:Email {email_id: row.email_id})
                MATCH (p:Project {project_id: row.project_id})
                MERGE (e)-[:ABOUT_PROJECT]->(p)
                """,
                rows=chunk,
            )
            n_proj += len(chunk)
        for i in range(0, len(topic_rows), chunk_size):
            chunk = topic_rows[i: i + chunk_size]
            session.run(
                """
                UNWIND $rows AS row
                MATCH (e:Email {email_id: row.email_id})
                MATCH (t:Topic {topic_id: row.topic_id})
                MERGE (e)-[:ABOUT_TOPIC]->(t)
                """,
                rows=chunk,
            )
            n_top += len(chunk)

    return (n_proj, n_top)


# ---------------------------------------------------------------------------
# Data loading from Postgres
# ---------------------------------------------------------------------------

def load_emails_from_pg(cur, limit: int | None = None) -> pd.DataFrame:
    """Read aggregatable columns from embeddings_email.

    Only fetches rows where at least Stage 1 embedding exists.
    Signals columns (speech_acts etc.) may be NULL for rows not yet processed
    by Stage 2a — they are included and treated as defaults in aggregation.
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    cur.execute(
        f"""
        SELECT
            email_id,
            thread_id,
            sender_resolved,
            recipients_resolved,
            date,
            subject,
            speech_acts,
            action_required,
            commitment_made,
            decision_carrying,
            sentiment,
            body_word_count,
            to_count,
            is_thread_initiator,
            is_off_hours,
            reply_latency_hours,
            politeness_score,
            hedge_count,
            topics,
            entities_mentioned
        FROM embeddings_email
        ORDER BY date
        {limit_clause}
        """
    )
    rows = cur.fetchall()
    cols = [
        "email_id", "thread_id", "sender_resolved", "recipients_resolved",
        "date", "subject", "speech_acts", "action_required", "commitment_made",
        "decision_carrying", "sentiment", "body_word_count", "to_count",
        "is_thread_initiator", "is_off_hours", "reply_latency_hours",
        "politeness_score", "hedge_count", "topics", "entities_mentioned",
    ]
    df = pd.DataFrame(rows, columns=cols)

    # Normalise list columns (Postgres JSONB comes back as Python list/None)
    for col in ("recipients_resolved", "speech_acts", "topics", "entities_mentioned"):
        df[col] = df[col].apply(lambda v: v if isinstance(v, list) else [])

    return df


# ---------------------------------------------------------------------------
# Optional: write Project / Topic nodes from CSVs
# ---------------------------------------------------------------------------

def write_project_nodes(driver, projects_csv: Path) -> int:
    """MERGE Project nodes keyed by ``project_id`` (e.g. 'P003').

    Switching the merge key from ``name`` to ``project_id`` lets edges like
    ``(:Email)-[:ABOUT_PROJECT]->(:Project)`` match by the stable cluster id
    (Stage 2b's ``f"P{label:03d}"``) regardless of LLM-named string drift.
    """
    df = pd.read_csv(projects_csv)
    rows = df.to_dict("records")
    query = """
    UNWIND $rows AS row
    MERGE (p:Project {project_id: row.project_id})
    SET p += row
    """
    with driver.session() as session:
        session.run(query, rows=rows)
    return len(rows)


def write_topic_nodes(driver, topics_csv: Path) -> int:
    """MERGE Topic nodes keyed by ``topic_id`` (e.g. 'T012').

    See ``write_project_nodes`` for the rationale on the id-not-name key.
    Note this Topic node represents the Stage 2b cluster topic
    (``topic_id`` like 'T012'), distinct from the canonicalized Topic created
    by Stage 6b sync_kg (``name`` like 'Trading'). The two will need to be
    reconciled in a follow-up — for now, edges from Email to cluster Topic
    use this id-keyed node.
    """
    df = pd.read_csv(topics_csv)
    rows = df.to_dict("records")
    query = """
    UNWIND $rows AS row
    MERGE (t:Topic {topic_id: row.topic_id})
    SET t += row
    """
    with driver.session() as session:
        session.run(query, rows=rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(limit: int | None = None) -> None:
    import time

    t0 = time.monotonic()

    print(f"[1/5] Connecting to Postgres ({PG_DSN.split('@')[-1]})...")
    pg = psycopg.connect(PG_DSN, autocommit=False)
    register_vector(pg)
    cur = pg.cursor()

    print("[2/5] Loading emails from embeddings_email...")
    df = load_emails_from_pg(cur, limit=limit)
    print(f"      {len(df):,} rows loaded")
    if df.empty:
        print("No rows to aggregate. Exiting.")
        pg.close()
        return

    print("[3/5] Computing pair_signals...")
    pairs = compute_pair_signals(df)
    print(f"      {len(pairs):,} directed pairs")

    print("      Upserting into pair_signals table...")
    n_upserted = 0
    for _, row in pairs.iterrows():
        upsert_pair_signals(cur, row.to_dict())
        n_upserted += 1
    pg.commit()
    print(f"      {n_upserted:,} rows upserted")

    print("[4/5] Computing Person aggregates + writing to Neo4j...")
    aggregates = compute_person_aggregates(df)
    print(f"      {len(aggregates):,} persons")

    try:
        from neo4j import GraphDatabase  # type: ignore

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        n_persons = write_person_aggregates_to_neo4j(driver, aggregates)
        print(f"      {n_persons:,} Person nodes updated in Neo4j")

        n_emails_written = write_decision_email_nodes(driver, df)
        print(f"      {n_emails_written:,} decision Email nodes written to Neo4j")

        print("[5/5] Optional: Project + Topic nodes from CSVs...")
        projects_csv = PROCESSED_DIR / "projects.csv"
        topics_csv = PROCESSED_DIR / "topics.csv"
        cluster_labels_path = PROCESSED_DIR / "cluster_labels.parquet"
        projects_written = topics_written = False
        if projects_csv.exists():
            n_proj = write_project_nodes(driver, projects_csv)
            projects_written = True
            print(f"      {n_proj:,} Project nodes written")
        else:
            print(f"      {projects_csv} not found — skipping Project nodes")
        if topics_csv.exists():
            n_topics = write_topic_nodes(driver, topics_csv)
            topics_written = True
            print(f"      {n_topics:,} Topic nodes written")
        else:
            print(f"      {topics_csv} not found — skipping Topic nodes")

        if projects_written or topics_written:
            n_proj_e, n_topic_e = write_email_cluster_edges(
                driver, df, cluster_labels_path,
            )
            print(
                f"      Email→cluster edges: ABOUT_PROJECT={n_proj_e:,}  "
                f"ABOUT_TOPIC={n_topic_e:,}"
            )

        driver.close()

    except Exception as e:  # noqa: BLE001
        print(f"[warn] Neo4j unreachable or error: {e}. Skipping graph writes.")

    pg.close()
    dt = time.monotonic() - t0

    print(f"\nDone in {dt:.1f}s.")
    print(f"  pair_signals rows upserted : {n_upserted:,}")
    print(f"  persons computed           : {len(aggregates):,}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: aggregate email signals into pair_signals + Person properties."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows from embeddings_email (dev/smoke).",
    )
    args = parser.parse_args(argv)
    run(limit=args.limit)


if __name__ == "__main__":
    main()
