"""Escape-hatch query tools: raw read-only Cypher and SQL.

These tools give simulation agents direct access to the underlying stores
when no structured tool fits the question. Both enforce strict read-only
allowlists and cap result sets so a single tool call cannot exhaust the
context window.

Factories follow the same KGTool contract as tools.py (Neo4j) and
tools_pg.py (Postgres). They are NOT registered in build_default_registry
here — Task 1.11 handles that wiring.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from orggraph.pipeline.agents.tools import KGTool


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MAX_ROWS_HARD_LIMIT = 200


def _safe_json(v: Any) -> Any:
    """Coerce a value returned by a Neo4j or psycopg driver into a type that
    json.dumps() can handle without raising TypeError."""
    # Lazy import so this module stays importable with no drivers installed.
    try:
        from neo4j.graph import Node, Relationship
    except ImportError:
        Node = None
        Relationship = None

    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if Node is not None and isinstance(v, Node):
        return dict(v)
    if Relationship is not None and isinstance(v, Relationship):
        return {
            "type": v.type,
            "start": v.start_node.element_id if v.start_node else None,
            "end": v.end_node.element_id if v.end_node else None,
            **dict(v),
        }
    # datetime-like objects (neo4j.time.DateTime, date, datetime, etc.)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    # Fall back: cast to string (list / dict values recurse on their own)
    if isinstance(v, (list, tuple)):
        return [_safe_json(i) for i in v]
    if isinstance(v, dict):
        return {k: _safe_json(val) for k, val in v.items()}
    return str(v)


# ---------------------------------------------------------------------------
# make_run_cypher_tool
# ---------------------------------------------------------------------------

# Keywords that indicate a write-mode Cypher statement.  Applied
# case-insensitively against the raw query string.
_CYPHER_DENY_RE = re.compile(
    r"\bcreate\b"
    r"|\bmerge\b"
    r"|\bdelete\b"
    r"|\bremove\b"
    r"|\bset\b"
    r"|\bdrop\b"
    r"|\bdetach\b"
    r"|load\s+csv"
    r"|call\s+apoc"
    r"|call\s+db\.",
    re.IGNORECASE,
)

_CYPHER_SCHEMA_REF = """\
Neo4j schema:
  (:Person {name, title, tier, tier_v3, role_summary,
            composite_score_v3, composite_score, pagerank, betweenness,
            in_degree, community})
  (:Email {email_id, subject, date, sender_resolved})
  (:Topic {topic_id, name, kind})  -- kind: 'cluster' or 'canonical'
  (:Project {project_id, name})
  (:ExternalEntity {name, kind})
Relationships:
  (Person)-[:REPORTS_TO]->(Person)
  (Person)-[:KNOWS_ABOUT]->(Topic)
  (Person)-[:MEMBER_OF]->(Project)
  (Email)-[:SENT_BY]->(Person)
  (Email)-[:SENT_TO]->(Person)
  (Email)-[:ABOUT_TOPIC]->(Topic)
  (Email)-[:MENTIONS]->(ExternalEntity)
Email.sentiment / speech_acts / etc. live in Postgres, NOT Neo4j —
use run_sql against embeddings_email for those fields."""


def make_run_cypher_tool(driver) -> KGTool:
    """Open a read-only Cypher query against the OrgGraph knowledge graph.

    The query is matched against an allowlist regex that rejects any
    write-mode operation. Results are capped at ``max_rows`` rows.
    """

    def _call(*, query: str, max_rows: int = 50) -> dict:
        max_rows = max(1, min(int(max_rows), _MAX_ROWS_HARD_LIMIT))

        m = _CYPHER_DENY_RE.search(query)
        if m:
            return {
                "error": "DisallowedCypher",
                "details": f"Disallowed keyword matched: {m.group(0)!r}",
            }

        rows: list[dict] = []
        truncated = False
        with driver.session(default_access_mode="READ") as session:
            result = session.run(query)
            count = 0
            for record in result:
                if count >= max_rows:
                    truncated = True
                    break
                row = {k: _safe_json(v) for k, v in dict(record).items()}
                rows.append(row)
                count += 1

        return {"rows": rows, "n_rows": len(rows), "truncated": truncated}

    return KGTool(
        name="run_cypher",
        description=(
            "ESCAPE HATCH — use ONLY when no structured tool fits the question. "
            "Executes a read-only Cypher query against the OrgGraph Neo4j instance. "
            "Write-mode keywords (CREATE, MERGE, DELETE, REMOVE, SET, DROP, DETACH, "
            "LOAD CSV, CALL apoc.*, CALL db.*) are rejected even inside string "
            "literals — this is a known false-positive the tool accepts. "
            "Results are capped at max_rows (default 50, max 200). "
            f"{_CYPHER_SCHEMA_REF}"
        ),
        args_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A read-only Cypher MATCH query.",
                },
                "max_rows": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": _MAX_ROWS_HARD_LIMIT,
                    "description": "Maximum rows to return. Hard cap is 200.",
                },
            },
            "required": ["query"],
        },
        callable=_call,
    )


# ---------------------------------------------------------------------------
# make_run_sql_tool
# ---------------------------------------------------------------------------

# Keywords that indicate a write or destructive SQL statement.
_SQL_DENY_RE = re.compile(
    r"\binsert\b"
    r"|\bupdate\b"
    r"|\bdelete\b"
    r"|\bdrop\b"
    r"|\bcreate\b"
    r"|\balter\b"
    r"|\btruncate\b"
    r"|\bcopy\b"
    r"|\bgrant\b"
    r"|\brevoke\b"
    r"|\bvacuum\b"
    r"|comment\s+on"
    r"|\breindex\b",
    re.IGNORECASE,
)

_SQL_SCHEMA_REF = """\
Postgres tables (canonical):
  embeddings_email(
    email_id TEXT PRIMARY KEY,            -- 16-char hex hash, not int
    thread_id TEXT, sender_email TEXT, sender_resolved TEXT,
    recipients_emails JSONB, recipients_resolved JSONB,
    date TIMESTAMPTZ, subject TEXT,
    body_truncated TEXT, body_chars INT,
    topics JSONB, intent TEXT, sentiment REAL, decision_carrying BOOL,
    mentions_money BOOL, mentions_regulator BOOL, entities_mentioned JSONB,
    speech_acts JSONB, action_required BOOL, commitment_made BOOL,
    body_word_count INT, reply_latency_hours REAL,
    is_thread_initiator BOOL, is_thread_closer BOOL, thread_position INT,
    politeness_score REAL, embedding vector(768))

  pair_signals(
    sender_id TEXT, recipient_id TEXT,    -- person names, not ids
    n_emails INT, n_to INT, n_cc INT,
    n_request_sent INT, n_commit_sent INT, n_deliver_sent INT,
    n_propose_sent INT, n_decision INT, n_action_required INT,
    request_commit_ratio REAL, mean_reply_latency_h REAL,
    mean_sentiment REAL, mean_body_words REAL, length_asymmetry REAL,
    first_email_date TIMESTAMPTZ, last_email_date TIMESTAMPTZ)

  embeddings_person(person_id TEXT, name TEXT, role TEXT,
                    department TEXT, embedding vector(768))
  embeddings_entity(...similar shape for ExternalEntity...)

There is NO clean_emails table. All email metadata + enrichment lives
in embeddings_email."""


def make_run_sql_tool(pg_conn_factory: Callable[[], Any]) -> KGTool:
    """Open a read-only SQL query against the OrgGraph Postgres warehouse.

    The query must start with SELECT or WITH and is matched against an
    allowlist regex that rejects any write-mode operation. The connection
    is set to TRANSACTION READ ONLY before execute. Results are capped at
    ``max_rows`` rows.
    """

    def _call(*, query: str, max_rows: int = 50) -> dict:
        max_rows = max(1, min(int(max_rows), _MAX_ROWS_HARD_LIMIT))

        # First token must be SELECT or WITH.
        first_token = query.lstrip().split()[0].upper() if query.strip() else ""
        if first_token not in ("SELECT", "WITH"):
            return {
                "error": "DisallowedSql",
                "details": "query must start with SELECT or WITH",
            }

        # Reject any dangerous keyword anywhere in the query string.
        m = _SQL_DENY_RE.search(query)
        if m:
            return {
                "error": "DisallowedSql",
                "details": f"Disallowed keyword matched: {m.group(0)!r}",
            }

        rows: list[dict] = []
        truncated = False
        with pg_conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(query)
                columns = [desc[0] for desc in cur.description]
                count = 0
                for raw_row in cur:
                    if count >= max_rows:
                        truncated = True
                        break
                    row = {
                        col: _safe_json(val)
                        for col, val in zip(columns, raw_row)
                    }
                    rows.append(row)
                    count += 1

        return {"rows": rows, "n_rows": len(rows), "truncated": truncated}

    return KGTool(
        name="run_sql",
        description=(
            "ESCAPE HATCH — use ONLY when no structured tool fits the question. "
            "Executes a read-only SQL query against the OrgGraph Postgres warehouse. "
            "The query must start with SELECT or WITH. Write-mode keywords "
            "(INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, COPY, GRANT, "
            "REVOKE, VACUUM, COMMENT ON, REINDEX) are rejected even inside string "
            "literals — this is a known false-positive the tool accepts. "
            "The session is set to TRANSACTION READ ONLY before the query runs. "
            "Results are capped at max_rows (default 50, max 200). "
            f"{_SQL_SCHEMA_REF}"
        ),
        args_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A read-only SQL query starting with SELECT or WITH.",
                },
                "max_rows": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": _MAX_ROWS_HARD_LIMIT,
                    "description": "Maximum rows to return. Hard cap is 200.",
                },
            },
            "required": ["query"],
        },
        callable=_call,
    )
