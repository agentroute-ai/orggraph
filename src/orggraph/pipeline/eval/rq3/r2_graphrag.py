"""R2 — Cypher-by-qtype GraphRAG over Neo4j, then MiniMax answer."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from orggraph.pipeline.eval.rq3.r1_vector import (
    ANSWER_PROMPT,
    EMBED_MODEL,
    VLLM_MODEL,
    call_minimax_answer,
    embed_query,
    format_passages,
    parse_citations,
    pg_cosine_top_k,
)
from orggraph.pipeline.validate.rq3_eval_set import classify_question_type

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "orggraph2026")


EXTRACT_ARGS_PROMPT = """Extract structured arguments from this question.
Return a JSON object with these optional keys:
  - person: the person name being asked about, or null
  - topic_keyword: the topic, project, or subject keyword, or null
  - date_range: {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null

For date_range:
  - "around <date>" → 30-day window centered on the date
  - "in <month> <year>" → first to last day of that month
  - "in <year>" → Jan 1 to Dec 31 of that year
  - "between <date_a> and <date_b>" → exactly that range
  - no explicit date → null

Question: {question}

JSON:
"""


def parse_permissive_json(text: str) -> dict:
    """Extract the first balanced ``{...}`` block from *text* and parse it.

    Returns ``{}`` if no valid JSON object can be found.
    """
    if not text:
        return {}
    # Find first opening brace
    start = text.find("{")
    if start < 0:
        return {}
    # Walk forward tracking depth, ignoring braces inside strings
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def extract_args(
    question: str,
    *,
    answer_client,
    model: str = VLLM_MODEL,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call MiniMax to extract {person, topic_keyword, date_range} from question."""
    prompt = EXTRACT_ARGS_PROMPT.format(question=question)
    raw = call_minimax_answer(
        prompt, answer_client=answer_client, model=model, temperature=temperature
    )
    return parse_permissive_json(raw)


CYPHER_FACTUAL = """
MATCH (e:Email)-[:ABOUT_TOPIC]->(t:Topic)
WHERE toLower(t.name) CONTAINS toLower($topic_keyword)
RETURN e.email_id AS email_id,
       e.subject AS subject,
       e.body_truncated AS body_truncated,
       toString(e.date) AS date,
       e.sender_resolved AS sender_resolved
ORDER BY e.date DESC
LIMIT 10
"""

CYPHER_RELATIONAL = """
MATCH (p:Person)<-[:SENT_BY|SENT_TO]-(e:Email)
WHERE toLower(p.name) CONTAINS toLower($person)
   OR toLower(coalesce(p.email, '')) CONTAINS toLower($person)
RETURN e.email_id AS email_id,
       e.subject AS subject,
       e.body_truncated AS body_truncated,
       toString(e.date) AS date,
       e.sender_resolved AS sender_resolved
ORDER BY e.date DESC
LIMIT 10
"""

CYPHER_TEMPORAL = """
MATCH (e:Email)
WHERE e.date >= date($date_start)
  AND e.date <= date($date_end)
  AND e.decision_carrying = true
RETURN e.email_id AS email_id,
       e.subject AS subject,
       e.body_truncated AS body_truncated,
       toString(e.date) AS date,
       e.sender_resolved AS sender_resolved
ORDER BY e.date ASC
LIMIT 10
"""

_CYPHER_BY_QTYPE = {
    "factual": CYPHER_FACTUAL,
    "relational": CYPHER_RELATIONAL,
    "temporal": CYPHER_TEMPORAL,
}


def cypher_for_qtype(qtype: str) -> str:
    if qtype not in _CYPHER_BY_QTYPE:
        raise ValueError(f"unknown qtype: {qtype!r}")
    return _CYPHER_BY_QTYPE[qtype]


def run_cypher(driver, qtype: str, args: dict) -> list[dict]:
    """Execute the qtype-specific Cypher against *driver* and return rows as dicts.

    Returns ``[]`` (without hitting the DB) when a required arg is missing:
      - factual: needs topic_keyword
      - relational: needs person
      - temporal: needs date_range with start and end
    """
    if qtype == "factual" and not args.get("topic_keyword"):
        return []
    if qtype == "relational" and not args.get("person"):
        return []
    if qtype == "temporal":
        dr = args.get("date_range") or {}
        if not dr.get("start") or not dr.get("end"):
            return []

    cypher = cypher_for_qtype(qtype)
    params: dict[str, Any] = {}
    if qtype == "factual":
        params["topic_keyword"] = args["topic_keyword"]
    elif qtype == "relational":
        params["person"] = args["person"]
    elif qtype == "temporal":
        params["date_start"] = args["date_range"]["start"]
        params["date_end"] = args["date_range"]["end"]

    with driver.session() as session:
        result = session.run(cypher, **params)
        return result.data()


from orggraph.pipeline.eval.rq3 import Answer  # noqa: E402


def answer_r2(
    question: str,
    *,
    neo4j_driver,
    answer_client,
    pg_conn=None,
    embed_client=None,
    k: int = 10,
    embed_model: str = EMBED_MODEL,
    answer_model: str = VLLM_MODEL,
    temperature: float = 0.0,
) -> Answer:
    """Run R2: classify qtype, extract args via MiniMax, dispatch Cypher,
    fall back to vector search if the result set is empty or args are missing,
    then MiniMax-answer with citations.

    The fallback uses *pg_conn* and *embed_client*. Pass ``None`` only if you
    can guarantee Cypher will succeed (e.g., in unit tests).
    """
    t0 = time.monotonic()
    qtype = classify_question_type(question)
    # Factual questions (per the qtype classifier) don't match the Cypher templates
    # well on this corpus — they ask about specific named values, not topics.
    # Skip the extract_args LLM hop and let the fallback do its job.
    if qtype == "factual":
        args = {}
        rows = []
    else:
        args = extract_args(question, answer_client=answer_client, model=answer_model, temperature=temperature)
        rows = run_cypher(neo4j_driver, qtype, args) if args else []

    # Backfill body_truncated from Postgres — Neo4j Email nodes don't store bodies
    if rows and pg_conn is not None:
        ids = [r["email_id"] for r in rows]
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT email_id, body_truncated FROM embeddings_email WHERE email_id = ANY(%s)",
                (ids,),
            )
            body_map = {r[0]: r[1] for r in cur.fetchall()}
        for r in rows:
            if not r.get("body_truncated"):
                r["body_truncated"] = body_map.get(r["email_id"], "") or ""

    used_fallback = False
    if not rows:
        if pg_conn is None or embed_client is None:
            rows = []
        else:
            q_vec = embed_query(question, embed_client=embed_client, model=embed_model)
            rows = pg_cosine_top_k(pg_conn, q_vec, k=k)
            used_fallback = True

    passages = format_passages(rows)
    prompt = ANSWER_PROMPT.format(passages=passages, question=question)
    answer_text = call_minimax_answer(
        prompt, answer_client=answer_client, model=answer_model, temperature=temperature
    )
    retrieved_ids = [r["email_id"] for r in rows]
    cited = parse_citations(answer_text, retrieved_ids=retrieved_ids)
    latency_ms = int((time.monotonic() - t0) * 1000)
    a = Answer(
        text=answer_text,
        cited_email_ids=cited,
        retrieved_email_ids=tuple(retrieved_ids),
        retrieved_bodies=tuple(r.get("body_truncated") or "" for r in rows),
        latency_ms=latency_ms,
        condition="R2",
    )
    # attach a private flag so the orchestrator can log fallback rate
    object.__setattr__(a, "_used_fallback", used_fallback)
    return a
