"""R1 — vector-only retrieval over embeddings_email, then MiniMax answer."""

from __future__ import annotations

import os
import re
from typing import Any

PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://{os.environ.get('POSTGRES_USER','orggraph')}:"
    f"{os.environ.get('POSTGRES_PASSWORD','orggraph2026')}@localhost:"
    f"{os.environ.get('POSTGRES_PORT','5432')}/"
    f"{os.environ.get('POSTGRES_DB','orggraph')}",
)


def pg_cosine_top_k(conn: Any, q_vec: list[float], k: int = 10) -> list[dict]:
    """Return the *k* nearest emails to *q_vec* by cosine distance.

    *conn* is a psycopg connection with pgvector registered. The function
    is connection-agnostic for testability; the orchestrator owns the
    connection lifecycle.

    Returns a list of dicts with keys:
        email_id, subject, body_truncated, date, sender_resolved
    """
    sql = (
        "SELECT email_id, subject, body_truncated, date, sender_resolved "
        "FROM embeddings_email "
        "ORDER BY embedding <=> %s::vector "
        "LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (q_vec, k))
        rows = cur.fetchall()
    return [
        {
            "email_id": r[0],
            "subject": r[1],
            "body_truncated": r[2],
            "date": r[3],
            "sender_resolved": r[4],
        }
        for r in rows
    ]


ANSWER_PROMPT = """You are answering a question using only the emails below.
Cite the email IDs you used in square brackets, e.g. [E1] or [E1, E3].
Give your final answer as a complete sentence prefixed with "ANSWER:".
The answer sentence should restate the subject of the question (e.g., for
"Who faxed X to Y?" answer "<person> faxed X to Y [E1].", not just "<person>").
If the emails do not contain the answer, say so explicitly.

Emails:
{passages}

Question: {question}
"""


def format_passages(rows: list[dict]) -> str:
    """Render retrieved rows as a numbered passage block."""
    if not rows:
        return "(no emails retrieved)"
    parts = []
    for i, r in enumerate(rows, start=1):
        date_str = str(r.get("date") or "unknown date")
        sender = r.get("sender_resolved") or "unknown sender"
        subject = (r.get("subject") or "").strip()
        body = (r.get("body_truncated") or "").strip()
        parts.append(f"[E{i}] ({date_str}, {sender}) {subject}\n{body}")
    return "\n\n".join(parts)


_CITE_RE = re.compile(r"\[E(\d+(?:\s*,\s*E?\d+)*)\]")


def parse_citations(answer_text: str, *, retrieved_ids: list[str]) -> tuple[str, ...]:
    """Extract cited [Ek] indices from the answer text and map to retrieved IDs.

    Out-of-range indices are silently dropped. Duplicates are deduplicated
    while preserving first-citation order.
    """
    cited: list[str] = []
    seen: set[str] = set()
    for match in _CITE_RE.finditer(answer_text):
        for tok in match.group(1).split(","):
            tok = tok.strip().lstrip("E").strip()
            if not tok.isdigit():
                continue
            idx = int(tok) - 1
            if 0 <= idx < len(retrieved_ids):
                eid = retrieved_ids[idx]
                if eid not in seen:
                    seen.add(eid)
                    cited.append(eid)
    return tuple(cited)


EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "ollama")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embeddinggemma")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit")


def make_embed_client():
    """Return a configured OpenAI SDK client pointing at the Ollama embedding endpoint."""
    from openai import OpenAI

    return OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)


def make_answer_client():
    """Return a configured OpenAI SDK client pointing at the vLLM MiniMax endpoint."""
    from openai import OpenAI

    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


def embed_query(question: str, *, embed_client, model: str = EMBED_MODEL) -> list[float]:
    """Embed a question with the Stage 1 encoder (Ollama embeddinggemma 768-dim)."""
    resp = embed_client.embeddings.create(model=model, input=[question])
    return list(resp.data[0].embedding)


def call_minimax_answer(
    prompt: str,
    *,
    answer_client,
    model: str = VLLM_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    seed: int = 42,
) -> str:
    """Send *prompt* as a single user message and return the completion content.

    Uses ``seed`` to force deterministic sampling at the vLLM layer — this
    eliminates output drift between R1 and R2 calls with identical prompts
    that arise from vLLM's continuous batching.
    """
    resp = answer_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    content = resp.choices[0].message.content
    return content or ""


import time  # noqa: E402

from orggraph.pipeline.eval.rq3 import Answer  # noqa: E402


def answer_r1(
    question: str,
    *,
    pg_conn,
    embed_client,
    answer_client,
    k: int = 10,
    embed_model: str = EMBED_MODEL,
    answer_model: str = VLLM_MODEL,
    temperature: float = 0.0,
) -> Answer:
    """Run R1: embed query, top-k cosine search, MiniMax answer with citations."""
    t0 = time.monotonic()
    q_vec = embed_query(question, embed_client=embed_client, model=embed_model)
    rows = pg_cosine_top_k(pg_conn, q_vec, k=k)
    passages = format_passages(rows)
    prompt = ANSWER_PROMPT.format(passages=passages, question=question)
    answer_text = call_minimax_answer(
        prompt, answer_client=answer_client, model=answer_model, temperature=temperature
    )
    retrieved_ids = [r["email_id"] for r in rows]
    cited = parse_citations(answer_text, retrieved_ids=retrieved_ids)
    latency_ms = int((time.monotonic() - t0) * 1000)
    return Answer(
        text=answer_text,
        cited_email_ids=cited,
        retrieved_email_ids=tuple(retrieved_ids),
        retrieved_bodies=tuple(r["body_truncated"] or "" for r in rows),
        latency_ms=latency_ms,
        condition="R1",
    )
