"""Stage 2a — Per-email LLM signal extraction.

Reads clean_emails.parquet, calls an LLM per email, writes structured signals
into embeddings_email (only the signal columns + signals_extracted_at).

Resumable: skips emails where signals_extracted_at IS NOT NULL.

Concurrency-safe: runs alongside Stage 1 (embed_emails.py) and Stage 1.5
(compute_email_metadata.py). All three write disjoint columns to
embeddings_email. Only rows already inserted by Stage 1 are processed here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import psycopg.types.json as pj
from pgvector.psycopg import register_vector
from tqdm import tqdm

from orggraph.config import OUTPUT_DIR
from orggraph.llm.client import LLMClient

PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://{os.environ.get('POSTGRES_USER','orggraph')}:"
    f"{os.environ.get('POSTGRES_PASSWORD','orggraph2026')}@localhost:"
    f"{os.environ.get('POSTGRES_PORT','5432')}/"
    f"{os.environ.get('POSTGRES_DB','orggraph')}",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_SPEECH_ACTS: frozenset[str] = frozenset({
    "request", "commit", "deliver", "propose",
    "amend", "meeting", "refuse", "accept",
})

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise an LLM JSON response.

    - Drops unknown speech_act values.
    - Coerces a single string speech_acts to a one-element list.
    - Defaults absent booleans to False.
    - Defaults absent sentiment to 0.0 and clamps to [-1.0, 1.0].
    - Defaults absent list fields to [].

    Returns a dict with all required keys.
    """
    # Normalise speech_acts (case-insensitive — LLMs vary on capitalisation)
    sa_raw = raw.get("speech_acts", [])
    if isinstance(sa_raw, str):
        sa_raw = [sa_raw]
    elif not isinstance(sa_raw, list):
        sa_raw = []
    speech_acts = [
        v.lower() for v in sa_raw
        if isinstance(v, str) and v.lower() in ALLOWED_SPEECH_ACTS
    ]

    # Booleans — default False if absent or not a bool
    def _bool(key: str) -> bool:
        val = raw.get(key, False)
        return bool(val)

    # Sentiment — clamp to [-1.0, 1.0]
    try:
        sentiment = float(raw.get("sentiment", 0.0))
    except (TypeError, ValueError):
        sentiment = 0.0
    sentiment = max(-1.0, min(1.0, sentiment))

    # Lists — default to []
    def _list(key: str) -> list:
        val = raw.get(key, [])
        if not isinstance(val, list):
            return []
        return val

    return {
        "speech_acts": speech_acts,
        "action_required": _bool("action_required"),
        "commitment_made": _bool("commitment_made"),
        "decision_carrying": _bool("decision_carrying"),
        "mentions_money": _bool("mentions_money"),
        "mentions_regulator": _bool("mentions_regulator"),
        "sentiment": sentiment,
        "topics": _list("topics"),
        "entities_mentioned": _list("entities_mentioned"),
    }


def _format_cluster_context(ctx: dict | None) -> str:
    """Render a [CONTEXT] block from a ClusterContext-like dict.

    Empty / None / missing fields → empty string (no context block emitted).
    The context block is purely for LLM grounding; we explicitly tell the LLM
    NOT to analyse the context, only the email under analysis.
    """
    if not ctx:
        return ""
    parts: list[str] = []
    if ctx.get("project_name"):
        line = f"  Project: \"{ctx['project_name']}\""
        if ctx.get("project_desc"):
            line += f" — {ctx['project_desc']}"
        parts.append(line)
    elif ctx.get("project_id"):
        parts.append(f"  Project: {ctx['project_id']} (unnamed)")
    if ctx.get("topic_name"):
        line = f"  Topic: \"{ctx['topic_name']}\""
        if ctx.get("topic_desc"):
            line += f" — {ctx['topic_desc']}"
        parts.append(line)
    elif ctx.get("topic_id"):
        parts.append(f"  Topic: {ctx['topic_id']} (unnamed)")

    rep_emails = ctx.get("representative_emails") or []
    if rep_emails:
        parts.append("  Representative emails from the same topic (for grounding only):")
        for i, rep in enumerate(rep_emails[:3], 1):
            subj = (rep.get("subject") or "").strip()[:120]
            body = (rep.get("body") or "").strip().replace("\n", " ")[:300]
            parts.append(f"    {i}. \"{subj}\" — {body}")

    if not parts:
        return ""
    return (
        "\n[CONTEXT — for grounding only; analyse only the email AFTER /CONTEXT]\n"
        + "\n".join(parts)
        + "\n[/CONTEXT]\n"
    )


def build_prompt(subject: str, body: str, cluster_context: dict | None = None) -> str:
    """Construct the LLM prompt for per-email signal extraction.

    Optional `cluster_context` injects the email's project/topic name +
    description + 1-3 representative emails to ground the analysis. Falls
    back to the bare prompt when context is None or empty.
    """
    acts_list = ", ".join(sorted(ALLOWED_SPEECH_ACTS))
    context_block = _format_cluster_context(cluster_context)
    return f"""You are an expert analyst of corporate email communication. Analyse the email below and return a single JSON object with exactly these keys:

- "speech_acts": list of speech-act labels from this set only: [{acts_list}]. Use as many as apply.
- "action_required": boolean — does this email explicitly request the recipient to take an action?
- "commitment_made": boolean — does the sender make a commitment or promise?
- "decision_carrying": boolean — does this email announce or carry a consequential decision?
- "mentions_money": boolean — does the email mention dollar amounts, prices, costs, or financial figures?
- "mentions_regulator": boolean — does the email mention a regulatory body (FERC, SEC, CFTC, ERCOT, etc.)?
- "sentiment": float between -1.0 (very negative) and 1.0 (very positive).
- "topics": list of short topic strings (max 5).
- "entities_mentioned": list of person or organisation names mentioned (max 10).

Return ONLY valid JSON. No markdown, no explanation.
{context_block}
EMAIL UNDER ANALYSIS:
SUBJECT: {subject}

BODY:
{body}
"""


def upsert_signals(cur, email_id: str, payload: dict[str, Any]) -> None:
    """Write signal columns via UPDATE on an existing embeddings_email row.

    Uses UPDATE rather than INSERT because the `embedding` column is NOT NULL —
    only rows already inserted by Stage 1 can receive signal data.
    """
    cur.execute(
        """
        UPDATE embeddings_email SET
            speech_acts         = %s,
            action_required     = %s,
            commitment_made     = %s,
            decision_carrying   = %s,
            mentions_money      = %s,
            mentions_regulator  = %s,
            sentiment           = %s,
            topics              = %s,
            entities_mentioned  = %s,
            signals_extracted_at = NOW()
        WHERE email_id = %s
        """,
        (
            pj.Jsonb(payload["speech_acts"]),
            payload["action_required"],
            payload["commitment_made"],
            payload["decision_carrying"],
            payload["mentions_money"],
            payload["mentions_regulator"],
            payload["sentiment"],
            pj.Jsonb(payload["topics"]),
            pj.Jsonb(payload["entities_mentioned"]),
            email_id,
        ),
    )


def load_existing_email_ids(conn) -> set[str]:
    """Return the set of email_id values where signals_extracted_at IS NOT NULL."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT email_id FROM embeddings_email WHERE signals_extracted_at IS NOT NULL"
        )
        return {r[0] for r in cur.fetchall()}


def load_cluster_context_map(
    df: pd.DataFrame,
    *,
    labels_path: Path | None = None,
    names_path: Path | None = None,
    n_reps: int = 3,
) -> dict[str, dict]:
    """Build email_id → cluster_context dict from Stage 2b artefacts.

    Returns an empty dict if the artefacts don't exist yet (graceful degradation).
    `df` must have columns: email_id, subject, body_truncated.
    """
    labels_path = labels_path or (OUTPUT_DIR / "cluster_labels.parquet")
    names_path = names_path or (OUTPUT_DIR / "cluster_names.jsonl")
    if not labels_path.exists():
        return {}

    labels = pd.read_parquet(labels_path)
    name_records: dict[str, dict] = {}
    if names_path.exists():
        with names_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    name_records[rec["cluster_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue

    # Build email_id → (subject, body) lookup from df for fast rep hydration
    email_lookup = {
        row["email_id"]: {
            "subject": row.get("subject") or "",
            "body": (row.get("body_truncated") or "")[:400],
        }
        for _, row in df.iterrows()
    }

    out: dict[str, dict] = {}
    for _, row in labels.iterrows():
        eid = row["email_id"]
        proj_label = int(row["project_label"])
        topic_label = int(row["topic_label"])
        proj_id = f"P{proj_label:03d}" if proj_label >= 0 else None
        topic_id = f"T{topic_label:03d}" if topic_label >= 0 else None

        ctx: dict = {}
        if proj_id:
            ctx["project_id"] = proj_id
            if proj_id in name_records:
                ctx["project_name"] = name_records[proj_id].get("name")
                ctx["project_desc"] = name_records[proj_id].get("description")
        if topic_id:
            ctx["topic_id"] = topic_id
            if topic_id in name_records:
                rec = name_records[topic_id]
                ctx["topic_name"] = rec.get("name")
                ctx["topic_desc"] = rec.get("description")
                # Hydrate up to n_reps representative emails (skip self).
                rep_ids = [
                    r for r in (rec.get("representative_email_ids") or [])
                    if r != eid
                ][:n_reps]
                ctx["representative_emails"] = [
                    email_lookup[r] for r in rep_ids if r in email_lookup
                ]
        if ctx:
            out[eid] = ctx
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    workers: int = 4,
    batch_size: int = 16,
    limit: int | None = None,
    base_urls: list[str] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    if base_urls is None:
        env_urls = os.environ.get(
            "OPENAI_BASE_URLS",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        )
        base_urls = [u.strip() for u in env_urls.split(",") if u.strip()]
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    model = os.environ.get("INFERENCE_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit")

    print(f"[backend] {len(base_urls)} endpoint(s):")
    for u in base_urls:
        print(f"          - {u}")
    print(f"[model]   {model}")

    parquet = output_dir / "clean_emails.parquet"
    if not parquet.exists():
        raise SystemExit(
            f"Missing {parquet}. Run `python -m orggraph.pipeline.corpus.filter` first."
        )

    print(f"[1/3] Loading {parquet}...")
    df = pd.read_parquet(parquet)
    if limit:
        df = df.head(limit)
    print(f"      {len(df):,} rows")

    print(f"[2/3] Connecting to pgvector at {PG_DSN.split('@')[-1]}...")
    pg = psycopg.connect(PG_DSN, autocommit=False)
    register_vector(pg)

    done = load_existing_email_ids(pg)
    print(f"      {len(done):,} already done; skipping")

    with pg.cursor() as cur:
        cur.execute("SELECT email_id FROM embeddings_email")
        embedded_ids = {r[0] for r in cur.fetchall()}

    pending = df[
        df["email_id"].isin(embedded_ids) & ~df["email_id"].isin(done)
    ].reset_index(drop=True)
    n_not_embedded = int((~df["email_id"].isin(embedded_ids)).sum())
    print(f"      {len(pending):,} pending; "
          f"{n_not_embedded:,} not yet embedded (re-run after Stage 1 progresses)")

    if pending.empty:
        print("Nothing to do.")
        pg.close()
        return

    # Optional: load Stage 2b cluster context for richer prompts.
    print("[2.5/3] Loading cluster context (project/topic + representative emails)...")
    ctx_map = load_cluster_context_map(df)
    if ctx_map:
        n_with_ctx = sum(1 for eid in pending["email_id"] if eid in ctx_map)
        print(f"      {n_with_ctx:,} of {len(pending):,} pending emails have cluster context")
    else:
        print("      no Stage 2b artefacts yet; running without cluster context")

    print(f"[3/3] Extracting signals across {len(base_urls)} endpoint(s)  workers={workers}...")
    clients = [LLMClient(base_url=u, api_key=api_key) for u in base_urls]
    # Thread-safe round-robin: itertools.count is atomic under the GIL
    rr_counter = itertools.count()

    def pick_client() -> LLMClient:
        return clients[next(rr_counter) % len(clients)]

    lock = threading.Lock()
    progress = tqdm(total=len(df), initial=len(done), desc="signals", unit="email")
    n_written = 0
    n_failed = 0

    def _llm_with_retry(prompt: str) -> dict | None:
        """Up to 3 attempts with exponential backoff on transient errors.

        Each retry picks a fresh client (round-robin), so a transient failure
        on one endpoint automatically routes the retry to another.
        """
        delays = [1.0, 4.0, 16.0]
        last_err = None
        for attempt, delay in enumerate(delays, 1):
            client = pick_client()
            try:
                return client.json_chat(model=model, prompt=prompt)
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                transient = any(t in msg for t in (
                    "timeout", "timed out", "connection", "502", "503", "504",
                    "reset", "broken pipe",
                ))
                if not transient or attempt == len(delays):
                    break
                time.sleep(delay)
        print(f"\n[error] LLM failed after retries: {last_err}")
        return None

    def process_one(row: dict) -> None:
        nonlocal n_written, n_failed
        eid = row["email_id"]
        subject = (row.get("subject") or "").strip()
        # Hard cap on body to keep prompt within sane token budget regardless
        # of what Stage 0 produced.
        body = (row.get("body_truncated") or "").strip()[:8000]
        ctx = ctx_map.get(eid)
        prompt = build_prompt(subject, body, cluster_context=ctx)
        raw = _llm_with_retry(prompt)

        if raw is None:
            with lock:
                n_failed += 1
            progress.update(1)
            return

        payload = validate_payload(raw)
        with lock:
            with pg.cursor() as cur:
                upsert_signals(cur, eid, payload)
            pg.commit()
            n_written += 1
        progress.update(1)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one, r.to_dict()) for _, r in pending.iterrows()]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(f"\n[error] future raised: {e}")

    progress.close()
    pg.close()

    print(f"\nDone. Wrote signals for {n_written:,} rows; {n_failed:,} LLM failures.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2a: per-email LLM signal extraction (speech acts, sentiment, etc.)"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--base-urls",
        type=str,
        default=None,
        help="Comma-separated list of OpenAI-compatible endpoints to round-robin "
             "across. Overrides OPENAI_BASE_URLS / OPENAI_BASE_URL env vars. "
             "Example: http://localhost:8000/v1,http://localhost:8001/v1",
    )
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (unused; kept for CLI consistency with other stages).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows from clean_emails.parquet.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Directory holding clean_emails.parquet (default: {OUTPUT_DIR})")
    args = parser.parse_args(argv)
    base_urls = (
        [u.strip() for u in args.base_urls.split(",") if u.strip()]
        if args.base_urls else None
    )
    run(
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        base_urls=base_urls,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
