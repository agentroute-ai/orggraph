"""Stage 1.5 — Deterministic per-email metadata.

Reads clean_emails.parquet, computes regex/lexicon/metadata features per email,
upserts into pgvector.embeddings_email (only the metadata columns).

Resumable: skips email_ids where metadata_extracted_at IS NOT NULL.

Concurrency-safe: runs alongside Stage 1 (embed_emails.py) which writes
disjoint columns (embedding, body_truncated).  Only rows already inserted by
Stage 1 are processed here; rows not yet embedded are silently skipped and will
be picked up on the next run.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg
import psycopg.types.json as pj
from pgvector.psycopg import register_vector
from tqdm import tqdm

from orggraph.config import OUTPUT_DIR

PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://{os.environ.get('POSTGRES_USER','orggraph')}:"
    f"{os.environ.get('POSTGRES_PASSWORD','orggraph2026')}@localhost:"
    f"{os.environ.get('POSTGRES_PORT','5432')}/"
    f"{os.environ.get('POSTGRES_DB','orggraph')}",
)

# ---------------------------------------------------------------------------
# Lexicons — DNM 2013-inspired minimal marker sets, NOT a faithful reproduction.
# Politeness uses 9 high-frequency markers as a coarse signal.
# Cite this as a methodological limitation (Ch.5); a richer politeness implementation
# (full Danescu-Niculescu-Mizil 2013 strategy set) is reserved for future work.
# ---------------------------------------------------------------------------

POLITENESS_TOKENS = {
    "please", "thank you", "thanks", "appreciate", "kindly",
    "would you mind", "if you could", "i was wondering", "sorry to bother",
}

HEDGE_TOKENS = {
    "maybe", "perhaps", "possibly", "might", "could be", "seems",
    "i think", "i guess", "kind of", "sort of", "probably", "likely",
}

MODAL_VERBS = ["will", "would", "should", "must", "could", "may", "might", "can", "shall"]

IMPERATIVE_VERBS = {
    "send", "review", "check", "submit", "complete", "approve", "sign",
    "call", "respond", "reply", "forward", "attach", "include", "remove",
    "update", "fix", "schedule", "confirm", "verify", "prepare", "make",
    "do", "go", "stop", "start", "see", "let", "give", "tell", "ask",
}

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"\b\w+\b")
QUESTION_RE = re.compile(r"\?")
EXCLAIM_RE = re.compile(r"!")
FIRST_PERSON_RE = re.compile(r"\b(I|me|my|mine|we|us|our|ours)\b")
SECOND_PERSON_RE = re.compile(r"\b(you|your|yours)\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[.!?]\s+|\n+")


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _count_modals(text: str) -> dict[str, int]:
    text_lower = text.lower()
    return {
        m: len(re.findall(rf"\b{m}\b", text_lower))
        for m in MODAL_VERBS
    }


def _count_imperatives(text: str) -> int:
    """Count sentence-opening tokens that are bare-infinitive imperatives."""
    sentences = SENTENCE_RE.split(text)
    n = 0
    for s in sentences:
        # Strip leading bullets / quotes / whitespace so "- Send", "* Send",
        # "> Send" all still surface "Send" as the first token.
        s = re.sub(r"^[^\w]+", "", s.strip())
        if not s:
            continue
        first_word = s.split(maxsplit=1)[0].lower().rstrip(",;:")
        if first_word in IMPERATIVE_VERBS:
            n += 1
    return n


def _politeness_score(text_lower: str) -> float:
    """0..1 score: number of distinct politeness tokens present, capped at 5."""
    hits = sum(1 for t in POLITENESS_TOKENS if t in text_lower)
    return min(hits / 5.0, 1.0)


def _hedge_count(text_lower: str) -> int:
    return sum(1 for t in HEDGE_TOKENS if t in text_lower)


# ---------------------------------------------------------------------------
# Public API (used directly by tests and by main())
# ---------------------------------------------------------------------------

def compute_metadata_for_row(
    *,
    body: str,
    sender_email: str | None,
    recipients_emails: list[str] | None,
    recipients_resolved: list[str] | None,
    date: datetime | None,
) -> dict:
    """Return a flat dict of deterministic metadata for a single email."""
    body = body or ""
    body_lower = body.lower()
    recipients_emails = list(recipients_emails or [])

    n_modals = _count_modals(body)

    if date is not None:
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        is_weekend = date.weekday() >= 5
        hour = date.hour
        is_off_hours = hour < 8 or hour >= 19
    else:
        is_weekend = False
        is_off_hours = False

    return {
        "body_word_count": len(WORD_RE.findall(body)),
        "n_questions": len(QUESTION_RE.findall(body)),
        "n_exclamations": len(EXCLAIM_RE.findall(body)),
        "n_imperatives": _count_imperatives(body),
        "n_modals": n_modals,
        "n_first_person": len(FIRST_PERSON_RE.findall(body)),
        "n_second_person": len(SECOND_PERSON_RE.findall(body)),
        # corbt parquet collapses cc/bcc/to into recipients_emails; treat as `to`.
        "to_count": len(recipients_emails),
        "cc_count": 0,
        "bcc_count": 0,
        "unique_recipients": len(set(recipients_emails)),
        "is_off_hours": is_off_hours,
        "is_weekend": is_weekend,
        "politeness_score": _politeness_score(body_lower),
        "hedge_count": _hedge_count(body_lower),
    }


def compute_thread_features(df: pd.DataFrame) -> dict[str, dict]:
    """Return per-email thread metadata keyed by email_id.

    Input df must have columns: email_id, thread_id, date.
    Returns a dict mapping email_id -> {is_thread_initiator, is_thread_closer,
    thread_position, reply_latency_hours}.
    """
    if df.empty:
        return {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.sort_values(["thread_id", "date", "email_id"])
    out: dict[str, dict] = {}
    for _tid, grp in df.groupby("thread_id", sort=False):
        rows = grp.to_dict("records")
        for i, row in enumerate(rows):
            prev_date = rows[i - 1]["date"] if i > 0 else None
            latency = None
            if prev_date is not None and pd.notna(prev_date) and pd.notna(row["date"]):
                latency = float((row["date"] - prev_date).total_seconds() / 3600.0)
            out[row["email_id"]] = {
                "is_thread_initiator": i == 0,
                "is_thread_closer": i == len(rows) - 1,
                "thread_position": i,
                "reply_latency_hours": latency,
            }
    return out


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def already_done(cur) -> set[str]:
    """Return set of email_ids where metadata has already been extracted."""
    cur.execute(
        "SELECT email_id FROM embeddings_email WHERE metadata_extracted_at IS NOT NULL"
    )
    return {r[0] for r in cur.fetchall()}


def update_metadata(cur, email_id: str, m: dict, t: dict) -> None:
    """UPDATE the metadata columns on an existing embeddings_email row.

    Uses UPDATE rather than INSERT ... ON CONFLICT because the `embedding`
    column is NOT NULL — we can only write metadata to rows that Stage 1
    has already inserted.  Rows not yet embedded are silently skipped.
    """
    cur.execute(
        """
        UPDATE embeddings_email SET
            body_word_count        = %s,
            n_questions            = %s,
            n_exclamations         = %s,
            n_imperatives          = %s,
            n_modals               = %s,
            n_first_person         = %s,
            n_second_person        = %s,
            to_count               = %s,
            cc_count               = %s,
            bcc_count              = %s,
            unique_recipients      = %s,
            is_thread_initiator    = %s,
            is_thread_closer       = %s,
            thread_position        = %s,
            reply_latency_hours    = %s,
            is_off_hours           = %s,
            is_weekend             = %s,
            politeness_score       = %s,
            hedge_count            = %s,
            metadata_extracted_at  = NOW()
        WHERE email_id = %s
        """,
        (
            m["body_word_count"],
            m["n_questions"],
            m["n_exclamations"],
            m["n_imperatives"],
            pj.Jsonb(m["n_modals"]),
            m["n_first_person"],
            m["n_second_person"],
            m["to_count"],
            m["cc_count"],
            m["bcc_count"],
            m["unique_recipients"],
            t["is_thread_initiator"],
            t["is_thread_closer"],
            t["thread_position"],
            t["reply_latency_hours"],
            m["is_off_hours"],
            m["is_weekend"],
            m["politeness_score"],
            m["hedge_count"],
            email_id,
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(limit: int | None = None, output_dir: Path = OUTPUT_DIR) -> None:
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

    print("[2/3] Computing thread features...")
    thread_feats = compute_thread_features(df[["email_id", "thread_id", "date"]])

    print("[3/3] Upserting metadata...")
    pg = psycopg.connect(PG_DSN, autocommit=False)
    register_vector(pg)
    cur = pg.cursor()
    done = already_done(cur)
    print(f"      {len(done):,} already done; skipping")

    # Only process rows that Stage 1 has already embedded.
    cur.execute("SELECT email_id FROM embeddings_email")
    embedded_ids = {r[0] for r in cur.fetchall()}

    n_written = 0
    n_skipped_not_embedded = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="metadata"):
        eid = row["email_id"]
        if eid in done:
            continue
        if eid not in embedded_ids:
            n_skipped_not_embedded += 1
            continue
        re_raw = row.get("recipients_emails")
        rr_raw = row.get("recipients_resolved")
        recipients_emails = list(re_raw) if re_raw is not None else []
        recipients_resolved = list(rr_raw) if rr_raw is not None else []
        m = compute_metadata_for_row(
            body=row.get("body_truncated"),
            sender_email=row.get("sender_email"),
            recipients_emails=recipients_emails,
            recipients_resolved=recipients_resolved,
            date=row.get("date"),
        )
        t = thread_feats.get(
            eid,
            {
                "is_thread_initiator": True,
                "is_thread_closer": True,
                "thread_position": 0,
                "reply_latency_hours": None,
            },
        )
        update_metadata(cur, eid, m, t)
        n_written += 1
        if n_written % 2000 == 0:
            pg.commit()

    pg.commit()
    pg.close()

    if n_skipped_not_embedded:
        print(
            f"\nNote: {n_skipped_not_embedded:,} rows skipped (not yet embedded by Stage 1)."
            " Re-run after Stage 1 completes to process them."
        )
    print(f"\nDone. Wrote metadata for {n_written:,} rows.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1.5: compute deterministic per-email metadata features."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows from clean_emails.parquet (for smoke-testing).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Directory holding clean_emails.parquet (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)
    run(limit=args.limit, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
