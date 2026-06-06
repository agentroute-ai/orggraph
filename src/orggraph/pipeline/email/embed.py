"""Stage 1 — Embed every filtered email into pgvector.

Reads clean_emails.parquet, embeds (subject + body_truncated) via local
Ollama (embeddinggemma 768-dim), upserts into pgvector.embeddings_email.
Resumable: skips email_ids already in the table.
"""

from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector
from tqdm import tqdm

from orggraph.config import OUTPUT_DIR

EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "ollama")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embeddinggemma")

PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://{os.environ.get('POSTGRES_USER','orggraph')}:"
    f"{os.environ.get('POSTGRES_PASSWORD','orggraph2026')}@localhost:"
    f"{os.environ.get('POSTGRES_PORT','5432')}/"
    f"{os.environ.get('POSTGRES_DB','orggraph')}",
)


def build_document(row: dict) -> str:
    subj = (row.get("subject") or "").strip()
    body = (row.get("body_truncated") or "").strip()
    return f"{subj}\n\n{body}".strip()


def existing_email_ids(cur) -> set[str]:
    cur.execute("SELECT email_id FROM embeddings_email")
    return {r[0] for r in cur.fetchall()}


def _to_list(v) -> list:
    if v is None:
        return []
    return list(v)


def upsert_one(cur, row: dict, embedding: list[float]) -> None:
    cur.execute(
        """
        INSERT INTO embeddings_email (
            email_id, thread_id, sender_email, sender_resolved,
            recipients_emails, recipients_resolved, date,
            subject, body_chars, body_truncated, embedding
        )
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
        ON CONFLICT (email_id) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                body_truncated = EXCLUDED.body_truncated,
                sender_resolved = EXCLUDED.sender_resolved,
                recipients_resolved = EXCLUDED.recipients_resolved
        """,
        (
            row["email_id"], row["thread_id"], row.get("sender_email"),
            row.get("sender_resolved"),
            psycopg.types.json.Jsonb(_to_list(row.get("recipients_emails"))),
            psycopg.types.json.Jsonb(_to_list(row.get("recipients_resolved"))),
            row.get("date"),
            (row.get("subject") or "")[:500],
            int(row.get("body_chars") or 0),
            row.get("body_truncated") or "",
            embedding,
        ),
    )


def embed_batch(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def run(workers: int, batch_size: int, limit: int | None, output_dir: Path = OUTPUT_DIR) -> None:
    parquet = output_dir / "clean_emails.parquet"
    if not parquet.exists():
        raise SystemExit(
            f"Missing {parquet}. Run `python -m orggraph.pipeline.corpus.filter` first."
        )

    print(f"[1/3] Loading {parquet}...")
    df = pd.read_parquet(parquet)
    if limit:
        df = df.head(limit)
    print(f"      {len(df):,} rows to embed")

    print(f"[2/3] Connecting to pgvector at {PG_DSN.split('@')[-1]}...")
    pg = psycopg.connect(PG_DSN, autocommit=False)
    register_vector(pg)
    cur = pg.cursor()
    existing = existing_email_ids(cur)
    print(f"      {len(existing):,} already embedded; skipping")

    pending = df[~df["email_id"].isin(existing)].reset_index(drop=True)
    print(f"      {len(pending):,} pending")
    if pending.empty:
        print("Nothing to do.")
        pg.close()
        return

    print(f"[3/3] Embedding via {EMBED_BASE_URL} model={EMBED_MODEL}...")
    client = OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)

    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc="embed", unit="email")

    def process_batch(start: int, stop: int) -> None:
        chunk = pending.iloc[start:stop]
        texts = [build_document(r.to_dict()) for _, r in chunk.iterrows()]
        try:
            vectors = embed_batch(client, EMBED_MODEL, texts)
        except Exception as e:  # noqa: BLE001
            print(f"\n[error] batch {start}-{stop}: {e}")
            progress.update(len(chunk))
            return
        with lock:
            for (_, row), vec in zip(chunk.iterrows(), vectors):
                upsert_one(cur, row.to_dict(), vec)
            pg.commit()
        progress.update(len(chunk))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        starts = list(range(0, len(pending), batch_size))
        futs = [ex.submit(process_batch, s, min(s + batch_size, len(pending))) for s in starts]
        for f in as_completed(futs):
            f.result()
    progress.close()

    cur.execute("SELECT count(*) FROM embeddings_email")
    n = cur.fetchone()[0]
    pg.close()
    print(f"\nDone. embeddings_email row count: {n:,}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Directory holding clean_emails.parquet (default: {OUTPUT_DIR})")
    args = parser.parse_args(argv)
    run(
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
