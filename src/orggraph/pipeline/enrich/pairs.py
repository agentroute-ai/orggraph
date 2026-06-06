"""Stage A.3 — Per-pair LLM deference scoring (Tier 2).

For each Person-Person pair with >= MIN_EXCHANGES email exchanges
(in either direction), the LLM scores the deference direction.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

from orggraph.config import OUTPUT_DIR
from orggraph.data.identity import build_alias_map, resolve_sender
from orggraph.data.loader import load_emails
from orggraph.llm.client import LLMClient
from orggraph.llm.prompts import PAIR_PROMPT, render_emails
from orggraph.llm.sampling import sample_emails_for_pair

OUT_CSV = OUTPUT_DIR / "pair_enrichment.csv"
MIN_EXCHANGES = 5

CSV_COLUMNS = [
    "person_a", "person_b", "deference_score", "deference_reasoning",
    "relationship_type", "shared_topics_json", "n_exchanges_sampled",
    "prompt_chars", "response_chars", "duration_s", "model", "timestamp",
]


def ordered(a: str, b: str) -> tuple[str, str]:
    """Sort the pair so resumes are deterministic."""
    return (a, b) if a <= b else (b, a)


def load_existing_pairs(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(zip(df["person_a"].astype(str), df["person_b"].astype(str)))
    except Exception:
        return set()


def fetch_eligible_pairs(driver, min_exchanges: int) -> list[tuple[str, str]]:
    """Return ordered pairs (A, B) with combined COMMUNICATES_WITH weight >= min."""
    q = """
    MATCH (a:Person)-[r:COMMUNICATES_WITH]-(b:Person)
    WHERE a.name < b.name
    WITH a, b, sum(r.weight) AS total
    WHERE total >= $min
    RETURN a.name AS person_a, b.name AS person_b, total
    ORDER BY total DESC
    """
    with driver.session() as s:
        return [(r["person_a"], r["person_b"]) for r in s.run(q, min=min_exchanges)]


def _resolve_recipients(val, alias_map) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        addrs = [a.strip() for a in val.split(",") if a.strip()]
    else:
        try:
            addrs = [str(a).strip() for a in val if str(a).strip()]
        except TypeError:
            return []
    out = []
    for a in addrs:
        n = resolve_sender(a.lower(), alias_map)
        if n:
            out.append(n)
    return out


def prepare_email_index(alias_map) -> pd.DataFrame:
    emails = load_emails()
    sender_col = next(c for c in ["from", "From", "sender"] if c in emails.columns)
    recipients_col = next(c for c in ["to", "To", "recipients"] if c in emails.columns)
    body_col = next(c for c in ["body", "Body", "text", "message", "content"] if c in emails.columns)
    subject_col = next((c for c in ["subject", "Subject"] if c in emails.columns), "subject")
    date_col = next((c for c in ["date", "Date", "timestamp"] if c in emails.columns), "date")
    emails = emails.rename(columns={
        sender_col: "from", recipients_col: "to",
        subject_col: "subject", body_col: "body", date_col: "date",
    })
    emails["sender_resolved"] = emails["from"].astype(str).str.lower().map(
        lambda s: resolve_sender(s, alias_map)
    )
    emails["recipients_resolved"] = emails["to"].apply(
        lambda r: _resolve_recipients(r, alias_map)
    )
    emails["body_len"] = emails["body"].astype(str).str.len()
    return emails


def build_pair_payload(
    person_a: str, person_b: str, llm_response: dict,
    n_exchanges: int, prompt_chars: int, response_chars: int,
    duration_s: float, model: str,
) -> dict:
    return {
        "person_a": person_a,
        "person_b": person_b,
        "deference_score": llm_response.get("deference_score"),
        "deference_reasoning": (llm_response.get("deference_reasoning") or "")[:500],
        "relationship_type": (llm_response.get("relationship_type") or "transactional")[:50],
        "shared_topics_json": json.dumps(
            llm_response.get("shared_topics") or [], ensure_ascii=False
        ),
        "n_exchanges_sampled": n_exchanges,
        "prompt_chars": prompt_chars,
        "response_chars": response_chars,
        "duration_s": round(duration_s, 1),
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run(model: str, workers: int, limit: int | None, min_exchanges: int) -> None:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_pw = os.environ.get("NEO4J_PASSWORD", "orggraph2026")

    print(f"[backend] {base_url} model={model} min_exchanges={min_exchanges}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_pairs(OUT_CSV)
    print(f"[resume] {len(existing)} pairs already in CSV")

    print("[1/3] Loading email corpus and resolving identities...")
    alias_map = build_alias_map()
    emails = prepare_email_index(alias_map)

    print("[2/3] Fetching eligible pairs from Neo4j...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
    try:
        pairs = fetch_eligible_pairs(driver, min_exchanges)
    finally:
        driver.close()
    pending = [(a, b) for (a, b) in pairs if (a, b) not in existing]
    if limit:
        pending = pending[:limit]
    print(f"[2/3] {len(pending)} pending out of {len(pairs)} eligible")

    if not pending:
        print("Nothing to do.")
        return

    client = LLMClient(base_url=base_url, api_key=api_key)
    csv_lock = threading.Lock()
    write_header = not OUT_CSV.exists()

    def score_one(pair: tuple[str, str]) -> dict:
        a, b = pair
        samples = sample_emails_for_pair(emails, a, b, k=5, min_exchanges=min_exchanges)
        prompt_chars = sum(len(e["body"]) for e in samples)
        if not samples:
            return build_pair_payload(a, b, {}, 0, 0, 0, 0.0, model)
        prompt = PAIR_PROMPT.format(person_a=a, person_b=b, emails=render_emails(samples))
        ts = time.monotonic()
        result = client.json_chat(model=model, prompt=prompt, max_tokens=4096)
        dur = time.monotonic() - ts
        response_chars = len(json.dumps(result)) if result else 0
        return build_pair_payload(a, b, result or {}, len(samples),
                                   prompt_chars, response_chars, dur, model)

    progress = tqdm(total=len(pending), desc="enrich-pairs", unit="pair")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_one, p): p for p in pending}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                a, b = futs[fut]
                row = build_pair_payload(a, b, {}, 0, 0, 0, 0.0, model)
                row["deference_reasoning"] = f"FUTURE_ERROR: {e!r}"[:500]
            with csv_lock:
                pd.DataFrame([row])[CSV_COLUMNS].to_csv(
                    OUT_CSV, mode="a", header=write_header, index=False
                )
                write_header = False
            progress.update(1)
            progress.set_postfix(d=row["deference_score"])
    progress.close()
    print(f"[3/3] Done. CSV: {OUT_CSV}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 4c: pair-deference LLM enrichment")
    parser.add_argument(
        "--model", default=os.environ.get("INFERENCE_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-exchanges", type=int, default=MIN_EXCHANGES)
    args = parser.parse_args(argv)
    run(model=args.model, workers=args.workers, limit=args.limit,
        min_exchanges=args.min_exchanges)


if __name__ == "__main__":
    main()
