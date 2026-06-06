"""Stage A.2 — Per-ExternalEntity LLM enrichment.

Reads ExternalEntity nodes from Neo4j, samples top-K emails for each domain,
calls the LLM, appends to datasets/enron/processed/entity_enrichment.csv.
Neo4j is not touched here.
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
from orggraph.data.loader import load_emails
from orggraph.llm.client import LLMClient
from orggraph.llm.prompts import ENTITY_PROMPT, render_emails
from orggraph.llm.sampling import sample_emails_for_entity

OUT_CSV = OUTPUT_DIR / "entity_enrichment.csv"

CSV_COLUMNS = [
    "domain", "name", "description", "business_relationship",
    "industry", "engagement_pattern", "tone", "topics_json",
    "prompt_chars", "response_chars", "duration_s", "model", "timestamp",
]


def load_existing_domains(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(df.loc[df["description"].notna(), "domain"].astype(str))
    except Exception:
        return set()


def fetch_entity_targets(driver) -> list[dict]:
    q = """
    MATCH (e:ExternalEntity)
    RETURN e.domain AS domain, e.name AS name, e.total_emails AS total_emails
    ORDER BY e.total_emails DESC
    """
    with driver.session() as s:
        return [dict(r) for r in s.run(q)]


def build_entity_payload(
    domain: str, name: str, llm_response: dict,
    prompt_chars: int, response_chars: int, duration_s: float, model: str,
) -> dict:
    return {
        "domain": domain,
        "name": name,
        "description": (llm_response.get("description") or "")[:500],
        "business_relationship": (llm_response.get("business_relationship") or "Other")[:50],
        "industry": (llm_response.get("industry") or "")[:200],
        "engagement_pattern": (llm_response.get("engagement_pattern") or "transactional")[:50],
        "tone": (llm_response.get("tone") or "transactional")[:50],
        "topics_json": json.dumps(llm_response.get("topics") or [], ensure_ascii=False),
        "prompt_chars": prompt_chars,
        "response_chars": response_chars,
        "duration_s": round(duration_s, 1),
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run(model: str, workers: int, limit: int | None, k_emails: int) -> None:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_pw = os.environ.get("NEO4J_PASSWORD", "orggraph2026")

    print(f"[backend] {base_url} model={model}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_domains(OUT_CSV)
    print(f"[resume] {len(existing)} entities already in CSV")

    print("[1/3] Loading email corpus...")
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
    emails["body_len"] = emails["body"].astype(str).str.len()

    print("[2/3] Fetching targets...")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
    try:
        targets = fetch_entity_targets(driver)
    finally:
        driver.close()
    pending = [t for t in targets if t["domain"] not in existing]
    if limit:
        pending = pending[:limit]
    print(f"[2/3] {len(pending)} pending out of {len(targets)}")

    if not pending:
        print("Nothing to do.")
        return

    client = LLMClient(base_url=base_url, api_key=api_key)
    csv_lock = threading.Lock()
    write_header = not OUT_CSV.exists()

    def score_one(t: dict) -> dict:
        samples = sample_emails_for_entity(emails, t["domain"], k=k_emails)
        prompt_chars = sum(len(e["body"]) for e in samples)
        if not samples:
            return build_entity_payload(t["domain"], t["name"], {}, 0, 0, 0.0, model)
        prompt = ENTITY_PROMPT.format(name=t["name"], domain=t["domain"], emails=render_emails(samples))
        ts = time.monotonic()
        result = client.json_chat(model=model, prompt=prompt, max_tokens=4096)
        dur = time.monotonic() - ts
        response_chars = len(json.dumps(result)) if result else 0
        return build_entity_payload(t["domain"], t["name"], result or {},
                                    prompt_chars, response_chars, dur, model)

    progress = tqdm(total=len(pending), desc="enrich-entities", unit="entity")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_one, t): t for t in pending}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                t = futs[fut]
                row = build_entity_payload(t["domain"], t["name"], {}, 0, 0, 0.0, model)
                row["description"] = f"FUTURE_ERROR: {e!r}"[:500]
            with csv_lock:
                pd.DataFrame([row])[CSV_COLUMNS].to_csv(
                    OUT_CSV, mode="a", header=write_header, index=False
                )
                write_header = False
            progress.update(1)
            progress.set_postfix(name=row["name"][:18])
    progress.close()
    print(f"[3/3] Done. CSV: {OUT_CSV}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 4b: external-entity LLM enrichment")
    parser.add_argument(
        "--model", default=os.environ.get("INFERENCE_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k-emails", type=int, default=10)
    args = parser.parse_args(argv)
    run(model=args.model, workers=args.workers, limit=args.limit, k_emails=args.k_emails)


if __name__ == "__main__":
    main()
