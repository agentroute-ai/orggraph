"""Stage 4a — Hybrid deterministic + LLM persona enrichment.

Reads Person nodes (with Stage 3 aggregates) from Neo4j, computes deterministic
signal dimensions, stratified-samples emails from Postgres, makes one LLM call
per Person for narrative dimensions, writes merged result to
``<output-dir>/person_enrichment_v2.csv``.

Resume-safe: skips Persons already present in the CSV.

Usage:
    source .env && python -m orggraph.pipeline.enrich.persons
    python -m orggraph.pipeline.enrich.persons --limit 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from orggraph.config import OUTPUT_DIR
from orggraph.llm.client import LLMClient

OUT_CSV = OUTPUT_DIR / "person_enrichment.csv"

PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://{os.environ.get('POSTGRES_USER', 'orggraph')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'orggraph2026')}@localhost:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'orggraph')}",
)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "orggraph2026")

CSV_COLUMNS = [
    "name",
    # deterministic layer
    "directiveness_signal",
    "agenda_setting_signal",
    "verbosity_signal",
    "expertise_topics",
    # LLM layer
    "formality",
    "authority_style",
    "communication_style",
    "role_summary",
    "expertise",
    "seniority_narrative",
    "confidence_self_report",
    # provenance
    "n_emails_sampled",
    "prompt_chars",
    "response_chars",
    "duration_s",
    "model",
    "timestamp",
]

AUTHORITY_STYLES = {"directive", "consultative", "delegating", "collaborative", "passive"}


SYSTEM_PROMPT = """You are an expert organizational analyst building persona profiles for Enron Corporation employees from email evidence.

For each employee, you receive a DETERMINISTIC SIGNAL PROFILE (computed from real email metadata — treat as ground truth, do not contradict) plus stratified samples of their actual emails (qualitative evidence). You must infer narrative persona dimensions that complement the deterministic numbers.

OUTPUT CONTRACT — strictly enforced:
- Return ONLY a single JSON object. No markdown fences. No prose preamble. No trailing commentary.
- The JSON object MUST have exactly these 7 keys, all populated, every time:
  {
    "formality": <integer 1-5; 1=very informal, 5=very formal>,
    "authority_style": <one of: "directive", "consultative", "delegating", "collaborative", "passive">,
    "communication_style": <1-2 sentence string describing HOW they communicate>,
    "role_summary": <1-2 sentence string describing WHAT they do at Enron>,
    "expertise": <list of 2-5 specific specialty strings inferred from email content>,
    "seniority_narrative": <1-2 sentence string describing apparent seniority>,
    "confidence_self_report": <integer 1-5; your confidence in this inference>
  }
- NEVER return {} or null fields. If evidence is thin, infer best-guess values from the deterministic signals and lower confidence_self_report — do NOT skip fields.
- authority_style MUST be one of the 5 enum values (lowercase, exact spelling).

CALIBRATION HINTS:
- directiveness_signal > 0.4 AND agenda_setting_signal > 0.3 → typically "directive"
- directiveness_signal < 0.2 AND agenda_setting_signal < 0.1 → typically "passive" or "collaborative"
- High politeness_baseline + high hedge_rate → typically "consultative"
- verbosity_signal: 0.5 ≈ corpus-average; > 0.7 = verbose; < 0.3 = terse
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_deterministic_layer(person_props: dict, corpus_mean_body_words: float) -> dict:
    """Compute deterministic signal dimensions from aggregated Person properties.

    Args:
        person_props: Neo4j Person node properties dict.
        corpus_mean_body_words: Corpus-wide average body word count from SQL.

    Returns:
        Dict with keys: directiveness_signal, agenda_setting_signal,
        verbosity_signal, expertise_topics.
    """
    directiveness_signal = float(person_props.get("pct_request_sent") or 0.0)
    agenda_setting_signal = float(person_props.get("pct_decision_carrying") or 0.0)

    mean_body_words = float(person_props.get("mean_body_words") or corpus_mean_body_words)
    if corpus_mean_body_words and corpus_mean_body_words > 0:
        z = (mean_body_words - corpus_mean_body_words) / corpus_mean_body_words
    else:
        z = 0.0
    verbosity_signal = 1.0 / (1.0 + math.exp(-z))  # sigmoid

    raw_topics = person_props.get("top3_topics") or []
    if isinstance(raw_topics, str):
        # Neo4j may return JSON string for list properties
        try:
            raw_topics = json.loads(raw_topics)
        except (json.JSONDecodeError, ValueError):
            raw_topics = [t.strip() for t in raw_topics.split(",") if t.strip()]
    expertise_topics = list(raw_topics)

    return {
        "directiveness_signal": directiveness_signal,
        "agenda_setting_signal": agenda_setting_signal,
        "verbosity_signal": verbosity_signal,
        "expertise_topics": expertise_topics,
    }


def stratified_sample_emails(
    emails: list[dict],
    *,
    top_topics: list[str],
    n_decision: int = 5,
    n_per_topic: int = 5,
    n_longest: int = 5,
) -> list[dict]:
    """Sample emails stratified across decision-carrying, topic diversity, and length.

    Picks up to n_decision decision-carrying emails (longest first),
    then up to n_per_topic emails per top_topic (longest unseen first),
    then tops up with globally-longest unseen emails.
    Deduplicates by email_id throughout.

    Args:
        emails: List of email dicts with keys email_id, decision_carrying,
                topics (list), body_word_count.
        top_topics: Topic labels to diversify across.
        n_decision: Max decision-carrying emails to pick.
        n_per_topic: Max emails per topic to pick.
        n_longest: Max globally-longest emails to top up with.

    Returns:
        Deduplicated list of sampled email dicts.
    """
    seen_ids: set[str] = set()
    result: list[dict] = []

    def _word_count(e: dict) -> int:
        return int(e.get("body_word_count") or 0)

    # Bucket 1: decision-carrying emails, longest first
    decision_emails = sorted(
        [e for e in emails if e.get("decision_carrying")],
        key=_word_count,
        reverse=True,
    )
    for e in decision_emails[:n_decision]:
        eid = e["email_id"]
        if eid not in seen_ids:
            seen_ids.add(eid)
            result.append(e)

    # Bucket 2: per-topic longest unseen
    for topic in top_topics:
        topic_emails = sorted(
            [e for e in emails if topic in (e.get("topics") or [])],
            key=_word_count,
            reverse=True,
        )
        added = 0
        for e in topic_emails:
            if added >= n_per_topic:
                break
            eid = e["email_id"]
            if eid not in seen_ids:
                seen_ids.add(eid)
                result.append(e)
                added += 1

    # Bucket 3: globally longest unseen top-up
    all_sorted = sorted(emails, key=_word_count, reverse=True)
    added = 0
    for e in all_sorted:
        if added >= n_longest:
            break
        eid = e["email_id"]
        if eid not in seen_ids:
            seen_ids.add(eid)
            result.append(e)
            added += 1

    return result


def build_user_prompt(person_props: dict, det: dict, sample_emails: list[dict]) -> str:
    """Build the per-person user message (data only — task framing is in SYSTEM_PROMPT)."""
    name = person_props.get("name", "Unknown")
    function = person_props.get("function") or "Unknown"
    n_sent = person_props.get("n_emails_sent") or 0
    politeness = person_props.get("politeness_baseline")
    hedge_rate = person_props.get("hedge_rate")

    politeness_str = f"{politeness:.3f}" if politeness is not None else "N/A"
    hedge_str = f"{hedge_rate:.2f}" if hedge_rate is not None else "N/A"

    topics_str = ", ".join(det["expertise_topics"]) if det["expertise_topics"] else "unknown"

    email_texts = []
    for i, e in enumerate(sample_emails, 1):
        subject = e.get("subject") or "(no subject)"
        body = e.get("body_truncated") or ""
        email_texts.append(f"[{i}] Subject: {subject}\n{body}")
    emails_block = "\n---\n".join(email_texts) if email_texts else "(no sample emails available)"

    return f"""EMPLOYEE: {name}
FUNCTION: {function}

DETERMINISTIC SIGNAL PROFILE (computed from {n_sent} emails sent):
  - directiveness_signal   : {det['directiveness_signal']:.2f}  (fraction of emails containing requests)
  - agenda_setting_signal  : {det['agenda_setting_signal']:.2f}  (fraction of decision-carrying emails)
  - verbosity_signal       : {det['verbosity_signal']:.3f}  (sigmoid-scaled relative to corpus mean)
  - expertise_topics       : {topics_str}
  - politeness_baseline    : {politeness_str}  (avg per-email politeness score)
  - hedge_rate             : {hedge_str}  (hedging markers per 100 words)

EMAIL SAMPLES ({len(sample_emails)}):
---
{emails_block}
---"""


def synthesize_heuristic(det: dict, person_props: dict) -> dict:
    """Generate plausible narrative fields purely from deterministic signals.

    Used as a final fallback when the LLM fails to produce content after retries,
    so no row ever has empty narrative columns. confidence_self_report is set to 1
    to flag these rows as heuristic, not LLM-grounded.
    """
    d = float(det.get("directiveness_signal") or 0.0)
    a = float(det.get("agenda_setting_signal") or 0.0)
    v = float(det.get("verbosity_signal") or 0.5)
    topics = list(det.get("expertise_topics") or [])
    function = person_props.get("function") or "Unknown"
    politeness = person_props.get("politeness_baseline")
    hedge_rate = person_props.get("hedge_rate")

    if d > 0.4 and a > 0.3:
        style = "directive"
    elif d < 0.2 and a < 0.1:
        style = "passive"
    elif d > 0.3:
        style = "consultative"
    elif a > 0.2:
        style = "collaborative"
    else:
        style = "collaborative"

    p = float(politeness) if politeness is not None else 0.5
    formality = 4 if p > 0.7 else 3 if p > 0.5 else 2

    if v > 0.7:
        verb_word = "verbose"
    elif v < 0.3:
        verb_word = "terse"
    else:
        verb_word = "moderately concise"

    topic_summary = ", ".join(topics[:3]) if topics else "general business operations"

    hedge_clause = (
        f" with hedge rate {hedge_rate:.2f}/100 words" if hedge_rate is not None else ""
    )

    return {
        "formality": formality,
        "authority_style": style,
        "communication_style": (
            f"Communicates in a {verb_word} style with a {style} authority pattern{hedge_clause}, "
            f"primarily addressing {topic_summary}."
        ),
        "role_summary": (
            f"Works in {function} at Enron, focused on {topic_summary}."
        ),
        "expertise": list(topics[:5]) if topics else ["Enron operations"],
        "seniority_narrative": (
            f"Signal-based estimate (directiveness={d:.2f}, agenda_setting={a:.2f}): "
            f"{'mid- to senior-level role with decision authority' if (d > 0.3 and a > 0.2) else 'support or operational role'}."
        ),
        "confidence_self_report": 1,
    }


def call_llm_with_retries(
    client,
    model: str,
    person_props: dict,
    det: dict,
    sample: list[dict],
) -> tuple[dict, int, int, float, int]:
    """Try LLM call up to 3 times with progressively different params.

    Returns (result_dict, last_prompt_chars, last_response_chars, total_duration_s, attempts).
    Empty result_dict means all attempts failed.
    """
    attempts_config = [
        {"max_tokens": 8192,  "temperature": 0.0, "seed": 42, "n_emails": None},
        {"max_tokens": 12288, "temperature": 0.3, "seed": 7,  "n_emails": None},
        {"max_tokens": 8192,  "temperature": 0.0, "seed": 42, "n_emails": 5},
    ]

    total_duration_s = 0.0
    last_prompt_chars = 0
    last_response_chars = 0

    for attempt_idx, cfg in enumerate(attempts_config, 1):
        attempt_sample = sample[: cfg["n_emails"]] if cfg["n_emails"] else sample
        prompt = build_user_prompt(person_props, det, attempt_sample)
        last_prompt_chars = len(prompt)

        ts = time.monotonic()
        try:
            result = client.json_chat(
                model=model,
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                seed=cfg["seed"],
            )
        except Exception:
            result = None
        total_duration_s += time.monotonic() - ts

        if result and isinstance(result, dict) and result.get("role_summary"):
            last_response_chars = len(json.dumps(result))
            return result, last_prompt_chars, last_response_chars, total_duration_s, attempt_idx

    return {}, last_prompt_chars, last_response_chars, total_duration_s, len(attempts_config)


def merge_layers(det: dict, llm: dict) -> dict:
    """Merge deterministic and LLM layers into a unified persona profile.

    Deterministic values always win for signal dimensions. LLM values are used
    for narrative dimensions. For expertise: LLM list is preferred if non-empty,
    otherwise falls back to det["expertise_topics"].

    Args:
        det: Output of compute_deterministic_layer.
        llm: Parsed LLM JSON response dict (may be empty / partially populated).

    Returns:
        Merged persona dict.
    """
    merged = dict(det)  # start with deterministic layer

    # LLM narrative keys — deterministic layer does NOT provide these
    merged["formality"] = llm.get("formality")
    merged["authority_style"] = llm.get("authority_style")
    merged["communication_style"] = llm.get("communication_style")
    merged["role_summary"] = llm.get("role_summary")
    merged["seniority_narrative"] = llm.get("seniority_narrative")
    merged["confidence_self_report"] = llm.get("confidence_self_report")

    # Expertise: LLM refines deterministic topics; fall back if absent/empty
    llm_expertise = llm.get("expertise")
    if llm_expertise:
        merged["expertise"] = list(llm_expertise)
    else:
        merged["expertise"] = list(det.get("expertise_topics") or [])

    return merged


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_existing_names(csv_path: Path) -> set[str]:
    """Return names already written to the output CSV."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(df["name"].dropna().astype(str))
    except Exception:
        return set()


def fetch_person_targets(driver) -> list[dict]:
    """Return Person nodes where pct_request_sent IS NOT NULL (Stage 3 has run)."""
    q = """
    MATCH (p:Person)
    WHERE p.pct_request_sent IS NOT NULL
    RETURN p {.*} AS props
    ORDER BY p.name
    """
    with driver.session() as s:
        return [dict(r["props"]) for r in s.run(q)]


def fetch_corpus_mean_body_words(cur) -> float:
    """Compute corpus-wide average body word count from Postgres."""
    cur.execute(
        "SELECT AVG(body_word_count) FROM embeddings_email WHERE body_word_count IS NOT NULL"
    )
    row = cur.fetchone()
    if row and row[0] is not None:
        return float(row[0])
    return 100.0  # safe fallback


def fetch_person_emails(cur, sender_resolved: str, limit: int = 200) -> list[dict]:
    """Fetch the most relevant emails for a given sender from Postgres."""
    cur.execute(
        """
        SELECT email_id, subject, body_truncated, body_word_count,
               topics, decision_carrying
        FROM embeddings_email
        WHERE sender_resolved = %s
        ORDER BY decision_carrying::int DESC, body_word_count DESC
        LIMIT %s
        """,
        (sender_resolved, limit),
    )
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        # topics may be stored as JSON string or list
        topics = d.get("topics")
        if isinstance(topics, str):
            try:
                topics = json.loads(topics)
            except (json.JSONDecodeError, ValueError):
                topics = [t.strip() for t in topics.split(",") if t.strip()]
        d["topics"] = topics or []
        result.append(d)
    return result


def _flatten_row(name: str, merged: dict, n_sampled: int, prompt_chars: int,
                 response_chars: int, duration_s: float, model: str) -> dict:
    """Flatten merged persona dict into a CSV row."""
    return {
        "name": name,
        "directiveness_signal": merged.get("directiveness_signal"),
        "agenda_setting_signal": merged.get("agenda_setting_signal"),
        "verbosity_signal": merged.get("verbosity_signal"),
        "expertise_topics": json.dumps(merged.get("expertise_topics") or [], ensure_ascii=False),
        "formality": merged.get("formality"),
        "authority_style": merged.get("authority_style"),
        "communication_style": (merged.get("communication_style") or "")[:500],
        "role_summary": (merged.get("role_summary") or "")[:500],
        "expertise": json.dumps(merged.get("expertise") or [], ensure_ascii=False),
        "seniority_narrative": (merged.get("seniority_narrative") or "")[:500],
        "confidence_self_report": merged.get("confidence_self_report"),
        "n_emails_sampled": n_sampled,
        "prompt_chars": prompt_chars,
        "response_chars": response_chars,
        "duration_s": round(duration_s, 1),
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(model: str, limit: int | None) -> None:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    print(f"[backend] {base_url}  model={model}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_names(OUT_CSV)
    print(f"[resume] {len(existing)} persons already in CSV")

    # --- Connect to Neo4j ---
    print("[1/4] Fetching Person targets from Neo4j...")
    try:
        from neo4j import GraphDatabase  # type: ignore
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        all_persons = fetch_person_targets(driver)
        driver.close()
    except Exception as e:
        print(f"[error] Neo4j connection failed: {e}")
        return

    pending = [p for p in all_persons if str(p.get("name", "")) not in existing]
    if limit is not None:
        pending = pending[:limit]
    print(f"[1/4] {len(pending)} pending out of {len(all_persons)} total (Stage 3 enriched)")

    if not pending:
        print("Nothing to do.")
        return

    # --- Connect to Postgres (context-managed for guaranteed cleanup) ---
    print("[2/4] Connecting to Postgres...")
    try:
        import psycopg  # type: ignore
    except Exception as e:
        print(f"[error] Postgres driver import failed: {e}")
        return

    try:
        with psycopg.connect(PG_DSN) as pg, pg.cursor() as cur:
            corpus_mean = fetch_corpus_mean_body_words(cur)
            print(f"[2/4] corpus_mean_body_words = {corpus_mean:.1f}")

            client = LLMClient(base_url=base_url, api_key=api_key)
            write_header = not OUT_CSV.exists()

            print(f"[3/4] Processing {len(pending)} persons...")
            counts = {"ok": 0, "heuristic": 0}
            pbar = tqdm(pending, desc="persona", unit="p", dynamic_ncols=True)
            for props in pbar:
                name = str(props.get("name", "unknown"))
                pbar.set_postfix_str(name[:30], refresh=False)

                det = compute_deterministic_layer(props, corpus_mean)

                try:
                    emails = fetch_person_emails(cur, name)
                except Exception as e:
                    tqdm.write(f"  [warn] email fetch failed for {name}: {e}")
                    emails = []

                sample = stratified_sample_emails(
                    emails,
                    top_topics=det["expertise_topics"],
                    n_decision=5,
                    n_per_topic=5,
                    n_longest=5,
                )

                llm_result, prompt_chars, response_chars, duration_s, attempts = (
                    call_llm_with_retries(client, model, props, det, sample)
                )
                if not llm_result.get("role_summary"):
                    llm_result = synthesize_heuristic(det, props)
                    response_chars = len(json.dumps(llm_result))
                    status = "heuristic"
                    counts["heuristic"] += 1
                else:
                    status = f"ok-att{attempts}"
                    counts["ok"] += 1

                merged = merge_layers(det, llm_result)
                row = _flatten_row(name, merged, len(sample), prompt_chars,
                                   response_chars, duration_s, model)

                pd.DataFrame([row])[CSV_COLUMNS].to_csv(
                    OUT_CSV, mode="a", header=write_header, index=False
                )
                write_header = False

                pbar.set_postfix(
                    name=name[:20], status=status, ok=counts["ok"], h=counts["heuristic"],
                )
                tqdm.write(
                    f"  [{status}] {name}: formality={row['formality']} "
                    f"style={row['authority_style']} ({duration_s:.1f}s)"
                )
            pbar.close()
    except Exception as e:
        print(f"[error] persona run aborted: {e}")
        raise

    print(f"[4/4] Done. CSV written to {OUT_CSV}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4a: hybrid deterministic + LLM persona enrichment."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("INFERENCE_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit"),
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only N persons (dev/smoke).")
    args = parser.parse_args(argv)
    run(model=args.model, limit=args.limit)


if __name__ == "__main__":
    main()
