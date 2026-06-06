"""Build the EnronQA -> project email id mapping via content-based heuristic.

The hard-questions dataset (``weaviate/hard-questions-enronqa``) does not
reliably index into the parent corpus via ``dataset_id``; row numbers are
misaligned. This module takes a different approach: for each hard question,
it extracts distinctive phrases from the question text and the GT candidate's
LLM-written content summary, then scores every email in ``embeddings_email``
by how many of those phrases appear in the subject, sender, and body fields.

Public API
----------
extract_signal_phrases(text) -> list[str]
score_email(phrases, email, weights) -> int
match_question_to_email(question, gt_candidate_content, emails) -> (id | None, debug)
load_pipeline_emails() -> list[dict]
load_hard_subset_with_gt_content() -> list[dict]
build_mapping(hard_rows, emails) -> pd.DataFrame
main(argv) -> int
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from orggraph.pipeline.agents.tools_pg import default_pg_conn_factory


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for phrase extraction
# ---------------------------------------------------------------------------

# Two or more consecutive Capitalized words (proper nouns)
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Mr./Mrs./Ms./Dr. + single capitalized name (Mr. Glynn, Dr. Smith)
_TITLED_NAME_RE = re.compile(r"\b((?:Mr|Mrs|Ms|Dr)\.\s+[A-Z][a-z]+)\b")

# Stylized company names with & or digits (PG&E, S&P, 8K, AT&T, 10-Q)
_STYLIZED_RE = re.compile(r"\b(\d+[A-Z]\b|[A-Z][A-Za-z0-9]*[&\d][A-Za-z0-9&]*)\b")

# All-caps tokens of 3+ characters (acronyms like CPUC, FERC, ISO)
_ACRONYM_RE = re.compile(r"\b([A-Z]{3,})\b")

# 'X's email' or 'email from X' or 'email by X' patterns — extract the person name
_SENDER_HINT_RE = re.compile(
    r"\b(?:from|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
    r"|\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:'s)?\s+email"
)

# Month-day date patterns like "September 25", "October 4"
_MONTH_DAY_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2}\b"
)

# Weekday names
_WEEKDAY_RE = re.compile(
    r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
)

# Single capitalized words that follow common trigger words
_TRIGGER_NAME_RE = re.compile(
    r"\b(?:by|for|from|to)\s+([A-Z][a-z]{2,})\b"
)

# Common stop-phrases that are not useful signals
_STOP_WORDS = frozenset({
    "The", "On", "This", "For", "In", "At", "An", "A", "Of", "To",
    "By", "Be", "It", "Is", "And", "Or", "But", "As", "With", "From",
    "That", "Are", "Was", "Has", "Had", "Not", "Can", "Its", "Our",
    "Their", "There", "These", "Those", "Have",
})


def extract_signal_phrases(text: str) -> list[str]:
    """Return a deduplicated list of distinctive phrases from *text*.

    Extracts:
    - Proper nouns (sequences of 2+ capitalized words)
    - Acronyms (all-caps tokens of 3+ chars)
    - Month-day date patterns ("September 25")
    - Weekday names ("Friday")
    - Single capitalized names following trigger words ("by Sarah")

    Filters common stop-words. Caps at 10 phrases, preferring longer ones.
    """
    found: list[str] = []

    # Proper nouns (2+ capitalized words) — highest signal
    for m in _PROPER_NOUN_RE.finditer(text):
        phrase = m.group(1)
        if phrase not in _STOP_WORDS:
            found.append(phrase)

    # Mr./Mrs./Ms./Dr. + name
    for m in _TITLED_NAME_RE.finditer(text):
        found.append(m.group(1))

    # Stylized company / ID tokens (PG&E, 8K, S&P)
    for m in _STYLIZED_RE.finditer(text):
        phrase = m.group(1)
        if phrase not in _STOP_WORDS and len(phrase) >= 2:
            found.append(phrase)

    # Acronyms
    for m in _ACRONYM_RE.finditer(text):
        phrase = m.group(1)
        if phrase not in _STOP_WORDS:
            found.append(phrase)

    # Month-day dates
    for m in _MONTH_DAY_RE.finditer(text):
        found.append(m.group(0))

    # Weekday names
    for m in _WEEKDAY_RE.finditer(text):
        day = m.group(1)
        if day not in _STOP_WORDS:
            found.append(day)

    # Single capitalized names after trigger words
    for m in _TRIGGER_NAME_RE.finditer(text):
        name = m.group(1)
        if name not in _STOP_WORDS and name not in found:
            found.append(name)

    # Deduplicate preserving first occurrence
    seen: set[str] = set()
    deduped: list[str] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    # Cap at 10, taking the longest phrases first (more specific = more useful)
    if len(deduped) > 10:
        deduped = sorted(deduped, key=len, reverse=True)[:10]

    return deduped


def score_email(
    phrases: list[str],
    email: dict,
    weights: dict | None = None,
) -> int:
    """Score an email against a list of signal phrases.

    Each phrase contributes at most once per field (no double-counting).
    Score = sum of weight * (1 if phrase appears in field else 0)
    across all phrases and all weighted fields.

    Default weights: subject=3, sender_resolved=4, body_truncated=1.
    """
    if weights is None:
        weights = {"subject": 3, "sender_resolved": 4, "body_truncated": 1}

    total = 0
    for field, weight in weights.items():
        field_text = (email.get(field) or "").lower()
        for phrase in phrases:
            if phrase.lower() in field_text:
                total += weight
    return total


def _extract_sender_hint(text: str) -> str | None:
    """Extract a sender name from patterns like 'Joseph Alamo's email' or
    'email from Sarah Novosel'. Returns the canonical-looking name or None.
    """
    m = _SENDER_HINT_RE.search(text)
    if not m:
        return None
    return m.group(1) or m.group(2)


def match_question_to_email(
    question: str,
    gt_candidate_content: str,
    emails: list[dict],
    max_tied_emails: int = 20,
) -> tuple[list[str], dict]:
    """Match a hard question to its source email(s) in *emails*.

    Two-stage strategy:
    1. Optional sender-hint prefilter: if the question contains an "X's
       email" pattern AND X is a sender in the corpus, narrow to those.
       Often the hinted name is in headers but not the canonical
       ``sender_resolved``; in that case the prefilter does nothing.
    2. Phrase-score every candidate by proper-noun / acronym overlap
       across the question + GT content.

    Returns ``(matches, debug)`` where ``matches`` is a list of email_ids
    sharing the top score (often >1 because the Enron corpus replicates
    each email across every custodian whose mailbox carried a copy).
    Empty list means no plausible match. ``debug`` carries ``top_score``,
    ``runner_up_score``, ``phrases``, ``sender_hint``, ``candidate_pool``,
    and ``n_tied``.
    """
    sender_hint = _extract_sender_hint(question) or _extract_sender_hint(gt_candidate_content)

    q_phrases = extract_signal_phrases(question)
    c_phrases = extract_signal_phrases(gt_candidate_content)

    seen: set[str] = set()
    phrases: list[str] = []
    for p in q_phrases + c_phrases:
        if p not in seen:
            seen.add(p)
            phrases.append(p)

    candidates = emails
    if sender_hint:
        sh = sender_hint.lower()
        narrowed = [e for e in emails if (e.get("sender_resolved") or "").lower() == sh]
        if narrowed:
            candidates = narrowed

    if not phrases:
        return [], {
            "top_score": 0, "runner_up_score": 0, "phrases": [],
            "sender_hint": sender_hint, "candidate_pool": len(candidates), "n_tied": 0,
        }

    scored: list[tuple[int, str]] = []
    for em in candidates:
        s = score_email(phrases, em)
        scored.append((s, em["email_id"]))

    scored.sort(key=lambda x: -x[0])

    top_score = scored[0][0] if scored else 0
    # Runner-up = the first score strictly below top_score.
    runner_up_score = next((s for s, _ in scored if s < top_score), 0)

    # All emails tied at the top score.
    top_emails = [eid for s, eid in scored if s == top_score]

    debug = {
        "top_score": top_score,
        "runner_up_score": runner_up_score,
        "phrases": phrases,
        "sender_hint": sender_hint,
        "candidate_pool": len(candidates),
        "n_tied": len(top_emails),
    }

    min_score = 2 if sender_hint and len(candidates) < len(emails) else 3
    if top_score < min_score:
        return [], debug

    # When dozens of emails share the top score, the signal is too weak
    # (likely a generic phrase like "Friday" or a famous name). Cap.
    if len(top_emails) > max_tied_emails:
        return [], debug

    return top_emails, debug


def load_pipeline_emails() -> list[dict]:
    """Read all rows from ``embeddings_email`` returning lightweight dicts.

    Returns a list of dicts with keys: ``email_id``, ``subject``,
    ``body_truncated``, ``sender_resolved``, ``date``.
    """
    factory = default_pg_conn_factory()
    with factory() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT email_id, subject, body_truncated, sender_resolved, date"
            " FROM embeddings_email"
        )
        rows = cur.fetchall()
    return [
        {
            "email_id": r[0],
            "subject": r[1] or "",
            "body_truncated": r[2] or "",
            "sender_resolved": r[3] or "",
            "date": str(r[4]) if r[4] else "",
        }
        for r in rows
    ]


def load_hard_subset_with_gt_content() -> list[dict]:
    """Fetch hard-questions dataset and resolve the GT candidate content.

    For each row, locates the candidate whose ``int(dataset_id)`` equals
    ``ground_truths[0]``. Returns rows shaped as::

        {"enronqa_doc_id": int, "question": str, "gt_content": str}

    Rows where no candidate matches the GT integer are logged as warnings
    and skipped.
    """
    from datasets import load_dataset

    ds = load_dataset("weaviate/hard-questions-enronqa", split="train")
    out: list[dict] = []
    for row in ds:
        gt_id = int(row["ground_truths"][0])
        gt_candidate = None
        for cand in row["shortlisted_candidates"]:
            if int(cand["dataset_id"]) == gt_id:
                gt_candidate = cand
                break
        if gt_candidate is None:
            logger.warning(
                "No candidate matches ground_truth %d for question: %s...",
                gt_id,
                row["question"][:60],
            )
            continue
        out.append({
            "enronqa_doc_id": gt_id,
            "question": row["question"],
            "gt_content": gt_candidate["content"],
        })
    return out


def build_mapping(
    hard_rows: list[dict],
    emails: list[dict],
) -> pd.DataFrame:
    """Orchestrate phrase-based matching for all hard-subset rows.

    For each hard row, calls ``match_question_to_email`` and emits ONE row
    per matched email_id. Hard rows often map to multiple emails because
    the Enron corpus replicates the same email across each custodian
    mailbox that received it. The eval treats any of these as a valid
    citation_hit.

    Returns a DataFrame with columns::

        enronqa_doc_id, orggraph_email_id, match_method,
        top_score, runner_up_score, n_phrases, n_tied
    """
    records: list[dict] = []
    for row in hard_rows:
        eids, info = match_question_to_email(
            question=row["question"],
            gt_candidate_content=row["gt_content"],
            emails=emails,
        )
        if not eids:
            logger.debug(
                "No match for enronqa_doc_id=%d (top_score=%d, n_tied=%d)",
                row["enronqa_doc_id"], info["top_score"], info["n_tied"],
            )
            continue
        for eid in eids:
            records.append({
                "enronqa_doc_id": row["enronqa_doc_id"],
                "orggraph_email_id": eid,
                "match_method": "content_heuristic",
                "top_score": info["top_score"],
                "runner_up_score": info["runner_up_score"],
                "n_phrases": len(info["phrases"]),
                "n_tied": info["n_tied"],
            })
    return pd.DataFrame.from_records(records)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the mapping build.

    Args:
        --out            Output parquet path (default datasets/enronqa/id_mapping.parquet)
        --min-coverage   Minimum fraction of hard rows that must match (default 0.70)
        --limit          Optional integer to cap the number of hard rows processed
    """
    p = argparse.ArgumentParser(
        description="Build EnronQA -> project email id mapping via content heuristic."
    )
    p.add_argument("--out", type=Path, default=Path("datasets/enronqa/id_mapping.parquet"))
    p.add_argument("--min-coverage", type=float, default=0.70)
    p.add_argument("--limit", type=int, default=None, help="Cap hard rows (for testing)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    hard_rows = load_hard_subset_with_gt_content()
    if args.limit is not None:
        hard_rows = hard_rows[: args.limit]
    print(f"Hard rows loaded: {len(hard_rows)}")

    emails = load_pipeline_emails()
    print(f"Pipeline emails loaded: {len(emails)}")

    df = build_mapping(hard_rows=hard_rows, emails=emails)

    total = len(hard_rows)
    matched_doc_ids = df["enronqa_doc_id"].nunique() if not df.empty else 0
    coverage = matched_doc_ids / max(1, total)

    for row in hard_rows:
        n = (df["enronqa_doc_id"] == row["enronqa_doc_id"]).sum() if not df.empty else 0
        status = f"MATCH x{n}" if n else "SKIP"
        print(f"  {status:<10} doc={row['enronqa_doc_id']}")

    print(f"Coverage: {coverage:.2%} ({matched_doc_ids} / {total} hard questions; {len(df)} (doc, email) rows)")

    if coverage < args.min_coverage:
        print(
            f"FAIL: coverage {coverage:.2%} below threshold {args.min_coverage:.2%}",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"Wrote {len(df)} (doc, email) rows to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
