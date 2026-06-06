"""RQ3 evaluation set builder (Task 9).

Samples 300-500 questions from MichaelR207/enron_qa_0922 stratified by
question type (factual / relational / temporal) and sender role
(executive / manager / IC). Writes a deterministic CSV to
``datasets/validation/rq3_eval_set.csv``.

## HF dataset schema (MichaelR207/enron_qa_0922, dev split)
Each row is expected to contain at minimum a question (or question list)
and a gold answer (or answer list).  Because the exact column names for
this community dataset can vary, the loader inspects ``ds[0]`` and adapts.
Known candidate column names are documented below.

## Classification rules
- Question type:
    - ``relational``: question contains the whole word 'who' (case-insensitive)
    - ``temporal``:   question contains 'when', 'date', 'in YYYY', or 'around Month YYYY'
    - ``factual``:    everything else

- Sender role (case-insensitive title matching):
    - ``executive``:  title contains CEO, COO, CFO, President, Chairman,
                      Vice Chairman
    - ``manager``:    title contains VP, SVP, Vice President, Director,
                      Manager
    - ``ic``:         empty / None / anything else
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from orggraph.config import REPO_ROOT

WORKTREE = REPO_ROOT

HF_DATASET_NAME = "MichaelR207/enron_qa_0922"
HF_SPLIT = "dev"

_OUT_DIR = WORKTREE / "datasets" / "validation"
_OUT_FILE = _OUT_DIR / "rq3_eval_set.csv"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_WHO_RE = re.compile(r"\bwho\b", re.IGNORECASE)
_TEMPORAL_RE = re.compile(
    r"\b(when|date|in\s+\d{4}|around\s+\w+\s+\d{4})\b",
    re.IGNORECASE,
)

# Executive title fragments (checked with case-insensitive search).
# "President" uses a negative lookbehind to exclude "Vice President"
# (which is a manager-level title).
_EXECUTIVE_PATTERNS = [
    r"\bCEO\b",
    r"\bCOO\b",
    r"\bCFO\b",
    r"(?<!Vice\s)\bPresident\b",        # President but NOT Vice President
    r"\bVice\s+Chairman\b",
    r"\bChairman\b",
    r"\bChief\s+\w+\s+Officer\b",       # "Chief Executive Officer", etc.
]
_EXECUTIVE_RE = re.compile("|".join(_EXECUTIVE_PATTERNS), re.IGNORECASE)

# Manager title fragments
_MANAGER_PATTERNS = [
    r"\bSVP\b",
    r"\bVP\b",
    r"\bVice\s+President\b",
    r"\bDirector\b",
    r"\bManager\b",
]
_MANAGER_RE = re.compile("|".join(_MANAGER_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_question_type(question: str) -> str:
    """Classify a question into one of ``"relational"``, ``"temporal"``, ``"factual"``.

    Priority order:
    1. ``relational`` — if the whole word 'who' appears (case-insensitive)
    2. ``temporal``   — if 'when', 'date', 'in YYYY', or 'around Month YYYY' appears
    3. ``factual``    — default
    """
    if _WHO_RE.search(question):
        return "relational"
    if _TEMPORAL_RE.search(question):
        return "temporal"
    return "factual"


def classify_sender_role(sender_title: str | None) -> str:
    """Classify a sender by job title into one of ``"executive"``, ``"manager"``, ``"ic"``.

    Matching is case-insensitive.

    - ``executive``: CEO, COO, CFO, President, Chairman, Vice Chairman,
                     Chief <Word> Officer patterns.
    - ``manager``:   VP, SVP, Vice President, Director, Manager.
    - ``ic``:        empty string, None, or any other title.
    """
    if not sender_title:
        return "ic"
    if _EXECUTIVE_RE.search(sender_title):
        return "executive"
    if _MANAGER_RE.search(sender_title):
        return "manager"
    return "ic"


def stratified_sample(
    items: list[dict],
    n: int,
    by: tuple[str, ...],
    seed: int = 42,
) -> list[dict]:
    """Return a stratified sample of up to *n* items from *items*.

    Buckets items by the composite key formed from the fields named in *by*.
    Within each bucket, items are shuffled deterministically using
    ``random.Random(seed)``.  The per-bucket quota is
    ``max(1, n // n_buckets)``.  Returns at most *n* items in total.

    Parameters
    ----------
    items:
        Source records (each a plain dict).
    n:
        Target sample size.
    by:
        Tuple of field names whose values form the stratification key.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    list[dict]
        Sampled records (order within each stratum is deterministic).
    """
    # Group into buckets
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        key = tuple(item.get(k) for k in by)
        buckets[key].append(item)

    n_buckets = len(buckets)
    if n_buckets == 0:
        return []

    per_bucket = max(1, n // n_buckets)
    rng = random.Random(seed)

    result: list[dict] = []
    for bucket_items in buckets.values():
        shuffled = list(bucket_items)
        rng.shuffle(shuffled)
        result.extend(shuffled[:per_bucket])

    # Trim to exactly n if we overshot (can happen when buckets are large
    # and n_buckets does not divide n evenly)
    return result[:n]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _inspect_columns(row: dict) -> tuple[str, str, str | None, str | None]:
    """Return (question_col, answer_col, sender_col, email_col) from a sample row."""
    keys = set(row.keys())

    # Question column candidates (ordered by preference)
    q_candidates = ["question", "questions", "query", "input"]
    q_col = next((c for c in q_candidates if c in keys), None)

    # Answer column candidates
    a_candidates = ["answers", "answer", "gold_answer", "gold_answers", "output", "label"]
    a_col = next((c for c in a_candidates if c in keys), None)

    # Optional: sender title
    sender_candidates = ["sender_title", "sender", "from", "author", "user"]
    sender_col = next((c for c in sender_candidates if c in keys), None)

    # Optional: email body / context
    email_candidates = ["email", "context", "body", "text", "email_text", "passage"]
    email_col = next((c for c in email_candidates if c in keys), None)

    return q_col, a_col, sender_col, email_col


def _coerce_to_str(value) -> str:
    """Flatten list/dict to a plain string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    return str(value)


def load_records_from_hf() -> list[dict]:
    """Load and normalise records from the HuggingFace dataset.

    Returns a list of dicts with keys:
        ``question``, ``gold_answer``, ``qtype``, ``role``,
        ``source_email``, ``sender``.

    If the dataset is unavailable, prints a message and returns an empty list.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("datasets library is required: pip install datasets") from exc

    try:
        ds = load_dataset(HF_DATASET_NAME, split=HF_SPLIT)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not load HF dataset '{HF_DATASET_NAME}': {exc}")
        return []

    if len(ds) == 0:
        print("[warn] Dataset loaded but is empty.")
        return []

    # Inspect the first row to find column names
    first_row = dict(ds[0])
    print(f"[info] Dataset columns: {list(first_row.keys())}")

    q_col, a_col, sender_col, email_col = _inspect_columns(first_row)

    if q_col is None:
        print(f"[warn] Cannot identify question column in: {list(first_row.keys())}")
        return []
    if a_col is None:
        print(f"[warn] Cannot identify answer column in: {list(first_row.keys())}")
        return []

    print(
        f"[info] Using columns — question: '{q_col}', answer: '{a_col}', "
        f"sender: '{sender_col}', email: '{email_col}'"
    )

    records: list[dict] = []
    for row in ds:
        question_raw = row.get(q_col)
        answer_raw = row.get(a_col)
        sender_raw = row.get(sender_col) if sender_col else None
        email_raw = row.get(email_col) if email_col else None

        # Handle datasets where a single row holds *multiple* Q/A pairs
        questions = question_raw if isinstance(question_raw, list) else [question_raw]
        answers = answer_raw if isinstance(answer_raw, list) else [answer_raw]

        # Zip Q/A pairs; pad with empty string if lengths differ
        for q, a in zip(
            questions,
            answers + [""] * max(0, len(questions) - len(answers)),
        ):
            q_str = _coerce_to_str(q).strip()
            a_str = _coerce_to_str(a).strip()
            if not q_str:
                continue
            sender_str = _coerce_to_str(sender_raw).strip()
            email_str = _coerce_to_str(email_raw).strip()

            records.append(
                {
                    "question": q_str,
                    "gold_answer": a_str,
                    "qtype": classify_question_type(q_str),
                    "role": classify_sender_role(sender_str or None),
                    "source_email": email_str,
                    "sender": sender_str,
                }
            )

    return records


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_COLUMNS = ["question", "gold_answer", "qtype", "role", "source_email", "sender"]


def write_csv(records: list[dict], path: Path) -> None:
    """Write *records* to a CSV file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"[info] Wrote {len(records)} rows to {path}")


# ---------------------------------------------------------------------------
# Distribution reporting
# ---------------------------------------------------------------------------


def _print_distribution(records: list[dict]) -> None:
    """Print a groupby summary of (qtype, role) counts."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        counts[(r.get("qtype", "?"), r.get("role", "?"))] += 1
    print("\n[info] Sample distribution (qtype × role):")
    for (qtype, role), count in sorted(counts.items()):
        print(f"  {qtype:12s}  {role:12s}  {count}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the RQ3 evaluation set from MichaelR207/enron_qa_0922."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=300,
        help="Target number of samples (default: 300).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic sampling (default: 42).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_OUT_FILE,
        help=f"Output CSV path (default: {_OUT_FILE}).",
    )
    args = parser.parse_args(argv)

    print(f"[info] Loading '{HF_DATASET_NAME}' split='{HF_SPLIT}' …")
    records = load_records_from_hf()

    if not records:
        print(
            "[error] No records loaded from HuggingFace dataset. "
            "Check your network connection or dataset availability."
        )
        return 0  # exit cleanly — not a fatal error in offline environments

    print(f"[info] Loaded {len(records)} Q/A records before sampling.")

    sampled = stratified_sample(records, n=args.n, by=("qtype", "role"), seed=args.seed)
    print(f"[info] Sampled {len(sampled)} records (target: {args.n}).")

    _print_distribution(sampled)
    write_csv(sampled, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
