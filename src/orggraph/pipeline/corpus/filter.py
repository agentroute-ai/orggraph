"""Stage 0 — Filter Enron corpus to emails involving resolved Persons.

Strict mode (per spec):
  - sender or any recipient resolves to one of the 139 Persons
  - body_chars >= 100
  - dedup by computed email_id

Output: ``<output-dir>/clean_emails.parquet``
       (default ``orggraph.config.OUTPUT_DIR``)
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from orggraph.config import OUTPUT_DIR
from orggraph.data.identity import build_alias_map, resolve_sender
from orggraph.data.loader import load_emails


def compute_email_id(sender: str, recipients: list[str], date: str, subject: str) -> str:
    """Stable 16-char sha256 of the email's natural key."""
    rec_sorted = ",".join(sorted([r.strip().lower() for r in recipients]))
    key = f"{sender.strip().lower()}|{rec_sorted}|{date}|{subject.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _split_recipients(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [a.strip() for a in val.split(",") if a.strip()]
    try:
        return [str(a).strip() for a in val if str(a).strip()]
    except TypeError:
        return []


def _resolve_recipients(addrs: list[str], alias_map: dict[str, str]) -> list[str]:
    out = []
    for a in addrs:
        n = resolve_sender(a.lower(), alias_map)
        if n:
            out.append(n)
    return out


def filter_emails(
    df: pd.DataFrame,
    alias_map: dict[str, str],
    min_body_chars: int = 100,
    max_body_chars: int = 2000,
) -> pd.DataFrame:
    """Apply the strict-mode filter and return a clean DataFrame."""
    sender_col = next((c for c in ["from", "From", "sender"] if c in df.columns), None)
    to_col = next((c for c in ["to", "To", "recipients"] if c in df.columns), None)
    body_col = next((c for c in ["body", "Body", "text"] if c in df.columns), None)
    subject_col = next((c for c in ["subject", "Subject"] if c in df.columns), "subject")
    date_col = next((c for c in ["date", "Date"] if c in df.columns), "date")
    if not all([sender_col, to_col, body_col]):
        raise RuntimeError(f"Required columns missing: {df.columns.tolist()}")

    df = df.rename(columns={
        sender_col: "from", to_col: "to",
        subject_col: "subject", body_col: "body", date_col: "date",
    })

    df["sender_resolved"] = df["from"].astype(str).str.lower().map(
        lambda s: resolve_sender(s, alias_map)
    )
    df["recipients_emails"] = df["to"].apply(_split_recipients)
    df["recipients_resolved"] = df["recipients_emails"].apply(
        lambda addrs: _resolve_recipients(addrs, alias_map)
    )

    keep_mask = df["sender_resolved"].notna() | df["recipients_resolved"].apply(lambda r: len(r) > 0)
    df = df[keep_mask].copy()

    df["body_chars"] = df["body"].astype(str).str.len()
    df = df[df["body_chars"] >= min_body_chars].copy()

    df["body_truncated"] = df["body"].astype(str).str[:max_body_chars]

    df["email_id"] = df.apply(
        lambda r: compute_email_id(
            str(r["from"]), r["recipients_emails"], str(r["date"]), str(r["subject"])
        ),
        axis=1,
    )
    df["thread_id"] = df["email_id"]

    df = df.drop_duplicates(subset=["email_id"]).reset_index(drop=True)

    df["sender_email"] = df["from"]
    df = df[[
        "email_id", "thread_id", "sender_email", "sender_resolved",
        "recipients_emails", "recipients_resolved",
        "date", "subject", "body_chars", "body_truncated",
    ]].copy()

    return df


def run(
    min_body_chars: int = 100,
    max_body_chars: int = 2000,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    out_parquet = output_dir / "clean_emails.parquet"

    print("[1/3] Loading Enron corpus from HuggingFace cache...", flush=True)
    df = load_emails()
    print(f"      raw rows: {len(df):,}")

    print("[2/3] Building alias map and filtering...", flush=True)
    alias_map = build_alias_map()
    out = filter_emails(df, alias_map, min_body_chars=min_body_chars, max_body_chars=max_body_chars)
    print(f"      filtered rows: {len(out):,} (kept {100 * len(out) / len(df):.1f}%)")

    print("[3/3] Writing parquet...", flush=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    print(f"      wrote {out_parquet}")
    print("\nSummary:")
    print(f"  total: {len(out):,}")
    if "date" in out.columns and len(out):
        print(f"  date range: {out['date'].min()} .. {out['date'].max()}")
    print(f"  unique senders: {out['sender_resolved'].nunique()}")
    print(f"  median body chars: {int(out['body_chars'].median())}")

    return out_parquet


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--min-body-chars", type=int, default=100)
    parser.add_argument("--max-body-chars", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for clean_emails.parquet (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)
    run(
        min_body_chars=args.min_body_chars,
        max_body_chars=args.max_body_chars,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
