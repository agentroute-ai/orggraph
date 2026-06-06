"""4-way model benchmark for per-email signal extraction.

Tests gemma3:4b, gemma4:e4b, phi4-mini, qwen3:4b on the same N emails.
Reports per-model: latency, JSON parse rate, per-field fill rate, raw outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import httpx
import pandas as pd

from orggraph.config import OUTPUT_DIR
from orggraph.data.identity import build_alias_map, resolve_sender
from orggraph.data.loader import load_emails

OLLAMA = "http://localhost:11434/api/chat"
MODELS = [
    "gemma3:latest",   # 4B baseline (no reasoning)
    "gemma4:latest",   # E4B (~8B, reasoning, native JSON)
    "phi4-mini:latest",  # 3.8B (Microsoft, prompt-sensitive)
    "qwen3:4b",        # 4B (tied #1 on tool-calling bench)
]
N_EMAILS = int(os.environ.get("N_EMAILS", "5"))


PROMPT = """You are extracting structured signals from a single email at Enron Corporation.
Return JSON ONLY, matching this exact schema:

{{
  "topics": ["topic 1", "topic 2"],
  "intent": "<directive | informational | social | question | decision>",
  "sentiment": <number from -1.0 to 1.0>,
  "decision_carrying": <true | false>,
  "mentions_money": <true | false>,
  "mentions_regulator": <true | false>,
  "entities_mentioned": ["name 1", "Williams Companies"]
}}

Field constraints:
- topics: 1-3 short concise themes
- entities_mentioned: 0-5 named people or organizations from the email body

EMAIL:
From: {from_}
To: {to}
Date: {date}
Subject: {subject}

Body:
{body}
"""


REQUIRED_FIELDS = [
    "topics", "intent", "sentiment",
    "decision_carrying", "mentions_money",
    "mentions_regulator", "entities_mentioned",
]


def sample_emails(n: int) -> pd.DataFrame:
    print("[1/3] Loading and resolving corpus...", flush=True)
    alias_map = build_alias_map()
    emails = load_emails()
    sender_col = next(c for c in ["from", "From", "sender"] if c in emails.columns)
    to_col = next(c for c in ["to", "To", "recipients"] if c in emails.columns)
    body_col = next(c for c in ["body", "Body", "text"] if c in emails.columns)
    subject_col = next((c for c in ["subject", "Subject"] if c in emails.columns), "subject")
    date_col = next((c for c in ["date", "Date"] if c in emails.columns), "date")

    emails = emails.rename(columns={
        sender_col: "from", to_col: "to",
        subject_col: "subject", body_col: "body", date_col: "date",
    })
    emails["sender_resolved"] = emails["from"].astype(str).str.lower().map(
        lambda s: resolve_sender(s, alias_map)
    )
    emails = emails[emails["sender_resolved"].notna()].copy()
    emails["body_len"] = emails["body"].astype(str).str.len()
    short = emails[emails["body_len"] < 500].sample(min(n // 3 + 1, len(emails)), random_state=42)
    medium = emails[(emails["body_len"] >= 500) & (emails["body_len"] < 2000)].sample(
        min(n // 3 + 1, len(emails)), random_state=42
    )
    long_ = emails[emails["body_len"] >= 2000].sample(
        min(n - len(short) - len(medium), len(emails)), random_state=42
    )
    sample = pd.concat([short, medium, long_]).sample(frac=1, random_state=42).reset_index(drop=True).head(n)
    print(f"      sampled {len(sample)} emails")
    return sample


def call_one(email: pd.Series, model: str) -> dict:
    body = str(email["body"])[:2000]
    prompt = PROMPT.format(
        from_=str(email["from"])[:200],
        to=str(email["to"])[:300],
        date=str(email["date"]),
        subject=str(email["subject"])[:120],
        body=body,
    )

    ts = time.monotonic()
    try:
        r = httpx.post(
            OLLAMA,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {
                    "num_ctx": 8192,
                    "num_predict": 1024,
                    "temperature": 0.0,
                    "seed": 42,
                },
            },
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        raw = (data.get("message") or {}).get("content", "") or ""
        eval_count = data.get("eval_count")
        done_reason = data.get("done_reason")
    except Exception as e:
        return {"latency_s": time.monotonic() - ts, "error": f"{type(e).__name__}: {e}",
                "raw": "", "parsed": None, "eval_count": None, "done_reason": None}

    dur = time.monotonic() - ts
    parsed = None
    err = None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            parsed = None
            err = "not_a_dict"
    except json.JSONDecodeError as e:
        err = f"json_decode_error: {e}"

    return {
        "latency_s": round(dur, 2),
        "raw": raw,
        "parsed": parsed,
        "error": err,
        "eval_count": eval_count,
        "done_reason": done_reason,
    }


def summarize_model(model: str, results: list[dict]) -> dict:
    n = len(results)
    parsed = [r for r in results if r.get("parsed") is not None]
    lats = [r["latency_s"] for r in results]
    eval_counts = [r["eval_count"] for r in results if r.get("eval_count")]

    fill_rates = {}
    for field in REQUIRED_FIELDS:
        non_null = sum(1 for r in parsed if r["parsed"].get(field) not in (None, []))
        fill_rates[field] = f"{non_null}/{len(parsed)}" if parsed else "n/a"

    return {
        "model": model,
        "n": n,
        "parsed_pct": int(100 * len(parsed) / n) if n else 0,
        "latency_p50": round(pd.Series(lats).median(), 1) if lats else 0,
        "latency_p95": round(pd.Series(lats).quantile(0.95), 1) if lats else 0,
        "latency_max": round(max(lats), 1) if lats else 0,
        "eval_p50": int(pd.Series(eval_counts).median()) if eval_counts else 0,
        "fill_rates": fill_rates,
    }


def print_summary_table(rows: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("MODEL BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Model':25s} {'N':>3} {'Parsed%':>8} {'p50 lat':>8} {'p95 lat':>8} {'max lat':>8} {'p50 tok':>8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['model']:25s} {r['n']:>3} {r['parsed_pct']:>7d}% "
              f"{r['latency_p50']:>7.1f}s {r['latency_p95']:>7.1f}s "
              f"{r['latency_max']:>7.1f}s {r['eval_p50']:>8d}")

    print("\nField fill rates (parsed responses only):")
    print(f"{'Field':27s} " + " ".join(f"{r['model']:14s}" for r in rows))
    for field in REQUIRED_FIELDS:
        line = f"{field:27s} " + " ".join(
            f"{r['fill_rates'][field]:>14s}" for r in rows
        )
        print(line)


def print_first_outputs(per_model: dict[str, list[dict]], emails: pd.DataFrame, n: int = 2) -> None:
    print("\n" + "=" * 90)
    print(f"FIRST {n} EMAILS — RAW PARSED OUTPUTS PER MODEL")
    print("=" * 90)
    for i in range(min(n, len(emails))):
        e = emails.iloc[i]
        print(f"\n--- email #{i+1}  from={str(e['from'])[:50]}  subject={str(e['subject'])[:60]} ---")
        for model in per_model:
            r = per_model[model][i]
            tag = "OK" if r.get("parsed") else "FAIL"
            print(f"\n  [{model}] {r['latency_s']:.1f}s {tag}")
            if r.get("parsed"):
                # Compact representation
                p = r["parsed"]
                print(f"    topics={p.get('topics')}")
                print(f"    intent={p.get('intent')} sentiment={p.get('sentiment')} "
                      f"decision={p.get('decision_carrying')} money={p.get('mentions_money')} "
                      f"reg={p.get('mentions_regulator')}")
                print(f"    entities={p.get('entities_mentioned')}")
            else:
                print(f"    error: {r.get('error')}  raw[:120]: {r.get('raw', '')[:120]!r}")


def run() -> None:
    sample = sample_emails(N_EMAILS)
    per_model: dict[str, list[dict]] = {}
    summaries = []

    for model in MODELS:
        print(f"\n[2/3] Running {model} on {len(sample)} emails...", flush=True)
        results = []
        for i, (_, email) in enumerate(sample.iterrows(), 1):
            r = call_one(email, model)
            results.append(r)
            tag = "ok" if r.get("parsed") else "FAIL"
            print(f"  [{i:2d}/{len(sample)}] {r['latency_s']:>5.1f}s  {tag}  "
                  f"eval={r.get('eval_count'):>4} done={r.get('done_reason')}", flush=True)
        per_model[model] = results
        summaries.append(summarize_model(model, results))

    print("\n[3/3] Reporting results...")
    print_summary_table(summaries)
    print_first_outputs(per_model, sample, n=2)

    out = OUTPUT_DIR / "bench_email_extraction.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_emails": N_EMAILS,
        "summaries": summaries,
        "results": {m: per_model[m] for m in per_model},
    }, indent=2, default=str))
    print(f"\nSaved full results to {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="4-way model benchmark for per-email signal extraction."
    )
    parser.parse_args(argv)
    run()


if __name__ == "__main__":
    main()
