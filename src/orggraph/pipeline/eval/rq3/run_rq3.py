"""RQ3 R1-vs-R2 pilot orchestrator."""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_and_stratify(csv_path: Path, n: int, seed: int = 42) -> list[dict]:
    """Load the eval CSV and return a stratified sample of size *n*.

    Buckets by (qtype, role). Shuffles within each bucket with
    ``random.Random(seed)``. Per-bucket quota is ``max(1, n // n_buckets)``.
    Trims to <= n in total.
    """
    rows: list[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["qtype"], r["role"])].append(r)

    n_buckets = len(buckets)
    if n_buckets == 0:
        return []
    per_bucket = max(1, n // n_buckets)
    rng = random.Random(seed)
    out: list[dict] = []
    for bucket_rows in buckets.values():
        shuffled = list(bucket_rows)
        rng.shuffle(shuffled)
        out.extend(shuffled[:per_bucket])
    return out[:n]


import re as _re  # noqa: E402

from orggraph.evaluation.text_metrics import best_f1  # noqa: E402
from orggraph.pipeline.eval.rq3 import Answer  # noqa: E402
from orggraph.pipeline.eval.rq3.score import recall_at_k_fuzzy  # noqa: E402

_ANSWER_RE = _re.compile(r"ANSWER\s*:\s*(.+)", _re.IGNORECASE | _re.DOTALL)
_CITE_STRIP_RE = _re.compile(r"\[E[\d,\s]+\]")


def _extract_answer_text(text: str) -> str:
    """Strip the 'ANSWER:' prefix and inline citation markers from answer text."""
    m = _ANSWER_RE.search(text)
    extracted = m.group(1) if m else text
    return _CITE_STRIP_RE.sub("", extracted).strip()


def completed_qids(path: Path) -> set[tuple[int, str]]:
    """Return the set of (qid, condition) pairs already present in *path*."""
    if not path.exists():
        return set()
    done: set[tuple[int, str]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            done.add((obj["qid"], obj["condition"]))
    return done


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append *row* as a JSON line to *path*, fsync'd."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build_row(qid: int, qa: dict, answer: Answer) -> dict[str, Any]:
    """Assemble the JSONL row for one (question, condition) pair."""
    pred_text = _extract_answer_text(answer.text)
    f1 = best_f1(pred_text, qa["gold_answer"])
    recall = recall_at_k_fuzzy(list(answer.retrieved_bodies), qa["source_email"])
    return {
        "qid": qid,
        "condition": answer.condition,
        "question": qa["question"],
        "gold": qa["gold_answer"],
        "qtype": qa["qtype"],
        "role": qa["role"],
        "answer": answer.text,
        "f1": f1,
        "recall_at_10": recall,
        "retrieved_email_ids": list(answer.retrieved_email_ids),
        "cited_email_ids": list(answer.cited_email_ids),
        "latency_ms": answer.latency_ms,
        "used_fallback": getattr(answer, "_used_fallback", False),
    }


from statistics import mean as _mean  # noqa: E402

from orggraph.pipeline.eval.rq3.score import paired_bootstrap  # noqa: E402

_SUBSET_DEFS: dict[str, callable] = {
    "overall": lambda r: True,
    "relational_temporal": lambda r: r["qtype"] in ("relational", "temporal"),
    "factual": lambda r: r["qtype"] == "factual",
    "relational": lambda r: r["qtype"] == "relational",
    "temporal": lambda r: r["qtype"] == "temporal",
    "executive": lambda r: r["role"] == "executive",
    "manager": lambda r: r["role"] == "manager",
    "ic": lambda r: r["role"] == "ic",
}


def _subset_stats(rows: list[dict], pred) -> dict:
    rows = [r for r in rows if pred(r)]
    r1 = [r for r in rows if r["condition"] == "R1"]
    r2 = [r for r in rows if r["condition"] == "R2"]
    # match by qid
    r1_by_qid = {r["qid"]: r for r in r1}
    r2_by_qid = {r["qid"]: r for r in r2}
    shared = sorted(set(r1_by_qid) & set(r2_by_qid))
    return {
        "R1": {
            "n": len(r1),
            "mean_f1": _mean([r["f1"] for r in r1]) if r1 else 0.0,
            "mean_recall_at_10": _mean([r["recall_at_10"] for r in r1]) if r1 else 0.0,
            "mean_latency_ms": _mean([r["latency_ms"] for r in r1]) if r1 else 0.0,
            "fallback_rate": (
                sum(1 for r in r1 if r.get("used_fallback")) / len(r1) if r1 else 0.0
            ),
        },
        "R2": {
            "n": len(r2),
            "mean_f1": _mean([r["f1"] for r in r2]) if r2 else 0.0,
            "mean_recall_at_10": _mean([r["recall_at_10"] for r in r2]) if r2 else 0.0,
            "mean_latency_ms": _mean([r["latency_ms"] for r in r2]) if r2 else 0.0,
            "fallback_rate": (
                sum(1 for r in r2 if r.get("used_fallback")) / len(r2) if r2 else 0.0
            ),
        },
        "shared_qids": shared,
    }


def aggregate(rows: list[dict], *, seed: int = 42, n_resamples: int = 1000) -> dict:
    """Compute per-subset summary stats + paired bootstrap on F1 and Recall@10."""
    summary: dict[str, dict] = {}
    for name, pred in _SUBSET_DEFS.items():
        stats = _subset_stats(rows, pred)
        shared = stats.pop("shared_qids")
        if shared:
            r1_by_qid = {r["qid"]: r for r in rows if r["condition"] == "R1" and r["qid"] in shared}
            r2_by_qid = {r["qid"]: r for r in rows if r["condition"] == "R2" and r["qid"] in shared}
            r1_f1 = [r1_by_qid[q]["f1"] for q in shared]
            r2_f1 = [r2_by_qid[q]["f1"] for q in shared]
            r1_rec = [r1_by_qid[q]["recall_at_10"] for q in shared]
            r2_rec = [r2_by_qid[q]["recall_at_10"] for q in shared]
            d_f1, lo_f1, hi_f1, p_f1 = paired_bootstrap(r1_f1, r2_f1, n_resamples=n_resamples, seed=seed)
            d_rec, lo_rec, hi_rec, p_rec = paired_bootstrap(r1_rec, r2_rec, n_resamples=n_resamples, seed=seed)
            stats["delta_f1"] = {"mean": d_f1, "ci_low": lo_f1, "ci_high": hi_f1, "p": p_f1}
            stats["delta_recall_at_10"] = {"mean": d_rec, "ci_low": lo_rec, "ci_high": hi_rec, "p": p_rec}
            stats["n_paired"] = len(shared)
        summary[name] = stats
    return summary


def _row(subset: str, cond: str, s: dict) -> list[str]:
    return [
        subset, cond, str(s["n"]),
        f"{s['mean_f1']:.3f}",
        f"{s['mean_recall_at_10']:.3f}",
        f"{s['mean_latency_ms']:.0f}",
        f"{s['fallback_rate']:.2f}",
    ]


def write_summary(rows: list[dict], out_dir: Path, *, seed: int = 42, n_resamples: int = 1000) -> None:
    """Write summary.md + summary.csv to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows, seed=seed, n_resamples=n_resamples)

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "condition", "n", "mean_f1", "mean_recall_at_10", "mean_latency_ms", "fallback_rate"])
        for subset, s in summary.items():
            writer.writerow(_row(subset, "R1", s["R1"]))
            writer.writerow(_row(subset, "R2", s["R2"]))

    md_lines: list[str] = ["# RQ3 R1-vs-R2 pilot — results\n"]
    rt = summary.get("relational_temporal", {})
    if "delta_f1" in rt:
        d = rt["delta_f1"]
        dr = rt["delta_recall_at_10"]
        pass_f1 = "PASS" if d["mean"] >= 0.05 and d["ci_low"] > 0 else "FAIL"
        pass_rec = "PASS" if dr["mean"] >= 0.10 and dr["ci_low"] > 0 else "FAIL"
        md_lines.append(
            f"\n## Headline (relational + temporal subset, n_paired={rt['n_paired']})\n\n"
            f"- Δ F1 (R2−R1): **{d['mean']:+.3f}** (95% CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], p={d['p']:.3f}) — success threshold >= +0.05: **{pass_f1}**\n"
            f"- Δ Recall@10 (R2−R1): **{dr['mean']:+.3f}** (95% CI [{dr['ci_low']:+.3f}, {dr['ci_high']:+.3f}], p={dr['p']:.3f}) — success threshold >= +0.10: **{pass_rec}**\n"
        )
    md_lines.append("\n## Per-subset means\n\n")
    md_lines.append("| Subset | Cond | n | F1 | Recall@10 | Latency (ms) | Fallback rate |\n")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for subset, s in summary.items():
        for cond in ("R1", "R2"):
            row = s[cond]
            md_lines.append(
                f"| {subset} | {cond} | {row['n']} | {row['mean_f1']:.3f} | "
                f"{row['mean_recall_at_10']:.3f} | {row['mean_latency_ms']:.0f} | "
                f"{row['fallback_rate']:.2f} |\n"
            )
    (out_dir / "summary.md").write_text("".join(md_lines))


import argparse  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

import psycopg  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

from orggraph.config import REPO_ROOT  # noqa: E402
from orggraph.pipeline.eval.rq3 import answer_r1, answer_r2  # noqa: E402
from orggraph.pipeline.eval.rq3.r1_vector import (  # noqa: E402
    EMBED_MODEL,
    PG_DSN,
    VLLM_MODEL,
    make_answer_client,
    make_embed_client,
)
from orggraph.pipeline.eval.rq3.r2_graphrag import (  # noqa: E402
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)

_DEFAULT_CSV = REPO_ROOT / "datasets" / "validation" / "rq3_eval_set.csv"
_DEFAULT_OUT = REPO_ROOT / "outputs" / "rq3_pilot"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the RQ3 R1-vs-R2 pilot.")
    p.add_argument("--n", type=int, default=50, help="Number of stratified questions (default: 50).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--k", type=int, default=10, help="Top-k for retrieval (default: 10).")
    p.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--embed-model", default=EMBED_MODEL)
    p.add_argument("--answer-model", default=VLLM_MODEL)
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true", help="Stratify and report, no LLM calls.")
    p.add_argument("--preflight", action="store_true",
                   help="Sanity-check source-email coverage in embeddings_email and exit.")
    return p.parse_args(argv)


def _process_one(
    qid: int, qa: dict, *, pg_conn, neo4j_driver, answer_client, embed_client, k: int,
    embed_model: str, answer_model: str,
) -> list[dict]:
    out_rows: list[dict] = []
    a1 = answer_r1(
        question=qa["question"],
        pg_conn=pg_conn,
        embed_client=embed_client,
        answer_client=answer_client,
        k=k,
        embed_model=embed_model,
        answer_model=answer_model,
    )
    out_rows.append(build_row(qid, qa, a1))
    a2 = answer_r2(
        question=qa["question"],
        neo4j_driver=neo4j_driver,
        answer_client=answer_client,
        pg_conn=pg_conn,
        embed_client=embed_client,
        k=k,
        embed_model=embed_model,
        answer_model=answer_model,
    )
    out_rows.append(build_row(qid, qa, a2))
    return out_rows


def _run_preflight(args: argparse.Namespace) -> int:
    """Sample 10 questions, embed their source_email, see if any of the
    top-10 vector hits has >= 0.5 Jaccard with the source. Reports
    coverage rate as a sanity check on the eval metric.
    """
    sample = load_and_stratify(args.csv, n=10, seed=args.seed)
    embed_client = make_embed_client()
    conn = psycopg.connect(PG_DSN)
    try:
        register_vector(conn)
        hits = 0
        for qa in sample:
            src = qa["source_email"]
            if not src.strip():
                continue
            q_vec = embed_client.embeddings.create(model=args.embed_model, input=[src]).data[0].embedding
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT body_truncated FROM embeddings_email ORDER BY embedding <=> %s::vector LIMIT 10",
                    (q_vec, ),
                )
                bodies = [r[0] or "" for r in cur.fetchall()]
            if recall_at_k_fuzzy(bodies, src, threshold=0.5) >= 1.0:
                hits += 1
        print(f"[preflight] {hits}/10 source emails findable in embeddings_email (>= 0.5 jaccard).")
        return 0 if hits >= 5 else 2
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight:
        return _run_preflight(args)
    sample = load_and_stratify(args.csv, n=args.n, seed=args.seed)
    print(f"[1/4] Stratified {len(sample)} questions from {args.csv}")

    if args.dry_run:
        from collections import Counter
        dist = Counter((r["qtype"], r["role"]) for r in sample)
        for k_, v in sorted(dist.items()):
            print(f"  {k_}: {v}")
        return 0

    rows_path = args.out / "rows.jsonl"
    done = completed_qids(rows_path)
    print(f"[2/4] {len(done)} (qid, condition) pairs already complete; resuming.")

    embed_client = make_embed_client()
    answer_client = make_answer_client()
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        def worker(qid_qa: tuple[int, dict]) -> list[dict]:
            qid, qa = qid_qa
            # Each thread opens its own pg connection (psycopg connections aren't thread-safe to share)
            conn = psycopg.connect(PG_DSN)
            try:
                register_vector(conn)
                return _process_one(
                    qid, qa, pg_conn=conn, neo4j_driver=neo4j_driver,
                    answer_client=answer_client, embed_client=embed_client,
                    k=args.k, embed_model=args.embed_model, answer_model=args.answer_model,
                )
            finally:
                conn.close()

        todo: list[tuple[int, dict]] = []
        for qid, qa in enumerate(sample):
            needed = {(qid, "R1"), (qid, "R2")} - done
            if needed:
                todo.append((qid, qa))

        print(f"[3/4] Running {len(todo)} questions with concurrency={args.concurrency} ...")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(worker, x): x for x in todo}
            for fut in as_completed(futs):
                qid_, _ = futs[fut]
                try:
                    produced = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  qid={qid_} FAILED: {e}", file=sys.stderr)
                    continue
                for row in produced:
                    if (row["qid"], row["condition"]) in done:
                        continue
                    append_row(rows_path, row)
                    done.add((row["qid"], row["condition"]))
                    print(f"  qid={row['qid']:>3} {row['condition']} f1={row['f1']:.2f} rec={row['recall_at_10']:.0f} t={row['latency_ms']}ms")

        print(f"[4/4] Aggregating {len(done)} rows → summary.md / summary.csv ...")
        if not rows_path.exists():
            print("      No rows.jsonl written — all questions failed. Skipping summary.", file=sys.stderr)
            return 1
        all_rows: list[dict] = []
        with rows_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))
        if not all_rows:
            print("      rows.jsonl is empty. Skipping summary.", file=sys.stderr)
            return 1
        write_summary(all_rows, args.out, seed=args.seed, n_resamples=args.n_resamples)
        print(f"      → {args.out / 'summary.md'}")
        return 0
    finally:
        neo4j_driver.close()


if __name__ == "__main__":
    sys.exit(main())
