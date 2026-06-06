"""Diff two KG-Chat eval runs into a side-by-side markdown report.

Usage:
    python scripts/07_compare_eval_runs.py \
        --baseline outputs/eval/kg_chat/<single_dir> \
        --candidate outputs/eval/kg_chat/<team_dir> \
        --out docs/eval/kg_chat/team_vs_single.md

Reads ``runs.jsonl`` from each directory and produces:
  - headline pass-rate delta
  - per-question PASS/FAIL diff
  - tool-call + latency aggregates
  - list of "interesting" questions where the two modes disagreed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_runs(path: Path) -> list[dict]:
    with (path / "runs.jsonl").open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def headline(rows: list[dict]) -> dict:
    passed = sum(1 for r in rows if r.get("pass"))
    return {
        "n": len(rows),
        "passed": passed,
        "pass_rate": passed / len(rows) if rows else 0.0,
        "total_tools": sum(r.get("n_tool_calls") or 0 for r in rows),
        "mean_tools": mean(r.get("n_tool_calls") or 0 for r in rows) if rows else 0.0,
        "mean_ms": mean(r.get("wall_time_ms") or 0 for r in rows) if rows else 0.0,
        "total_ms": sum(r.get("wall_time_ms") or 0 for r in rows),
    }


def render(baseline: list[dict], candidate: list[dict],
           baseline_label: str, candidate_label: str) -> str:
    b_idx = {r["question_id"]: r for r in baseline}
    c_idx = {r["question_id"]: r for r in candidate}
    qids = sorted(set(b_idx) | set(c_idx))

    h_b = headline(baseline)
    h_c = headline(candidate)

    out: list[str] = []
    out.append(f"# A/B comparison: {baseline_label} vs {candidate_label}\n")
    out.append(f"_Generated from {len(qids)} questions._\n\n")
    out.append("## Headline\n\n")
    out.append(f"| Metric | {baseline_label} | {candidate_label} | Δ |\n")
    out.append("|---|---:|---:|---:|\n")
    out.append(f"| Pass rate | {h_b['pass_rate']:.1%} ({h_b['passed']}/{h_b['n']}) "
               f"| {h_c['pass_rate']:.1%} ({h_c['passed']}/{h_c['n']}) "
               f"| {(h_c['pass_rate']-h_b['pass_rate'])*100:+.1f} pp |\n")
    out.append(f"| Total tool calls | {h_b['total_tools']} | {h_c['total_tools']} "
               f"| {h_c['total_tools']-h_b['total_tools']:+d} |\n")
    out.append(f"| Mean tool calls / Q | {h_b['mean_tools']:.1f} | {h_c['mean_tools']:.1f} "
               f"| {h_c['mean_tools']-h_b['mean_tools']:+.1f} |\n")
    out.append(f"| Mean latency / Q | {h_b['mean_ms']/1000:.1f} s "
               f"| {h_c['mean_ms']/1000:.1f} s "
               f"| {(h_c['mean_ms']-h_b['mean_ms'])/1000:+.1f} s |\n")
    out.append(f"| Total wall time | {h_b['total_ms']/60_000:.1f} min "
               f"| {h_c['total_ms']/60_000:.1f} min "
               f"| {(h_c['total_ms']-h_b['total_ms'])/60_000:+.1f} min |\n\n")

    out.append("## Per-question diff\n\n")
    out.append(f"| ID | Category | {baseline_label} | {candidate_label} | Δ |\n")
    out.append("|---|---|:-:|:-:|---|\n")
    flips_b_to_c: list[str] = []  # questions baseline passed, candidate failed
    flips_c_to_b: list[str] = []  # questions candidate passed, baseline failed
    for qid in qids:
        b = b_idx.get(qid)
        c = c_idx.get(qid)
        b_mark = "✓" if (b and b.get("pass")) else "✗" if b else "-"
        c_mark = "✓" if (c and c.get("pass")) else "✗" if c else "-"
        cat = (b or c or {}).get("category", "?")
        delta = ""
        if b and c:
            if b.get("pass") and not c.get("pass"):
                delta = "**regression**"
                flips_b_to_c.append(qid)
            elif not b.get("pass") and c.get("pass"):
                delta = "**improvement**"
                flips_c_to_b.append(qid)
        out.append(f"| {qid} | {cat} | {b_mark} | {c_mark} | {delta} |\n")
    out.append("\n")

    if flips_c_to_b or flips_b_to_c:
        out.append("## Disagreements (sanity check these)\n\n")
        for qid in flips_c_to_b:
            r = c_idx[qid]
            out.append(f"### {qid} - {candidate_label} improved\n\n")
            out.append(f"**Question:** {r.get('question','')}\n\n")
            out.append(f"**{candidate_label} answer:**\n\n> "
                       f"{(r.get('final_body') or '').strip()[:600]}\n\n")
            b_ans = (b_idx.get(qid) or {}).get("final_body") or ""
            out.append(f"**{baseline_label} answer:**\n\n> {b_ans.strip()[:600]}\n\n")
            out.append(f"**Grader reason ({candidate_label}):** {r.get('reason','')}\n\n")
        for qid in flips_b_to_c:
            r = b_idx[qid]
            out.append(f"### {qid} - {candidate_label} regressed\n\n")
            out.append(f"**Question:** {r.get('question','')}\n\n")
            c_ans = (c_idx.get(qid) or {}).get("final_body") or ""
            out.append(f"**{candidate_label} answer:**\n\n> {c_ans.strip()[:600]}\n\n")
            out.append(f"**{baseline_label} answer:**\n\n> "
                       f"{(r.get('final_body') or '').strip()[:600]}\n\n")
            c_reason = (c_idx.get(qid) or {}).get("reason") or ""
            out.append(f"**Grader reason ({candidate_label}):** {c_reason}\n\n")

    out.append("## Caveats\n\n")
    out.append("- Single run per mode. MiniMax M2.7 4-bit AWQ is stochastic at temp=0 due to quantization, so per-question pass/fail can vary across reruns. Treat single-question flips as suggestive, not definitive.\n")
    out.append("- Pass rate delta of one or two questions is within the noise floor on n=10.\n")
    out.append("- Team mode uses k_responders, n_rounds defaults from the run - check `summary.json` for exact config.\n")
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--baseline-label", default="single")
    p.add_argument("--candidate-label", default="team")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    baseline_rows = load_runs(args.baseline)
    candidate_rows = load_runs(args.candidate)
    md = render(baseline_rows, candidate_rows,
                args.baseline_label, args.candidate_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"Wrote {args.out}  ({len(md)} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
