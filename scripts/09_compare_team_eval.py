"""Compare single-agent vs A2A cascade on the team-eval question set.

Reads two runs.jsonl directories (one per mode), scores routing +
dispatch metrics on the A2A run via team_metrics, and writes a
markdown report covering:

  - Headline pass rates per arm.
  - Routing accuracy + dispatch precision/coverage for A2A.
  - Per-question table with pass/fail diff and the agents each side
    consulted.
  - "Where A2A helps" - questions A2A passes that single agent fails.
  - "Where A2A loses" - questions single passes that A2A fails.
  - Latency + tool-call deltas.
  - Caveats.

Usage:
    python scripts/09_compare_team_eval.py \
        --single outputs/eval/kg_chat/team_questions_baseline \
        --a2a outputs/eval/kg_chat/team_questions_a2a \
        --questions src/orggraph/pipeline/eval/kg_chat/questions_team.yaml \
        --out docs/eval/team_chat/a2a_vs_single.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_runs(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    runs_path = path / "runs.jsonl"
    if not runs_path.exists():
        return rows
    with runs_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["question_id"]] = r
    return rows


def headline(rows: dict[str, dict]) -> dict:
    if not rows:
        return {"n": 0, "passed": 0, "pass_rate": 0.0,
                "mean_tools": 0.0, "mean_ms": 0.0}
    passed = sum(1 for r in rows.values() if r.get("pass"))
    return {
        "n": len(rows),
        "passed": passed,
        "pass_rate": passed / len(rows),
        "mean_tools": mean(r.get("n_tool_calls") or 0 for r in rows.values()),
        "mean_ms": mean(r.get("wall_time_ms") or 0 for r in rows.values()),
    }


def render(single: dict[str, dict], a2a: dict[str, dict],
           team_scored: dict) -> str:
    """team_scored is the output of team_metrics.score_runs_file on the A2A run."""
    qids = sorted(set(single) | set(a2a),
                  key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    h_s = headline(single)
    h_a = headline(a2a)
    team_summary = team_scored.get("summary") or {}
    team_per_q = {r["question_id"]: r for r in team_scored.get("per_question") or []}

    out: list[str] = []
    out.append("# Team-chat eval: A2A cascade vs single-agent\n\n")
    out.append("_10 multi-perspective workplace questions designed so the right\n"
               "answer requires consulting ≥2 colleagues from different domains.\n"
               "Both modes use the same 15-tool GraphRAG catalog and the same\n"
               "graders. Only the orchestration differs._\n\n")

    out.append("## Headline\n\n")
    out.append("| Metric | Single agent | A2A cascade | Δ |\n")
    out.append("|---|---:|---:|---:|\n")
    delta_pp = (h_a["pass_rate"] - h_s["pass_rate"]) * 100
    out.append(f"| **Answer pass rate** | "
               f"**{h_s['pass_rate']:.1%}** ({h_s['passed']}/{h_s['n']}) | "
               f"**{h_a['pass_rate']:.1%}** ({h_a['passed']}/{h_a['n']}) | "
               f"{delta_pp:+.1f} pp |\n")
    out.append(f"| Mean tool calls / Q | {h_s['mean_tools']:.1f} | "
               f"{h_a['mean_tools']:.1f} | "
               f"{h_a['mean_tools']-h_s['mean_tools']:+.1f} |\n")
    out.append(f"| Mean latency / Q | {h_s['mean_ms']/1000:.1f} s | "
               f"{h_a['mean_ms']/1000:.1f} s | "
               f"{(h_a['mean_ms']-h_s['mean_ms'])/1000:+.1f} s |\n\n")

    if team_summary:
        out.append("## A2A cascade quality (routing + dispatch)\n\n")
        out.append("| Metric | Value |\n|---|---:|\n")
        if team_summary.get("routing_accuracy") is not None:
            out.append(f"| Routing accuracy (initial pick in expected set) | "
                       f"{team_summary['routing_accuracy']:.1%} "
                       f"({team_summary['n_with_routing_gt']} of {team_summary['n']}) |\n")
        if team_summary.get("mean_dispatch_precision") is not None:
            out.append(f"| Mean dispatch precision | "
                       f"{team_summary['mean_dispatch_precision']:.1%} "
                       f"({team_summary['n_with_dispatches']} with any dispatch) |\n")
        if team_summary.get("mean_dispatch_coverage") is not None:
            out.append(f"| Mean dispatch coverage (categories hit / total) | "
                       f"{team_summary['mean_dispatch_coverage']:.1%} |\n")
        if team_summary.get("min_categories_met_rate") is not None:
            out.append(f"| Min-categories-met rate | "
                       f"{team_summary['min_categories_met_rate']:.1%} |\n")
        out.append("\n")

    out.append("## Per-question results\n\n")
    out.append("| ID | Category | Single | A2A | Initial pick | Dispatched | Categories hit |\n")
    out.append("|---|---|:-:|:-:|---|---|---|\n")
    helps: list[str] = []
    loses: list[str] = []
    for qid in qids:
        s = single.get(qid, {})
        a = a2a.get(qid, {})
        t = team_per_q.get(qid, {}).get("team") or {}
        s_mark = "✓" if s.get("pass") else "✗" if s else "-"
        a_mark = "✓" if a.get("pass") else "✗" if a else "-"
        cat = (s or a or {}).get("category", "?")
        initial = t.get("initial_pick") or "-"
        dispatched_list = t.get("dispatched") or []
        dispatched_str = ", ".join(dispatched_list) if dispatched_list else "-"
        covered = t.get("covered_categories") or []
        expected_cats = t.get("expected_categories") or []
        cats_str = f"{len(covered)}/{len(expected_cats)}" if expected_cats else "-"
        out.append(f"| {qid} | {cat} | {s_mark} | {a_mark} | `{initial}` "
                   f"| `{dispatched_str}` | {cats_str} |\n")
        if (not s.get("pass")) and a.get("pass"):
            helps.append(qid)
        if s.get("pass") and (not a.get("pass")):
            loses.append(qid)
    out.append("\n")

    if helps:
        out.append("## Where A2A helps (cascade fixed a single-agent failure)\n\n")
        for qid in helps:
            s = single.get(qid, {})
            a = a2a.get(qid, {})
            out.append(f"### {qid} - {a.get('category')}\n\n")
            out.append(f"**Question:** {a.get('question','')}\n\n")
            out.append(f"**Single-agent answer (FAIL):**\n\n> "
                       f"{(s.get('final_body') or '').strip()[:400] or '_no reply_'}\n\n")
            out.append(f"**A2A cascade answer (PASS):**\n\n> "
                       f"{(a.get('final_body') or '').strip()[:600]}\n\n")
            t = team_per_q.get(qid, {}).get("team") or {}
            if t.get("dispatched"):
                out.append(f"**Consulted:** {', '.join(t['dispatched'])}\n\n")

    if loses:
        out.append("## Where A2A loses (cascade regressed a single-agent pass)\n\n")
        for qid in loses:
            s = single.get(qid, {})
            a = a2a.get(qid, {})
            out.append(f"### {qid} - {s.get('category')}\n\n")
            out.append(f"**Question:** {s.get('question','')}\n\n")
            out.append(f"**Single-agent answer (PASS):**\n\n> "
                       f"{(s.get('final_body') or '').strip()[:400]}\n\n")
            out.append(f"**A2A cascade answer (FAIL):**\n\n> "
                       f"{(a.get('final_body') or '').strip()[:400] or '_no reply_'}\n\n")
            out.append(f"**Grader reason:** {a.get('reason','')}\n\n")

    out.append("## Caveats\n\n")
    out.append("- Single run per mode. MiniMax M2.7 AWQ-4bit is stochastic at temp>0 due to quantisation; per-question pass/fail can vary across reruns. Treat single-question flips as suggestive, not definitive.\n")
    out.append("- A2A wall time includes router + initial agent + dispatched colleagues + (optional) auto-reply synthesis. Latency overhead is expected.\n")
    out.append("- Routing and dispatch GT (`team_grading` in questions_team.yaml) reflects the author's read of the Enron org from the KG; not an oracle.\n")
    out.append("- LLM-as-judge graders (T1, T2, T4, T5, T7, T8) carry the judge's own bias. Deterministic graders (name_in_set, contains) cover T3, T6, T9, T10.\n")
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    from orggraph.pipeline.eval.kg_chat.team_metrics import score_runs_file
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--single", type=Path, required=True)
    p.add_argument("--a2a", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    single = load_runs(args.single)
    a2a = load_runs(args.a2a)
    team_scored = score_runs_file(args.a2a / "runs.jsonl", args.questions)

    md = render(single, a2a, team_scored)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)

    h_s = headline(single)
    h_a = headline(a2a)
    print(f"Wrote {args.out}  ({len(md)} bytes)")
    print(f"  Single : {h_s['passed']}/{h_s['n']} = {h_s['pass_rate']:.1%}")
    print(f"  A2A    : {h_a['passed']}/{h_a['n']} = {h_a['pass_rate']:.1%}")
    s = team_scored.get("summary") or {}
    if s.get("routing_accuracy") is not None:
        print(f"  Routing accuracy: {s['routing_accuracy']:.1%}")
    if s.get("mean_dispatch_precision") is not None:
        print(f"  Mean dispatch precision: {s['mean_dispatch_precision']:.1%}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
