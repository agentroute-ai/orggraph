"""Routing + dispatch metrics for A2A cascade runs.

Reads ``a2a_cascade`` blocks from a team-mode runs.jsonl and the
matching ``team_grading`` clauses from the question set, and computes
per-question + aggregate:

  - **routing_hit**          : did the initial pick land in expected_initial?
  - **dispatch_precision**   : of dispatched colleagues, how many were in
                               any expected_targets category?
  - **dispatch_coverage**    : how many expected_targets categories were
                               covered by at least one dispatch?
  - **min_categories_met**   : did dispatch cover >= min_categories_covered?

Aggregates are means across the question set, plus the answer-grade
pass rate from the main grader for context.

This is the team-eval companion to ``metrics.py``; the existing
``metrics.aggregate`` already handles answer-pass aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


def _norm_slug(s: str) -> str:
    """Normalise a slug-or-display-name to a slug-like form for matching.

    Display names ("Mark Haedicke") come from the model's
    address_colleague call sometimes; we want to match against the
    roster slugs ("mark_haedicke") declared in team_grading.
    """
    return s.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")


def score_one(run_row: dict, team_grading: dict) -> dict:
    """Score one question's cascade against its team_grading clause.

    Returns a dict with the four metrics + the raw evidence.
    """
    cascade = run_row.get("a2a_cascade") or {}
    initial_pick = cascade.get("initial_pick")
    edges = cascade.get("dispatches") or []   # list of (src, dst)

    expected_initial = {_norm_slug(s) for s in (team_grading.get("expected_initial") or [])}
    expected_targets = team_grading.get("expected_targets") or {}
    min_categories = team_grading.get("min_categories_covered", 1)

    # --- routing ---
    routing_hit = (_norm_slug(initial_pick or "") in expected_initial) if expected_initial else None

    # --- dispatches ---
    # All dispatched destinations except "user" and the initial agent
    # itself (the seed edge ("user", initial_slug) shouldn't count
    # as a dispatch).
    dispatched: list[str] = [
        _norm_slug(dst) for (src, dst) in edges
        if src != "user" and dst != "user"
    ]
    # The same target may be dispatched to multiple times; dedupe.
    dispatched_uniq = list(dict.fromkeys(dispatched))

    # Build the set of all expected targets across categories (for precision)
    all_expected: set[str] = set()
    for cat_slugs in expected_targets.values():
        for s in cat_slugs:
            all_expected.add(_norm_slug(s))

    if dispatched_uniq:
        hits = [d for d in dispatched_uniq if d in all_expected]
        dispatch_precision = len(hits) / len(dispatched_uniq)
    else:
        dispatch_precision = None  # no dispatches — undefined

    # Category coverage
    covered_categories: list[str] = []
    for cat_name, cat_slugs in expected_targets.items():
        cat_set = {_norm_slug(s) for s in cat_slugs}
        if any(d in cat_set for d in dispatched_uniq):
            covered_categories.append(cat_name)
    dispatch_coverage = (
        len(covered_categories) / len(expected_targets) if expected_targets else None
    )
    min_categories_met = (len(covered_categories) >= min_categories) if expected_targets else None

    return {
        "routing_hit": routing_hit,
        "dispatch_precision": dispatch_precision,
        "dispatch_coverage": dispatch_coverage,
        "min_categories_met": min_categories_met,
        # raw evidence for the report
        "initial_pick": initial_pick,
        "dispatched": dispatched_uniq,
        "covered_categories": covered_categories,
        "expected_categories": list(expected_targets.keys()),
        "n_dispatches": len(dispatched_uniq),
    }


def aggregate_team(rows: list[dict]) -> dict:
    """Aggregate per-question team metrics across the question set.

    ``rows`` is a list of dicts with shape:
      {"question_id": str, "pass": bool, "team": <score_one output>}
    """
    def _mean(values: list[float | int | None]) -> float | None:
        vals = [v for v in values if v is not None]
        return mean(vals) if vals else None

    routing_hits = [r["team"]["routing_hit"] for r in rows]
    precisions = [r["team"]["dispatch_precision"] for r in rows]
    coverages = [r["team"]["dispatch_coverage"] for r in rows]
    min_cats = [r["team"]["min_categories_met"] for r in rows]
    answer_pass = [r["pass"] for r in rows]

    return {
        "n": len(rows),
        "answer_pass_rate": (sum(1 for p in answer_pass if p) / len(rows)) if rows else 0.0,
        "routing_accuracy": _mean(routing_hits),
        "mean_dispatch_precision": _mean(precisions),
        "mean_dispatch_coverage": _mean(coverages),
        "min_categories_met_rate": _mean(min_cats),
        "n_with_routing_gt": sum(1 for r in rows if r["team"]["routing_hit"] is not None),
        "n_with_dispatches": sum(1 for r in rows if r["team"]["n_dispatches"] > 0),
    }


def score_runs_file(
    runs_path: Path,
    questions_path: Path,
) -> dict:
    """End-to-end: read a runs.jsonl + questions.yaml, score each, aggregate.

    Returns ``{"per_question": [...], "summary": {...}}``.
    """
    from orggraph.pipeline.eval.kg_chat.dataset import load_questions

    questions = {q.id: q for q in load_questions(questions_path)}
    rows: list[dict] = []
    with runs_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            q = questions.get(qid)
            if not q or not q.team_grading:
                continue   # no team GT for this question — skip
            team = score_one(row, q.team_grading)
            rows.append({
                "question_id": qid,
                "category": row.get("category"),
                "question": row.get("question"),
                "pass": bool(row.get("pass")),
                "wall_time_ms": row.get("wall_time_ms"),
                "n_tool_calls": row.get("n_tool_calls"),
                "team": team,
            })

    return {
        "per_question": rows,
        "summary": aggregate_team(rows),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: orggraph-eval-team-metrics <runs.jsonl> <questions.yaml>"""
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", type=Path, help="Path to runs.jsonl from --mode a2a run.")
    p.add_argument("questions", type=Path, help="Path to questions YAML with team_grading.")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the scored JSON to this path (default: stdout).")
    args = p.parse_args(argv)

    result = score_runs_file(args.runs, args.questions)
    payload = json.dumps(result, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
        print(f"Wrote {args.out}")
    else:
        print(payload)

    s = result["summary"]
    print()
    print(f"Answer pass rate           : {s['answer_pass_rate']:.1%} ({s['n']} questions)")
    if s["routing_accuracy"] is not None:
        print(f"Routing accuracy           : {s['routing_accuracy']:.1%}  "
              f"({s['n_with_routing_gt']} with GT)")
    if s["mean_dispatch_precision"] is not None:
        print(f"Mean dispatch precision    : {s['mean_dispatch_precision']:.1%}  "
              f"({s['n_with_dispatches']} with any dispatch)")
    if s["mean_dispatch_coverage"] is not None:
        print(f"Mean dispatch coverage     : {s['mean_dispatch_coverage']:.1%}")
    if s["min_categories_met_rate"] is not None:
        print(f"Min-categories-met rate    : {s['min_categories_met_rate']:.1%}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
