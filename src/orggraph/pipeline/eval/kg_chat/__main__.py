"""End-to-end runner for the KG-Chat structural eval.

Usage:
    orggraph-eval-kg-chat [--limit N] [--output-dir PATH] [--model NAME]
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path

from orggraph.agents.agent import OpenAIChatClient
from orggraph.pipeline.agents.tools import (
    RAG_ENHANCED_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    build_default_registry,
    build_rag_enhanced_registry,
    build_rag_only_registry,
)
from orggraph.pipeline.eval.kg_chat.dataset import load_questions
from orggraph.pipeline.eval.kg_chat.grader import grade
from orggraph.pipeline.eval.kg_chat.judge import judge_answer
from orggraph.pipeline.eval.kg_chat.metrics import aggregate
from orggraph.pipeline.eval.kg_chat.report import render_report
from orggraph.pipeline.eval.kg_chat.runner import load_system_prompt, run_one
from orggraph.pipeline.eval.kg_chat.team_flow import run_team_turn
from orggraph.pipeline.eval.kg_chat.team_a2a_flow import run_a2a_turn


def _to_jsonable(rr) -> dict:
    out = {
        "question_id": rr.question_id,
        "category": rr.category,
        "question": rr.question,
        "tool_calls": [
            {"name": tc.name, "args": tc.args, "result_summary": tc.result_summary,
             "latency_ms": tc.latency_ms, "error": tc.error}
            for tc in rr.tool_calls
        ],
        "n_tool_calls": len(rr.tool_calls),
        "final_body": rr.final_body,
        "wall_time_ms": rr.wall_time_ms,
    }
    # A2A flow attaches cascade metadata as a side channel so the team
    # metrics scorer can read it from the runs.jsonl downstream.
    cascade = getattr(rr, "a2a_cascade", None)
    if cascade:
        out["a2a_cascade"] = cascade
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the KG-Chat structural eval.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--questions-file", type=Path, default=None,
        help="Path to questions.yaml. Defaults to the core 27-Q set in the package.",
    )
    p.add_argument("--model", default=os.environ.get("EVAL_LLM_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit"))
    p.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "anything"))
    p.add_argument("--max-tools", type=int, default=8)
    p.add_argument(
        "--mode", choices=["single", "team", "a2a"], default="single",
        help="single: one tool-calling agent (baseline). "
             "team: route -> N-round persona discussion -> moderator synthesis. "
             "a2a: cascade dispatcher — router picks one expert who fans out "
             "via address_colleague; replies auto-route back and consolidate.",
    )
    p.add_argument("--n-rounds", type=int, default=2, help="Team mode: discussion rounds.")
    p.add_argument("--k-responders", type=int, default=3, help="Team mode: max responders.")
    p.add_argument(
        "--retrieval", choices=["graphrag", "rag", "rag_enhanced"], default="graphrag",
        help="graphrag: full 15-tool catalog (structured retrieval + signals). "
             "rag: naive vector RAG baseline (search_emails_semantic + get_email_content). "
             "rag_enhanced: naive RAG + get_thread for parent-document expansion.",
    )
    args = p.parse_args(argv)

    if args.retrieval in {"rag", "rag_enhanced"} and args.mode in {"team", "a2a"}:
        p.error(f"--retrieval {args.retrieval} is incompatible with --mode {args.mode} "
                "(team/a2a router needs Neo4j tools; RAG baselines have none).")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "" if args.mode == "single" else f"_{args.mode}"
    retr_tag = "" if args.retrieval == "graphrag" else f"_{args.retrieval}"
    out_dir = args.output_dir or Path(f"outputs/eval/kg_chat/{ts}{mode_tag}{retr_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mode: {args.mode}  |  Retrieval: {args.retrieval}  |  Out: {out_dir}")

    questions = load_questions(args.questions_file) if args.questions_file else load_questions()
    if args.limit:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} questions")

    if args.retrieval == "rag":
        # Naive RAG baseline — no Neo4j, just pgvector for semantic search.
        registry = build_rag_only_registry()
        system_prompt = RAG_SYSTEM_PROMPT
        print(f"Retrieval: rag (naive vector baseline, {sum(1 for _ in registry)} tools)")
    elif args.retrieval == "rag_enhanced":
        # Enhanced RAG: naive + parent-document (thread) expansion.
        registry = build_rag_enhanced_registry()
        system_prompt = RAG_ENHANCED_SYSTEM_PROMPT
        print(f"Retrieval: rag_enhanced (RAG + thread expansion, {sum(1 for _ in registry)} tools)")
    else:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "orggraph2026"),
            ),
        )
        registry = build_default_registry(driver)
        system_prompt = load_system_prompt()
        print(f"Retrieval: graphrag ({sum(1 for _ in registry)} tools)")
    client = OpenAIChatClient(base_url=args.base_url, api_key=args.api_key)

    rows: list[dict] = []
    runs_path = out_dir / "runs.jsonl"
    with runs_path.open("w") as fh:
        for q in questions:
            if args.mode == "team":
                res = run_team_turn(
                    q, registry=registry, client=client, model=args.model,
                    system_prompt=system_prompt, max_tools=args.max_tools,
                    n_rounds=args.n_rounds, k_responders=args.k_responders,
                )
            elif args.mode == "a2a":
                res = run_a2a_turn(
                    q, registry=registry, client=client, model=args.model,
                    system_prompt=system_prompt, max_tools=args.max_tools,
                )
            else:
                res = run_one(
                    q, registry=registry, client=client, model=args.model,
                    system_prompt=system_prompt, max_tools=args.max_tools,
                )
            verdict, reason = grade(res.final_body, q.grading)
            if verdict is None:  # judge sentinel
                criterion = q.grading.get("prompt") or "Does the answer address the question correctly?"
                verdict, reason = judge_answer(
                    client=client, model=args.model,
                    question=q.question, answer=res.final_body, criterion=criterion,
                )
            rows.append({"result": res, "pass": bool(verdict), "reason": reason})
            tag = "PASS" if verdict else "FAIL"
            print(f"  [{q.id:>4}] {tag} {q.category:<10} {res.wall_time_ms:>6} ms  {reason[:60]}")
            fh.write(json.dumps({
                **_to_jsonable(res),
                "pass": bool(verdict),
                "reason": reason,
                "grading_kind": q.grading.get("kind"),
            }, default=str) + "\n")

    summary = aggregate(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "report.md").write_text(render_report(summary=summary, rows=rows))

    print()
    print(f"Joint pass rate: {summary['joint_pass_rate']:.1%}")
    for cat, info in summary["per_category"].items():
        mark = "✓" if info["meets_threshold"] else "✗"
        print(f"  {cat:<10} {info['pass_rate']:.1%}  (threshold {info['threshold']:.0%}) {mark}")
    print(f"\nWrote {runs_path}, summary.json, report.md to {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
