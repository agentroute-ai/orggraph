"""Task-completion evaluation for RQ2 (and RQ3 retrieval) on enron_qa_0922.

Each QA pair from MichaelR207/enron_qa_0922 is wrapped as a 2-agent
dialogue: "Sara and Louise, work together to answer this question;
provide a single concise final answer prefixed with 'ANSWER:'". Three
conditions are run per question:

  C1       multi-agent A2A with persona-only system prompts
  C1_tools multi-agent A2A with the KG tool registry attached
  C2       single-LLM long-context (no per-agent isolation)

The final answer is extracted from the dialogue tail (the last
'ANSWER: ...' line, falling back to the last non-END turn) and scored
with SQuAD-style token-overlap F1 against the gold answer and any
alternates. The five RQ2 naturalness dimensions are also scored on the
same transcripts via the existing dialogue judge, so each row of the
output has both task-completion (F1) and dialogue-quality numbers.

Usage:
    python scripts/06_rq2_qa_task_eval.py --n 30 \\
        --model cyankiwi/MiniMax-M2.7-AWQ-4bit \\
        --base-url http://localhost:8000/v1

Outputs:
    outputs/rq2_qa_task/results.json     per-(question, condition) rows
    outputs/rq2_qa_task/summary.md       aggregated comparison table
    outputs/rq2_qa_task/transcripts/     JSONL per (question, condition)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset
from neo4j import GraphDatabase

from orggraph.evaluation.text_metrics import best_f1
from orggraph.agents.agent import OpenAIChatClient, PersonaAgent
from orggraph.agents.persona import load_personas_from_csv
from orggraph.config import OUTPUT_DIR
from orggraph.evaluation.dialogue_judge import (
    DIMENSIONS,
    judge_transcript,
)
from orggraph.pipeline.agents.tools import build_default_registry
from orggraph.pipeline.agents.tools_logging import ToolCallLog
from orggraph.simulation.runner import run_multi_agent
from orggraph.simulation.scenario import Scenario
from orggraph.simulation.single_llm import run_single_llm

DEFAULT_AGENT_A = "Sara Shackleton"
DEFAULT_AGENT_B = "Louise Kitchen"
DEFAULT_DATASET = "MichaelR207/enron_qa_0922"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "rq2_qa_task"


ANS_RE = re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE | re.DOTALL)


def extract_answer_from_messages(messages) -> str:
    for m in reversed(messages):
        body = (m.body or "").strip()
        if not body or body.upper() == "END":
            continue
        match = ANS_RE.search(body)
        if match:
            return match.group(1).strip().split("\n")[0]
    for m in reversed(messages):
        body = (m.body or "").strip()
        if body and body.upper() != "END":
            return body[:400]
    return ""


def sample_questions(n: int, dataset: str) -> list[dict]:
    """Stream the first ``n`` questions whose row has at least one gold answer."""
    print(f"[1] Loading {n} questions from {dataset}...")
    ds = load_dataset(dataset, split="train", streaming=True)
    out = []
    for i, row in enumerate(ds):
        questions = row.get("questions") or []
        gold_answers = row.get("gold_answers") or []
        alternates = row.get("alternate_answers") or []
        if not questions or not gold_answers:
            continue
        out.append({
            "id": i,
            "question": questions[0],
            "gold": gold_answers[0],
            "alternates": alternates[0] if alternates else [],
            "path": row.get("path"),
        })
        if len(out) >= n:
            break
    print(f"    Loaded {len(out)} questions.")
    return out


def build_scenario(qa: dict, agent_a: str, agent_b: str, max_turns: int = 6) -> Scenario:
    return Scenario(
        name=f"qa_q{qa['id']}",
        flow="activity",
        brief=(
            f"You and your colleague need to answer this question for the "
            f"organisation. Discuss briefly, then in your final turn give a "
            f"single concise answer prefixed with 'ANSWER:'. Question: "
            f"{qa['question']}"
        ),
        participants=(agent_a, agent_b),
        starter=agent_a,
        seed_message="",
        max_turns=max_turns,
    )


def run_one(
    qa: dict,
    condition: str,
    personas,
    client,
    registry,
    model: str,
    transcripts_dir: Path,
    temperature: float = 0.7,
) -> dict:
    scen = build_scenario(qa, DEFAULT_AGENT_A, DEFAULT_AGENT_B)
    t0 = time.time()
    log = None
    if condition == "C1":
        agents = {
            n: PersonaAgent(persona=personas[n], client=client, model=model,
                            temperature=temperature)
            for n in scen.participants
        }
        transcript = run_multi_agent(scen, agents, condition="C1")
    elif condition == "C1_tools":
        log = ToolCallLog()
        agents = {
            n: PersonaAgent(persona=personas[n], client=client, model=model,
                            temperature=temperature,
                            tools=registry, tool_call_log=log)
            for n in scen.participants
        }
        transcript = run_multi_agent(scen, agents, condition="C1_tools")
    elif condition == "C2":
        transcript = run_single_llm(
            scen, personas, client, model=model, temperature=temperature,
        )
    else:
        raise ValueError(f"unknown condition {condition!r}")
    duration = time.time() - t0
    answer = extract_answer_from_messages(transcript.messages)
    f1 = best_f1(answer, qa["gold"], qa["alternates"])
    out_path = transcripts_dir / f"qa_q{qa['id']}__{condition}.jsonl"
    transcript.to_jsonl(out_path)
    if log is not None:
        log_path = transcripts_dir / f"qa_q{qa['id']}__{condition}__tool_calls.jsonl"
        log.to_jsonl(log_path)
    return {
        "condition": condition,
        "answer": answer,
        "f1": f1,
        "n_messages": len(transcript),
        "n_tool_calls": len(log.entries) if log else 0,
        "duration_s": round(duration, 1),
        "transcript_path": str(out_path),
    }


def run_judge(qa: dict, condition: str, transcript_path: Path, personas, client,
              model: str) -> dict:
    """Run the dialogue judge on a transcript to produce 5 naturalness scores."""
    from orggraph.simulation.transcript import Transcript
    transcript = Transcript.from_jsonl(transcript_path)
    scen = build_scenario(qa, DEFAULT_AGENT_A, DEFAULT_AGENT_B)
    try:
        verdict = judge_transcript(
            transcript,
            scenario_brief=scen.brief,
            participants=list(scen.participants),
            personas=personas,
            client=client,
            model=model,
            temperature=0.0,
        )
        scores = {d: verdict.scores[d].score for d in DIMENSIONS}
        return {
            "judge_ok": True,
            **scores,
            "judge_mean": verdict.mean_score() if callable(verdict.mean_score) else verdict.mean_score,
            "judge_flags": len(verdict.turn_flags),
        }
    except Exception as e:  # noqa: BLE001
        return {"judge_ok": False, "judge_error": str(e)[:200]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--model", default="cyankiwi/MiniMax-M2.7-AWQ-4bit")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="orggraph2026")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip the 5-dimension naturalness judge (faster).")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = args.out_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    qas = sample_questions(args.n, DEFAULT_DATASET)

    print(f"[2] Loading personas from {OUTPUT_DIR / 'person_enrichment.csv'}...")
    personas_all = load_personas_from_csv(OUTPUT_DIR / "person_enrichment.csv")
    personas = {n: personas_all[n] for n in (DEFAULT_AGENT_A, DEFAULT_AGENT_B)
                if n in personas_all}
    if len(personas) < 2:
        print(f"    ERROR: need {DEFAULT_AGENT_A} and {DEFAULT_AGENT_B} in CSV")
        sys.exit(1)

    print(f"[3] Connecting to LLM endpoint at {args.base_url}...")
    client = OpenAIChatClient(base_url=args.base_url, api_key="EMPTY")

    print("[4] Building KG tool registry...")
    drv = GraphDatabase.driver(args.neo4j_uri,
                               auth=(args.neo4j_user, args.neo4j_password))
    registry = build_default_registry(drv)
    print(f"    {len(registry.to_openai_tools())} tools registered.")

    print(f"[5] Running {len(qas)} questions × 3 conditions = "
          f"{len(qas) * 3} dialogues...")
    results = []
    for qa in qas:
        print(f"\n--- Q{qa['id']} ---")
        print(f"  Q: {qa['question'][:120]}")
        print(f"  Gold: {qa['gold'][:120]}")
        for condition in ("C1", "C1_tools", "C2"):
            try:
                r = run_one(qa, condition, personas, client, registry,
                            model=args.model, transcripts_dir=transcripts_dir)
            except Exception as e:  # noqa: BLE001
                print(f"  [{condition:9s}] ERROR: {e}")
                continue
            print(f"  [{condition:9s}] f1={r['f1']:.3f}  msgs={r['n_messages']}  "
                  f"tools={r['n_tool_calls']:>2d}  {r['duration_s']:.1f}s")
            results.append({
                "qa_id": qa["id"], "question": qa["question"],
                "gold": qa["gold"], **r,
            })

    if not args.skip_judge:
        print(f"\n[6] Judging {len(results)} transcripts on 5 naturalness "
              f"dimensions...")
        for i, r in enumerate(results):
            qa = next(q for q in qas if q["id"] == r["qa_id"])
            j = run_judge(qa, r["condition"], Path(r["transcript_path"]),
                          personas, client, model=args.model)
            r.update(j)
            ok = "ok" if j.get("judge_ok") else "fail"
            mean = j.get("judge_mean", "n/a")
            print(f"  judged {i+1}/{len(results)} ({r['condition']}): {ok}, "
                  f"mean={mean}")

    # Persist
    out_path = args.out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[7] Saved {len(results)} rows → {out_path}")

    # Aggregate
    print("\n[8] Aggregation by condition:")
    print(f"{'cond':10s} {'mean F1':>9s} {'mean tools':>11s} "
          f"{'mean dur(s)':>12s} {'mean nat':>9s}")
    for cond in ("C1", "C1_tools", "C2"):
        sub = [r for r in results if r["condition"] == cond]
        if not sub:
            continue
        mean_f1 = sum(r["f1"] for r in sub) / len(sub)
        mean_tools = sum(r["n_tool_calls"] for r in sub) / len(sub)
        mean_dur = sum(r["duration_s"] for r in sub) / len(sub)
        nat_scores = [r["judge_mean"] for r in sub if r.get("judge_ok")]
        mean_nat = (sum(nat_scores) / len(nat_scores)) if nat_scores else float("nan")
        print(f"{cond:10s} {mean_f1:>9.3f} {mean_tools:>11.1f} "
              f"{mean_dur:>12.1f} {mean_nat:>9.3f}")


if __name__ == "__main__":
    main()
