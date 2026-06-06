"""Persona-grounded Q&A demo against the OrgGraph chat tools.

For each (persona, question) pair, swaps the chat's company-wide system
prompt for the named employee's persona prompt (with a short append
telling them they have the tools), runs the same tool-calling loop the
KG-Chat eval uses, and prints a full transcript.

Output: outputs/eval/kg_chat/persona_sim/<timestamp>/transcripts.md
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT.parents[1] / "datasets" / "enron" / "processed" / "persona_prompts"

from neo4j import GraphDatabase
from orggraph.agents.agent import OpenAIChatClient
from orggraph.pipeline.agents.tools import build_default_registry
from orggraph.pipeline.eval.kg_chat.dataset import Question
from orggraph.pipeline.eval.kg_chat.runner import run_one


PERSONA_SUFFIX = """

## Tools you have access to

You have the OrgGraph tool catalog (get_person, find_person, get_org_chart,
find_topic_collaborators, get_thread_history, get_email_content, get_thread,
get_emails_bulk, search_emails_semantic, get_pair_signals,
find_emails_with_speech_act, get_centrality, get_dominance_score,
run_cypher, run_sql). Use them when you need a fact you don't already
know. Answer in your own voice, the way you would write to a colleague.
Cite specific people or emails when relevant; do not fabricate.
"""


PAIRS = [
    {
        "persona_file": "jeff_dasovich.txt",
        "question": (
            "What's the most pressing California regulatory matter we should "
            "be tracking right now, and who else needs to be in the loop?"
        ),
    },
    {
        "persona_file": "sara_shackleton.txt",
        "question": (
            "A new counterparty wants to start trading with us. Walk me "
            "through what we need to put an ISDA Master Agreement in place "
            "for them - who handles which piece, and how long does it "
            "usually take?"
        ),
    },
]


def load_persona(persona_file: str) -> str:
    p = PERSONAS_DIR / persona_file
    body = p.read_text()
    return body + PERSONA_SUFFIX


def main() -> int:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "eval" / "kg_chat" / f"persona_sim_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "orggraph2026"),
        ),
    )
    registry = build_default_registry(driver)
    client = OpenAIChatClient(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "anything"),
    )
    model = os.environ.get("EVAL_LLM_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit")

    transcripts: list[str] = ["# Persona Q&A demo", ""]

    for pair in PAIRS:
        persona_name = pair["persona_file"].replace(".txt", "").replace("_", " ").title()
        question = pair["question"]
        system_prompt = load_persona(pair["persona_file"])

        print(f"\n=== {persona_name} ===")
        print(f"Q: {question}\n")

        # Reuse the same runner; we don't grade, just capture.
        q = Question(
            id=pair["persona_file"].split(".")[0],
            category="persona_qa",
            question=question,
            expected_tools=tuple(),
            grading={"kind": "judge"},  # unused
        )
        result = run_one(
            q,
            registry=registry,
            client=client,
            model=model,
            system_prompt=system_prompt,
            max_tools=12,
        )

        # Print + capture transcript
        print(f"[{len(result.tool_calls)} tool calls, {result.wall_time_ms} ms]")
        for i, tc in enumerate(result.tool_calls, 1):
            err = f" ERROR={tc.error}" if tc.error else ""
            print(f"  {i}. {tc.name}({json.dumps(tc.args)[:100]}){err}")
        print("\nFinal answer:")
        print(result.final_body)
        print("=" * 70)

        transcripts.append(f"## {persona_name} answers")
        transcripts.append("")
        transcripts.append(f"**Question:** {question}")
        transcripts.append("")
        transcripts.append(f"**Tools called:** {len(result.tool_calls)} · wall time: {result.wall_time_ms} ms")
        transcripts.append("")
        for i, tc in enumerate(result.tool_calls, 1):
            err = f" 🛑 `{tc.error}`" if tc.error else ""
            transcripts.append(f"{i}. `{tc.name}({json.dumps(tc.args)})`{err}")
            transcripts.append(f"   - result: `{tc.result_summary[:160]}`")
        transcripts.append("")
        transcripts.append("**Answer (in persona voice):**")
        transcripts.append("")
        transcripts.append(result.final_body or "_(no body)_")
        transcripts.append("")
        transcripts.append("---")
        transcripts.append("")

    (out_dir / "transcripts.md").write_text("\n".join(transcripts))
    print(f"\nWrote transcripts to {out_dir / 'transcripts.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
