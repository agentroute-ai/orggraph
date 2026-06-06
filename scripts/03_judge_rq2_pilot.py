"""Run the LLM-as-judge over the latest RQ2 pilot transcripts.

Usage:
    LLM_BASE_URL=http://localhost:8000/v1 \
    LLM_API_KEY=anything \
    LLM_MODEL=cyankiwi/MiniMax-M2.7-AWQ-4bit \
    python scripts/03_judge_rq2_pilot.py

By default scores both ``__multi_agent.jsonl`` and ``__single_llm.jsonl``
files for the named scenario, prints a side-by-side comparison, and
writes the verdicts to ``outputs/rq2_pilot/judge/<scenario>__verdict.json``.

Use ``--judge-model`` to use a separate model from the agents (e.g.
score multi-agent runs that used model A with judge model B for
self-preference-bias avoidance). Default is to use the same model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from orggraph.agents.agent import OpenAIChatClient, TextChatClient
from orggraph.agents.persona import load_personas_from_csv
from orggraph.config import OUTPUT_DIR
from orggraph.evaluation.dialogue_judge import (
    DIMENSIONS,
    JudgeError,
    JudgeResult,
    judge_transcript,
)
from orggraph.simulation.scenario import SAMPLE_SCENARIOS
from orggraph.simulation.transcript import Transcript

DEFAULT_PERSONAS_CSV = OUTPUT_DIR / "person_enrichment.csv"
DEFAULT_PILOT_DIR = Path(__file__).resolve().parents[1] / "outputs/rq2_pilot"


def _build_client() -> TextChatClient:
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"
    if not base_url:
        raise SystemExit("Set LLM_BASE_URL (or OPENAI_BASE_URL) in the environment.")
    return OpenAIChatClient(base_url=base_url, api_key=api_key)


def _format_verdict(result: JudgeResult) -> str:
    lines = [
        f"  scenario: {result.scenario_name}",
        f"  condition: {result.condition}",
        f"  mean score: {result.mean_score():.2f} / 5",
        "",
        "  scores:",
    ]
    for dim in DIMENSIONS:
        s = result.scores.get(dim)
        if not s:
            lines.append(f"    {dim:25s} -")
            continue
        lines.append(f"    {dim:25s} {s.score}/5  ({s.justification})")
    if result.turn_flags:
        lines.append("")
        lines.append("  per-turn issues:")
        for f in result.turn_flags:
            lines.append(f"    [turn {f.turn_id}] {f.issue}: {f.detail}")
    if result.overall_summary:
        lines.append("")
        lines.append(f"  overall: {result.overall_summary}")
    return "\n".join(lines)


def _comparison_table(verdicts: dict[str, JudgeResult]) -> str:
    if not verdicts:
        return "(no verdicts to compare)"
    conditions = list(verdicts)
    header = f"{'dimension':<27}" + "".join(f"{c:>14}" for c in conditions)
    rows = [header, "-" * len(header)]
    for dim in DIMENSIONS:
        cells = []
        for c in conditions:
            s = verdicts[c].scores.get(dim)
            cells.append(f"{s.score}/5" if s else "-")
        rows.append(f"{dim:<27}" + "".join(f"{c:>14}" for c in cells))
    rows.append("-" * len(header))
    rows.append(
        f"{'mean':<27}"
        + "".join(f"{verdicts[c].mean_score():>13.2f}" for c in conditions)
    )
    rows.append(
        f"{'turn flags':<27}"
        + "".join(f"{len(verdicts[c].turn_flags):>14}" for c in conditions)
    )
    return "\n".join(rows)


def _judge_one(scenario, personas, args, client, out_dir) -> dict:
    """Judge both conditions for one scenario; return verdicts dict."""
    print(f"\nScenario: {scenario.name}")
    print(f"Participants: {', '.join(scenario.participants)}")

    missing = [n for n in scenario.participants if n not in personas]
    if missing:
        print(f"  [skip] personas missing from CSV: {missing}")
        return {}

    verdicts: dict[str, JudgeResult] = {}
    for cond_label, suffix in (("multi_agent", "__multi_agent.jsonl"),
                                ("multi_agent_tools", "__multi_agent_tools.jsonl"),
                                ("single_llm", "__single_llm.jsonl")):
        path = args.pilot_dir / f"{scenario.name}{suffix}"
        if not path.exists():
            print(f"  [skip] {cond_label}: {path.name} not found")
            continue
        verdict_path = out_dir / f"{scenario.name}__{cond_label}__verdict.json"
        if args.skip_existing and verdict_path.exists():
            print(f"  [skip] {cond_label}: verdict already on disk ({verdict_path.name})")
            continue
        print(f"  judging {cond_label}: {path.name}")
        transcript = Transcript.from_jsonl(path)
        try:
            result = judge_transcript(
                transcript,
                scenario_brief=scenario.brief,
                participants=list(scenario.participants),
                personas=personas,
                client=client,
                model=args.judge_model,
                temperature=args.temperature,
            )
        except JudgeError as exc:
            print(f"    [error] {cond_label}: {exc}")
            print("    [error] continuing with next transcript")
            continue
        verdicts[cond_label] = result
        verdict_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"    mean={result.mean_score():.2f}/5, "
              f"flags={len(result.turn_flags)} → {verdict_path.name}")
    return verdicts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--scenario",
        default="activity_coordination_q4_review",
        help="Scenario slug; transcripts must already exist under --pilot-dir (ignored if --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Judge every scenario in SAMPLE_SCENARIOS that has transcripts on disk.",
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=DEFAULT_PILOT_DIR,
        help=f"Where transcripts live (default: {DEFAULT_PILOT_DIR})",
    )
    parser.add_argument(
        "--personas-csv",
        type=Path,
        default=DEFAULT_PERSONAS_CSV,
        help=f"Persona CSV (default: {DEFAULT_PERSONAS_CSV})",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("LLM_MODEL", "gemma3:4b"),
        help="Model to use as judge (default: $LLM_MODEL or gemma3:4b)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature (default: 0.0 for reproducibility)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write verdict JSONs (default: <pilot-dir>/judge)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip transcripts whose verdict JSON already exists on disk.",
    )
    args = parser.parse_args(argv)

    if args.all:
        scenarios = list(SAMPLE_SCENARIOS.values())
    else:
        if args.scenario not in SAMPLE_SCENARIOS:
            raise SystemExit(
                f"Unknown scenario {args.scenario!r}. Available: {list(SAMPLE_SCENARIOS)}"
            )
        scenarios = [SAMPLE_SCENARIOS[args.scenario]]

    out_dir = args.out_dir or (args.pilot_dir / "judge")
    out_dir.mkdir(parents=True, exist_ok=True)

    personas = load_personas_from_csv(args.personas_csv)
    client = _build_client()
    print(f"Judge model: {args.judge_model}, temperature: {args.temperature}")

    all_verdicts: dict[str, dict[str, JudgeResult]] = {}
    for scenario in scenarios:
        verdicts = _judge_one(scenario, personas, args, client, out_dir)
        if verdicts:
            all_verdicts[scenario.name] = verdicts

    if len(scenarios) == 1 and all_verdicts:
        single = next(iter(all_verdicts.values()))
        if len(single) >= 2:
            print("\n" + "=" * 60)
            print("Side-by-side comparison")
            print("=" * 60)
            print(_comparison_table(single))
    else:
        n_judged = sum(len(v) for v in all_verdicts.values())
        print(f"\nDone. Judged {n_judged} transcripts across "
              f"{len(all_verdicts)} scenarios. "
              f"Run scripts/04_aggregate_rq2_pilot.py next for the comparison table.")


if __name__ == "__main__":
    main()
