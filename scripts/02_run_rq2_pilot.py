"""RQ2 pilot smoke test - run one scenario through both conditions.

Usage:
    # With a real LLM (set in .env):
    python scripts/02_run_rq2_pilot.py

    # With a deterministic mock (no LLM required):
    python scripts/02_run_rq2_pilot.py --mock

Outputs two transcripts under ``outputs/rq2_pilot/`` for the same
scenario, one per condition. Use them to eyeball whether the
multi-agent dialogue and the single-LLM long-context dialogue look
materially different at this scale.

The full pilot (5 scenarios x 2 conditions = 10 dialogues) is driven by
the ``make rq2-*`` targets (run, judge, aggregate).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from orggraph.agents.agent import OpenAIChatClient, PersonaAgent, TextChatClient
from orggraph.agents.persona import load_personas_from_csv
from orggraph.agents.samples import load_sample_messages
from orggraph.config import OUTPUT_DIR
from orggraph.simulation.runner import run_multi_agent
from orggraph.simulation.scenario import SAMPLE_SCENARIOS, Scenario
from orggraph.simulation.single_llm import run_single_llm

DEFAULT_PERSONAS_CSV = OUTPUT_DIR / "person_enrichment.csv"
DEFAULT_PARQUET = OUTPUT_DIR / "clean_emails.parquet"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "outputs/rq2_pilot"


class _MockClient:
    """Deterministic mock used when --mock is passed. Loops through a
    short script of turn-templated replies so the smoke test produces
    meaningful-looking transcripts without an LLM running."""

    _SCRIPT = [
        "Let me get the confirmations onto Friday's agenda - can you pull a "
        "summary of the three weeks of backlog?",
        "Yes, I can have a summary by Wednesday EOD. Should it cover only "
        "the counterparty deal or every backlogged confirmation?",
        "Both, separated by counterparty. The risk committee will want the "
        "regulatory uncertainty isolated.",
        "Understood. I'll flag any items that touch FERC reporting in red "
        "and circulate a draft for your review by Thursday morning.",
        "Good. Loop in legal if any item needs counsel sign-off - Sara, you "
        "make the call there.",
        "Will do. I'll start the review tonight and forward anything "
        "ambiguous tomorrow.",
        "END",
    ]

    def __init__(self) -> None:
        self._idx = 0

    def chat(self, *, system, messages, model, temperature) -> str:
        out = self._SCRIPT[min(self._idx, len(self._SCRIPT) - 1)]
        self._idx += 1
        return out


def _build_real_client() -> TextChatClient:
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"
    if not base_url:
        raise SystemExit(
            "Set LLM_BASE_URL (or OPENAI_BASE_URL) in the environment, "
            "or pass --mock for a deterministic smoke test."
        )
    return OpenAIChatClient(base_url=base_url, api_key=api_key)


def _run_one(scenario: Scenario, personas, args, client, registry=None) -> None:
    """Run pilot conditions for one scenario and write transcripts.

    Conditions:
      - multi_agent (always): persona-only A2A
      - multi_agent_tools (if registry is not None): A2A + KG tool registry
      - single_llm (always): single-LLM long-context baseline
    """
    print(f"\n=== Scenario: {scenario.name} ({scenario.flow}) ===")
    print(f"Participants: {', '.join(scenario.participants)}")
    print(f"Brief: {scenario.brief[:120]}{'…' if len(scenario.brief) > 120 else ''}")
    print(f"Model: {args.model}, temperature: {args.temperature}, "
          f"mode: {'MOCK' if args.mock else 'live'}, "
          f"tools: {'on' if registry is not None else 'off'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- multi-agent condition (persona only) ------------------------
    print("\n--- multi_agent (A2A, persona only) ---")
    agents = {
        n: PersonaAgent(
            persona=personas[n], client=client, model=args.model,
            temperature=args.temperature,
        )
        for n in scenario.participants
    }
    if args.mock:
        for n, agent in agents.items():
            agent.client = _MockClient()
    multi_transcript = run_multi_agent(scenario, agents)
    multi_path = args.out_dir / f"{scenario.name}__multi_agent.jsonl"
    multi_transcript.to_jsonl(multi_path)
    print(f"Wrote {len(multi_transcript)} messages → {multi_path}")
    for m in multi_transcript.messages:
        print(f"  [{m.turn_id}] {m.sender}: {m.body[:80]}{'…' if len(m.body) > 80 else ''}")

    # --- multi-agent + tools condition -------------------------------
    if registry is not None:
        from orggraph.pipeline.agents.tools_logging import ToolCallLog

        print("\n--- multi_agent_tools (A2A + KG tool registry) ---")
        log = ToolCallLog()
        agents_tools = {
            n: PersonaAgent(
                persona=personas[n], client=client, model=args.model,
                temperature=args.temperature,
                tools=registry, tool_call_log=log,
            )
            for n in scenario.participants
        }
        tools_transcript = run_multi_agent(
            scenario, agents_tools, condition="multi_agent_tools"
        )
        tools_path = args.out_dir / f"{scenario.name}__multi_agent_tools.jsonl"
        tools_transcript.to_jsonl(tools_path)
        # Persist the ToolCallLog (even if empty) so the run is auditable.
        # Empty log = the model never decided to call any registered tool;
        # this is itself diagnostic information.
        log_path = args.out_dir / f"{scenario.name}__multi_agent_tools__tool_calls.jsonl"
        log.to_jsonl(log_path)
        print(f"Wrote {len(tools_transcript)} messages → {tools_path}; "
              f"{len(log.entries)} tool calls logged → {log_path.name}")
        for m in tools_transcript.messages:
            print(f"  [{m.turn_id}] {m.sender}: {m.body[:80]}{'…' if len(m.body) > 80 else ''}")

    # --- single-LLM baseline -----------------------------------------
    print("\n--- single_llm long-context ---")
    single_client = _MockClient() if args.mock else client
    single_transcript = run_single_llm(
        scenario, personas, single_client,
        model=args.model, temperature=args.temperature,
    )
    single_path = args.out_dir / f"{scenario.name}__single_llm.jsonl"
    single_transcript.to_jsonl(single_path)
    print(f"Wrote {len(single_transcript)} messages → {single_path}")
    for m in single_transcript.messages:
        print(f"  [{m.turn_id}] {m.sender}: {m.body[:80]}{'…' if len(m.body) > 80 else ''}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--scenario",
        default="activity_coordination_q4_review",
        help="Scenario name to run (ignored if --all is passed)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every scenario in SAMPLE_SCENARIOS sequentially (overrides --scenario).",
    )
    parser.add_argument(
        "--personas-csv",
        type=Path,
        default=DEFAULT_PERSONAS_CSV,
        help=f"Persona CSV path (default: {DEFAULT_PERSONAS_CSV})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output dir (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", "gemma3:4b"),
        help="Model identifier (default: $LLM_MODEL or gemma3:4b)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a deterministic mock instead of a real LLM (no env required).",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Skip few-shot writing-sample loading (smaller prompts, no style grounding).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Few-shot writing samples per persona (default: 3).",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help=f"clean_emails.parquet path for sample loading (default: {DEFAULT_PARQUET})",
    )
    parser.add_argument(
        "--with-tools",
        action="store_true",
        help="Also run a 'multi_agent_tools' condition that attaches the KG "
             "tool registry to each PersonaAgent. Requires Neo4j reachable "
             "via NEO4J_BOLT_PORT (default 7687); without --with-tools, only "
             "the persona-only multi_agent and single_llm conditions run.",
    )
    args = parser.parse_args(argv)

    # Resolve scenario(s)
    if args.all:
        scenarios = list(SAMPLE_SCENARIOS.values())
    else:
        if args.scenario not in SAMPLE_SCENARIOS:
            raise SystemExit(
                f"Unknown scenario {args.scenario!r}. "
                f"Available: {list(SAMPLE_SCENARIOS)}"
            )
        scenarios = [SAMPLE_SCENARIOS[args.scenario]]

    # Resolve personas (load once, reused across scenarios)
    personas = load_personas_from_csv(args.personas_csv)
    all_participants = {p for s in scenarios for p in s.participants}
    missing = [n for n in all_participants if n not in personas]
    if missing:
        raise SystemExit(
            f"Personas missing from {args.personas_csv}: {missing}. "
            f"Run Stage 4a first or drop the affected scenarios."
        )

    client: TextChatClient = _MockClient() if args.mock else _build_real_client()

    # Build the KG tool registry once if --with-tools is on. Connects to
    # Neo4j directly (the dashboard's get_driver() is Streamlit-cached and
    # not safe to call from a CLI). Falls back to None (and prints a
    # warning) if Neo4j isn't reachable, so the persona-only conditions
    # still run.
    registry = None
    if args.with_tools and not args.mock:
        try:
            from neo4j import GraphDatabase
            from orggraph.pipeline.agents.tools import build_default_registry

            bolt_port = os.environ.get("NEO4J_BOLT_PORT", "7687")
            user = os.environ.get("NEO4J_USER", "neo4j")
            password = os.environ.get("NEO4J_PASSWORD", "orggraph2026")
            driver = GraphDatabase.driver(
                f"bolt://localhost:{bolt_port}", auth=(user, password)
            )
            with driver.session() as s:
                s.run("RETURN 1").consume()
            registry = build_default_registry(driver)
            tool_names = [t.name for t in registry]
            print(f"\nKG tool registry attached: {len(tool_names)} tools "
                  f"({', '.join(tool_names[:6])}{'…' if len(tool_names) > 6 else ''})")
        except Exception as e:  # noqa: BLE001
            print(f"\n[warn] --with-tools requested but Neo4j unreachable: {e}")
            print("[warn] Falling back to persona-only multi-agent.")
            registry = None

    for scenario in scenarios:
        # Attach few-shot writing samples per scenario's participants
        scenario_personas = dict(personas)
        if not args.no_samples and not args.mock:
            attached = 0
            for name in scenario.participants:
                samples = load_sample_messages(
                    name, parquet_path=args.parquet, n_samples=args.n_samples,
                )
                if samples:
                    scenario_personas[name] = scenario_personas[name].with_samples(samples)
                    attached += 1
            print(f"\nScenario {scenario.name}: attached writing samples to "
                  f"{attached}/{len(scenario.participants)} personas "
                  f"(n_samples={args.n_samples})")

        _run_one(scenario, scenario_personas, args, client, registry=registry)

    print(f"\nDone. {len(scenarios)} scenario(s) x 2 conditions in {args.out_dir}/. "
          f"Run scripts/03_judge_rq2_pilot.py --all next.")


if __name__ == "__main__":
    main()
