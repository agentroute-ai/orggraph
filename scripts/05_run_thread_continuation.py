"""Thread-continuation experiment for RQ2.

Pulls a real Enron email thread, holds out the last email, asks both
the multi-agent A2A and single-LLM long-context conditions to predict
it, then compares the predictions to the ground truth using cheap
textual overlap and an LLM judge.

Usage:
    LLM_BASE_URL=http://localhost:8000/v1 \
    LLM_API_KEY=anything \
    LLM_MODEL=cyankiwi/MiniMax-M2.7-AWQ-4bit \
    python scripts/05_run_thread_continuation.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from orggraph.agents.agent import OpenAIChatClient, TextChatClient
from orggraph.agents.kg_context import load_kg_context
from orggraph.agents.persona import load_personas_from_csv
from orggraph.agents.samples import load_pair_samples, load_sample_messages
from orggraph.config import OUTPUT_DIR
from orggraph.evaluation.continuation_compare import (
    CONTINUATION_DIMENSIONS,
    ContinuationVerdict,
    judge_continuation_match,
    textual_overlap,
)
from orggraph.simulation.thread_continuation import (
    load_thread,
    predict_next_multi_agent,
    predict_next_single_llm,
)
from orggraph.simulation.transcript import Message

DEFAULT_PARQUET = OUTPUT_DIR / "clean_emails.parquet"
DEFAULT_PERSONAS = OUTPUT_DIR / "person_enrichment.csv"
DEFAULT_PROMPTS_DIR = OUTPUT_DIR / "persona_prompts"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "outputs/rq2_thread_continuation"

# Tonight's chosen thread - selected from 212 candidates because it
# has 6 emails, no forwarded content, late-Enron-panic stakes, and
# both senders are in person_enrichment.csv.
DEFAULT_PARTICIPANTS = ("Tracy Geaccone", "Rod Hayslett")
DEFAULT_SUBJECT = "five day rolling forecast"


def _build_client() -> TextChatClient:
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"
    if not base_url:
        raise SystemExit("Set LLM_BASE_URL (or OPENAI_BASE_URL).")
    return OpenAIChatClient(base_url=base_url, api_key=api_key)


def _format_message(label: str, msg: Message, max_lines: int = 0) -> str:
    body = msg.body
    if max_lines:
        lines = body.splitlines()
        if len(lines) > max_lines:
            body = "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} more lines]"
    return f"--- {label} ({msg.sender} → {', '.join(msg.recipients)}, {len(msg.body)} chars) ---\n{body}"


def _comparison_table(
    overlaps: dict[str, dict],
    verdicts: dict[str, ContinuationVerdict],
) -> str:
    conditions = list(overlaps)
    header = f"{'metric':<25}" + "".join(f"{c:>16}" for c in conditions)
    lines = [header, "-" * len(header)]

    # textual overlap rows
    for key, label in [
        ("jaccard", "jaccard"),
        ("shared_tokens", "shared tokens"),
        ("length_ratio", "length ratio"),
        ("generated_chars", "generated chars"),
    ]:
        cells = "".join(f"{overlaps[c][key]:>16}" for c in conditions)
        lines.append(f"{label:<25}{cells}")
    lines.append("")

    # judge dimension rows
    for dim in CONTINUATION_DIMENSIONS:
        parts = []
        for c in conditions:
            v = verdicts.get(c)
            if v and dim in v.scores:
                parts.append(f"{v.scores[dim].score}/5".rjust(16))
            else:
                parts.append("-".rjust(16))
        lines.append(f"  judge: {dim:<18}{''.join(parts)}")
    lines.append("-" * len(header))
    means = "".join(f"{verdicts[c].mean_score():>15.2f}" for c in conditions)
    lines.append(f"{'judge mean':<25}{means}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--personas-csv", type=Path, default=DEFAULT_PERSONAS)
    p.add_argument("--participants", nargs=2, default=list(DEFAULT_PARTICIPANTS),
                   help="Two canonical sender names.")
    p.add_argument("--subject", default=DEFAULT_SUBJECT,
                   help="Normalised subject (lowercase, no Re:/Fw:).")
    p.add_argument("--prefix-len", type=int, default=None,
                   help="How many emails to feed as prefix. Default = N-1 (predict the last).")
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", "gemma3:4b"))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--judge-model",
                   default=os.environ.get("LLM_MODEL", "gemma3:4b"),
                   help="Judge model (defaults to same as agents).")
    p.add_argument("--no-samples", action="store_true",
                   help="Skip few-shot writing-sample loading.")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument(
        "--prompts-dir",
        type=Path,
        default=DEFAULT_PROMPTS_DIR,
        help=("Directory of LLM-prebuilt persona prompts (Stage 4b output). "
              f"Default: {DEFAULT_PROMPTS_DIR}"),
    )
    p.add_argument(
        "--no-prebuilt",
        action="store_true",
        help="Ignore prebuilt prompts; use the runtime template only.",
    )
    p.add_argument("--no-kg", action="store_true",
                   help="Skip Neo4j KG context loading.")
    p.add_argument(
        "--strict-temporal",
        action="store_true",
        default=True,
        help=("Filter writing samples to those sent BEFORE the actual email's "
              "timestamp, AND skip KG context (which is corpus-wide and leaks "
              "future information). Default ON for the continuation experiment."),
    )
    p.add_argument(
        "--allow-temporal-leak",
        dest="strict_temporal",
        action="store_false",
        help="DISABLE temporal hygiene. Use only for diagnostic comparison.",
    )
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "orggraph2026"))
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load thread + personas + samples ----------------------------
    thread = load_thread(args.parquet, tuple(args.participants), args.subject)
    if len(thread) < 2:
        raise SystemExit(
            f"Thread has {len(thread)} emails; need at least 2. "
            f"Check --participants and --subject (subject must be the *normalised* form)."
        )
    prefix_len = args.prefix_len if args.prefix_len is not None else len(thread) - 1
    if prefix_len < 1 or prefix_len >= len(thread):
        raise SystemExit(
            f"--prefix-len must be in [1, {len(thread) - 1}]; got {prefix_len}"
        )
    prefix = thread[:prefix_len]
    actual = thread[prefix_len]
    print(f"Thread: {args.participants} | subject={args.subject!r} | {len(thread)} emails")
    print(f"Prefix: {prefix_len} emails; predicting email index {prefix_len} "
          f"(real sender: {actual.sender})")

    prompts_dir = None if args.no_prebuilt else args.prompts_dir
    personas = load_personas_from_csv(args.personas_csv, prompts_dir=prompts_dir)
    for n in args.participants:
        if n not in personas:
            raise SystemExit(f"Persona missing for {n!r}; check {args.personas_csv}")
    n_with_prebuilt = sum(
        1 for n in args.participants if personas[n].prebuilt_prompt
    )
    if n_with_prebuilt:
        print(f"Loaded {n_with_prebuilt}/{len(args.participants)} prebuilt prompts "
              f"from {prompts_dir}")
    else:
        print("No prebuilt prompts loaded (using runtime template)")

    cutoff = actual.timestamp if args.strict_temporal else None
    if cutoff:
        print(f"Temporal cutoff: {cutoff} (samples + KG must be <strictly< before this)")

    if not args.no_samples:
        attached_samples = 0
        for n in args.participants:
            other = next(p_ for p_ in args.participants if p_ != n)
            # Try pair-specific samples first (n→other), fall back to general
            # sender samples if the pair is too sparse. Pair samples expose
            # the relationship-conditional register; general samples expose
            # the persona's overall voice.
            pair_samples = load_pair_samples(
                n, other,
                parquet_path=args.parquet,
                n_samples=max(args.n_samples - 1, 1),
                before_date=cutoff,
            )
            general_samples = load_sample_messages(
                n, parquet_path=args.parquet,
                n_samples=args.n_samples,
                before_date=cutoff,
            )
            # Combine: at most n_samples total, pair-specific first
            combined = list(pair_samples) + [s for s in general_samples if s not in pair_samples]
            combined = tuple(combined[: args.n_samples])
            if combined:
                personas[n] = personas[n].with_samples(combined)
                attached_samples += 1
                print(
                    f"  {n}: {len(pair_samples)} pair-specific (→{other}) + "
                    f"{len(combined) - len(pair_samples)} general samples"
                )
        print(
            f"Attached writing samples to {attached_samples}/{len(args.participants)} "
            f"personas (n={args.n_samples}, before_date={cutoff or 'unrestricted'})"
        )

    # KG context: the Neo4j graph is built from the FULL corpus, so any
    # facts pulled from it (REPORTS_TO chains, top collaborators, recurring
    # topics) inherently include hindsight. In strict-temporal mode we
    # therefore skip KG context entirely. Building a date-aware Neo4j
    # snapshot per cutoff would be the principled fix; out of scope tonight.
    if args.no_kg:
        kg_skip_reason = "explicitly disabled (--no-kg)"
    elif args.strict_temporal:
        kg_skip_reason = (
            "skipped: KG is corpus-wide and would leak post-cutoff information. "
            "Use --allow-temporal-leak to include it (diagnostic only)."
        )
    else:
        kg_skip_reason = None

    if kg_skip_reason:
        print(f"KG context: {kg_skip_reason}")
    else:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password),
            )
            try:
                attached = 0
                for n in args.participants:
                    kg = load_kg_context(n, driver)
                    if kg.title or kg.tier is not None or kg.top_collaborators or kg.topics:
                        personas[n] = personas[n].with_kg_context(kg)
                        attached += 1
                print(f"Attached KG context to {attached}/{len(args.participants)} personas")
            finally:
                driver.close()
        except Exception as e:  # noqa: BLE001
            print(f"WARN: could not load KG context ({e}); proceeding without")

    client = _build_client()

    next_sender = actual.sender
    other = next(p_ for p_ in args.participants if p_ != next_sender)

    # ---- Generate both candidates ------------------------------------
    print("\n--- generating multi-agent prediction ---")
    multi = predict_next_multi_agent(
        prefix, next_sender=next_sender, next_recipient=other,
        personas=personas, client=client,
        model=args.model, temperature=args.temperature,
    )
    print(_format_message("multi-agent A2A", multi, max_lines=12))

    print("\n--- generating single-LLM prediction ---")
    single = predict_next_single_llm(
        prefix, next_sender=next_sender, next_recipient=other,
        personas=personas, client=client,
        model=args.model, temperature=args.temperature,
    )
    print(_format_message("single-LLM", single, max_lines=12))

    print("\n--- ACTUAL email (ground truth) ---")
    print(_format_message("REAL", actual, max_lines=12))

    # ---- Compare -----------------------------------------------------
    overlaps = {
        "multi_agent": textual_overlap(multi.body, actual.body),
        "single_llm": textual_overlap(single.body, actual.body),
    }
    verdicts: dict[str, ContinuationVerdict] = {}
    for label, generated in [("multi_agent", multi), ("single_llm", single)]:
        verdicts[label] = judge_continuation_match(
            prefix=prefix, actual=actual, generated=generated,
            client=client, condition=label,
            model=args.judge_model, temperature=0.0,
        )

    print("\n" + "=" * 60)
    print("Comparison vs ground truth")
    print("=" * 60)
    print(_comparison_table(overlaps, verdicts))

    # ---- Persist -----------------------------------------------------
    record = {
        "participants": list(args.participants),
        "subject": args.subject,
        "prefix_len": prefix_len,
        "actual": {"sender": actual.sender, "body": actual.body, "chars": len(actual.body)},
        "multi_agent": {
            "body": multi.body, "overlap": overlaps["multi_agent"],
            "verdict": verdicts["multi_agent"].to_dict(),
        },
        "single_llm": {
            "body": single.body, "overlap": overlaps["single_llm"],
            "verdict": verdicts["single_llm"].to_dict(),
        },
        "model": args.model,
        "judge_model": args.judge_model,
        "temperature": args.temperature,
    }
    pair_slug = "_".join(sorted(args.participants)).replace(" ", "_")
    out_path = args.out_dir / f"{pair_slug}__{args.subject.replace(' ', '_')[:40]}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\n→ verdict saved to {out_path}")


if __name__ == "__main__":
    main()
