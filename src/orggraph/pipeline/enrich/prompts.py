"""Stage 4b — Per-Person system-prompt build.

Reads the Stage 4a ``person_enrichment.csv`` and, for each Person,
makes a single LLM call that synthesises a narrative system prompt
from the enrichment fields plus a stratified set of real writing
samples (and optionally the KG position from Stage 6's Neo4j sync).

The resulting prompts are written one-per-file to
``<output-dir>/persona_prompts/<slug>.txt`` and can be loaded at
runtime by :func:`orggraph.agents.persona.load_personas_from_csv`
with ``prompts_dir=...``.

Why this is a separate stage rather than runtime work:

* Each prompt costs one LLM call (~5–10 s on the local vLLM endpoint).
  Pre-baking 140 of them is ~25 minutes; doing it per-simulation-run
  multiplies the cost N times for no benefit since the prompts are
  stable for a given persona.
* The artefacts are inspectable and version-controllable. A reviewer
  can read each persona's prompt and form an opinion about whether the
  agent's voice is grounded; templated prompts buried in code are not
  reviewable in the same way.
* Edits compound. If you find a phrasing that helps Sara's prompt,
  you can hand-edit her file once; next simulation run picks it up.

Resume-safe: skips any persona whose prompt file already exists.
Use ``--force`` to rebuild.

Temporal-hygiene caveat: the prompt is built using corpus-wide samples
and KG. For *strict-temporal* continuation experiments, regenerate
the prompts with a date cutoff, or fall back to the runtime template.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time
from pathlib import Path

from orggraph.agents.kg_context import KGContext, load_kg_context
from orggraph.agents.persona import load_personas_from_csv
from orggraph.agents.prompt_builder import (
    CandidateRound,
    _slug,
    build_persona_prompt_n_best,
)
from orggraph.agents.prompt_judge import DIMENSIONS
from orggraph.agents.samples import load_sample_messages
from orggraph.config import OUTPUT_DIR

DEFAULT_PERSONAS_CSV = OUTPUT_DIR / "person_enrichment.csv"
DEFAULT_PARQUET = OUTPUT_DIR / "clean_emails.parquet"
DEFAULT_OUT_DIR = OUTPUT_DIR / "persona_prompts"

QUALITY_THRESHOLD = 3

QUALITY_CSV_FIELDS = (
    "name",
    "slug",
    "n_candidates",
    "winner_index",
    *DIMENSIONS,
    "total",
    "all_scores_json",
    "judge_model",
    "builder_model",
    "timestamp",
)


def _save_prompt_with_gate(
    *,
    out_dir: Path,
    slug: str,
    winner: CandidateRound,
    all_candidates: list[CandidateRound],
    threshold: int = QUALITY_THRESHOLD,
) -> Path:
    """Write the winning prompt. Pass→main dir; fail→_failed/.

    Returns the path actually written.
    """
    if winner.scores.all_pass(threshold):
        target = out_dir / f"{slug}.txt"
        target.write_text(winner.text)
        return target

    failed_dir = out_dir / "_failed"
    failed_dir.mkdir(exist_ok=True)
    target = failed_dir / f"{slug}.txt"
    target.write_text(winner.text)
    (failed_dir / f"{slug}.scores.json").write_text(
        json.dumps(
            {
                "winner": winner.scores.to_dict(),
                "all_candidates": [c.scores.to_dict() for c in all_candidates],
                "threshold": threshold,
            },
            indent=2,
        )
    )
    return target


def _append_quality_row(
    csv_path: Path,
    *,
    name: str,
    slug: str,
    candidates: list[CandidateRound],
    winner_index: int,
    judge_model: str,
    builder_model: str,
) -> None:
    """Append one row to prompt_quality_scores.csv. Writes header on first row."""
    winner = candidates[winner_index]
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUALITY_CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "name": name,
            "slug": slug,
            "n_candidates": len(candidates),
            "winner_index": winner_index,
            **winner.scores.to_dict(),
            "total": winner.scores.total(),
            "all_scores_json": json.dumps(
                [c.scores.to_dict() for c in candidates],
                separators=(",", ":"),
            ),
            "judge_model": judge_model,
            "builder_model": builder_model,
            "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        })


def _build_chat_client():
    """Lazily import to keep tests import-light."""
    from orggraph.agents.agent import OpenAIChatClient

    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"
    if not base_url:
        raise SystemExit(
            "Set LLM_BASE_URL (or OPENAI_BASE_URL) to a vLLM/Ollama endpoint."
        )
    return OpenAIChatClient(base_url=base_url, api_key=api_key)


def run(
    personas_csv: Path,
    parquet_path: Path,
    out_dir: Path,
    *,
    model: str,
    judge_model: str | None = None,
    temperature: float = 0.6,
    judge_temperature: float = 0.0,
    n_best: int = 3,
    n_samples: int = 5,
    variant_name: str = "v1_freeform",
    only_names: list[str] | None = None,
    force: bool = False,
    use_kg: bool = True,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    quality_csv: Path | None = None,
) -> dict:
    """Run Stage 4b with the Tier 1 quality gate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    quality_csv = quality_csv or (out_dir.parent / "prompt_quality_scores.csv")
    judge_model = judge_model or model

    personas = load_personas_from_csv(personas_csv)
    if only_names:
        personas = {n: personas[n] for n in only_names if n in personas}
        missing = [n for n in only_names if n not in personas]
        if missing:
            print(f"WARN: requested personas not in {personas_csv}: {missing}")
    if not personas:
        raise SystemExit(
            f"No personas to build prompts for in {personas_csv} "
            f"(filter={only_names!r})"
        )

    builder_client = _build_chat_client()
    judge_client = builder_client  # same endpoint, different system prompt

    driver = None
    if use_kg:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                auth=(
                    neo4j_user or os.environ.get("NEO4J_USER", "neo4j"),
                    neo4j_password or os.environ.get("NEO4J_PASSWORD", "orggraph2026"),
                ),
            )
        except Exception as e:  # noqa: BLE001
            print(f"WARN: Neo4j unreachable ({e}); building prompts without KG context")
            driver = None

    summary = {
        "built": 0, "skipped": 0, "failed_gate": 0, "errored": 0,
        "model": model, "judge_model": judge_model, "n_best": n_best,
    }
    try:
        for name, persona in personas.items():
            slug = _slug(name)
            target_main = out_dir / f"{slug}.txt"
            target_failed = out_dir / "_failed" / f"{slug}.txt"
            if (target_main.exists() or target_failed.exists()) and not force:
                summary["skipped"] += 1
                continue

            samples = load_sample_messages(
                name, parquet_path=parquet_path, n_samples=n_samples,
            )
            kg = (
                load_kg_context(name, driver) if driver is not None
                else KGContext(name=name)
            )

            t0 = time.time()
            try:
                winner, all_cands = build_persona_prompt_n_best(
                    persona=persona, samples=samples, kg=kg,
                    builder_client=builder_client,
                    judge_client=judge_client,
                    model=model, judge_model=judge_model,
                    n_best=n_best,
                    builder_temperature=temperature,
                    judge_temperature=judge_temperature,
                    variant_name=variant_name,
                )
            except Exception as e:  # noqa: BLE001
                summary["errored"] += 1
                print(f"  ERR   {name}: {e}")
                continue

            written = _save_prompt_with_gate(
                out_dir=out_dir, slug=slug,
                winner=winner, all_candidates=all_cands,
            )
            winner_index = next(
                i for i, c in enumerate(all_cands) if c is winner
            )
            _append_quality_row(
                quality_csv, name=name, slug=slug,
                candidates=all_cands, winner_index=winner_index,
                judge_model=judge_model, builder_model=model,
            )

            if winner.scores.all_pass(QUALITY_THRESHOLD):
                summary["built"] += 1
                tag = "OK   "
            else:
                summary["failed_gate"] += 1
                tag = "FAIL "
            print(
                f"  {tag} {name:<30s}  total={winner.scores.total():>2d}  "
                f"min={winner.scores.min_score()}  "
                f"{time.time() - t0:.1f}s → {written.name}"
            )
    finally:
        if driver is not None:
            driver.close()

    print(
        f"\nDone. built={summary['built']} failed_gate={summary['failed_gate']} "
        f"errored={summary['errored']} skipped={summary['skipped']} → {out_dir}"
    )
    print(f"Scores: {quality_csv}")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 4b — build per-Person system prompts with N-best + judge gate.")
    p.add_argument("--personas-csv", type=Path, default=DEFAULT_PERSONAS_CSV)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit"),
        help="Builder model id (defaults to $LLM_MODEL).",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Judge model id (defaults to --model).",
    )
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Builder temperature for diversity within N-best.")
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--n-best", type=int, default=3,
                   help="Number of candidates generated per persona.")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument(
        "--variant", default="v1_freeform",
        choices=("v1_freeform", "v2_sections", "v3_idcard"),
        help="Builder variant from BUILDER_VARIANTS (run structure search to pick).",
    )
    p.add_argument(
        "--names", default=None,
        help="Comma-separated subset of canonical names. Default: all personas.",
    )
    p.add_argument("--force", action="store_true",
                   help="Rebuild prompts even if they exist on disk.")
    p.add_argument("--no-kg", action="store_true",
                   help="Skip Neo4j KG context.")
    p.add_argument("--quality-csv", type=Path, default=None,
                   help="Override scores CSV path.")
    args = p.parse_args(argv)

    only_names = (
        [n.strip() for n in args.names.split(",") if n.strip()]
        if args.names else None
    )
    run(
        personas_csv=args.personas_csv,
        parquet_path=args.parquet,
        out_dir=args.out_dir,
        model=args.model,
        judge_model=args.judge_model,
        temperature=args.temperature,
        judge_temperature=args.judge_temperature,
        n_best=args.n_best,
        n_samples=args.n_samples,
        variant_name=args.variant,
        only_names=only_names,
        force=args.force,
        use_kg=not args.no_kg,
        quality_csv=args.quality_csv,
    )


if __name__ == "__main__":
    main()
