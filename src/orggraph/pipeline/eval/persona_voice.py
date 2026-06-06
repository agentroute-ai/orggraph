"""Tier 3 — voice fidelity validation for persona prompts.

Holds out 5 real emails per persona, generates persona-vs-control
replies, runs 3-way attribution + a deterministic stylometric panel.
Design at docs/plans/2026-05-07-persona-prompt-quality-design.md §3.2.
"""

from __future__ import annotations

import argparse as _argparse
import ast as _ast
import json as _json
import os as _os
import random as _random
import re as _re
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from orggraph.agents.agent import TextChatClient
from orggraph.config import OUTPUT_DIR
from orggraph.evaluation.voice_metrics import (
    blackberry_tag_rate,
    char_bigram_jaccard,
    mean_sentence_length,
    salutation_pattern,
    signoff_pattern,
)

# Hierarchy tier bands. Senior = tier 1-2, mid = tier 3-4, IC = tier 5+.
_BANDS = (
    ("senior", lambda t: t in {1, 2}),
    ("mid", lambda t: t in {3, 4}),
    ("ic", lambda t: t >= 5),
)


def _band_for_tier(tier: int) -> str | None:
    for name, pred in _BANDS:
        if pred(tier):
            return name
    return None


def select_stratified_personas(
    person_enrichment_csv: Path,
    n_per_band: int = 5,
) -> list[str]:
    """Pick n_per_band personas from each hierarchy band.

    Within each band:
      1. Sort by n_emails_sampled desc, name asc (deterministic).
      2. Take top n_per_band.
      3. If formality range is collapsed (max - min < 3), swap the
         lowest-email pick for the highest-email persona in an absent
         formality bucket.
    """
    df = pd.read_csv(person_enrichment_csv)
    if "hierarchy_tier" not in df.columns:
        # Fallback: assume tier 4 (mid) for personas without explicit tier
        df["hierarchy_tier"] = 4
    df["band"] = df["hierarchy_tier"].apply(_band_for_tier)
    df = df.dropna(subset=["band"]).sort_values(
        ["n_emails_sampled", "name"], ascending=[False, True]
    )

    selected: list[str] = []
    for band, _ in _BANDS:
        band_df = df[df["band"] == band].reset_index(drop=True)
        if band_df.empty:
            continue
        picks = band_df.head(n_per_band).copy()
        if len(picks) >= 2 and picks["formality"].max() - picks["formality"].min() < 3:
            # Find candidates in absent formality buckets
            picked_names = set(picks["name"])
            outside = band_df[~band_df["name"].isin(picked_names)]
            high = outside[outside["formality"] >= 4].head(1)
            low = outside[outside["formality"] <= 2].head(1)
            swap_ins = [s for s in (high, low) if not s.empty]
            if swap_ins:
                # Replace the lowest-email current picks (one per swap-in).
                # `picks` is already sorted by n_emails_sampled desc within
                # the band, so the tail rows are the lowest-email picks.
                n_drop = min(len(swap_ins), len(picks) - 1)
                picks = picks.iloc[: len(picks) - n_drop]
                picks = pd.concat([picks, *swap_ins[:n_drop]], ignore_index=True)
        selected.extend(picks["name"].tolist())
    return selected


_FORWARD_MARKER = _re.compile(
    r"-{3,}\s*forwarded\s*by\s|---{2,}\s*original\s+message",
    _re.IGNORECASE,
)


def _quoted_fraction(body: str) -> float:
    """Fraction of lines starting with '>' or following a forward marker."""
    lines = body.split("\n")
    if not lines:
        return 0.0
    forwarded_idx = None
    for i, ln in enumerate(lines):
        if _FORWARD_MARKER.search(ln):
            forwarded_idx = i
            break
    quoted = 0
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(">"):
            quoted += 1
        elif forwarded_idx is not None and i >= forwarded_idx:
            quoted += 1
    return quoted / len(lines)


@dataclass(frozen=True)
class HeldoutEmail:
    """One real email held out for voice-fidelity evaluation."""

    message_id: str
    subject: str
    body: str
    date: pd.Timestamp


def select_holdout_emails(
    name: str,
    parquet_path: Path,
    n: int = 5,
) -> list[HeldoutEmail]:
    """Pick the most recent n emails by `name` that pass quality filters.

    Filters:
      - sender_resolved == name
      - body has >= 50 words
      - quoted/forwarded fraction <= 30%
    """
    df = pd.read_parquet(parquet_path)
    sender_col = "sender_resolved" if "sender_resolved" in df.columns else "sender_canonical"
    body_col = "body_truncated" if "body_truncated" in df.columns else "body"
    id_col = "email_id" if "email_id" in df.columns else "message_id"

    mask = df[sender_col] == name
    candidates = df[mask].copy()
    candidates["word_count"] = candidates[body_col].astype(str).str.split().str.len()
    candidates = candidates[candidates["word_count"] >= 50]
    candidates["quoted_frac"] = candidates[body_col].astype(str).apply(_quoted_fraction)
    candidates = candidates[candidates["quoted_frac"] <= 0.30]
    candidates = candidates.sort_values("date", ascending=False).head(n)

    return [
        HeldoutEmail(
            message_id=str(row[id_col]),
            subject=str(row.get("subject", "")),
            body=str(row[body_col]),
            date=pd.to_datetime(row["date"]),
        )
        for _, row in candidates.iterrows()
    ]


GENERIC_CONTROL_TEMPLATE = (
    "You are an Enron employee with expertise in {topics}. "
    "Reply to incoming emails using a typical professional business tone."
)


def build_control_prompt(persona_row: dict) -> str:
    """Generic control prompt: top-3 expertise topics, no voice description."""
    raw = persona_row.get("expertise_topics", "[]")
    try:
        topics = _ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
    except (ValueError, SyntaxError):
        topics = []
    topics = list(topics)[:3]
    topics_text = ", ".join(topics) if topics else "general business operations"
    return GENERIC_CONTROL_TEMPLATE.format(topics=topics_text)


@dataclass(frozen=True)
class GeneratedReplies:
    """Persona-prompt reply and control-prompt reply for one held-out email."""

    persona_reply: str
    control_reply: str


def _format_reply_user_message(holdout: HeldoutEmail) -> str:
    return (
        f"Subject: {holdout.subject}\n\n"
        f"{holdout.body}\n\n"
        "Write a reply now. Output the body of the email only - no headers, "
        "no quoted text, no meta-commentary."
    )


def generate_replies(
    holdout: HeldoutEmail,
    *,
    persona_prompt: str,
    control_prompt: str,
    client: TextChatClient,
    model: str,
    temperature: float = 0.7,
) -> GeneratedReplies:
    """Generate one persona reply and one control reply for a held-out email."""
    user = _format_reply_user_message(holdout)
    msg = [{"role": "user", "content": user}]
    persona_reply = client.chat(
        system=persona_prompt, messages=msg, model=model, temperature=temperature,
    )
    control_reply = client.chat(
        system=control_prompt, messages=msg, model=model, temperature=temperature,
    )
    return GeneratedReplies(
        persona_reply=persona_reply.strip(),
        control_reply=control_reply.strip(),
    )


ATTRIBUTION_SYSTEM_PROMPT = """You are an expert reader of business correspondence.

You will see three short messages labelled A, B, and C, plus a name and role of an Enron employee. One of the three was actually written by that employee; the other two are imitations or generic business writing.

Pick the letter (A, B, or C) most likely written by the named employee. Base your decision on voice — sentence rhythm, sign-off habits, salutation style, register — not topic.

Return ONLY a JSON object:
{"pick": "A"|"B"|"C", "reasoning": "<one short sentence>"}
"""


@dataclass(frozen=True)
class AttributionVerdict:
    """One judge verdict on a 3-way attribution call."""

    picked_letter: str  # "A" | "B" | "C"
    picked: Literal["actual", "persona", "control"]
    reasoning: str
    shuffle_order: tuple[str, str, str]


def _resolve_pick(
    picked_letter: str, order: tuple[str, str, str]
) -> Literal["actual", "persona", "control"]:
    idx = "ABC".index(picked_letter)
    label = order[idx]
    if label not in {"actual", "persona", "control"}:
        raise ValueError(f"Unexpected label in shuffle order: {label}")
    return label  # type: ignore[return-value]


def judge_attribution(
    actual: str,
    persona_reply: str,
    control_reply: str,
    *,
    name: str,
    role: str,
    client: TextChatClient,
    model: str,
    temperature: float = 0.0,
    seed: int = 0,
) -> AttributionVerdict:
    """3-way attribution: which message did {name} actually write?

    Order is shuffled deterministically by `seed` so the test set can be
    fully replicated and the judge is not given a positional hint.
    """
    rng = _random.Random(seed)
    items = [("actual", actual), ("persona", persona_reply), ("control", control_reply)]
    rng.shuffle(items)
    order = tuple(label for label, _ in items)
    body = "\n\n".join(
        f"--- {letter} ---\n{text}"
        for letter, (_, text) in zip("ABC", items)
    )
    user = (
        f"Name: {name}\nRole: {role}\n\n"
        f"{body}\n\n"
        "Which message did this person actually write?"
    )
    raw = client.chat(
        system=ATTRIBUTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        model=model,
        temperature=temperature,
    )
    data = _json.loads(raw.strip().lstrip("`json").rstrip("`").strip())
    picked_letter = data["pick"].strip().upper()
    if picked_letter not in {"A", "B", "C"}:
        raise ValueError(f"Bad pick letter: {picked_letter!r}")
    return AttributionVerdict(
        picked_letter=picked_letter,
        picked=_resolve_pick(picked_letter, order),
        reasoning=str(data.get("reasoning", "")),
        shuffle_order=order,
    )


DEFAULT_PERSONAS_CSV = OUTPUT_DIR / "person_enrichment.csv"
DEFAULT_PARQUET = OUTPUT_DIR / "clean_emails.parquet"
DEFAULT_PROMPTS_DIR = OUTPUT_DIR / "persona_prompts"
DEFAULT_OUT_DIR = OUTPUT_DIR / "persona_voice_eval"


def _stylometry_panel(actual: str, persona: str, control: str) -> dict:
    """Compute stylometric features and persona-closer-than-control flags."""
    feats = {}
    for name, body in (("actual", actual), ("persona", persona), ("control", control)):
        feats[name] = {
            "mean_sentence_length": mean_sentence_length(body),
            "salutation": salutation_pattern(body),
            "signoff": signoff_pattern(body),
            "blackberry": blackberry_tag_rate(body),
        }
    feats["bigram_jaccard_persona_actual"] = char_bigram_jaccard(persona, actual)
    feats["bigram_jaccard_control_actual"] = char_bigram_jaccard(control, actual)
    feats["persona_closer_jaccard"] = (
        feats["bigram_jaccard_persona_actual"]
        > feats["bigram_jaccard_control_actual"]
    )
    feats["persona_matches_signoff"] = (
        feats["persona"]["signoff"] == feats["actual"]["signoff"]
    )
    feats["control_matches_signoff"] = (
        feats["control"]["signoff"] == feats["actual"]["signoff"]
    )
    feats["persona_matches_salutation"] = (
        feats["persona"]["salutation"] == feats["actual"]["salutation"]
    )
    return feats


def _build_default_client():
    from orggraph.agents.agent import OpenAIChatClient
    base = _os.environ.get("LLM_BASE_URL") or _os.environ.get("OPENAI_BASE_URL")
    if not base:
        raise SystemExit("Set LLM_BASE_URL to a vLLM endpoint.")
    return OpenAIChatClient(
        base_url=base,
        api_key=_os.environ.get("LLM_API_KEY") or _os.environ.get("OPENAI_API_KEY") or "EMPTY",
    )


def run(
    personas_csv: Path,
    parquet_path: Path,
    prompts_dir: Path,
    out_dir: Path,
    *,
    model: str,
    n_per_band: int = 5,
    n_holdout: int = 5,
    client: TextChatClient | None = None,
    seed_base: int = 0,
) -> dict:
    """Run Tier 3 voice fidelity eval over the stratified persona sample."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_persona").mkdir(exist_ok=True)
    client = client or _build_default_client()

    selected = select_stratified_personas(personas_csv, n_per_band=n_per_band)
    persona_df = pd.read_csv(personas_csv).set_index("name")

    summary_rows: list[dict] = []
    for i, name in enumerate(selected):
        slug = _re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
        persona_prompt_path = prompts_dir / f"{slug}.txt"
        if not persona_prompt_path.exists():
            print(f"  SKIP  {name}: no prompt at {persona_prompt_path}")
            continue
        persona_prompt = persona_prompt_path.read_text()
        row = persona_df.loc[name].to_dict()
        control_prompt = build_control_prompt({"expertise_topics": row.get("expertise_topics", "[]")})

        holdouts = select_holdout_emails(name=name, parquet_path=parquet_path, n=n_holdout)
        if not holdouts:
            print(f"  SKIP  {name}: no held-out emails")
            continue

        per_holdout: list[dict] = []
        persona_above_control = 0
        for j, h in enumerate(holdouts):
            t0 = _time.time()
            replies = generate_replies(
                holdout=h, persona_prompt=persona_prompt,
                control_prompt=control_prompt,
                client=client, model=model,
            )
            verdict = judge_attribution(
                actual=h.body,
                persona_reply=replies.persona_reply,
                control_reply=replies.control_reply,
                name=name, role=str(row.get("role_summary", "")),
                client=client, model=model, seed=seed_base + i * 100 + j,
            )
            stylo = _stylometry_panel(h.body, replies.persona_reply, replies.control_reply)
            persona_above_control += int(
                verdict.picked == "persona"
                or (verdict.picked == "actual" and stylo["persona_closer_jaccard"])
            )
            per_holdout.append({
                "message_id": h.message_id,
                "subject": h.subject,
                "actual": h.body,
                "persona_reply": replies.persona_reply,
                "control_reply": replies.control_reply,
                "verdict": {
                    "picked_letter": verdict.picked_letter,
                    "picked": verdict.picked,
                    "reasoning": verdict.reasoning,
                    "shuffle_order": list(verdict.shuffle_order),
                },
                "stylometry": stylo,
                "duration_s": _time.time() - t0,
            })

        rate = persona_above_control / len(per_holdout)
        summary_rows.append({
            "name": name,
            "n_holdouts_evaluated": len(per_holdout),
            "attribution_persona_above_control_rate": rate,
            "stylo_persona_closer_jaccard_rate": sum(
                int(h["stylometry"]["persona_closer_jaccard"]) for h in per_holdout
            ) / len(per_holdout),
            "stylo_persona_signoff_match_rate": sum(
                int(h["stylometry"]["persona_matches_signoff"]) for h in per_holdout
            ) / len(per_holdout),
        })
        (out_dir / "per_persona" / f"{slug}.json").write_text(
            _json.dumps({"name": name, "holdouts": per_holdout}, indent=2)
        )
        print(
            f"  OK  {name:<30s}  attr_rate={rate:.2f}  "
            f"jaccard_closer_rate={summary_rows[-1]['stylo_persona_closer_jaccard_rate']:.2f}"
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    return {"personas_evaluated": len(summary_rows), "out_dir": str(out_dir)}


from orggraph.agents.persona import load_personas_from_csv
from orggraph.agents.prompt_builder import (
    BUILDER_VARIANTS,
    build_persona_prompt_n_best,
)
from orggraph.agents.kg_context import KGContext
from orggraph.agents.samples import load_sample_messages


def run_structure_search(
    *,
    calibration_names: list[str],
    variant_names: list[str],
    personas_csv: Path,
    parquet_path: Path,
    out_dir: Path,
    builder_client: TextChatClient,
    judge_client: TextChatClient,
    model: str,
    judge_model: str,
    n_best: int = 3,
    n_holdout: int = 5,
    builder_temperature: float = 0.6,
    judge_temperature: float = 0.0,
    seed_base: int = 0,
) -> dict:
    """Compare builder variants on a fixed calibration set.

    For each variant × calibration persona:
      1. Build N-best prompts via that variant; pick winner by Tier 1 score.
      2. Run Tier 3 voice fidelity on the winning prompt vs a generic
         control prompt across n_holdout real held-out emails.

    Aggregates per-variant means and picks the variant with the
    highest Tier 3 attribution rate (tie-break: Tier 1 total).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    persona_df = pd.read_csv(personas_csv).set_index("name")
    personas = load_personas_from_csv(personas_csv)

    summary_rows: list[dict] = []
    for variant_name in variant_names:
        if variant_name not in BUILDER_VARIANTS:
            raise ValueError(f"Unknown variant: {variant_name!r}")
        variant = BUILDER_VARIANTS[variant_name]
        variant_dir = out_dir / variant_name
        prompts_dir = variant_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        per_persona_tier1: list[dict] = []
        per_persona_tier3: list[dict] = []

        for i, name in enumerate(calibration_names):
            if name not in personas:
                print(f"  WARN: {name} not in personas CSV, skipping")
                continue
            persona = personas[name]
            slug = _re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()

            # Tier 1
            try:
                samples = load_sample_messages(
                    name, parquet_path=parquet_path, n_samples=5,
                )
            except Exception:
                samples = ()
            kg = KGContext(name=name)  # No KG enrichment in structure search to remove a confounder.
            winner, all_cands = build_persona_prompt_n_best(
                persona=persona, samples=samples, kg=kg,
                builder_client=builder_client, judge_client=judge_client,
                model=model, judge_model=judge_model,
                n_best=n_best,
                builder_temperature=builder_temperature,
                judge_temperature=judge_temperature,
                variant_name=variant_name,
            )
            (prompts_dir / f"{slug}.txt").write_text(winner.text)
            per_persona_tier1.append({
                "name": name,
                "winner_total": winner.scores.total(),
                "winner_min": winner.scores.min_score(),
                "all_totals": [c.scores.total() for c in all_cands],
            })

            # Tier 3
            row = persona_df.loc[name].to_dict()
            control_prompt = build_control_prompt(
                {"expertise_topics": row.get("expertise_topics", "[]")}
            )
            holdouts = select_holdout_emails(
                name=name, parquet_path=parquet_path, n=n_holdout,
            )
            persona_above_control = 0
            for j, h in enumerate(holdouts):
                replies = generate_replies(
                    holdout=h, persona_prompt=winner.text,
                    control_prompt=control_prompt,
                    client=builder_client, model=model,
                )
                # Distinct seed namespace from `run` (which uses
                # seed_base + i*100 + j) so the structure search isn't
                # judging on the same shuffles that the per-persona Tier 3
                # eval will reuse downstream.
                verdict = judge_attribution(
                    actual=h.body,
                    persona_reply=replies.persona_reply,
                    control_reply=replies.control_reply,
                    name=name, role=str(row.get("role_summary", "")),
                    client=judge_client, model=judge_model,
                    seed=seed_base + i * 100 + j + 4,
                )
                stylo = _stylometry_panel(
                    h.body, replies.persona_reply, replies.control_reply,
                )
                persona_above_control += int(
                    verdict.picked == "persona"
                    or (
                        verdict.picked == "actual"
                        and stylo["persona_closer_jaccard"]
                    )
                )
            attr_rate = (
                persona_above_control / len(holdouts) if holdouts else 0.0
            )
            per_persona_tier3.append({"name": name, "attr_rate": attr_rate})

        mean_tier1 = (
            sum(r["winner_total"] for r in per_persona_tier1)
            / max(1, len(per_persona_tier1))
        )
        mean_tier3 = (
            sum(r["attr_rate"] for r in per_persona_tier3)
            / max(1, len(per_persona_tier3))
        )
        summary_rows.append({
            "variant": variant_name,
            "n_calibration_personas": len(per_persona_tier1),
            "mean_tier1_total": mean_tier1,
            "mean_tier3_attribution_rate": mean_tier3,
            "description": variant.description,
        })
        (variant_dir / "tier1.json").write_text(
            _json.dumps(per_persona_tier1, indent=2)
        )
        (variant_dir / "tier3.json").write_text(
            _json.dumps(per_persona_tier3, indent=2)
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    winner_row = summary_df.sort_values(
        ["mean_tier3_attribution_rate", "mean_tier1_total"],
        ascending=False,
    ).iloc[0]
    return {
        "summary": summary_rows,
        "winner": str(winner_row["variant"]),
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> None:
    p = _argparse.ArgumentParser(description="Tier 3 voice fidelity validation.")
    p.add_argument("--personas-csv", type=Path, default=DEFAULT_PERSONAS_CSV)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--model",
        default=_os.environ.get("LLM_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit"),
    )
    p.add_argument("--n-per-band", type=int, default=5)
    p.add_argument("--n-holdout", type=int, default=5)
    p.add_argument("--seed-base", type=int, default=0)
    args = p.parse_args(argv)
    run(
        personas_csv=args.personas_csv, parquet_path=args.parquet,
        prompts_dir=args.prompts_dir, out_dir=args.out_dir,
        model=args.model,
        n_per_band=args.n_per_band, n_holdout=args.n_holdout,
        seed_base=args.seed_base,
    )


def main_structure_search(argv: list[str] | None = None) -> None:
    p = _argparse.ArgumentParser(
        description="A/B-test builder variants on a calibration persona set."
    )
    p.add_argument("--personas-csv", type=Path, default=DEFAULT_PERSONAS_CSV)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument(
        "--out-dir", type=Path,
        default=OUTPUT_DIR / "persona_structure_search",
    )
    p.add_argument(
        "--variants", default="v1_freeform,v2_sections,v3_idcard",
        help="Comma-separated variant names from BUILDER_VARIANTS.",
    )
    p.add_argument(
        "--names", required=True,
        help="Comma-separated calibration persona names (canonical form).",
    )
    p.add_argument(
        "--model",
        default=_os.environ.get("LLM_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit"),
    )
    p.add_argument("--n-best", type=int, default=3)
    p.add_argument("--n-holdout", type=int, default=5)
    p.add_argument("--seed-base", type=int, default=0)
    args = p.parse_args(argv)

    client = _build_default_client()
    summary = run_structure_search(
        calibration_names=[n.strip() for n in args.names.split(",") if n.strip()],
        variant_names=[v.strip() for v in args.variants.split(",") if v.strip()],
        personas_csv=args.personas_csv,
        parquet_path=args.parquet,
        out_dir=args.out_dir,
        builder_client=client, judge_client=client,
        model=args.model, judge_model=args.model,
        n_best=args.n_best, n_holdout=args.n_holdout,
        seed_base=args.seed_base,
    )
    print("\n=== Structure search results ===")
    for row in summary["summary"]:
        print(
            f"  {row['variant']:<14s}  "
            f"tier1={row['mean_tier1_total']:5.1f}  "
            f"tier3={row['mean_tier3_attribution_rate']:.2f}  "
            f"({row['description']})"
        )
    print(f"\nWinner: {summary['winner']}")
    print(f"Detail: {summary['out_dir']}")


if __name__ == "__main__":
    main()
