"""Stage 4a addendum: canonicalize free-text persona roles to a closed tier vocabulary.

Per docs/plans/2026-05-08-role-canonicalization.md, this module reads
person_enrichment.csv (the Stage 4a output), classifies each Person's
free-text role_summary into the six-tier vocabulary
{Employee, Manager, Director, VP, SVP, C-Suite} (or Unknown), and
writes a sidecar person_enrichment_canonical.csv keyed on Person name.

The classifier is a single-call zero-shot LLM prompt with strict JSON
output, followed by a deterministic post-LLM normalization layer that
maps the LLM's free-text canonical_role field to the closed enum.
The normalization layer is what gets unit-tested in
tests/pipeline/enrich/test_canonicalize_roles.py.

Why the addendum design: Stage 4a's free-text role_summary is the
right shape for human reviewers but the wrong shape for the RQ1
evaluator (which needs an ordinal feature for the A5 ablation rung).
Adding canonical_role_level as a sidecar column unifies the two paths
under a single per-Person record without re-running the expensive
Stage 4a calibration.

Sidecar columns:
    name                            str   primary key
    canonical_role                  str   one of {Employee, Manager, Director, VP, SVP, C-Suite, Unknown}
    canonical_role_confidence       float 0.0-1.0
    canonical_role_rationale        str   <= 200 chars
    canonical_role_level            int   0-6 per the table in this module's docstring; NaN for Unknown
    model                           str   LLM model identifier
    temperature                     float
    timestamp                       str   ISO 8601

Console script: orggraph-enrich-canonicalize (registered in pyproject.toml).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from orggraph.agents.agent import OpenAIChatClient, TextChatClient
from orggraph.config import OUTPUT_DIR


# ---------------------------------------------------------------------------
# Vocabulary (per plan §B.1, §B.4)
# ---------------------------------------------------------------------------

CANONICAL_TIERS: tuple[str, ...] = (
    "Employee",
    "Manager",
    "Director",
    "VP",
    "SVP",
    "C-Suite",
)
"""The six-tier canonical vocabulary, ordered by increasing seniority."""

UNKNOWN = "Unknown"
"""The abstain bucket. Used when the classifier's confidence is below threshold
or when the post-LLM normalization cannot map the LLM's output to any canonical
tier. Treated as missing data in the RQ1 evaluator."""

TIER_LEVELS: dict[str, int] = {
    "Employee": 0,
    "Manager": 1,
    "Director": 2,
    "VP": 3,
    "SVP": 4,
    "C-Suite": 6,
}
"""Numeric levels matching the GT v2 level_numeric scale. Note the gap at 5;
this mirrors the GT idiosyncrasy where SVP and C-Suite are not adjacent."""

CONFIDENCE_THRESHOLD = 0.6
"""Per plan §B.4: any classification with confidence < threshold falls through
to Unknown. The classifier is asked to self-report confidence and to choose
Unknown explicitly when uncertain; this threshold is a safety net."""

DEFAULT_INPUT = OUTPUT_DIR / "person_enrichment.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "person_enrichment_canonical.csv"

SIDECAR_COLUMNS = (
    "name",
    "canonical_role",
    "canonical_role_confidence",
    "canonical_role_rationale",
    "canonical_role_level",
    "model",
    "temperature",
    "timestamp",
)


# ---------------------------------------------------------------------------
# Prompt (per plan §D.1)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a role classifier for Enron Corporation employees. Given a free-text role description and short context, you must assign the employee to exactly one of the following six canonical organizational tiers, or "Unknown" if you cannot make the call with confidence >= 0.6.

CANONICAL TIERS (in increasing seniority):
- Employee  : individual contributors (analysts, traders, specialists, associates)
- Manager   : first-line managers, in-house lawyers, team leads, senior counsel
- Director  : directors of a function or region (NOT Managing Director)
- VP        : Vice President, V.P.; NOT preceded by "Senior" or "Executive"
- SVP       : Senior VP, Executive VP, EVP, Managing Director (Enron convention)
- C-Suite   : CEO, CFO, COO, CRO, Chairman, President, any "Chief X Officer"; also Vice Chairman per GT convention

EDGE-CASE RULES:
- "Senior Vice President" / "Sr. VP" / "Executive VP" -> SVP, NOT VP.
- "Managing Director" -> SVP (Enron convention).
- "President" (standalone or "& COO") -> C-Suite.
- "Trader", "Analyst", "Specialist" alone -> Employee.
- "In House Lawyer" / "Senior Counsel" -> Manager.
- Compound titles: classify on the highest-tier component present.

OUTPUT CONTRACT (strict JSON, no code fences, no prose outside the object):
{
  "canonical_role": "<one of: Employee, Manager, Director, VP, SVP, C-Suite, Unknown>",
  "confidence": <float 0.0-1.0>,
  "rationale": "<one short sentence, <= 200 chars, citing the tier-determining phrase>"
}

If confidence < 0.6, you MUST output "Unknown"."""


USER_TEMPLATE = """EMPLOYEE: {name}
ROLE_SUMMARY: {role_summary}
SENIORITY_NARRATIVE: {seniority_narrative}
COMMUNICATION_STYLE: {communication_style}
EXPERTISE: {expertise}

Classify this employee into exactly one canonical tier."""


# ---------------------------------------------------------------------------
# Post-LLM normalization (the deterministic layer that the unit tests cover)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationResult:
    """Normalized classifier output, ready to write to the sidecar."""

    canonical_role: str
    canonical_role_level: int | None
    canonical_role_confidence: float
    canonical_role_rationale: str


def _normalize_role_string(raw_role: str) -> str | None:
    """Map a free-text role string to a canonical tier, or None if no match.

    Applies the edge-case rules from plan §B.2 (alias rules per tier). Rules
    are tried in seniority order from most-senior to least-senior, so that
    compound titles like "Sr. VP & General Counsel" classify on the highest
    tier present.
    """
    if not raw_role:
        return None
    text = raw_role.lower().strip()

    # Exact match against the canonical enum first (cheap path)
    for tier in CANONICAL_TIERS:
        if text == tier.lower():
            return tier
    if text in ("unknown", "n/a", "na", "none"):
        return None

    # C-Suite (highest tier, checked first)
    csuite_substrings = (
        "ceo", "cfo", "coo", "cro", "cao",
        "chief executive officer",
        "chief financial officer",
        "chief operating officer",
        "chief risk officer",
        "chief accounting officer",
        "chairman", "chairwoman", "chairperson",
        "vice chairman", "vice-chairman",
    )
    if any(p in text for p in csuite_substrings):
        return "C-Suite"
    # Any "Chief ... Officer" pattern (e.g. "Chief Compliance Officer")
    if re.search(r"\bchief\b.*\bofficer\b", text):
        return "C-Suite"
    # "President" matches C-Suite per the Enron GT convention, but only when
    # it is the noun head, not e.g. "Vice President".
    if re.search(r"\bpresident\b", text) and "vice president" not in text and "vice-president" not in text:
        return "C-Suite"

    # SVP (must check before VP, since "senior vice president" contains "vice president")
    svp_substrings = (
        "senior vice president",
        "sr vice president", "sr. vice president",
        "executive vice president",
        "executive vp", "exec vp",
        "managing director",
    )
    if any(p in text for p in svp_substrings):
        return "SVP"
    # EVP / SVP abbreviations
    if re.search(r"\bevp\b", text) or re.search(r"\bsvp\b", text):
        return "SVP"
    # "Sr. VP" or "Sr VP"
    if re.search(r"\bsr\.?\s+vp\b", text):
        return "SVP"

    # VP (after the senior-modifier check above)
    if "vice president" in text or "vice-president" in text:
        return "VP"
    if re.search(r"\bvp\b", text) or re.search(r"\bv\.p\.", text):
        return "VP"

    # Director (after Managing Director, which is SVP)
    if "director" in text and "managing director" not in text and "executive director" not in text:
        return "Director"

    # Manager-tier
    manager_substrings = (
        "manager", "mgr",
        "head of",
        "team lead", "team leader",
        "in house lawyer", "in-house lawyer",
        "senior counsel", "sr. counsel", "sr counsel",
    )
    if any(p in text for p in manager_substrings):
        return "Manager"

    # Employee (individual contributors)
    employee_substrings = (
        "analyst", "trader", "specialist", "associate", "assistant",
        "coordinator", "consultant", "engineer", "scheduler", "controller",
    )
    if any(p in text for p in employee_substrings):
        return "Employee"

    return None


def normalize(parsed: dict) -> ClassificationResult:
    """Take a parsed LLM JSON dict and produce the final ClassificationResult.

    Applies the confidence threshold and the alias rules. Always returns a
    valid ClassificationResult (uses Unknown / NaN for unparseable cases).
    """
    raw_role = str(parsed.get("canonical_role", "")).strip()
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    rationale = str(parsed.get("rationale", ""))[:200]

    if confidence < CONFIDENCE_THRESHOLD:
        return ClassificationResult(
            canonical_role=UNKNOWN,
            canonical_role_level=None,
            canonical_role_confidence=confidence,
            canonical_role_rationale=rationale,
        )

    canonical = _normalize_role_string(raw_role)
    if canonical is None:
        return ClassificationResult(
            canonical_role=UNKNOWN,
            canonical_role_level=None,
            canonical_role_confidence=confidence,
            canonical_role_rationale=rationale,
        )

    return ClassificationResult(
        canonical_role=canonical,
        canonical_role_level=TIER_LEVELS[canonical],
        canonical_role_confidence=confidence,
        canonical_role_rationale=rationale,
    )


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def parse_llm_response(raw: str) -> dict:
    """Extract a JSON object from the LLM's raw text response.

    Tolerates code fences (```json ... ```), leading/trailing whitespace, and
    a trailing newline. Returns an empty dict on parse failure; the caller's
    normalize() turns that into Unknown.
    """
    if not raw:
        return {}
    s = raw.strip()
    # Strip code fences if present
    if s.startswith("```"):
        lines = s.split("\n")
        # Drop first line (e.g. "```json") and find closing fence
        lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Strip any trailing text after the closing brace
    end = s.rfind("}")
    if end >= 0:
        s = s[: end + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Classifier (LLM call)
# ---------------------------------------------------------------------------


def classify_one(
    *,
    name: str,
    role_summary: str,
    seniority_narrative: str,
    communication_style: str,
    expertise: str,
    client: TextChatClient,
    model: str,
    temperature: float = 0.0,
) -> ClassificationResult:
    """Classify a single Person's role via the LLM + post-LLM normalization."""
    user_msg = USER_TEMPLATE.format(
        name=name or "",
        role_summary=role_summary or "",
        seniority_narrative=seniority_narrative or "",
        communication_style=communication_style or "",
        expertise=expertise or "",
    )
    try:
        raw = client.chat(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            model=model,
            temperature=temperature,
        )
    except Exception as e:  # noqa: BLE001 - we want resilience over correctness
        return ClassificationResult(
            canonical_role=UNKNOWN,
            canonical_role_level=None,
            canonical_role_confidence=0.0,
            canonical_role_rationale=f"classifier error: {type(e).__name__}",
        )

    parsed = parse_llm_response(raw)
    return normalize(parsed)


# ---------------------------------------------------------------------------
# CLI orchestrator
# ---------------------------------------------------------------------------


def _build_client() -> TextChatClient:
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    if not base_url:
        raise SystemExit(
            "Set LLM_BASE_URL (or OPENAI_BASE_URL) in the environment."
        )
    return OpenAIChatClient(base_url=base_url, api_key=api_key)


def _read_existing_sidecar(path: Path) -> dict[str, dict]:
    """Return {name: row_dict} from an existing sidecar, or empty if missing."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["name"]] = row
    return out


def _write_sidecar(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SIDECAR_COLUMNS))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run(
    *,
    input_csv: Path,
    output_csv: Path,
    model: str,
    temperature: float = 0.0,
    force: bool = False,
    client: TextChatClient | None = None,
    progress_every: int = 10,
) -> dict:
    """Run the canonicalizer end-to-end.

    Resume-safe: rows already present in the sidecar with a non-Unknown
    canonical_role are skipped unless force=True.

    Returns a summary dict (counts per tier, Unknown rate, total classified).
    """
    if client is None:
        client = _build_client()

    # Load persons
    with input_csv.open() as f:
        persons = list(csv.DictReader(f))

    existing = _read_existing_sidecar(output_csv)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

    out_rows: list[dict] = []
    classified = 0
    skipped = 0
    counts: dict[str, int] = {t: 0 for t in CANONICAL_TIERS}
    counts[UNKNOWN] = 0

    for i, p in enumerate(persons):
        name = p["name"]
        # Resume guard
        if (
            not force
            and name in existing
            and existing[name].get("canonical_role") not in (UNKNOWN, "", None)
        ):
            out_rows.append(existing[name])
            counts[existing[name]["canonical_role"]] = (
                counts.get(existing[name]["canonical_role"], 0) + 1
            )
            skipped += 1
            continue

        result = classify_one(
            name=name,
            role_summary=p.get("role_summary", ""),
            seniority_narrative=p.get("seniority_narrative", ""),
            communication_style=p.get("communication_style", ""),
            expertise=p.get("expertise", ""),
            client=client,
            model=model,
            temperature=temperature,
        )
        counts[result.canonical_role] = counts.get(result.canonical_role, 0) + 1
        classified += 1

        out_rows.append({
            "name": name,
            "canonical_role": result.canonical_role,
            "canonical_role_confidence": f"{result.canonical_role_confidence:.3f}",
            "canonical_role_rationale": result.canonical_role_rationale,
            "canonical_role_level": (
                "" if result.canonical_role_level is None
                else str(result.canonical_role_level)
            ),
            "model": model,
            "temperature": f"{temperature:.2f}",
            "timestamp": timestamp,
        })

        if progress_every and (i + 1) % progress_every == 0:
            print(
                f"  classified {classified}, skipped {skipped} "
                f"({i + 1}/{len(persons)}); current row: {name} -> {result.canonical_role}"
            )

    _write_sidecar(output_csv, out_rows)

    total = sum(counts.values())
    unknown_rate = counts.get(UNKNOWN, 0) / total if total else 0.0
    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "total": total,
        "classified": classified,
        "skipped": skipped,
        "counts": counts,
        "unknown_rate": unknown_rate,
        "model": model,
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Input Stage 4a CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Sidecar output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model", default=os.environ.get("LLM_MODEL", "gemma3:4b"),
        help="LLM model identifier (default: $LLM_MODEL or gemma3:4b)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Classifier temperature (default: 0.0 for stability)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-classify rows even if they already have a non-Unknown row in the sidecar.",
    )
    args = parser.parse_args(argv)

    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Model:       {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Force:       {args.force}")

    summary = run(
        input_csv=args.input,
        output_csv=args.output,
        model=args.model,
        temperature=args.temperature,
        force=args.force,
    )

    print()
    print(f"Total persons:    {summary['total']}")
    print(f"Newly classified: {summary['classified']}")
    print(f"Skipped (cached): {summary['skipped']}")
    print(f"Unknown rate:     {summary['unknown_rate']:.1%}")
    print("Counts per tier:")
    for tier in [*CANONICAL_TIERS, UNKNOWN]:
        n = summary["counts"].get(tier, 0)
        print(f"  {tier:<10} {n}")


if __name__ == "__main__":
    main()
