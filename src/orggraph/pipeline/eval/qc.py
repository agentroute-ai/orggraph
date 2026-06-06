"""Stage 8 — Quality check for the v2 LLM persona enrichment.

Reads datasets/enron/processed/person_enrichment.csv (v2 schema, 18 columns)
and produces datasets/enron/processed/llm_quality_report.md with green/yellow/red
verdicts on coverage, distribution diversity, known-executive sanity,
signal-style calibration, role_summary diversity, and confidence calibration.

Replaces the v1 QC script that read the v1 schema (seniority/key_collaborators_json).
The v2 schema has formality, authority_style, communication_style, role_summary,
expertise, seniority_narrative, confidence_self_report.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from statistics import mean, median, stdev

from orggraph.config import OUTPUT_DIR

PERSON_CSV = OUTPUT_DIR / "person_enrichment.csv"
REPORT_PATH = OUTPUT_DIR / "llm_quality_report.md"

# Authority style ordering for calibration check (low-power → high-power)
AUTHORITY_ORDINAL = {
    "passive": 0,
    "collaborative": 1,
    "consultative": 2,
    "delegating": 3,
    "directive": 4,
}

# Known executives — accept name variants observed in the corpus.
KNOWN_EXECS = {
    "Kenneth Lay":     ["chairman", "ceo", "founder", "executive"],
    "Jeff Skilling":   ["president", "ceo", "chief"],     # appears as "Jeffery Skilling"
    "John Lavorato":   ["president", "trading", "head", "executive"],
    "Louise Kitchen":  ["president", "enrononline", "head", "executive"],
    "David Delainey":  ["president", "ceo", "americas", "executive"],
}

NAME_VARIANTS = {
    "jeffery skilling": "Jeff Skilling",
    "jeffrey skilling": "Jeff Skilling",
    "kenneth l. lay":   "Kenneth Lay",
    "kenneth l lay":    "Kenneth Lay",
    "ken lay":          "Kenneth Lay",
}


def _color(value, green, yellow=None, lower_is_better=False):
    if lower_is_better:
        if value <= green:
            return "green"
        if yellow is not None and value <= yellow:
            return "yellow"
        return "red"
    if value >= green:
        return "green"
    if yellow is not None and value >= yellow:
        return "yellow"
    return "red"


def _resolve_name(raw: str) -> str:
    return NAME_VARIANTS.get(raw.strip().lower(), raw)


def load_rows() -> list[dict]:
    if not PERSON_CSV.exists():
        print(f"[error] {PERSON_CSV} not found. Run Stage 4a first.")
        sys.exit(1)
    with open(PERSON_CSV) as f:
        rows = list(csv.DictReader(f))
    # Drop any "NaN" / empty-name rows (data hygiene)
    return [r for r in rows if r.get("name") and r["name"] not in ("NaN", "nan")]


def is_heuristic_row(r: dict) -> bool:
    """Heuristic fallback rows have confidence=1 and a templated communication_style."""
    return (
        r.get("confidence_self_report") == "1"
        and r.get("communication_style", "").startswith("Communicates in a")
    )


# ---------------------------------------------------------------------------
# Section A — Coverage and grounding
# ---------------------------------------------------------------------------

def section_coverage(rows: list[dict], target_persons: int) -> list[tuple]:
    n = len(rows)
    n_grounded = sum(1 for r in rows if r.get("role_summary", "").strip())
    n_heuristic = sum(1 for r in rows if is_heuristic_row(r))
    return [
        ("Person CSV coverage", n, target_persons,
         _color(n, green=target_persons, yellow=int(0.95 * target_persons))),
        ("Narrative populated pct", f"{100*n_grounded/max(1,n):.1f}%", "100%",
         _color(100*n_grounded/max(1,n), green=99, yellow=90)),
        ("Heuristic fallback pct", f"{100*n_heuristic/max(1,n):.1f}%", "0%",
         _color(100*n_heuristic/max(1,n), green=0, yellow=5, lower_is_better=True)),
    ]


# ---------------------------------------------------------------------------
# Section B — Distribution diversity
# ---------------------------------------------------------------------------

def section_diversity(rows: list[dict]) -> tuple[list[tuple], dict]:
    fdist = Counter(r.get("formality", "") for r in rows if r.get("formality"))
    adist = Counter(r.get("authority_style", "") for r in rows if r.get("authority_style"))
    cdist = Counter(r.get("confidence_self_report", "") for r in rows
                    if r.get("confidence_self_report"))

    n_formality = len(fdist)
    n_auth = len(adist)
    n_conf = len(cdist)

    metrics = [
        ("Formality levels used (of 5)", n_formality, ">=4",
         _color(n_formality, green=4, yellow=3)),
        ("Authority styles used (of 5)", n_auth, ">=4",
         _color(n_auth, green=4, yellow=3)),
        ("Confidence levels used", n_conf, ">=3",
         _color(n_conf, green=3, yellow=2)),
    ]
    return metrics, {"formality": dict(fdist), "authority_style": dict(adist),
                     "confidence": dict(cdist)}


# ---------------------------------------------------------------------------
# Section C — Known executive sanity check
# ---------------------------------------------------------------------------

def section_known_execs(rows: list[dict]) -> tuple[list[tuple], list[str]]:
    by_name = {_resolve_name(r["name"]): r for r in rows}
    detail_lines = []
    hits = 0
    for canonical, keywords in KNOWN_EXECS.items():
        r = by_name.get(canonical)
        if not r:
            detail_lines.append(f"  - {canonical}: NOT FOUND in CSV")
            continue
        role_lc = r.get("role_summary", "").lower()
        senior = any(kw in role_lc for kw in keywords)
        marker = "✓" if senior else "✗"
        if senior:
            hits += 1
        detail_lines.append(
            f"  - {canonical} ({r['name']}): {marker} \"{r['role_summary'][:80]}…\""
        )
    metric = (f"Known execs identified ({hits}/{len(KNOWN_EXECS)})",
              f"{hits}/{len(KNOWN_EXECS)}", f">={len(KNOWN_EXECS)-1}/{len(KNOWN_EXECS)}",
              _color(hits, green=len(KNOWN_EXECS)-1, yellow=len(KNOWN_EXECS)-2))
    return [metric], detail_lines


# ---------------------------------------------------------------------------
# Section D — Calibration: signal-to-style coherence
# ---------------------------------------------------------------------------

def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation. Returns None if degenerate."""
    if len(xs) < 3 or len(ys) < 3:
        return None
    n = len(xs)
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rank_x = [0] * n
    rank_y = [0] * n
    for r, i in enumerate(rx):
        rank_x[i] = r
    for r, i in enumerate(ry):
        rank_y[i] = r
    mean_rx = sum(rank_x) / n
    mean_ry = sum(rank_y) / n
    num = sum((rank_x[i] - mean_rx) * (rank_y[i] - mean_ry) for i in range(n))
    dx = sum((r - mean_rx) ** 2 for r in rank_x) ** 0.5
    dy = sum((r - mean_ry) ** 2 for r in rank_y) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def section_calibration(rows: list[dict]) -> tuple[list[tuple], dict]:
    pairs = []
    for r in rows:
        try:
            d = float(r.get("directiveness_signal") or 0)
            a = float(r.get("agenda_setting_signal") or 0)
            style = r.get("authority_style", "").lower()
            if style not in AUTHORITY_ORDINAL:
                continue
            pairs.append((d, a, AUTHORITY_ORDINAL[style]))
        except ValueError:
            continue

    if len(pairs) < 10:
        return [("Signal-style calibration", "n<10", ">=0.3", "yellow")], {}

    rho_d = spearman([p[0] for p in pairs], [p[2] for p in pairs])
    rho_a = spearman([p[1] for p in pairs], [p[2] for p in pairs])
    rho_combined = spearman([p[0] + p[1] for p in pairs], [p[2] for p in pairs])

    metrics = [
        ("Directiveness vs authority_style (Spearman ρ)",
         f"{rho_d:.3f}" if rho_d is not None else "N/A",
         ">=0.3",
         _color(rho_d if rho_d is not None else -1, green=0.3, yellow=0.15)),
        ("Agenda-setting vs authority_style (Spearman ρ)",
         f"{rho_a:.3f}" if rho_a is not None else "N/A",
         ">=0.2",
         _color(rho_a if rho_a is not None else -1, green=0.2, yellow=0.10)),
        ("Combined signals vs authority_style (Spearman ρ)",
         f"{rho_combined:.3f}" if rho_combined is not None else "N/A",
         ">=0.4",
         _color(rho_combined if rho_combined is not None else -1, green=0.4, yellow=0.2)),
    ]
    return metrics, {"rho_d": rho_d, "rho_a": rho_a, "rho_combined": rho_combined,
                     "n": len(pairs)}


# ---------------------------------------------------------------------------
# Section E — Role summary diversity (no template collapse)
# ---------------------------------------------------------------------------

def section_diversity_role_summary(rows: list[dict]) -> tuple[list[tuple], dict]:
    role_summaries = [r.get("role_summary", "").strip() for r in rows
                      if r.get("role_summary", "").strip()]
    if not role_summaries:
        return [("Role summary uniqueness", "0", ">=95%", "red")], {}

    n_total = len(role_summaries)
    n_unique = len(set(role_summaries))
    pct_unique = 100 * n_unique / n_total

    lengths = [len(s) for s in role_summaries]
    avg_len = mean(lengths)
    med_len = median(lengths)
    std_len = stdev(lengths) if len(lengths) > 1 else 0

    metrics = [
        ("Role summary uniqueness pct", f"{pct_unique:.1f}%", ">=95%",
         _color(pct_unique, green=95, yellow=85)),
        ("Avg role summary length (chars)", f"{avg_len:.0f}", ">=80",
         _color(avg_len, green=80, yellow=50)),
    ]
    return metrics, {"avg_len": avg_len, "median_len": med_len, "std_len": std_len,
                     "unique": n_unique, "total": n_total}


# ---------------------------------------------------------------------------
# Section F — Confidence calibration
# ---------------------------------------------------------------------------

def section_confidence_calibration(rows: list[dict]) -> tuple[list[tuple], dict]:
    by_conf: dict[str, list[int]] = {}
    for r in rows:
        c = r.get("confidence_self_report", "")
        try:
            chars = int(r.get("response_chars", 0))
        except ValueError:
            continue
        by_conf.setdefault(c, []).append(chars)

    avgs = {c: round(mean(xs)) for c, xs in by_conf.items() if xs}
    sorted_confs = sorted(avgs.keys())

    monotonic = all(avgs[sorted_confs[i]] <= avgs[sorted_confs[i+1]]
                    for i in range(len(sorted_confs) - 1))

    metrics = [
        ("Confidence-response richness monotonic", "yes" if monotonic else "no",
         "yes",
         "green" if monotonic else "yellow"),
    ]
    return metrics, {"avg_response_chars_by_confidence": avgs}


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_markdown(sections: list[tuple[str, list[tuple]]],
                    extras: dict, exec_detail: list[str]) -> str:
    all_metrics = [m for _, ms in sections for m in ms]
    greens = sum(1 for m in all_metrics if m[3] == "green")
    yellows = sum(1 for m in all_metrics if m[3] == "yellow")
    reds = sum(1 for m in all_metrics if m[3] == "red")

    if reds > 0:
        verdict = f"⚠ REVIEW — {reds} red metric(s)."
    elif yellows > 0:
        verdict = f"🟡 PROCEED WITH CAUTION — {yellows} yellow metric(s)."
    else:
        verdict = "✅ GO — all green."

    out = [
        "# v2 Persona Enrichment Quality Report",
        "",
        f"**{greens} green / {yellows} yellow / {reds} red**",
        "",
        f"**Verdict:** {verdict}",
        "",
    ]

    for title, metrics in sections:
        out.append(f"## {title}")
        out.append("")
        out.append("| Metric | Value | Target | Status |")
        out.append("|---|---|---|---|")
        for name, val, tgt, status in metrics:
            icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[status]
            out.append(f"| {name} | `{val}` | {tgt} | {icon} {status} |")
        out.append("")

    out.append("## Known executives detail")
    out.append("")
    out.extend(exec_detail)
    out.append("")

    if "distributions" in extras:
        out.append("## Distributions")
        out.append("")
        for k, v in extras["distributions"].items():
            out.append(f"- **{k}**: {v}")
        out.append("")

    if extras.get("calibration"):
        c = extras["calibration"]
        out.append("## Calibration details")
        out.append("")
        out.append(f"- N persons with valid signal+style: {c.get('n')}")
        rd = c.get("rho_d")
        ra = c.get("rho_a")
        rc = c.get("rho_combined")
        out.append(f"- ρ(directiveness, authority_style) = "
                   f"`{rd:.3f}`" if rd is not None else "- ρ(directiveness): N/A")
        out.append(f"- ρ(agenda_setting, authority_style) = "
                   f"`{ra:.3f}`" if ra is not None else "- ρ(agenda_setting): N/A")
        out.append(f"- ρ(directiveness+agenda, authority_style) = "
                   f"`{rc:.3f}`" if rc is not None else "- ρ(combined): N/A")
        out.append("")

    if extras.get("role_summary"):
        rs = extras["role_summary"]
        out.append("## Role summary stats")
        out.append("")
        out.append(f"- Unique / total: {rs['unique']} / {rs['total']}")
        out.append(f"- Length: avg={rs['avg_len']:.0f}, median={rs['median_len']:.0f}, "
                   f"std={rs['std_len']:.0f}")
        out.append("")

    if extras.get("confidence_richness"):
        out.append("## Avg response length by confidence")
        out.append("")
        for c, avg in sorted(extras["confidence_richness"].items()):
            out.append(f"- confidence_self_report=`{c}` → {avg} chars avg")
        out.append("")

    return "\n".join(out)


def run() -> None:
    rows = load_rows()
    target = len(rows)  # 139 (after NaN/Steven Merriss cleanup)

    cov_metrics = section_coverage(rows, target)
    div_metrics, dist_extras = section_diversity(rows)
    exec_metrics, exec_detail = section_known_execs(rows)
    cal_metrics, cal_extras = section_calibration(rows)
    rs_metrics, rs_extras = section_diversity_role_summary(rows)
    conf_metrics, conf_extras = section_confidence_calibration(rows)

    sections = [
        ("Section A — Coverage", cov_metrics),
        ("Section B — Distribution diversity", div_metrics),
        ("Section C — Known executive sanity", exec_metrics),
        ("Section D — Signal-style calibration", cal_metrics),
        ("Section E — Role summary diversity", rs_metrics),
        ("Section F — Confidence calibration", conf_metrics),
    ]

    extras = {
        "distributions": dist_extras,
        "calibration": cal_extras,
        "role_summary": rs_extras,
        "confidence_richness": conf_extras.get("avg_response_chars_by_confidence"),
    }

    md = render_markdown(sections, extras, exec_detail)
    REPORT_PATH.write_text(md)

    print(md)
    print(f"\nReport saved to {REPORT_PATH}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 8a: QC of Stage-4 LLM persona enrichment")
    parser.parse_args(argv)
    run()


if __name__ == "__main__":
    main()
