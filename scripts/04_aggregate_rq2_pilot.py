"""Aggregate RQ2 pilot judge verdicts into a comparison table.

Usage:
    python scripts/04_aggregate_rq2_pilot.py

Reads every ``*__verdict.json`` under ``outputs/rq2_pilot/judge/`` and emits:

- ``summary.csv``  - long-format: one row per (scenario, condition, dimension)
- ``summary.md``   - headline comparison + per-scenario breakdown + turn-flag
  counts.

Only conditions present on disk are aggregated. Scenarios with one missing
condition are reported as such; they do not contribute to the paired delta
column.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from orggraph.evaluation.dialogue_judge import DIMENSIONS
from orggraph.simulation.scenario import SAMPLE_SCENARIOS

DEFAULT_JUDGE_DIR = Path(__file__).resolve().parents[1] / "outputs/rq2_pilot/judge"


def _load_verdicts(judge_dir: Path) -> dict[str, dict[str, dict]]:
    """Return ``{scenario_name: {condition: verdict_dict}}``."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in sorted(judge_dir.glob("*__verdict.json")):
        data = json.loads(p.read_text())
        out[data["scenario_name"]][data["condition"]] = data
    return dict(out)


def _flow_of(scenario_name: str) -> str:
    s = SAMPLE_SCENARIOS.get(scenario_name)
    return s.flow if s else "(unknown)"


def _write_csv(verdicts: dict[str, dict[str, dict]], out_path: Path) -> int:
    """Long-format CSV: one row per (scenario, condition, dimension)."""
    rows: list[dict[str, str]] = []
    for scen_name, by_cond in verdicts.items():
        flow = _flow_of(scen_name)
        for cond, v in by_cond.items():
            for dim in DIMENSIONS:
                s = v["scores"].get(dim, {})
                rows.append({
                    "scenario": scen_name,
                    "flow": flow,
                    "condition": cond,
                    "dimension": dim,
                    "score": s.get("score", ""),
                    "justification": s.get("justification", ""),
                })
            rows.append({
                "scenario": scen_name,
                "flow": flow,
                "condition": cond,
                "dimension": "_mean",
                "score": f"{v['mean_score']:.3f}",
                "justification": "",
            })
            rows.append({
                "scenario": scen_name,
                "flow": flow,
                "condition": cond,
                "dimension": "_turn_flag_count",
                "score": str(len(v.get("turn_flags", []))),
                "justification": "",
            })
    if not rows:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _per_dim_table(
    verdicts: dict[str, dict[str, dict]], conditions: list[str]
) -> str:
    """Per-dimension means by condition + delta (cond1 - cond2)."""
    per_cond: dict[str, dict[str, list[int]]] = {
        c: {d: [] for d in DIMENSIONS} for c in conditions
    }
    for by_cond in verdicts.values():
        for c in conditions:
            v = by_cond.get(c)
            if not v:
                continue
            for d in DIMENSIONS:
                s = v["scores"].get(d, {}).get("score")
                if s is not None:
                    per_cond[c][d].append(s)

    if not conditions:
        return "(no conditions found)"

    head = ["dimension"] + conditions
    if len(conditions) == 2:
        head.append(f"Δ ({conditions[0]} − {conditions[1]})")
    rows = [head]
    for d in DIMENSIONS:
        row = [d]
        means = []
        for c in conditions:
            vals = per_cond[c][d]
            m = mean(vals) if vals else float("nan")
            means.append(m)
            row.append(f"{m:.2f}" if vals else "-")
        if len(conditions) == 2 and all(per_cond[c][d] for c in conditions):
            row.append(f"{means[0] - means[1]:+.2f}")
        elif len(conditions) == 2:
            row.append("-")
        rows.append(row)

    # Mean across dimensions per condition
    overall = ["mean"]
    overall_means: list[float] = []
    for c in conditions:
        all_vals = [s for d in DIMENSIONS for s in per_cond[c][d]]
        m = mean(all_vals) if all_vals else float("nan")
        overall_means.append(m)
        overall.append(f"{m:.2f}" if all_vals else "-")
    if len(conditions) == 2 and all(overall_means):
        overall.append(f"{overall_means[0] - overall_means[1]:+.2f}")
    elif len(conditions) == 2:
        overall.append("-")
    rows.append(overall)

    return _md_table(rows)


def _per_scenario_table(
    verdicts: dict[str, dict[str, dict]], conditions: list[str]
) -> str:
    head = ["scenario", "flow"] + [f"{c} mean" for c in conditions]
    if len(conditions) == 2:
        head.append(f"Δ ({conditions[0]} − {conditions[1]})")
    head += [f"{c} flags" for c in conditions]
    rows = [head]
    for scen_name in sorted(verdicts):
        by_cond = verdicts[scen_name]
        flow = _flow_of(scen_name)
        row = [scen_name, flow]
        scores: list[float | None] = []
        for c in conditions:
            v = by_cond.get(c)
            scores.append(float(v["mean_score"]) if v else None)
            row.append(f"{v['mean_score']:.2f}" if v else "-")
        if len(conditions) == 2:
            if scores[0] is not None and scores[1] is not None:
                row.append(f"{scores[0] - scores[1]:+.2f}")
            else:
                row.append("-")
        for c in conditions:
            v = by_cond.get(c)
            row.append(str(len(v.get("turn_flags", []))) if v else "-")
        rows.append(row)
    return _md_table(rows)


def _md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    head = rows[0]
    sep = ["---"] * len(head)
    body = rows[1:]
    lines = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _write_md(verdicts: dict[str, dict[str, dict]], out_path: Path) -> None:
    if not verdicts:
        out_path.write_text("# RQ2 Pilot Aggregate\n\n(no verdicts on disk)\n")
        return
    # Stable canonical condition order: multi_agent first if present
    all_conds: list[str] = []
    for by_cond in verdicts.values():
        for c in by_cond:
            if c not in all_conds:
                all_conds.append(c)
    canonical: list[str] = []
    for c in ("multi_agent", "multi_agent_tools", "single_llm"):
        if c in all_conds:
            canonical.append(c)
    for c in all_conds:
        if c not in canonical:
            canonical.append(c)

    n_scen = len(verdicts)
    n_complete = sum(
        1 for by_cond in verdicts.values()
        if all(c in by_cond for c in canonical[:2])
    )

    lines = [
        "# RQ2 Pilot Aggregate",
        "",
        f"Scenarios aggregated: **{n_scen}**, with both conditions present in **{n_complete}** of them.",
        f"Conditions: {', '.join(canonical)}.",
        "",
        "## Per-dimension means (across scenarios)",
        "",
        _per_dim_table(verdicts, canonical),
        "",
        "## Per-scenario means",
        "",
        _per_scenario_table(verdicts, canonical),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--judge-dir",
        type=Path,
        default=DEFAULT_JUDGE_DIR,
        help=f"Judge verdict dir (default: {DEFAULT_JUDGE_DIR})",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="CSV output path (default: <judge-dir>/summary.csv)",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Markdown output path (default: <judge-dir>/summary.md)",
    )
    args = parser.parse_args(argv)

    if not args.judge_dir.is_dir():
        raise SystemExit(f"Judge dir not found: {args.judge_dir}")

    verdicts = _load_verdicts(args.judge_dir)
    if not verdicts:
        raise SystemExit(
            f"No *__verdict.json files in {args.judge_dir}. "
            f"Run scripts/03_judge_rq2_pilot.py --all first."
        )

    csv_path = args.csv_out or (args.judge_dir / "summary.csv")
    md_path = args.md_out or (args.judge_dir / "summary.md")

    n_rows = _write_csv(verdicts, csv_path)
    _write_md(verdicts, md_path)

    print(f"CSV  → {csv_path} ({n_rows} rows)")
    print(f"MD   → {md_path}")
    print()
    print(md_path.read_text())


if __name__ == "__main__":
    main()
