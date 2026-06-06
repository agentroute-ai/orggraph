"""Render the eval summary + failure table as Markdown."""

from __future__ import annotations
from typing import Iterable


CATEGORY_ORDER = ["hierarchy", "identity", "dominance", "dyadic", "topics", "escape"]


def render_report(*, summary: dict, rows: Iterable[dict], worst_n: int = 15) -> str:
    """rows: list of {result: RunResult, pass: bool, reason: str}."""
    rows = list(rows)
    lines: list[str] = []
    lines.append("# KG Chat structural eval")
    lines.append("")
    lines.append("## Top-line")
    lines.append("")
    lines.append(f"- Questions total: **{summary['n_total']}**")
    lines.append(f"- Passed: **{summary['n_passed']}**")
    joint = summary['joint_pass_rate']
    joint_t = 0.65
    joint_mark = "✓" if joint >= joint_t else "✗"
    lines.append(f"- Joint pass rate: **{joint:.1%}** (threshold {joint_t:.0%}) {joint_mark}")
    lines.append(f"- Mean tool calls per question: {summary['mean_tool_calls']}")
    lines.append(f"- p50 wall time: {summary['p50_wall_time_ms']} ms")
    lines.append(f"- Max wall time: {summary['max_wall_time_ms']} ms")
    lines.append("")

    lines.append("## Per category")
    lines.append("")
    lines.append("| Category | n | Passed | Rate | Threshold | Met |")
    lines.append("| --- | ---: | ---: | ---: | ---: | :-: |")
    for cat in CATEGORY_ORDER:
        info = summary["per_category"].get(cat)
        if not info:
            continue
        mark = "✓" if info["meets_threshold"] else "✗"
        lines.append(
            f"| {cat} | {info['n']} | {info['n_passed']} | {info['pass_rate']:.1%} | "
            f"{info['threshold']:.0%} | {mark} |"
        )
    lines.append("")

    lines.append("## Tool usage")
    lines.append("")
    lines.append("| Tool | Calls | Error rate |")
    lines.append("| --- | ---: | ---: |")
    for name, count in sorted(summary["tool_usage"].items(), key=lambda kv: -kv[1]):
        err = summary["error_rate_by_tool"].get(name, 0.0)
        lines.append(f"| {name} | {count} | {err:.1%} |")
    lines.append("")

    failures = [r for r in rows if not r["pass"]]
    failures.sort(key=lambda r: r["result"].question_id)
    lines.append(f"## Failures ({len(failures)})")
    lines.append("")
    if not failures:
        lines.append("None — all questions passed.")
    else:
        lines.append("| qid | category | reason | tools used | final body (truncated) |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in failures[:worst_n]:
            tools_used = ",".join(tc.name for tc in r["result"].tool_calls) or "(none)"
            body = (r["result"].final_body or "").replace("\n", " ").replace("|", "\\|")
            if len(body) > 80:
                body = body[:79] + "…"
            lines.append(f"| {r['result'].question_id} | {r['result'].category} | "
                         f"{r['reason']} | {tools_used} | {body} |")
    lines.append("")

    lines.append("## Threshold provenance")
    lines.append("")
    lines.append("Thresholds were committed at SHA a29e8e7 before any runner code "
                 "executed; see `docs/superpowers/specs/2026-05-24-structural-eval-questions-draft.md` "
                 "for the pre-run reasoning.")
    lines.append("")
    return "\n".join(lines)
