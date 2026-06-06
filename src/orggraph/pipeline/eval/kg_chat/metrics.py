"""Aggregate per-question results into a summary."""

from __future__ import annotations
import statistics
from collections import Counter
from typing import Iterable


THRESHOLDS = {
    "joint": 0.65,
    "hierarchy": 0.80,
    "dominance": 0.60,
    "topics": 0.50,
    "dyadic": 0.50,
    "identity": 0.65,   # not pre-registered, but reported for completeness
    "escape": 0.50,
}


def aggregate(rows: Iterable[dict]) -> dict:
    """Each row: {result: RunResult, pass: bool, reason: str}.

    Returns a dict with:
      n_total, n_passed, joint_pass_rate
      per_category: {cat: {n, n_passed, pass_rate, threshold, meets_threshold}}
      mean_tool_calls, p50_wall_time_ms, max_wall_time_ms
      tool_usage: {tool_name: count}
      error_rate_by_tool: {tool_name: rate}
      thresholds_met: {joint, hierarchy, dominance, topics, dyadic, identity, escape}
    """
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return {"n_total": 0, "n_passed": 0, "joint_pass_rate": 0.0,
                "per_category": {}, "mean_tool_calls": 0.0,
                "p50_wall_time_ms": 0, "max_wall_time_ms": 0,
                "tool_usage": {}, "error_rate_by_tool": {},
                "thresholds_met": {}}

    n_passed = sum(1 for r in rows if r["pass"])
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["result"].category, []).append(r)

    per_category = {}
    for cat, cat_rows in by_cat.items():
        c_n = len(cat_rows)
        c_passed = sum(1 for r in cat_rows if r["pass"])
        rate = c_passed / max(1, c_n)
        threshold = THRESHOLDS.get(cat, 0.0)
        per_category[cat] = {
            "n": c_n,
            "n_passed": c_passed,
            "pass_rate": round(rate, 4),
            "threshold": threshold,
            "meets_threshold": rate >= threshold,
        }

    # Tool usage
    usage = Counter()
    errors = Counter()
    for r in rows:
        for tc in r["result"].tool_calls:
            usage[tc.name] += 1
            if tc.error:
                errors[tc.name] += 1
    error_rate = {name: round(errors[name] / count, 4) for name, count in usage.items()}

    wall_times = [r["result"].wall_time_ms for r in rows]
    mean_calls = sum(len(r["result"].tool_calls) for r in rows) / n

    joint = n_passed / n
    thresholds_met = {
        "joint": joint >= THRESHOLDS["joint"],
        **{cat: info["meets_threshold"] for cat, info in per_category.items()},
    }

    return {
        "n_total": n,
        "n_passed": n_passed,
        "joint_pass_rate": round(joint, 4),
        "per_category": per_category,
        "mean_tool_calls": round(mean_calls, 2),
        "p50_wall_time_ms": int(statistics.median(wall_times)),
        "max_wall_time_ms": max(wall_times),
        "tool_usage": dict(usage),
        "error_rate_by_tool": error_rate,
        "thresholds_met": thresholds_met,
    }
