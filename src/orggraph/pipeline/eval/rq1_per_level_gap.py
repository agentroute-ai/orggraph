"""RQ1 per-level-gap accuracy breakdown.

Stratifies the A0-A5 ablation by the ordinal distance between the
superior and subordinate in each GT dominance pair, so a single
accuracy number is decomposed into "easy" pairs (CEO vs. analyst,
gap 4) and "hard" pairs (adjacent levels, gap 1).

Reuses the rq1 helpers (fetch_persons, _enrich_for_ablation,
fetch_dominance_pairs, f1_per_pair) so the scoring path is identical
to the headline ablation; only the aggregation differs.

Outputs:
    outputs/rq1_per_level_gap.csv      long-form per (stage, gap)
    thesis/figures/results/per_level_gap.tex   LaTeX tabular for Ch.6
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from orggraph.config import OUTPUT_DIR, REPO_ROOT
from orggraph.pipeline.eval.rq1 import (
    FEATURE_SETS,
    _STAGE_ORDER,
    _enrich_for_ablation,
    bootstrap_ci,
    f1_per_pair,
    fetch_dominance_pairs,
    fetch_persons,
)

log = logging.getLogger(__name__)

# Ordinal rank used to compute gaps. GT v2 uses 6 distinct tiers
# (Employee, Manager, Director, VP, SVP, C-Suite); the raw level_numeric
# in the CSV skips 5 and goes 0,1,2,3,4,6, so we use ordinal position
# rather than the literal numeric value.
_LEVEL_RANK = {
    "Employee": 1,
    "Manager": 2,
    "Director": 3,
    "VP": 4,
    "SVP": 5,
    "C-Suite": 6,
}

OUT_CSV = OUTPUT_DIR / "rq1_per_level_gap.csv"
OUT_TEX = REPO_ROOT / "thesis" / "figures" / "results" / "per_level_gap.tex"


def _add_gap_column(pairs: pd.DataFrame) -> pd.DataFrame:
    """Attach a `gap` column (1-4) to the pairs DataFrame.

    Pairs whose level strings don't map cleanly are dropped (should be
    zero rows on GT v2; logged when it happens).
    """
    sup = pairs["superior_level"].map(_LEVEL_RANK)
    sub = pairs["subordinate_level"].map(_LEVEL_RANK)
    out = pairs.assign(gap=sup - sub)
    dropped = out["gap"].isna().sum()
    if dropped:
        log.warning("Dropping %d pairs with unmapped level strings", dropped)
    return out.dropna(subset=["gap"]).assign(gap=lambda d: d["gap"].astype(int))


def _per_stage_correct(persons: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    """For each ablation rung, return the per-pair 0/1 correctness vector
    aligned to the order of `pairs`. Skips rungs whose features are not
    all populated in Neo4j (logged).
    """
    out: dict[str, np.ndarray] = {}
    for stage in _STAGE_ORDER:
        feats = FEATURE_SETS[stage]
        available = [f for f in feats if f in persons.columns and persons[f].notna().any()]
        missing = [f for f in feats if f not in available]
        if missing:
            log.warning("Stage %s missing features %s — skipping in gap breakdown", stage, missing)
            continue
        scores = dict(zip(persons["name"], persons[available].sum(axis=1).astype(float)))
        out[stage] = f1_per_pair(pairs, scores)
    return out


def _aggregate(pairs_with_gap: pd.DataFrame, per_stage: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build the long-form (stage, gap) accuracy table with bootstrap CIs."""
    gaps = sorted(pairs_with_gap["gap"].unique())
    rows: list[dict] = []
    for stage, correct in per_stage.items():
        for gap in gaps:
            mask = (pairs_with_gap["gap"] == gap).to_numpy()
            n = int(mask.sum())
            if n == 0:
                continue
            sub = correct[mask]
            acc = float(sub.mean())
            lo, _, hi = bootstrap_ci(sub)
            rows.append(
                {
                    "stage": stage,
                    "gap": int(gap),
                    "n_pairs": n,
                    "accuracy": acc,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        # Also store the overall row for consistency-checking against the
        # headline table.
        n = len(correct)
        acc = float(correct.mean())
        lo, _, hi = bootstrap_ci(correct)
        rows.append(
            {
                "stage": stage,
                "gap": 0,  # 0 == "all pairs"
                "n_pairs": n,
                "accuracy": acc,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def _to_latex(df: pd.DataFrame) -> str:
    """Render the per-gap breakdown as a compact LaTeX tabular.

    Rows: A0-A5. Columns: gap=1, gap=2, gap=3, gap=4, plus the overall
    headline accuracy in a final column. Each cell is the accuracy
    formatted to 3 decimal places.
    """
    gaps = sorted(g for g in df["gap"].unique() if g != 0)
    stages = [s for s in _STAGE_ORDER if s in df["stage"].unique()]

    pivot = df.pivot(index="stage", columns="gap", values="accuracy")
    counts = df[df["stage"] == stages[0]].set_index("gap")["n_pairs"]

    header = "Rung & " + " & ".join(
        [f"$\\Delta=${g} ($n=${counts.get(g, 0)})" for g in gaps]
    ) + f" & All ($n=${counts.get(0, 0)}) \\\\"

    lines = [
        r"% Generated by src/orggraph/pipeline/eval/rq1_per_level_gap.py",
        r"\begin{tabular}{l" + "r" * len(gaps) + "r}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for stage in stages:
        cells = [f"{pivot.loc[stage, g]:.3f}" for g in gaps]
        overall = df[(df["stage"] == stage) & (df["gap"] == 0)]["accuracy"].iloc[0]
        cells.append(f"{overall:.3f}")
        lines.append(f"{stage} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def run() -> pd.DataFrame:
    """Run the per-gap breakdown end-to-end and write artifacts.

    Returns the long-form DataFrame for further analysis / testing.
    """
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_pw = os.environ.get("NEO4J_PASSWORD", "orggraph2026")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
    persons = fetch_persons(driver)
    driver.close()
    persons = _enrich_for_ablation(persons)

    pairs = fetch_dominance_pairs()
    pairs_with_gap = _add_gap_column(pairs)

    per_stage = _per_stage_correct(persons, pairs_with_gap)
    if not per_stage:
        raise RuntimeError("No ablation rungs could be scored; Neo4j features missing.")

    df = _aggregate(pairs_with_gap, per_stage)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    log.info("Wrote per-gap breakdown CSV: %s", OUT_CSV)

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(_to_latex(df))
    log.info("Wrote LaTeX tabular: %s", OUT_TEX)

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress INFO log lines"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    df = run()

    # Compact human-readable summary
    print()
    print("=" * 64)
    print("RQ1 — accuracy by hierarchy gap (Δ = level_sup − level_sub)")
    print("=" * 64)
    pivot = df.pivot(index="stage", columns="gap", values="accuracy")
    counts = df[df["stage"] == df["stage"].iloc[0]].set_index("gap")["n_pairs"]
    cols = [c for c in pivot.columns if c != 0]
    header = f"{'Rung':<6}" + "".join(f"  Δ={g} (n={counts[g]:>4})" for g in cols) + f"  ALL (n={counts.get(0, 0)})"
    print(header)
    print("-" * len(header))
    for stage in _STAGE_ORDER:
        if stage not in pivot.index:
            continue
        row = pivot.loc[stage]
        overall = df[(df["stage"] == stage) & (df["gap"] == 0)]["accuracy"].iloc[0]
        line = f"{stage:<6}" + "".join(f"  {row[g]:>11.3f}" for g in cols) + f"  {overall:>11.3f}"
        print(line)
    print()
    print(f"Long-form CSV : {OUT_CSV}")
    print(f"LaTeX tabular : {OUT_TEX}")


if __name__ == "__main__":
    main()
