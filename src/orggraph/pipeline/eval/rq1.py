"""Stage B.3 — RQ1 evaluation with bootstrap confidence intervals.

Reads enriched Neo4j and ground-truth CSVs. For each ablation stage,
computes pair-classification accuracy, Spearman rho, and NMI with 95%
bootstrap CIs and paired-bootstrap deltas vs A0.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score

from orggraph.config import OUTPUT_DIR

OUT_CSV = OUTPUT_DIR / "rq1_ablation.csv"
GT_PAIRS_FILE = "dominance_pairs_v2.csv"  # EnronData GT v2 (3,814 pairs); falls back to legacy
GT_LEVELS_FILE = "employees_ground_truth_v2.csv"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A0-A5 ablation feature sets
# Each stage is a strict superset of the previous (cumulative ladder).
# ---------------------------------------------------------------------------

_A0 = ["pagerank", "betweenness", "in_degree", "response_time", "cc_frequency"]

_A1 = _A0 + [
    "is_thread_initiator_rate",
    "mean_to_count",
    "pct_off_hours",
    "mean_body_words",
]

_A2 = _A1 + [
    "pct_request_sent",
    "pct_commit_sent",
    "pct_decision_carrying",
    "pct_action_required",
]

_A3 = _A2 + [
    "request_commit_ratio",
    "length_asymmetry",
    "mean_reply_latency_h",
]

_A4 = _A3 + [
    "project_one_hot",
    "topic_embedding",
]

_A5 = _A4 + [
    "seniority_narrative_numeric",
    "authority_style_one_hot",
]

FEATURE_SETS: dict[str, list[str]] = {
    "A0": _A0,
    "A1": _A1,
    "A2": _A2,
    "A3": _A3,
    "A4": _A4,
    "A5": _A5,
}

_STUB_RESULT: dict = {
    "f1": 0.0,
    "f1_ci": (0.0, 0.0),
    "spearman": 0.0,
    "nmi": 0.0,
    "delta_f1_vs_a0": 0.0,
    "p_value": 1.0,
}

_STAGE_ORDER = ["A0", "A1", "A2", "A3", "A4", "A5"]

_STAGE_DESCRIPTIONS: dict[str, str] = {
    "A0": "Pure structural — graph centrality (PageRank + betweenness + in-degree)",
    "A1": "+ Stage 1.5 deterministic email patterns (thread-init, recipient count, off-hours, body length)",
    "A2": "+ Stage 2a LLM speech-act %ages (request, commit, decision-carrying, action-required)",
    "A3": "+ Stage 3 dyadic asymmetry (request/commit ratio; length-asym + reply-latency are pair-level, zero-fill)",
    "A4": "+ Stage 2b cluster memberships (project + topic — currently zero-filled, awaiting Email→cluster traversal)",
    "A5": "+ Stage 4a LLM persona signals (canonical_role_level when the canonicalization sidecar is present, else proxied by persona_directiveness + persona_formality)",
}


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Constant arrays return zeros."""
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _enrich_for_ablation(persons: pd.DataFrame) -> pd.DataFrame:
    """Populate every feature name expected by ``FEATURE_SETS``.

    Some features are direct Neo4j renames (``pct_thread_initiator`` →
    ``is_thread_initiator_rate``); some are derived from per-Person
    counts (``pct_action_required`` from ``n_action_required_sent`` /
    ``n_emails_sent``); a few are genuinely pair-level (``length_asymmetry``,
    ``mean_reply_latency_h``, ``response_time``) and zero-filled here so
    the rung that "should" include them runs at the prior rung's signal
    strength rather than returning a stub. Two A5 features
    (``seniority_narrative_numeric``, ``authority_style_one_hot``) are
    proxied by their numeric LLM-persona counterparts since the original
    text columns aren't directly summable.

    All features are then min-max normalised so the score sum is not
    dominated by features with larger absolute scales (PageRank ~0.01
    vs ``mean_body_words`` ~100).
    """
    p = persons.copy()

    # ------------- Renames (A0 / A1 directly) -----------------
    if "pct_thread_initiator" in p.columns:
        p["is_thread_initiator_rate"] = p["pct_thread_initiator"]

    # ------------- Derivations ---------------------------------
    n_emails = p.get("n_emails_sent", pd.Series([0.0] * len(p))).fillna(0.0).clip(lower=1.0)
    if "n_action_required_sent" in p.columns:
        p["pct_action_required"] = p["n_action_required_sent"].fillna(0.0) / n_emails
    if {"pct_request_sent", "pct_commit_sent"}.issubset(p.columns):
        p["request_commit_ratio"] = p["pct_request_sent"].fillna(0.0) / (
            p["pct_commit_sent"].fillna(0.0) + 0.01
        )

    # ------------- Pair-level features: zero-fill --------------
    # response_time, cc_frequency, length_asymmetry, mean_reply_latency_h
    # are pair-level signals (sender→recipient asymmetries). They live
    # in the pair_signals table, not on Person, so they cannot
    # contribute to a per-Person score directly. Zero-filling means the
    # rung runs but those features add no signal — equivalent to "this
    # rung tested only the persisted per-Person signals".
    for col in ("response_time", "cc_frequency", "length_asymmetry", "mean_reply_latency_h"):
        if col not in p.columns:
            p[col] = 0.0

    # ------------- A4 features: cluster memberships ------------
    # project_one_hot / topic_embedding need feature engineering from
    # Email→Project / Email→Topic relationships in Neo4j (not yet in
    # the per-Person property set). Zero-fill until that lands.
    for col in ("project_one_hot", "topic_embedding"):
        if col not in p.columns:
            p[col] = 0.0

    # ------------- A5 features: canonical role level + style ---
    # The pre-registered A5 rung consumes a numeric seniority signal and
    # an authority-style proxy. After the role-canonicalization addendum
    # (docs/plans/2026-05-08-role-canonicalization.md), the per-Person
    # canonical_role_level ordinal is the right source for the seniority
    # signal: it maps directly to the GT v2 level_numeric scale and is
    # not contaminated by the per-email directiveness pattern (which is
    # closer to a behavioural-style signal than a position signal).
    #
    # Fallback order:
    #   1. canonical_role_level (post-canonicalization sidecar)
    #   2. persona_directiveness (the pre-canonicalization proxy)
    #   3. 0.0 (no signal)
    if "canonical_role_level" in p.columns:
        p["seniority_narrative_numeric"] = p["canonical_role_level"].fillna(0.0)
    elif "persona_directiveness" in p.columns:
        p["seniority_narrative_numeric"] = p["persona_directiveness"].fillna(0.0)
    else:
        p["seniority_narrative_numeric"] = 0.0
    if "persona_formality" in p.columns:
        p["authority_style_one_hot"] = p["persona_formality"].fillna(0.0)
    else:
        p["authority_style_one_hot"] = 0.0

    # ------------- Min-max normalise all feature columns -------
    all_features: set[str] = set()
    for feats in FEATURE_SETS.values():
        all_features.update(feats)
    for col in all_features:
        if col in p.columns:
            p[col] = _minmax(p[col].fillna(0.0).to_numpy())

    return p


def run_ablation_ladder(
    stage_filter: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Run the A0-A5 ablation ladder and return per-stage metrics.

    Parameters
    ----------
    stage_filter:
        Subset of stages to run, e.g. ``["A0", "A1"]``.  ``None`` runs all.
    dry_run:
        When *True* every stage returns the stub result without touching Neo4j
        or any data files.  Useful for CI / smoke testing.

    Returns
    -------
    dict mapping stage key to a metrics dict with keys:
    ``f1``, ``f1_ci``, ``spearman``, ``nmi``, ``delta_f1_vs_a0``, ``p_value``.
    """
    stages = stage_filter if stage_filter is not None else _STAGE_ORDER
    # Preserve canonical ordering
    stages = [s for s in _STAGE_ORDER if s in stages]

    if dry_run:
        return {stage: dict(_STUB_RESULT) for stage in stages}

    # ------------------------------------------------------------------
    # Non-dry-run path: attempt to fetch data from Neo4j/CSV.
    # Stages whose required features are not yet written to Neo4j will
    # gracefully fall back to the stub result with a warning.
    # ------------------------------------------------------------------
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_pw = os.environ.get("NEO4J_PASSWORD", "orggraph2026")

    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
        persons = fetch_persons(driver)
        driver.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("Cannot connect to Neo4j (%s) — returning stubs for all stages.", exc)
        return {stage: dict(_STUB_RESULT) for stage in stages}

    persons = _enrich_for_ablation(persons)

    try:
        pairs = fetch_dominance_pairs()
        gt_levels = fetch_gt_levels()
    except Exception as exc:  # noqa: BLE001
        log.warning("Cannot load ground-truth files (%s) — returning stubs for all stages.", exc)
        return {stage: dict(_STUB_RESULT) for stage in stages}

    results: dict[str, dict] = {}
    a0_correct: np.ndarray | None = None

    for stage in stages:
        features = FEATURE_SETS[stage]
        # Check which features are available as columns on the persons DataFrame.
        available = [f for f in features if f in persons.columns and persons[f].notna().any()]
        missing = [f for f in features if f not in available]

        if missing:
            log.warning(
                "Stage %s: features not yet in Neo4j: %s — using stub result.",
                stage,
                missing,
            )
            results[stage] = dict(_STUB_RESULT)
            continue

        try:
            score_series = persons[available].sum(axis=1)
            scores = dict(zip(persons["name"], score_series.astype(float)))
            correct = f1_per_pair(pairs, scores)
            f1_val = float(correct.mean())
            lo, _, hi = bootstrap_ci(correct)

            if a0_correct is None:
                a0_correct = correct
                delta = 0.0
                p_value = 1.0
            else:
                delta_lo, delta_md, delta_hi = paired_bootstrap_ci(correct, a0_correct)
                delta = delta_md
                p_value = paired_bootstrap_pvalue(correct, a0_correct)

            results[stage] = {
                "f1": f1_val,
                "f1_ci": (lo, hi),
                "spearman": hierarchy_spearman(scores, gt_levels),
                "nmi": community_nmi(
                    gt_levels,
                    dict(
                        zip(
                            persons["name"],
                            persons["community"].fillna(-1).astype(int)
                            if "community" in persons.columns
                            else [-1] * len(persons),
                        )
                    ),
                ),
                "delta_f1_vs_a0": delta,
                "p_value": p_value,
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("Stage %s: scoring failed (%s) — using stub result.", stage, exc)
            results[stage] = dict(_STUB_RESULT)

    return results


# Reference rows from cited prior work
# NOTE: Agarwal reports pair-classification accuracy (NOT F1, despite the
# column name being kept here for backward compatibility). The 0.7931
# Core-Core figure is the closest like-for-like comparison since our
# extracted graph is custodian-only.
BASELINE_REFS = [
    {
        "stage": "Ref-A-CoreCore",
        "method": "Agarwal et al. (2012) — SNA degree (Core-Core, 440 pairs)",
        "f1": 0.7931,
        "spearman": None,
        "nmi": None,
        "note": "Closest like-for-like comparison to our custodian-only graph",
    },
    {
        "stage": "Ref-A-Full",
        "method": "Agarwal et al. (2012) — SNA degree (all 13,724 pairs)",
        "f1": 0.8388,
        "spearman": None,
        "nmi": None,
        "note": "Includes 1,360 non-core people not present in our extracted graph",
    },
    {
        "stage": "Ref-C",
        "method": "Creamer et al. (2022)",
        "f1": None,
        "spearman": None,
        "nmi": None,
        "note": "social-power scores; different protocol",
    },
]


def fetch_persons(driver) -> pd.DataFrame:
    """Pull every Neo4j Person property the ablation feature map can use,
    and merge the canonical-role sidecar (per the role-canonicalization
    addendum in docs/plans/2026-05-08-role-canonicalization.md) when it
    exists on disk.

    See :func:`_enrich_for_ablation` for how these columns are renamed,
    derived, and normalised before scoring.
    """
    q = """
    MATCH (p:Person)
    RETURN p.name AS name,
           p.pagerank AS pagerank,
           p.betweenness AS betweenness,
           p.in_degree AS in_degree,
           p.composite_score AS composite_v2,
           p.composite_score_v3 AS composite_v3,
           p.community AS community,
           p.function AS function,
           p.pct_thread_initiator AS pct_thread_initiator,
           p.mean_to_count AS mean_to_count,
           p.pct_off_hours AS pct_off_hours,
           p.mean_body_words AS mean_body_words,
           p.pct_request_sent AS pct_request_sent,
           p.pct_commit_sent AS pct_commit_sent,
           p.pct_decision_carrying AS pct_decision_carrying,
           p.n_action_required_sent AS n_action_required_sent,
           p.n_emails_sent AS n_emails_sent,
           p.persona_directiveness AS persona_directiveness,
           p.persona_agenda_setting AS persona_agenda_setting,
           p.persona_formality AS persona_formality,
           p.persona_verbosity AS persona_verbosity
    """
    with driver.session() as s:
        df = pd.DataFrame([dict(r) for r in s.run(q)])

    # Merge canonical-role sidecar when present. The sidecar is produced
    # by orggraph-enrich-canonicalize and is keyed on Person `name`.
    sidecar = OUTPUT_DIR / "person_enrichment_canonical.csv"
    if sidecar.is_file():
        try:
            canon = pd.read_csv(sidecar)
            keep_cols = ["name", "canonical_role", "canonical_role_level"]
            canon = canon[[c for c in keep_cols if c in canon.columns]]
            canon["canonical_role_level"] = pd.to_numeric(
                canon["canonical_role_level"], errors="coerce"
            )
            df = df.merge(canon, on="name", how="left")
        except Exception:  # noqa: BLE001
            # Sidecar exists but unparseable; leave df unchanged.
            pass

    return df


def fetch_dominance_pairs() -> pd.DataFrame:
    """Prefer the EnronData GT v2 (3,814 pairs over 103 employees) when
    available; fall back to the legacy 30-person hand-curated GT
    (349 pairs)."""
    p_v2 = OUTPUT_DIR / GT_PAIRS_FILE
    if p_v2.exists():
        return pd.read_csv(p_v2)
    return pd.read_csv(OUTPUT_DIR / "dominance_pairs.csv")


def fetch_gt_levels() -> dict[str, int]:
    p_v2 = OUTPUT_DIR / GT_LEVELS_FILE
    p = p_v2 if p_v2.exists() else OUTPUT_DIR / "employees_ground_truth.csv"
    df = pd.read_csv(p)
    return dict(zip(df["name"], df["level_numeric"]))


def dominance_f1(pairs: pd.DataFrame, scores: dict[str, float]) -> float:
    """Fraction of dominance pairs correctly ordered by scores."""
    if pairs.empty:
        return 0.0
    correct = 0
    for _, row in pairs.iterrows():
        sup, sub = row["superior"], row["subordinate"]
        if sup in scores and sub in scores and scores[sup] > scores[sub]:
            correct += 1
    return correct / len(pairs)


def hierarchy_spearman(scores: dict[str, float], gt: dict[str, int]) -> float:
    """Spearman correlation between predicted scores and ground-truth levels."""
    common = set(scores) & set(gt)
    if len(common) < 3:
        return 0.0
    a = [scores[k] for k in common]
    b = [gt[k] for k in common]
    rho, _ = spearmanr(a, b)
    return float(rho) if not np.isnan(rho) else 0.0


def community_nmi(true_labels: dict, pred_labels: dict) -> float:
    """Normalised Mutual Information between two label dicts."""
    common = sorted(set(true_labels) & set(pred_labels))
    if len(common) < 2:
        return 0.0
    y_t = [true_labels[k] for k in common]
    y_p = [pred_labels[k] for k in common]
    return float(normalized_mutual_info_score(y_t, y_p))


def bootstrap_ci(
    values: np.ndarray, B: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Return (low2.5, median, high97.5) bootstrap percentiles of an array."""
    rng = np.random.default_rng(seed)
    boots = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(B)]
    )
    return (
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 50)),
        float(np.percentile(boots, 97.5)),
    )


def _paired_bootstrap_means(
    a: np.ndarray, b: np.ndarray, B: int = 1000, seed: int = 42
) -> np.ndarray:
    """Return the bootstrap distribution of the paired mean delta (a - b).

    Resamples the per-element delta vector ``a - b`` with replacement ``B``
    times and returns the mean of each resample. Used as the shared engine
    for ``paired_bootstrap_ci`` (percentile CI) and ``paired_bootstrap_pvalue``
    (one-sided proportion).
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = a - b
    return np.array(
        [rng.choice(deltas, size=n, replace=True).mean() for _ in range(B)]
    )


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, B: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """CI on the per-element delta (a - b). a and b must be aligned."""
    boots = _paired_bootstrap_means(a, b, B=B, seed=seed)
    return (
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 50)),
        float(np.percentile(boots, 97.5)),
    )


def paired_bootstrap_pvalue(
    a: np.ndarray, b: np.ndarray, B: int = 1000, seed: int = 42
) -> float:
    """One-sided paired-bootstrap p-value for H1: mean(a) > mean(b).

    Returns the fraction of bootstrap resamples in which the mean delta
    (a - b) is ≤ 0. Small values indicate evidence that a is reliably
    larger than b across paired observations.
    """
    boots = _paired_bootstrap_means(a, b, B=B, seed=seed)
    return float(np.mean(boots <= 0))


def f1_per_pair(pairs: pd.DataFrame, scores: dict[str, float]) -> np.ndarray:
    """Return a 0/1 vector indicating per-pair correctness for paired bootstrap."""
    out = np.zeros(len(pairs))
    for i, row in enumerate(pairs.itertuples(index=False)):
        sup = getattr(row, "superior")
        sub = getattr(row, "subordinate")
        if sup in scores and sub in scores and scores[sup] > scores[sub]:
            out[i] = 1
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RQ1 evaluation — ablation ladder")
    parser.add_argument(
        "--stages",
        default=",".join(_STAGE_ORDER),
        help="Comma-separated ablation stages to run, e.g. A0,A1,A2 (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Return stub results without connecting to Neo4j (for testing)",
    )
    args, _ = parser.parse_known_args(argv)

    stage_filter = [s.strip() for s in args.stages.split(",") if s.strip()]

    if args.dry_run:
        table = run_ablation_ladder(stage_filter=stage_filter, dry_run=True)
        ablation_df = pd.DataFrame(
            [{"stage": k, **v} for k, v in table.items()]
        )
        print("\n=== Ablation ladder (dry-run stubs) ===")
        print(ablation_df.to_string(index=False))
        return

    table = run_ablation_ladder(stage_filter=stage_filter)

    rows: list[dict] = []
    for ref in BASELINE_REFS:
        rows.append(
            {
                "stage": ref["stage"],
                "method": ref["method"],
                "pair_accuracy": ref.get("f1"),
                "f1_ci_lo": None,
                "f1_ci_hi": None,
                "spearman": ref.get("spearman"),
                "nmi": ref.get("nmi"),
                "delta_vs_a0": None,
                "p_value": None,
                "note": ref.get("note"),
            }
        )
    for stage, m in table.items():
        rows.append(
            {
                "stage": stage,
                "method": _STAGE_DESCRIPTIONS.get(stage, stage),
                "pair_accuracy": m["f1"],
                "f1_ci_lo": m["f1_ci"][0],
                "f1_ci_hi": m["f1_ci"][1],
                "spearman": m["spearman"],
                "nmi": m["nmi"],
                "delta_vs_a0": m["delta_f1_vs_a0"],
                "p_value": m["p_value"],
                "note": "",
            }
        )

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    print("\n=== A0–A5 ablation table ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
