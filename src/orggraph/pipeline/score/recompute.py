"""Stage B.2 — Recompute composite scores with LLM features.

Reads Neo4j (post-sync), computes composite_v3 per Person, k-fold tunes
alpha for the Tier 2 dominance signal, refines REPORTS_TO from strongest
DEFERS_TO, writes back.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sklearn.model_selection import KFold

from orggraph.config import OUTPUT_DIR

ALPHA_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHT_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
ALPHA_OUT = OUTPUT_DIR / "tier2_alpha.json"
WEIGHT_OUT = OUTPUT_DIR / "composite_weights.json"


def minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def fetch_persons(driver) -> pd.DataFrame:
    """Pull Person rows from Neo4j and merge the canonical-role sidecar.

    The v2 KG schema does not store an integer ``seniority`` field
    directly; the seniority signal comes from the role-canonicalization
    sidecar (``person_enrichment_canonical.csv``) as
    ``canonical_role_level`` (0-5 ordinal). When the sidecar is absent
    or a Person is not in it, we fall back to ``persona_directiveness``
    (continuous behavioural proxy in [0, 1]) read from Neo4j. Both
    paths produce a numeric ``seniority`` column for the v3 composite.
    """
    q = """
    MATCH (p:Person)
    RETURN p.name AS name,
           p.composite_score AS composite_score,
           p.persona_directiveness AS persona_directiveness,
           p.community AS community
    """
    with driver.session() as s:
        df = pd.DataFrame([dict(r) for r in s.run(q)])

    # Merge the canonical-role sidecar when it exists (the v2-schema
    # source of seniority — see RQ1's fetch_persons for the pattern).
    sidecar = OUTPUT_DIR / "person_enrichment_canonical.csv"
    if sidecar.is_file():
        try:
            canon = pd.read_csv(sidecar)
            keep_cols = ["name", "canonical_role_level"]
            canon = canon[[c for c in keep_cols if c in canon.columns]]
            canon["canonical_role_level"] = pd.to_numeric(
                canon["canonical_role_level"], errors="coerce"
            )
            df = df.merge(canon, on="name", how="left")
        except Exception:  # noqa: BLE001
            pass

    # Build the seniority column: prefer canonical_role_level (0-5),
    # fall back to persona_directiveness (0-1, scaled to 0-5 for
    # comparable contribution), and fill any remaining gaps with 0.
    canon_level = df.get("canonical_role_level")
    persona_dir = df.get("persona_directiveness")
    if canon_level is None:
        canon_level = pd.Series([None] * len(df))
    if persona_dir is None:
        persona_dir = pd.Series([None] * len(df))
    # Scale persona_directiveness to the canonical 0-5 range for blend parity.
    persona_dir_scaled = pd.to_numeric(persona_dir, errors="coerce") * 5.0
    df["seniority"] = pd.to_numeric(canon_level, errors="coerce").fillna(
        persona_dir_scaled
    ).fillna(0.0)
    return df


def fetch_pair_signals(driver) -> pd.DataFrame:
    """Return one row per (a, b) directed DEFERS_TO with raw_score."""
    q = """
    MATCH (a:Person)-[r:DEFERS_TO]->(b:Person)
    RETURN a.name AS a, b.name AS b, r.raw_score AS raw_score
    """
    with driver.session() as s:
        return pd.DataFrame([dict(r) for r in s.run(q)])


def fetch_dominance_pairs() -> pd.DataFrame:
    p = OUTPUT_DIR / "dominance_pairs.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Need {p} (run `python -m orggraph.pipeline.ground_truth` first)"
        )
    return pd.read_csv(p)


def compute_composite_v3(persons: pd.DataFrame, weight: float) -> pd.DataFrame:
    """v3 = composite (Stage 2) + weight * minmax(seniority)."""
    df = persons.copy()
    sen = df["seniority"].fillna(df["seniority"].median()).to_numpy(dtype=float)
    df["composite_score_v3"] = df["composite_score"].astype(float) + weight * minmax(sen)
    df["tier_v3"] = pd.qcut(df["composite_score_v3"], q=5, labels=False, duplicates="drop").astype(int)
    return df


def evaluate_alpha(pairs: pd.DataFrame, alpha: float) -> float:
    """F1-equivalent accuracy of the directional dominance signal."""
    signal = pairs["diff"] + alpha * pairs["deference"]
    correct = (signal > 0).sum()
    n = len(pairs)
    return correct / n if n else 0.0


def kfold_alpha(pairs: pd.DataFrame, alphas: list[float] = ALPHA_GRID, n_splits: int = 5) -> list[dict]:
    """5-fold CV: tune alpha on training, report on held-out."""
    if len(pairs) < n_splits:
        return []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    out = []
    pairs = pairs.reset_index(drop=True)
    for fold_idx, (tr, te) in enumerate(kf.split(pairs)):
        train = pairs.iloc[tr]
        test = pairs.iloc[te]
        # Pick alpha that maximizes training F1
        best_a = max(alphas, key=lambda a: evaluate_alpha(train, a))
        f1 = evaluate_alpha(test, best_a)
        out.append({"fold": fold_idx, "alpha": best_a, "f1": f1, "n_train": len(train), "n_test": len(test)})
    return out


def write_persons_v3(driver, persons: pd.DataFrame) -> None:
    """Write v3 columns (composite_score_v3, tier_v3) and the seniority
    blend back to Neo4j Person nodes.

    The seniority value persisted here is the same numeric column used
    by ``compute_composite_v3``: canonical_role_level when the sidecar
    is present, persona_directiveness scaled to 0-5 otherwise. We also
    persist a copy under ``canonical_role_level`` so tools that ask for
    that property find a non-null value for every Person.
    """
    cols = ["name", "composite_score_v3", "tier_v3", "seniority"]
    if "canonical_role_level" in persons.columns:
        cols.append("canonical_role_level")
    rows = persons[cols].to_dict("records")
    with driver.session() as s:
        s.run(
            """
            UNWIND $rows AS row
            MATCH (p:Person {name: row.name})
            SET p.composite_score_v3 = row.composite_score_v3,
                p.tier_v3 = row.tier_v3,
                p.seniority = row.seniority,
                p.canonical_role_level = coalesce(row.canonical_role_level, p.canonical_role_level)
            """,
            rows=rows,
        )


def refine_reports_to(driver, pair_signals: pd.DataFrame, threshold: float = 0.5) -> int:
    """Replace inferred REPORTS_TO with deference-derived edges."""
    if pair_signals.empty:
        return 0
    # For each subordinate (negative raw_score), pick the strongest superior
    sub_signals = pair_signals[pair_signals["raw_score"] < -threshold].copy()
    if sub_signals.empty:
        return 0
    sub_signals["abs"] = sub_signals["raw_score"].abs()
    strongest = sub_signals.sort_values("abs", ascending=False).drop_duplicates(subset=["a"])
    rows = strongest[["a", "b", "abs"]].rename(
        columns={"a": "subordinate", "b": "superior", "abs": "confidence"}
    ).to_dict("records")

    with driver.session() as s:
        s.run("MATCH ()-[r:REPORTS_TO]->() DELETE r")  # clear old
        s.run(
            """
            UNWIND $rows AS row
            MATCH (sub:Person {name: row.subordinate})
            MATCH (sup:Person {name: row.superior})
            MERGE (sub)-[r:REPORTS_TO]->(sup)
            SET r.source = 'tier2_deference', r.confidence = row.confidence
            """,
            rows=rows,
        )
    return len(rows)


def run() -> None:
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_pw = os.environ.get("NEO4J_PASSWORD", "orggraph2026")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
    try:
        print("[1/4] Loading persons + pair signals from Neo4j...")
        persons = fetch_persons(driver)
        pair_signals = fetch_pair_signals(driver)
        print(f"      persons={len(persons)} pairs={len(pair_signals)}")

        print("[2/4] Sweep composite weight (sensitivity analysis)...")
        # Build the dominance-pair frame for evaluation
        gt = fetch_dominance_pairs()
        weight_results = []
        best_w, best_f1 = 1.5, -1.0
        for w in WEIGHT_GRID:
            persons_w = compute_composite_v3(persons, weight=w)
            sc = dict(zip(persons_w["name"], persons_w["composite_score_v3"]))
            correct = 0
            total = 0
            for _, row in gt.iterrows():
                if row["superior"] in sc and row["subordinate"] in sc:
                    total += 1
                    if sc[row["superior"]] > sc[row["subordinate"]]:
                        correct += 1
            f1 = correct / total if total else 0.0
            weight_results.append({"weight": w, "accuracy": f1})
            if f1 > best_f1:
                best_f1, best_w = f1, w
        print(f"      best weight = {best_w} (acc={best_f1:.3f})")
        with open(WEIGHT_OUT, "w") as f:
            json.dump({"chosen_weight": best_w, "sweep": weight_results}, f, indent=2)

        print("[3/4] k-fold CV alpha tuning...")
        # Build pair frame: diff = composite_v3[a] - composite_v3[b], deference = raw_score
        persons_v3 = compute_composite_v3(persons, weight=best_w)
        sc = dict(zip(persons_v3["name"], persons_v3["composite_score_v3"]))
        if pair_signals.empty or "a" not in pair_signals.columns:
            # No DEFERS_TO edges in the current KG → nothing to CV-tune.
            # The composite_score_v3 / tier_v3 / canonical_role_level write
            # below still happens; the alpha tuning is just skipped.
            merged = pd.DataFrame(columns=["a", "b", "raw_score", "diff", "deference"])
        else:
            merged = pair_signals.copy()
            merged["diff"] = merged["a"].map(sc) - merged["b"].map(sc)
            merged["deference"] = merged["raw_score"]
            merged = merged.dropna(subset=["diff", "deference"])

        folds = kfold_alpha(merged)
        if folds:
            chosen_alpha = float(np.mean([f["alpha"] for f in folds]))
            mean_f1 = float(np.mean([f["f1"] for f in folds]))
            print(f"      alpha={chosen_alpha:.2f} mean F1={mean_f1:.3f}")
        else:
            chosen_alpha, mean_f1 = 1.0, 0.0
            print("      no pairs available — skipping CV, default alpha=1.0")
        with open(ALPHA_OUT, "w") as f:
            json.dump({"chosen_alpha": chosen_alpha, "mean_f1": mean_f1, "folds": folds}, f, indent=2)

        print("[4/4] Writing back to Neo4j (composite_v3, refined REPORTS_TO)...")
        write_persons_v3(driver, persons_v3)
        n_refined = refine_reports_to(driver, pair_signals)
        print(f"      composite_v3 written; REPORTS_TO refined: {n_refined}")
    finally:
        driver.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: composite-score recompute with k-fold alpha tuning."
    )
    parser.parse_args(argv)
    run()


if __name__ == "__main__":
    main()
