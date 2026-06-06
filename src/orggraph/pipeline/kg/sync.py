"""Stage B.1 — Sync the CSV/JSON artifacts produced by Stage A into Neo4j.

This script is the only place LLM-derived data lands in the KG.
All operations use MERGE; rerunning is idempotent and converges to
the same KG.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

from orggraph.config import OUTPUT_DIR

PERSON_CSV = OUTPUT_DIR / "person_enrichment.csv"


def _safe_json_list(blob) -> list[str]:
    if not isinstance(blob, str) or not blob.strip():
        return []
    try:
        v = json.loads(blob)
        return [str(x) for x in v if isinstance(x, (str, int, float))]
    except Exception:
        return []


def load_canonical_map(path: Path) -> dict[str, str]:
    with open(path) as f:
        d = json.load(f)
    return {k.strip().lower(): v for k, v in d.get("raw_to_canonical", {}).items()}


def _to_canonical(raw: str, canon_map: dict[str, str]) -> str:
    return canon_map.get(raw.strip().lower(), raw.title())


def sync_persons(session, csv_path: Path, canon_map: dict[str, str]) -> int:
    """Set Person properties + create Topic nodes + KNOWS_ABOUT edges.

    Reads person_enrichment_v2.csv (18-column schema from Stage 4a).
    Column mapping to Neo4j Person properties:
      formality              -> persona_formality  (int 1-5)
      directiveness_signal   -> persona_directiveness (float signal)
      agenda_setting_signal  -> persona_agenda_setting (float signal)
      verbosity_signal       -> persona_verbosity (float signal)
      expertise              -> expertise_topics_v2 (JSON list, LLM-refined)
      expertise_topics       -> expertise_topics_det (JSON list, deterministic)
      authority_style        -> authority_style
      communication_style    -> communication_style
      role_summary           -> role_summary
      seniority_narrative    -> seniority_narrative
      confidence_self_report -> confidence_self_report
    """
    if not csv_path.exists():
        return 0
    df = pd.read_csv(csv_path)
    # v2 requires at least the name column; filter rows where name is present
    df = df[df["name"].notna()]
    if df.empty:
        return 0

    rows = []
    for _, r in df.iterrows():
        # Expertise topics: prefer LLM-refined "expertise" list; fall back to
        # deterministic "expertise_topics" list.
        expertise_llm = _safe_json_list(r.get("expertise"))
        expertise_det = _safe_json_list(r.get("expertise_topics"))
        topics_raw = expertise_llm if expertise_llm else expertise_det
        topics_canonical = sorted({_to_canonical(t, canon_map) for t in topics_raw})
        rows.append({
            "name": r["name"],
            "role_summary": r.get("role_summary") or "",
            "persona_formality": _to_int(r.get("formality")),
            "persona_directiveness": float(r["directiveness_signal"]) if pd.notna(r.get("directiveness_signal")) else None,
            "persona_agenda_setting": float(r["agenda_setting_signal"]) if pd.notna(r.get("agenda_setting_signal")) else None,
            "persona_verbosity": float(r["verbosity_signal"]) if pd.notna(r.get("verbosity_signal")) else None,
            "authority_style": r.get("authority_style") or "technical",
            "communication_style": r.get("communication_style") or "",
            "seniority_narrative": r.get("seniority_narrative") or "",
            "confidence_self_report": _to_int(r.get("confidence_self_report")),
            "expertise_topics_v2": expertise_llm,
            "expertise_topics_det": expertise_det,
            "topics_canonical": topics_canonical,
        })

    # 1. Person properties (v2 schema — no Function node, no seniority int)
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:Person {name: row.name})
        SET p.role_summary = row.role_summary,
            p.persona_formality = row.persona_formality,
            p.persona_verbosity = row.persona_verbosity,
            p.persona_directiveness = row.persona_directiveness,
            p.persona_agenda_setting = row.persona_agenda_setting,
            p.authority_style = row.authority_style,
            p.communication_style = row.communication_style,
            p.seniority_narrative = row.seniority_narrative,
            p.confidence_self_report = row.confidence_self_report,
            p.expertise_topics_v2 = row.expertise_topics_v2,
            p.expertise_topics_det = row.expertise_topics_det,
            p.llm_enriched_at = datetime()
        """,
        rows=rows,
    )

    # 2. Topic nodes + KNOWS_ABOUT edges
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:Person {name: row.name})
        UNWIND row.topics_canonical AS topic_name
        MERGE (t:Topic {name: topic_name, kind: 'canonical'})
        MERGE (p)-[:KNOWS_ABOUT]->(t)
        """,
        rows=rows,
    )
    return len(rows)


def sync_entities(session, csv_path: Path, canon_map: dict[str, str]) -> int:
    if not csv_path.exists():
        return 0
    df = pd.read_csv(csv_path)
    df = df[df["description"].notna()]
    if df.empty:
        return 0

    rows = []
    for _, r in df.iterrows():
        topics_raw = _safe_json_list(r.get("topics_json"))
        rows.append({
            "domain": r["domain"],
            "description": r.get("description") or "",
            "business_relationship": r.get("business_relationship") or "Other",
            "industry": r.get("industry") or "",
            "engagement_pattern": r.get("engagement_pattern") or "transactional",
            "tone": r.get("tone") or "transactional",
            "topics_canonical": sorted({_to_canonical(t, canon_map) for t in topics_raw}),
        })

    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:ExternalEntity {domain: row.domain})
        SET e.description = row.description,
            e.business_relationship = row.business_relationship,
            e.industry = row.industry,
            e.engagement_pattern = row.engagement_pattern,
            e.tone = row.tone,
            e.llm_enriched_at = datetime()
        """,
        rows=rows,
    )
    session.run(
        """
        UNWIND $rows AS row
        MATCH (e:ExternalEntity {domain: row.domain})
        UNWIND row.topics_canonical AS topic_name
        MERGE (t:Topic {name: topic_name, kind: 'canonical'})
        MERGE (e)-[:DISCUSSES]->(t)
        """,
        rows=rows,
    )
    return len(rows)


def sync_pairs(session, csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    df = pd.read_csv(csv_path)
    df = df[df["deference_score"].notna()]
    if df.empty:
        return 0

    out_rows = []  # A defers to B
    in_rows = []   # B defers to A
    rel_rows = []
    for _, r in df.iterrows():
        s = float(r["deference_score"])
        rel = r.get("relationship_type") or "transactional"
        if s < 0:
            out_rows.append({"a": r["person_a"], "b": r["person_b"], "score": abs(s), "raw": s})
        elif s > 0:
            in_rows.append({"a": r["person_a"], "b": r["person_b"], "score": s, "raw": s})
        rel_rows.append({"a": r["person_a"], "b": r["person_b"], "type": rel})

    if out_rows:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (a:Person {name: row.a})
            MATCH (b:Person {name: row.b})
            MERGE (a)-[r:DEFERS_TO]->(b)
            SET r.score = row.score, r.raw_score = row.raw
            """,
            rows=out_rows,
        )
    if in_rows:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (a:Person {name: row.a})
            MATCH (b:Person {name: row.b})
            MERGE (b)-[r:DEFERS_TO]->(a)
            SET r.score = row.score, r.raw_score = row.raw
            """,
            rows=in_rows,
        )
    session.run(
        """
        UNWIND $rows AS row
        MATCH (a:Person {name: row.a})
        MATCH (b:Person {name: row.b})
        MERGE (a)-[r:RELATES_AS]->(b)
        SET r.type = row.type
        """,
        rows=rel_rows,
    )
    return len(df)


def maybe_load_pair_enrichment() -> "pd.DataFrame | None":
    """Return pair_enrichment.csv as a DataFrame, or None if the file is absent.

    Stage 4c (pair enrichment) is optional and may be deferred. Callers must
    check for None before using the result.
    """
    pair_csv = OUTPUT_DIR / "pair_enrichment.csv"
    if not pair_csv.exists():
        print("[sync] pair_enrichment.csv not found — Stage 4c deferred, skipping pairs")
        return None
    return pd.read_csv(pair_csv)


def _to_int(x) -> int | None:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def run() -> None:
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_pw = os.environ.get("NEO4J_PASSWORD", "orggraph2026")

    persons_csv = PERSON_CSV
    entities_csv = OUTPUT_DIR / "entity_enrichment.csv"
    canon_json = OUTPUT_DIR / "topic_canonicalization.json"

    canon_map = load_canonical_map(canon_json) if canon_json.exists() else {}

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
    try:
        with driver.session() as session:
            n_p = sync_persons(session, persons_csv, canon_map)
            print(f"[sync] persons enriched: {n_p}")
            n_e = sync_entities(session, entities_csv, canon_map)
            print(f"[sync] entities enriched: {n_e}")
            pairs_df = maybe_load_pair_enrichment()
            if pairs_df is not None:
                pairs_csv = OUTPUT_DIR / "pair_enrichment.csv"
                n_pr = sync_pairs(session, pairs_csv)
                print(f"[sync] pairs enriched: {n_pr}")
            else:
                n_pr = 0
                print(f"[sync] pairs enriched: {n_pr} (skipped)")

            # Quick KG audit (Section §11.F in the spec)
            stats = session.run(
                """
                CALL () { MATCH (p:Person) WHERE p.seniority IS NOT NULL RETURN count(p) AS persons_enriched }
                CALL () { MATCH (e:ExternalEntity) WHERE e.description IS NOT NULL RETURN count(e) AS entities_enriched }
                CALL () { MATCH (f:Function) RETURN count(f) AS functions }
                CALL () { MATCH (t:Topic) RETURN count(t) AS topics }
                CALL () { MATCH ()-[r:KNOWS_ABOUT]->() RETURN count(r) AS knows_about }
                CALL () { MATCH ()-[r:DISCUSSES]->() RETURN count(r) AS discusses }
                CALL () { MATCH ()-[r:DEFERS_TO]->() RETURN count(r) AS defers_to }
                CALL () { MATCH ()-[r:RELATES_AS]->() RETURN count(r) AS relates_as }
                RETURN persons_enriched, entities_enriched, functions, topics,
                       knows_about, discusses, defers_to, relates_as
                """
            ).single()
            print("[audit]", dict(stats))
    finally:
        driver.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 6b: sync enrichment CSVs into Neo4j")
    parser.parse_args(argv)
    run()


if __name__ == "__main__":
    main()
