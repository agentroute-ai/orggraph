"""Load the MVP organizational knowledge graph into Neo4j.

Builds a demo-ready KG from the existing RQ1 outputs:

    Node types        Count (approx)     Source
    ---------------   ----------------   ----------------------------------
    Person            139                datasets/enron/processed/extracted_hierarchy.csv
                                         + employees_ground_truth.csv for titles / levels
    Team              8                  Louvain on communication_graph.gpickle
    ExternalEntity    425                All clients/suppliers/regulators/law firms
                                         from clients_suppliers.json (--top-externals N
                                         to cap)

    Edge types                   Count      Properties
    --------------------------   --------   -----------------------------
    MEMBER_OF                    139        (Person -> Team)
    COMMUNICATES_WITH (intra)    ~1.9k      Person <-> Person, weight = email count
    COMMUNICATES_WITH (ext)      ~3.3k      Person <-> ExternalEntity, directed
    REPORTS_TO                   ~130       (Person -> Person; inferred by
                                             top-of-team or score delta >= 0.1
                                             within the same Team)

Requires Neo4j running locally (docker compose up -d).

Usage:
    python scripts/load_kg_mvp.py
    python scripts/load_kg_mvp.py --reset            # wipe graph first
    python scripts/load_kg_mvp.py --top-externals 50 # more external entities
    python scripts/load_kg_mvp.py --skip-external-edges  # skip email-corpus parse
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import pandas as pd
from neo4j import GraphDatabase

from orggraph.config import DATASETS_DIR, OUTPUT_DIR
from orggraph.data.identity import build_alias_map, resolve_sender
from orggraph.data.loader import load_emails
from orggraph.extraction.communities import detect_communities

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "orggraph2026")


def load_inputs(top_externals: int):
    """Load all input artifacts from disk."""
    hierarchy = pd.read_csv(OUTPUT_DIR / "extracted_hierarchy.csv")
    gt = pd.read_csv(OUTPUT_DIR / "employees_ground_truth.csv")

    with open(OUTPUT_DIR / "communication_graph.gpickle", "rb") as f:
        graph = pickle.load(f)

    with open(DATASETS_DIR / "clients_suppliers.json") as f:
        external_raw = json.load(f)["organizations"]
    externals = sorted(external_raw, key=lambda o: -o.get("total_emails", 0))[:top_externals]

    communities = detect_communities(graph)

    return hierarchy, gt, graph, externals, communities


def enrich_hierarchy(hierarchy: pd.DataFrame, gt: pd.DataFrame, communities: dict) -> pd.DataFrame:
    """Merge ground truth titles/levels + community IDs onto the hierarchy frame."""
    merged = hierarchy.merge(
        gt[["name", "email", "title", "level", "level_numeric"]],
        left_on="node",
        right_on="name",
        how="left",
    )
    merged["community"] = merged["node"].map(communities).fillna(-1).astype(int)
    merged["gt_level_numeric"] = merged["level_numeric"].astype("Int64")
    merged["gt_level"] = merged["level"].fillna("Unknown")
    merged["title"] = merged["title"].fillna("Unknown")
    merged["email"] = merged["email"].fillna("")
    return merged


def infer_reports_to(persons: pd.DataFrame) -> list[dict]:
    """Infer REPORTS_TO edges within each team.

    Simple rule: person A reports to person B iff they are in the same Team,
    B has the highest composite_score in that team, and B != A. This produces
    a flat, one-level hierarchy per team — enough to light up the graph view.
    """
    edges = []
    for team_id, group in persons.groupby("community"):
        if team_id < 0 or len(group) < 2:
            continue
        boss = group.sort_values("composite_score", ascending=False).iloc[0]
        for _, row in group.iterrows():
            if row["node"] == boss["node"]:
                continue
            edges.append({"subordinate": row["node"], "superior": boss["node"]})
    return edges


def cypher_reset(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def cypher_constraints(tx):
    tx.run("CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT team_id IF NOT EXISTS FOR (t:Team) REQUIRE t.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT entity_domain IF NOT EXISTS FOR (e:ExternalEntity) REQUIRE e.domain IS UNIQUE")


def load_persons(tx, persons: pd.DataFrame):
    rows = []
    for _, r in persons.iterrows():
        rows.append({
            "name": r["node"],
            "email": r["email"],
            "title": r["title"],
            "gt_level": r["gt_level"],
            "gt_level_numeric": int(r["gt_level_numeric"]) if pd.notna(r["gt_level_numeric"]) else None,
            "tier": int(r["tier"]),
            "composite_score": float(r["composite_score"]),
            "pagerank": float(r["pagerank"]),
            "betweenness": float(r["betweenness"]),
            "in_degree": float(r["in_degree"]),
            "community": int(r["community"]),
        })
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (p:Person {name: row.name})
        SET p.email = row.email,
            p.title = row.title,
            p.gt_level = row.gt_level,
            p.gt_level_numeric = row.gt_level_numeric,
            p.tier = row.tier,
            p.composite_score = row.composite_score,
            p.pagerank = row.pagerank,
            p.betweenness = row.betweenness,
            p.in_degree = row.in_degree,
            p.community = row.community
        """,
        rows=rows,
    )


def load_teams(tx, persons: pd.DataFrame):
    # Name each team after its highest-composite-score member ("captain")
    team_rows = []
    for community_id, group in persons.groupby("community"):
        if community_id < 0:
            continue
        captain = group.sort_values("composite_score", ascending=False).iloc[0]
        team_rows.append({
            "id": int(community_id),
            "name": f"Team {int(community_id)} ({captain['node']})",
            "size": int(len(group)),
            "captain": captain["node"],
        })
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (t:Team {id: row.id})
        SET t.name = row.name, t.size = row.size, t.captain = row.captain
        """,
        rows=team_rows,
    )


def load_member_of(tx):
    tx.run(
        """
        MATCH (p:Person), (t:Team)
        WHERE p.community = t.id AND p.community >= 0
        MERGE (p)-[:MEMBER_OF]->(t)
        """
    )


def load_communicates(tx, graph, person_names: set[str]):
    rows = []
    for u, v, data in graph.edges(data=True):
        if u not in person_names or v not in person_names:
            continue
        rows.append({
            "source": u,
            "target": v,
            "weight": int(data.get("weight", 1)),
        })
    # Chunked write to avoid giant transactions
    chunk = 500
    for i in range(0, len(rows), chunk):
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (a:Person {name: row.source})
            MATCH (b:Person {name: row.target})
            MERGE (a)-[r:COMMUNICATES_WITH]->(b)
            SET r.weight = row.weight
            """,
            rows=rows[i : i + chunk],
        )
    return len(rows)


def load_reports_to(tx, edges: list[dict]):
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (sub:Person {name: row.subordinate})
        MATCH (sup:Person {name: row.superior})
        MERGE (sub)-[:REPORTS_TO]->(sup)
        """,
        rows=edges,
    )


def extract_person_external_edges(
    external_domains: set[str],
    person_names: set[str],
    alias_map: dict[str, str],
    email_limit: int | None = None,
) -> list[dict]:
    """Walk the raw email corpus and produce Person<->ExternalEntity edge tuples.

    For each email, the sender domain and every recipient domain is inspected.
    If one side is a known internal Person (resolved via alias_map) and the
    other side is one of our top-N ExternalEntity domains, a directed edge
    is emitted (out = Person sent, in = Person received).

    Returns:
        List of dicts with keys person, domain, weight, direction ('out' | 'in').
    """
    print(f"      Loading email corpus (limit={email_limit or 'all'})...")
    emails = load_emails(limit=email_limit)

    # Detect HF column names (same logic as run_rq1.py)
    sender_col = next((c for c in ["from", "From", "sender"] if c in emails.columns), None)
    recipients_col = next((c for c in ["to", "To", "recipients"] if c in emails.columns), None)
    if sender_col is None or recipients_col is None:
        raise RuntimeError(f"Could not find sender/recipient columns in {list(emails.columns)}")

    out_counts: dict[tuple[str, str], int] = {}  # (person, domain) -> emails person->ext
    in_counts: dict[tuple[str, str], int] = {}   # (person, domain) -> emails ext->person

    def _addrs(val) -> list[str]:
        if val is None:
            return []
        if isinstance(val, str):
            return [a.strip() for a in val.split(",") if a.strip()]
        try:
            return [str(a).strip() for a in val if str(a).strip()]
        except TypeError:
            return []

    def _domain(addr: str) -> str | None:
        if "@" not in addr:
            return None
        return addr.split("@", 1)[1].lower().strip("<>\" ")

    print(f"      Scanning {len(emails):,} emails for external edges...")
    for sender_raw, recips_raw in zip(emails[sender_col], emails[recipients_col]):
        sender_str = str(sender_raw).lower().strip() if sender_raw is not None else ""
        sender_person = resolve_sender(sender_str, alias_map) if sender_str else None
        sender_domain = _domain(sender_str)

        recipient_addrs = _addrs(recips_raw)

        # Sender internal, any recipient external -> Person->Ext
        if sender_person in person_names:
            for r in recipient_addrs:
                dom = _domain(r.lower())
                if dom and dom in external_domains:
                    out_counts[(sender_person, dom)] = out_counts.get((sender_person, dom), 0) + 1

        # Sender external, any recipient internal -> Ext->Person
        if sender_domain and sender_domain in external_domains:
            for r in recipient_addrs:
                recip_person = resolve_sender(r.lower(), alias_map)
                if recip_person in person_names:
                    key = (recip_person, sender_domain)
                    in_counts[key] = in_counts.get(key, 0) + 1

    edges = []
    for (person, domain), weight in out_counts.items():
        edges.append({"person": person, "domain": domain, "weight": weight, "direction": "out"})
    for (person, domain), weight in in_counts.items():
        edges.append({"person": person, "domain": domain, "weight": weight, "direction": "in"})

    print(f"      Person->External: {len(out_counts)} edges, External->Person: {len(in_counts)} edges")
    return edges


def load_external_edges(tx, edges: list[dict]):
    """Write Person<->ExternalEntity edges. Direction encoded via two relationship patterns."""
    out_edges = [e for e in edges if e["direction"] == "out"]
    in_edges = [e for e in edges if e["direction"] == "in"]

    if out_edges:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (p:Person {name: row.person})
            MATCH (e:ExternalEntity {domain: row.domain})
            MERGE (p)-[r:COMMUNICATES_WITH]->(e)
            SET r.weight = row.weight
            """,
            rows=out_edges,
        )
    if in_edges:
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (p:Person {name: row.person})
            MATCH (e:ExternalEntity {domain: row.domain})
            MERGE (e)-[r:COMMUNICATES_WITH]->(p)
            SET r.weight = row.weight
            """,
            rows=in_edges,
        )


def load_externals(tx, externals: list[dict]):
    rows = []
    for o in externals:
        rows.append({
            "domain": o["domain"],
            "name": o.get("company_name", o["domain"]),
            "category": o.get("category", "Unknown"),
            "relationship_type": o.get("relationship_type", "Unknown"),
            "total_emails": int(o.get("total_emails", 0)),
            "emails_from_enron": int(o.get("emails_from_enron", 0)),
            "emails_to_enron": int(o.get("emails_to_enron", 0)),
        })
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (e:ExternalEntity {domain: row.domain})
        SET e.name = row.name,
            e.category = row.category,
            e.relationship_type = row.relationship_type,
            e.total_emails = row.total_emails,
            e.emails_from_enron = row.emails_from_enron,
            e.emails_to_enron = row.emails_to_enron
        """,
        rows=rows,
    )


def summarize(session):
    q = session.run(
        """
        CALL () { MATCH (p:Person) RETURN count(p) AS persons }
        CALL () { MATCH (t:Team) RETURN count(t) AS teams }
        CALL () { MATCH (e:ExternalEntity) RETURN count(e) AS externals }
        CALL () {
          MATCH (:Person)-[r:COMMUNICATES_WITH]->(:Person)
          RETURN count(r) AS comm_internal
        }
        CALL () {
          MATCH (:Person)-[r:COMMUNICATES_WITH]->(:ExternalEntity)
          RETURN count(r) AS comm_outbound
        }
        CALL () {
          MATCH (:ExternalEntity)-[r:COMMUNICATES_WITH]->(:Person)
          RETURN count(r) AS comm_inbound
        }
        CALL () { MATCH ()-[r:MEMBER_OF]->() RETURN count(r) AS member_of }
        CALL () { MATCH ()-[r:REPORTS_TO]->() RETURN count(r) AS reports_to }
        RETURN persons, teams, externals,
               comm_internal, comm_outbound, comm_inbound,
               member_of, reports_to
        """
    ).single()
    return dict(q) if q else {}


def run(reset: bool, top_externals: int, skip_external_edges: bool, email_limit: int | None):
    print("[1/6] Loading inputs...")
    hierarchy, gt, graph, externals, communities = load_inputs(top_externals)
    persons = enrich_hierarchy(hierarchy, gt, communities)
    reports_edges = infer_reports_to(persons)
    print(f"      persons={len(persons)}  teams={persons['community'].nunique()}  "
          f"externals={len(externals)}  comm_edges={graph.number_of_edges()}  "
          f"reports_to={len(reports_edges)}")

    external_edges: list[dict] = []
    if not skip_external_edges:
        print("[2/6] Extracting Person<->ExternalEntity edges from raw email corpus...")
        alias_map = build_alias_map()
        person_names = set(persons["node"])
        external_domains = {o["domain"].lower() for o in externals}
        external_edges = extract_person_external_edges(
            external_domains, person_names, alias_map, email_limit=email_limit,
        )
    else:
        print("[2/6] Skipping external edges (--skip-external-edges).")

    print(f"[3/6] Connecting to Neo4j at {NEO4J_URI} as {NEO4J_USER}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            if reset:
                print("[4/6] Resetting graph (MATCH (n) DETACH DELETE n)...")
                session.execute_write(cypher_reset)

            print("[4/6] Creating constraints...")
            session.execute_write(cypher_constraints)

            print("[5/6] Loading nodes + edges...")
            session.execute_write(load_persons, persons)
            session.execute_write(load_teams, persons)
            session.execute_write(load_member_of)

            person_names = set(persons["node"])
            n_comm = session.execute_write(load_communicates, graph, person_names)
            print(f"      COMMUNICATES_WITH (internal) loaded: {n_comm}")

            session.execute_write(load_reports_to, reports_edges)
            session.execute_write(load_externals, externals)

            if external_edges:
                session.execute_write(load_external_edges, external_edges)
                print(f"      COMMUNICATES_WITH (external) loaded: {len(external_edges)}")

            print("[6/6] Summary:")
            stats = summarize(session)
            for k, v in stats.items():
                print(f"      {k:16} {v}")
    finally:
        driver.close()

    print("\nOpen http://localhost:7474 to explore.")
    print("Sample Cypher:")
    print("""
    MATCH (lay:Person {name: "Kenneth Lay"})
    OPTIONAL MATCH (lay)-[c:COMMUNICATES_WITH]-(other:Person)
    WITH lay, other, c ORDER BY c.weight DESC LIMIT 10
    OPTIONAL MATCH (lay)-[:MEMBER_OF]->(lt:Team)
    OPTIONAL MATCH (other)-[:MEMBER_OF]->(ot:Team)
    RETURN lay, other, c, lt, ot
    """)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 6a: load MVP KG into Neo4j")
    parser.add_argument("--reset", action="store_true", help="Wipe graph before loading")
    parser.add_argument("--top-externals", type=int, default=425,
                        help="Top-N external entities by email volume (default: all 425)")
    parser.add_argument("--skip-external-edges", action="store_true",
                        help="Don't parse raw corpus for Person<->ExternalEntity edges")
    parser.add_argument("--email-limit", type=int, default=None,
                        help="Limit emails scanned for external edges (dev only)")
    args = parser.parse_args(argv)
    run(reset=args.reset, top_externals=args.top_externals,
        skip_external_edges=args.skip_external_edges, email_limit=args.email_limit)


if __name__ == "__main__":
    main()
