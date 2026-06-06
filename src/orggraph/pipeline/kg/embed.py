"""Embed all Person and ExternalEntity nodes into pgvector.

For each node we build a short textual description ("document"), embed it via
the configured OpenAI-compatible endpoint (Ollama on Mac dev, vLLM on DGX),
and upsert into the pgvector tables created by docker/postgres/init.sql.

Usage:
    source .env && python scripts/embed_kg.py
    python scripts/embed_kg.py --limit 10          # dev preview
    python scripts/embed_kg.py --model embeddinggemma
"""

from __future__ import annotations

import argparse
import os
import time

import psycopg
from neo4j import GraphDatabase
from openai import OpenAI
from pgvector.psycopg import register_vector
from tqdm import tqdm

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "orggraph2026")

PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://{os.environ.get('POSTGRES_USER','orggraph')}:"
    f"{os.environ.get('POSTGRES_PASSWORD','orggraph2026')}@localhost:"
    f"{os.environ.get('POSTGRES_PORT','5432')}/"
    f"{os.environ.get('POSTGRES_DB','orggraph')}",
)

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embeddinggemma")


# ---------------------------------------------------------------------------
# Document building — these short descriptions are what gets embedded
# ---------------------------------------------------------------------------

def person_document(p: dict) -> str:
    """Construct a richer text passage when LLM enrichment is present."""
    name = p["name"]
    title = p.get("title") or "Unknown role"
    role = p.get("role_summary") or ""
    function = p.get("function") or "Unknown"
    seniority = p.get("seniority")
    persona = []
    for dim in ("formality", "verbosity", "directiveness", "agenda_setting"):
        v = p.get(f"persona_{dim}")
        if v is not None:
            persona.append(f"{dim}={v}")
    auth = p.get("authority_style") or ""
    style = p.get("communication_style") or ""
    topics = p.get("topics") or []
    expertise = p.get("expertise") or []
    contacts = p.get("top_contacts") or []

    parts = [f"{name} — {title}."]
    if role:
        parts.append(f"Role: {role}")
    parts.append(f"Function: {function}.")
    if seniority is not None:
        parts.append(f"Recovered seniority: {seniority}/10.")
    if auth:
        parts.append(f"Authority style: {auth}.")
    if persona:
        parts.append("Persona: " + ", ".join(persona) + ".")
    if style:
        parts.append(f"Style: {style}")
    if topics:
        parts.append("Topics: " + ", ".join(topics) + ".")
    if expertise:
        parts.append("Expertise: " + ", ".join(expertise) + ".")
    if contacts:
        parts.append("Frequent contacts: " + ", ".join(contacts[:5]) + ".")
    return " ".join(parts)


def entity_document(e: dict) -> str:
    name = e["name"]
    domain = e["domain"]
    category = e.get("category") or "Unknown"
    rel = e.get("relationship_type") or "Unknown"
    total = e.get("total_emails") or 0
    contacts = e.get("enron_contacts") or []

    parts = [
        f"{name} (domain {domain}) is an external organization in the category '{category}'.",
        f"Relationship type to Enron: {rel}.",
        f"Total observed email volume with Enron employees: {total}.",
    ]
    if contacts:
        parts.append(
            "Enron employees in direct contact include: " + ", ".join(contacts[:5]) + "."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Neo4j: pull nodes with the context needed for the document
# ---------------------------------------------------------------------------

def fetch_persons(driver) -> list[dict]:
    q = """
    MATCH (p:Person)
    OPTIONAL MATCH (p)-[c:COMMUNICATES_WITH]->(other:Person)
    WITH p, other, c ORDER BY c.weight DESC
    WITH p, collect(DISTINCT other.name)[..5] AS top_contacts
    OPTIONAL MATCH (p)-[:KNOWS_ABOUT]->(t:Topic)
    WITH p, top_contacts, collect(DISTINCT t.name)[..7] AS topics
    RETURN p.name AS name, p.title AS title, p.role_summary AS role_summary,
           p.function AS function, p.seniority AS seniority,
           p.persona_formality AS persona_formality,
           p.persona_verbosity AS persona_verbosity,
           p.persona_directiveness AS persona_directiveness,
           p.persona_agenda_setting AS persona_agenda_setting,
           p.authority_style AS authority_style,
           p.communication_style AS communication_style,
           p.tier AS tier, p.composite_score AS composite_score,
           p.composite_score_v3 AS composite_score_v3,
           p.community AS community,
           p.gt_level AS gt_level,
           top_contacts, topics
    ORDER BY name
    """
    with driver.session() as s:
        return [dict(r) for r in s.run(q)]


def fetch_entities(driver) -> list[dict]:
    q = """
    MATCH (e:ExternalEntity)
    OPTIONAL MATCH (e)-[c:COMMUNICATES_WITH]-(p:Person)
    WITH e, p, c ORDER BY c.weight DESC
    WITH e, collect(DISTINCT p.name)[..5] AS enron_contacts
    RETURN e.domain AS domain, e.name AS name, e.category AS category,
           e.relationship_type AS relationship_type,
           e.total_emails AS total_emails,
           e.emails_from_enron AS emails_from_enron,
           e.emails_to_enron AS emails_to_enron,
           enron_contacts
    ORDER BY e.total_emails DESC
    """
    with driver.session() as s:
        return [dict(r) for r in s.run(q)]


# ---------------------------------------------------------------------------
# Embedding + upsert
# ---------------------------------------------------------------------------

def embed_many(client: OpenAI, model: str, texts: list[str], batch: int = 16) -> list[list[float]]:
    """Embed texts one batch at a time. Ollama handles arbitrary batch sizes but
    smaller batches give faster feedback via the progress bar."""
    out: list[list[float]] = []
    for i in tqdm(range(0, len(texts), batch), desc=f"embed [{model}]", unit="batch"):
        chunk = texts[i : i + batch]
        resp = client.embeddings.create(model=model, input=chunk)
        # openai SDK returns data in input order
        out.extend([d.embedding for d in resp.data])
    return out


def upsert_persons(cur, persons: list[dict], embeddings: list[list[float]]):
    cur.executemany(
        """
        INSERT INTO embeddings_person (person_id, name, role, department, embedding, metadata)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (person_id) DO UPDATE
           SET name = EXCLUDED.name,
               role = EXCLUDED.role,
               department = EXCLUDED.department,
               embedding = EXCLUDED.embedding,
               metadata = EXCLUDED.metadata
        """,
        [
            (
                p["name"],
                p["name"],
                p.get("title") or "Unknown",
                p.get("team_name") or "Unknown",
                emb,
                psycopg.types.json.Jsonb({
                    "tier": p.get("tier"),
                    "composite_score": p.get("composite_score"),
                    "community": p.get("community"),
                    "gt_level": p.get("gt_level"),
                    "gt_level_numeric": p.get("gt_level_numeric"),
                    "pagerank": p.get("pagerank"),
                    "betweenness": p.get("betweenness"),
                }),
            )
            for p, emb in zip(persons, embeddings)
        ],
    )


def upsert_entities(cur, entities: list[dict], embeddings: list[list[float]]):
    cur.executemany(
        """
        INSERT INTO embeddings_entity (entity_id, name, category, embedding, metadata)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (entity_id) DO UPDATE
           SET name = EXCLUDED.name,
               category = EXCLUDED.category,
               embedding = EXCLUDED.embedding,
               metadata = EXCLUDED.metadata
        """,
        [
            (
                e["domain"],
                e["name"],
                e.get("category") or "Unknown",
                emb,
                psycopg.types.json.Jsonb({
                    "relationship_type": e.get("relationship_type"),
                    "total_emails": e.get("total_emails"),
                    "emails_from_enron": e.get("emails_from_enron"),
                    "emails_to_enron": e.get("emails_to_enron"),
                }),
            )
            for e, emb in zip(entities, embeddings)
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(limit: int | None, model: str, skip_persons: bool, skip_entities: bool, refresh: bool = False):
    t0 = time.monotonic()
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print("[1/4] Fetching nodes from Neo4j...")
    persons = fetch_persons(neo4j_driver)
    entities = fetch_entities(neo4j_driver)
    if limit:
        persons = persons[:limit]
        entities = entities[:limit]
    print(f"      persons={len(persons)} entities={len(entities)}")

    print("[2/4] Building documents...")
    person_docs = [person_document(p) for p in persons]
    entity_docs = [entity_document(e) for e in entities]
    print(f"      example person doc: {person_docs[0]}")
    print(f"      example entity doc: {entity_docs[0]}")

    print(f"[3/4] Embedding via {OPENAI_BASE_URL} model={model}...")
    person_embs: list[list[float]] = []
    entity_embs: list[list[float]] = []
    if not skip_persons:
        person_embs = embed_many(client, model, person_docs)
    if not skip_entities:
        entity_embs = embed_many(client, model, entity_docs)

    neo4j_driver.close()

    print(f"[4/4] Upserting into pgvector ({PG_DSN.split('@')[-1]})...")
    with psycopg.connect(PG_DSN) as pg:
        register_vector(pg)
        with pg.cursor() as cur:
            if refresh:
                cur.execute("TRUNCATE TABLE embeddings_person, embeddings_entity")
                pg.commit()
                print("[refresh] truncated embeddings_person + embeddings_entity")
            if person_embs:
                upsert_persons(cur, persons, person_embs)
            if entity_embs:
                upsert_entities(cur, entities, entity_embs)
            pg.commit()

            cur.execute("SELECT count(*) FROM embeddings_person")
            n_p = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM embeddings_entity")
            n_e = cur.fetchone()[0]
            cur.execute("""
                SELECT array_length(embedding::real[], 1)
                FROM embeddings_person LIMIT 1
            """)
            dim = cur.fetchone()[0] if n_p else None

    dt = time.monotonic() - t0
    print(f"\n  Done in {dt:.1f}s. pgvector rows: person={n_p}, entity={n_e}, dim={dim}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 7: embed Person + Entity descriptions to pgvector")
    parser.add_argument("--limit", type=int, default=None, help="Limit nodes per label (dev preview)")
    parser.add_argument("--model", default=EMBED_MODEL, help=f"Embed model (default env EMBED_MODEL={EMBED_MODEL})")
    parser.add_argument("--skip-persons", action="store_true")
    parser.add_argument("--skip-entities", action="store_true")
    parser.add_argument("--refresh", action="store_true",
                        help="Wipe pgvector tables before inserting (force full rebuild)")
    args = parser.parse_args(argv)
    run(limit=args.limit, model=args.model, skip_persons=args.skip_persons,
        skip_entities=args.skip_entities, refresh=args.refresh)


if __name__ == "__main__":
    main()
