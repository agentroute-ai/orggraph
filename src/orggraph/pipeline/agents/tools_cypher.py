"""Parametrised Cypher templates for the KG Chat tool catalog.

Templates are exposed as module-level constants so a reviewer can
audit query logic in one file without reading the tool API. Each
template uses parameter-placeholder syntax ($name, $before_date, $n,
etc.) — no string interpolation for query values, since that would
risk Cypher injection. Activity fields (counts, recent threads, top
topics) are *always* derived from filtered Email edges, never from
corpus-wide pre-computed Person properties.

Per the design: identity fields (name, title, tier, team, role
summary) are returned as-is from the Person node. Activity fields
are recomputed at query time using $before_date.

Known schema caveats:

- Topic node ambiguity: the graph contains two Topic populations,
  cluster Topics (keyed by topic_id, with [:ABOUT_TOPIC] edges from
  Email) and canonical Topics (keyed by {name, kind: 'canonical'},
  reachable only from Person/ExternalEntity). Queries that match
  Topic by name may pick up the canonical Topic when names overlap;
  the cluster Topic is what carries the Email volume signal. A
  follow-up commit will disambiguate.

- Email nodes carry no body. Stage 3 writes only
  subject/date/sender_resolved/decision_carrying to Email nodes; the
  raw body lives in Postgres but is not loaded into Neo4j. Both
  thread-history and recent-activity queries return subject + date
  only. If body content is needed in a future iteration, propagate
  ``body_truncated`` from the source parquet through Stage 3 first.

These caveats do not affect the structural unit tests (which run
against mocked Neo4j) but will surface against the real graph. The
in-plan integration tests at tests/integration/test_tools_neo4j.py
are out of scope here and are tracked as a follow-up.
"""

from __future__ import annotations

# 1. get_person — identity card + recent-window topic activity
GET_PERSON = """
MATCH (p:Person {name: $name})
OPTIONAL MATCH (p)-[:REPORTS_TO]->(boss:Person)
OPTIONAL MATCH (sub:Person)-[:REPORTS_TO]->(p)
OPTIONAL MATCH (p)-[:MEMBER_OF]->(team:Team)
OPTIONAL MATCH (p)<-[:SENT_BY]-(e:Email)
WHERE ($before_date IS NULL OR e.date < $before_date)
WITH p, boss,
     collect(DISTINCT sub.name) AS direct_reports,
     collect(DISTINCT team.name) AS teams,
     count(DISTINCT e) AS n_emails_sent
OPTIONAL MATCH (p)<-[:SENT_BY]-(e2:Email)-[:ABOUT_TOPIC]->(t:Topic)
WHERE ($before_date IS NULL OR e2.date < $before_date)
WITH p, boss, direct_reports, teams, n_emails_sent,
     t.name AS topic_name, count(t) AS topic_count
ORDER BY topic_count DESC
WITH p, boss, direct_reports, teams, n_emails_sent,
     collect({name: topic_name, n: topic_count})[..5] AS top_topics
RETURN p.name           AS name,
       p.title          AS title,
       p.tier           AS tier,
       p.role_summary   AS role_summary,
       boss.name        AS reports_to,
       direct_reports,
       teams,
       n_emails_sent,
       top_topics
"""

# 2. get_thread_history — last n emails sender → recipient
GET_THREAD_HISTORY = """
MATCH (sender:Person {name: $sender})<-[:SENT_BY]-(e:Email)-[:SENT_TO]->(recipient:Person {name: $recipient})
WHERE ($before_date IS NULL OR e.date < $before_date)
RETURN e.email_id AS email_id,
       toString(e.date) AS date,
       e.subject AS subject
ORDER BY e.date DESC
LIMIT $n
"""

# 3. find_topic_collaborators — top n people on a topic
# Uses case-insensitive substring match on Topic.name because the canonical
# topic names are LLM-generated specific phrases ("ISDA Agreement
# Administration", "Online Trade Confirmation") and the calling agent
# usually queries with shorter keywords ("ISDA", "confirmations"). Exact
# match was returning 0 results in ~85% of calls during the RQ2 C1' pilot.
FIND_TOPIC_COLLABORATORS = """
MATCH (t:Topic)<-[:ABOUT_TOPIC]-(e:Email)-[:SENT_BY]->(p:Person)
WHERE toLower(t.name) CONTAINS toLower($topic)
  AND ($before_date IS NULL OR e.date < $before_date)
WITH p, count(DISTINCT e) AS n_emails_on_topic
ORDER BY n_emails_on_topic DESC
LIMIT $n
RETURN p.name         AS name,
       p.title        AS title,
       p.role_summary AS role_summary,
       n_emails_on_topic
"""

# --- Network analytics (Tier 3) -------------------------------------------
#
# These templates expose the per-Person centrality / score columns
# computed by Stage 2c (clusters.discover) and Stage 5 (score.recompute).
# Available metric columns: composite_score, composite_score_v3,
# pagerank, betweenness, in_degree, tier, tier_v3, gt_level_numeric.
#
# get_centrality interpolates the metric column name into the Cypher.
# That sounds dangerous (Cypher injection) — and it would be, if the
# metric came straight from the model. The wrapper validates against an
# allowlist before substitution, so the SQL-injection equivalent here
# is contained at the Python boundary.

# 9. get_centrality — top N people by a chosen metric, optionally tier-filtered
GET_CENTRALITY_TEMPLATE = """
MATCH (p:Person)
WHERE p.{metric} IS NOT NULL
  AND ($tier IS NULL OR p.tier = $tier)
RETURN p.name              AS name,
       p.title             AS title,
       p.tier              AS tier,
       p.{metric}          AS score
ORDER BY p.{metric} DESC
LIMIT $top_n
"""

# 10. get_dominance_score — every score column for one person + their rank
# on composite_score_v3 (the canonical RQ1 metric). Rank uses a window
# pattern (count of others with strictly higher score, +1).
GET_DOMINANCE_SCORE = """
MATCH (p:Person {name: $name})
OPTIONAL MATCH (other:Person)
WHERE other.composite_score_v3 IS NOT NULL
  AND other.composite_score_v3 > coalesce(p.composite_score_v3, -1.0)
WITH p, count(other) AS n_above
RETURN p.name                  AS name,
       p.title                 AS title,
       p.tier                  AS tier,
       p.tier_v3               AS tier_v3,
       p.composite_score       AS composite_score,
       p.composite_score_v3    AS composite_score_v3,
       p.pagerank              AS pagerank,
       p.betweenness           AS betweenness,
       p.in_degree             AS in_degree,
       p.community             AS community,
       p.gt_level              AS gt_level,
       p.gt_level_numeric      AS gt_level_numeric,
       n_above + 1             AS rank_v3
"""

# 8. get_org_chart — multi-hop REPORTS_TO traversal (bosses up + reports down)
# Variable-length paths capped at $depth on both sides. Self is returned
# separately so the model can anchor.
GET_ORG_CHART = """
MATCH (p:Person {name: $name})
OPTIONAL MATCH bosses_path = (p)-[:REPORTS_TO*1..]->(b:Person)
WHERE length(bosses_path) <= $depth
WITH p,
     collect(DISTINCT {name: b.name, title: b.title, tier: b.tier,
                       depth: length(bosses_path)}) AS bosses_raw
OPTIONAL MATCH reports_path = (s:Person)-[:REPORTS_TO*1..]->(p)
WHERE length(reports_path) <= $depth
WITH p, bosses_raw,
     collect(DISTINCT {name: s.name, title: s.title, tier: s.tier,
                       depth: length(reports_path)}) AS reports_raw
RETURN p.name  AS name,
       p.title AS title,
       p.tier  AS tier,
       [b IN bosses_raw  WHERE b.name IS NOT NULL] AS bosses,
       [r IN reports_raw WHERE r.name IS NOT NULL] AS reports
"""
