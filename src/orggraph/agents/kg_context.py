"""Knowledge-graph context for persona-grounded agents.

Pulls a compact set of organisationally-relevant facts from Neo4j —
hierarchical position (REPORTS_TO + tier), top human correspondents
(COMMUNICATES_WITH), expertise topics (KNOWS_ABOUT), and team affiliation
(MEMBER_OF) — and renders them as a context block that's appended to
the persona system prompt.

Why: in early continuation experiments the multi-agent agent locked
onto the literal email subject line and missed the broader
conversational context, while the single-LLM baseline (which has
*everyone's* perspective in one prompt) inferred it from the longer
window. Giving each agent its KG context narrows that gap without
breaking the architectural principle that agents have *independent*
contexts — each agent only gets KG facts about *itself*, not about
the other participants.

Loader is forgiving: missing edges, missing properties, or missing
Neo4j connection all produce empty fields rather than raising. The
serialiser omits any section whose data is absent, so partial KG
coverage degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KGContext:
    """Per-Person facts pulled from the Neo4j knowledge graph."""

    name: str
    title: str = ""
    tier: int | None = None
    community: int | None = None
    function: str = ""

    reports_to: str = ""
    direct_reports: tuple[str, ...] = ()

    top_collaborators: tuple[tuple[str, float], ...] = ()
    teams: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()

    n_emails_sent: int | None = None


_CYPHER = """
MATCH (p:Person {name: $name})
OPTIONAL MATCH (p)-[:REPORTS_TO]->(boss:Person)
OPTIONAL MATCH (sub:Person)-[:REPORTS_TO]->(p)
OPTIONAL MATCH (p)-[r:COMMUNICATES_WITH]-(other:Person)
WHERE other <> p
WITH p, boss,
     collect(DISTINCT sub.name) AS direct_reports,
     collect(DISTINCT {
       name: other.name,
       weight: coalesce(r.weight, r.count, r.n_emails, 0.0)
     }) AS comms_raw
OPTIONAL MATCH (p)-[:KNOWS_ABOUT]->(t:Topic)
WITH p, boss, direct_reports, comms_raw, collect(DISTINCT t.name) AS topics_kg
OPTIONAL MATCH (p)-[:MEMBER_OF]->(team:Team)
RETURN p.name           AS name,
       p.title          AS title,
       p.tier           AS tier,
       p.community      AS community,
       p.function       AS function,
       p.n_emails_sent  AS n_emails_sent,
       boss.name        AS reports_to,
       direct_reports,
       comms_raw,
       topics_kg,
       collect(DISTINCT team.name) AS teams
"""


def load_kg_context(name: str, driver) -> KGContext:
    """Fetch one Person's KG context.

    ``driver`` is a ``neo4j.GraphDatabase`` driver. If the Person does
    not exist or fields are absent, the returned ``KGContext`` carries
    sensible empty values rather than raising — let the formatter omit
    sections gracefully.
    """
    if driver is None:
        return KGContext(name=name)

    with driver.session() as session:
        record = session.run(_CYPHER, name=name).single()
    if record is None:
        return KGContext(name=name)

    comms = sorted(
        (c for c in record.get("comms_raw", []) or [] if c.get("name")),
        key=lambda c: float(c.get("weight") or 0.0),
        reverse=True,
    )[:5]
    top_collabs = tuple((c["name"], float(c.get("weight") or 0.0)) for c in comms)

    return KGContext(
        name=str(record.get("name") or name),
        title=_safe_str(record.get("title")),
        tier=_safe_int(record.get("tier")),
        community=_safe_int(record.get("community")),
        function=_safe_str(record.get("function")),
        n_emails_sent=_safe_int(record.get("n_emails_sent")),
        reports_to=_safe_str(record.get("reports_to")),
        direct_reports=tuple(
            n for n in (record.get("direct_reports") or []) if n
        ),
        top_collaborators=top_collabs,
        teams=tuple(t for t in (record.get("teams") or []) if t),
        topics=tuple(t for t in (record.get("topics_kg") or []) if t),
    )


def format_kg_context(ctx: KGContext) -> str:
    """Render a KGContext as a system-prompt block.

    Sections with no data are omitted so partial KG coverage produces
    a partial — but still coherent — prompt rather than empty
    placeholders.
    """
    lines: list[str] = []

    # Hierarchy line: title + tier + community + reporting relationships
    pos_parts: list[str] = []
    if ctx.title:
        pos_parts.append(ctx.title)
    if ctx.tier is not None:
        pos_parts.append(f"hierarchy tier {ctx.tier}/6")
    if ctx.function:
        pos_parts.append(f"function: {ctx.function}")
    if pos_parts:
        lines.append("Position: " + "; ".join(pos_parts) + ".")

    if ctx.reports_to:
        lines.append(f"You report to: {ctx.reports_to}.")
    if ctx.direct_reports:
        # Cap at 6 to keep prompts compact; longer lists rarely add signal
        names = ", ".join(ctx.direct_reports[:6])
        more = "" if len(ctx.direct_reports) <= 6 else f" (and {len(ctx.direct_reports) - 6} others)"
        lines.append(f"Your direct reports include: {names}{more}.")

    if ctx.top_collaborators:
        # Format as "name (n emails)" if weights look like counts
        parts = []
        for name, weight in ctx.top_collaborators:
            if weight >= 1.0:
                parts.append(f"{name} ({int(weight)} interactions)")
            else:
                parts.append(name)
        lines.append("Your most frequent correspondents are: " + "; ".join(parts) + ".")

    if ctx.teams:
        lines.append(f"Team affiliation: {', '.join(ctx.teams[:3])}.")

    if ctx.topics:
        lines.append(f"Recurring topics in your communication: {', '.join(ctx.topics[:5])}.")

    if ctx.n_emails_sent is not None and ctx.n_emails_sent > 0:
        lines.append(f"You have sent {ctx.n_emails_sent:,} emails in this corpus window.")

    if not lines:
        return ""
    return "\n\n## Organisational position (from the org-graph)\n\n" + "\n".join(f"- {line}" for line in lines)


# --- private helpers ---------------------------------------------------


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return s if s.lower() != "nan" else ""


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
