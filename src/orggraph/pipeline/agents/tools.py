"""Public surface for the OrgGraph KG-as-tool stage.

Defines the KGTool dataclass — one tool with a name, JSON-schema args,
description, and Python callable — and a ToolRegistry that lets the
simulation runner register tools and convert them to the OpenAI
chat-completions tools format for function calling.

Tool implementations themselves live in the same package
(get_person.py, thread_history.py, etc.). build_default_registry()
in this module is the canonical factory: given a Neo4j driver, it
wires up the KG Chat tool catalog and returns a populated ToolRegistry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from orggraph.pipeline.agents import tools_cypher


@dataclass(frozen=True)
class KGTool:
    """One tool definition exposed to simulation agents.

    Attributes
    ----------
    name:
        Stable identifier the agent (and the OpenAI function-calling
        protocol) uses to invoke the tool. Must be unique within a
        registry.
    description:
        One-sentence description shown to the model. Kept short — the
        full schema documents arguments separately.
    args_schema:
        JSON-schema-style description of the function's parameters.
        Used directly in the OpenAI tools list.
    callable:
        Python callable invoked with kwargs unpacked from the model's
        tool_call arguments. Returns a JSON-serialisable dict / list /
        scalar.
    """

    name: str
    description: str
    args_schema: dict[str, Any]
    callable: Callable[..., Any]


class ToolRegistry:
    """Insertion-ordered registry of KGTools.

    Provides:
    - register / get / __contains__ / __iter__ for in-Python use
    - to_openai_tools() to feed the OpenAI chat-completions tools list
    - call(name, args) to dispatch a tool_call
    """

    def __init__(self) -> None:
        self._tools: dict[str, KGTool] = {}

    def register(self, tool: KGTool) -> None:
        if tool.name in self._tools:
            raise ValueError(
                f"Tool {tool.name!r} already registered on this ToolRegistry"
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> KGTool:
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Export as OpenAI chat-completions `tools` list."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema,
                },
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch a tool_call. Validates args against ``args_schema`` before
        invoking the callable. Returns a structured error dict for invalid
        inputs or unexpected exceptions instead of propagating them out of
        the registry — the chat loop must never crash on a single tool call.
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool {name!r}; registered: {sorted(self._tools)}")
        tool = self._tools[name]

        import jsonschema  # local import keeps the dep optional at import time

        try:
            jsonschema.validate(instance=args, schema=tool.args_schema)
        except jsonschema.ValidationError as exc:
            return {"error": "InvalidArgs", "tool": name, "details": exc.message}

        try:
            return tool.callable(**args)
        except Exception as exc:  # noqa: BLE001 — uniform error contract
            return {"error": "ToolError", "tool": name, "args": args, "details": str(exc)}


# --- Tool factories ----------------------------------------------------
#
# Each factory takes a Neo4j driver and returns a configured KGTool.
# Wiring is deferred to build_default_registry() at runtime.


def _serialise_top_topics(raw):
    """Neo4j returns a list of {name, n} dicts; pass through as-is."""
    if not raw:
        return []
    return list(raw)


def _safe(v) -> str:
    """Coerce None / NaN / empty strings into empty string for JSON safety."""
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def make_get_person_tool(driver) -> KGTool:
    """Identity card + recent-window topic activity for a single Person."""

    def _call(*, name: str, before_date: str | None = None) -> dict:
        with driver.session() as session:
            record = session.run(
                tools_cypher.GET_PERSON,
                name=name,
                before_date=before_date,
            ).single()
        if record is None:
            return {"name": name, "found": False}
        return {
            "name": str(record.get("name") or name),
            "title": _safe(record.get("title")),
            "tier": record.get("tier"),
            "role_summary": _safe(record.get("role_summary")),
            "reports_to": _safe(record.get("reports_to")),
            "direct_reports": list(record.get("direct_reports") or []),
            "teams": list(record.get("teams") or []),
            "n_emails_sent": int(record.get("n_emails_sent") or 0),
            "top_topics": _serialise_top_topics(record.get("top_topics")),
        }

    return KGTool(
        name="get_person",
        description=(
            "Get a person's identity card: title, hierarchy tier, team, who they "
            "report to, who reports to them, role summary, and their top topics by "
            "email volume."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Canonical full name as it appears in employees.json.",
                },
                "before_date": {
                    "type": ["string", "null"],
                    "description": "ISO-8601 cutoff for activity fields. None = no cutoff.",
                },
            },
            "required": ["name"],
        },
        callable=_call,
    )


def make_get_thread_history_tool(driver) -> KGTool:
    """Last n emails between sender → recipient (or both directions)."""

    def _call(
        *,
        sender: str,
        recipient: str,
        before_date: str | None = None,
        n: int = 5,
    ) -> list[dict]:
        out: list[dict] = []
        with driver.session() as session:
            for rec in session.run(
                tools_cypher.GET_THREAD_HISTORY,
                sender=sender,
                recipient=recipient,
                before_date=before_date,
                n=n,
            ):
                out.append({
                    "email_id": rec.get("email_id"),
                    "date": rec.get("date"),
                    "subject": _safe(rec.get("subject")),
                })
        return out

    return KGTool(
        name="get_thread_history",
        description=(
            "Last n email subject lines sent from `sender` to `recipient`, most "
            "recent first. Subject + date only — full body content is not stored "
            "in the graph. Use to recall the topics two people have been "
            "discussing."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "sender": {"type": "string"},
                "recipient": {"type": "string"},
                "before_date": {
                    "type": ["string", "null"],
                    "description": "ISO-8601 cutoff. Only emails strictly before this are returned.",
                },
                "n": {"type": "integer", "minimum": 1, "default": 5},
            },
            "required": ["sender", "recipient"],
        },
        callable=_call,
    )


ROLE_SUMMARY_SHORT_TRUNC = 160


def make_find_topic_collaborators_tool(driver) -> KGTool:
    """Top n people working on a topic by email volume."""

    def _call(*, topic: str, before_date: str | None = None, n: int = 5) -> list[dict]:
        out: list[dict] = []
        with driver.session() as session:
            for rec in session.run(
                tools_cypher.FIND_TOPIC_COLLABORATORS,
                topic=topic,
                before_date=before_date,
                n=n,
            ):
                role = _safe(rec.get("role_summary"))
                if len(role) > ROLE_SUMMARY_SHORT_TRUNC:
                    role = role[: ROLE_SUMMARY_SHORT_TRUNC - 1].rstrip() + "…"
                out.append({
                    "name": _safe(rec.get("name")),
                    "title": _safe(rec.get("title")),
                    "role_summary_short": role,
                    "n_emails_on_topic": int(rec.get("n_emails_on_topic") or 0),
                })
        return out

    return KGTool(
        name="find_topic_collaborators",
        description=(
            "Top n people working on a given topic, ranked by email volume on "
            "that topic. Use to identify who else cares about a subject before "
            "drafting a reply."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "before_date": {"type": ["string", "null"]},
                "n": {"type": "integer", "minimum": 1, "default": 5},
            },
            "required": ["topic"],
        },
        callable=_call,
    )


def make_get_org_chart_tool(driver) -> KGTool:
    """Multi-hop hierarchy view: bosses up to depth + reports down to depth."""

    def _call(*, name: str, depth: int = 2) -> dict:
        depth = max(1, min(int(depth), 5))  # cap so the query can't run away
        with driver.session() as session:
            record = session.run(
                tools_cypher.GET_ORG_CHART,
                name=name,
                depth=depth,
            ).single()
        if record is None:
            return {"name": name, "found": False}

        def _people(rows):
            return [
                {
                    "name": _safe(r.get("name")),
                    "title": _safe(r.get("title")),
                    "tier": r.get("tier"),
                    "depth": int(r.get("depth") or 0),
                }
                for r in (rows or [])
            ]

        return {
            "name": str(record.get("name") or name),
            "found": True,
            "title": _safe(record.get("title")),
            "tier": record.get("tier"),
            "bosses": sorted(
                _people(record.get("bosses")), key=lambda r: r["depth"]
            ),
            "reports": sorted(
                _people(record.get("reports")), key=lambda r: (r["depth"], r["name"])
            ),
        }

    return KGTool(
        name="get_org_chart",
        description=(
            "Multi-hop hierarchy view for one person: bosses up the chain and "
            "reports (plus their reports) down to `depth` levels. depth=1 is "
            "just direct manager + direct reports; depth=2 adds boss-of-boss "
            "and reports-of-reports. Capped at depth=5."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Canonical person name to anchor the chart on.",
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 2,
                },
            },
            "required": ["name"],
        },
        callable=_call,
    )


# Allowlist for the metric column substituted into GET_CENTRALITY_TEMPLATE.
# The Cypher template uses str.format() to fill in the metric name (Cypher
# doesn't support parametrised property access for ORDER BY); this list is
# the trust boundary that prevents arbitrary property names from reaching
# the database.
ALLOWED_CENTRALITY_METRICS = (
    "pagerank",
    "betweenness",
    "in_degree",
    "composite_score",
    "composite_score_v3",
)


def make_get_centrality_tool(driver) -> KGTool:
    """Top-N people by a chosen network-centrality / dominance metric."""

    def _call(
        *,
        metric: str = "composite_score_v3",
        top_n: int = 10,
        tier: int | None = None,
    ) -> list[dict] | dict:
        if metric not in ALLOWED_CENTRALITY_METRICS:
            return {
                "error": f"Unknown metric {metric!r}; allowed: {list(ALLOWED_CENTRALITY_METRICS)}",
            }
        query = tools_cypher.GET_CENTRALITY_TEMPLATE.format(metric=metric)
        out: list[dict] = []
        with driver.session() as session:
            for rec in session.run(query, top_n=int(top_n), tier=tier):
                score = rec.get("score")
                out.append({
                    "name": _safe(rec.get("name")),
                    "title": _safe(rec.get("title")),
                    "tier": rec.get("tier"),
                    "metric": metric,
                    "score": float(score) if score is not None else None,
                })
        return out

    return KGTool(
        name="get_centrality",
        description=(
            "Top-N people by a network-centrality / dominance metric "
            "(pagerank, betweenness, in_degree, composite_score, or "
            "composite_score_v3). Optional tier filter (1=executive ... 5=junior). "
            "Use to answer 'who's most central / influential / dominant in the org?'."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": list(ALLOWED_CENTRALITY_METRICS),
                    "default": "composite_score_v3",
                    "description": (
                        "Which score to rank by. composite_score_v3 is the "
                        "canonical RQ1 dominance metric."
                    ),
                },
                "top_n": {"type": "integer", "minimum": 1, "default": 10},
                "tier": {
                    "type": ["integer", "null"],
                    "description": "Optional tier filter (1-5 quintiles).",
                },
            },
            "required": [],
        },
        callable=_call,
    )


def make_get_dominance_score_tool(driver) -> KGTool:
    """Every centrality / dominance score for one person, plus their
    rank on composite_score_v3 (1 = highest)."""

    def _call(*, name: str) -> dict:
        with driver.session() as session:
            record = session.run(tools_cypher.GET_DOMINANCE_SCORE, name=name).single()
        if record is None:
            return {"name": name, "found": False}

        def _f(v):
            return float(v) if v is not None else None

        return {
            "name": str(record.get("name") or name),
            "found": True,
            "title": _safe(record.get("title")),
            "tier": record.get("tier"),
            "tier_v3": record.get("tier_v3"),
            "rank_v3": int(record.get("rank_v3") or 0),
            "scores": {
                "composite_score": _f(record.get("composite_score")),
                "composite_score_v3": _f(record.get("composite_score_v3")),
                "pagerank": _f(record.get("pagerank")),
                "betweenness": _f(record.get("betweenness")),
                "in_degree": _f(record.get("in_degree")),
            },
            "community": record.get("community"),
            "gt_level": _safe(record.get("gt_level")),
            "gt_level_numeric": _f(record.get("gt_level_numeric")),
        }

    return KGTool(
        name="get_dominance_score",
        description=(
            "Every dominance / centrality score for one person plus their "
            "rank on composite_score_v3 (the canonical RQ1 metric, 1=highest). "
            "Use to answer 'how dominant / central is X in the org?'."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Canonical person name.",
                },
            },
            "required": ["name"],
        },
        callable=_call,
    )


def build_default_registry(
    driver,
    *,
    pg_conn_factory: Callable[[], Any] | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> ToolRegistry:
    """Default factory: wires up the KG Chat tool catalog (15 tools).

    The catalog spans content retrieval (5), identity / disambiguation (2),
    uniquely-KG graph traversal (3), RQ1 metric alignment (2), bulk fetch (1),
    and escape hatches (2). See design spec for rationale.
    """
    from orggraph.pipeline.agents import tools_pg
    from orggraph.pipeline.agents import tools_query

    reg = ToolRegistry()

    # Identity (Neo4j)
    reg.register(make_get_person_tool(driver))
    reg.register(make_get_org_chart_tool(driver))

    # Topic / activity (Neo4j)
    reg.register(make_find_topic_collaborators_tool(driver))

    # Email metadata (Neo4j)
    reg.register(make_get_thread_history_tool(driver))

    # Network analytics (Neo4j)
    reg.register(make_get_centrality_tool(driver))
    reg.register(make_get_dominance_score_tool(driver))

    # Escape hatch (Neo4j)
    reg.register(tools_query.make_run_cypher_tool(driver))

    if pg_conn_factory is None:
        try:
            pg_conn_factory = tools_pg.default_pg_conn_factory()
        except ImportError:
            pg_conn_factory = None
    if embed_fn is None:
        try:
            embed_fn = tools_pg.default_embed_fn()
        except ImportError:
            embed_fn = None

    if pg_conn_factory is not None:
        reg.register(tools_pg.make_get_email_content_tool(pg_conn_factory))
        reg.register(tools_pg.make_get_thread_tool(pg_conn_factory))
        reg.register(tools_pg.make_get_pair_signals_tool(pg_conn_factory))
        reg.register(tools_pg.make_find_emails_tool(pg_conn_factory))
        reg.register(tools_pg.make_get_emails_bulk_tool(pg_conn_factory))
        # Escape hatch (Postgres)
        reg.register(tools_query.make_run_sql_tool(pg_conn_factory))
        if embed_fn is not None:
            reg.register(tools_pg.make_search_emails_semantic_tool(pg_conn_factory, embed_fn))
            reg.register(tools_pg.make_find_person_tool(pg_conn_factory, embed_fn))
    return reg


def build_rag_only_registry(
    *,
    pg_conn_factory: Callable[[], Any] | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> ToolRegistry:
    """RQ3 baseline: pure naive-RAG over the email corpus, no graph.

    Exposes exactly two tools:
      - ``search_emails_semantic``: cosine-similarity retrieval over the
        768-d email embeddings.
      - ``get_email_content``: fetch a full email by id.

    Deliberately omits structured tools (find_emails, get_pair_signals,
    get_org_chart, run_cypher, run_sql, etc.) and the person-resolver
    (find_person, which is itself vector-over-person-embeddings, would
    blur the "email-only RAG" baseline). This is the unembellished RAG
    setup that productionized text-RAG systems start from before any
    enhancement, against which GraphRAG's structured retrieval is the
    thesis-claim contrast.
    """
    from orggraph.pipeline.agents import tools_pg

    reg = ToolRegistry()

    if pg_conn_factory is None:
        pg_conn_factory = tools_pg.default_pg_conn_factory()
    if embed_fn is None:
        embed_fn = tools_pg.default_embed_fn()

    reg.register(tools_pg.make_search_emails_semantic_tool(pg_conn_factory, embed_fn))
    reg.register(tools_pg.make_get_email_content_tool(pg_conn_factory))
    return reg


RAG_SYSTEM_PROMPT = """You are an assistant answering questions about Enron Corporation based on the company's email corpus.

You have two tools:
- search_emails_semantic(query, k): semantic search over the corpus. Returns top-k matching emails with sender, recipient, date, snippet.
- get_email_content(email_id): fetch the full text of one email.

Strategy:
1. Run one or more semantic searches with focused queries.
2. If a snippet looks promising, fetch the full email with get_email_content.
3. Compose your answer from what you found.

Important constraints:
- You cannot count, aggregate, rank across people, or traverse reporting relationships. The only signal you have is what text comes back from semantic search.
- If the question requires aggregation, ranking, or relational reasoning, retrieve relevant emails and report what they actually say. Be honest if the retrieved content does not establish a clear answer — "the retrieved emails do not establish this" is better than fabricating.

Answer concisely. Do not hedge if the retrieved emails give a clear answer."""


def build_rag_enhanced_registry(
    *,
    pg_conn_factory: Callable[[], Any] | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> ToolRegistry:
    """RQ3 mid-tier: naive RAG plus thread (parent-document) expansion.

    Exposes three tools:
      - ``search_emails_semantic``: cosine-similarity retrieval over emails.
      - ``get_email_content``: fetch one email by id.
      - ``get_thread``: fetch the entire thread containing a given email.

    "Parent-document retrieval" is the canonical first enhancement applied
    to naive RAG in production text-RAG systems: retrieve at the chunk
    level, then expand to the surrounding document (here, the thread).
    Including this arm guards the RQ3 result against the reviewer
    objection that we compared GraphRAG only to a deliberately weak RAG
    baseline. If structured retrieval still wins against enhanced RAG,
    the claim survives.

    Deliberately omits structured tools (find_emails, get_pair_signals,
    get_org_chart, run_cypher, run_sql, find_person, etc.) and graph
    traversal -- those are GraphRAG-specific. This stays strictly within
    the text-RAG family.
    """
    from orggraph.pipeline.agents import tools_pg

    reg = ToolRegistry()

    if pg_conn_factory is None:
        pg_conn_factory = tools_pg.default_pg_conn_factory()
    if embed_fn is None:
        embed_fn = tools_pg.default_embed_fn()

    reg.register(tools_pg.make_search_emails_semantic_tool(pg_conn_factory, embed_fn))
    reg.register(tools_pg.make_get_email_content_tool(pg_conn_factory))
    reg.register(tools_pg.make_get_thread_tool(pg_conn_factory))
    return reg


RAG_ENHANCED_SYSTEM_PROMPT = """You are an assistant answering questions about Enron Corporation based on the company's email corpus.

You have three tools:
- search_emails_semantic(query, k): semantic search over the corpus. Returns top-k matching emails with sender, recipient, date, snippet.
- get_email_content(email_id): fetch the full text of one email.
- get_thread(email_id): fetch every email in the thread that contains the given email -- useful for surrounding context, the chain of replies, who else was involved.

Strategy:
1. Run one or more semantic searches with focused queries.
2. If a snippet looks promising, fetch the full email with get_email_content.
3. If the email's context matters (a deal discussion, an escalation chain, a back-and-forth), expand to the full thread with get_thread to see the surrounding conversation.
4. Compose your answer from what you retrieved.

Important constraints:
- You cannot count, aggregate, rank across people, or traverse reporting relationships. The only signal you have is what text comes back from semantic search and thread expansion.
- If the question requires aggregation, ranking, or relational reasoning beyond a single thread, retrieve relevant emails and threads and report what they actually say. Be honest if the retrieved content does not establish a clear answer.

Answer concisely. Do not hedge if the retrieved emails give a clear answer."""

