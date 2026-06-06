"""End-to-end OrgGraph pipeline visualization for the supervisor demo."""
from __future__ import annotations

import streamlit as st

from lib.header import render_header

st.set_page_config(page_title="Pipeline - OrgGraph", layout="wide")
render_header(
    title="Pipeline",
    subtitle=(
        "End-to-end Stage 0 → Stage 9 flow from raw maildir to thesis figures. "
        "Each box is a pipeline module under `src/orggraph/pipeline/`."
    ),
)


# ---------------------------------------------------------------------------
# Pipeline stages: kept in sync with the Stage 0-9 DAG documented in the README.
# ---------------------------------------------------------------------------

STAGES = [
    {
        "id": "raw",
        "label": "Raw maildir\\n517K emails",
        "title": "Raw Enron corpus",
        "phase": "Source",
        "module": "—",
        "tagline": "517K raw emails from 158 custodian inboxes",
        "inputs": "FERC's 2003 release, 158 custodian inboxes",
        "outputs": "517,401 raw email messages",
        "description": (
            "The Enron corpus was released by the Federal Energy Regulatory Commission in 2003 "
            "as part of the company's bankruptcy investigation. We read it from the public "
            "HuggingFace mirror (`sujan-maharjan/enron_email_dataset`). No filtering yet."
        ),
    },
    {
        "id": "s0",
        "label": "Stage 0:corpus.filter\\nclean_emails.parquet",
        "title": "Filter raw corpus and resolve identities",
        "phase": "Data",
        "module": "`orggraph.pipeline.corpus.filter`",
        "tagline": "Filter, resolve sender + recipient identities",
        "inputs": "Raw maildir parquet (517K rows)",
        "outputs": "`clean_emails.parquet` (~157K rows with `sender_resolved`, `recipients_resolved`)",
        "description": (
            "Drops automated, foreign-language, and out-of-band emails. Keeps bodies between "
            "100 and 2,000 characters. Resolves each `From:` / `To:` address against a "
            "schema.org alias map (`givenName`, `additionalName`, `familyName`, `alternateName`) "
            "to attach a canonical Person name. The alias map is what the Whalley fix patched."
        ),
    },
    {
        "id": "s1",
        "label": "Stage 1:email.embed\\nembeddings_email (pgvector, 768-d)",
        "title": "Embed each email body",
        "phase": "Data",
        "module": "`orggraph.pipeline.email.embed`",
        "tagline": "768-d email embeddings (EmbeddingGemma, local)",
        "inputs": "`clean_emails.parquet`",
        "outputs": "`embeddings_email` table in pgvector (768-d per row)",
        "description": (
            "Embeds each email body with EmbeddingGemma (768 dimensions) running locally via "
            "Ollama. Output goes to a pgvector column. Resumable: skips email_ids already "
            "embedded. Embeddings are reused by Stage 2b clustering and by RQ3 retrieval."
        ),
    },
    {
        "id": "s1_5",
        "label": "Stage 1.5:email.metadata\\n20 deterministic per-email cols",
        "title": "Deterministic per-email features",
        "phase": "Data",
        "module": "`orggraph.pipeline.email.metadata`",
        "tagline": "20 regex/lexicon features per email (no LLM)",
        "inputs": "`embeddings_email` rows from Stage 1",
        "outputs": "20 metadata columns added in place (`is_thread_initiator`, `is_off_hours`, "
                   "`politeness_score`, `hedge_count`, `body_word_count`, etc.)",
        "description": (
            "Pure regex and lexicon features over the email body and headers. Computes "
            "politeness markers, hedges, off-hours flag, thread-initiator boolean, recipient "
            "counts, and similar deterministic signals. No LLM call. These features become "
            "the A1 ablation rung on top of A0 structural centrality."
        ),
    },
    {
        "id": "s2a",
        "label": "Stage 2a:signals.extract\\nspeech_acts JSONB · sentiment · topics",
        "title": "LLM speech-act extraction",
        "phase": "Signals",
        "module": "`orggraph.pipeline.signals.extract`",
        "tagline": "LLM tags speech acts and decisions per email",
        "inputs": "`embeddings_email` rows with metadata from Stage 1.5",
        "outputs": "`speech_acts` JSONB, `sentiment`, `topics`, `decision_carrying`, "
                   "`action_required`, `commitment_made` columns",
        "description": (
            "LLM (`MiniMax-M2.7-AWQ-4bit` via vLLM) tags each email with speech acts "
            "(`request`, `commit`, `deliver`, `propose`), decision/action flags, and a sentiment "
            "score. ~157K LLM calls:the most expensive stage by far. Resumable: skips emails "
            "with `signals_extracted_at IS NOT NULL`. These become the A2 ablation rung."
        ),
    },
    {
        "id": "s2b",
        "label": "Stage 2b:clusters.discover\\nprojects.csv · topics.csv\\n(UMAP + HDBSCAN + LLM naming)",
        "title": "Unsupervised topic and project discovery",
        "phase": "Signals",
        "module": "`orggraph.pipeline.clusters.discover`",
        "tagline": "UMAP + HDBSCAN find Projects/Topics; LLM names them",
        "inputs": "Email embeddings from Stage 1",
        "outputs": "`projects.csv` (450 large clusters), `topics.csv` (~2,800 finer topics), "
                   "`cluster_labels.parquet`",
        "description": (
            "UMAP reduces the 768-d email embeddings to 2D. HDBSCAN then clusters them at two "
            "granularities:large Projects and finer Topics. The LLM names each cluster from "
            "a sample of its representative emails. The Project / Topic graph nodes downstream "
            "all originate here."
        ),
    },
    {
        "id": "s2c",
        "label": "Stage 2c:clusters.canonicalize\\ntopic_canonicalization.json",
        "title": "Merge near-duplicate cluster names",
        "phase": "Signals",
        "module": "`orggraph.pipeline.clusters.canonicalize`",
        "tagline": "Merge near-duplicate cluster names",
        "inputs": "`projects.csv`, `topics.csv`",
        "outputs": "`topic_canonicalization.json` (alias map of duplicates)",
        "description": (
            "Sibling clusters with semantically equivalent LLM-generated names "
            "(e.g. \"Q3 forecast\" and \"Q3 forecasting\") get merged into a single "
            "canonical bucket. Cuts topic dimensionality and removes the LLM-naming "
            "drift that would otherwise look like distinct topics."
        ),
    },
    {
        "id": "s3",
        "label": "Stage 3:signals.aggregate\\npair_signals · Person/Email/Project/Topic in Neo4j",
        "title": "Roll up to Person and pair level",
        "phase": "Aggregation",
        "module": "`orggraph.pipeline.signals.aggregate`",
        "tagline": "Per-Person and per-pair rollups → Neo4j",
        "inputs": "`embeddings_email` (post-Stage-2a)",
        "outputs": "`pair_signals` table (postgres), Person / Email / Project / Topic nodes "
                   "and `COMMUNICATES_WITH` edges in Neo4j",
        "description": (
            "Aggregates per-email signals up to per-Person rollups (counts, percentages, "
            "means) and per-pair rollups (request/commit ratio, length asymmetry, reply "
            "latency). Writes to postgres `pair_signals` and merges Person nodes into Neo4j. "
            "This is where Lawrence Whalley's Person node gets created post-fix."
        ),
    },
    {
        "id": "s4",
        "label": "Stage 4:enrich.persons / entities / pairs\\nperson_enrichment.csv et al.",
        "title": "LLM persona and pair-deference enrichment",
        "phase": "Enrichment",
        "module": "`orggraph.pipeline.enrich.{persons,entities,pairs}`",
        "tagline": "LLM persona profiles + per-pair deference scores",
        "inputs": "Person nodes from Stage 3, sample emails per Person",
        "outputs": "`person_enrichment.csv`, `entity_enrichment.csv`, `pair_enrichment.csv` "
                   "(role summary, persona attributes, deference scores)",
        "description": (
            "Three LLM passes. `enrich.persons` produces a hybrid deterministic + LLM persona "
            "profile per employee with N-best generation and a quality gate. `enrich.entities` "
            "does the same for the 425 external organizations. `enrich.pairs` LLM-scores "
            "deference direction for every pair with at least 5 exchanges (~700 pairs)."
        ),
    },
    {
        "id": "s5",
        "label": "Stage 5:score.recompute\\ncomposite_v3 (k-fold alpha tuning)",
        "title": "Recompute composite seniority score",
        "phase": "Scoring",
        "module": "`orggraph.pipeline.score.recompute`",
        "tagline": "Composite seniority score (k-fold alpha tuning)",
        "inputs": "Neo4j Person nodes, `DEFERS_TO` edges",
        "outputs": "Updated `Person.composite_score_v3`, `tier_v3`, `composite_weights.json`, "
                   "`tier2_alpha.json`",
        "description": (
            "Sweeps a weight grid for the LLM-seniority blend, k-fold cross-validates the "
            "Tier-2 dominance alpha, and writes back a refreshed composite score per Person. "
            "Refines `REPORTS_TO` from the strongest `DEFERS_TO` edge per subordinate. "
            "Depends on Stage 6b having seeded `DEFERS_TO`."
        ),
    },
    {
        "id": "s6",
        "label": "Stage 6:kg.load · kg.sync\\nMVP load + enrichment sync to Neo4j",
        "title": "Build the knowledge graph",
        "phase": "Knowledge graph",
        "module": "`orggraph.pipeline.kg.{load,sync}`",
        "tagline": "Build typed KG; sync LLM data into Neo4j",
        "inputs": "Person CSVs, `clients_suppliers.json`, `pair_enrichment.csv`",
        "outputs": "Neo4j: Person, Team, ExternalEntity, Function nodes; `MEMBER_OF`, "
                   "`COMMUNICATES_WITH`, `REPORTS_TO`, `DEFERS_TO`, `KNOWS_ABOUT` edges",
        "description": (
            "`kg.load` builds the MVP graph (Persons, Teams, External entities, "
            "intra/external communication edges). `kg.sync` is the only place LLM-derived "
            "data lands in the graph: persona properties on Person nodes, plus `DEFERS_TO` "
            "edges materialised from `pair_enrichment.csv` deference scores."
        ),
    },
    {
        "id": "s7",
        "label": "Stage 7:kg.embed\\nPerson + Entity descriptions → pgvector",
        "title": "Embed Person and Entity descriptions",
        "phase": "Knowledge graph",
        "module": "`orggraph.pipeline.kg.embed`",
        "tagline": "Embed Person/Entity descriptions for GraphRAG",
        "inputs": "Person and ExternalEntity nodes (post-sync), with role summaries and "
                   "narrative descriptions",
        "outputs": "`embeddings_person`, `embeddings_entity` tables in pgvector",
        "description": (
            "Re-embeds each Person and ExternalEntity using their narrative description "
            "(role summary, expertise topics, communication style). Output feeds RQ3's "
            "graph-vs-vector retrieval comparator: implicit hierarchy queries can hit "
            "either the typed graph or the dense semantic index."
        ),
    },
    {
        "id": "s8",
        "label": "Stage 8:eval.qc / rq1 / run_rq1\\nllm_quality_report.md · RQ1 ablation A0–A5",
        "title": "Quality control and RQ1 evaluation",
        "phase": "Evaluation",
        "module": "`orggraph.pipeline.eval.{qc,rq1,run_rq1}`",
        "tagline": "QC + RQ1 A0–A5 ablation vs ground truth",
        "inputs": "Neo4j Person nodes (with all features), `dominance_pairs_v2.csv` GT",
        "outputs": "`llm_quality_report.md`, `rq1_ablation.csv` (A0–A5 with bootstrap CIs), "
                   "`rq1_results.json`, `extracted_hierarchy.csv`",
        "description": (
            "`eval.qc` audits LLM enrichment for missing fields and quality-gate failures. "
            "`eval.rq1` runs the A0–A5 ablation against the EnronData GT v2 (103 employees, "
            "3,814 pairs) with paired-bootstrap 95% confidence intervals over the evaluable "
            "pair set. `run_rq1` is the single-shot A0 entry point used as a smoke test."
        ),
    },
    {
        "id": "s9",
        "label": "Stage 9:figures\\nthesis/figures/results/",
        "title": "Render thesis figures",
        "phase": "Output",
        "module": "`orggraph.pipeline.figures`",
        "tagline": "Render thesis figures (network plots, ablation table)",
        "inputs": "All artifacts produced by Stages 0–8",
        "outputs": "PDFs and PNGs under `thesis/figures/results/`",
        "description": (
            "Generates the figures cited from the thesis: ablation table, hierarchy comparison, "
            "network community plots, RQ3 retrieval-comparator screenshots. Output is what "
            "actually ends up in `\\includegraphics` calls in the LaTeX source."
        ),
    },
]


PHASE_COLORS = {
    "Source":          "#94a3b8",
    "Data":            "#60a5fa",
    "Signals":         "#34d399",
    "Aggregation":     "#a78bfa",
    "Enrichment":      "#fbbf24",
    "Scoring":         "#fb923c",
    "Knowledge graph": "#f472b6",
    "Evaluation":      "#22d3ee",
    "Output":          "#facc15",
}


def _build_dot() -> str:
    lines: list[str] = [
        'digraph orggraph_pipeline {',
        '  rankdir=TB;',
        '  bgcolor="white";',
        '  size="5,11";',
        '  ratio=compress;',
        '  nodesep=0.18;',
        '  ranksep=0.32;',
        '  node ['
        '    shape=box, style="filled,rounded", fontname="Helvetica",'
        '    fontsize=10, margin="0.12,0.06", penwidth=0,'
        '    width=3.4, fixedsize=false',
        '  ];',
        '  edge [color="#475569", arrowhead=vee, arrowsize=0.6, penwidth=1.0];',
    ]

    for stage in STAGES:
        color = PHASE_COLORS.get(stage["phase"], "#cbd5e1")
        font_color = "white" if stage["phase"] != "Output" else "#0b1220"
        label = stage["label"]
        lines.append(
            f'  {stage["id"]} ['
            f'label="{label}", fillcolor="{color}", fontcolor="{font_color}"];'
        )

    # Linear chain Stage 0 → 9
    chain = [s["id"] for s in STAGES]
    for u, v in zip(chain, chain[1:]):
        lines.append(f"  {u} -> {v};")

    lines.append("}")
    return "\n".join(lines)


col_chart, col_legend = st.columns([3, 2])

with col_chart:
    st.graphviz_chart(_build_dot(), use_container_width=False)

with col_legend:
    st.subheader("Stages")
    for stage in STAGES:
        color = PHASE_COLORS.get(stage["phase"], "#cbd5e1")
        # First line of label, e.g. "Stage 0:corpus.filter"
        head = stage["label"].split("\\n")[0]
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:.55rem;margin:.35rem 0;">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{color};border-radius:2px;margin-top:.35rem;flex-shrink:0;"></span>'
            f'<div style="line-height:1.25;">'
            f'<div style="font-size:0.88rem;font-weight:600;color:#0f172a;">{head}</div>'
            f'<div style="font-size:0.8rem;color:#64748b;">{stage["tagline"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "Linear chain:each stage reads what the previous produced. "
        "Color indicates phase."
    )

st.divider()

# ---------------------------------------------------------------------------
# Stage selector:pick a stage to see what it does
# ---------------------------------------------------------------------------

st.subheader("Stage detail")
st.caption("Pick a stage to read what it does, what it reads, and what it produces.")

_STAGE_OPTIONS = {f'{s["label"].split(chr(92) + "n")[0]}': s["id"] for s in STAGES}
_default_label = next(iter(_STAGE_OPTIONS))
selected_label = st.selectbox(
    "Stage",
    options=list(_STAGE_OPTIONS.keys()),
    index=1,  # default to Stage 0:first interesting stage
    label_visibility="collapsed",
)
selected_id = _STAGE_OPTIONS[selected_label]
selected = next(s for s in STAGES if s["id"] == selected_id)

phase_color = PHASE_COLORS.get(selected["phase"], "#cbd5e1")
st.markdown(
    f'<div style="display:flex;align-items:center;gap:.6rem;margin:0.5rem 0 0.25rem;">'
    f'<span style="display:inline-block;width:14px;height:14px;'
    f'background:{phase_color};border-radius:3px;"></span>'
    f'<span style="font-size:1.1rem;font-weight:600;">{selected["title"]}</span>'
    f'<span style="color:#64748b;font-size:0.9rem;">· {selected["phase"]} phase</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown(selected["description"])

col_in, col_out = st.columns(2)
with col_in:
    st.markdown("**Reads**")
    st.markdown(selected["inputs"])
with col_out:
    st.markdown("**Writes**")
    st.markdown(selected["outputs"])

st.markdown(f"**Module:** {selected['module']}")

st.divider()

st.markdown(
    """
**How to read this**

- Each box is a numbered pipeline stage backed by a module under `src/orggraph/pipeline/`.
- Arrows show data flow: each stage reads the artifacts produced by the stage(s) above it.
- Color groups stages by phase. The full chain runs end-to-end via `make pipeline`.
- Sibling subpackages `validate/` (external-corpus validation) and `bench/` (performance benchmarks) sit
  outside this chain and are opt-in via `make validate` / `make bench`.
"""
)

with st.expander("CLI shortcuts"):
    st.code(
        """make pipeline           # Stage 0 → Stage 9 end-to-end
make corpus             # Stage 0
make email              # Stage 1 + 1.5
make signals            # Stage 2a + Stage 3
make clusters           # Stage 2b + 2c
make enrich             # Stage 4
make score              # Stage 5
make kg                 # Stage 6 + 7
make eval               # Stage 8
make figures            # Stage 9""",
        language="bash",
    )
