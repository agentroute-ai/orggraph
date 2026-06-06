"""About / methodology - what OrgGraph is, the research questions, and the data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.data import DATA_DIR, PROCESSED, load_extracted_hierarchy
from lib.header import render_header

st.set_page_config(page_title="About - OrgGraph", layout="wide")
render_header(
    title="About OrgGraph",
    subtitle="Unsupervised organizational knowledge extraction from email.",
)

st.markdown(
    "**OrgGraph** recovers organizational structure from an email corpus without "
    "supervision, then uses it to ground multi-agent simulations. It is the software "
    "for a Master's thesis in Artificial Intelligence at Università della Svizzera "
    "italiana (USI), by **Riccardo Sacco**, supervised by **Prof. Lonneke van der Plas**."
)

# --- by the numbers (from committed data) -----------------------------------
def _n(path: Path, kind: str) -> int:
    try:
        if kind == "json_list":
            return len(json.loads(path.read_text()))
        if kind == "csv":
            return len(pd.read_csv(path))
        if kind == "dir_txt":
            return sum(1 for f in path.iterdir() if f.suffix == ".txt")
    except Exception:
        return 0
    return 0

corpus = {}
cs = Path(__file__).resolve().parents[2] / "data" / "corpus_stats.json"
if cs.is_file():
    corpus = json.loads(cs.read_text())

cols = st.columns(4)
cols[0].metric("Emails analyzed", f"{corpus.get('total_emails', 0):,}" if corpus else "157K+")
cols[1].metric("Maildir custodians", f"{_n(DATA_DIR / 'employees.json', 'json_list')}")
cols[2].metric("People in the org graph", f"{len(load_extracted_hierarchy())}")
cols[3].metric("LLM personas", f"{_n(PROCESSED / 'persona_prompts', 'dir_txt')}")

st.divider()

st.subheader("Research questions")
st.markdown(
    """
- **RQ1 - Structure recovery.** How much organizational structure can be recovered
  *unsupervised* from email? Evaluated as **pair-classification accuracy** on
  manager/subordinate dominance pairs. Agarwal et al. (2012) report ~83.9% on **their
  own** pair set; that is a reference point, **not a head-to-head baseline** (different
  pairs and employee subset).
- **RQ2 - Dialogue naturalness.** Do multi-agent systems produce more natural dialogue
  than a single LLM with long context? Evaluated with human review and an LLM judge over
  five naturalness dimensions. The dashboard shows a **pilot** (5 scenarios x 3 conditions).
- **RQ3 - Retrieval and coherence.** Does graph-structured retrieval (GraphRAG) improve
  agent coherence versus vector-only retrieval? An **exploratory** A/B comparison, reported
  as found (no win for either side in the pilot).

Full pre-registered success criteria live in the thesis.
"""
)

st.subheader("Data and ground truth")
st.markdown(
    """
- **Corpus:** the public Enron email dataset (~517K raw messages; this build filters to a
  clean working set).
- **Maildir custodians:** 148 inbox owners after de-duplication.
- **Ground truth:** the Agarwal et al. (2012) *Comprehensive Gold Standard for the Enron
  Organizational Hierarchy* (dominance pairs), plus a 103-person hierarchy (v2) built from
  EnronData.org custodian titles (CC BY 3.0). A 30-person hand-curated senior-executive set
  is kept as a subset evaluation.
"""
)

st.subheader("How the pipeline works")
st.markdown(
    "A nine-stage pipeline: filter the corpus, embed emails, extract LLM signals "
    "(speech acts, sentiment, topics), discover project/topic clusters, aggregate dyadic "
    "signals into a knowledge graph, enrich people into personas, score a composite "
    "dominance ranking, load the graph into Neo4j, and evaluate. See the **Pipeline** page "
    "for the stage-by-stage detail."
)

st.subheader("About this dashboard")
st.markdown(
    """
This is a **static explorer**: every page reads committed export files, so it runs with
**no database and no model endpoint**. The genuinely live agent chat (which calls a local
model and Neo4j) is kept as a local/defence demo; the **Agent demo** page here replays
recorded dialogues instead.
"""
)

st.divider()
st.markdown(
    """
**License:** MIT (code).
**Repository:** https://github.com/agentroute-ai/orggraph
**Related project:** [AgentRoute](https://github.com/agentroute-ai) - the productised,
general-purpose descendant of this simulation runtime (not a dependency of OrgGraph).
"""
)
