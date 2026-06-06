# OrgGraph

**Unsupervised Organizational Knowledge Extraction and Agent Simulation from Email Corpora**

Master's thesis software, MSc in Artificial Intelligence, Università della Svizzera italiana (USI)

**Author:** Riccardo Sacco
**Supervisor:** Prof. Lonneke van der Plas

[![tests](https://github.com/agentroute-ai/orggraph/actions/workflows/tests.yml/badge.svg)](https://github.com/agentroute-ai/orggraph/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python: 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

## Abstract

OrgGraph recovers organizational structure from email metadata and content without
supervision, evaluates multi-agent dialogue naturalness against single-LLM baselines,
and compares graph-structured (GraphRAG) versus vector-only retrieval for agent
communication coherence. This repository contains the library and command-line
pipeline (corpus filter, email embedding, LLM signal extraction, project/topic
clustering, knowledge-graph construction, persona enrichment, RQ1 evaluation) together
with the multi-agent simulation runtime used for the RQ2 and RQ3 studies.

## Research questions

| RQ | Question | Evaluation |
|----|----------|------------|
| **RQ1** | How much organizational structure can be recovered unsupervised from email? | Pair-classification accuracy on Agarwal-style dominance pairs. Agarwal et al. (2012) report accuracy on their own pair set, which is a reference point and not a head-to-head baseline (different pairs, different employee subset). |
| **RQ2** | Do multi-agent systems produce more natural dialogue than a single LLM with long context? | Human evaluation plus LLM-as-judge over G-Eval style dimensions. |
| **RQ3** | Does graph-structured retrieval (GraphRAG) improve agent coherence vs vector-only? | A/B comparison on stratified question/answer pairs from `enron_qa_0922`. |

## Repository structure

```
orggraph/
├── README.md
├── LICENSE                          # MIT
├── CITATION.cff                     # machine-readable citation
├── Makefile                         # one entry point for every workflow
├── pyproject.toml                   # editable install + console scripts
├── docker-compose.yml               # postgres+pgvector, neo4j, vllm, litellm
├── litellm-config.example.yaml      # multi-endpoint LLM router (copy to litellm-config.yaml)
├── .env.example                     # copy to .env and edit
│
├── src/orggraph/                    # library + orggraph-* CLI
│   ├── data/                          # corpus loader, identity resolution, network
│   ├── extraction/                    # community detection, hierarchy
│   ├── evaluation/                    # RQ1 metrics, dialogue judge, text metrics
│   ├── llm/                           # client, prompts, sampling, parsing
│   ├── agents/                        # persona agents
│   ├── simulation/                    # A2A bus, runner, scenarios (RQ2/RQ3 runtime)
│   └── pipeline/                      # 9-stage pipeline (see below)
│
├── dashboard/                       # static Streamlit explorer (no DB/LLM needed)
├── scripts/                         # numbered reproduction scripts (01-10)
├── docker/                          # neo4j + postgres init scripts
├── docs/                            # dev-setup, ground-truth research
├── data/                            # committed Neo4j graph export (cypher.gz)
└── datasets/enron/                  # committed derived artifacts (ground truth,
                                     #   hierarchy, clusters, personas, comm. graph)
```

The repository ships the **derived knowledge graph and analysis artifacts** so the
dashboard, the RQ1 evaluation, and the figures work without re-running the LLM pipeline.
The raw corpus and the large embedding artifacts are not committed (regenerate them from
HuggingFace via `make corpus` / the pipeline); see [Reproducing results](#reproducing-results).

## Pipeline

```
Stage 0    corpus.filter             raw maildir to clean parquet
Stage 1    email.embed               768-d embeddings to pgvector
Stage 1.5  email.metadata            deterministic per-email features
Stage 2a   signals.extract           LLM speech acts, sentiment, topics
Stage 2b   clusters.discover         UMAP + HDBSCAN project/topic discovery
Stage 2c   clusters.canonicalize     LLM topic canonicalization
Stage 3    signals.aggregate         dyadic pair_signals + Person aggregates to Neo4j
Stage 4    enrich.{persons,entities,pairs}  hybrid deterministic + LLM persona
Stage 5    score.recompute           composite v3 with k-fold alpha tuning
Stage 6    kg.{load,sync}            Neo4j MVP load + enrichment sync
Stage 7    kg.embed                  Person + Entity description embeddings to pgvector
Stage 8    eval.{qc,rq1,run_rq1}     QC gates + RQ1 ablation A0 to A5
Stage 9    figures                   render result figures
```

Sibling, opt-in subpackages: `validate/` (BC3 speech-act agreement, Berkeley genre ARI,
RQ3 eval-set builder) and `bench/` (LLM concurrency, email-extraction throughput).

## Quickstart

```bash
git clone https://github.com/agentroute-ai/orggraph.git
cd orggraph
python3.12 -m venv .venv && source .venv/bin/activate
make install                       # pip install -e ".[dev]" + pre-commit hooks
```

## Dashboard

A static Streamlit explorer of the extracted organization (communication network,
LLM personas, project/topic clusters, pipeline overview). It reads the committed
exports only, so it needs no database and no model endpoint:

```bash
make dashboard                     # pip install -e ".[dashboard]" + streamlit run dashboard/Home.py
```

It also deploys as-is to Streamlit Community Cloud (main file `dashboard/Home.py`,
Python 3.12); see [dashboard/README.md](dashboard/README.md).

## Reproducing results

Two paths, depending on whether you have GPU/LLM access.

### Path A - verify from the committed artifacts (no GPU, no LLM)

The knowledge graph and analysis artifacts are committed, so you can rebuild the
headline results directly:

```bash
docker compose up -d               # start postgres + neo4j
make restore                       # load data/neo4j_export.cypher.gz into Neo4j
make eval                          # Stage 8: RQ1 evaluation on the committed hierarchy
make figures                       # Stage 9: result figures
make dashboard                     # explore the extracted organization
```

### Path B - run the full pipeline from scratch (needs an LLM endpoint)

```bash
cp litellm-config.example.yaml litellm-config.yaml   # edit endpoint hostnames
cp .env.example .env                                 # corpus + model settings
docker compose up -d               # postgres + neo4j + vllm + litellm
make pipeline                      # Stage 0 to Stage 9 end-to-end (downloads the
                                   #   Enron corpus from HuggingFace, calls the LLM)

# RQ2 multi-agent dialogue study
make rq2-pilot                     # run + judge + aggregate (needs LLM_BASE_URL, LLM_MODEL)

# opt-in (slow, external corpora needed)
make validate                      # BC3 + Berkeley + RQ3 eval-set
make bench                         # LLM concurrency benchmarks
```

Each pipeline stage is also a console script (`orggraph-corpus-filter`,
`orggraph-signals-extract`, and so on) and a `python -m` target. The numbered scripts in
`scripts/` build the ground truth (`01`) and run the RQ2/RQ3 experiments (`02` to `10`).
Run `make help` for the full target list.

## Citation

```bibtex
@mastersthesis{sacco2026orggraph,
  title  = {OrgGraph: Unsupervised Organizational Knowledge Extraction and Agent Simulation from Email Corpora},
  author = {Sacco, Riccardo},
  school = {Università della Svizzera italiana (USI)},
  year   = {2026},
  type   = {Master's thesis}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## Related project

[AgentRoute](https://github.com/agentroute-ai) (`pip install agentroute`) is the
productised, general-purpose descendant of this repository's agent-to-agent simulation
runtime. It is a separate project and not a dependency of OrgGraph.

## License

Code is released under the MIT License. See [LICENSE](LICENSE).
