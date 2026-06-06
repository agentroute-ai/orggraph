# OrgGraph explorer (Streamlit)

A static, read-only Streamlit app for exploring the organizational knowledge
extracted from the Enron email corpus. Every page reads the committed export
artifacts in `datasets/enron/` and `data/` - **no live database and no LLM are
required**, so the app runs anywhere, including free public hosting.

## Run locally

From the repo root:

```bash
pip install -e ".[dashboard]"      # or: make dashboard
streamlit run dashboard/Home.py
```

Default URL: <http://localhost:8501>.

## Pages

| Page | Reads | Shows |
|------|-------|-------|
| Home | headline metrics | corpus size, employees mapped, personas, tiers, topics |
| The Graph | `processed/communication_graph.gpickle`, `extracted_hierarchy.csv` | network topology, top contributors, interactive top-50 subgraph, most-senior ranking |
| Personas | hierarchy artifacts + `persona_prompts/` | per-employee profile cards: scores, LLM role summary, persona prompt |
| Projects | `processed/projects.csv` | discovered project clusters by email volume |
| Topics | `processed/topics.csv`, `cluster_names.jsonl` | discovered topic clusters by email volume |
| Pipeline | (static) | the Stage 0-9 pipeline diagram and per-stage detail |

Some per-person drill-downs on the Personas page (the raw email list and the
per-person project/topic breakdown) need the full `clean_emails.parquet`, which
is not shipped. They degrade to empty until you regenerate it with `make corpus`.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (already public at `agentroute-ai/orggraph`).
2. On <https://share.streamlit.io>, create an app from the repo.
3. Set the main file path to `dashboard/Home.py` and the Python version to **3.12**.
4. The root `requirements.txt` (`.[dashboard]`) and `.streamlit/config.toml` are
   picked up automatically.

Because the app only reads the committed exports, the deployed instance needs no
secrets, no database, and no model endpoint.
