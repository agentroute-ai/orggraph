"""Cached loaders for thesis data files.

Resolves paths against the repo root regardless of where Streamlit is launched.
All loaders are cached so the multi-page app stays snappy.
"""
from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

from orggraph.config import DATASETS_DIR, OUTPUT_DIR, REPO_ROOT
from orggraph.data.identity import build_alias_map
from orggraph.data.loader import load_emails as _load_emails

# Aliases preserved for backwards-compat with pages/ that import these names.
DATA_DIR = DATASETS_DIR
PROCESSED = OUTPUT_DIR


@lru_cache(maxsize=1)
def repo_root() -> Path:
    return REPO_ROOT


def _exists(p: Path) -> bool:
    return p.is_file()


# ---------------------------------------------------------------------------
# Direct file loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_employees() -> pd.DataFrame:
    path = DATA_DIR / "employees.json"
    if not _exists(path):
        return pd.DataFrame()
    raw = json.loads(path.read_text())
    rows = []
    for r in raw:
        rows.append({
            "name": r.get("name"),
            "email": r.get("email"),
            "given_name": r.get("givenName"),
            "family_name": r.get("familyName"),
            "alternate_name": r.get("alternateName"),
            "affiliation": (r.get("affiliation") or {}).get("legalName"),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_clients_suppliers() -> pd.DataFrame:
    path = DATA_DIR / "clients_suppliers.json"
    if not _exists(path):
        return pd.DataFrame()
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return pd.json_normalize(raw)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ground_truth() -> pd.DataFrame:
    path = PROCESSED / "employees_ground_truth.csv"
    return pd.read_csv(path) if _exists(path) else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_dominance_pairs() -> pd.DataFrame:
    path = PROCESSED / "dominance_pairs.csv"
    return pd.read_csv(path) if _exists(path) else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_extracted_hierarchy() -> pd.DataFrame:
    path = PROCESSED / "extracted_hierarchy.csv"
    return pd.read_csv(path) if _exists(path) else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_tier1_seniority() -> pd.DataFrame:
    path = PROCESSED / "tier1_seniority.csv"
    return pd.read_csv(path) if _exists(path) else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_rq1_results() -> dict:
    path = PROCESSED / "rq1_results.json"
    return json.loads(path.read_text()) if _exists(path) else {}


# ---------------------------------------------------------------------------
# Stage 2b cluster output (UMAP+HDBSCAN + LLM naming)
# ---------------------------------------------------------------------------


def _load_jsonl_clusters(kind: str) -> pd.DataFrame:
    """Live read of cluster_names.jsonl — appended after each LLM call.

    `kind` is 'project' or 'topic'.
    """
    path = PROCESSED / "cluster_names.jsonl"
    if not _exists(path):
        return pd.DataFrame()
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("cluster_kind") != kind:
                continue
            rows.append({
                f"{kind}_id": rec.get("cluster_id"),
                "name": rec.get("name", ""),
                "description": rec.get("description", ""),
                "n_emails": int(rec.get("n_emails", 0)),
                "_rep_ids": list(rec.get("representative_email_ids") or []),
                "representative_emails": json.dumps(
                    rec.get("representative_email_ids") or []
                ),
            })
    return pd.DataFrame(rows)


def _load_clusters(kind_singular: str, csv_filename: str) -> pd.DataFrame:
    """Prefer the finalized CSV; fall back to the live JSONL checkpoint."""
    csv_path = PROCESSED / csv_filename
    # Use the JSONL when no CSV yet, OR when the JSONL is newer than the CSV
    # (e.g. user is doing a `--rerun-naming` against a different LLM and
    # wants to see live progress).
    jsonl_path = PROCESSED / "cluster_names.jsonl"
    if csv_path.is_file() and (
        not jsonl_path.is_file()
        or csv_path.stat().st_mtime >= jsonl_path.stat().st_mtime
    ):
        df = pd.read_csv(csv_path)
        if "representative_emails" in df.columns:
            df["_rep_ids"] = df["representative_emails"].apply(_parse_rep_ids)
        return df
    return _load_jsonl_clusters(kind_singular)


@st.cache_data(show_spinner=False, ttl=10)
def load_projects() -> pd.DataFrame:
    """Load projects from CSV when present, else from the live JSONL (TTL 10s)."""
    return _load_clusters("project", "projects.csv")


@st.cache_data(show_spinner=False, ttl=10)
def load_topics() -> pd.DataFrame:
    """Load topics from CSV when present, else from the live JSONL (TTL 10s)."""
    return _load_clusters("topic", "topics.csv")


def _parse_rep_ids(s) -> list[str]:
    if isinstance(s, list):
        return s
    if isinstance(s, str) and s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            return []
    return []


@st.cache_data(show_spinner=False)
def load_topic_name_map() -> dict[str, str]:
    """topic_id (e.g. 'T012') → human-readable name from topics.csv."""
    df = load_topics()
    if df.empty or "topic_id" not in df.columns or "name" not in df.columns:
        return {}
    return dict(zip(df["topic_id"], df["name"]))


@st.cache_resource(show_spinner=False)
def load_communication_graph() -> nx.Graph | None:
    path = PROCESSED / "communication_graph.gpickle"
    if not _exists(path):
        return None
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Email loading (HuggingFace, slow on first call)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading emails from HuggingFace…")
def load_emails(limit: int | None = 50_000) -> pd.DataFrame:
    """Lazy-load Enron emails from HuggingFace.

    The full corpus is ~517k messages; default limit keeps the dashboard responsive.
    """
    return _load_emails(limit=limit)


@st.cache_data(show_spinner="Resolving sender aliases…")
def load_alias_map() -> dict[str, str]:
    return build_alias_map()


# ---------------------------------------------------------------------------
# Status / lineage
# ---------------------------------------------------------------------------


def _file_meta(p: Path) -> dict:
    if not p.is_file():
        return {"present": False, "size": 0, "mtime": None}
    stat = p.stat()
    return {
        "present": True,
        "size": stat.st_size,
        "mtime": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC"),
    }


def data_status() -> dict[str, bool]:
    """Quick presence check (used on the overview)."""
    return {k: v["present"] for k, v in pipeline_status().items()}


def pipeline_status() -> dict[str, dict]:
    """Per-file presence + size + mtime, ordered roughly by pipeline stage."""
    files = [
        ("employees.json", DATA_DIR / "employees.json"),
        ("clients_suppliers.json", DATA_DIR / "clients_suppliers.json"),
        ("processed/employees_ground_truth.csv", PROCESSED / "employees_ground_truth.csv"),
        ("processed/dominance_pairs.csv", PROCESSED / "dominance_pairs.csv"),
        ("processed/communication_graph.gpickle", PROCESSED / "communication_graph.gpickle"),
        ("processed/extracted_hierarchy.csv", PROCESSED / "extracted_hierarchy.csv"),
        ("processed/tier1_seniority.csv", PROCESSED / "tier1_seniority.csv"),
        ("processed/rq1_results.json", PROCESSED / "rq1_results.json"),
    ]
    return {label: _file_meta(p) for label, p in files}


# ---------------------------------------------------------------------------
# Derived analytics
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def network_summary() -> dict:
    """Dense network metrics — clustering, diameter, components, density."""
    G = load_communication_graph()
    if G is None:
        return {}
    UG = G.to_undirected()
    out = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(UG),
        "transitivity": nx.transitivity(UG),
        "weakly_connected_components": nx.number_weakly_connected_components(G),
        "strongly_connected_components": nx.number_strongly_connected_components(G),
    }
    largest = max(nx.weakly_connected_components(G), key=len)
    H = G.subgraph(largest).to_undirected()
    out["largest_wcc_nodes"] = H.number_of_nodes()
    out["largest_wcc_edges"] = H.number_of_edges()
    try:
        out["diameter"] = nx.diameter(H)
    except Exception:  # noqa: BLE001
        out["diameter"] = None
    try:
        out["avg_shortest_path"] = nx.average_shortest_path_length(H)
    except Exception:  # noqa: BLE001
        out["avg_shortest_path"] = None
    return out


@st.cache_data(show_spinner=False)
def degree_distribution() -> pd.DataFrame:
    G = load_communication_graph()
    if G is None:
        return pd.DataFrame()
    rows = []
    for n in G.nodes():
        rows.append({
            "node": n,
            "in_degree": G.in_degree(n),
            "out_degree": G.out_degree(n),
            "total_degree": G.in_degree(n) + G.out_degree(n),
            "weighted_in": G.in_degree(n, weight="weight"),
            "weighted_out": G.out_degree(n, weight="weight"),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def rq1_pair_predictions() -> pd.DataFrame:
    """Recompute per-pair correctness from extracted_hierarchy.composite_score."""
    hier = load_extracted_hierarchy()
    pairs = load_dominance_pairs()
    if hier.empty or pairs.empty:
        return pd.DataFrame()
    rank = dict(zip(hier["node"], hier["composite_score"]))
    rows = []
    for _, r in pairs.iterrows():
        sup, sub = r["superior"], r["subordinate"]
        sup_score = rank.get(sup)
        sub_score = rank.get(sub)
        if sup_score is None or sub_score is None:
            status = "missing"
            correct = False
        else:
            correct = sup_score > sub_score
            status = "correct" if correct else "wrong_order"
        rows.append({
            "superior": sup,
            "subordinate": sub,
            "superior_level": r.get("superior_level"),
            "subordinate_level": r.get("subordinate_level"),
            "superior_score": sup_score,
            "subordinate_score": sub_score,
            "score_gap": (sup_score - sub_score) if (sup_score is not None and sub_score is not None) else None,
            "status": status,
            "correct": correct,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def corpus_stats(limit: int = 50_000) -> dict:
    """Per-corpus stats: dates, sender domains, volume timeline."""
    df = load_emails(limit=limit)
    if df.empty:
        return {}
    out: dict = {"emails_loaded": len(df), "limit": limit}
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    invalid = dates.isna().sum()
    out["invalid_dates"] = int(invalid)
    out["valid_dates"] = int(dates.notna().sum())
    valid = dates.dropna()
    if not valid.empty:
        out["date_min"] = str(valid.min())
        out["date_max"] = str(valid.max())
    # Domain split
    domains = df["from"].astype(str).str.lower().str.extract(r"@([\w.\-]+)$")[0]
    domain_counts = domains.value_counts()
    out["unique_sender_domains"] = int(domain_counts.shape[0])
    out["enron_emails"] = int((domains == "enron.com").sum())
    out["external_emails"] = int((domains != "enron.com").sum())
    out["top_external_domains"] = domain_counts[domain_counts.index != "enron.com"].head(15).to_dict()
    # Volume timeline (monthly)
    monthly = (
        valid.dt.tz_convert("UTC").dt.to_period("M").value_counts().sort_index()
    )
    out["monthly_volume"] = {str(k): int(v) for k, v in monthly.items()}
    # Body / subject sizes
    out["empty_subjects"] = int((df["subject"].astype(str).str.strip() == "").sum())
    out["empty_bodies"] = int((df["body"].astype(str).str.strip() == "").sum())
    return out


@st.cache_data(show_spinner=False)
def persona_for(node: str, top_k: int = 8) -> dict:
    """Compose a persona profile for a single node from existing artifacts."""
    G = load_communication_graph()
    hier = load_extracted_hierarchy()
    gt = load_ground_truth()
    employees = load_employees()
    tier1 = load_tier1_seniority()

    profile: dict = {"node": node}
    if not employees.empty:
        emp = employees[employees["name"] == node]
        if not emp.empty:
            profile["email"] = emp.iloc[0]["email"]
            profile["alternate_name"] = emp.iloc[0]["alternate_name"]
    if not gt.empty:
        row = gt[gt["name"] == node]
        if not row.empty:
            profile["gt_level"] = row.iloc[0]["level"]
            profile["gt_level_numeric"] = int(row.iloc[0]["level_numeric"])
            profile["gt_title"] = row.iloc[0]["title"]
    if not hier.empty:
        row = hier[hier["node"] == node]
        if not row.empty:
            for col in ("composite_score", "tier", "pagerank", "betweenness",
                        "in_degree", "out_degree", "total_degree",
                        "response_time_ratio", "initiation_ratio",
                        "cc_frequency", "communication_breadth"):
                if col in row.columns:
                    profile[col] = float(row.iloc[0][col]) if col != "tier" else int(row.iloc[0][col])
    if not tier1.empty:
        row = tier1[tier1["name"] == node]
        if not row.empty:
            profile["llm_seniority"] = int(row.iloc[0].get("seniority", 0))
            profile["llm_reasoning"] = row.iloc[0].get("reasoning")
            profile["llm_model"] = row.iloc[0].get("model")
    if G is not None and node in G:
        # Top recipients (out) and senders (in)
        out_edges = sorted(G.out_edges(node, data=True), key=lambda e: -e[2].get("weight", 0))[:top_k]
        in_edges = sorted(G.in_edges(node, data=True), key=lambda e: -e[2].get("weight", 0))[:top_k]
        profile["top_recipients"] = [(u, int(d.get("weight", 0))) for _, u, d in out_edges]
        profile["top_senders"] = [(u, int(d.get("weight", 0))) for u, _, d in in_edges]
    return profile


@st.cache_data(show_spinner=False)
def load_persona_prompt(name: str | None) -> str | None:
    """Resolve a display name (e.g. 'Sara Shackleton') to its persona prompt text.

    Lookup order:
      1. Slug = lowercase, punctuation stripped, whitespace → underscore.
         If `<slug>.txt` exists in `persona_prompts/`, return its text.
      2. Fallback: scan the directory for any file whose stem starts with
         the slug's first token (the given name) and contains the last token
         (family name). Handles middle-initial / nickname mismatches.
      3. Otherwise return None.
    """
    if not name:
        return None
    prompts_dir = PROCESSED / "persona_prompts"
    if not prompts_dir.is_dir():
        return None

    slug = re.sub(r"[^a-z0-9\s]", "", name.lower()).strip()
    slug = re.sub(r"\s+", "_", slug)
    direct = prompts_dir / f"{slug}.txt"
    if direct.is_file():
        return direct.read_text()

    tokens = slug.split("_")
    if len(tokens) >= 2:
        first, last = tokens[0], tokens[-1]
        for f in prompts_dir.iterdir():
            if not f.is_file() or f.suffix != ".txt":
                continue
            stem = f.stem
            if stem.startswith(first + "_") and stem.endswith("_" + last):
                return f.read_text()
    return None
