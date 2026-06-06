"""Stage 2b — UMAP dim-reduction + HDBSCAN clustering + LLM cluster naming.

Pipeline:
    1. Load 768-dim embeddings from `embeddings_email`.
    2. UMAP reduce 768 → 5 dims (cosine metric). Visible tqdm progress.
    3. HDBSCAN on the 5-dim space, twice (project min_cluster_size + topic).
    4. LLM names each cluster.
    5. Write CSVs and update `embeddings_email.topics`.

Why UMAP first: HDBSCAN on raw 768-dim embeddings is broken — distance
concentration kills density estimation, runtime balloons (O(n²) brute-force
nearest-neighbour). BERTopic, the canonical text-clustering library, uses
UMAP→HDBSCAN by default. Documented benchmarks: 26 min → 5 sec on MNIST after
UMAP preprocessing.

Resumability:
    - HDBSCAN labels are saved to `cluster_labels.parquet` after the
      clustering phase. With `--rerun-naming`, the script loads the cache
      and skips UMAP+HDBSCAN, letting you re-name clusters with a different LLM.
    - With `--model-slug X`, output is written to `projects_X.csv` /
      `topics_X.csv` instead of the default paths so multiple LLMs can be
      compared side-by-side.
    - On rerun-naming, the `embeddings_email.topics` DB update is skipped
      (labels haven't changed).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import psycopg.types.json as pj
from pgvector.psycopg import register_vector
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

from orggraph.config import OUTPUT_DIR
from orggraph.llm.client import LLMClient

PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://{os.environ.get('POSTGRES_USER','orggraph')}:"
    f"{os.environ.get('POSTGRES_PASSWORD','orggraph2026')}@localhost:"
    f"{os.environ.get('POSTGRES_PORT','5432')}/"
    f"{os.environ.get('POSTGRES_DB','orggraph')}",
)

PROJECTS_CSV = OUTPUT_DIR / "projects.csv"
TOPICS_CSV = OUTPUT_DIR / "topics.csv"
CLUSTER_LABELS_PATH = OUTPUT_DIR / "cluster_labels.parquet"
UMAP_EMBEDDINGS_PATH = OUTPUT_DIR / "umap_embeddings.npy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    random_state: int = 42,
) -> np.ndarray:
    """UMAP reduction tuned for downstream HDBSCAN clustering.

    Defaults follow BERTopic's canonical config (n_components=5, n_neighbors=15,
    min_dist=0.0, metric='cosine'). UMAP's `verbose=True` prints progress.
    """
    import umap  # imported lazily so tests don't require umap-learn

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        verbose=True,
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Run HDBSCAN; returns cluster labels (-1 = noise).

    Expects already-dim-reduced embeddings (≤ 50 dims). Uses euclidean which
    is equivalent to cosine on L2-normalised vectors.
    """
    if len(embeddings) < min_cluster_size:
        return np.full(len(embeddings), -1, dtype=int)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(embeddings)


def representative_emails_for_cluster(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    k: int = 5,
) -> list[int]:
    """Return indices of the K emails closest to the centroid of cluster_id."""
    indices = np.where(labels == cluster_id)[0]
    if len(indices) == 0:
        return []
    cluster_emb = embeddings[indices]
    centroid = cluster_emb.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    row_norms = np.linalg.norm(cluster_emb, axis=1)
    sims = cluster_emb @ centroid / (row_norms * centroid_norm + 1e-9)
    top_k = min(k, len(indices))
    nearest_within = np.argsort(-sims)[:top_k]
    return indices[nearest_within].tolist()


def build_naming_prompt(sample_emails: list[dict], cluster_kind: str) -> str:
    """Build a prompt asking the LLM to name a cluster.

    cluster_kind must be one of: "project", "topic".
    """
    samples_text = "\n\n---\n\n".join(
        f"Subject: {e.get('subject') or ''}\nBody: {(e.get('body_truncated') or '')[:500]}"
        for e in sample_emails[:5]
    )
    return (
        f"Below are representative emails from a single {cluster_kind} cluster "
        f"discovered by HDBSCAN over an Enron email corpus. "
        f"Read them and output a JSON object with two keys:\n"
        f'- "name": a short human-readable {cluster_kind} name (2-4 words)\n'
        f'- "description": a one-sentence summary of what this {cluster_kind} is about\n\n'
        f"Output JSON only, no prose.\n\n"
        f"EMAILS:\n{samples_text}\n"
    )


def name_cluster(
    client: Any,
    model: str,
    sample_emails: list[dict],
    cluster_kind: str,
) -> tuple[str, str]:
    """Call the LLM, return (name, description) for the cluster.

    Falls back to placeholder strings on any error.
    """
    prompt = build_naming_prompt(sample_emails, cluster_kind)
    try:
        result = client.json_chat(model=model, prompt=prompt)
        if not result:
            return f"unnamed_{cluster_kind}", ""
        return (
            str(result.get("name", f"unnamed_{cluster_kind}")),
            str(result.get("description", "")),
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] LLM naming failed: {e}")
        return f"unnamed_{cluster_kind}", ""


def save_cluster_labels(
    email_ids: list[str],
    project_labels: np.ndarray,
    topic_labels: np.ndarray,
    path: Path = CLUSTER_LABELS_PATH,
) -> None:
    pd.DataFrame({
        "email_id": email_ids,
        "project_label": project_labels.astype(int),
        "topic_label": topic_labels.astype(int),
    }).to_parquet(path, index=False)


def load_cluster_labels(path: Path = CLUSTER_LABELS_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def model_slug(model: str) -> str:
    """Filesystem-safe slug for a model identifier."""
    last = model.split("/")[-1]
    return re.sub(r"[^\w]+", "-", last).strip("-")[:40] or "model"


def append_cluster_name(
    path: Path,
    *,
    cluster_kind: str,
    cluster_id: str,
    name: str,
    description: str,
    n_emails: int,
    representative_email_ids: list[str],
) -> None:
    """Append one cluster's name to a JSONL checkpoint, fsync'd per write."""
    rec = {
        "cluster_kind": cluster_kind,
        "cluster_id": cluster_id,
        "name": name,
        "description": description,
        "n_emails": n_emails,
        "representative_email_ids": representative_email_ids,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_named_clusters(path: Path) -> dict[str, dict]:
    """Load already-named clusters keyed by cluster_id (e.g. 'P003', 'T012')."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["cluster_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    min_project_size: int = 50,
    min_topic_size: int = 10,
    sample_limit: int | None = None,
    rerun_naming: bool = False,
    force_recluster: bool = False,
    output_slug: str | None = None,
    n_components: int = 5,
    workers: int = 8,
) -> None:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    model = os.environ.get("INFERENCE_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit")

    if output_slug:
        projects_csv = OUTPUT_DIR / f"projects_{output_slug}.csv"
        topics_csv = OUTPUT_DIR / f"topics_{output_slug}.csv"
        names_jsonl = OUTPUT_DIR / f"cluster_names_{output_slug}.jsonl"
    else:
        projects_csv = PROJECTS_CSV
        topics_csv = TOPICS_CSV
        names_jsonl = OUTPUT_DIR / "cluster_names.jsonl"
    print(f"[backend] {base_url}  model={model}")
    print(f"[outputs] {projects_csv.name}  {topics_csv.name}  ({names_jsonl.name})")

    print("[1/6] Loading embeddings from Postgres...")
    pg = psycopg.connect(PG_DSN)
    register_vector(pg)
    cur = pg.cursor()
    sql = (
        "SELECT email_id, sender_resolved, subject, body_truncated, embedding "
        "FROM embeddings_email WHERE embedding IS NOT NULL"
    )
    if sample_limit:
        sql += f" ORDER BY random() LIMIT {sample_limit}"
    cur.execute(sql)
    rows = cur.fetchall()
    print(f"      {len(rows):,} rows")

    df = pd.DataFrame(
        rows,
        columns=["email_id", "sender_resolved", "subject", "body_truncated", "embedding"],
    )
    raw_embeddings = np.stack([np.array(e, dtype=np.float32) for e in df["embedding"]])

    cached_labels = None if force_recluster else load_cluster_labels()
    use_cache = cached_labels is not None
    if rerun_naming and not use_cache:
        print(f"[warn] --rerun-naming set but no cache at {CLUSTER_LABELS_PATH}; running fresh")
    if force_recluster and CLUSTER_LABELS_PATH.exists():
        print("[info] --force-recluster: ignoring existing cache; will re-run UMAP+HDBSCAN")
    elif use_cache and not rerun_naming:
        print(f"[info] using cached UMAP+HDBSCAN labels from {CLUSTER_LABELS_PATH.name} "
              "(pass --force-recluster to redo)")

    if use_cache:
        print(f"[2-3/6] Loading cached labels from {CLUSTER_LABELS_PATH}...")
        merged = df.merge(cached_labels, on="email_id", how="inner")
        if len(merged) < len(df):
            print(f"      [warn] {len(df) - len(merged):,} rows in DB lack cached labels; "
                  "they will be excluded from this run")
        df = merged.reset_index(drop=True)
        raw_embeddings = np.stack([np.array(e, dtype=np.float32) for e in df["embedding"]])
        project_labels = df["project_label"].to_numpy(dtype=int)
        topic_labels = df["topic_label"].to_numpy(dtype=int)
        # Reduced embeddings used for representative-email centroids.
        reduced = (
            np.load(UMAP_EMBEDDINGS_PATH)
            if UMAP_EMBEDDINGS_PATH.exists()
            else raw_embeddings  # fall back to raw if the .npy was lost
        )
        if len(reduced) != len(df):
            # cache mismatch (e.g. embedding count changed); fall back to raw
            reduced = raw_embeddings
        n_projects = len({int(lab) for lab in project_labels if lab >= 0})
        n_topics = len({int(lab) for lab in topic_labels if lab >= 0})
        print(f"      cached: {n_projects} projects, {n_topics} topics")
    else:
        print(f"[2/6] UMAP {raw_embeddings.shape[1]} → {n_components} dims "
              f"(cosine, n_neighbors=15)...")
        reduced = reduce_dimensions(raw_embeddings, n_components=n_components)
        np.save(UMAP_EMBEDDINGS_PATH, reduced)
        print(f"      reduced shape: {reduced.shape}")

        print(f"[3/6] HDBSCAN — Projects (min_cluster_size={min_project_size}) "
              f"and Topics (min_cluster_size={min_topic_size})...")
        project_labels = cluster_embeddings(reduced, min_project_size)
        topic_labels = cluster_embeddings(reduced, min_topic_size)
        n_projects = len({int(lab) for lab in project_labels if lab >= 0})
        n_topics = len({int(lab) for lab in topic_labels if lab >= 0})
        print(f"      {n_projects} project clusters, {n_topics} topic clusters")

        save_cluster_labels(df["email_id"].tolist(), project_labels, topic_labels)
        print(f"      cached labels to {CLUSTER_LABELS_PATH}")

    print(f"[4/6] Naming clusters via LLM (workers={workers}, "
          f"per-cluster checkpoint to JSONL)...")
    client = LLMClient(base_url=base_url, api_key=api_key)
    already_named = load_named_clusters(names_jsonl)
    if already_named:
        print(f"      {len(already_named)} clusters already named; resuming")

    jsonl_lock = threading.Lock()

    def _name_one(kind: str, cluster_id: int, labels: np.ndarray) -> dict:
        cid_str = f"{'P' if kind == 'project' else 'T'}{cluster_id:03d}"
        if cid_str in already_named:
            return already_named[cid_str]
        rep_indices = representative_emails_for_cluster(reduced, labels, cluster_id, k=5)
        sample_emails = df.iloc[rep_indices].to_dict("records")
        rep_email_ids = df.iloc[rep_indices]["email_id"].tolist()
        name, desc = name_cluster(client, model, sample_emails, kind)
        rec = {
            "cluster_kind": kind,
            "cluster_id": cid_str,
            "name": name,
            "description": desc,
            "n_emails": int(np.sum(labels == cluster_id)),
            "representative_email_ids": rep_email_ids,
        }
        with jsonl_lock:
            append_cluster_name(names_jsonl, **rec)
            already_named[cid_str] = rec
        return rec

    def _name_in_parallel(kind: str, cluster_ids: list[int], labels: np.ndarray) -> list[dict]:
        results: dict[int, dict] = {}
        with tqdm(total=len(cluster_ids), desc=f"{kind}s") as pbar:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(_name_one, kind, cid, labels): cid
                    for cid in cluster_ids
                }
                for fut in as_completed(futs):
                    cid = futs[fut]
                    try:
                        results[cid] = fut.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"\n[error] {kind} cluster {cid}: {e}")
                    pbar.update(1)
        # Preserve cluster_id order for deterministic CSV output
        return [results[cid] for cid in cluster_ids if cid in results]

    project_cluster_ids = sorted({int(lab) for lab in project_labels if lab >= 0})
    project_records = _name_in_parallel("project", project_cluster_ids, project_labels)

    topic_cluster_ids = sorted({int(lab) for lab in topic_labels if lab >= 0})
    topic_records = _name_in_parallel("topic", topic_cluster_ids, topic_labels)

    project_rows = [{
        "project_id": r["cluster_id"],
        "name": r["name"],
        "description": r["description"],
        "n_emails": r["n_emails"],
        "representative_emails": json.dumps(r["representative_email_ids"]),
    } for r in project_records]
    topic_rows = [{
        "topic_id": r["cluster_id"],
        "name": r["name"],
        "description": r["description"],
        "n_emails": r["n_emails"],
        "representative_emails": json.dumps(r["representative_email_ids"]),
    } for r in topic_records]

    projects_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(project_rows).to_csv(projects_csv, index=False)
    pd.DataFrame(topic_rows).to_csv(topics_csv, index=False)
    print(f"[5/6] Wrote {projects_csv.name} ({len(project_rows)} rows) "
          f"and {topics_csv.name} ({len(topic_rows)} rows)")

    if rerun_naming:
        print("[6/6] Skipping DB topics update (--rerun-naming: labels unchanged).")
    else:
        n_to_update = int(np.sum(topic_labels >= 0))
        print(f"[6/6] Updating embeddings_email.topics on {n_to_update:,} rows...")
        n_updated = 0
        progress = tqdm(total=n_to_update, desc="db-update", unit="row")
        for i, label in enumerate(topic_labels):
            if label < 0:
                continue
            topic_id = f"T{int(label):03d}"
            cur.execute(
                "UPDATE embeddings_email SET topics = %s WHERE email_id = %s",
                (pj.Jsonb([topic_id]), df.iloc[i]["email_id"]),
            )
            n_updated += 1
            progress.update(1)
            if n_updated % 5000 == 0:
                pg.commit()
        pg.commit()
        progress.close()
        print(f"      Updated topics on {n_updated:,} emails.")

    pg.close()
    print(f"\nDone. {n_projects} projects, {n_topics} topics.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2b: UMAP + HDBSCAN clustering + LLM cluster naming."
    )
    parser.add_argument("--min-project-size", type=int, default=50)
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--n-components", type=int, default=5,
                        help="UMAP target dimensions (BERTopic default: 5).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit input embeddings (random sample).")
    parser.add_argument("--rerun-naming", action="store_true",
                        help="LLM-swap mode: reuse cached HDBSCAN labels, re-run naming "
                             "with a different model, skip DB topics update.")
    parser.add_argument("--force-recluster", action="store_true",
                        help="Ignore any existing cluster_labels.parquet/umap_embeddings.npy "
                             "and re-run UMAP+HDBSCAN from scratch.")
    parser.add_argument("--model-slug", type=str, default=None,
                        help="Append slug to output CSVs to compare LLMs side-by-side. "
                             "If omitted, auto-derived from INFERENCE_MODEL when --rerun-naming is set.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent LLM calls during cluster naming (default: 8).")
    args = parser.parse_args(argv)

    slug = args.model_slug
    if slug is None and args.rerun_naming:
        slug = model_slug(os.environ.get("INFERENCE_MODEL", "default"))

    run(
        min_project_size=args.min_project_size,
        min_topic_size=args.min_topic_size,
        sample_limit=args.limit,
        rerun_naming=args.rerun_naming,
        force_recluster=args.force_recluster,
        output_slug=slug,
        n_components=args.n_components,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
