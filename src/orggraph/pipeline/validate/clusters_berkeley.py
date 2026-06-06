"""Berkeley genre cluster validation harness (Task 8).

Loads the neelblabla/enron_labeled_email-llama2-7b_finetuning HuggingFace
dataset (1,339 emails with 8 Berkeley genre labels), assigns each labeled
email to its nearest HDBSCAN cluster centroid (read from embeddings_email),
and computes ARI / NMI / purity vs the Berkeley genre labels.

## HF dataset schema
The dataset has a single column: ``prompts`` (no separate ``label`` column).
Each row follows the LLaMA-2 instruction template:

    <s>[INST] <<SYS>> ... <</SYS>>
    Mail: ...
    [/INST] Category: <label>(detail suffix)

The genre label is extracted by regex from the [/INST] response block.
The label ends at the first ``(`` character (the detail suffix follows).
The 8 Berkeley genre labels used in this dataset are:
    'Company Business, Strategy, etc.'
    'Purely Personal'
    'Personal but in professional context'
    'Logistic Arrangements'
    'Employment arrangements'
    'Document editing/checking'
    'Empty message (due to missing attachment)'
    'Empty message'

## Cluster centroid strategy
Cluster centroids are derived from embeddings_email rows where
``topics IS NOT NULL``.  The ``topics`` column stores a JSONB list
(e.g. ["energy trading", "contracts"]).  We use the *first element* of
the topics list as the cluster key.  Groups with fewer than 5 emails
are skipped (too small to yield a meaningful centroid).

## Two-mode operation
1. If embeddings_email.topics is populated for at least one row:
   compute centroids, embed labeled prompts, assign to nearest centroid,
   compute ARI / NMI / purity, write JSON results.
2. If no clustered rows exist:
   print "Stage 2b not yet run; clusters unavailable" and exit cleanly
   with status 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from orggraph.config import REPO_ROOT

WORKTREE = REPO_ROOT

# ---------------------------------------------------------------------------
# Environment / connection settings
# ---------------------------------------------------------------------------

EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "ollama")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embeddinggemma")

PG_DSN = os.environ.get(
    "PG_DSN",
    f"postgresql://{os.environ.get('POSTGRES_USER', 'orggraph')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'orggraph2026')}@localhost:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'orggraph')}",
)

HF_DATASET_NAME = "neelblabla/enron_labeled_email-llama2-7b_finetuning"
HF_SPLIT = "train"

_RESULTS_DIR = WORKTREE / "datasets" / "validation"
_RESULTS_FILE = _RESULTS_DIR / "berkeley_results.json"

# Regex to extract the label from the LLaMA-2 instruction template.
# The actual format is: [/INST] Category: <label>(detail suffix)
# The label ends at the first '(' character that begins the detail.
_LABEL_RE = re.compile(r"\[/INST\]\s*Category:\s*([^(\n\r<\[]+)", re.IGNORECASE)

# Minimum number of emails in a topic group to compute a centroid.
_MIN_CLUSTER_SIZE = 5


# ---------------------------------------------------------------------------
# Public API — load_berkeley_labels
# ---------------------------------------------------------------------------


def load_berkeley_labels(limit: int | None = None) -> list[dict]:
    """Load the Berkeley-labeled email dataset from HuggingFace.

    Uses the ``neelblabla/enron_labeled_email-llama2-7b_finetuning`` dataset.
    The dataset has one column: ``prompts``.  Each prompt follows the LLaMA-2
    instruction template; the category label is embedded after
    ``[/INST] Category:`` and ends before the ``(`` of the detail suffix.

    Parameters
    ----------
    limit:
        If given, return only the first *limit* rows (for smoke tests).

    Returns
    -------
    list[dict]
        One dict per row: ``{"prompt": str, "label": str}``.
        Rows where the label cannot be extracted are silently skipped.

    Notes
    -----
    If the dataset is gated or unavailable (network error, auth error), the
    function prints a clear message and returns an empty list so the caller
    can handle the missing-data path.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "datasets library is required: pip install datasets"
        ) from exc

    try:
        ds = load_dataset(HF_DATASET_NAME, split=HF_SPLIT)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not load HF dataset '{HF_DATASET_NAME}': {exc}")
        return []

    # The dataset may have column name 'prompts' (note plural) or 'prompt'.
    # Detect which one is present.
    col_name: str
    if "prompts" in ds.column_names:
        col_name = "prompts"
    elif "prompt" in ds.column_names:
        col_name = "prompt"
    else:
        print(
            f"[warn] Unexpected column names in dataset: {ds.column_names}. "
            "Expected 'prompts' or 'prompt'."
        )
        return []

    rows = ds[col_name]
    if limit is not None:
        rows = rows[:limit]

    records: list[dict] = []
    for prompt_text in rows:
        m = _LABEL_RE.search(prompt_text)
        if m is None:
            continue
        label = m.group(1).strip()
        if not label:
            continue
        records.append({"prompt": prompt_text, "label": label})

    return records


# ---------------------------------------------------------------------------
# Public API — embed_texts
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts via Ollama embeddinggemma at localhost:11434.

    Uses the OpenAI-compatible endpoint (same pattern as embed_emails.py).

    Parameters
    ----------
    texts:
        List of strings to embed.

    Returns
    -------
    np.ndarray
        Shape (N, 768) array of float32 embeddings.

    Raises
    ------
    RuntimeError
        If the Ollama server is unreachable or embedding fails.
    """
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("openai package is required: pip install openai") from exc

    client = OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY)

    batch_size = 16
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            all_embeddings.extend([d.embedding for d in resp.data])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Embedding failed for batch [{start}:{start + batch_size}]: {exc}"
            ) from exc

    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API — load_cluster_centroids
# ---------------------------------------------------------------------------


def _load_centroids_from_parquet(level: str) -> tuple[list[str], np.ndarray]:
    """Load centroids from cluster_labels.parquet + embeddings_email.

    `level` is 'project' or 'topic'; corresponds to project_label /
    topic_label columns in cluster_labels.parquet. Centroids are computed
    over the FULL 768-dim embeddings (not the 5-dim UMAP) because the
    Berkeley validation embeds incoming prompts with embeddinggemma at 768
    dim — the centroid space must match.
    """
    from orggraph.config import OUTPUT_DIR  # noqa: PLC0415
    labels_parquet = OUTPUT_DIR / "cluster_labels.parquet"
    if not labels_parquet.exists():
        print(f"[warn] {labels_parquet} not found; falling back to topics column.")
        return [], np.empty((0, 768), dtype=np.float32)

    label_col = "project_label" if level == "project" else "topic_label"
    id_prefix = "P" if level == "project" else "T"
    labels = pd.read_parquet(labels_parquet)
    labels = labels[labels[label_col] >= 0]  # drop noise
    if labels.empty:
        return [], np.empty((0, 768), dtype=np.float32)

    # Pull embeddings for those email_ids in one query
    try:
        import psycopg  # noqa: PLC0415
        from pgvector.psycopg import register_vector  # noqa: PLC0415
        conn = psycopg.connect(PG_DSN, autocommit=True)
        register_vector(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT email_id, embedding FROM embeddings_email "
            "WHERE email_id = ANY(%s) AND embedding IS NOT NULL",
            (labels["email_id"].tolist(),),
        )
        rows = {eid: np.asarray(emb, dtype=np.float32) for eid, emb in cur.fetchall()}
        conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not query embeddings: {exc}")
        return [], np.empty((0, 768), dtype=np.float32)

    # Group embeddings by cluster label
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for _, row in labels.iterrows():
        emb = rows.get(row["email_id"])
        if emb is None:
            continue
        cluster_id = f"{id_prefix}{int(row[label_col]):03d}"
        groups[cluster_id].append(emb)

    cluster_ids: list[str] = []
    centroids: list[np.ndarray] = []
    for cid, embs in sorted(groups.items()):
        if len(embs) < _MIN_CLUSTER_SIZE:
            continue
        centroid = np.mean(np.stack(embs, axis=0), axis=0)
        cluster_ids.append(cid)
        centroids.append(centroid)

    if not cluster_ids:
        return [], np.empty((0, 768), dtype=np.float32)
    return cluster_ids, np.stack(centroids, axis=0).astype(np.float32)


def load_cluster_centroids(level: str = "topic") -> tuple[list[str], np.ndarray]:
    """Read cluster centroids.

    `level='topic'` (default) reads from `embeddings_email.topics` JSONB
    column populated by Stage 2b — backward-compatible behaviour.

    `level='project'` reads project labels from cluster_labels.parquet
    instead. Useful for thesis comparison: 450 projects map closer to
    Berkeley's 8 coarse genres than 2,827 topics, giving more interpretable
    ARI/NMI numbers.

    Notes
    -----
    Database connection errors are caught; the function prints a warning and
    returns the empty-centroids tuple so the caller can handle the path
    gracefully.
    """
    if level == "project":
        return _load_centroids_from_parquet("project")
    if level != "topic":
        raise ValueError(f"level must be 'project' or 'topic', got {level!r}")
    try:
        import psycopg  # noqa: PLC0415
        from pgvector.psycopg import register_vector  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "psycopg and pgvector are required: pip install psycopg pgvector"
        ) from exc

    try:
        conn = psycopg.connect(PG_DSN, autocommit=True)
        register_vector(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT topics, embedding
            FROM embeddings_email
            WHERE topics IS NOT NULL AND embedding IS NOT NULL
            """
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not query embeddings_email: {exc}")
        return [], np.empty((0, 768), dtype=np.float32)

    if not rows:
        return [], np.empty((0, 768), dtype=np.float32)

    # Group embeddings by the first element of the topics list.
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for topics_val, embedding in rows:
        if topics_val is None:
            continue
        # topics_val may be a list (returned from JSONB) or a raw JSON string.
        if isinstance(topics_val, str):
            try:
                topics_val = json.loads(topics_val)
            except json.JSONDecodeError:
                continue
        if not isinstance(topics_val, list) or len(topics_val) == 0:
            continue
        primary_topic = str(topics_val[0]).strip()
        if not primary_topic:
            continue
        emb_arr = np.asarray(embedding, dtype=np.float32)
        groups[primary_topic].append(emb_arr)

    # Build centroids for groups large enough.
    cluster_ids: list[str] = []
    centroids: list[np.ndarray] = []
    for topic, embs in sorted(groups.items()):
        if len(embs) < _MIN_CLUSTER_SIZE:
            continue
        centroid = np.mean(np.stack(embs, axis=0), axis=0)
        cluster_ids.append(topic)
        centroids.append(centroid)

    if not cluster_ids:
        return [], np.empty((0, 768), dtype=np.float32)

    return cluster_ids, np.stack(centroids, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API — assign_to_nearest_cluster
# ---------------------------------------------------------------------------


def assign_to_nearest_cluster(emb: np.ndarray, centroids: np.ndarray) -> int:
    """Return the index of the nearest centroid by cosine similarity.

    Parameters
    ----------
    emb:
        1-D embedding vector of shape (D,).
    centroids:
        2-D array of shape (K, D).

    Returns
    -------
    int
        Index into ``centroids`` of the centroid with highest cosine similarity
        to ``emb``.
    """
    emb = np.asarray(emb, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)

    # Normalise both sides to unit length for cosine similarity via dot product.
    emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
    cent_norms = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    similarities = cent_norms @ emb_norm
    return int(np.argmax(similarities))


# ---------------------------------------------------------------------------
# Public API — compute_metrics
# ---------------------------------------------------------------------------


def compute_metrics(gold: list[str], pred: list[int]) -> dict[str, float]:
    """Compute ARI, NMI, and purity of predicted cluster assignments.

    Parameters
    ----------
    gold:
        Ground-truth string labels (Berkeley genre strings), one per sample.
    pred:
        Integer cluster assignments, one per sample.

    Returns
    -------
    dict[str, float]
        Keys: ``"ari"``, ``"nmi"``, ``"purity"``.

    Notes
    -----
    Purity: for each predicted cluster, identify the most common gold label;
    sum those counts and divide by the total number of samples.
    """
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

    n = len(gold)
    if n == 0:
        return {"ari": 0.0, "nmi": 0.0, "purity": 0.0}

    ari = float(adjusted_rand_score(gold, pred))
    nmi = float(normalized_mutual_info_score(gold, pred, average_method="arithmetic"))

    # Purity: group by predicted cluster, find dominant gold label, sum.
    cluster_to_golds: dict[int, list[str]] = defaultdict(list)
    for g, p in zip(gold, pred):
        cluster_to_golds[p].append(g)

    correct = sum(Counter(golds).most_common(1)[0][1] for golds in cluster_to_golds.values())
    purity = correct / n

    return {"ari": ari, "nmi": nmi, "purity": purity}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(limit: int | None, output: Path, cluster_level: str = "topic") -> None:
    """Run the full Berkeley genre cluster validation pipeline."""

    # 1. Load Berkeley labels
    print(f"[1/5] Loading Berkeley genre labels (limit={limit or 'all'}) ...")
    records = load_berkeley_labels(limit=limit)
    if not records:
        print(
            "[warn] No Berkeley labels loaded (dataset unavailable or empty). "
            "Exiting."
        )
        sys.exit(0)
    print(f"      {len(records):,} labeled emails loaded")

    # 2. Load cluster centroids at the chosen granularity
    print(f"[2/5] Loading {cluster_level} cluster centroids ...")
    cluster_ids, centroids = load_cluster_centroids(level=cluster_level)
    if len(cluster_ids) == 0:
        print(
            f"Stage 2b not yet run for {cluster_level}-level clusters; "
            "clusters unavailable."
        )
        sys.exit(0)
    print(f"      {len(cluster_ids)} {cluster_level} clusters found: {cluster_ids[:5]}...")

    # 3. Embed the labeled prompts
    print(f"[3/5] Embedding {len(records)} labeled prompts via {EMBED_MODEL} ...")
    texts = [r["prompt"] for r in records]
    try:
        embeddings = embed_texts(texts)
    except RuntimeError as exc:
        print(f"[error] Embedding failed: {exc}")
        sys.exit(1)
    print(f"      Embedding shape: {embeddings.shape}")

    # 4. Assign each embedding to nearest cluster centroid
    print("[4/5] Assigning to nearest cluster ...")
    gold_labels = [r["label"] for r in records]
    pred_clusters = [
        assign_to_nearest_cluster(emb, centroids) for emb in embeddings
    ]

    # 5. Compute metrics
    print("[5/5] Computing ARI / NMI / purity ...")
    metrics = compute_metrics(gold_labels, pred_clusters)
    print(f"      ARI    = {metrics['ari']:.4f}")
    print(f"      NMI    = {metrics['nmi']:.4f}")
    print(f"      Purity = {metrics['purity']:.4f}")

    if metrics["ari"] < 0.30:
        print(
            "\n[WARN] ARI < 0.30 — weak cluster-genre alignment. "
            "Consider re-running Stage 2b or reviewing the topic extraction prompt."
        )

    # Write results
    results = {
        "cluster_level": cluster_level,
        "n_labeled": len(records),
        "n_clusters": len(cluster_ids),
        "cluster_ids": cluster_ids,
        **metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Task 8: Berkeley genre cluster validation harness."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Load only the first N Berkeley labels (for smoke tests).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_RESULTS_FILE,
        help="Path for the JSON results file.",
    )
    parser.add_argument(
        "--cluster-level",
        choices=["topic", "project"],
        default="topic",
        help="Granularity of cluster centroids: 'topic' (~2,827 fine-grained) "
             "or 'project' (~450 coarse). Default: topic.",
    )
    args = parser.parse_args(argv)
    # When using projects, write to a separate output file by default
    output = args.output
    if args.cluster_level == "project" and output == _RESULTS_FILE:
        output = output.with_name("berkeley_results_projects.json")
    run(limit=args.limit, output=output, cluster_level=args.cluster_level)


if __name__ == "__main__":
    main()
