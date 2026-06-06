"""Stage A.4 — Topic canonicalization.

One LLM call. Reads raw topic strings from person_enrichment.csv and
entity_enrichment.csv, asks the model to cluster them into ~30 canonical
topics, and writes datasets/enron/processed/topic_canonicalization.json.

Neo4j is NOT touched here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from orggraph.config import OUTPUT_DIR
from orggraph.llm.client import LLMClient
from orggraph.llm.prompts import TOPIC_CANON_PROMPT

OUT_JSON = OUTPUT_DIR / "topic_canonicalization.json"


def collect_raw_topics(persons_csv: Path, entities_csv: Path) -> list[str]:
    """Union all raw topics from the two CSVs, normalized + deduplicated."""
    seen: set[str] = set()

    def _add(json_blob: str | float):
        if not isinstance(json_blob, str) or not json_blob.strip():
            return
        try:
            items = json.loads(json_blob)
        except json.JSONDecodeError:
            return
        for s in items:
            if not isinstance(s, str):
                continue
            seen.add(s.strip().lower())

    if persons_csv.exists():
        df = pd.read_csv(persons_csv)
        for col in ("topics_json", "expertise_json"):
            if col in df.columns:
                df[col].apply(_add)
    if entities_csv.exists():
        df = pd.read_csv(entities_csv)
        if "topics_json" in df.columns:
            df["topics_json"].apply(_add)

    return sorted(seen)


def invert_canonical_map(canonicals: dict[str, list[str]]) -> dict[str, str]:
    """Build raw -> canonical lookup, normalized."""
    out: dict[str, str] = {}
    for canon, variants in canonicals.items():
        for v in variants:
            out[v.strip().lower()] = canon
    return out


def run(model: str) -> None:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    persons_csv = OUTPUT_DIR / "person_enrichment.csv"
    entities_csv = OUTPUT_DIR / "entity_enrichment.csv"

    raw = collect_raw_topics(persons_csv, entities_csv)
    print(f"[1/3] Collected {len(raw)} unique raw topics")
    if not raw:
        print("No raw topics. Did you run enrich_persons / enrich_entities first?")
        return

    print("[2/3] Calling LLM for canonicalization...")
    client = LLMClient(base_url=base_url, api_key=api_key)
    prompt = TOPIC_CANON_PROMPT.format(raw_topics="\n".join(raw))
    result = client.json_chat(model=model, prompt=prompt, max_tokens=8192)

    if not result or "canonicals" not in result:
        print("LLM returned no canonicals. Saving identity map as fallback.")
        canonicals = {t.title(): [t] for t in raw}
    else:
        canonicals = result["canonicals"]

    inverted = invert_canonical_map(canonicals)
    # Make sure every raw topic has a canonical — assign Other for stragglers
    for t in raw:
        if t not in inverted:
            inverted[t] = "Other"
            canonicals.setdefault("Other", []).append(t)

    payload = {
        "canonicals": canonicals,
        "raw_to_canonical": inverted,
        "n_raw": len(raw),
        "n_canonical": len(canonicals),
        "model": model,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[3/3] Wrote {len(canonicals)} canonicals to {OUT_JSON}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--model", default=os.environ.get("INFERENCE_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit"),
    )
    args = parser.parse_args(argv)
    run(model=args.model)


if __name__ == "__main__":
    main()
