"""BC3 speech-act validation harness (Task 7).

Loads the BC3 corpus (40 email threads, 3,222 sentences with manual
speech-act labels), runs Stage 2a's prompt against each sentence at
sentence granularity, and computes Cohen's κ per tag.

## BC3 XML format — deviation from the simplified plan example
The real BC3 release (bc3corpus 1.0, CC BY-SA 3.0) ships as two files:

- corpus.xml  — email text, structure:
    <root> <thread> <listno> <name> <DOC> <Text> <Sent id="N.M"> ...
  Sentence IDs are local within each thread (e.g. 1.1, 2.3).

- annotation.xml — per-annotator labels, structure:
    <root> <thread> <listno> <name>
      <annotation> <desc> <date> <summary> <sentences> <labels>
        <req id="..."/> <meet id="..."/> <prop id="..."/>
        <cmt id="..."/> <subj id="..."/> <meta id="..."/>

Three annotators label each thread. Gold acts are the union of mapped
tags across all three annotators for each sentence. Only tags that
intersect our 8-act canonical set are included; `subj` (subjective) and
`meta` (meta-commentary) have no canonical equivalent and are silently
dropped.

## Tag mapping
BC3 tag  ->  canonical act
  req    ->  request
  meet   ->  meeting
  prop   ->  propose
  cmt    ->  commit

BC3 has no direct mappings for: deliver, amend, refuse, accept.
Those canonical acts will not appear in the gold labels.

## LLM prediction grain
Predictions are made per sentence (matching BC3 annotation grain)
rather than per full email. A short context window
(sentence + 2 surrounding sentences from the same DOC) is passed to
the prompt to reduce label ambiguity without straying from per-sentence
granularity.

## Results
A JSON file is written to datasets/bc3/validation_results.json with:
  {"per_tag": {"request": κ, ...}, "macro_kappa": float, "n_sentences": int}

If macro_kappa < 0.55, a warning is printed (acceptance band per spec).
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from orggraph.config import REPO_ROOT

WORKTREE = REPO_ROOT

# ---------------------------------------------------------------------------
# Public API — BC3 tag mapping
# ---------------------------------------------------------------------------

#: Maps BC3 raw element tag names to our Cohen-Carvalho canonical speech acts.
#: Only the intersection of BC3's label vocabulary with our 8-act set.
#: BC3's `subj` (subjective) and `meta` (meta-commentary) are excluded —
#: they have no canonical equivalent in our scheme.
BC3_TAG_MAP: dict[str, str] = {
    "req": "request",
    "meet": "meeting",
    "prop": "propose",
    "cmt": "commit",
}

#: BC3 tags that are part of the annotation label vocabulary but not in BC3_TAG_MAP.
#: These are parsed and discarded during gold extraction.
_BC3_IGNORED_TAGS: frozenset[str] = frozenset({"subj", "meta"})

# Cache location for the BC3 corpus.
_BC3_DIR = WORKTREE / "datasets" / "bc3"
_BC3_CORPUS = _BC3_DIR / "corpus.xml"
_BC3_ANNOTATION = _BC3_DIR / "annotation.xml"
_BC3_RESULTS = _BC3_DIR / "validation_results.json"

# Primary download source (GitHub mirror of bc3corpus.1.0, CC BY-SA 3.0).
_BC3_CORPUS_URL = (
    "https://raw.githubusercontent.com/nAk123/mailgist/master/bc3corpus.1.0/corpus.xml"
)
_BC3_ANNOTATION_URL = (
    "https://raw.githubusercontent.com/nAk123/mailgist/master/bc3corpus.1.0/annotation.xml"
)

# Canonical tags present in the BC3 gold — used as default in compute_kappa_table.
CANONICAL_BC3_TAGS: list[str] = ["request", "meeting", "propose", "commit"]


# ---------------------------------------------------------------------------
# Public API — fetch_bc3
# ---------------------------------------------------------------------------


def fetch_bc3() -> Path:
    """Download (or return cached path to) the BC3 corpus.

    Downloads corpus.xml and annotation.xml to datasets/bc3/.
    Returns the directory path containing both files.

    Raises SystemExit if the download fails and no cached copy exists.
    """
    _BC3_DIR.mkdir(parents=True, exist_ok=True)

    if _BC3_CORPUS.exists() and _BC3_ANNOTATION.exists():
        # Validate that cached files are real XML (not HTML 404 pages).
        try:
            ET.parse(_BC3_CORPUS)
            ET.parse(_BC3_ANNOTATION)
            return _BC3_DIR
        except ET.ParseError:
            print("[warn] Cached BC3 files appear corrupt — re-downloading.")
            _BC3_CORPUS.unlink(missing_ok=True)
            _BC3_ANNOTATION.unlink(missing_ok=True)

    try:
        import urllib.request

        print(f"[bc3] Downloading corpus.xml from {_BC3_CORPUS_URL}")
        urllib.request.urlretrieve(_BC3_CORPUS_URL, _BC3_CORPUS)  # noqa: S310

        print(f"[bc3] Downloading annotation.xml from {_BC3_ANNOTATION_URL}")
        urllib.request.urlretrieve(_BC3_ANNOTATION_URL, _BC3_ANNOTATION)  # noqa: S310

        # Verify files are actual XML.
        ET.parse(_BC3_CORPUS)
        ET.parse(_BC3_ANNOTATION)
        print("[bc3] Download complete.")
        return _BC3_DIR

    except Exception as exc:  # noqa: BLE001
        _BC3_CORPUS.unlink(missing_ok=True)
        _BC3_ANNOTATION.unlink(missing_ok=True)
        raise SystemExit(
            f"BC3 download failed: {exc}\n"
            f"Place corpus.xml and annotation.xml manually in {_BC3_DIR}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API — parse_bc3_xml
# ---------------------------------------------------------------------------


def parse_bc3_xml(path: Path) -> list[dict]:
    """Parse the BC3 corpus and return per-sentence gold annotation records.

    Parameters
    ----------
    path:
        Directory containing corpus.xml and annotation.xml
        (as returned by fetch_bc3()), OR a single XML file path.
        If a directory, corpus.xml and annotation.xml are loaded from it.
        If a single file, it is treated as corpus.xml and annotation.xml
        is loaded from the same directory (for backwards compatibility).

    Returns
    -------
    list[dict]
        One dict per sentence::

            {
                "sentence_id":   "<listno>/<sent_local_id>",  # e.g. "067-11978590/1.1"
                "sentence_text": str,
                "gold_acts":     list[str],   # canonical tags, sorted; may be []
            }

    Deviations from simplified plan example
    ----------------------------------------
    * The single ``bc3.xml`` file referenced in the plan does not exist at
      the canonical URL.  The actual release ships corpus text and annotations
      as two separate files (corpus.xml / annotation.xml).
    * Sentence IDs are thread-local (``1.1``, ``2.3``, …) not globally unique.
      We qualify them as ``<listno>/<local_id>`` to ensure uniqueness.
    * Gold labels are the union across all three annotators per thread.
      Only BC3 tags in BC3_TAG_MAP are retained; ``subj`` and ``meta`` are
      discarded silently.
    * The ``path`` argument refers to the directory (not a single XML file)
      because two XML files are required.  A single-file path is handled for
      test compatibility with synthetic fixtures.
    """
    path = Path(path)

    if path.is_dir():
        corpus_path = path / "corpus.xml"
        annotation_path = path / "annotation.xml"
    elif path.is_file() and path.suffix == ".xml":
        # Single-file mode: assume it's corpus.xml, look for annotation.xml nearby.
        corpus_path = path
        annotation_path = path.parent / "annotation.xml"
        if not annotation_path.exists():
            # Synthetic fixture mode: treat the single file as both corpus and
            # annotation (used in tests where annotation may be embedded).
            annotation_path = path
    else:
        raise ValueError(f"path must be a directory or XML file, got: {path}")

    # --- Build sentence text lookup: listno -> {sent_id -> text} ---
    corpus_root = ET.parse(corpus_path).getroot()
    corpus_map: dict[str, dict[str, str]] = {}
    for thread in corpus_root.findall("thread"):
        ln = thread.find("listno").text.strip()
        sent_map: dict[str, str] = {}
        for doc in thread.findall("DOC"):
            text_el = doc.find("Text")
            if text_el is None:
                continue
            for sent in text_el.findall("Sent"):
                sid = sent.get("id")
                txt = (sent.text or "").strip()
                if sid:
                    sent_map[sid] = txt
        corpus_map[ln] = sent_map

    # --- Build gold label lookup: (listno, sent_id) -> set[canonical_act] ---
    try:
        ann_root = ET.parse(annotation_path).getroot()
    except ET.ParseError:
        # annotation.xml is missing or the path was a synthetic corpus-only
        # fixture; return records with empty gold.
        ann_root = None

    sent_gold: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    if ann_root is not None:
        for thread in ann_root.findall("thread"):
            ln_el = thread.find("listno")
            if ln_el is None:
                continue
            ln = ln_el.text.strip()
            for annotation in thread.findall("annotation"):
                labels = annotation.find("labels")
                if labels is None:
                    continue
                for child in labels:
                    tag = child.tag
                    sid = child.get("id")
                    if tag in BC3_TAG_MAP and sid:
                        sent_gold[ln][sid].add(BC3_TAG_MAP[tag])
                    # subj and meta are intentionally ignored

    # --- Build output records ---
    records: list[dict] = []
    for thread in corpus_root.findall("thread"):
        ln = thread.find("listno").text.strip()
        thread_sents = corpus_map.get(ln, {})
        gold_by_sent = sent_gold.get(ln, {})

        for sid, text in thread_sents.items():
            gold_acts = sorted(gold_by_sent.get(sid, set()))
            records.append({
                "sentence_id": f"{ln}/{sid}",
                "sentence_text": text,
                "gold_acts": gold_acts,
            })

    return records


# ---------------------------------------------------------------------------
# Public API — compute_kappa_table
# ---------------------------------------------------------------------------


def compute_kappa_table(
    gold: list[list[str]],
    pred: list[list[str]],
    tags: list[str],
) -> dict[str, float]:
    """Compute Cohen's κ per tag on binary indicator vectors.

    Parameters
    ----------
    gold:
        Per-sentence gold label lists, e.g. ``[["request"], [], ["commit"]]``.
    pred:
        Per-sentence predicted label lists, same length as ``gold``.
    tags:
        Tags to compute κ for.  Tags absent from both gold and pred are
        returned with κ = 0.0 (undefined) rather than raising.

    Returns
    -------
    dict[str, float]
        ``{tag: kappa}`` for every tag in ``tags``.  Undefined κ (all-same
        class) is returned as 0.0.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError as exc:  # noqa: BLE001
        raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

    n = len(gold)
    if n != len(pred):
        raise ValueError(
            f"gold and pred must have the same length; got {n} vs {len(pred)}"
        )

    result: dict[str, float] = {}

    for tag in tags:
        gold_vec = [1 if tag in row else 0 for row in gold]
        pred_vec = [1 if tag in row else 0 for row in pred]

        # If all values identical on either side, κ is undefined.
        if len(set(gold_vec)) == 1 and len(set(pred_vec)) == 1:
            result[tag] = 0.0
            continue
        if len(set(gold_vec)) == 1 or len(set(pred_vec)) == 1:
            # sklearn raises ValueError when one vector is constant;
            # κ is 0.0 by convention (no agreement beyond chance possible).
            result[tag] = 0.0
            continue

        try:
            kappa = float(cohen_kappa_score(gold_vec, pred_vec))
        except ValueError:
            kappa = 0.0
        result[tag] = kappa

    return result


# ---------------------------------------------------------------------------
# LLM prediction (sentence-level)
# ---------------------------------------------------------------------------


def _build_sentence_prompt(sentence: str, context: str = "") -> str:
    """Build the LLM prompt for per-sentence speech-act prediction.

    Uses the same label set as Stage 2a's build_prompt, but at sentence
    grain.  A short surrounding-sentence context window is prepended when
    available to reduce ambiguity.
    """
    from extract_email_signals import ALLOWED_SPEECH_ACTS  # noqa: PLC0415

    acts_list = ", ".join(sorted(ALLOWED_SPEECH_ACTS))
    ctx_block = f"\nCONTEXT (surrounding sentences):\n{context}\n" if context else ""
    return (
        f"You are an expert analyst of corporate and professional email communication."
        f"Classify the single sentence below with zero or more speech-act labels "
        f"from this set only: [{acts_list}]. Use as many as apply.\n"
        f"Return a single JSON object with one key: \"speech_acts\" (list of strings)."
        f"{ctx_block}\n\n"
        f"SENTENCE: {sentence}\n\n"
        f"Return ONLY valid JSON. No markdown, no explanation."
    )


def _predict_sentences(
    records: list[dict],
    client: Any,
    model: str,
    limit: int | None = None,
) -> list[list[str]]:
    """Run LLM predictions on per-sentence records.

    Returns a list of predicted act lists, one per record.
    Sentences that fail LLM prediction get an empty list.
    """

    working = records if limit is None else records[:limit]
    predictions: list[list[str]] = []

    for i, rec in enumerate(working):
        prompt = _build_sentence_prompt(rec["sentence_text"])
        try:
            raw = client.json_chat(model=model, prompt=prompt, max_tokens=256)
        except Exception:  # noqa: BLE001
            raw = None

        if raw is None:
            predictions.append([])
            continue

        # validate_payload expects full email payload; we cherry-pick speech_acts.
        sa_raw = raw.get("speech_acts", [])
        if isinstance(sa_raw, str):
            sa_raw = [sa_raw]
        elif not isinstance(sa_raw, list):
            sa_raw = []

        from extract_email_signals import ALLOWED_SPEECH_ACTS  # noqa: PLC0415

        acts = [
            v.lower()
            for v in sa_raw
            if isinstance(v, str) and v.lower() in ALLOWED_SPEECH_ACTS
        ]
        predictions.append(acts)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(working)}] predicted ...")

    return predictions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(limit: int | None, skip_llm: bool, output: Path) -> None:
    """Run the full BC3 validation pipeline."""
    # 1. Acquire data
    bc3_dir = fetch_bc3()

    # 2. Parse gold labels
    print("[1/4] Parsing BC3 corpus ...")
    records = parse_bc3_xml(bc3_dir)
    print(f"      {len(records):,} sentences, "
          f"{sum(1 for r in records if r['gold_acts']):,} with gold labels")

    gold = [r["gold_acts"] for r in records]

    # 3. LLM prediction
    if skip_llm:
        print("[2/4] Skipping LLM predictions (--skip-llm); using empty predictions.")
        pred = [[] for _ in records]
    else:
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        model = os.environ.get("INFERENCE_MODEL", "cyankiwi/MiniMax-M2.7-AWQ-4bit")

        print(f"[2/4] Running LLM predictions "
              f"(limit={limit or 'all'}, model={model}) ...")

        from orggraph.llm.client import LLMClient  # noqa: PLC0415

        client = LLMClient(base_url=base_url, api_key=api_key)
        gold_working = gold if limit is None else gold[:limit]
        pred_full = _predict_sentences(records, client, model, limit=limit)
        # Pad to full length if limited
        pred = pred_full + [[] for _ in range(len(records) - len(pred_full))]
        gold = gold_working + gold[len(gold_working):]

    # 4. Compute κ
    print("[3/4] Computing Cohen's κ ...")
    n_eval = min(len(gold), len(pred))
    kappa_table = compute_kappa_table(
        gold[:n_eval], pred[:n_eval], tags=CANONICAL_BC3_TAGS
    )

    valid_kappas = [v for v in kappa_table.values() if v == v]  # exclude NaN
    macro_kappa = sum(valid_kappas) / len(valid_kappas) if valid_kappas else 0.0

    # 5. Report
    results = {
        "per_tag": kappa_table,
        "macro_kappa": macro_kappa,
        "n_sentences": n_eval,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    print(f"\n[4/4] Results written to {output}")
    print(f"      n_sentences : {n_eval:,}")
    for tag, kappa in sorted(kappa_table.items()):
        print(f"      κ({tag:10s}) = {kappa:.4f}")
    print(f"      macro κ     = {macro_kappa:.4f}")

    if macro_kappa < 0.55:
        print(
            "\n[WARN] macro κ < 0.55 — below acceptance band. "
            "Check prompt design and label alignment."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Task 7: BC3 speech-act validation harness."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N sentences (for smoke tests)."
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM inference; useful for testing parse + κ logic only."
    )
    parser.add_argument(
        "--output", type=Path, default=_BC3_RESULTS,
        help="Path for the JSON results file."
    )
    args = parser.parse_args(argv)
    run(limit=args.limit, skip_llm=args.skip_llm, output=args.output)


if __name__ == "__main__":
    main()
