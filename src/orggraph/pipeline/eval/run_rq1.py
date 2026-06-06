"""End-to-end RQ1 pipeline: load data → build graph → extract hierarchy → evaluate.

Usage:
    python -m orggraph.pipeline.eval.run_rq1 [--email-limit 10000]
"""

import argparse
import json

import pandas as pd

from orggraph.config import OUTPUT_DIR
from orggraph.data.loader import load_emails, load_employees
from orggraph.data.identity import build_alias_map, resolve_sender
from orggraph.data.network import build_graph, compute_centrality, compute_communication_patterns
from orggraph.extraction.hierarchy import assign_hierarchy_tiers
from orggraph.extraction.communities import detect_communities
from orggraph.evaluation.rq1_metrics import dominance_f1, hierarchy_spearman

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_recipients(recipients, alias_map: dict[str, str]) -> list[str]:
    """Resolve a list of recipient email addresses to canonical names.

    Handles list, numpy array, and comma-separated string formats.
    Returns only resolved (known employee) names.
    """
    if recipients is None:
        return []

    # Convert to a plain Python list regardless of input type
    if isinstance(recipients, str):
        addresses = [r.strip() for r in recipients.split(",")]
    else:
        # Handles list, numpy.ndarray, or any iterable
        try:
            addresses = list(recipients)
        except TypeError:
            return []

    resolved = []
    for addr in addresses:
        addr_str = str(addr).strip()
        if not addr_str:
            continue
        name = resolve_sender(addr_str, alias_map)
        if name:
            resolved.append(name)
    return resolved


def run(email_limit: int | None = None):
    print("=" * 60)
    print("RQ1: Unsupervised Organizational Structure Extraction")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading employees...")
    employees = load_employees()
    print(f"  Loaded {len(employees)} employees")

    print("\n[2/6] Loading emails...")
    emails = load_emails(limit=email_limit)
    print(f"  Loaded {len(emails)} emails")

    # 2. Identity resolution
    print("\n[3/6] Resolving identities...")
    alias_map = build_alias_map()

    # Detect column names (HuggingFace dataset uses lowercase)
    sender_col = "from" if "from" in emails.columns else "From" if "From" in emails.columns else "sender"
    recipients_col = "to" if "to" in emails.columns else "To" if "To" in emails.columns else "recipients"

    # Resolve both senders and recipients to canonical names
    emails["sender_resolved"] = emails[sender_col].apply(
        lambda x: resolve_sender(str(x), alias_map)
    )
    emails["recipients_resolved"] = emails[recipients_col].apply(
        lambda x: resolve_recipients(x, alias_map)
    )

    resolved_senders = emails["sender_resolved"].notna().sum()
    resolved_recipients = emails["recipients_resolved"].apply(len).sum()
    print(f"  Resolved {resolved_senders}/{len(emails)} senders")
    print(f"  Resolved {int(resolved_recipients)} recipient addresses")

    # Filter to emails where sender is a known employee
    emails_internal = emails[emails["sender_resolved"].notna()].copy()
    # Further filter to emails with at least one resolved recipient
    emails_internal = emails_internal[
        emails_internal["recipients_resolved"].apply(len) > 0
    ].copy()
    print(f"  Internal emails (resolved sender + recipient): {len(emails_internal)}")

    # 3. Build communication graph using resolved names
    print("\n[4/6] Building communication graph...")
    G = build_graph(
        emails_internal,
        sender_col="sender_resolved",
        recipients_col="recipients_resolved",
    )
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_nodes() == 0:
        print("  ERROR: Empty graph. Check identity resolution.")
        return

    # 4. Extract hierarchy
    print("\n[5/6] Extracting hierarchy...")
    centrality = compute_centrality(G)

    # Compute communication pattern features from raw emails
    print("  Computing communication pattern features...")
    patterns = compute_communication_patterns(emails, G, alias_map)
    print(f"  Computed patterns for {len(patterns)} nodes")

    # Merge patterns into centrality DataFrame
    centrality = centrality.merge(patterns, on="node", how="left")
    # Fill any missing values with column medians
    for col in ["response_time_ratio", "initiation_ratio", "cc_frequency", "communication_breadth"]:
        if col in centrality.columns:
            median_val = centrality[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            centrality[col] = centrality[col].fillna(median_val)

    # Enhanced weights combining centrality + communication patterns
    weights = {
        "pagerank": 1.0,
        "betweenness": 1.0,
        "in_degree": 1.0,
        "response_time_ratio": 1.5,
        "cc_frequency": 1.5,
        "initiation_ratio": 0.5,
        "communication_breadth": 0.5,
    }
    # Features where lower raw value = higher in hierarchy
    invert = {"response_time_ratio", "initiation_ratio"}
    # Highly skewed features benefit from log transform before normalization
    log_transform = {"in_degree", "cc_frequency", "communication_breadth", "response_time_ratio"}

    hierarchy = assign_hierarchy_tiers(
        centrality, n_tiers=5, weights=weights, invert=invert, log_transform=log_transform,
    )
    print(f"  Assigned {len(hierarchy)} nodes to 5 tiers")
    print()
    print(hierarchy.head(15).to_string(index=False))

    # 5. Detect communities
    communities = detect_communities(G)
    n_communities = len(set(communities.values()))
    print(f"\n  Detected {n_communities} communities")

    # 6. Evaluate against ground truth
    print("\n[6/6] Evaluating against ground truth...")
    gt_path = OUTPUT_DIR / "dominance_pairs.csv"
    gt_emp_path = OUTPUT_DIR / "employees_ground_truth.csv"

    if not gt_path.exists():
        print("  Ground truth not found. Run: python -m orggraph.pipeline.ground_truth")
        return

    gt_pairs = pd.read_csv(gt_path)
    gt_employees = pd.read_csv(gt_emp_path)

    # Build extracted ranks dict (node -> composite_score)
    extracted_ranks = dict(zip(hierarchy["node"], hierarchy["composite_score"]))

    # Normalize GT names to canonical employee names (handle nickname/spelling mismatches).
    # Build a richer index than fuzzy-only: also map "{additional_name} {family_name}"
    # to the canonical name so a "go-by" GT name like "Greg Whalley" finds its
    # formal canonical "Lawrence Greg Whalley". Fuzzy matching alone misses this
    # because the strings are too dissimilar (ratio ~0.73 < 0.8 cutoff).
    employee_names = set(hierarchy["node"])
    from difflib import get_close_matches

    employees_df = load_employees()
    name_to_canonical: dict[str, str] = {}
    for _, e in employees_df.iterrows():
        canonical = e["name"]
        if canonical not in employee_names:
            continue  # only consider employees actually in the extracted graph
        additional = (e.get("additional_name") or "").strip().rstrip(".")
        family = (e.get("family_name") or "").strip()
        given = (e.get("given_name") or "").strip()
        # Register the bare "Go-by + family" form (e.g. "Greg Whalley")
        if additional and len(additional) > 1 and family:
            name_to_canonical[f"{additional} {family}"] = canonical
        # And the formal "Given + family" form, in case the GT uses that
        if given and family:
            name_to_canonical[f"{given} {family}"] = canonical

    gt_name_map: dict[str, str] = {}  # gt_name -> canonical_name
    all_gt_names = set(gt_pairs["superior"]) | set(gt_pairs["subordinate"]) | set(gt_employees["name"])
    for gt_name in all_gt_names:
        if gt_name in employee_names:
            gt_name_map[gt_name] = gt_name
        elif gt_name in name_to_canonical:
            gt_name_map[gt_name] = name_to_canonical[gt_name]
            print(f"  Name mapping (go-by): '{gt_name}' -> '{name_to_canonical[gt_name]}'")
        else:
            matches = get_close_matches(gt_name, employee_names, n=1, cutoff=0.8)
            if matches:
                gt_name_map[gt_name] = matches[0]
                print(f"  Name mapping (fuzzy): '{gt_name}' -> '{matches[0]}'")

    # Apply name normalization to GT data. Pairs whose names cannot be mapped
    # to any extracted-hierarchy node are KEPT (so dominance_f1 can count them
    # as `missing`). Dropping them silently inflates pair accuracy by hiding
    # coverage gaps — see the metric audit on 2026-05-07.
    n_pairs_before = len(gt_pairs)
    gt_pairs = gt_pairs.copy()
    gt_pairs["superior"] = gt_pairs["superior"].map(lambda n: gt_name_map.get(n, n))
    gt_pairs["subordinate"] = gt_pairs["subordinate"].map(lambda n: gt_name_map.get(n, n))
    print(f"  Ground-truth pairs preserved: {len(gt_pairs)}/{n_pairs_before}")

    gt_employees = gt_employees.copy()
    gt_employees["name"] = gt_employees["name"].map(lambda n: gt_name_map.get(n, n))

    # Pair-classification accuracy on dominance pairs
    f1_result = dominance_f1(gt_pairs, extracted_ranks)
    print("\n  Dominance Pair Classification:")
    print(f"    Pair accuracy: {f1_result['pair_accuracy']:.3f}")
    print(f"    Coverage:      {f1_result['coverage']:.3f}")
    print(
        f"    Breakdown:     correct={f1_result['correct']} "
        f"wrong_order={f1_result['wrong_order']} "
        f"missing={f1_result['missing']} / total={f1_result['total_pairs']}"
    )

    # Spearman correlation
    gt_levels = dict(zip(gt_employees["name"], gt_employees["level_numeric"]))
    rho = hierarchy_spearman(gt_levels, extracted_ranks)
    print(f"\n  Hierarchy Spearman rho: {rho:.3f}")

    # Community info
    print(f"\n  Communities detected: {n_communities}")

    # Save results
    results = {
        "f1": f1_result,
        "spearman_rho": rho,
        "n_communities": n_communities,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "email_limit": email_limit,
        "internal_emails": len(emails_internal),
    }
    with open(OUTPUT_DIR / "rq1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'rq1_results.json'}")

    hierarchy.to_csv(OUTPUT_DIR / "extracted_hierarchy.csv", index=False)
    print(f"  Hierarchy saved to {OUTPUT_DIR / 'extracted_hierarchy.csv'}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 8c: RQ1 ablation orchestrator (A0–A5)")
    parser.add_argument("--email-limit", type=int, default=None, help="Limit emails for dev")
    args = parser.parse_args(argv)
    run(email_limit=args.email_limit)


if __name__ == "__main__":
    main()
