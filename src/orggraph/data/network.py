"""Build directed communication graph from email data."""

from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from orggraph.data.identity import resolve_sender


def build_graph(
    emails: pd.DataFrame,
    sender_col: str = "sender",
    recipients_col: str = "recipients",
) -> nx.DiGraph:
    """Build a directed weighted graph from email communication data.

    Nodes are person identifiers. Edge weight is the number of emails sent
    from sender to recipient.
    """
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)

    for _, row in emails.iterrows():
        sender = str(row[sender_col]).strip()
        if not sender:
            continue

        raw_recipients = row[recipients_col]
        if isinstance(raw_recipients, str):
            recipient_list = [r.strip() for r in raw_recipients.split(",") if r.strip()]
        elif isinstance(raw_recipients, list):
            recipient_list = [str(r).strip() for r in raw_recipients if str(r).strip()]
        else:
            continue

        for recipient in recipient_list:
            if recipient != sender:
                edge_counts[(sender, recipient)] += 1

    G = nx.DiGraph()
    for (s, r), count in edge_counts.items():
        G.add_edge(s, r, weight=count)

    return G


def compute_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """Compute centrality metrics for all nodes in the graph.

    Returns DataFrame with columns: node, in_degree, out_degree, total_degree,
    betweenness, pagerank
    """
    pagerank = nx.pagerank(G, weight="weight")
    betweenness = nx.betweenness_centrality(G, weight="weight")
    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))

    records = []
    for node in G.nodes():
        records.append({
            "node": node,
            "in_degree": in_degree.get(node, 0),
            "out_degree": out_degree.get(node, 0),
            "total_degree": in_degree.get(node, 0) + out_degree.get(node, 0),
            "betweenness": betweenness.get(node, 0.0),
            "pagerank": pagerank.get(node, 0.0),
        })

    return pd.DataFrame(records).sort_values("pagerank", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Communication pattern features (RQ1 enhancement)
# ---------------------------------------------------------------------------

_RE_PREFIX = re.compile(r"^(re|fwd?|fw)\s*:\s*", re.IGNORECASE)


def _normalize_subject(subj: str) -> str:
    """Strip Re:/Fwd: prefixes and collapse whitespace."""
    s = str(subj) if subj is not None else ""
    while _RE_PREFIX.search(s):
        s = _RE_PREFIX.sub("", s, count=1)
    return " ".join(s.split()).lower().strip()


def _resolve_addresses(
    addrs,
    alias_map: dict[str, str],
) -> list[str]:
    """Resolve an array/list/string of email addresses to canonical names."""
    if addrs is None:
        return []
    if isinstance(addrs, str):
        items = [a.strip() for a in addrs.split(",")]
    else:
        try:
            items = list(addrs)
        except TypeError:
            return []
    resolved = []
    for a in items:
        a_str = str(a).strip()
        if not a_str:
            continue
        name = resolve_sender(a_str, alias_map)
        if name:
            resolved.append(name)
    return resolved


def compute_communication_patterns(
    emails: pd.DataFrame,
    G: nx.DiGraph,
    alias_map: dict[str, str],
) -> pd.DataFrame:
    """Compute per-node communication pattern features from raw email data.

    Returns a DataFrame with columns:
        node, response_time_ratio, cc_frequency, initiation_ratio,
        communication_breadth
    """
    nodes = set(G.nodes())

    # ------------------------------------------------------------------
    # Prepare a working copy with parsed dates and resolved identities
    # Column names use "r_" prefix (for "resolved") to avoid clashes
    # with original columns and to stay compatible with itertuples().
    # ------------------------------------------------------------------
    df = emails.copy()

    # Parse dates
    df["r_dt"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # Resolve sender
    sender_col = "from" if "from" in df.columns else "sender"
    df["r_sender"] = df[sender_col].apply(
        lambda x: resolve_sender(str(x), alias_map)
    )

    # Resolve To recipients
    to_col = "to" if "to" in df.columns else "recipients"
    df["r_to"] = df[to_col].apply(lambda x: _resolve_addresses(x, alias_map))

    # Resolve CC recipients
    if "cc" in df.columns:
        df["r_cc"] = df["cc"].apply(lambda x: _resolve_addresses(x, alias_map))
    else:
        df["r_cc"] = [[] for _ in range(len(df))]

    # Normalize subjects for thread detection
    subject_col = "subject" if "subject" in df.columns else None
    if subject_col:
        df["r_thread"] = df[subject_col].apply(_normalize_subject)
    else:
        df["r_thread"] = ""

    # Keep only rows where sender is a known node and date is valid
    df = df[df["r_sender"].isin(nodes) & df["r_dt"].notna()].copy()

    # ------------------------------------------------------------------
    # 1. Response time asymmetry
    # ------------------------------------------------------------------
    # For each thread, sort by date and look at consecutive messages
    # between pairs. A->B then B->A gives response time for B replying to A.
    pair_response_times: dict[tuple[str, str], list[float]] = defaultdict(list)

    if subject_col:
        for thread_key, thread_df in df.groupby("r_thread"):
            if len(thread_df) < 2 or thread_key == "":
                continue
            thread_sorted = thread_df.sort_values("r_dt")
            rows = list(thread_sorted.itertuples(index=False))
            for i in range(1, len(rows)):
                prev_sender = rows[i - 1].r_sender
                curr_sender = rows[i].r_sender
                if prev_sender and curr_sender and prev_sender != curr_sender:
                    delta = (rows[i].r_dt - rows[i - 1].r_dt).total_seconds()
                    if 0 < delta < 7 * 24 * 3600:  # cap at 7 days
                        # curr_sender is responding to prev_sender
                        pair_response_times[(curr_sender, prev_sender)].append(delta)

    # Per-node: median time others respond to me vs median time I respond to others
    response_to_me: dict[str, list[float]] = defaultdict(list)
    my_response: dict[str, list[float]] = defaultdict(list)
    for (responder, original_sender), times in pair_response_times.items():
        # responder responded to original_sender
        response_to_me[original_sender].extend(times)
        my_response[responder].extend(times)

    # ------------------------------------------------------------------
    # 2. Initiation ratio: fraction of threads where node is initiator
    # ------------------------------------------------------------------
    thread_initiators: dict[str, str] = {}  # thread_key -> sender of first msg
    thread_participants: dict[str, set[str]] = defaultdict(set)

    if subject_col:
        for thread_key, thread_df in df.groupby("r_thread"):
            if thread_key == "":
                continue
            first_row = thread_df.sort_values("r_dt").iloc[0]
            initiator = first_row["r_sender"]
            if initiator:
                thread_initiators[thread_key] = initiator
            for _, row in thread_df.iterrows():
                s = row["r_sender"]
                if s:
                    thread_participants[thread_key].add(s)
                for r in row["r_to"]:
                    if r in nodes:
                        thread_participants[thread_key].add(r)

    node_initiated: dict[str, int] = defaultdict(int)
    node_participated: dict[str, int] = defaultdict(int)
    for tkey, initiator in thread_initiators.items():
        for participant in thread_participants.get(tkey, set()):
            node_participated[participant] += 1
        node_initiated[initiator] += 1

    # ------------------------------------------------------------------
    # 3. CC frequency: how often is each node in CC (not To)?
    # ------------------------------------------------------------------
    cc_counts: dict[str, int] = defaultdict(int)
    for cc_list in df["r_cc"]:
        for name in cc_list:
            if name in nodes:
                cc_counts[name] += 1

    # ------------------------------------------------------------------
    # 4. Communication breadth: unique recipients per node
    # ------------------------------------------------------------------
    unique_recipients: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        s = row["r_sender"]
        if not s:
            continue
        for r in row["r_to"]:
            if r in nodes and r != s:
                unique_recipients[s].add(r)

    # ------------------------------------------------------------------
    # Assemble the feature DataFrame
    # ------------------------------------------------------------------
    records = []
    for node in nodes:
        # Response time ratio
        med_others_respond = float(np.median(response_to_me[node])) if response_to_me[node] else np.nan
        med_i_respond = float(np.median(my_response[node])) if my_response[node] else np.nan
        if not np.isnan(med_others_respond) and not np.isnan(med_i_respond) and med_i_respond > 0:
            rt_ratio = med_others_respond / med_i_respond
        else:
            rt_ratio = np.nan

        # Initiation ratio
        participated = node_participated.get(node, 0)
        initiated = node_initiated.get(node, 0)
        init_ratio = initiated / participated if participated > 0 else np.nan

        # CC frequency
        cc_freq = cc_counts.get(node, 0)

        # Communication breadth
        breadth = len(unique_recipients.get(node, set()))

        records.append({
            "node": node,
            "response_time_ratio": rt_ratio,
            "initiation_ratio": init_ratio,
            "cc_frequency": cc_freq,
            "communication_breadth": breadth,
        })

    result = pd.DataFrame(records)

    # Fill NaN with median for each feature (graceful fallback)
    for col in ["response_time_ratio", "initiation_ratio"]:
        median_val = result[col].median()
        if pd.isna(median_val):
            median_val = 1.0  # neutral fallback
        result[col] = result[col].fillna(median_val)

    return result
