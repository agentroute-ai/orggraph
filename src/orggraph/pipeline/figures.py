"""Generate publication-quality figures for the thesis (RQ1 results).

Produces three figures in ``thesis/figures/results/``:

1. ``hierarchy_comparison.{pdf,png}`` - Top-20 extracted employees vs.
   ground truth level.
2. ``metric_comparison.{pdf,png}`` - Baseline vs. pattern-augmented
   metrics (F1, Recall, Spearman, correct pairs).
3. ``network_communities.{pdf,png}`` - Spring-layout network diagram
   coloured by Louvain community.

Style: USI thesis compliant -- serif text, black/white/grayscale with a
single muted accent (``#2F5F8F``), 300 DPI, PDF + PNG output.

Usage:
    .venv/bin/python scripts/generate_figures.py

The script reuses cached outputs in ``datasets/enron/processed/`` where
possible. It rebuilds (and caches) the communication graph once as
``communication_graph.gpickle`` for reproducibility.
"""

from __future__ import annotations

import pickle
from difflib import get_close_matches
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from orggraph.config import OUTPUT_DIR, REPO_ROOT

# ---------------------------------------------------------------------------
# Paths and global style
# ---------------------------------------------------------------------------

FIG_DIR = REPO_ROOT / "thesis" / "figures" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_PICKLE = OUTPUT_DIR / "communication_graph.gpickle"

ACCENT = "#2F5F8F"  # muted neutral blue
GRAY_DARK = "#333333"
GRAY_MID = "#777777"
GRAY_LIGHT = "#BBBBBB"

# Applied globally so every figure stays consistent.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Charter", "PT Serif", "Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,  # TrueType, safe for LaTeX
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_gt_name_map(
    gt_names: set[str],
    extracted_nodes: set[str],
) -> dict[str, str]:
    """Replicate the difflib GT-name normalisation used in run_rq1."""
    mapping: dict[str, str] = {}
    for gt_name in gt_names:
        if gt_name in extracted_nodes:
            mapping[gt_name] = gt_name
        else:
            matches = get_close_matches(gt_name, list(extracted_nodes), n=1, cutoff=0.8)
            if matches:
                mapping[gt_name] = matches[0]
    return mapping


def _load_or_build_graph() -> nx.DiGraph:
    """Load the communication graph from cache, or rebuild + cache it."""
    if GRAPH_PICKLE.exists():
        print(f"  Loading cached graph: {GRAPH_PICKLE}")
        with open(GRAPH_PICKLE, "rb") as f:
            return pickle.load(f)

    print("  No cached graph; rebuilding from raw emails...")
    # Imported lazily so that users without HuggingFace set up can still
    # run the first two figures if a pickle already exists.
    from orggraph.data.identity import build_alias_map, resolve_sender
    from orggraph.data.loader import load_emails
    from orggraph.data.network import build_graph

    emails = load_emails()
    print(f"    {len(emails)} emails loaded")

    alias_map = build_alias_map()

    sender_col = "from" if "from" in emails.columns else "From" if "From" in emails.columns else "sender"
    recipients_col = "to" if "to" in emails.columns else "To" if "To" in emails.columns else "recipients"

    def _resolve_recipients(val):
        if val is None:
            return []
        if isinstance(val, str):
            items = [r.strip() for r in val.split(",")]
        else:
            try:
                items = list(val)
            except TypeError:
                return []
        out = []
        for a in items:
            a_str = str(a).strip()
            if not a_str:
                continue
            name = resolve_sender(a_str, alias_map)
            if name:
                out.append(name)
        return out

    emails["sender_resolved"] = emails[sender_col].apply(
        lambda x: resolve_sender(str(x), alias_map)
    )
    emails["recipients_resolved"] = emails[recipients_col].apply(_resolve_recipients)

    internal = emails[
        emails["sender_resolved"].notna()
        & (emails["recipients_resolved"].apply(len) > 0)
    ].copy()

    G = build_graph(
        internal,
        sender_col="sender_resolved",
        recipients_col="recipients_resolved",
    )

    with open(GRAPH_PICKLE, "wb") as f:
        pickle.dump(G, f)
    print(f"    Graph cached to {GRAPH_PICKLE}")
    return G


# ---------------------------------------------------------------------------
# Figure 1: Hierarchy comparison
# ---------------------------------------------------------------------------

def figure_hierarchy_comparison() -> Path:
    """Top-20 extracted employees with ground truth level annotation.

    Horizontal bar chart. Bar length = composite score. Bar shading
    encodes ground-truth level (higher level = darker), and employees
    absent from ground truth are drawn hollow.
    """
    extracted = pd.read_csv(OUTPUT_DIR / "extracted_hierarchy.csv")
    gt = pd.read_csv(OUTPUT_DIR / "employees_ground_truth.csv")

    extracted_nodes = set(extracted["node"])
    gt_name_map = _build_gt_name_map(set(gt["name"]), extracted_nodes)

    gt = gt.copy()
    gt["canonical_name"] = gt["name"].map(gt_name_map)
    gt_by_canonical = gt.dropna(subset=["canonical_name"]).set_index("canonical_name")

    # Prefer extracted employees that ARE in GT. If fewer than 20 such
    # employees in the whole extracted table we top up with extras so
    # that the figure always shows 20 rows.
    matched_mask = extracted["node"].isin(gt_by_canonical.index)
    top_matched = extracted[matched_mask].head(20).copy()
    if len(top_matched) < 20:
        padding = extracted[~matched_mask].head(20 - len(top_matched))
        top = pd.concat([top_matched, padding], ignore_index=True)
    else:
        top = top_matched
    top = top.head(20).reset_index(drop=True)

    # Build the level annotation
    levels_numeric: list[float | None] = []
    levels_label: list[str] = []
    for _, row in top.iterrows():
        node = row["node"]
        if node in gt_by_canonical.index:
            levels_numeric.append(float(gt_by_canonical.loc[node, "level_numeric"]))
            levels_label.append(str(gt_by_canonical.loc[node, "level"]))
        else:
            levels_numeric.append(None)
            levels_label.append("not in GT")

    # Colour scale from white to accent; hollow for not-in-GT
    max_level = 6.0  # C-Suite
    bar_colors = []
    edge_colors = []
    for lvl in levels_numeric:
        if lvl is None:
            bar_colors.append("white")
            edge_colors.append(GRAY_MID)
        else:
            # Interpolate between light-gray and accent
            t = lvl / max_level
            base = np.array(mpl.colors.to_rgb(ACCENT))
            col = (1 - t) * np.array([0.95, 0.95, 0.95]) + t * base
            bar_colors.append(tuple(col))
            edge_colors.append(GRAY_DARK)

    fig, ax = plt.subplots(figsize=(7.0, 6.2))

    y_pos = np.arange(len(top))[::-1]  # top=rank 1 at the top
    ax.barh(
        y_pos,
        top["composite_score"],
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=0.7,
        height=0.75,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["node"])
    ax.invert_yaxis()  # already reversed via y_pos; keep labels correct
    # Because we reversed y_pos, top element is at top; inverting axis
    # would flip it. Re-flip: use ascending y_pos instead.
    y_pos = np.arange(len(top))
    ax.clear()

    ax.barh(
        y_pos,
        top["composite_score"],
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=0.7,
        height=0.75,
    )

    # Level label on top of each bar
    for i, (score, lvl_text) in enumerate(zip(top["composite_score"], levels_label)):
        ax.text(
            score + 0.05,
            i,
            lvl_text,
            va="center",
            ha="left",
            fontsize=8,
            color=GRAY_DARK if lvl_text != "not in GT" else GRAY_MID,
            style="italic" if lvl_text == "not in GT" else "normal",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["node"], fontsize=9)
    ax.invert_yaxis()  # rank 1 on top
    ax.set_xlabel("Composite score (centrality + communication patterns)")
    ax.set_title(
        "Top-20 extracted employees and ground-truth level",
        loc="left",
        pad=8,
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(0, max(top["composite_score"]) * 1.25)

    # Small legend: colour scale + not-in-GT swatch
    legend_elems = [
        mpl.patches.Patch(
            facecolor=ACCENT, edgecolor=GRAY_DARK,
            linewidth=0.7, label="C-Suite (level 6)",
        ),
        mpl.patches.Patch(
            facecolor=np.array([0.5, 0.5, 0.5]) * 0 + (
                0.5 * np.array([0.95, 0.95, 0.95])
                + 0.5 * np.array(mpl.colors.to_rgb(ACCENT))
            ),
            edgecolor=GRAY_DARK, linewidth=0.7, label="VP / SVP (level 3-4)",
        ),
        mpl.patches.Patch(
            facecolor=(0.95, 0.95, 0.95), edgecolor=GRAY_DARK,
            linewidth=0.7, label="Manager / Employee (level 0-1)",
        ),
        mpl.patches.Patch(
            facecolor="white", edgecolor=GRAY_MID,
            linewidth=0.7, label="Not in ground truth",
        ),
    ]
    ax.legend(
        handles=legend_elems,
        loc="lower right",
        frameon=False,
        fontsize=8,
        title="Ground-truth level",
        title_fontsize=9,
    )

    out = FIG_DIR / "hierarchy_comparison"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Wrote {out.with_suffix('.pdf').name} / .png")
    return out.with_suffix(".pdf")


# ---------------------------------------------------------------------------
# Figure 2: Baseline vs With-Patterns
# ---------------------------------------------------------------------------

def figure_metric_comparison() -> Path:
    """Grouped bar chart of baseline vs. pattern-augmented metrics."""
    metrics = [
        ("F1",            0.459, 0.741),
        ("Recall",        0.298, 0.589),
        ("Spearman rho",  0.042, 0.181),
    ]
    # Correct pairs is shown on a second subplot with its own scale.
    correct_pairs = ("Correct pairs", 104, 349, 136, 231)  # label, num_base, denom_base, num_new, denom_new

    fig, (ax, ax2) = plt.subplots(
        1, 2,
        figsize=(7.5, 3.8),
        gridspec_kw={"width_ratios": [2.6, 1.0]},
    )

    # ----- Left subplot: unit-interval metrics -----
    labels = [m[0] for m in metrics]
    baseline = np.array([m[1] for m in metrics])
    enhanced = np.array([m[2] for m in metrics])

    x = np.arange(len(labels))
    w = 0.38

    b1 = ax.bar(
        x - w / 2, baseline, width=w,
        color="white", edgecolor=GRAY_DARK, linewidth=0.9,
        hatch="////", label="Baseline (centrality only)",
    )
    b2 = ax.bar(
        x + w / 2, enhanced, width=w,
        color=ACCENT, edgecolor=GRAY_DARK, linewidth=0.9,
        label="With communication patterns",
    )

    # Target line at F1 = 0.65 (applies to the F1 metric)
    ax.axhline(
        0.65, color=GRAY_MID, linestyle="--", linewidth=0.9,
        zorder=0,
    )
    ax.text(
        len(labels) - 0.55, 0.66,
        "target F1 = 0.65",
        fontsize=8, color=GRAY_MID, style="italic",
        ha="right", va="bottom",
    )

    # Value labels on top of bars
    for rect, v in zip(b1, baseline):
        ax.text(
            rect.get_x() + rect.get_width() / 2, v + 0.015,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8,
            color=GRAY_DARK,
        )
    for rect, v in zip(b2, enhanced):
        ax.text(
            rect.get_x() + rect.get_width() / 2, v + 0.015,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8,
            color=GRAY_DARK, weight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 0.95)
    ax.set_title("Dominance and rank-correlation metrics", loc="left", pad=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # ----- Right subplot: correct-pair counts -----
    lbl, num_b, den_b, num_n, den_n = correct_pairs
    bars = ax2.bar(
        [0, 1],
        [num_b, num_n],
        width=0.55,
        color=["white", ACCENT],
        edgecolor=GRAY_DARK, linewidth=0.9,
        hatch=["////", None],
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Baseline", "With\npatterns"])
    ax2.set_ylabel("Correct dominance pairs")
    ax2.set_ylim(0, max(num_b, num_n) * 1.25)
    ax2.set_title("Correct pairs", loc="left", pad=8)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

    for rect, num, den in zip(bars, [num_b, num_n], [den_b, den_n]):
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + max(num_b, num_n) * 0.02,
            f"{num}/{den}",
            ha="center", va="bottom", fontsize=8, color=GRAY_DARK,
            weight="bold" if num == num_n else "normal",
        )

    # GT-normalised note under the right subplot
    ax2.text(
        0.5, -0.23,
        "(GT names normalised)",
        transform=ax2.transAxes,
        ha="center", va="top", fontsize=7.5,
        color=GRAY_MID, style="italic",
    )

    fig.suptitle(
        "RQ1: communication-pattern features raise dominance F1 from 0.459 to 0.741",
        fontsize=11, y=1.02,
    )

    out = FIG_DIR / "metric_comparison"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Wrote {out.with_suffix('.pdf').name} / .png")
    return out.with_suffix(".pdf")


# ---------------------------------------------------------------------------
# Figure 3: Network visualization with communities
# ---------------------------------------------------------------------------

def figure_network_communities() -> Path:
    """Spring layout of the 139-node communication graph.

    Nodes coloured by Louvain community, sized by PageRank. Top-10
    PageRank nodes are labelled.
    """
    from orggraph.extraction.communities import detect_communities

    G = _load_or_build_graph()

    # Detect communities on the same graph used in run_rq1.
    communities = detect_communities(G)
    community_ids = sorted(set(communities.values()))
    n_comm = len(community_ids)

    # PageRank for sizing / labelling
    pagerank = nx.pagerank(G, weight="weight")
    top_by_pr = sorted(pagerank.items(), key=lambda kv: -kv[1])[:10]
    top_labels = {n for n, _ in top_by_pr}

    # Layout on the undirected projection. Kamada-Kawai gives a cleaner
    # placement than spring for graphs with hub-and-spoke structure,
    # followed by a few spring iterations to spread out overlapping
    # nodes. Use unweighted positions so dense edges don't dominate.
    G_und = G.to_undirected()
    try:
        pos_init = nx.kamada_kawai_layout(G_und)
    except Exception:
        pos_init = nx.spring_layout(G_und, seed=42)
    pos = nx.spring_layout(
        G_und,
        pos=pos_init,
        seed=42,
        k=3.0 / np.sqrt(G_und.number_of_nodes()),
        iterations=120,
        scale=1.6,
    )

    # Muted sequential palette for communities. We use a tasteful
    # qualitative set rather than pure grayscale: 8 distinguishable but
    # subdued hues. Ordering is stable for reproducibility.
    palette = [
        "#2F5F8F",  # accent blue
        "#6B7A8F",  # cool gray-blue
        "#A38560",  # warm tan
        "#7C6E8B",  # muted violet
        "#6A8E7F",  # sage
        "#8F5F5F",  # dusty brick
        "#B3B3B3",  # light gray
        "#4D4D4D",  # dark gray
    ]
    # Extend palette if we ever see more than 8 communities
    while len(palette) < n_comm:
        palette.append("#333333")

    node_colors = [palette[communities[n] % len(palette)] for n in G.nodes()]

    # Node sizes proportional to sqrt(PageRank), normalised. The sqrt
    # keeps mid-tier nodes visible while still emphasising the hubs.
    pr_vals = np.array([pagerank[n] for n in G.nodes()])
    pr_sqrt = np.sqrt(pr_vals)
    pr_min, pr_max = pr_sqrt.min(), pr_sqrt.max()
    if pr_max > pr_min:
        pr_norm = (pr_sqrt - pr_min) / (pr_max - pr_min)
    else:
        pr_norm = np.zeros_like(pr_sqrt)
    node_sizes = 25 + pr_norm * 400

    fig, ax = plt.subplots(figsize=(8.5, 7.3))

    # Draw edges first (dense graph, so subtle)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=GRAY_LIGHT,
        width=0.25,
        alpha=0.35,
        arrows=False,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=GRAY_DARK,
        linewidths=0.4,
        alpha=0.92,
    )

    # Label the top-10 nodes by PageRank. Place labels radially away
    # from the centroid, then run a short repulsion pass so overlapping
    # labels separate. Leader lines connect each label to its node.
    labeled = list(top_labels)
    xs = np.array([pos[n][0] for n in labeled])
    ys = np.array([pos[n][1] for n in labeled])
    cx, cy = xs.mean(), ys.mean()
    x_span = max(p[0] for p in pos.values()) - min(p[0] for p in pos.values())
    y_span = max(p[1] for p in pos.values()) - min(p[1] for p in pos.values())
    max_span = max(x_span, y_span)
    radial = 0.10 * max_span
    min_dist = 0.16 * max_span  # minimum spacing between label centres

    # Initial radial positions
    label_pos_arr = []
    for n in labeled:
        x, y = pos[n]
        vx, vy = x - cx, y - cy
        norm = (vx ** 2 + vy ** 2) ** 0.5
        if norm < 1e-9:
            vx, vy, norm = 1.0, 0.0, 1.0
        lx = x + radial * vx / norm
        ly = y + radial * vy / norm
        label_pos_arr.append([lx, ly])
    label_pos_arr = np.array(label_pos_arr)

    # Repulsion passes: push overlapping labels apart
    for _ in range(80):
        moved = False
        for i in range(len(labeled)):
            for j in range(i + 1, len(labeled)):
                dx = label_pos_arr[j, 0] - label_pos_arr[i, 0]
                dy = label_pos_arr[j, 1] - label_pos_arr[i, 1]
                d = (dx ** 2 + dy ** 2) ** 0.5
                if d < min_dist and d > 1e-9:
                    # Push each label half the overlap along the axis
                    push = (min_dist - d) / 2
                    ux, uy = dx / d, dy / d
                    label_pos_arr[i, 0] -= ux * push
                    label_pos_arr[i, 1] -= uy * push
                    label_pos_arr[j, 0] += ux * push
                    label_pos_arr[j, 1] += uy * push
                    moved = True
        if not moved:
            break

    for n, (lx, ly) in zip(labeled, label_pos_arr):
        x, y = pos[n]
        ax.annotate(
            n,
            xy=(x, y),
            xytext=(lx, ly),
            fontsize=7.5,
            fontfamily="serif",
            color=GRAY_DARK,
            ha="center", va="center",
            bbox=dict(
                facecolor="white",
                edgecolor=GRAY_LIGHT,
                linewidth=0.3,
                alpha=0.92,
                pad=1.2,
                boxstyle="round,pad=0.25",
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=GRAY_MID,
                linewidth=0.4,
                alpha=0.7,
            ),
            zorder=10,
        )

    ax.set_axis_off()
    ax.set_title(
        f"Internal communication network (Enron, N={G.number_of_nodes()} employees, "
        f"E={G.number_of_edges()} edges); Louvain communities",
        loc="left", pad=8,
    )

    # Legend: community swatches
    legend_elems = [
        mpl.patches.Patch(
            facecolor=palette[i % len(palette)],
            edgecolor=GRAY_DARK, linewidth=0.4,
            label=f"Community {i + 1}",
        )
        for i in range(n_comm)
    ]
    ax.legend(
        handles=legend_elems,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.04),
        frameon=False,
        fontsize=8,
        ncol=4,
        title="Louvain communities",
        title_fontsize=9,
    )

    # Text annotation: node size scale, placed below the plot
    fig.text(
        0.99, 0.01,
        "Node size proportional to PageRank; labels on top-10 nodes.",
        ha="right", va="bottom",
        fontsize=8, color=GRAY_MID, style="italic",
    )

    out = FIG_DIR / "network_communities"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Wrote {out.with_suffix('.pdf').name} / .png")
    return out.with_suffix(".pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 60)
    print("Generating RQ1 thesis figures")
    print("=" * 60)
    print(f"  Output directory: {FIG_DIR}")

    print("\n[1/3] Figure 1: hierarchy_comparison")
    figure_hierarchy_comparison()

    print("\n[2/3] Figure 2: metric_comparison")
    figure_metric_comparison()

    print("\n[3/3] Figure 3: network_communities")
    figure_network_communities()

    print("\nDone. Files in:", FIG_DIR)


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Stage 9: generate thesis figures")
    parser.parse_args(argv)
    run()


if __name__ == "__main__":
    main()
