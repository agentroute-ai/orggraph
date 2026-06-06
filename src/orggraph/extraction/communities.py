"""Community detection for organizational department identification."""

import networkx as nx
from networkx.algorithms.community import louvain_communities


def detect_communities(
    G: nx.DiGraph,
    resolution: float = 1.0,
    seed: int = 42,
) -> dict[str, int]:
    """Detect communities using the Louvain algorithm.

    Args:
        G: Directed communication graph.
        resolution: Louvain resolution parameter. Higher = more communities.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping node -> community_id (int).
    """
    G_undirected = G.to_undirected()

    communities = louvain_communities(
        G_undirected,
        weight="weight",
        resolution=resolution,
        seed=seed,
    )

    node_to_community: dict[str, int] = {}
    for community_id, members in enumerate(communities):
        for node in members:
            node_to_community[node] = community_id

    return node_to_community
