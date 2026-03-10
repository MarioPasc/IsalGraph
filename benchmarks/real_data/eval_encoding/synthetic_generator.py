# ruff: noqa: N803, N806
"""Synthetic graph generation for encoding complexity analysis.

Generates 10 families of graphs with controlled structural properties
for benchmarking IsalGraph encoding complexity:

- Deterministic: path, star, cycle, complete, binary_tree, grid
- Stochastic:    ba_m1, ba_m2, gnp_03, gnp_05

All generated graphs are connected undirected graphs with contiguous
integer node labels starting from 0.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx

# =============================================================================
# Family configurations
# =============================================================================

FAMILY_CONFIGS: dict[str, dict[str, Any]] = {
    "path": {"n_min": 3, "n_max": 50, "deterministic": True},
    "star": {"n_min": 3, "n_max": 50, "deterministic": True},
    "cycle": {"n_min": 3, "n_max": 50, "deterministic": True},
    "complete": {"n_min": 3, "n_max": 15, "deterministic": True},
    "binary_tree": {"n_min": 3, "n_max": 50, "deterministic": True},
    "ba_m1": {"n_min": 5, "n_max": 50, "deterministic": False},
    "ba_m2": {"n_min": 5, "n_max": 50, "deterministic": False},
    "gnp_03": {"n_min": 5, "n_max": 30, "deterministic": False},
    "gnp_05": {"n_min": 5, "n_max": 20, "deterministic": False},
    "grid": {"n_min": 4, "n_max": 49, "deterministic": True},
}

ALL_FAMILIES = list(FAMILY_CONFIGS.keys())


# =============================================================================
# Graph generation
# =============================================================================


def generate_graph_family(
    family: str,
    n_nodes: int,
    seed: int = 42,
    instance: int = 0,
) -> nx.Graph:
    """Generate a connected undirected graph from a specified family.

    Args:
        family: One of the keys in FAMILY_CONFIGS.
        n_nodes: Target number of nodes.
        seed: Base random seed.
        instance: Instance index (for stochastic families).

    Returns:
        Connected undirected NetworkX graph with contiguous int labels.

    Raises:
        ValueError: If family is unknown or n_nodes is invalid.
    """
    if family not in FAMILY_CONFIGS:
        msg = f"Unknown family: {family}. Valid: {list(FAMILY_CONFIGS.keys())}"
        raise ValueError(msg)

    cfg = FAMILY_CONFIGS[family]
    if n_nodes < cfg["n_min"]:
        msg = f"n_nodes={n_nodes} < n_min={cfg['n_min']} for family {family}"
        raise ValueError(msg)

    # Deterministic seed for stochastic families
    rng_seed = seed + instance * 1000 + n_nodes

    if family == "path":
        return nx.path_graph(n_nodes)

    if family == "star":
        return nx.star_graph(n_nodes - 1)  # star_graph(k) has k+1 nodes

    if family == "cycle":
        return nx.cycle_graph(n_nodes)

    if family == "complete":
        return nx.complete_graph(n_nodes)

    if family == "binary_tree":
        return _generate_binary_tree(n_nodes)

    if family == "ba_m1":
        return nx.barabasi_albert_graph(n_nodes, 1, seed=rng_seed)

    if family == "ba_m2":
        return nx.barabasi_albert_graph(n_nodes, 2, seed=rng_seed)

    if family == "gnp_03":
        return _generate_connected_gnp(n_nodes, 0.3, rng_seed)

    if family == "gnp_05":
        return _generate_connected_gnp(n_nodes, 0.5, rng_seed)

    if family == "grid":
        return _generate_grid(n_nodes)

    msg = f"Unhandled family: {family}"
    raise ValueError(msg)


def _generate_binary_tree(n_nodes: int) -> nx.Graph:
    """Generate a balanced binary tree trimmed to n_nodes.

    Starts with a balanced tree of sufficient depth, then removes
    highest-labeled leaves until the target size is reached.
    Verifies connectivity after trimming.
    """
    depth = max(1, int(math.ceil(math.log2(n_nodes + 1))) - 1)
    G = nx.balanced_tree(r=2, h=depth)

    # If tree is already the right size or smaller, return as-is
    if G.number_of_nodes() <= n_nodes:
        return nx.convert_node_labels_to_integers(G)

    # Remove highest-labeled leaves (preserves connectivity for balanced trees)
    while G.number_of_nodes() > n_nodes:
        # Find leaves (degree 1) and remove the highest-labeled one
        leaves = [v for v in G.nodes() if G.degree(v) == 1]
        if not leaves:
            break
        G.remove_node(max(leaves))

    # Safety check: ensure connected
    if not nx.is_connected(G):
        # Take largest component
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()

    return nx.convert_node_labels_to_integers(G)


def _generate_connected_gnp(
    n_nodes: int,
    p: float,
    seed: int,
    max_attempts: int = 10,
) -> nx.Graph:
    """Generate a connected GNP(n, p) graph.

    Takes the largest connected component. If LCC < 0.7*n,
    retries with a different seed up to max_attempts times.

    Args:
        n_nodes: Target number of nodes.
        p: Edge probability.
        seed: Random seed.
        max_attempts: Maximum regeneration attempts.

    Returns:
        Connected graph (may have fewer than n_nodes if LCC is small).
    """
    min_acceptable = max(3, int(0.7 * n_nodes))

    for attempt in range(max_attempts):
        attempt_seed = seed + attempt * 7919  # Large prime offset
        G = nx.gnp_random_graph(n_nodes, p, seed=attempt_seed)

        if nx.is_connected(G):
            return G

        # Take largest connected component
        lcc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc).copy()

        if G_lcc.number_of_nodes() >= min_acceptable:
            return nx.convert_node_labels_to_integers(G_lcc)

    # Last resort: return whatever we have
    G = nx.gnp_random_graph(n_nodes, p, seed=seed)
    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()
    return nx.convert_node_labels_to_integers(G)


def _generate_grid(n_nodes: int) -> nx.Graph:
    """Generate a grid graph with approximately n_nodes nodes.

    Finds r, c such that r*c is close to n_nodes, then trims
    by removing highest-labeled nodes.
    """
    r = max(2, int(math.ceil(math.sqrt(n_nodes))))
    c = max(2, int(math.ceil(n_nodes / r)))
    G = nx.grid_2d_graph(r, c)
    G = nx.convert_node_labels_to_integers(G)

    # Trim to exact size by removing highest-labeled nodes
    while G.number_of_nodes() > n_nodes:
        node_to_remove = max(G.nodes())
        G.remove_node(node_to_remove)

    # Ensure connected after trimming
    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


# =============================================================================
# Utilities
# =============================================================================


def get_family_n_range(
    family: str,
    max_n: int | None = None,
) -> list[int]:
    """Get the range of valid node counts for a family.

    Args:
        family: Graph family name.
        max_n: Optional upper bound override.

    Returns:
        Sorted list of node counts.
    """
    cfg = FAMILY_CONFIGS[family]
    n_min = cfg["n_min"]
    n_max = min(cfg["n_max"], max_n) if max_n is not None else cfg["n_max"]
    return list(range(n_min, n_max + 1))


def generate_density_sweep(
    n: int,
    p_values: list[float],
    n_instances: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate GNP(n, p) graphs for density dependence analysis.

    Args:
        n: Fixed number of nodes.
        p_values: List of edge probabilities.
        n_instances: Number of instances per p value.
        seed: Base random seed.

    Returns:
        List of dicts with keys: p, instance, graph, density, n_nodes, n_edges.
    """
    results: list[dict[str, Any]] = []

    for p in p_values:
        for inst in range(n_instances):
            rng_seed = seed + int(p * 1000) + inst * 7919
            G = _generate_connected_gnp(n, p, rng_seed)
            actual_n = G.number_of_nodes()
            n_edges = G.number_of_edges()
            max_edges = actual_n * (actual_n - 1) / 2
            density = n_edges / max_edges if max_edges > 0 else 0.0

            results.append(
                {
                    "p": p,
                    "instance": inst,
                    "graph": G,
                    "density": density,
                    "n_nodes": actual_n,
                    "n_edges": n_edges,
                }
            )

    return results


def compute_density(G: nx.Graph) -> float:
    """Compute edge density of an undirected graph."""
    n = G.number_of_nodes()
    max_edges = n * (n - 1) / 2
    if max_edges == 0:
        return 0.0
    return G.number_of_edges() / max_edges
