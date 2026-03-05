"""Round-trip correctness tests (Phase 1 + Phase 2).

Phase 1: Short, manually inspectable strings.
For each string w:
  1. S2G(w)           -> G1
  2. G2S(G1, 0)       -> w'
  3. S2G(w')          -> G2
  4. Assert G1 ~ G2   (graph isomorphism)

Phase 2: Random strings (see also benchmarks/random_roundtrip.py).
"""

from __future__ import annotations

import random

import pytest

from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.core.string_to_graph import StringToGraph


def _roundtrip(w: str, directed: bool) -> None:
    """Run the full round-trip check for string *w*."""
    # Forward: string -> graph
    stg1 = StringToGraph(w, directed_graph=directed)
    g1, _ = stg1.run()

    # Inverse: graph -> string
    gts = GraphToString(g1)
    w_prime, _ = gts.run(0)

    # Forward again: string' -> graph'
    stg2 = StringToGraph(w_prime, directed_graph=directed)
    g2, _ = stg2.run()

    # Isomorphism check
    assert g1.is_isomorphic(g2), (
        f"Round-trip failed for '{w}' (directed={directed}).\n"
        f"  G1: {g1.node_count()} nodes, {g1.logical_edge_count()} edges\n"
        f"  G2: {g2.node_count()} nodes, {g2.logical_edge_count()} edges\n"
        f"  w' = '{w_prime}'"
    )

    # Sanity: node and edge counts must match.
    assert g1.node_count() == g2.node_count(), (
        f"Node count mismatch: {g1.node_count()} vs {g2.node_count()}"
    )
    assert g1.logical_edge_count() == g2.logical_edge_count(), (
        f"Edge count mismatch: {g1.logical_edge_count()} vs {g2.logical_edge_count()}"
    )


# ======================================================================
# Phase 1: Short strings (from ISALGRAPH_AGENT_CONTEXT.md Section 3.1)
# ======================================================================

# Single-instruction strings
PHASE1_SINGLE = ["V", "v"]

# Two-instruction strings
PHASE1_DOUBLE = ["VV", "Vv", "vV", "vv", "VC", "vC", "Vc", "NV", "nv", "PV", "pv"]

# Three-instruction strings
PHASE1_TRIPLE = ["VNV", "VnC", "vNv", "vvc", "VVN", "VNC"]

PHASE1_ALL = PHASE1_SINGLE + PHASE1_DOUBLE + PHASE1_TRIPLE


@pytest.mark.parametrize("w", PHASE1_ALL)
def test_roundtrip_undirected(w: str) -> None:
    """Phase 1: round-trip for short strings (undirected)."""
    _roundtrip(w, directed=False)


@pytest.mark.parametrize("w", PHASE1_ALL)
def test_roundtrip_directed(w: str) -> None:
    """Phase 1: round-trip for short strings (directed)."""
    _roundtrip(w, directed=True)


# ======================================================================
# Phase 1 extension: specific graph structures
# ======================================================================


def test_roundtrip_triangle_undirected(triangle_undirected: SparseGraph) -> None:
    """Round-trip starting from an explicitly constructed triangle."""
    gts = GraphToString(triangle_undirected)
    w, _ = gts.run(0)

    stg = StringToGraph(w, directed_graph=False)
    g2, _ = stg.run()

    assert triangle_undirected.is_isomorphic(g2)


def test_roundtrip_triangle_directed(triangle_directed: SparseGraph) -> None:
    """Round-trip starting from an explicitly constructed directed triangle."""
    gts = GraphToString(triangle_directed)
    w, _ = gts.run(0)

    stg = StringToGraph(w, directed_graph=True)
    g2, _ = stg.run()

    assert triangle_directed.is_isomorphic(g2)


def test_roundtrip_path_3_undirected(path_3_undirected: SparseGraph) -> None:
    """Round-trip for 3-node path."""
    gts = GraphToString(path_3_undirected)
    w, _ = gts.run(0)

    stg = StringToGraph(w, directed_graph=False)
    g2, _ = stg.run()

    assert path_3_undirected.is_isomorphic(g2)


def test_roundtrip_empty_string() -> None:
    """Empty string -> 1-node graph -> empty string."""
    stg1 = StringToGraph("", directed_graph=False)
    g1, _ = stg1.run()
    gts = GraphToString(g1)
    w, _ = gts.run(0)
    assert w == ""

    stg2 = StringToGraph(w, directed_graph=False)
    g2, _ = stg2.run()
    assert g1.is_isomorphic(g2)


def test_roundtrip_star_graph() -> None:
    """Star graph: center node 0 connected to 1,2,3,4."""
    g = SparseGraph(5, directed_graph=False)
    for _ in range(5):
        g.add_node()
    for i in range(1, 5):
        g.add_edge(0, i)

    gts = GraphToString(g)
    w, _ = gts.run(0)

    stg = StringToGraph(w, directed_graph=False)
    g2, _ = stg.run()

    assert g.is_isomorphic(g2)


def test_roundtrip_complete_4() -> None:
    """Complete graph K4 (undirected)."""
    g = SparseGraph(4, directed_graph=False)
    for _ in range(4):
        g.add_node()
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)

    gts = GraphToString(g)
    w, _ = gts.run(0)

    stg = StringToGraph(w, directed_graph=False)
    g2, _ = stg.run()

    assert g.is_isomorphic(g2)
    assert g.logical_edge_count() == g2.logical_edge_count()


def test_roundtrip_cycle_5() -> None:
    """Cycle C5 (undirected): 0-1-2-3-4-0."""
    g = SparseGraph(5, directed_graph=False)
    for _ in range(5):
        g.add_node()
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)

    gts = GraphToString(g)
    w, _ = gts.run(0)

    stg = StringToGraph(w, directed_graph=False)
    g2, _ = stg.run()

    assert g.is_isomorphic(g2)


# ======================================================================
# Phase 2 (mini): small random-ish tests
# ======================================================================


@pytest.mark.parametrize("seed", range(20))
def test_roundtrip_random_undirected(seed: int) -> None:
    """Random string round-trip (undirected), length 1..15."""
    rng = random.Random(seed)
    length = rng.randint(1, 15)
    alphabet = "NnPpVvCcW"
    w = "".join(rng.choice(alphabet) for _ in range(length))
    _roundtrip(w, directed=False)


@pytest.mark.parametrize("seed", range(20))
def test_roundtrip_random_directed(seed: int) -> None:
    """Random string round-trip (directed), length 1..15."""
    rng = random.Random(seed)
    length = rng.randint(1, 15)
    alphabet = "NnPpVvCcW"
    w = "".join(rng.choice(alphabet) for _ in range(length))
    _roundtrip(w, directed=True)
