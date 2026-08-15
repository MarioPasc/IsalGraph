"""Shared graph fixtures.  Standard library only.

Graphs are returned as ``(n_nodes, edges)`` pairs so that this module has no
``networkx`` import and can be used by a test that runs with ``networkx``
uninstalled.  :func:`to_networkx` is the one function that needs it, and its
import lives in the function body.

Two fixtures carry arguments the paper makes:

**The running example** ``C4(0,1,2,3) + K3(3,4,5)`` is the graph every
backend's expected output is written against.

**K33 vs the triangular prism** is the completeness witness: both connected,
both 3-regular on six vertices, not isomorphic.  WL gives them kernel
distance exactly ``0.0000`` while every other pool member separates them.
1-WL cannot distinguish regular graphs of the same degree and order -- the
colouring is constant after round 1, so refinement never starts, and no
number of rounds fixes it.  That six-node witness is the cleanest evidence
for R1.2's uniqueness axis and it costs one small figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random

    import networkx as nx

#: ``(n_nodes, edges)`` for each fixture.
Fixture = tuple[int, tuple[tuple[int, int], ...]]

RUNNING_EXAMPLE: Fixture = (
    6,
    ((0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 3)),
)
"""``C4(0,1,2,3) + K3(3,4,5)``: n = 6, m = 7, |Aut| = 4."""

RUNNING_EXAMPLE_MINUS_EDGE: Fixture = (
    6,
    ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 3)),
)
"""``H = G - (0,3)``: n = 6, m = 6, |Aut| = 2."""

K33: Fixture = (
    6,
    ((0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)),
)
"""Complete bipartite ``K_{3,3}``: connected, 3-regular on six vertices."""

PRISM: Fixture = (
    6,
    ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)),
)
"""The triangular prism ``C3 x K2``: connected, 3-regular, **not** ``K_{3,3}``."""

C4_PLUS_K3_DISJOINT: Fixture = (
    7,
    ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 4)),
)
"""A disconnected fixture: ``C4`` and ``K3`` with no bridge.

min-DFS and IsalGraph raise on it; adjacency, graph6, sparse6 and AGM do
not.  That asymmetry is a documented row in the AE.3 properties table, so
the fixture exists to make it testable rather than asserted.
"""

PATH_2: Fixture = (2, ((0, 1),))
"""The smallest connected graph, for boundary tests."""

EMPTY_3: Fixture = (3, ())
"""Three isolated vertices: no edges, still a valid encode for the n^2 family."""

ALL_FIXTURES: dict[str, Fixture] = {
    "running_example": RUNNING_EXAMPLE,
    "running_example_minus_edge": RUNNING_EXAMPLE_MINUS_EDGE,
    "k33": K33,
    "prism": PRISM,
    "c4_plus_k3_disjoint": C4_PLUS_K3_DISJOINT,
    "path_2": PATH_2,
    "empty_3": EMPTY_3,
}

#: Fixtures every backend must encode: connected, no isolated vertices.
CONNECTED_FIXTURES: tuple[str, ...] = (
    "running_example",
    "running_example_minus_edge",
    "k33",
    "prism",
    "path_2",
)


def to_networkx(fixture: Fixture) -> nx.Graph:
    """Build a ``networkx.Graph`` from a fixture.

    Nodes are added before edges and in ascending order, so the incident
    labelling is deterministic.  That is determinism, **not** isomorphism
    invariance -- the order-dependent formats are order-dependent by design
    and their F3 failure is the finding.
    """
    import networkx as nx

    n, edges = fixture
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    return graph


def relabelled(fixture: Fixture, rng: random.Random) -> nx.Graph:
    """A relabelled copy of *fixture* with a genuinely fresh insertion order.

    ``nx.relabel_nodes(copy=True)`` alone **preserves insertion order**, so
    an F3 harness built on it makes order-dependent formats look invariant
    (finding 13).  This rebuilds the copy: the node mapping is shuffled, the
    insertion order is shuffled independently, and the edge list is shuffled
    again.  An F3 harness that cannot fail is worthless, so a test asserts
    this relabeller *can* make ``graph6`` fail.
    """
    import networkx as nx

    n, edges = fixture
    mapping = list(range(n))
    rng.shuffle(mapping)
    order = list(mapping)
    rng.shuffle(order)
    graph = nx.Graph()
    graph.add_nodes_from(order)
    relabelled_edges = [(mapping[u], mapping[v]) for u, v in edges]
    rng.shuffle(relabelled_edges)
    graph.add_edges_from(relabelled_edges)
    return graph


def shuffled_copy(graph: nx.Graph, rng: random.Random) -> nx.Graph:
    """A relabelled copy of a live graph, with a fresh insertion order.

    The same construction as :func:`relabelled`, for graphs that did not come
    from a fixture.  **The rng draw sequence is identical to
    ``scratch/sweep.py::shuffled_copy``** -- three shuffles, in the order
    mapping, insertion order, edges -- because the reproduction gate replays
    that stream and any change to the draw count desynchronises it.
    """
    import networkx as nx

    nodes = list(graph.nodes())
    new = list(range(len(nodes)))
    rng.shuffle(new)
    mapping = dict(zip(nodes, new, strict=True))
    order = list(new)
    rng.shuffle(order)
    out = nx.Graph()
    out.add_nodes_from(order)
    edges = [(mapping[u], mapping[v]) for u, v in graph.edges()]
    rng.shuffle(edges)
    out.add_edges_from(edges)
    return out


__all__ = [
    "ALL_FIXTURES",
    "C4_PLUS_K3_DISJOINT",
    "CONNECTED_FIXTURES",
    "EMPTY_3",
    "K33",
    "PATH_2",
    "PRISM",
    "RUNNING_EXAMPLE",
    "RUNNING_EXAMPLE_MINUS_EDGE",
    "Fixture",
    "relabelled",
    "shuffled_copy",
    "to_networkx",
]
