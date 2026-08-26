"""Symmetry and partition-resolution measurements for T-13.

This module is the instrument behind the finding that replaces `corrections.md`
§5 item 4 and `decisions.md` §17.  Those two files assert that the encoder's
structural-triplet pruning key is *"provably coarser than 1-WL"* and gives
*"2.4-2.6x fewer classes"*.  Both are false:

- **Not ordered.**  The two partitions are incomparable in general.  The witness
  is exact and needs no enumeration: :func:`witness_prism_k33` -- a connected,
  3-regular, 12-node graph -- has a stable 1-WL partition of **one** class and a
  triplet partition of **four**, so 1-WL does not refine the triplet key there.
  :func:`witness_incomparable` goes further and gives a graph on which *neither*
  partition refines the other while both have the same number of classes.
- **Not 2.4-2.6x.**  That ratio came from one hand-picked graph.  The cohort
  median is 1.021.

The replacement claim is a theorem rather than a measurement:

    **Proposition 1.**  Let ``f`` be any node invariant, i.e. any map assigning
    ``f_G(v)`` such that ``f_G(v) = f_{G^sigma}(v^sigma)`` for every
    ``sigma in Aut(G)``.  Then the partition induced by ``f`` is coarser than or
    equal to the orbit partition of ``Aut(G)``.

    *Proof.*  Let ``u, v`` lie in one orbit, so ``v = u^sigma`` for some
    ``sigma in Aut(G)``.  Invariance gives ``f_G(v) = f_{G^sigma}(u^sigma)``,
    and ``G^sigma = G`` because ``sigma`` is an automorphism, hence
    ``f_G(v) = f_G(u)``.  Every orbit therefore lies inside one ``f``-class,
    which is exactly the statement that the orbit partition refines the
    ``f``-partition.  QED

Both 1-WL and the triplet key are node invariants, so neither can be finer than
the orbits -- there is **no headroom for a finer pruning invariant at all**, and
measurement shows 1-WL already sits at that floor on real graphs.  That is the
characterised worst case R3.7d asks for.  :func:`resolution_record` reports the
gap to the floor for both; the property test asserts the floor itself.

Two rules from `T-13-design.md` §3 are enforced here rather than documented:

4. ``log10|Aut|`` is ``log10(mantissa) + exponent``, **never** a float product.
   ``|Aut(K_200)| = 200!`` is about ``1e374`` and the product overflows.
5. The refinement test is **exact class containment**.  A class-count comparison
   is not a refinement test and may not be substituted for one -- that
   substitution is the error T-13 exists to correct.  See
   :func:`witness_incomparable` for a graph on which the counts agree in both
   directions and containment holds in neither.

No ``grakel``: it is unusable on Picasso under numpy 2, so :func:`wl_partition`
is self-contained and needs no numpy at all.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Mapping
from types import ModuleType
from typing import TypeVar

import networkx as nx

__all__ = [
    "TRIPLET_RADIUS",
    "log10_aut",
    "orbits",
    "refines",
    "resolution_record",
    "triplet_partition",
    "witness_incomparable",
    "witness_prism_k33",
    "wl_partition",
]

H = TypeVar("H", bound=Hashable)

#: Radius at which the incumbent pruning key truncates its BFS shells.  Frozen
#: by ``isalgraph.core.canonical_pruned.compute_structural_triplets``; the key is
#: ``(|N_1(v)|, |N_2(v)|, |N_3(v)|)``.
TRIPLET_RADIUS = 3


# ----------------------------------------------------------------------------
# pynauty access
# ----------------------------------------------------------------------------


def _pynauty() -> ModuleType:
    """Import ``pynauty`` lazily, as ``competitors.backends.nauty`` does.

    Returns:
        The ``pynauty`` module.
    """
    import pynauty

    module: ModuleType = pynauty
    return module


def _autgrp(graph: nx.Graph) -> tuple[float, int, tuple[int, ...], list[Hashable]]:
    """One ``pynauty.autgrp`` call, returning everything T-13 reads from it.

    ``competitors.backends.nauty`` already exposes ``automorphism_group_size``
    and ``automorphism_orbits``, but they are not reused here for two reasons.
    First, they run nauty twice for what T-13 always needs together, and
    :func:`resolution_record` is called once per graph in a campaign of ~1e4
    graphs.  Second, ``automorphism_group_size`` returns
    ``mantissa * 10 ** exponent``, which is exactly the overflowing product
    `T-13-design.md` §3 rule 4 forbids: it returns ``inf`` for ``K_200``.

    Args:
        graph: an undirected ``networkx`` graph.  Node labels may be arbitrary.

    Returns:
        ``(mantissa, exponent, orbit_array, nodes)``.  ``orbit_array[i]`` is the
        orbit representative of the vertex nauty indexed as ``i``, and
        ``nodes[i]`` is that vertex's ``networkx`` label.  ``|Aut(G)|`` is
        ``mantissa * 10 ** exponent``, never formed as such.
    """
    pynauty = _pynauty()
    nodes: list[Hashable] = list(graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    adjacency = {i: [index[w] for w in graph.neighbors(v)] for i, v in enumerate(nodes)}
    pg = pynauty.Graph(len(nodes), directed=False, adjacency_dict=adjacency)
    _generators, mantissa, exponent, orbit_array, _n_orbits = pynauty.autgrp(pg)
    return float(mantissa), int(exponent), tuple(int(o) for o in orbit_array), nodes


# ----------------------------------------------------------------------------
# |Aut(G)| and its orbits
# ----------------------------------------------------------------------------


def log10_aut(graph: nx.Graph) -> float:
    """``log10|Aut(G)|``, from ``pynauty.autgrp``'s (mantissa, exponent) pair.

    Computed as ``log10(mantissa) + exponent``.  **Never** as
    ``mantissa * 10 ** exponent``: ``|Aut|`` reaches ``n!``, the product
    overflows a IEEE-754 double above about ``1e308``, and ``K_200`` alone is
    ``200! ~ 1e374``.  Working in the log throughout also matches the analysis,
    which regresses ``log t`` on ``log|Aut|``.

    Args:
        graph: an undirected ``networkx`` graph.

    Returns:
        ``log10|Aut(G)|``.  ``0.0`` for a rigid graph, and for the empty and
        one-vertex graphs, whose automorphism groups are trivial.

    Raises:
        ValueError: if nauty reports a non-positive mantissa, which would make
            the logarithm undefined.  This has never been observed and the guard
            exists so that it cannot pass silently as ``-inf``.
    """
    if graph.number_of_nodes() <= 1:
        return 0.0
    mantissa, exponent, _orbit_array, _nodes = _autgrp(graph)
    if mantissa <= 0.0:
        raise ValueError(
            f"pynauty.autgrp returned mantissa={mantissa!r}, which is not positive; "
            f"log10|Aut| is undefined"
        )
    return math.log10(mantissa) + exponent


def orbits(graph: nx.Graph) -> dict[Hashable, int]:
    """Vertex orbits under ``Aut(G)``, as node -> dense orbit id from 0.

    Orbit ids are assigned in order of first appearance over ``graph.nodes()``,
    so the mapping is deterministic for a fixed node order.  The ids carry no
    meaning beyond "same id iff same orbit"; only the induced partition is used.

    Args:
        graph: an undirected ``networkx`` graph.

    Returns:
        ``{node: orbit_id}`` with ids ``0 .. n_orbits - 1``.
    """
    if graph.number_of_nodes() == 0:
        return {}
    _mantissa, _exponent, orbit_array, nodes = _autgrp(graph)
    dense: dict[int, int] = {}
    out: dict[Hashable, int] = {}
    for i, node in enumerate(nodes):
        representative = orbit_array[i]
        if representative not in dense:
            dense[representative] = len(dense)
        out[node] = dense[representative]
    return out


# ----------------------------------------------------------------------------
# 1-WL colour refinement
# ----------------------------------------------------------------------------


def wl_partition(graph: nx.Graph, *, rounds: int | None = None) -> dict[Hashable, int]:
    """1-WL colour refinement, run to stability by default.

    The graphs in this project carry no node labels, so refinement starts from
    the uniform colouring; round 1 therefore separates by degree and every later
    round by the multiset of neighbour colours.  Colours are re-compressed each
    round by sorting the signatures, which keeps the ids deterministic and
    isomorphism-invariant (the signature of a vertex depends only on the
    partition, never on the node labels).

    Stability is detected by the class count, which is sound because refinement
    is monotone: a round that does not increase the number of classes cannot
    have changed the partition.  It therefore terminates in at most ``n`` rounds.

    Self-contained by design: no ``grakel`` and no ``numpy``.  ``grakel`` is
    unusable on Picasso under numpy 2, and this is the arm that has to run there.

    Args:
        graph: an undirected ``networkx`` graph.
        rounds: number of refinement rounds.  ``None`` -- the default -- runs to
            stability.  An explicit value runs exactly that many rounds, which
            is what makes the k-round approximations comparable to
            ``wl_subtree``'s fixed depth.

    Returns:
        ``{node: colour_id}`` with dense ids from 0.

    Raises:
        ValueError: if *rounds* is negative.
    """
    if rounds is not None and rounds < 0:
        raise ValueError(f"rounds must be non-negative, got {rounds}")
    nodes: list[Hashable] = list(graph.nodes())
    if not nodes:
        return {}

    colour: dict[Hashable, int] = dict.fromkeys(nodes, 0)
    n_classes = 1
    limit = len(nodes) if rounds is None else rounds

    for _ in range(limit):
        signature = {
            v: (colour[v], tuple(sorted(colour[u] for u in graph.neighbors(v)))) for v in nodes
        }
        order = {sig: i for i, sig in enumerate(sorted(set(signature.values())))}
        refined = {v: order[signature[v]] for v in nodes}
        if rounds is None and len(order) == n_classes:
            # Refinement is monotone, so a round that adds no class is a fixed
            # point.  Return the previous colouring: it induces the same
            # partition and avoids a gratuitous relabelling.
            return colour
        colour = refined
        n_classes = len(order)

    return colour


# ----------------------------------------------------------------------------
# The incumbent pruning key
# ----------------------------------------------------------------------------


def _shell_sizes(graph: nx.Graph, source: Hashable) -> tuple[int, ...]:
    """BFS from *source*, counting vertices at distance exactly 1, 2, ..., R.

    Mirrors ``isalgraph.core.canonical_pruned._bfs_distance_counts``: same early
    stop at radius :data:`TRIPLET_RADIUS`, same treatment of unreachable
    vertices (they are simply never counted, so a disconnected graph yields
    smaller shells rather than an error).

    Args:
        graph: an undirected ``networkx`` graph.
        source: the vertex to expand from.

    Returns:
        ``(|N_1|, ..., |N_R|)`` with ``R = TRIPLET_RADIUS``.
    """
    distance: dict[Hashable, int] = {source: 0}
    counts = [0] * TRIPLET_RADIUS
    queue: deque[Hashable] = deque((source,))
    while queue:
        u = queue.popleft()
        d = distance[u]
        if d >= TRIPLET_RADIUS:
            continue
        for w in graph.neighbors(u):
            if w not in distance:
                distance[w] = d + 1
                counts[d] += 1
                queue.append(w)
    return tuple(counts)


def triplet_partition(graph: nx.Graph) -> dict[Hashable, tuple[int, int, int]]:
    """The encoder's pruning key: ``(|N_1(v)|, |N_2(v)|, |N_3(v)|)`` per vertex.

    This is the *incumbent* invariant -- what
    ``isalgraph.core.canonical_pruned.pruned_canonical_string`` actually prunes
    with -- reproduced on ``networkx`` graphs so that it can be compared against
    1-WL and the orbits on the same footing.  ``tests/test_symmetry.py`` asserts
    byte-equality with ``compute_structural_triplets`` on 500+ graphs; the
    reimplementation exists only because the frozen reference takes a
    ``SparseGraph`` and this package works in ``networkx``.

    Unlike :func:`wl_partition` the returned values are the invariant itself,
    not compressed ids, because the triplet is interpretable and its components
    are reported separately in the analysis.  :func:`refines` and the class
    counts treat them as opaque labels either way.

    Args:
        graph: an undirected ``networkx`` graph.

    Returns:
        ``{node: (|N_1|, |N_2|, |N_3|)}``.
    """
    out: dict[Hashable, tuple[int, int, int]] = {}
    for v in graph.nodes():
        shells = _shell_sizes(graph, v)
        out[v] = (shells[0], shells[1], shells[2])
    return out


# ----------------------------------------------------------------------------
# The refinement test
# ----------------------------------------------------------------------------


def refines(fine: Mapping[H, object], coarse: Mapping[H, object]) -> bool:
    """``True`` iff every *fine* class lies inside a single *coarse* class.

    **Exact class containment.**  A class-count comparison is not a refinement
    test and may not be used as one: on :func:`witness_incomparable` the two
    partitions have the same number of classes and neither refines the other, so
    every count-based rule reports refinement in both directions and is wrong in
    both.  That substitution is the error `corrections.md` §5 made and the one
    T-13 exists to correct, so the correct test is the primitive here and the
    counts are reported only alongside it.

    Args:
        fine: the candidate finer partition, as ``{key: class_label}``.
        coarse: the candidate coarser partition, over the same key set.

    Returns:
        ``True`` iff *fine* refines *coarse*.  Note that equality of partitions
        is refinement in both directions, and that the trivial partition into
        singletons refines everything.

    Raises:
        ValueError: if the two mappings are over different key sets.  A
            refinement relation between partitions of different sets is not
            defined, and silently intersecting the domains would let a truncated
            partition pass the test.
    """
    if fine.keys() != coarse.keys():
        missing = set(fine.keys()) ^ set(coarse.keys())
        raise ValueError(
            f"refines() needs two partitions of the same set; {len(missing)} key(s) "
            f"appear in exactly one of them, e.g. {sorted(map(repr, missing))[:5]}"
        )
    images: dict[object, object] = {}
    for key, fine_class in fine.items():
        coarse_class = coarse[key]
        if fine_class in images:
            if images[fine_class] != coarse_class:
                return False
        else:
            images[fine_class] = coarse_class
    return True


def _equal_partitions(left: Mapping[H, object], right: Mapping[H, object]) -> bool:
    """``True`` iff *left* and *right* induce the same partition."""
    return refines(left, right) and refines(right, left)


# ----------------------------------------------------------------------------
# The record
# ----------------------------------------------------------------------------


def resolution_record(graph: nx.Graph) -> dict[str, object]:
    """The nine symmetry fields of a T-13 record, for one graph.

    Every field is either the orbit partition, an invariant partition, or the
    exact relation between two of them.  ``wl_equals_orbits`` and
    ``triplet_equals_orbits`` are the two that carry the argument: by
    Proposition 1 neither invariant can beat the orbits, so ``True`` there means
    the invariant has *attained* the theoretical floor and no finer node
    invariant exists to be built.

    Args:
        graph: an undirected ``networkx`` graph.

    Returns:
        Exactly the keys ``log10_aut, n_orbits, max_orbit_size, n_wl_classes,
        n_triplet_classes, wl_refines_triplet, triplet_refines_wl,
        wl_equals_orbits, triplet_equals_orbits`` -- no more, no fewer, because
        ``schema.py`` copies them into the record verbatim.
    """
    if graph.number_of_nodes() == 0:
        orbit_map: dict[Hashable, int] = {}
        log10_size = 0.0
        max_orbit = 0
    else:
        mantissa, exponent, orbit_array, nodes = _autgrp(graph)
        if mantissa <= 0.0:
            raise ValueError(
                f"pynauty.autgrp returned mantissa={mantissa!r}, which is not positive; "
                f"log10|Aut| is undefined"
            )
        log10_size = math.log10(mantissa) + exponent
        dense: dict[int, int] = {}
        orbit_map = {}
        sizes: dict[int, int] = {}
        for i, node in enumerate(nodes):
            representative = orbit_array[i]
            if representative not in dense:
                dense[representative] = len(dense)
            orbit_map[node] = dense[representative]
            sizes[representative] = sizes.get(representative, 0) + 1
        max_orbit = max(sizes.values())

    wl = wl_partition(graph)
    triplet = triplet_partition(graph)

    return {
        "log10_aut": log10_size,
        "n_orbits": len(set(orbit_map.values())),
        "max_orbit_size": max_orbit,
        "n_wl_classes": len(set(wl.values())),
        "n_triplet_classes": len(set(triplet.values())),
        "wl_refines_triplet": refines(wl, triplet),
        "triplet_refines_wl": refines(triplet, wl),
        "wl_equals_orbits": _equal_partitions(wl, orbit_map),
        "triplet_equals_orbits": _equal_partitions(triplet, orbit_map),
    }


#: The exact key set :func:`resolution_record` returns.  ``schema.py`` may
#: assert against this rather than restating the nine names.
RESOLUTION_FIELDS: tuple[str, ...] = (
    "log10_aut",
    "n_orbits",
    "max_orbit_size",
    "n_wl_classes",
    "n_triplet_classes",
    "wl_refines_triplet",
    "triplet_refines_wl",
    "wl_equals_orbits",
    "triplet_equals_orbits",
)


# ----------------------------------------------------------------------------
# Witnesses
# ----------------------------------------------------------------------------


def witness_prism_k33() -> nx.Graph:
    """The 3-prism spliced to ``K_{3,3}``: 1-WL does **not** refine the triplet key.

    Connected, 3-regular, ``n = 12``, ``m = 18``.  Because it is regular and
    vertex-transitive under 1-WL's eyes, the stable 1-WL partition is a single
    class; the triplet key sees the two halves' differing distance-2 and
    distance-3 shells and produces four.  So 1-WL does not refine the triplet
    partition here, which refutes `corrections.md` §5's *"provably coarser than
    1-WL"* -- a single exact counterexample, no enumeration needed.

    On this graph the triplet key does refine 1-WL, trivially, because 1-WL has
    only one class.  That is why it cannot also serve as the witness for
    :func:`refines` versus a class-count comparison; see
    :func:`witness_incomparable`.

    Returns:
        A fresh ``networkx.Graph`` on ``range(12)``.
    """
    prism = nx.circular_ladder_graph(3)
    bipartite = nx.complete_bipartite_graph(3, 3)
    graph = nx.disjoint_union(prism, bipartite)
    graph.remove_edge(0, 1)
    graph.remove_edge(6, 9)
    graph.add_edge(0, 6)
    graph.add_edge(1, 9)
    return graph


#: Edges of :func:`witness_incomparable`.  Found by a seeded sweep over
#: ``G(n, p)``, ``6 <= n <= 12``, and then frozen: the graph is a fixture, and
#: regenerating it from a search would make a regression test depend on a random
#: draw.
_INCOMPARABLE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 2),
    (0, 3),
    (0, 8),
    (1, 2),
    (1, 5),
    (1, 8),
    (2, 5),
    (2, 7),
    (3, 6),
    (3, 7),
    (4, 5),
    (4, 7),
    (4, 8),
    (6, 7),
    (6, 8),
)


def witness_incomparable() -> nx.Graph:
    """A graph where 1-WL and the triplet key are **incomparable** at equal counts.

    Connected, ``n = 9``, ``m = 15``.  Both partitions have four classes and
    neither refines the other.  This is the graph that separates :func:`refines`
    from a class-count comparison: every count-based rule -- ``|P| >= |Q|``,
    ``|P| == |Q| => equal`` -- concludes refinement in *both* directions, and
    exact containment holds in *neither*.

    It also strengthens the §1.3 finding.  On the cohort the two partitions were
    never incomparable (0/250) and on :func:`witness_prism_k33` they are ordered
    the opposite way from the plan's claim; here they are genuinely unordered,
    so no orientation of the claim survives.

    Returns:
        A fresh ``networkx.Graph`` on ``range(9)``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(9))
    graph.add_edges_from(_INCOMPARABLE_EDGES)
    return graph
