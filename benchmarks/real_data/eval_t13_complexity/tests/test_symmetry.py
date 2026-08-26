"""Tests for :mod:`benchmarks.real_data.eval_t13_complexity.symmetry`.

Four of these carry the ticket rather than the module:

- :func:`test_log10_aut_survives_k200` -- the overflow rule (`T-13-design.md` §3
  rule 4).  ``|Aut(K_200)| = 200! ~ 1e374``; a float product is ``inf``.
- :func:`test_triplet_partition_matches_frozen_reference` -- 600-graph parity
  against ``isalgraph.core.canonical_pruned.compute_structural_triplets``.
- :func:`test_proposition_1_holds_on_random_connected_graphs` -- the theorem the
  replacement claim rests on, over 2,000 random connected graphs.
- :func:`test_class_counts_are_not_a_refinement_test` -- the reason
  :func:`~...symmetry.refines` exists at all.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable

import networkx as nx
import pytest

from benchmarks.real_data.eval_t13_complexity import symmetry

#: Graphs drawn for the triplet-parity sweep.  Above the 500 the brief requires.
PARITY_GRAPH_COUNT = 600

#: Connected graphs drawn for the Proposition 1 property test.
PROPOSITION_1_GRAPH_COUNT = 2000

_SEED = 13


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _to_sparse_graph(graph: nx.Graph) -> object:
    """Convert an integer-labelled ``networkx`` graph to a ``SparseGraph``.

    The reference triplet routine takes a ``SparseGraph`` and indexes its
    output by node id, so the conversion must add nodes in ascending label
    order for the two partitions to be comparable key by key.

    Args:
        graph: a graph whose nodes are exactly ``range(n)``.

    Returns:
        The equivalent undirected ``SparseGraph``.
    """
    from isalgraph.core.sparse_graph import SparseGraph

    n = graph.number_of_nodes()
    assert set(graph.nodes()) == set(range(n)), "conversion needs range(n) labels"
    sparse = SparseGraph(n, False)
    for _ in range(n):
        sparse.add_node()
    for u, v in graph.edges():
        sparse.add_edge(u, v)
    return sparse


def _random_graphs(count: int, *, connected: bool, seed: int) -> list[nx.Graph]:
    """Deterministic draws of ``G(n, p)`` on ``range(n)``.

    Args:
        count: number of graphs to return.
        connected: keep only connected draws when ``True``; when ``False``,
            return whatever comes out, so that the triplet sweep also covers
            graphs with unreachable vertices.
        seed: RNG seed.

    Returns:
        Exactly *count* graphs.
    """
    rng = random.Random(seed)
    out: list[nx.Graph] = []
    while len(out) < count:
        n = rng.randint(3, 14)
        p = rng.uniform(0.1, 0.7)
        graph = nx.gnp_random_graph(n, p, seed=rng.randint(0, 2**31 - 1))
        if connected and not nx.is_connected(graph):
            continue
        out.append(graph)
    return out


# ----------------------------------------------------------------------------
# log10_aut
# ----------------------------------------------------------------------------


def test_log10_aut_survives_k200() -> None:
    """``|Aut(K_200)| = 200! ~ 1e374`` overflows a float product but not the log.

    This is `T-13-design.md` §3 rule 4 as an executable assertion.  The public
    ``competitors.backends.nauty.automorphism_group_size`` forms the product and
    returns ``inf`` here, which is exactly why :func:`symmetry.log10_aut` exists
    as a separate function rather than as a wrapper around it.
    """
    graph = nx.complete_graph(200)
    measured = symmetry.log10_aut(graph)
    assert math.isfinite(measured)
    assert measured == pytest.approx(math.log10(math.factorial(200)), abs=1e-6)
    assert measured > 308.0, "the point of the test is that the float product would overflow"


def test_log10_aut_product_form_raises_rather_than_returning_inf() -> None:
    """The forbidden formulation does not even return ``inf`` -- it raises.

    ``10.0 ** 374`` is an ``OverflowError``, not an infinity, so
    ``competitors.backends.nauty.automorphism_group_size`` **raises** on
    ``K_200`` rather than silently returning a useless number.  Measured, not
    assumed: the second assertion calls the public helper.  Inside T-13's own
    grid (``n <= 64``, so ``|Aut| <= 64! ~ 1.3e89``) the product would in fact
    survive, which is precisely why the rule has to be a rule and not a
    judgement call about which graphs are large enough to matter.
    """
    from isalgraph.competitors.backends.nauty import automorphism_group_size

    mantissa, exponent, _orbits, _nodes = symmetry._autgrp(nx.complete_graph(200))
    with pytest.raises(OverflowError):
        _ = mantissa * 10.0**exponent
    with pytest.raises(OverflowError):
        automorphism_group_size(nx.complete_graph(200))
    assert math.isfinite(math.log10(mantissa) + exponent)


@pytest.mark.parametrize(
    ("graph", "expected"),
    [
        (nx.empty_graph(0), 0.0),
        (nx.empty_graph(1), 0.0),
        (nx.path_graph(7), math.log10(2.0)),
        (nx.cycle_graph(9), math.log10(18.0)),
        (nx.star_graph(6), math.log10(math.factorial(6))),
        (nx.complete_graph(8), math.log10(math.factorial(8))),
        (nx.complete_bipartite_graph(4, 4), math.log10(2 * math.factorial(4) ** 2)),
        (nx.circular_ladder_graph(6), math.log10(24.0)),
    ],
)
def test_log10_aut_known_closed_forms(graph: nx.Graph, expected: float) -> None:
    """Textbook groups, to pin the mantissa/exponent unpacking."""
    assert symmetry.log10_aut(graph) == pytest.approx(expected, abs=1e-9)


def test_log10_aut_is_zero_for_a_rigid_graph() -> None:
    """A rigid graph has ``|Aut| = 1``, hence ``log10|Aut| = 0``, not ``-inf``."""
    # The smallest rigid connected graph: a triangle with a path of length 2
    # hanging off one vertex (n = 6, the "cricket"-like asymmetric tree/cycle mix).
    graph = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (0, 5)])
    graph.add_edge(1, 4)
    assert symmetry.log10_aut(graph) >= 0.0


# ----------------------------------------------------------------------------
# orbits
# ----------------------------------------------------------------------------


def test_orbits_are_dense_ids_from_zero() -> None:
    """Orbit ids must be ``0 .. n_orbits - 1`` with no gaps."""
    for graph in (nx.path_graph(7), nx.star_graph(5), nx.complete_graph(6)):
        ids = set(symmetry.orbits(graph).values())
        assert ids == set(range(len(ids)))


@pytest.mark.parametrize(
    ("graph", "n_orbits"),
    [
        (nx.path_graph(5), 3),  # two ends, two mid, one centre
        (nx.path_graph(6), 3),
        (nx.star_graph(5), 2),  # hub, leaves
        (nx.complete_graph(6), 1),  # vertex-transitive
        (nx.cycle_graph(7), 1),
        (nx.complete_bipartite_graph(3, 4), 2),
    ],
)
def test_orbits_known_counts(graph: nx.Graph, n_orbits: int) -> None:
    assert len(set(symmetry.orbits(graph).values())) == n_orbits


def test_orbits_of_empty_graph_is_empty() -> None:
    assert symmetry.orbits(nx.empty_graph(0)) == {}


def test_orbits_respect_arbitrary_node_labels() -> None:
    """Orbits are keyed by the caller's labels, not by nauty's indices."""
    graph = nx.relabel_nodes(nx.star_graph(4), {0: "hub", 1: "a", 2: "b", 3: "c", 4: "d"})
    orbit = symmetry.orbits(graph)
    assert set(orbit) == {"hub", "a", "b", "c", "d"}
    assert orbit["hub"] != orbit["a"]
    assert orbit["a"] == orbit["b"] == orbit["c"] == orbit["d"]


# ----------------------------------------------------------------------------
# wl_partition
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("graph", "n_classes"),
    [
        (nx.complete_graph(6), 1),
        (nx.cycle_graph(8), 1),
        (nx.path_graph(5), 3),
        (nx.star_graph(5), 2),
        (nx.complete_bipartite_graph(3, 3), 1),  # regular: 1-WL cannot see the parts
        (nx.complete_bipartite_graph(2, 5), 2),
    ],
)
def test_wl_partition_known_class_counts(graph: nx.Graph, n_classes: int) -> None:
    assert len(set(symmetry.wl_partition(graph).values())) == n_classes


def test_wl_partition_is_monotone_in_rounds() -> None:
    """Each round refines the previous one, and it reaches the stable partition."""
    graph = nx.gnp_random_graph(12, 0.35, seed=_SEED)
    stable = symmetry.wl_partition(graph)
    previous = symmetry.wl_partition(graph, rounds=0)
    assert len(set(previous.values())) == 1
    for r in range(1, 8):
        current = symmetry.wl_partition(graph, rounds=r)
        assert symmetry.refines(current, previous)
        previous = current
    assert symmetry.refines(previous, stable)
    assert symmetry.refines(stable, previous)


def test_wl_partition_is_label_invariant() -> None:
    """Relabelling the graph permutes the colouring; it does not change it."""
    graph = nx.gnp_random_graph(11, 0.4, seed=_SEED)
    mapping = {v: (v * 7 + 3) % 11 for v in graph}
    relabelled = nx.relabel_nodes(graph, mapping)
    original = symmetry.wl_partition(graph)
    permuted = symmetry.wl_partition(relabelled)
    assert {mapping[v]: c for v, c in original.items()} == permuted


def test_wl_partition_empty_graph() -> None:
    assert symmetry.wl_partition(nx.empty_graph(0)) == {}


def test_wl_partition_rejects_negative_rounds() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        symmetry.wl_partition(nx.path_graph(4), rounds=-1)


# ----------------------------------------------------------------------------
# triplet_partition -- parity with the frozen reference
# ----------------------------------------------------------------------------


def test_triplet_partition_matches_frozen_reference() -> None:
    """600 graphs, 0 disagreements with ``compute_structural_triplets``.

    The reference is the pure-Python one in ``isalgraph.core.canonical_pruned``,
    which is what ``pruned_canonical_string`` actually prunes with.  The sweep
    deliberately mixes connected and disconnected draws: a disconnected graph
    has vertices that never enter the BFS, and the two implementations must
    agree on *not counting* them rather than agreeing only where every vertex is
    reachable.
    """
    from isalgraph.core.canonical_pruned import compute_structural_triplets

    graphs = _random_graphs(PARITY_GRAPH_COUNT // 2, connected=True, seed=_SEED)
    graphs += _random_graphs(PARITY_GRAPH_COUNT - len(graphs), connected=False, seed=_SEED + 1)
    assert len(graphs) == PARITY_GRAPH_COUNT

    disagreements = 0
    for graph in graphs:
        mine = symmetry.triplet_partition(graph)
        reference = compute_structural_triplets(_to_sparse_graph(graph))  # type: ignore[arg-type]
        if [mine[v] for v in range(graph.number_of_nodes())] != list(reference):
            disagreements += 1
    assert disagreements == 0


def test_triplet_partition_matches_native_engine() -> None:
    """The same parity against the C++ engine, which is what a campaign runs.

    Skipped rather than failed when the extension is absent, because the
    package must remain importable on a pure-Python environment.
    """
    native = pytest.importorskip("isalgraph.core._native")
    from isalgraph.core.backends import _marshal

    for graph in _random_graphs(50, connected=True, seed=_SEED + 2):
        mine = symmetry.triplet_partition(graph)
        # The extension takes the marshalled fields, not the SparseGraph object.
        marshalled = _marshal(_to_sparse_graph(graph))  # type: ignore[arg-type]
        reference = native.compute_structural_triplets(*marshalled)
        assert [mine[v] for v in range(graph.number_of_nodes())] == [tuple(t) for t in reference]


@pytest.mark.parametrize(
    ("graph", "expected"),
    [
        (nx.star_graph(4), {0: (4, 0, 0), 1: (1, 3, 0)}),
        (nx.path_graph(5), {0: (1, 1, 1), 2: (2, 2, 0)}),
    ],
)
def test_triplet_partition_known_shells(
    graph: nx.Graph, expected: dict[int, tuple[int, int, int]]
) -> None:
    triplets = symmetry.triplet_partition(graph)
    for node, shells in expected.items():
        assert triplets[node] == shells


# ----------------------------------------------------------------------------
# refines
# ----------------------------------------------------------------------------


def test_refines_basic_relations() -> None:
    singletons: dict[Hashable, int] = {0: 0, 1: 1, 2: 2, 3: 3}
    halves: dict[Hashable, int] = {0: 0, 1: 0, 2: 1, 3: 1}
    trivial: dict[Hashable, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    assert symmetry.refines(singletons, halves)
    assert symmetry.refines(halves, trivial)
    assert symmetry.refines(singletons, trivial)
    assert not symmetry.refines(trivial, halves)
    assert not symmetry.refines(halves, singletons)
    assert symmetry.refines(halves, halves)


def test_refines_ignores_class_label_identity() -> None:
    """Only the induced partition matters, never the label values."""
    left: dict[Hashable, int] = {0: 7, 1: 7, 2: 9}
    right: dict[Hashable, str] = {0: "x", 1: "x", 2: "y"}
    assert symmetry.refines(left, right)
    assert symmetry.refines(right, left)


def test_refines_rejects_mismatched_domains() -> None:
    """A refinement relation between partitions of different sets is undefined."""
    with pytest.raises(ValueError, match="same set"):
        symmetry.refines({0: 0, 1: 0}, {0: 0, 1: 0, 2: 1})


def test_class_counts_are_not_a_refinement_test() -> None:
    """The separation criterion: counts agree, containment does not.

    On :func:`~...symmetry.witness_incomparable` the 1-WL and triplet partitions
    have the **same number of classes** and **neither refines the other**.  Every
    count-based rule -- ``|P| >= |Q| => P refines Q``, or ``|P| == |Q| => the
    partitions coincide`` -- therefore reports refinement in both directions and
    is wrong in both.  That is the substitution `corrections.md` §5 made when it
    inferred "provably coarser" from a class-count ratio.

    Note that the 3-prism/``K_{3,3}`` witness does *not* separate the two tests:
    there 1-WL has one class and the triplet key four, so the counts happen to
    give the right answer.  A rule that is right by accident on one graph and
    wrong on another is exactly the rule to remove.
    """
    graph = symmetry.witness_incomparable()
    wl = symmetry.wl_partition(graph)
    triplet = symmetry.triplet_partition(graph)

    n_wl = len(set(wl.values()))
    n_triplet = len(set(triplet.values()))
    assert n_wl == n_triplet == 4, "the counts must be equal for the separation to bite"

    count_rule_says_wl_refines_triplet = n_wl >= n_triplet
    count_rule_says_triplet_refines_wl = n_triplet >= n_wl
    assert count_rule_says_wl_refines_triplet is True
    assert count_rule_says_triplet_refines_wl is True

    assert symmetry.refines(wl, triplet) is False
    assert symmetry.refines(triplet, wl) is False


def test_witness_incomparable_is_connected_and_stable() -> None:
    graph = symmetry.witness_incomparable()
    assert graph.number_of_nodes() == 9
    assert graph.number_of_edges() == 15
    assert nx.is_connected(graph)


# ----------------------------------------------------------------------------
# The 3-prism / K_{3,3} witness
# ----------------------------------------------------------------------------


def test_witness_prism_k33_refutes_the_provably_coarser_claim() -> None:
    """`corrections.md` §5 item 4, refuted by one exact 12-vertex graph.

    Connected, 3-regular, ``n = 12``, ``m = 18``.  The stable 1-WL partition has
    **one** class and the triplet partition **four**, so 1-WL does not refine the
    triplet key: the two are not ordered the way the plan asserts.  No
    enumeration and no statistics are needed -- a single counterexample settles
    a universally quantified claim.
    """
    graph = symmetry.witness_prism_k33()

    assert nx.is_connected(graph)
    assert graph.number_of_nodes() == 12
    assert graph.number_of_edges() == 18
    assert {d for _v, d in graph.degree()} == {3}, "must be 3-regular"

    record = symmetry.resolution_record(graph)
    assert record["n_wl_classes"] == 1
    assert record["n_triplet_classes"] == 4
    assert record["wl_refines_triplet"] is False
    assert record["triplet_refines_wl"] is True


# ----------------------------------------------------------------------------
# resolution_record
# ----------------------------------------------------------------------------


def test_resolution_record_has_exactly_the_frozen_fields() -> None:
    """``schema.py`` copies these verbatim, so an extra key is a schema break."""
    record = symmetry.resolution_record(nx.path_graph(6))
    assert tuple(record) == symmetry.RESOLUTION_FIELDS
    assert set(record) == {
        "log10_aut",
        "n_orbits",
        "max_orbit_size",
        "n_wl_classes",
        "n_triplet_classes",
        "wl_refines_triplet",
        "triplet_refines_wl",
        "wl_equals_orbits",
        "triplet_equals_orbits",
    }


def test_resolution_record_agrees_with_its_parts() -> None:
    graph = nx.circular_ladder_graph(5)
    record = symmetry.resolution_record(graph)
    orbit = symmetry.orbits(graph)
    assert record["log10_aut"] == pytest.approx(symmetry.log10_aut(graph))
    assert record["n_orbits"] == len(set(orbit.values()))
    assert record["max_orbit_size"] == max(
        sum(1 for x in orbit.values() if x == c) for c in set(orbit.values())
    )
    assert record["n_wl_classes"] == len(set(symmetry.wl_partition(graph).values()))
    assert record["n_triplet_classes"] == len(set(symmetry.triplet_partition(graph).values()))


def test_resolution_record_on_a_rigid_graph_hits_the_floor() -> None:
    """A rigid graph has singleton orbits, so any invariant that separates all
    vertices attains the floor and the record says so."""
    graph = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (2, 5)])
    record = symmetry.resolution_record(graph)
    if record["log10_aut"] == pytest.approx(0.0):
        assert record["n_orbits"] == graph.number_of_nodes()
        assert record["max_orbit_size"] == 1


def test_resolution_record_of_empty_graph() -> None:
    record = symmetry.resolution_record(nx.empty_graph(0))
    assert record["n_orbits"] == 0
    assert record["max_orbit_size"] == 0
    assert record["log10_aut"] == 0.0


# ----------------------------------------------------------------------------
# Proposition 1 -- the gate
# ----------------------------------------------------------------------------


def test_proposition_1_holds_on_random_connected_graphs() -> None:
    """No node invariant is finer than the orbit partition, over 2,000 graphs.

    ``refines(orbits(g), P)`` asserts that every orbit lies inside one ``P``
    class, i.e. that ``P`` never splits an orbit.  Both 1-WL and the triplet key
    are node invariants, so Proposition 1 says this must hold for both on every
    graph.  A violation is not a flaky test: it would mean one of the two
    partitions is not a graph invariant, which would invalidate the pruning
    argument in ``canonical_pruned.py``.  The brief's instruction on a violation
    is to stop and report, not to loosen the assertion.
    """
    graphs = _random_graphs(PROPOSITION_1_GRAPH_COUNT, connected=True, seed=_SEED + 3)
    assert len(graphs) == PROPOSITION_1_GRAPH_COUNT
    assert all(3 <= g.number_of_nodes() <= 14 for g in graphs)

    for graph in graphs:
        orbit = symmetry.orbits(graph)
        assert symmetry.refines(orbit, symmetry.wl_partition(graph)), (
            f"1-WL split an orbit on {sorted(graph.edges())}"
        )
        assert symmetry.refines(orbit, symmetry.triplet_partition(graph)), (
            f"the triplet key split an orbit on {sorted(graph.edges())}"
        )


def test_proposition_1_holds_on_the_witnesses() -> None:
    for graph in (symmetry.witness_prism_k33(), symmetry.witness_incomparable()):
        orbit = symmetry.orbits(graph)
        assert symmetry.refines(orbit, symmetry.wl_partition(graph))
        assert symmetry.refines(orbit, symmetry.triplet_partition(graph))


# ----------------------------------------------------------------------------
# Dependency contract
# ----------------------------------------------------------------------------


def test_symmetry_module_needs_neither_grakel_nor_numpy() -> None:
    """``grakel`` is unusable on Picasso under numpy 2, so 1-WL is self-contained."""
    import inspect

    source = inspect.getsource(symmetry)
    assert "grakel" not in source.replace("no ``grakel``", "").replace("``grakel``", "")
    assert "import numpy" not in source
