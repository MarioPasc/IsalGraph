"""Track B: the two canonical-labelling backends and the canonicalised sparse6.

Acceptance criteria 1, 2, 4, 6, 7 and the suite-scope rule live here; the
AGM brute-force oracle and the ceiling table live in ``test_agm_cam.py``.

Criterion 6 is **not** written the way the brief specifies, and the reason
is measured rather than argued -- see
:func:`test_inverted_canon_label_fails_f3` and
:func:`test_isomorphism_assertion_cannot_catch_the_inversion`.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import networkx as nx

from isalgraph.competitors import bits as bits_module
from isalgraph.competitors import datasets, fixtures
from isalgraph.competitors.base import Budget, Capability, ReprBackend
from isalgraph.competitors.registry import get_repr_backend, registered_backends
from isalgraph.errors import BackendUnavailableError, CompetitorError, SuiteScopeError

pytestmark = pytest.mark.unit

pynauty = pytest.importorskip("pynauty")
nx = pytest.importorskip("networkx")

from isalgraph.competitors.backends import agm as agm_module  # noqa: E402
from isalgraph.competitors.backends import nauty as nauty_module  # noqa: E402

# --------------------------------------------------------------------------
# The frozen targets.  Every one was confirmed by hand on 2026-08-15 before
# any of this module existed, so a failure here is a fault in the port.
# --------------------------------------------------------------------------

RUNNING_EXAMPLE_NAUTY = "E@ro"
MINUS_EDGE_NAUTY = "E@po"
K33_NAUTY = "Es\\o"
PRISM_NAUTY = "E{Sw"

RUNNING_EXAMPLE_AGM = "000001110011110"
MINUS_EDGE_AGM = "000001011111000"
K33_AGM = "000111111011100"
PRISM_AGM = "001101110111100"

#: The strict upper triangle of the running example under its **incident**
#: labelling, read column-wise -- what ``adjacency.symbols`` must equal and
#: what graph6 ``'ElCW'`` unpacks to (CONTRACTS.md §9, wave-0 finding 2).
RUNNING_EXAMPLE_IDENTITY_TRIANGLE = "101101000100011"
MINUS_EDGE_IDENTITY_TRIANGLE = "101001000100011"


def _graph(name: str) -> nx.Graph:
    return fixtures.to_networkx(fixtures.ALL_FIXTURES[name])


@pytest.fixture
def nauty_backend() -> ReprBackend:
    return get_repr_backend("nauty_graph6")


@pytest.fixture
def agm_backend() -> ReprBackend:
    return get_repr_backend("agm_cam")


# ==========================================================================
# Criterion 1 -- the running example reproduces exactly
# ==========================================================================


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("running_example", RUNNING_EXAMPLE_NAUTY),
        ("running_example_minus_edge", MINUS_EDGE_NAUTY),
    ],
)
def test_nauty_running_example(nauty_backend: ReprBackend, fixture: str, expected: str) -> None:
    assert nauty_backend.encode(_graph(fixture)).text == expected


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("running_example", RUNNING_EXAMPLE_AGM),
        ("running_example_minus_edge", MINUS_EDGE_AGM),
    ],
)
def test_agm_running_example(agm_backend: ReprBackend, fixture: str, expected: str) -> None:
    assert agm_backend.encode(_graph(fixture)).text == expected


def test_running_example_shape() -> None:
    graph = _graph("running_example")
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (6, 7)


def test_automorphism_group_size_of_the_running_example() -> None:
    """``|Aut(G)| = 4``.  Free from ``pynauty.autgrp``; T-13 needs it."""
    assert nauty_module.automorphism_group_size(_graph("running_example")) == pytest.approx(4.0)


def test_automorphism_group_size_of_known_graphs() -> None:
    assert nauty_module.automorphism_group_size(_graph("k33")) == pytest.approx(72.0)
    assert nauty_module.automorphism_group_size(_graph("prism")) == pytest.approx(12.0)
    assert nauty_module.automorphism_group_size(nx.complete_graph(6)) == pytest.approx(720.0)


def test_automorphism_orbits_are_exposed() -> None:
    """Orbits exist for AGM's optional pruning; four orbits on the example."""
    orbits = nauty_module.automorphism_orbits(_graph("running_example"))
    assert len(set(orbits)) == 4


# ==========================================================================
# Criterion 2 -- K3,3 vs the triangular prism
# ==========================================================================


def test_k33_and_prism_are_the_witness_pair() -> None:
    k33, prism = _graph("k33"), _graph("prism")
    for graph in (k33, prism):
        assert graph.number_of_nodes() == 6
        assert nx.is_connected(graph)
        assert sorted(d for _, d in graph.degree()) == [3] * 6
    assert not nx.is_isomorphic(k33, prism)


def test_nauty_separates_k33_from_the_prism(nauty_backend: ReprBackend) -> None:
    assert nauty_backend.encode(_graph("k33")).text == K33_NAUTY
    assert nauty_backend.encode(_graph("prism")).text == PRISM_NAUTY
    assert K33_NAUTY != PRISM_NAUTY


def test_agm_separates_k33_from_the_prism(agm_backend: ReprBackend) -> None:
    assert agm_backend.encode(_graph("k33")).text == K33_AGM
    assert agm_backend.encode(_graph("prism")).text == PRISM_AGM
    assert K33_AGM != PRISM_AGM


def test_certificates_separate_k33_from_the_prism() -> None:
    """``pynauty.certificate`` is used for isomorphism assertions only."""
    assert nauty_module.certificate(_graph("k33")) != nauty_module.certificate(_graph("prism"))


# ==========================================================================
# Criterion 7 -- the reading order
# ==========================================================================


def test_agm_on_the_identity_permutation_is_the_column_wise_triangle() -> None:
    """AGM's code order, evaluated at the identity, is the frozen order.

    In an isolated wave-1 worktree agent A's ``adjacency`` module does not
    exist, so the assertion is against the literal that CONTRACTS.md §9
    fixes and that graph6 ``'ElCW'`` unpacks to.  The cross-backend form is
    :func:`test_agm_agrees_with_agent_a_adjacency`, which skips until the
    branches merge.
    """
    for name, expected in (
        ("running_example", RUNNING_EXAMPLE_IDENTITY_TRIANGLE),
        ("running_example_minus_edge", MINUS_EDGE_IDENTITY_TRIANGLE),
    ):
        assert agm_module.identity_code(_graph(name)) == expected


def test_identity_code_pins_the_labelling_against_insertion_order() -> None:
    """The labelling is pinned by rebuilding, not by renaming.

    ``nx.convert_node_labels_to_integers(ordering="sorted")`` renames node
    values and leaves insertion order alone; ``to_graph6_bytes`` re-derives
    its labelling from insertion order, so the two disagree on almost every
    scrambled graph.  ``identity_code`` therefore rebuilds, and a graph
    carrying the same labels in a scrambled insertion order must give the
    same answer.
    """
    graph = _graph("running_example")
    scrambled = nx.Graph()
    scrambled.add_nodes_from([4, 1, 5, 0, 3, 2])
    scrambled.add_edges_from(sorted(graph.edges(), reverse=True))
    assert list(scrambled.nodes()) != sorted(scrambled.nodes())
    assert agm_module.identity_code(scrambled) == RUNNING_EXAMPLE_IDENTITY_TRIANGLE
    # And the unpinned reading really is order-dependent, so the pinning is
    # doing work rather than restating what networkx already guarantees.
    adjacency, n = agm_module._adjacency_sets(scrambled)
    assert agm_module._code_from_perm(adjacency, list(range(n))) != (
        RUNNING_EXAMPLE_IDENTITY_TRIANGLE
    )


def test_agm_identity_code_equals_the_graph6_payload_of_the_same_labelling() -> None:
    """The family identity, executable without agent A's modules.

    ``adjacency``, graph6's payload and AGM at the identity permutation are
    the *same bit sequence*.  Here the graph6 side is produced from the
    incident labelling, not the canonical one, so this is the reading-order
    check rather than a canonicity check.
    """
    for name in fixtures.ALL_FIXTURES:
        graph = _graph(name)
        n = graph.number_of_nodes()
        pinned = nx.Graph()
        pinned.add_nodes_from(sorted(graph.nodes()))
        pinned.add_edges_from(graph.edges())
        wire = nx.to_graph6_bytes(pinned, header=False).strip()
        payload = "".join(nauty_module.graph6_payload_bits(wire, n))
        assert agm_module.identity_code(graph) == payload, name


def test_agm_frame_pairs_match_the_code_order() -> None:
    """``frame.pairs[k]`` is the cell ``frame.bits[k]`` reports."""
    graph = _graph("running_example")
    encoding = get_repr_backend("agm_cam").encode(graph)
    frame = encoding.frame
    assert frame is not None
    assert frame.pairs[:6] == ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))
    rebuilt = agm_module.code_to_graph("".join(frame.bits), frame.n_nodes)
    for k, (i, j) in enumerate(frame.pairs):
        assert rebuilt.has_edge(i, j) == (frame.bits[k] == "1")


def test_nauty_frame_carries_triangle_bits_not_symbols() -> None:
    """graph6's symbols are six-bit bytes; its frame is the bit triangle.

    Zipping ``pairs`` against ``symbols`` would compare a byte to a cell,
    which is why ``PositionalFrame`` carries its own ``bits``.
    """
    encoding = get_repr_backend("nauty_graph6").encode(_graph("running_example"))
    frame = encoding.frame
    assert frame is not None
    assert len(encoding.symbols) == 4
    assert len(frame.bits) == 15
    assert "".join(frame.bits) == "000001110011110"


@pytest.mark.parametrize("name", sorted(fixtures.ALL_FIXTURES))
def test_nauty_frame_bits_describe_the_canonical_graph(name: str) -> None:
    graph = _graph(name)
    encoding = get_repr_backend("nauty_graph6").encode(graph)
    frame = encoding.frame
    assert frame is not None
    rebuilt = nx.Graph()
    rebuilt.add_nodes_from(range(frame.n_nodes))
    rebuilt.add_edges_from(
        pair for pair, bit in zip(frame.pairs, frame.bits, strict=True) if bit == "1"
    )
    assert nx.is_isomorphic(graph, rebuilt)


def test_agm_agrees_with_agent_a_adjacency() -> None:
    """Cross-backend form of criterion 7.  Closed by the orchestrator at merge."""
    adjacency_backend = pytest.importorskip(
        "isalgraph.competitors.backends.adjacency",
        reason="agent A's module is absent in an isolated wave-1 worktree",
    )
    del adjacency_backend
    backend = get_repr_backend("adjacency")
    rng = random.Random(42)
    for name in fixtures.ALL_FIXTURES:
        for graph in (_graph(name), fixtures.shuffled_copy(_graph(name), rng)):
            assert "".join(backend.encode(graph).symbols) == agm_module.identity_code(graph), name


# ==========================================================================
# Criterion 6 -- the inversion guard, and what it can actually catch
# ==========================================================================


def _relabel_with_inverted_label(graph: nx.Graph) -> nx.Graph:
    """:func:`canonical_relabel` with the ``canon_label`` direction reversed.

    ``lab[i]`` is the OLD vertex at NEW position ``i``.  Using it directly
    as ``old -> new`` is the documented trap.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    index = {v: i for i, v in enumerate(nodes)}
    adjacency = {i: [index[w] for w in graph.neighbors(v)] for i, v in enumerate(nodes)}
    labelling = pynauty.canon_label(pynauty.Graph(n, directed=False, adjacency_dict=adjacency))
    position = {i: int(labelling[i]) for i in range(n)}  # WRONG DIRECTION
    out = nx.Graph()
    out.add_nodes_from(range(n))
    out.add_edges_from((position[index[u]], position[index[v]]) for u, v in graph.edges())
    return out


def _graph6(graph: nx.Graph) -> str:
    return nx.to_graph6_bytes(graph, header=False).strip().decode()


def test_inverted_canon_label_fails_f3() -> None:
    """**The brief's premise is wrong: the inversion fails F3, loudly.**

    ``competitors/nauty.md`` §1 states the inverted labelling *"will pass an
    invariance test and be wrong"*.  It does not.  For ``G' = G^tau`` nauty
    returns ``lab_{G'} = pi_G^{-1} tau``, so the wrong-direction image is
    ``G^{tau pi_G^{-1} tau}``, which depends on ``tau``.

    Measured here: the correct labelling gives one code over 20 genuine
    relabellings of every connected fixture; the inverted one gives many.

    ``path_2`` is exempt and the exemption is mathematics, not a weakened
    assertion: ``|Aut(K2)| = 2 = 2!``, so *every* labelling of it produces
    ``'A_'`` and no labelling scheme can be distinguished on it.  The guard
    below states that condition rather than hard-coding the exception.
    """
    import math

    rng = random.Random(42)
    checked = 0
    for name in fixtures.CONNECTED_FIXTURES:
        graph = _graph(name)
        correct: set[str] = set()
        inverted: set[str] = set()
        for _ in range(20):
            copy = fixtures.shuffled_copy(graph, rng)
            correct.add(_graph6(nauty_module.canonical_relabel(copy)))
            inverted.add(_graph6(_relabel_with_inverted_label(copy)))
        assert len(correct) == 1, f"the CORRECT labelling is not invariant on {name}"
        if math.factorial(graph.number_of_nodes()) == nauty_module.automorphism_group_size(graph):
            continue  # every labelling gives one code; nothing to distinguish
        checked += 1
        assert len(inverted) > 1, (
            f"the inverted labelling was invariant on {name}; if this ever holds, "
            f"F3 has stopped being the guard that catches the inversion"
        )
    assert checked >= 4


@pytest.mark.slow
def test_inverted_canon_label_fails_f3_on_random_graphs() -> None:
    """The same finding at scale: 30 of 30 random ``n = 8`` graphs.

    ``nauty.md`` §1's claim that the inversion *"will pass an invariance
    test"* is refuted with a margin, not marginally.
    """
    rng = random.Random(7)
    non_invariant = 0
    for _ in range(30):
        graph = nx.gnm_random_graph(8, 12, seed=rng.randrange(10**6))
        correct: set[str] = set()
        inverted: set[str] = set()
        for _ in range(15):
            copy = fixtures.shuffled_copy(graph, rng)
            correct.add(_graph6(nauty_module.canonical_relabel(copy)))
            inverted.add(_graph6(_relabel_with_inverted_label(copy)))
        assert len(correct) == 1
        non_invariant += len(inverted) > 1
    assert non_invariant == 30


def test_isomorphism_assertion_cannot_catch_the_inversion() -> None:
    """``nx.is_isomorphic(G, relabelled)`` is vacuous against the inversion.

    Any bijective relabelling of ``G`` is isomorphic to ``G`` by
    construction, so the assertion the brief prescribes holds for *every*
    permutation, correct or inverted.  This test exists to pin that, so no
    later ticket re-adds the assertion believing it guards the inversion.
    """
    rng = random.Random(7)
    for _ in range(20):
        graph = nx.gnm_random_graph(8, 12, seed=rng.randrange(10**6))
        wrong = _relabel_with_inverted_label(graph)
        assert nx.is_isomorphic(graph, wrong)


def test_isomorphism_guard_catches_a_broken_index_map() -> None:
    """What the guard *does* catch: a non-bijective vertex map.

    ``pynauty`` indexes vertices ``0..n-1`` while a ``networkx`` graph may
    carry arbitrary labels, so a wrong index map is the realistic fault.
    :func:`canonical_relabel` refuses it before any encode is emitted.
    """
    graph = _graph("running_example")
    original = pynauty.canon_label

    def broken(pg: object) -> list[int]:
        del pg
        return [0, 0, 1, 2, 3, 4]  # not a permutation

    pynauty.canon_label = broken
    try:
        with pytest.raises(CompetitorError, match="not a permutation"):
            nauty_module.canonical_relabel(graph)
    finally:
        pynauty.canon_label = original


def test_canonical_relabel_verify_flag_is_honoured() -> None:
    """``verify=False`` skips the VF2 call and returns the same graph.

    It exists for the language-matched Fig. 2 timing mode: ``nx.is_isomorphic``
    costs 6.7 ms at ``n = 96`` against 0.33 ms for the relabelling itself.
    """
    graph = nx.gnm_random_graph(20, 40, seed=3)
    fast = nauty_module.canonical_relabel(graph, verify=False)
    slow = nauty_module.canonical_relabel(graph, verify=True)
    assert sorted(fast.edges()) == sorted(slow.edges())


# ==========================================================================
# Canonicity, reversibility, capabilities
# ==========================================================================


@pytest.mark.parametrize("backend_name", ["nauty_graph6", "agm_cam"])
@pytest.mark.parametrize("name", sorted(fixtures.ALL_FIXTURES))
def test_backends_are_invariant_on_fixtures(backend_name: str, name: str) -> None:
    backend = get_repr_backend(backend_name)
    graph = _graph(name)
    rng = random.Random(42)
    codes = {backend.encode(graph).text}
    for _ in range(20):
        codes.add(backend.encode(fixtures.shuffled_copy(graph, rng)).text)
    assert len(codes) == 1


@pytest.mark.parametrize("backend_name", ["nauty_graph6", "agm_cam"])
@pytest.mark.parametrize("name", sorted(fixtures.ALL_FIXTURES))
def test_backends_are_reversible(backend_name: str, name: str) -> None:
    backend = get_repr_backend(backend_name)
    graph = _graph(name)
    assert nx.is_isomorphic(graph, backend.decode(backend.encode(graph)))


def test_disconnected_fixture_encodes() -> None:
    """Both backends declare ``HANDLES_DISCONNECTED`` and both must honour it."""
    graph = _graph("c4_plus_k3_disjoint")
    for name in ("nauty_graph6", "agm_cam"):
        backend = get_repr_backend(name)
        assert nx.is_isomorphic(graph, backend.decode(backend.encode(graph)))


def test_isolated_vertices_encode() -> None:
    graph = _graph("empty_3")
    for name in ("nauty_graph6", "agm_cam"):
        encoding = get_repr_backend(name).encode(graph)
        assert encoding.n_nodes == 3
        assert encoding.n_edges == 0


def test_empty_graph_encodes() -> None:
    graph = nx.Graph()
    for name in ("nauty_graph6", "agm_cam"):
        encoding = get_repr_backend(name).encode(graph)
        assert encoding.n_nodes == 0


def test_declared_capabilities() -> None:
    expected_nauty = {
        Capability.POSITIONAL_FRAME,
        Capability.CANONICAL,
        Capability.COMPLETE_INVARIANT,
        Capability.REVERSIBLE,
        Capability.HANDLES_DISCONNECTED,
    }
    assert get_repr_backend("nauty_graph6").capabilities == frozenset(expected_nauty)
    assert get_repr_backend("agm_cam").capabilities == frozenset(
        expected_nauty | {Capability.SUITE1_ONLY}
    )
    assert Capability.SUITE1_ONLY not in get_repr_backend("nauty_graph6").capabilities


def test_both_names_register_from_one_module() -> None:
    names = registered_backends()
    assert "nauty_graph6" in names
    assert "sparse6_nauty" in names
    assert "agm_cam" in names


def test_encode_accepts_arbitrary_node_labels() -> None:
    """The realistic index-map fault: nodes that are not ``0..n-1``."""
    graph = nx.relabel_nodes(_graph("running_example"), {i: f"v{i}" for i in range(6)})
    assert get_repr_backend("nauty_graph6").encode(graph).text == RUNNING_EXAMPLE_NAUTY
    assert get_repr_backend("agm_cam").encode(graph).text == RUNNING_EXAMPLE_AGM


# ==========================================================================
# Bit accounting -- measured, never computed
# ==========================================================================


def test_nauty_bit_count_is_identical_to_graph6_by_construction() -> None:
    """Canonicalisation permutes bits; it does not change how many there are."""
    for name in fixtures.ALL_FIXTURES:
        graph = _graph(name)
        n = graph.number_of_nodes()
        incident = nx.to_graph6_bytes(graph, header=False).strip()
        encoding = get_repr_backend("nauty_graph6").encode(graph)
        assert encoding.wire is not None
        assert len(encoding.wire) == len(incident), name
        counted = get_repr_backend("nauty_graph6").bits(encoding)
        assert counted.entropy_bits == 6.0 * len(incident)
        assert counted.realised_bits == 8 * len(incident)
        assert counted.payload_bits == n * (n - 1) // 2


def test_nauty_wire_carries_no_header_and_no_newline() -> None:
    encoding = get_repr_backend("nauty_graph6").encode(_graph("running_example"))
    assert encoding.wire == b"E@ro"
    assert b"\n" not in (encoding.wire or b"")
    assert not (encoding.wire or b"").startswith(b">>graph6<<")


def test_graph6_closed_form_holds_inside_its_range_only() -> None:
    """``1 + ceil(n(n-1)/12)`` is a **test oracle**, valid to ``n = 62``."""
    import math

    for n in (2, 6, 12, 40, 62):
        graph = nx.gnm_random_graph(n, min(2 * n, n * (n - 1) // 2), seed=n)
        encoding = get_repr_backend("nauty_graph6").encode(graph)
        assert encoding.wire is not None
        assert len(encoding.wire) == 1 + math.ceil(n * (n - 1) / 12)


def test_graph6_prefix_branch_is_live_above_62() -> None:
    """Suite 2 reaches ``n = 98``, where ``N(n)`` is four bytes."""
    assert nauty_module.graph6_prefix_bytes(62) == 1
    assert nauty_module.graph6_prefix_bytes(63) == 4
    assert nauty_module.graph6_prefix_bytes(98) == 4
    graph = nx.gnm_random_graph(98, 190, seed=5)
    encoding = get_repr_backend("nauty_graph6").encode(graph)
    frame = encoding.frame
    assert frame is not None
    assert len(frame.bits) == 98 * 97 // 2
    assert sum(bit == "1" for bit in frame.bits) == graph.number_of_edges()


def test_agm_bit_count_is_the_adjacency_row() -> None:
    """``n(n-1)/2``, identical to adjacency.  AGM adds nothing to Claim A."""
    for name in fixtures.ALL_FIXTURES:
        graph = _graph(name)
        n = graph.number_of_nodes()
        backend = get_repr_backend("agm_cam")
        counted = backend.bits(backend.encode(graph))
        assert counted.entropy_bits == float(n * (n - 1) // 2)
        assert counted.payload_bits == n * (n - 1) // 2


def test_agm_is_not_charged_eight_bits_per_character() -> None:
    """``'000001...'`` is a debugging view, not a serialisation.

    Charging it at eight bits per character inflates the ``n^2`` family
    eightfold and hands us a baseline we beat for free.  Equality is
    admissible only for a payload of eight bits or fewer, where one byte is
    both the packing and the rendering.
    """
    for name in fixtures.ALL_FIXTURES:
        graph = _graph(name)
        backend = get_repr_backend("agm_cam")
        encoding = backend.encode(graph)
        if not encoding.text:
            continue
        realised = backend.bits(encoding).realised_bits
        assert realised <= 8 * len(encoding.text), name
        if len(encoding.text) > 8:
            assert realised < 8 * len(encoding.text), name


def test_nauty_realised_bits_are_the_emitted_bytes() -> None:
    """For graph6 the text *is* the wire, so ``8 * len`` is correct, not inflated.

    The 8x-inflation trap applies to bit-string renderings; asserting
    ``realised < 8 * len(text)`` here would be asserting that graph6 is
    stored in fewer than eight bits per emitted ASCII byte.
    """
    for name in fixtures.ALL_FIXTURES:
        graph = _graph(name)
        backend = get_repr_backend("nauty_graph6")
        encoding = backend.encode(graph)
        assert encoding.wire is not None
        assert encoding.text == encoding.wire.decode("ascii")
        counted = backend.bits(encoding)
        assert counted.realised_bits == 8 * len(encoding.wire)
        assert counted.entropy_bits == 6.0 * len(encoding.wire)


def test_agm_realised_bits_match_the_frozen_formula() -> None:
    """``8*ceil(n(n-1)/16)``, and never below the payload it stores."""
    import math

    for name in fixtures.ALL_FIXTURES:
        graph = _graph(name)
        n = graph.number_of_nodes()
        backend = get_repr_backend("agm_cam")
        counted = backend.bits(backend.encode(graph))
        assert counted.realised_bits == 8 * math.ceil(n * (n - 1) / 16), name
        assert counted.entropy_bits <= counted.realised_bits, name
        assert counted.realised_bits < counted.entropy_bits + 8, name


def test_bits_module_is_the_only_producer() -> None:
    """Neither backend overrides ``ReprBackend.bits``."""
    assert agm_module.AGMBackend.bits is ReprBackend.bits
    assert nauty_module.NautyGraph6Backend.bits is ReprBackend.bits
    assert nauty_module.Sparse6NautyBackend.bits is ReprBackend.bits
    assert bits_module.count is not None


# ==========================================================================
# Suite scope -- enforced, not documented
# ==========================================================================


def test_agm_refuses_above_suite_1() -> None:
    graph = nx.gnm_random_graph(agm_module.SUITE1_MAX_NODES + 1, 20, seed=1)
    with pytest.raises(SuiteScopeError, match="Suite-1 only"):
        get_repr_backend("agm_cam").encode(graph)


def test_agm_accepts_the_largest_suite_1_graph() -> None:
    graph = nx.path_graph(agm_module.SUITE1_MAX_NODES)
    assert len(get_repr_backend("agm_cam").encode(graph).symbols) == 12 * 11 // 2


def test_agm_suite_1_bound_admits_every_suite_1_graph() -> None:
    """``SUITE1_MAX_NODES = 12`` is Suite 1's true maximum, on AIDS."""
    if "aids" not in datasets.available_datasets():
        pytest.skip("Suite-1 cohorts not on this workstation")
    for name in datasets.SUITE1:
        cohort = datasets.load(name)
        assert max(g.number_of_nodes() for g in cohort.graphs) <= agm_module.SUITE1_MAX_NODES


def test_nauty_has_no_suite_restriction() -> None:
    graph = nx.gnm_random_graph(98, 190, seed=2)
    assert get_repr_backend("nauty_graph6").encode(graph).n_nodes == 98


def test_agm_budget_is_frozen() -> None:
    assert agm_module.SUITE1_NODE_BUDGET == 200_000
    assert agm_module.SUITE2_NODE_BUDGET == 100_000
    assert agm_module.DEFAULT_NODE_BUDGET == 200_000


def test_agm_budget_override_is_honoured() -> None:
    """A tiny budget raises rather than returning the greedy incumbent."""
    from isalgraph.errors import AGMBudgetExceeded

    graph = nx.complete_graph(9)
    with pytest.raises(AGMBudgetExceeded, match="NOT canonical"):
        get_repr_backend("agm_cam").encode(graph, budget=Budget(search_nodes=10))


# ==========================================================================
# sparse6_nauty -- the one cross-edge
# ==========================================================================


def _sparse6_available() -> bool:
    try:
        import isalgraph.competitors.backends.sparse6  # noqa: F401
    except ImportError:
        return False
    return True


def test_sparse6_nauty_registers_even_without_agent_a() -> None:
    """Registration must succeed; only invocation may fail."""
    assert "sparse6_nauty" in registered_backends()


def test_sparse6_nauty_reports_unavailable_without_agent_a() -> None:
    if _sparse6_available():
        pytest.skip("agent A's sparse6 module is present")
    with pytest.raises(BackendUnavailableError):
        get_repr_backend("sparse6_nauty")


@pytest.mark.skipif(not _sparse6_available(), reason="agent A's sparse6.py is not in this worktree")
def test_sparse6_serialise_matches_the_frozen_signature() -> None:
    """The cross-edge is resolved by ``cast``; a cast can hide a mismatch.

    CONTRACTS.md §4 freezes ``serialise(graph: nx.Graph) -> Encoding`` as a
    module-level function, so the conformance is asserted rather than
    assumed.
    """
    import inspect

    from isalgraph.competitors.base import Encoding

    serialise = nauty_module._sparse6_serialise()
    signature = inspect.signature(serialise)
    assert len(signature.parameters) == 1
    encoding = serialise(_graph("running_example"))
    assert isinstance(encoding, Encoding)
    assert encoding.backend == "sparse6"


@pytest.mark.skipif(not _sparse6_available(), reason="agent A's sparse6.py is not in this worktree")
def test_sparse6_nauty_is_canonical() -> None:
    backend = get_repr_backend("sparse6_nauty")
    rng = random.Random(42)
    for name in fixtures.CONNECTED_FIXTURES:
        graph = _graph(name)
        codes = {backend.encode(graph).text}
        for _ in range(20):
            codes.add(backend.encode(fixtures.shuffled_copy(graph, rng)).text)
        assert len(codes) == 1, name


@pytest.mark.skipif(not _sparse6_available(), reason="agent A's sparse6.py is not in this worktree")
def test_sparse6_nauty_carries_its_own_provenance() -> None:
    encoding = get_repr_backend("sparse6_nauty").encode(_graph("running_example"))
    assert encoding.backend == "sparse6_nauty"
    assert Capability.POSITIONAL_FRAME not in get_repr_backend("sparse6_nauty").capabilities
    assert encoding.frame is None


# ==========================================================================
# Criterion 4 -- F3 on the real cohort, 50 graphs x 20 relabellings, seed 42
# ==========================================================================


def _f3(backend: ReprBackend, graphs: list[nx.Graph], seed: int) -> int:
    rng = random.Random(seed)
    invariant = 0
    for graph in graphs:
        try:
            codes = {backend.encode(graph).text}
            for _ in range(20):
                codes.add(backend.encode(fixtures.shuffled_copy(graph, rng)).text)
        except CompetitorError:
            continue
        if len(codes) == 1:
            invariant += 1
    return invariant


@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(datasets.SUITE1))
@pytest.mark.parametrize("backend_name", ["nauty_graph6", "agm_cam"])
def test_f3_on_the_real_cohort(dataset: str, backend_name: str) -> None:
    """50 / 50 on every Suite-1 dataset, for both canonical backends."""
    if dataset not in datasets.available_datasets():
        pytest.skip(f"cohort {dataset!r} not on this workstation")
    cohort = datasets.load(dataset)
    graphs = [cohort.graphs[i] for i in cohort.sample(200, seed=42)][:50]
    assert _f3(get_repr_backend(backend_name), graphs, 42) == len(graphs)


@pytest.mark.slow
def test_the_f3_harness_can_fail() -> None:
    """An F3 harness that cannot fail is worthless.

    Plain graph6 on the *incident* labelling must fail on the same cohort
    and the same relabeller, or the 50/50 above measures nothing.
    """
    if "iam_letter_low" not in datasets.available_datasets():
        pytest.skip("cohort not on this workstation")
    cohort = datasets.load("iam_letter_low")
    graphs = [cohort.graphs[i] for i in cohort.sample(200, seed=42)][:50]
    rng = random.Random(42)
    invariant = 0
    for graph in graphs:
        codes = {_graph6(graph)}
        for _ in range(20):
            codes.add(_graph6(fixtures.shuffled_copy(graph, rng)))
        invariant += len(codes) == 1
    assert invariant < len(graphs), "the relabeller cannot make graph6 fail"
