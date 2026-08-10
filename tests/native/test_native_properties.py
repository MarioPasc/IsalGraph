"""Mathematical properties of the encoding, proved THROUGH the cpp backend.

These are the acceptance criteria: round-trip, isomorphism invariance and
deduplication, canonicality, and idempotence.  Every call passes
``backend="cpp"`` explicitly -- a property that held only because the Python
reference happened to run would prove nothing about the port.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip(
    "isalgraph.core._native",
    reason="C++ extension not built",
    # exc_type is load-bearing. pytest only catches ModuleNotFoundError by
    # default, but a DELETED .so under a scikit-build-core editable install
    # still resolves through the import redirect and fails later with a plain
    # ImportError ("cannot open shared object file"). Without this the
    # extension-absent run dies at collection instead of skipping.
    exc_type=ImportError,
)

import graphs as G

from isalgraph.core import backends

STRUCTURED = G.structured_corpus()
RANDOM_U = G.random_corpus(220, max_n=8)
RANDOM_D = G.random_corpus(120, max_n=7, directed=True)
ALL_GRAPHS = [g for _, g in STRUCTURED] + RANDOM_U + RANDOM_D


def _safe_canonical(g, **kw):
    try:
        return backends.canonical_string(g, backend="cpp", **kw)
    except G.ENCODING_ERRORS:
        return None


# ----------------------------------------------------------------------
# Round-trip:  S2G(w) ~= S2G(G2S(S2G(w), v0))
# ----------------------------------------------------------------------


def _random_strings(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = "NnPpVvCcW"
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 22))) for _ in range(count)]


@pytest.mark.parametrize("directed", [False, True])
def test_round_trip_from_random_strings(directed: bool) -> None:
    checked = 0
    for w in _random_strings(700, seed=101 if directed else 100):
        g1 = backends.string_to_graph(w, directed, backend="cpp")
        for v0 in range(g1.node_count()):
            try:
                w2 = backends.graph_to_string(g1, v0, backend="cpp")
            except G.ENCODING_ERRORS:
                continue
            g2 = backends.string_to_graph(w2, directed, backend="cpp")
            assert g1.is_isomorphic(g2), (w, v0, w2)
            checked += 1
            break
    assert checked > 400, checked


def test_round_trip_from_structured_graphs() -> None:
    for name, g in STRUCTURED:
        for v0 in range(g.node_count()):
            try:
                w = backends.graph_to_string(g, v0, backend="cpp")
            except G.ENCODING_ERRORS:
                continue
            back = backends.string_to_graph(w, g.directed(), backend="cpp")
            assert g.is_isomorphic(back), (name, v0, w)


# ----------------------------------------------------------------------
# Isomorphism invariance and deduplication
# ----------------------------------------------------------------------


@pytest.mark.parametrize("pruned", [False, True])
def test_canonical_string_is_relabelling_invariant(pruned: bool) -> None:
    """A complete invariant must be blind to node numbering."""
    fn = backends.pruned_canonical_string if pruned else backends.canonical_string
    rng = random.Random(4242)
    checked = 0
    for name, g in STRUCTURED:
        n = g.node_count()
        try:
            base = fn(g, backend="cpp")
        except G.ENCODING_ERRORS:
            continue
        for _ in range(8):
            perm = list(range(n))
            rng.shuffle(perm)
            h = G.relabel(g, perm)
            assert fn(h, backend="cpp") == base, (name, perm)
            checked += 1
    assert checked > 200, checked


@pytest.mark.parametrize("pruned", [False, True])
def test_non_isomorphic_graphs_get_different_strings(pruned: bool) -> None:
    """Deduplication: equal canonical strings must imply isomorphism.

    Checked in the contrapositive over every pair of a structurally distinct
    family, using the reference isomorphism test as the ground truth.
    """
    fn = backends.pruned_canonical_string if pruned else backends.canonical_string
    family = [
        G.path(5),
        G.cycle(5),
        G.star(5),
        G.complete(5),
        G.grid(2, 3),
        G.path(6),
        G.cycle(6),
        G.star(6),
        G.random_tree(6, 1),
        G.random_tree(6, 2),
        G.barabasi_albert(6, 2, 0),
    ]
    strings = [fn(g, backend="cpp") for g in family]
    for i in range(len(family)):
        for j in range(i + 1, len(family)):
            same_string = strings[i] == strings[j]
            isomorphic = family[i].is_isomorphic(family[j])
            assert same_string == isomorphic, (i, j, strings[i], strings[j])


def test_equal_strings_decode_to_isomorphic_graphs() -> None:
    """The other direction, over a corpus with many collisions by construction.

    Buckets are keyed on ``(directed, string)``, not on the string alone.  The
    canonical string does NOT encode directedness: the 3-node directed path
    0->1->2 and the 3-node undirected path both canonicalise to the same
    instruction sequence, because C and c differ only in which endpoint the
    edge is written from and an undirected encoder never needs c.  The string
    is a complete invariant *within* a directedness class, and any
    deduplication over a mixed corpus must carry the flag alongside it.
    """
    buckets: dict[tuple[bool, str], list[object]] = {}
    for g in [G.relabel(gg, list(range(gg.node_count()))) for gg in ALL_GRAPHS[:200]]:
        w = _safe_canonical(g)
        if w is None:
            continue
        buckets.setdefault((g.directed(), w), []).append(g)
    collided = 0
    for members in buckets.values():
        for other in members[1:]:
            assert members[0].is_isomorphic(other)
            collided += 1
    assert collided > 0, "corpus produced no canonical-string collisions to check"


# ----------------------------------------------------------------------
# Canonicality:  lexmin among shortest over all starting nodes
# ----------------------------------------------------------------------


def test_canonical_is_lexmin_among_shortest_greedy_encodings() -> None:
    """The canonical string is never worse, by (length, lex), than the best
    greedy encoding from any starting node -- it minimises over a strict
    superset of the greedy execution paths."""
    for name, g in STRUCTURED:
        canon = _safe_canonical(g)
        if canon is None:
            continue
        greedy: list[str] = []
        for v in range(g.node_count()):
            try:
                greedy.append(backends.graph_to_string(g, v, backend="cpp"))
            except G.ENCODING_ERRORS:
                continue
        if not greedy:
            continue
        best_greedy = min(greedy, key=lambda w: (len(w), w))
        assert (len(canon), canon) <= (len(best_greedy), best_greedy), (name, canon, best_greedy)


def test_canonical_decodes_back_to_the_same_graph() -> None:
    """Canonicality is worthless if the string does not describe the graph."""
    for name, g in STRUCTURED:
        canon = _safe_canonical(g)
        if canon is None:
            continue
        decoded = backends.string_to_graph(canon, g.directed(), backend="cpp")
        assert g.is_isomorphic(decoded), (name, canon)


def test_pruned_agrees_with_unpruned_on_invariance() -> None:
    """The pruned form defines a different canonical string but must be an
    equally complete invariant."""
    rng = random.Random(31337)
    for name, g in STRUCTURED:
        try:
            a = backends.pruned_canonical_string(g, backend="cpp")
        except G.ENCODING_ERRORS:
            continue
        n = g.node_count()
        for _ in range(4):
            perm = list(range(n))
            rng.shuffle(perm)
            assert backends.pruned_canonical_string(G.relabel(g, perm), backend="cpp") == a, name


# ----------------------------------------------------------------------
# Idempotence:  canonical(S2G(canonical(G))) == canonical(G)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("pruned", [False, True])
def test_canonical_is_idempotent(pruned: bool) -> None:
    fn = backends.pruned_canonical_string if pruned else backends.canonical_string
    checked = 0
    for name, g in STRUCTURED:
        try:
            first = fn(g, backend="cpp")
        except G.ENCODING_ERRORS:
            continue
        decoded = backends.string_to_graph(first, g.directed(), backend="cpp")
        assert fn(decoded, backend="cpp") == first, (name, first)
        checked += 1
    assert checked > 30, checked


def test_idempotence_on_random_graphs() -> None:
    checked = 0
    for g in RANDOM_U[:120] + RANDOM_D[:60]:
        first = _safe_canonical(g)
        if first is None:
            continue
        decoded = backends.string_to_graph(first, g.directed(), backend="cpp")
        assert _safe_canonical(decoded) == first
        checked += 1
    assert checked > 100, checked


# ----------------------------------------------------------------------
# Threading must not change the answer
# ----------------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_threaded_search_returns_the_same_string(threads: int) -> None:
    """Parallelism is over the starting-node loop, which is embarrassingly
    parallel; the minimum is order-independent, so the result must be
    bit-identical to the serial run."""
    for _, g in STRUCTURED[:40]:
        serial = _safe_canonical(g, threads=1)
        if serial is None:
            continue
        assert backends.canonical_string(g, threads=threads, backend="cpp") == serial
        assert backends.pruned_canonical_string(
            g, threads=threads, backend="cpp"
        ) == backends.pruned_canonical_string(g, threads=1, backend="cpp")
