"""Differential parity: the C++ engine against the frozen Python reference.

Every check here drives the specific function through the specific backend.
A harness that merely proves the extension imported would report PASS while
computing nothing, so each test calls ``backend="cpp"`` explicitly and
compares against the reference module imported directly.

Byte-exactness is the criterion for every function.  The canonical searches
minimise over their whole candidate set and are therefore order-independent;
the greedy encoder is order-dependent and is byte-exact only because
``backends._marshal`` preserves CPython's set-iteration order across the FFI.
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
from isalgraph.core.canonical import canonical_string as ref_canonical
from isalgraph.core.canonical import levenshtein as ref_levenshtein
from isalgraph.core.canonical_pruned import compute_structural_triplets as ref_triplets
from isalgraph.core.canonical_pruned import pruned_canonical_string as ref_pruned
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph
from isalgraph.types import VALID_INSTRUCTIONS

ALPHABET = "".join(sorted(VALID_INSTRUCTIONS))

# Sized so the exhaustive PYTHON reference stays tractable -- the very cost
# this port exists to lift.  Measured on this workstation, the reference costs
# ~0.4 ms at n=4, ~3 ms at n=5, ~18 ms at n=6, ~87 ms at n=7 and ~1.2 s at
# n=8, so a uniform spread over 2..8 would spend 95% of the wall clock on 5%
# of the corpus.  The distribution below buys >3,000 comparisons in roughly a
# minute of reference time while still covering the expensive tail.
_RANDOM_UNDIRECTED = G.sized_corpus(
    {2: 300, 3: 400, 4: 500, 5: 500, 6: 400, 7: 120, 8: 25}, seed0=0
)
_RANDOM_DIRECTED = G.sized_corpus(
    {2: 150, 3: 200, 4: 200, 5: 150, 6: 60, 7: 20}, directed=True, seed0=90_000
)
_STRUCTURED = G.structured_corpus()


def _corpus() -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = [(f"struct:{name}", g) for name, g in _STRUCTURED]
    items += [(f"rand_u:{i}", g) for i, g in enumerate(_RANDOM_UNDIRECTED)]
    items += [(f"rand_d:{i}", g) for i, g in enumerate(_RANDOM_DIRECTED)]
    return items


CORPUS = _corpus()


def test_corpus_meets_the_acceptance_size() -> None:
    assert len(CORPUS) >= 3000, len(CORPUS)


# ----------------------------------------------------------------------
# canonical_string / pruned_canonical_string -- byte-exact
# ----------------------------------------------------------------------


def test_canonical_string_is_byte_exact() -> None:
    mismatches: list[tuple[str, str, str]] = []
    compared = 0
    for label, g in CORPUS:
        try:
            expected = ref_canonical(g)  # type: ignore[arg-type]
        except G.ENCODING_ERRORS:
            continue  # the error paths have their own parity test
        got = backends.canonical_string(g, backend="cpp")  # type: ignore[arg-type]
        compared += 1
        if got != expected:
            mismatches.append((label, expected, got))
    assert compared >= 3000, compared
    assert not mismatches, f"{len(mismatches)}/{compared} mismatched: {mismatches[:5]}"


def test_pruned_canonical_string_is_byte_exact() -> None:
    mismatches: list[tuple[str, str, str]] = []
    compared = 0
    for label, g in CORPUS:
        try:
            expected = ref_pruned(g)  # type: ignore[arg-type]
        except G.ENCODING_ERRORS:
            continue
        got = backends.pruned_canonical_string(g, backend="cpp")  # type: ignore[arg-type]
        compared += 1
        if got != expected:
            mismatches.append((label, expected, got))
    assert compared >= 3000, compared
    assert not mismatches, f"{len(mismatches)}/{compared} mismatched: {mismatches[:5]}"


def test_structural_triplets_are_byte_exact() -> None:
    from isalgraph.core import _native as ext

    for label, g in CORPUS[:600]:
        n, max_nodes, directed, edges, adjacency = backends._marshal(g)  # noqa: SLF001
        got = ext.compute_structural_triplets(n, max_nodes, directed, edges, adjacency)
        assert got == ref_triplets(g), label  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# graph_to_string (greedy) -- byte-exact, the order-dependent path
# ----------------------------------------------------------------------


def test_greedy_graph_to_string_is_byte_exact() -> None:
    """The claim §2.3 exists to support: identical strings, not merely
    identical lengths or isomorphic decodings."""
    mismatches: list[tuple[str, int, str, str]] = []
    compared = 0
    for label, g in CORPUS:
        for v in range(g.node_count()):  # type: ignore[attr-defined]
            try:
                expected, _ = GraphToString(g).run(v)  # type: ignore[arg-type]
            except G.ENCODING_ERRORS:
                continue
            got = backends.graph_to_string(g, v, backend="cpp")  # type: ignore[arg-type]
            compared += 1
            if got != expected:
                mismatches.append((label, v, expected, got))
    assert compared >= 3000, compared
    assert not mismatches, f"{len(mismatches)}/{compared} mismatched: {mismatches[:5]}"


def test_greedy_parity_survives_relabelling() -> None:
    """Relabelling permutes CPython's set slots, so it is the sharpest probe
    of the ordered-marshalling contract."""
    rng = random.Random(20260810)
    compared = 0
    for _, g in _STRUCTURED:
        n = g.node_count()
        for _ in range(6):
            perm = list(range(n))
            rng.shuffle(perm)
            h = G.relabel(g, perm)
            for v in range(n):
                try:
                    expected, _ = GraphToString(h).run(v)
                except G.ENCODING_ERRORS:
                    continue
                assert backends.graph_to_string(h, v, backend="cpp") == expected
                compared += 1
    assert compared > 500, compared


# ----------------------------------------------------------------------
# string_to_graph -- identical graphs
# ----------------------------------------------------------------------


def _random_strings(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    for _ in range(count):
        length = rng.randint(0, 24)
        out.append("".join(rng.choice(ALPHABET) for _ in range(length)))
    return out


HANDWRITTEN = [
    "",
    "W",
    "V",
    "v",
    "C",
    "c",
    "N",
    "P",
    "n",
    "p",
    "VV",
    "Vv",
    "vV",
    "VVV",
    "VNV",
    "VPV",
    "VnV",
    "VpV",
    "VC",
    "Vc",
    "VVC",
    "VVc",
    "VVNC",
    "VVPC",
    "VVnC",
    "VVpC",
    "CC",  # duplicate self edge on the initial node
    "cc",
    "NNNN",  # pointer walks with a single-element list
    "PPPP",
    "nnnn",
    "pppp",
    "WWWW",
    "VWNWVWCW",
    "VVVVNNPPCcnp",
    "V" * 20,
    "Vv" * 12,
    "VNVNVNVNC",
]


@pytest.mark.parametrize("directed", [False, True])
def test_string_to_graph_produces_identical_graphs(directed: bool) -> None:
    strings = HANDWRITTEN + _random_strings(1100, seed=7 if directed else 3)
    for s in strings:
        expected, _ = StringToGraph(s, directed).run()
        got = backends.string_to_graph(s, directed, backend="cpp")
        assert G.graphs_equal(expected, got), (directed, s)
        assert got.node_count() == expected.node_count(), s
        assert got.edge_count() == expected.edge_count(), s
        assert got.logical_edge_count() == expected.logical_edge_count(), s


def test_string_to_graph_corpus_meets_the_acceptance_size() -> None:
    assert len(HANDWRITTEN) + 1100 >= 1000
    assert 2 * (len(HANDWRITTEN) + 1100) >= 2000


def test_string_to_graph_adjacency_iterates_identically() -> None:
    """Beyond set equality: the replayed insertion order must reproduce
    CPython's slot layout, or a decoded graph fed to the greedy encoder would
    diverge from the reference."""
    for s in HANDWRITTEN + _random_strings(400, seed=11):
        for directed in (False, True):
            expected, _ = StringToGraph(s, directed).run()
            got = backends.string_to_graph(s, directed, backend="cpp")
            for u in range(expected.node_count()):
                assert list(expected.neighbors(u)) == list(got.neighbors(u)), (s, directed, u)


def test_decoded_graphs_encode_identically() -> None:
    """End-to-end consequence of the previous test."""
    for s in _random_strings(300, seed=13):
        expected, _ = StringToGraph(s, False).run()
        got = backends.string_to_graph(s, False, backend="cpp")
        for v in range(expected.node_count()):
            try:
                a, _ = GraphToString(expected).run(v)
            except G.ENCODING_ERRORS:
                continue
            assert backends.graph_to_string(got, v, backend="cpp") == a


# ----------------------------------------------------------------------
# levenshtein -- exact integer equality
# ----------------------------------------------------------------------


def test_levenshtein_matches_on_random_pairs() -> None:
    rng = random.Random(20260811)
    for _ in range(4000):
        a = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 30)))
        b = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 30)))
        assert backends.levenshtein(a, b, backend="cpp") == ref_levenshtein(a, b), (a, b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("", ""),
        ("", "abc"),
        ("abc", ""),
        ("abc", "abc"),
        ("kitten", "sitting"),
        ("flaw", "lawn"),
        ("V" * 200, "V" * 199 + "C"),
    ],
)
def test_levenshtein_edge_cases(a: str, b: str) -> None:
    assert backends.levenshtein(a, b, backend="cpp") == ref_levenshtein(a, b)


# ----------------------------------------------------------------------
# distances
# ----------------------------------------------------------------------


def test_graph_distance_matches_reference() -> None:
    from isalgraph.core.canonical import graph_distance as ref_gd
    from isalgraph.core.canonical_pruned import pruned_graph_distance as ref_pgd

    sample = [g for _, g in CORPUS[:120]]
    for i in range(0, len(sample) - 1, 2):
        g1, g2 = sample[i], sample[i + 1]
        try:
            expected = ref_gd(g1, g2)  # type: ignore[arg-type]
            expected_p = ref_pgd(g1, g2)  # type: ignore[arg-type]
        except G.ENCODING_ERRORS:
            continue
        assert backends.graph_distance(g1, g2, backend="cpp") == expected  # type: ignore[arg-type]
        assert (
            backends.pruned_graph_distance(g1, g2, backend="cpp") == expected_p  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------
# Both engines through the dispatcher agree
# ----------------------------------------------------------------------


def test_dispatcher_backends_agree() -> None:
    for _, g in CORPUS[:400]:
        try:
            py = backends.canonical_string(g, backend="python")  # type: ignore[arg-type]
        except G.ENCODING_ERRORS:
            continue
        assert backends.canonical_string(g, backend="cpp") == py  # type: ignore[arg-type]
