"""Track A: the three serialisation backends that need only ``networkx``.

The three are in the pool as **controls**.  ``adjacency`` and ``graph6`` are
deliberately weak on canonicity and deliberately strong on bits; their
failure to be isomorphism-invariant is the finding, not a defect in the
code.  ``sparse6`` is IsalGraph's only genuine rival on message length.
**If these backends make IsalGraph look good, there is a bug**, so several
tests below assert against IsalGraph's interest -- most sharply
:func:`test_claim_a_medians_reproduce_readme`, which pins the row saying
IsalGraph is shorter than the adjacency matrix on 0.0 % of Letter graphs.

Two numbers in the T-04 brief do **not** reproduce, and both are recorded
here as executable evidence rather than tuned away:

* ``bits.py`` computes the adjacency realised cost as
  ``8 ceil(n(n-1)/2 / 16)``, but the design note and CONTRACTS §5 specify
  ``8 ceil(n(n-1)/16)``.  Those differ by a factor of two --
  :func:`test_adjacency_realised_bits_match_the_frozen_closed_form` carries
  it as a strict xfail.
* README §4.3's ``sparse6`` column counts the ``':'`` in the entropy bound;
  ``bits.py`` excludes it, per the design note.  Every dataset is therefore
  exactly six bits lower.  :func:`test_claim_a_medians_reproduce_readme`
  asserts both halves so the delta is provably the prefix and nothing else.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import math
import pathlib
import random
import statistics

import pytest

from isalgraph.competitors import datasets, fixtures
from isalgraph.competitors.backends import adjacency as adjacency_mod
from isalgraph.competitors.backends import graph6 as graph6_mod
from isalgraph.competitors.backends import sparse6 as sparse6_mod
from isalgraph.competitors.base import Capability, Encoding, ReprBackend
from isalgraph.competitors.registry import get_repr_backend

# After the imports, deliberately: every module above is importable with
# networkx absent -- that is the subpackage's dependency contract, and
# importing them here would be a poor test of it if the skip fired first.
nx = pytest.importorskip("networkx")

TRACK_A = ("adjacency", "graph6", "sparse6")

#: Sizes the round-trip and length tests must cover.  ``63`` and ``98``
#: exercise graph6's four-byte ``N(n)`` header; ``16``, ``32`` and ``64``
#: exercise sparse6's power-of-two ``k = ceil(log2 n)`` special case.
BOUNDARY_SIZES = (2, 15, 16, 32, 62, 63, 64, 98)


def backend(name: str) -> ReprBackend:
    """The registered backend called *name*."""
    return get_repr_backend(name)


def cycle(n: int) -> nx.Graph:
    """``C_n`` for ``n >= 3``, ``P_2`` for ``n == 2``."""
    return nx.cycle_graph(n) if n >= 3 else nx.path_graph(n)


def random_graph(rng: random.Random, *, lo: int = 2, hi: int = 20) -> nx.Graph:
    """A ``G(n, m)`` draw with a scrambled insertion order."""
    n = rng.randint(lo, hi)
    m = rng.randint(0, n * (n - 1) // 2)
    base = nx.gnm_random_graph(n, m, seed=rng.randint(0, 10**6))
    order = list(base.nodes())
    rng.shuffle(order)
    out = nx.Graph()
    out.add_nodes_from(order)
    edges = list(base.edges())
    rng.shuffle(edges)
    out.add_edges_from(edges)
    return out


# --------------------------------------------------------------------------- #
# Criterion 1 -- the running example reproduces exactly
# --------------------------------------------------------------------------- #

#: The frozen running-example strings.  The adjacency literals are the
#: **column-wise** triangle; the row-major ones the brief originally quoted
#: were corrected on 2026-08-15 and are wrong.
RUNNING_EXAMPLE_EXPECTED = {
    "adjacency": ("101101000100011", "101001000100011"),
    "graph6": ("ElCW", "EhCW"),
    "sparse6": (":EaWIzR", ":EaYms"),
}


@pytest.mark.parametrize("name", TRACK_A)
def test_running_example_strings(name: str) -> None:
    """``G = C4(0,1,2,3) + K3(3,4,5)`` and ``H = G - (0,3)``, exactly."""
    expected_g, expected_h = RUNNING_EXAMPLE_EXPECTED[name]
    backend_obj = backend(name)
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    minus = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE_MINUS_EDGE)
    assert backend_obj.encode(graph).text == expected_g
    assert backend_obj.encode(minus).text == expected_h


def test_running_example_shape() -> None:
    """``n = 6``, ``m = 7``; sparse6 is 7 bytes for ``G`` and 6 for ``H``."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    encoding = backend("sparse6").encode(graph)
    assert (encoding.n_nodes, encoding.n_edges) == (6, 7)
    assert encoding.wire is not None
    assert len(encoding.wire) == 7
    minus = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE_MINUS_EDGE)
    minus_wire = backend("sparse6").encode(minus).wire
    assert minus_wire is not None
    assert len(minus_wire) == 6


def test_graph6_wire_carries_no_trailing_newline() -> None:
    """``to_graph6_bytes`` appends one; leaving it costs eight realised bits."""
    for name in ("graph6", "sparse6"):
        for fixture in fixtures.ALL_FIXTURES.values():
            encoding = backend(name).encode(fixtures.to_networkx(fixture))
            assert encoding.wire is not None
            assert not encoding.wire.endswith(b"\n")
            assert "\n" not in encoding.text


# --------------------------------------------------------------------------- #
# Criterion 6 -- the family identity, and the reading order it pins
# --------------------------------------------------------------------------- #


def test_adjacency_reading_order_is_column_wise() -> None:
    """``a(0,1) a(0,2) a(1,2) a(0,3) ...`` -- not row-major."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    frame = backend("adjacency").encode(graph).frame
    assert frame is not None
    assert frame.pairs[:6] == ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))
    row_major = tuple((i, j) for i in range(frame.n_nodes) for j in range(i + 1, frame.n_nodes))
    assert frame.pairs != row_major, "column-wise and row-major must not coincide at n=6"


@pytest.mark.parametrize("fixture_name", sorted(fixtures.ALL_FIXTURES))
def test_family_identity_on_fixtures(fixture_name: str) -> None:
    """``adjacency.symbols`` equals graph6's unpacked payload, bit for bit."""
    graph = fixtures.to_networkx(fixtures.ALL_FIXTURES[fixture_name])
    triangle = backend("adjacency").encode(graph)
    packed = backend("graph6").encode(graph)
    assert packed.wire is not None
    unpacked = graph6_mod.unpack_payload(packed.wire, packed.n_nodes)
    assert triangle.symbols == unpacked
    frame = packed.frame
    assert frame is not None
    assert frame.bits == unpacked, "graph6's frame carries triangle bits, not its bytes"


def test_family_identity_on_random_graphs() -> None:
    """The identity holds across sizes, densities and insertion orders."""
    rng = random.Random(42)
    for _ in range(300):
        graph = random_graph(rng)
        triangle = backend("adjacency").encode(graph)
        packed = backend("graph6").encode(graph)
        assert packed.wire is not None
        assert triangle.symbols == graph6_mod.unpack_payload(packed.wire, packed.n_nodes)


@pytest.mark.parametrize("n", BOUNDARY_SIZES)
def test_family_identity_at_boundary_sizes(n: int) -> None:
    """Including ``n = 63`` and ``n = 98``, where ``N(n)`` is four bytes."""
    graph = cycle(n)
    triangle = backend("adjacency").encode(graph)
    packed = backend("graph6").encode(graph)
    assert packed.wire is not None
    assert graph6_mod.header_length(packed.wire) == (1 if n <= 62 else 4)
    assert triangle.symbols == graph6_mod.unpack_payload(packed.wire, n)


# --------------------------------------------------------------------------- #
# Determinism is not invariance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", TRACK_A)
def test_encoding_is_independent_of_insertion_order(name: str) -> None:
    """Scrambling insertion order while holding labels fixed changes nothing.

    ``competitors/graph6.md`` §7 prescribes
    ``nx.convert_node_labels_to_integers(G, ordering="sorted")`` for this,
    but that call relabels values and leaves the insertion order, and
    ``to_graph6_bytes`` re-derives its own labels from the insertion order.
    :func:`test_convert_node_labels_alone_does_not_pin_the_labelling`
    records the size of the gap.
    """
    rng = random.Random(11)
    for _ in range(100):
        base = nx.gnm_random_graph(
            rng.randint(2, 15), rng.randint(0, 20), seed=rng.randint(0, 10**6)
        )
        order = list(base.nodes())
        rng.shuffle(order)
        scrambled = nx.Graph()
        scrambled.add_nodes_from(order)
        edges = list(base.edges())
        rng.shuffle(edges)
        scrambled.add_edges_from(edges)
        assert backend(name).encode(base).text == backend(name).encode(scrambled).text


def test_convert_node_labels_alone_does_not_pin_the_labelling() -> None:
    """The brief's prescription is insufficient, and this measures by how much.

    Reported, not worked around: the fix lives in
    :func:`isalgraph.competitors.backends.adjacency.normalised`.
    """
    rng = random.Random(7)
    disagreements = 0
    trials = 300
    for _ in range(trials):
        scrambled = random_graph(rng)
        via_convert = nx.to_graph6_bytes(
            nx.convert_node_labels_to_integers(scrambled, ordering="sorted"), header=False
        ).rstrip(b"\n")
        via_rebuild = nx.to_graph6_bytes(adjacency_mod.normalised(scrambled), header=False).rstrip(
            b"\n"
        )
        disagreements += via_convert != via_rebuild
    assert disagreements > trials // 2, (
        "expected the two normalisations to disagree on most scrambled graphs; "
        "if this ever drops to zero, networkx changed its writer and "
        "adjacency.normalised can be simplified"
    )


# --------------------------------------------------------------------------- #
# Criterion 2 -- F3, and a harness that can fail
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", TRACK_A)
def test_relabeller_can_make_the_backend_fail(name: str) -> None:
    """**An F3 harness that cannot fail is worthless.**

    ``nx.relabel_nodes(copy=True)`` alone preserves insertion order and
    makes order-dependent formats look invariant (finding 13).
    ``fixtures.shuffled_copy`` rebuilds the graph, and this asserts that it
    genuinely breaks all three non-canonical formats.
    """
    rng = random.Random(3)
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    codes = {backend(name).encode(graph).text}
    for _ in range(20):
        codes.add(backend(name).encode(fixtures.shuffled_copy(graph, rng)).text)
    assert len(codes) > 1, f"{name} looked invariant; the relabeller is broken"


@pytest.mark.parametrize("name", TRACK_A)
def test_relabel_nodes_copy_alone_would_hide_the_failure(name: str) -> None:
    """The trap itself, made executable: the wrong relabeller finds nothing."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    rng = random.Random(3)
    codes = {backend(name).encode(graph).text}
    for _ in range(20):
        nodes = list(graph.nodes())
        new = list(range(len(nodes)))
        rng.shuffle(new)
        # relabel_nodes(copy=True) preserves the ORIGINAL insertion order,
        # so this only permutes label values -- which our normalisation
        # then re-sorts away.  It cannot detect an order-dependent format.
        codes.add(
            backend(name)
            .encode(nx.relabel_nodes(graph, dict(zip(nodes, new, strict=True)), copy=True))
            .text
        )
    assert len(codes) > 1, "sanity: even this weak relabeller changes labels"


@pytest.mark.parametrize("name", TRACK_A)
def test_complete_graphs_are_the_only_f3_successes(name: str) -> None:
    """``Aut(K_n) = S_n``, so ``K_n`` is invariant; nothing else in Letter is.

    This is the whole explanation for the non-zero Letter F3 counts.
    ``competitors/graph6.md`` §2 attributes them to "20 draws can miss every
    distinguishable labelling"; measured exhaustively over all ``n!``
    relabellings of the 50-graph Letter draws, the sampled count and the
    exhaustive count are **equal** (6, 5 and 9 of 50), and every invariant
    graph is a complete graph.  The successes are structural, not sampling.
    """
    rng = random.Random(5)
    for n in (3, 4, 5):
        complete = nx.complete_graph(n)
        codes = {backend(name).encode(complete).text}
        for _ in range(20):
            codes.add(backend(name).encode(fixtures.shuffled_copy(complete, rng)).text)
        assert len(codes) == 1, f"K_{n} must be invariant under relabelling"

    for fixture_name in ("running_example", "k33", "prism"):
        graph = fixtures.to_networkx(fixtures.ALL_FIXTURES[fixture_name])
        codes = {backend(name).encode(graph).text}
        for _ in range(20):
            codes.add(backend(name).encode(fixtures.shuffled_copy(graph, rng)).text)
        assert len(codes) > 1, f"{fixture_name} is not complete and must not be invariant"


def test_exhaustive_f3_over_all_relabellings_of_the_running_example() -> None:
    """122 of 720 distinct graph6 codes: not invariant, and not close."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    codes = set()
    for perm in itertools.permutations(range(6)):
        relabelled = nx.Graph()
        relabelled.add_nodes_from(range(6))
        relabelled.add_edges_from((perm[u], perm[v]) for u, v in graph.edges())
        codes.add(backend("graph6").encode(relabelled).text)
    # |Aut(G)| = 4, so at most 720/4 = 180 distinct labelled codes exist.
    assert 1 < len(codes) <= 180
    assert Capability.CANONICAL not in backend("graph6").capabilities


@pytest.mark.slow
@pytest.mark.parametrize("dataset", datasets.SUITE1)
@pytest.mark.parametrize("name", TRACK_A)
def test_f3_on_the_real_cohort(dataset: str, name: str) -> None:
    """50 graphs x 20 relabellings, seed 42, per Suite-1 dataset.

    **The exact per-dataset count is stream-dependent and is not the claim.**
    The claim is the contrast: all three land in single digits out of 50,
    i.e. they fail F3 as controls, while a canonical backend would be 50/50.
    Measured on this draw: Letter LOW 6, MED 5, HIGH 9, LINUX 0, AIDS 0 --
    identical for all three backends, because all three serialise the same
    normalised labelling.
    """
    pytest.importorskip("numpy")
    try:
        cohort = datasets.load(dataset)
    except datasets.DatasetNotFoundError as exc:
        pytest.skip(str(exc))
    graphs = [cohort.graphs[i] for i in cohort.sample(200, seed=42)][:50]
    rng = random.Random(42)
    backend_obj = backend(name)
    invariant = 0
    for graph in graphs:
        codes = {backend_obj.encode(graph).text}
        for _ in range(20):
            codes.add(backend_obj.encode(fixtures.shuffled_copy(graph, rng)).text)
        invariant += len(codes) == 1
    assert invariant <= 10, (
        f"{name} on {dataset}: {invariant}/50 invariant. These are controls; a "
        f"double-digit count means the relabeller stopped changing the labelling"
    )
    if dataset in ("linux", "aids"):
        assert invariant == 0, "the representative F3 result is 0/50 on the larger graphs"


# --------------------------------------------------------------------------- #
# Criterion 5 -- round trip, exactly, not up to isomorphism
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", BOUNDARY_SIZES)
@pytest.mark.parametrize("name", TRACK_A)
def test_round_trip_at_boundary_sizes(name: str, n: int) -> None:
    """``decode(encode(G))`` is **edge-set identical**, not merely isomorphic."""
    graph = cycle(n)
    backend_obj = backend(name)
    recovered = backend_obj.decode(backend_obj.encode(graph))
    assert recovered.number_of_nodes() == n
    assert set(map(frozenset, recovered.edges())) == set(map(frozenset, graph.edges()))


@pytest.mark.parametrize("name", TRACK_A)
def test_round_trip_on_fixtures(name: str) -> None:
    """Including the disconnected fixture and the three isolated vertices."""
    backend_obj = backend(name)
    for fixture in fixtures.ALL_FIXTURES.values():
        graph = fixtures.to_networkx(fixture)
        recovered = backend_obj.decode(backend_obj.encode(graph))
        assert recovered.number_of_nodes() == graph.number_of_nodes()
        assert set(map(frozenset, recovered.edges())) == set(map(frozenset, graph.edges()))


@pytest.mark.parametrize("name", TRACK_A)
def test_round_trip_on_random_graphs(name: str) -> None:
    """Densities from empty to complete, sizes 2..20, 300 draws."""
    rng = random.Random(2026)
    backend_obj = backend(name)
    for _ in range(300):
        graph = adjacency_mod.normalised(random_graph(rng))
        recovered = backend_obj.decode(backend_obj.encode(graph))
        assert set(map(frozenset, recovered.edges())) == set(map(frozenset, graph.edges()))
        assert recovered.number_of_nodes() == graph.number_of_nodes()


@pytest.mark.parametrize("n", (16, 32, 64))
def test_sparse6_round_trip_at_powers_of_two(n: int) -> None:
    """``k = ceil(log2 n)`` has an off-by-one special case at ``n = 2^k``.

    Asserted by round trip on every encode rather than by trusting a length
    formula, per ``competitors/sparse6.md`` §7.
    """
    rng = random.Random(n)
    backend_obj = backend("sparse6")
    for _ in range(20):
        graph = nx.gnm_random_graph(n, rng.randint(0, 3 * n), seed=rng.randint(0, 10**6))
        encoding = backend_obj.encode(graph)
        recovered = backend_obj.decode(encoding)
        assert set(map(frozenset, recovered.edges())) == set(map(frozenset, graph.edges()))
        assert recovered.number_of_nodes() == n


@pytest.mark.parametrize("name", TRACK_A)
def test_decode_rejects_a_foreign_encoding(name: str) -> None:
    """A decoder never guesses at another backend's wire."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    foreign = backend("graph6" if name != "graph6" else "sparse6").encode(graph)
    with pytest.raises(ValueError, match="is from"):
        backend(name).decode(foreign)


# --------------------------------------------------------------------------- #
# graph6's four-byte header: the closed form is a test oracle, not production
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", (2, 3, 15, 16, 32, 62))
def test_graph6_closed_form_holds_up_to_62(n: int) -> None:
    """``1 + ceil(n(n-1)/12)`` bytes -- valid **only** in this range."""
    encoding = backend("graph6").encode(cycle(n))
    assert encoding.wire is not None
    assert len(encoding.wire) == 1 + math.ceil(n * (n - 1) / 12)


@pytest.mark.parametrize("n", (63, 64, 98))
def test_graph6_closed_form_fails_above_62(n: int) -> None:
    """The live branch: ``N(n)`` costs four bytes, so the closed form is short.

    Suite 2 reaches ``n = 98``.  This is why ``bits.py`` measures ``wire``.
    """
    encoding = backend("graph6").encode(cycle(n))
    assert encoding.wire is not None
    closed_form = 1 + math.ceil(n * (n - 1) / 12)
    assert len(encoding.wire) == closed_form + 3, "N(n) is '~' plus three bytes"
    assert graph6_mod.header_length(encoding.wire) == 4
    counted = backend("graph6").bits(encoding)
    assert counted.entropy_bits == 6.0 * len(encoding.wire)
    assert counted.entropy_bits != 6.0 * closed_form


def test_graph6_payload_bits_are_recorded_separately() -> None:
    """Claim A's two conventions are not recoverable from each other."""
    for n in BOUNDARY_SIZES:
        encoding = backend("graph6").encode(cycle(n))
        assert encoding.wire is not None
        counted = backend("graph6").bits(encoding)
        assert counted.payload_bits == n * (n - 1) // 2
        assert counted.realised_bits == 8 * len(encoding.wire)
        assert counted.entropy_bits == 6.0 * len(encoding.wire)


# --------------------------------------------------------------------------- #
# Criterion 4 -- no eightfold inflation, and no reader of Encoding.text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", sorted(fixtures.ALL_FIXTURES))
def test_adjacency_bits_are_not_inflated_eightfold(fixture_name: str) -> None:
    """``'101001...'`` is a debugging view; it is never eight bits a character."""
    graph = fixtures.to_networkx(fixtures.ALL_FIXTURES[fixture_name])
    encoding = backend("adjacency").encode(graph)
    counted = backend("adjacency").bits(encoding)
    assert counted.entropy_bits == encoding.n_nodes * (encoding.n_nodes - 1) / 2
    assert counted.realised_bits < 8 * len(encoding.text) or len(encoding.text) <= 1
    assert not counted.inflated


@pytest.mark.parametrize(
    "fixture_name",
    [k for k, v in sorted(fixtures.ALL_FIXTURES.items()) if v[0] >= 5],
)
def test_adjacency_realised_bits_beat_one_bit_per_character(fixture_name: str) -> None:
    """The brief's literal criterion 4, on the fixtures where it is arithmetic.

    ``realised_bits < len(text)`` requires ``8 ceil(T/16) < T`` with
    ``T = n(n-1)/2``, i.e. ``T >= 9``, i.e. ``n >= 5``.  ``PATH_2`` (T = 1)
    and ``EMPTY_3`` (T = 3) cannot satisfy it: one byte is the floor.
    """
    graph = fixtures.to_networkx(fixtures.ALL_FIXTURES[fixture_name])
    encoding = backend("adjacency").encode(graph)
    assert backend("adjacency").bits(encoding).realised_bits < len(encoding.text)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CONTRACT DEFECT, orchestrator's bits.py, reported not fixed. The design "
        "note §4.2 and CONTRACTS §5 specify realised_bits = 8*ceil(n(n-1)/16), "
        "i.e. the triangle packed 8 bits to a byte = 8*ceil(T/8) with T = "
        "n(n-1)/2. bits.py:46 calls _packed_bits(triangle, word=16), which is "
        "8*ceil(T/16) = 8*ceil(n(n-1)/32) -- half the specified value. At n=6 it "
        "returns 8 bits (one byte) for a 15-bit triangle, which does not fit."
    ),
)
@pytest.mark.parametrize("n", (5, 6, 7, 10, 20))
def test_adjacency_realised_bits_match_the_frozen_closed_form(n: int) -> None:
    """``8 ceil(n(n-1)/16)``, the value the design note freezes."""
    encoding = backend("adjacency").encode(cycle(n))
    assert backend("adjacency").bits(encoding).realised_bits == 8 * math.ceil(n * (n - 1) / 16)


@pytest.mark.parametrize(
    "module", (adjacency_mod, graph6_mod, sparse6_mod), ids=lambda m: m.__name__
)
def test_no_track_a_module_reads_encoding_text(module: object) -> None:
    """``Encoding.text`` is a debugging view and is never read by our code."""
    source = pathlib.Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")
    tree = ast.parse(source)
    reads = [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "text"
    ]
    assert not reads, f"{module} reads .text at lines {[n.lineno for n in reads]}"


# --------------------------------------------------------------------------- #
# Criterion 3 -- Claim A on the real cohort
# --------------------------------------------------------------------------- #

#: ``competitors/README.md`` §4.3, median entropy-bound bits.  The sparse6
#: column is the README's own convention -- ``6 len(wire)``, with the ``':'``
#: counted.  ``bits.py`` excludes it, so the shipped value is six lower.
README_43_MEDIAN_ENTROPY_BITS = {
    "iam_letter_low": (6.0, 12.0, 24.0),
    "iam_letter_med": (6.0, 12.0, 24.0),
    "iam_letter_high": (10.0, 18.0, 36.0),
    "linux": (36.0, 42.0, 60.0),
    "aids": (55.0, 66.0, 72.0),
    "grec": (55.0, 66.0, 78.0),
    "aids_iam": (55.0, 66.0, 72.0),
    "coil_del": (153.0, 162.0, 282.0),
    "mutagenicity": (300.0, 306.0, 168.0),
    "protein": (465.0, 474.0, 390.0),
}


@pytest.mark.slow
@pytest.mark.parametrize("dataset", sorted(README_43_MEDIAN_ENTROPY_BITS))
def test_claim_a_medians_reproduce_readme(dataset: str) -> None:
    """README §4.3, all ten datasets, all three track-A columns.

    ``adjacency`` and ``graph6`` reproduce **exactly**.  ``sparse6``
    reproduces exactly under the README's own ``6 len(wire)`` convention and
    is exactly six bits lower under the frozen ``bits.py`` convention, which
    excludes the ``':'`` framing byte.  Both halves are asserted, so the
    delta is provably the prefix and cannot drift into anything else.

    Suite 2 rows use a 400-graph sample at seed 42; Suite 1 rows use every
    retained graph, matching the README's own caption.
    """
    pytest.importorskip("numpy")
    try:
        cohort = datasets.load(dataset)
    except datasets.DatasetNotFoundError as exc:
        pytest.skip(str(exc))
    graphs = (
        [cohort.graphs[i] for i in cohort.sample(400, seed=42)]
        if cohort.suite == "suite2"
        else list(cohort.graphs)
    )
    expected_adjacency, expected_graph6, expected_sparse6 = README_43_MEDIAN_ENTROPY_BITS[dataset]

    def median_entropy(name: str) -> float:
        obj = backend(name)
        return statistics.median(obj.bits(obj.encode(g)).entropy_bits for g in graphs)

    assert median_entropy("adjacency") == expected_adjacency
    assert median_entropy("graph6") == expected_graph6

    sparse6_backend = backend("sparse6")
    wires = [sparse6_backend.encode(g).wire for g in graphs]
    assert all(w is not None for w in wires)
    readme_convention = statistics.median(6.0 * len(w) for w in wires if w is not None)
    assert readme_convention == expected_sparse6, "README §4.3 counts the ':' prefix"
    assert median_entropy("sparse6") == expected_sparse6 - 6.0, (
        "bits.py excludes the ':' from the entropy bound, per T-04-design §4.2. "
        "README §4.3's sparse6 column predates that decision and is six bits high"
    )


@pytest.mark.slow
def test_isalgraph_is_never_shorter_than_adjacency_on_letter() -> None:
    """The load-bearing row: ``n(n-1)/2`` is minimal when the graph is small.

    Asserted here as a property of the *denominator* rather than of
    IsalGraph: the adjacency entropy bound on every Letter graph is at most
    12 bits, so the 0.0 % figure is not a fluke of one encoder.
    """
    pytest.importorskip("numpy")
    try:
        cohort = datasets.load("iam_letter_low")
    except datasets.DatasetNotFoundError as exc:
        pytest.skip(str(exc))
    obj = backend("adjacency")
    values = [obj.bits(obj.encode(g)).entropy_bits for g in cohort.graphs]
    assert statistics.median(values) == 6.0
    assert max(values) <= 21.0


# --------------------------------------------------------------------------- #
# Capabilities, frames and the sparse6 conventions
# --------------------------------------------------------------------------- #


def test_positional_frame_is_declared_iff_populated() -> None:
    """``padded_hamming`` is gated on the capability and reads only the frame."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    for name in ("adjacency", "graph6"):
        obj = backend(name)
        assert Capability.POSITIONAL_FRAME in obj.capabilities
        frame = obj.encode(graph).frame
        assert frame is not None
        assert len(frame.pairs) == 15
        assert len(frame.bits) == 15
    sparse = backend("sparse6")
    assert Capability.POSITIONAL_FRAME not in sparse.capabilities
    assert sparse.encode(graph).frame is None, (
        "sparse6 is not a positional bit vector; padded_hamming must report "
        "undefined there, and that undefined is a reported F1 result"
    )


def test_adjacency_frame_bits_equal_its_symbols_and_graph6_frame_bits_do_not() -> None:
    """The reason ``PositionalFrame`` carries its own bits."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    triangle = backend("adjacency").encode(graph)
    assert triangle.frame is not None
    assert triangle.frame.bits == triangle.symbols

    packed = backend("graph6").encode(graph)
    assert packed.frame is not None
    assert packed.frame.bits != packed.symbols
    assert len(packed.symbols) == 4
    assert len(packed.frame.bits) == 15


def test_sparse6_symbols_exclude_the_colon_and_text_includes_it() -> None:
    """``':'`` is framing, not payload -- and not a unit of edit."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    encoding = backend("sparse6").encode(graph)
    assert encoding.text.startswith(":")
    assert ":" not in encoding.symbols
    assert encoding.wire is not None
    assert len(encoding.symbols) == len(encoding.wire) - 1


def test_sparse6_bit_conventions() -> None:
    """Entropy excludes the ``':'``; realised includes it."""
    for n in BOUNDARY_SIZES:
        encoding = backend("sparse6").encode(cycle(n))
        assert encoding.wire is not None
        counted = backend("sparse6").bits(encoding)
        assert counted.entropy_bits == 6.0 * len(encoding.wire) - 6.0
        assert counted.realised_bits == 8 * len(encoding.wire)
        assert counted.payload_bits is None


def test_sparse6_length_varies_with_m_at_fixed_n() -> None:
    """Which is why plain Hamming is undefined on most sparse6 pairs."""
    lengths = set()
    for m in (3, 6, 9, 12):
        graph = nx.gnm_random_graph(10, m, seed=m)
        wire = backend("sparse6").encode(graph).wire
        assert wire is not None
        lengths.add(len(wire))
    assert len(lengths) > 1

    graph6_lengths = set()
    for m in (3, 6, 9, 12):
        wire = backend("graph6").encode(nx.gnm_random_graph(10, m, seed=m)).wire
        assert wire is not None
        graph6_lengths.add(len(wire))
    assert len(graph6_lengths) == 1, "graph6's length is a function of n alone"


@pytest.mark.parametrize("name", TRACK_A)
def test_capabilities_are_honest(name: str) -> None:
    """Declared, never inferred -- and none of the three is canonical."""
    obj = backend(name)
    assert Capability.REVERSIBLE in obj.capabilities
    assert Capability.HANDLES_DISCONNECTED in obj.capabilities
    assert Capability.CANONICAL not in obj.capabilities
    assert Capability.COMPLETE_INVARIANT not in obj.capabilities
    assert Capability.SUITE1_ONLY not in obj.capabilities
    assert Capability.BASELINE not in obj.capabilities


@pytest.mark.parametrize("name", TRACK_A)
def test_disconnected_and_isolated_vertices_encode(name: str) -> None:
    """All three handle what IsalGraph raises on; that is a properties-table row."""
    obj = backend(name)
    for fixture in (fixtures.C4_PLUS_K3_DISJOINT, fixtures.EMPTY_3):
        encoding = obj.encode(fixtures.to_networkx(fixture))
        assert encoding.n_nodes == fixture[0]
        assert encoding.n_edges == len(fixture[1])


def test_k33_and_prism_are_separated() -> None:
    """The completeness witness: WL cannot separate them; all three of ours can."""
    k33 = fixtures.to_networkx(fixtures.K33)
    prism = fixtures.to_networkx(fixtures.PRISM)
    for name in TRACK_A:
        assert backend(name).encode(k33).text != backend(name).encode(prism).text


# --------------------------------------------------------------------------- #
# The cross-track contract and the registry
# --------------------------------------------------------------------------- #


def test_sparse6_serialise_has_the_frozen_signature() -> None:
    """Agent B imports this to register ``sparse6_nauty``.  CONTRACTS §4."""
    signature = inspect.signature(sparse6_mod.serialise)
    assert list(signature.parameters) == ["graph"]
    assert signature.parameters["graph"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    encoding = sparse6_mod.serialise(fixtures.to_networkx(fixtures.RUNNING_EXAMPLE))
    assert isinstance(encoding, Encoding)
    assert encoding.backend == "sparse6"


def test_module_level_serialise_matches_the_backend_method() -> None:
    """``serialise`` and ``encode`` are one code path, not two."""
    rng = random.Random(99)
    for _ in range(50):
        graph = random_graph(rng)
        assert adjacency_mod.serialise(graph) == backend("adjacency").encode(graph)
        assert graph6_mod.serialise(graph) == backend("graph6").encode(graph)
        assert sparse6_mod.serialise(graph) == backend("sparse6").encode(graph)


@pytest.mark.parametrize("name", TRACK_A)
def test_backend_is_registered_and_available(name: str) -> None:
    """Registration happens at module import, through the lazy registry."""
    obj = backend(name)
    assert obj.name == name
    assert type(obj).is_available()
    assert obj.encode(fixtures.to_networkx(fixtures.PATH_2)).backend == name


@pytest.mark.parametrize("name", TRACK_A)
def test_bits_is_not_overridden(name: str) -> None:
    """``bits.py`` is the only producer of a ``BitCount``."""
    assert type(backend(name)).bits is ReprBackend.bits
