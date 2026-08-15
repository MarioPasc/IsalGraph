"""The WL subtree kernel: two independent implementations, and the witness.

Two things this file exists to pin forever.

**The convention.**  ``grakel(n_iter=k) == ours(h=k)``.  There is no
off-by-one -- ``grakel/kernels/weisfeiler_lehman.py`` sets
``self._n_iter = self.n_iter + 1`` and loops ``range(1, self._n_iter)``, so
``n_iter`` counts refinements on top of the base histogram, exactly as our
``h`` does.  ``wl-subtree-kernel.md`` §1 claims ``grakel(n_iter=3)`` equals
``ours(h=2)``; it does not, and the assertion below is what stops that being
re-derived.  Two independent implementations agreeing to machine precision
is what makes the WL row auditable without a third-party version pin.

**The witness.**  ``K_{3,3}`` and the triangular prism are both connected
and 3-regular on six vertices and are not isomorphic, and 1-WL cannot
separate them at any ``h``: the colouring is constant after round 1, so
refinement never starts.  Kernel distance is **exactly 0.0000**, while the
minimum DFS code separates them.  That is the cleanest evidence in the
folder for R1.2's uniqueness axis.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

import pytest

pytestmark = pytest.mark.unit

nx = pytest.importorskip("networkx")

from isalgraph.competitors import bits as bits_module  # noqa: E402
from isalgraph.competitors import fixtures  # noqa: E402
from isalgraph.competitors.backends.min_dfs import min_dfs_code, render  # noqa: E402
from isalgraph.competitors.backends.wl import (  # noqa: E402
    BASE_COLOUR,
    WL_ROUNDS,
    WLSubtree,
    grakel_available,
    grakel_distance,
    grakel_gram,
    wl_colours,
    wl_features,
)
from isalgraph.competitors.base import Capability, Encoding, VectorBackend  # noqa: E402
from isalgraph.competitors.metrics.kernel import KernelDistance, linear_kernel  # noqa: E402
from isalgraph.competitors.registry import get_vector_backend  # noqa: E402
from isalgraph.errors import BitCountUndefined  # noqa: E402

requires_grakel = pytest.mark.skipif(not grakel_available(), reason="grakel is not installed")


def _graph(name: str) -> nx.Graph:
    return fixtures.to_networkx(fixtures.ALL_FIXTURES[name])


def _sample_graphs(seed: int = 11, count: int = 12) -> list[nx.Graph]:
    """A mixed fixture set: the named fixtures plus random connected graphs."""
    graphs = [_graph(name) for name in fixtures.ALL_FIXTURES]
    rng = random.Random(seed)
    while len(graphs) < len(fixtures.ALL_FIXTURES) + count:
        n = rng.randint(4, 12)
        candidate = nx.gnp_random_graph(n, 0.35, seed=rng.randrange(10**9))
        graphs.append(candidate)
    return graphs


# ---------------------------------------------------------------------------
# The convention -- criterion 2
# ---------------------------------------------------------------------------


@requires_grakel
def test_grakel_n_iter_equals_our_h() -> None:
    """``grakel(n_iter=2) == ours(h=2) == 5.830952``, exactly.

    The brief's ``grakel(n_iter=3) == ours(h=2)`` is wrong; ``n_iter=3``
    gives 7.211103, which is ``ours(h=3)``.  All four rows are asserted so
    the mapping cannot drift by one in either direction.
    """
    g = _graph("running_example")
    h = _graph("running_example_minus_edge")
    expected = {1: 2.0, 2: 5.830951894845301, 3: 7.211102550927978}
    for rounds, value in expected.items():
        ours = WLSubtree(h=rounds).distance(g, h)
        assert ours == pytest.approx(value, abs=1e-12)
        assert grakel_distance(g, h, h=rounds) == pytest.approx(value, abs=1e-12)
    assert round(WLSubtree(h=2).distance(g, h), 6) == 5.830952


@requires_grakel
def test_grakel_gram_entries_match_ours() -> None:
    """The Gram matrix agrees entrywise, not just the distance.

    ``K(G,G) = 62`` at ``h = 1`` is ``36`` (base: six identical labels) plus
    ``26`` (the degree histogram ``5^2 + 1^2``), which is the arithmetic
    that fixes the convention independently of any implementation.
    """
    g = _graph("running_example")
    h = _graph("running_example_minus_edge")
    gram = grakel_gram([g, h], h=1)
    assert gram[0][0] == pytest.approx(62.0)
    fg, fh = wl_features(g, 1), wl_features(h, 1)
    assert linear_kernel(fg, fg) == pytest.approx(gram[0][0])
    assert linear_kernel(fh, fh) == pytest.approx(gram[1][1])
    assert linear_kernel(fg, fh) == pytest.approx(gram[0][1])


@requires_grakel
@pytest.mark.parametrize("rounds", [0, 1, 2, 3, 5])
def test_two_implementations_agree_on_a_fixture_set(rounds: int) -> None:
    """Machine-precision agreement over every pair of a mixed fixture set."""
    graphs = _sample_graphs()
    gram = grakel_gram(graphs, h=rounds) if rounds else None
    features = [wl_features(g, rounds) for g in graphs]
    if gram is None:  # grakel rejects n_iter=0; check ours is the base histogram
        for g, f in zip(graphs, features, strict=True):
            assert sum(f.values()) == g.number_of_nodes()
        return
    for i, j in itertools.combinations_with_replacement(range(len(graphs)), 2):
        assert linear_kernel(features[i], features[j]) == pytest.approx(
            gram[i][j], rel=1e-12, abs=1e-9
        )


# ---------------------------------------------------------------------------
# The completeness witness -- criterion 7
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rounds", [1, 2, 3, 5])
def test_k33_and_prism_have_distance_exactly_zero(rounds: int) -> None:
    """Distance **exactly** 0.0000, at every ``h``.  Not approximately."""
    k33, prism = _graph("k33"), _graph("prism")
    assert not nx.is_isomorphic(k33, prism)
    assert WLSubtree(h=rounds).distance(k33, prism) == 0.0


@requires_grakel
@pytest.mark.parametrize("rounds", [1, 2, 3, 5])
def test_k33_and_prism_zero_under_grakel_too(rounds: int) -> None:
    """The witness does not depend on our implementation."""
    assert grakel_distance(_graph("k33"), _graph("prism"), h=rounds) == pytest.approx(0.0)


def test_min_dfs_separates_the_witness_that_wl_cannot() -> None:
    """The two-line regression test: WL 0, min-DFS non-zero.

    A broken canonical backend would show up here instantly, which is why
    ``K33`` and ``PRISM`` are shared fixtures rather than local constants.
    """
    k33, prism = _graph("k33"), _graph("prism")
    assert WLSubtree().distance(k33, prism) == 0.0
    assert render(min_dfs_code(k33)) == "0-1 1-2 2-3 3-0 3-4 4-1 4-5 5-0 5-2"
    assert render(min_dfs_code(prism)) == "0-1 1-2 2-0 2-3 3-4 4-0 4-5 5-1 5-3"
    assert min_dfs_code(k33) != min_dfs_code(prism)


def test_incompleteness_is_not_specific_to_six_vertices() -> None:
    """Any two connected ``k``-regular graphs of the same order collapse.

    The Petersen graph and the pentagonal prism are both cubic on ten
    vertices and not isomorphic.  1-WL gives them distance 0 as well, so the
    ``K_{3,3}`` witness is an instance of a rule, not a curiosity.
    """
    petersen = nx.petersen_graph()
    pentagonal_prism = nx.circular_ladder_graph(5)
    assert not nx.is_isomorphic(petersen, pentagonal_prism)
    for rounds in (1, 2, 3, 5):
        assert WLSubtree(h=rounds).distance(petersen, pentagonal_prism) == 0.0


def test_the_kernel_distance_is_declared_a_pseudometric() -> None:
    """F2 requires the declaration; the witness makes it concrete."""
    assert KernelDistance().is_pseudometric is True
    assert KernelDistance().consumes == "features"


# ---------------------------------------------------------------------------
# No fabricated bit count
# ---------------------------------------------------------------------------


def test_wl_has_no_bits_method_at_all() -> None:
    """Unreachable rather than forbidden: ``VectorBackend`` has no ``bits``."""
    backend = get_vector_backend("wl_subtree")
    assert isinstance(backend, VectorBackend)
    assert not hasattr(backend, "bits")
    assert not hasattr(VectorBackend, "bits")


def test_bits_count_raises_for_wl_subtree() -> None:
    """``bits.count`` refuses the name, with the reason printed."""
    encoding = Encoding(
        backend="wl_subtree",
        symbols=(),
        alphabet_size=1,
        n_nodes=6,
        n_edges=9,
        text="",
    )
    with pytest.raises(BitCountUndefined, match="not a serialisation"):
        bits_module.count(encoding)


def test_bit_count_reason_is_printable() -> None:
    """The Claim A cell is empty **and the reason is printed**."""
    reason = get_vector_backend("wl_subtree").bit_count_reason()
    assert "container" in reason


# ---------------------------------------------------------------------------
# Frozen parameters
# ---------------------------------------------------------------------------


def test_frozen_h_is_two() -> None:
    """``h = 2``.  ``h = 3`` is below it on all five Suite-1 datasets."""
    assert WL_ROUNDS == 2
    assert WLSubtree().h == 2
    assert get_vector_backend("wl_subtree").h == 2


def test_h_is_a_constructor_keyword() -> None:
    """``get_vector_backend('wl_subtree', h=k)`` -- the identity check needs it."""
    for k in (1, 2, 3):
        assert get_vector_backend("wl_subtree", h=k).h == k


def test_normalize_true_is_refused() -> None:
    """``normalize=True`` removes the size signal GED depends on."""
    with pytest.raises(ValueError, match="graph-size signal"):
        WLSubtree(normalize=True)
    assert WLSubtree().normalize is False


def test_negative_h_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        WLSubtree(h=-1)


# ---------------------------------------------------------------------------
# The per-batch-fit trap, made impossible rather than discouraged
# ---------------------------------------------------------------------------


def test_features_do_not_depend_on_what_was_fitted() -> None:
    """The batching-order bug is structurally unreachable.

    ``fit_transform`` on a subset produces a different colour vocabulary in
    an implementation that compresses colours per corpus, so the distance
    matrix would depend on batching order -- silent corruption of the same
    family as GEDLIB's ``get_lower_bound()`` trap.  Here a colour is a
    digest of a canonical signature, so it cannot happen.
    """
    graphs = _sample_graphs()
    whole = WLSubtree()
    whole.fit(graphs)
    reference = [dict(whole.features(g)) for g in graphs]

    for chunk in (1, 3, 7):
        batched = WLSubtree()
        for start in range(0, len(graphs), chunk):
            batched.fit(graphs[start : start + chunk])
            for offset, graph in enumerate(graphs[start : start + chunk]):
                assert dict(batched.features(graph)) == reference[start + offset]


def test_distance_matrix_is_independent_of_batching() -> None:
    """The property that actually matters: the Gram matrix does not move."""
    graphs = _sample_graphs()
    metric = KernelDistance()

    def matrix(chunk: int) -> list[list[float]]:
        backend = WLSubtree()
        features: list[dict[str, int]] = []
        for start in range(0, len(graphs), chunk):
            backend.fit(graphs[start : start + chunk])
            features.extend(dict(backend.features(g)) for g in graphs[start : start + chunk])
        return [[metric.distance(a, b) for b in features] for a in features]

    reference = matrix(len(graphs))
    for chunk in (1, 4):
        assert matrix(chunk) == reference


def test_fit_records_the_vocabulary() -> None:
    """``fit`` is inert on the distance but still reports what it saw."""
    graphs = _sample_graphs()
    backend = WLSubtree()
    assert backend.vocabulary == ()
    backend.fit(graphs)
    assert backend.n_fitted == len(graphs)
    assert len(backend.vocabulary) > 0
    assert set(backend.vocabulary) >= set(wl_features(graphs[0], WL_ROUNDS))


def test_features_are_never_restricted_to_the_fitted_vocabulary() -> None:
    """Dropping unseen colours would reintroduce the batching dependence."""
    graphs = _sample_graphs()
    backend = WLSubtree()
    backend.fit([graphs[0]])
    unseen = graphs[-1]
    assert set(backend.features(unseen)) - set(backend.vocabulary)


# ---------------------------------------------------------------------------
# Structure of the feature map
# ---------------------------------------------------------------------------


def test_feature_counts_on_the_running_example() -> None:
    """Criterion 1's WL half: 10 non-zero features for ``G``, 13 for ``H``.

    Those are the counts at ``h = 3`` under this module's convention, which
    is the same ``h`` ``wl-subtree-kernel.md`` §2 quotes.
    """
    g, h = _graph("running_example"), _graph("running_example_minus_edge")
    assert len(wl_features(g, 3)) == 10
    assert len(wl_features(h, 3)) == 13


def test_round_zero_is_a_single_constant_colour() -> None:
    """Topology-only: no vertex labels, so the base histogram is ``n`` of one colour."""
    graph = _graph("running_example")
    rounds = wl_colours(graph, 2)
    assert set(rounds[0].values()) == {BASE_COLOUR}
    base = wl_features(graph, 0)
    assert base == {f"h0:{BASE_COLOUR}": graph.number_of_nodes()}


def test_round_one_colours_are_degrees() -> None:
    """With one base colour, round 1 partitions vertices exactly by degree."""
    graph = _graph("running_example")
    colour = wl_colours(graph, 1)[1]
    by_degree: dict[int, set[str]] = {}
    for v, c in colour.items():
        by_degree.setdefault(graph.degree(v), set()).add(c)
    assert all(len(s) == 1 for s in by_degree.values())
    assert len({next(iter(s)) for s in by_degree.values()}) == len(by_degree)


def test_total_feature_mass_is_n_times_h_plus_one() -> None:
    """Every vertex contributes one colour per round, so the mass is fixed."""
    for name in fixtures.ALL_FIXTURES:
        graph = _graph(name)
        for rounds in (0, 1, 2, 3):
            assert sum(wl_features(graph, rounds).values()) == graph.number_of_nodes() * (
                rounds + 1
            )


def test_more_rounds_never_merge_colour_classes() -> None:
    """WL refinement is monotone: the partition only ever gets finer."""
    graph = _graph("running_example_minus_edge")
    rounds = wl_colours(graph, 4)
    sizes = [len(set(c.values())) for c in rounds]
    assert sizes == sorted(sizes)


# ---------------------------------------------------------------------------
# Invariance, scope and registry
# ---------------------------------------------------------------------------


def test_f3_isomorphism_invariance_on_fixtures() -> None:
    """Features are identical under a relabelling with a fresh insertion order."""
    rng = random.Random(42)
    backend = WLSubtree()
    for name in fixtures.ALL_FIXTURES:
        graph = _graph(name)
        base = dict(backend.features(graph))
        for _ in range(20):
            assert dict(backend.features(fixtures.shuffled_copy(graph, rng))) == base, name


def test_handles_disconnected_graphs() -> None:
    """WL is defined where the DFS code is not -- an AE.3 row, both ways."""
    disconnected = _graph("c4_plus_k3_disjoint")
    assert Capability.HANDLES_DISCONNECTED in WLSubtree.capabilities
    assert sum(WLSubtree().features(disconnected).values()) == 21
    empty = _graph("empty_3")
    assert WLSubtree().features(empty) == {f"h{r}:{BASE_COLOUR}": 3 for r in range(1)} | {
        f"h{r}:{c}": 3 for r in (1, 2) for c in {next(iter(set(wl_colours(empty, 2)[r].values())))}
    }


def test_registry_returns_a_vector_backend_not_a_repr_backend() -> None:
    """WL is registered as the one :class:`VectorBackend` in the pool."""
    from isalgraph.competitors.registry import get_repr_backend

    assert isinstance(get_vector_backend("wl_subtree"), WLSubtree)
    with pytest.raises(Exception, match="wl_subtree"):
        get_repr_backend("wl_subtree")


def test_shipped_backend_needs_no_third_party_library() -> None:
    """``grakel`` is the cross-check, not a runtime dependency of the backend."""
    assert WLSubtree.is_available() is True


def _pairwise(graphs: Sequence[nx.Graph], rounds: int) -> list[float]:
    backend = WLSubtree(h=rounds)
    backend.fit(graphs)
    return [backend.distance(a, b) for a, b in itertools.combinations(graphs, 2)]


def test_e10_h2_versus_h5_changes_the_distances() -> None:
    """E10: ``wl_kernel_computer.py`` defaults to ``n_iter = 5``, i.e. ``h = 5``.

    Under the corrected convention that is **three** refinement rounds past
    the frozen ``h = 2``, and past the ``h = 3`` already measured strictly
    worse on all five Suite-1 datasets.  This test records that the two are
    not interchangeable; reconciling them belongs to T-06, and this file
    does not edit ``wl_kernel_computer.py``.
    """
    graphs = _sample_graphs()
    d2, d5 = _pairwise(graphs, 2), _pairwise(graphs, 5)
    assert d2 != d5
    assert all(b >= a - 1e-9 for a, b in zip(d2, d5, strict=True))
    # The witness is the one pair that does not move.
    assert WLSubtree(h=2).distance(_graph("k33"), _graph("prism")) == 0.0
    assert WLSubtree(h=5).distance(_graph("k33"), _graph("prism")) == 0.0
