"""T-04a annex E1: invariance and the separation ratio ``psi``.

The claims E1 puts in front of a reviewer are all of the form "this
representation is / is not a well-defined function on isomorphism classes",
and every one of them can be produced by a harness that is simply broken in
a way that looks like a result.  Each test here pins one of those ways.

Harness -> test:

1. A relabeller that cannot make an order-dependent format fail makes every
   format look invariant (finding 13) ->
   :func:`test_shuffled_copy_can_make_graph6_non_invariant` and
   :func:`test_relabel_nodes_copy_would_have_hidden_the_failure`, which
   measures the alternative and shows it hiding exactly this defect.
2. A wrong atlas measures the characterisation on the wrong population ->
   :func:`test_connected_atlas_matches_oeis_a001349`,
   :func:`test_exhaustive_invariance_refuses_a_broken_atlas`.
3. A permutation sweep that silently skips labelled copies would report
   invariance it never checked ->
   :func:`test_orbit_total_is_n_factorial_over_the_automorphism_group`.
4. A ``psi`` denominator that assumes a draw holds no repeated graphs
   divides by a mean contaminated with exact zeros ->
   :func:`test_psi_denominator_excludes_isomorphic_pairs`,
   :func:`test_isomorphism_classes_are_settled_by_an_exact_test`.
5. Conflating "this relabelling gave 0" with "every relabelling gave 0"
   quotes an invariance rate that is not the grid's F3 ->
   :func:`test_per_pair_and_per_graph_invariance_are_different_quantities`.

The atlas and fixture tests need no cohort on disk.  The two that read the
real draws carry ``@pytest.mark.integration`` and skip when the external
drive is not mounted.
"""

from __future__ import annotations

import itertools
import json
import math
import random

import pytest

from isalgraph.competitors import datasets, fixtures, registry
from isalgraph.competitors.admissibility import common
from isalgraph.competitors.admissibility import e1_invariance as e1

nx = pytest.importorskip("networkx")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _have(backend: str) -> bool:
    """Whether *backend*'s optional dependency imports."""
    return backend in registry.available_backends()


def _grid_file(tmp_path, primary: dict[str, str | None]) -> str:  # noqa: ANN001
    """A minimal grid JSON carrying only the primary-distance block."""
    path = tmp_path / "grid.json"
    path.write_text(json.dumps({"primary_distance": primary}), encoding="utf-8")
    return str(path)


def _fixture_cohort(names: tuple[str, ...], *, relabellings: int, seed: int) -> e1.Cohort:
    graphs = [fixtures.to_networkx(fixtures.ALL_FIXTURES[name]) for name in names]
    return e1.build_cohort("fixtures", graphs, relabellings=relabellings, seed=seed)


CONNECTED_FIXTURES = fixtures.CONNECTED_FIXTURES


# --------------------------------------------------------------------------
# The population: the atlas has to be intact before anything else means much
# --------------------------------------------------------------------------


def test_connected_atlas_matches_oeis_a001349() -> None:
    """995 connected graphs on 2..7 nodes, 1/2/6/21/112/853 by node count."""
    atlas = common.connected_atlas(common.EXHAUSTIVE_N_INVARIANCE)
    assert len(atlas) == 995
    counts: dict[int, int] = {}
    for graph in atlas:
        counts[graph.number_of_nodes()] = counts.get(graph.number_of_nodes(), 0) + 1
    assert counts == {2: 1, 3: 2, 4: 6, 5: 21, 6: 112, 7: 853}
    assert counts == e1.A001349
    assert sum(e1.A001349.values()) == 995


def test_atlas_holds_one_graph_per_isomorphism_class() -> None:
    """No two atlas graphs on the same node count are isomorphic."""
    atlas = [g for g in common.connected_atlas(6) if g.number_of_nodes() == 5]
    for a, b in itertools.combinations(atlas, 2):
        assert not nx.is_isomorphic(a, b)


def test_exhaustive_invariance_refuses_a_broken_atlas(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short atlas raises rather than reporting a smaller invariant set."""
    full = common.connected_atlas(4)
    monkeypatch.setattr(e1.common, "connected_atlas", lambda max_n: full[:-1])
    with pytest.raises(common.AdmissibilityError, match="A001349"):
        e1.exhaustive_invariance("graph6", "levenshtein", max_n=4)


# --------------------------------------------------------------------------
# Part A: the characterisation
# --------------------------------------------------------------------------


def test_permuted_graph_is_isomorphic_and_inserted_in_ascending_order() -> None:
    """Only the labelling varies between two permutations of one graph."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    for perm in itertools.islice(itertools.permutations(range(6)), 12):
        copy = e1.permuted_graph(graph, perm)
        assert nx.is_isomorphic(graph, copy)
        assert list(copy.nodes()) == sorted(copy.nodes())


@pytest.mark.parametrize("backend", ["adjacency", "graph6", "sparse6"])
def test_n_squared_family_is_invariant_exactly_on_the_complete_graphs(backend: str) -> None:
    """T-04's claim, re-verified: the invariant set is exactly ``{K_n}``.

    The strict upper triangle of ``P A P^T`` is the same for every ``P`` iff
    ``A`` is constant off the diagonal.  Over connected graphs that leaves the
    complete graph and nothing else, at every node count.
    """
    rows = e1.exhaustive_invariance(backend, "levenshtein", max_n=5)
    assert [row.n_nodes for row in rows] == [2, 3, 4, 5]
    for row in rows:
        assert row.mode == "exhaustive"
        assert row.exhaustive
        assert row.n_settled == row.n_graphs
        assert row.n_invariant == 1, f"n={row.n_nodes} invariant set {row.invariant_graph6}"
        assert row.n_no_counterexample is None
        assert row.invariant_set_is_complete_graph


@pytest.mark.skipif(not _have("nauty_graph6"), reason="pynauty not installed")
def test_a_canonical_backend_is_invariant_on_every_atlas_graph() -> None:
    """A canonical labelling makes every graph invariant, ``K_n`` included.

    The complementary reading of the same sweep: ``invariant_set_is_complete_
    graph`` is ``False`` here precisely because the invariant set is *larger*
    than ``{K_n}``, which is what class III means.
    """
    rows = e1.exhaustive_invariance("nauty_graph6", "levenshtein", max_n=5)
    for row in rows:
        assert row.n_invariant == row.n_graphs
        assert row.invariant_set_is_complete_graph is (row.n_nodes == 2)


def test_orbit_total_is_n_factorial_over_the_automorphism_group() -> None:
    """Deduplication drops repeats, never labelled copies.

    ``|orbit(G)| = n!/|Aut(G)|`` by orbit-stabiliser.  At ``n = 3`` that is
    ``3`` for the path and ``1`` for ``K_3``, so a sweep that covered both
    fully reports ``orbit_total == 4``.
    """
    rows = e1.exhaustive_invariance("adjacency", "levenshtein", max_n=3)
    by_n = {row.n_nodes: row for row in rows}
    assert by_n[3].orbit_total == 1  # only K_3 is invariant, and |orbit(K_3)| = 1

    if _have("nauty_graph6"):
        canonical = {
            row.n_nodes: row
            for row in e1.exhaustive_invariance("nauty_graph6", "levenshtein", max_n=3)
        }
        assert canonical[3].orbit_total == 4  # 3!/2 for P_3 plus 3!/6 for K_3
        assert canonical[3].encodes == 4


def test_a_full_sweep_covers_every_labelled_connected_graph() -> None:
    """``orbit_total`` reproduces OEIS A001187 -- the dedup skips nothing.

    ``sum_G n!/|Aut(G)|`` over the connected graphs on ``n`` nodes up to
    isomorphism *is* the number of **labelled** connected graphs on ``n``
    nodes: 1, 4, 38, 728 for ``n = 2..5``.  A representation invariant on
    every atlas graph therefore had to encode exactly that many distinct
    labelled graphs.  If deduplication ever dropped a labelled copy the sum
    would fall short, and this is the cheapest way to see it.
    """
    if not _have("nauty_graph6"):
        pytest.skip("pynauty not installed")
    labelled_connected = {2: 1, 3: 4, 4: 38, 5: 728}
    for row in e1.exhaustive_invariance("nauty_graph6", "levenshtein", max_n=5):
        assert row.n_invariant == row.n_graphs
        assert row.orbit_total == labelled_connected[row.n_nodes]
        assert row.encodes == labelled_connected[row.n_nodes]


def test_a_bound_sweep_reports_itself_as_not_exhaustive() -> None:
    """Hitting the encode ceiling is declared, never silently truncated."""
    rows = e1.exhaustive_invariance("adjacency", "levenshtein", max_n=4, max_encodes=1)
    assert any(not row.exhaustive for row in rows)
    for row in rows:
        if not row.exhaustive:
            assert row.invariant_set_is_complete_graph is False


# --------------------------------------------------------------------------
# Part A's declared exhaustive / sampled split
# --------------------------------------------------------------------------


def test_n_up_to_six_is_exhaustive_for_every_backend() -> None:
    """The characterisation lives at n <= 6 and is never sampled."""
    for name in registry.available_backends():
        for n in range(2, e1.PART_A_ALWAYS_EXHAUSTIVE_MAX_N + 1):
            mode, perms = e1.permutation_plan(n, name, exhaustive_n7=(), sample=8, seed=common.SEED)
            assert mode == "exhaustive", f"{name} at n={n}"
            assert len(perms) == math.factorial(n)


def test_only_the_declared_backends_enumerate_n_seven() -> None:
    """The split is a property of the protocol, not of the machine's speed."""
    for name in e1.PART_A_EXHAUSTIVE_N7:
        mode, perms = e1.permutation_plan(
            7, name, exhaustive_n7=e1.PART_A_EXHAUSTIVE_N7, sample=200, seed=common.SEED
        )
        assert mode == "exhaustive"
        assert len(perms) == 5040
    for name in ("min_dfs", "agm_cam", "isalgraph_pruned", "wl_subtree"):
        mode, perms = e1.permutation_plan(
            7, name, exhaustive_n7=e1.PART_A_EXHAUSTIVE_N7, sample=200, seed=common.SEED
        )
        assert mode == "sampled"
        assert len(perms) == 200
        assert len(set(perms)) == 200, "the draw must be without replacement"
        assert perms[0] == tuple(range(7)), "the identity is always the base"


def test_the_permutation_sample_is_shared_by_every_graph_and_backend() -> None:
    """Sampled cells stay comparable because they see the same permutations."""
    first = e1.permutation_plan(7, "min_dfs", exhaustive_n7=(), sample=64, seed=common.SEED)[1]
    second = e1.permutation_plan(7, "wl_subtree", exhaustive_n7=(), sample=64, seed=common.SEED)[1]
    other_seed = e1.permutation_plan(7, "min_dfs", exhaustive_n7=(), sample=64, seed=7)[1]
    assert first == second
    assert first != other_seed


def test_a_sampled_cell_never_claims_invariance() -> None:
    """Sampling can fail to refute; it cannot decide.  The fields enforce it.

    ``n_invariant`` is ``None`` under ``mode = "sampled"`` and
    ``invariant_set_is_complete_graph`` is ``False``, so a downstream table
    cannot print a sampled count in a column that means "invariant".  What the
    cell *does* license is a rule-of-three bound on the chance that a random
    relabelling breaks a graph the draw did not break.
    """
    rows = e1.exhaustive_invariance(
        "adjacency",
        "levenshtein",
        max_n=5,
        exhaustive_n7=(),
        sample_n7=16,
    )
    # n <= 5 is below the always-exhaustive ceiling, so force the sampled path
    # through permutation_plan directly and through a cell built on it.
    mode, perms = e1.permutation_plan(7, "min_dfs", exhaustive_n7=(), sample=16, seed=common.SEED)
    assert mode == "sampled"
    assert all(row.mode == "exhaustive" for row in rows)

    sampled = e1.exhaustive_invariance(
        "adjacency", "levenshtein", max_n=7, exhaustive_n7=(), sample_n7=16
    )
    seven = next(row for row in sampled if row.n_nodes == 7)
    assert seven.mode == "sampled"
    assert seven.permutations_per_graph == 16
    assert seven.sample_seed == common.SEED
    assert seven.n_invariant is None
    assert seven.n_no_counterexample is not None
    assert seven.invariant_set_is_complete_graph is False
    assert seven.exhaustive is False
    assert seven.per_graph_non_invariance_upper == pytest.approx(common.rule_of_three(16))
    assert len(perms) == 16


def test_a_sample_at_least_as_large_as_n_factorial_is_exhaustive() -> None:
    """Asking for more permutations than exist enumerates them, and says so."""
    mode, perms = e1.permutation_plan(4, "min_dfs", exhaustive_n7=(), sample=100, seed=1)
    assert mode == "exhaustive"
    assert len(perms) == 24


def test_invariant_certificates_are_readable_graph6_strings() -> None:
    """The invariant list is checkable by a reader, not just by us."""
    rows = e1.exhaustive_invariance("adjacency", "levenshtein", max_n=4)
    for row in rows:
        for certificate in row.invariant_graph6:
            decoded = nx.from_graph6_bytes(certificate.encode("ascii"))
            n = decoded.number_of_nodes()
            assert decoded.number_of_edges() == n * (n - 1) // 2


# --------------------------------------------------------------------------
# The relabeller: a harness that cannot fail is worthless
# --------------------------------------------------------------------------


def test_shuffled_copy_can_make_graph6_non_invariant() -> None:
    """``fixtures.shuffled_copy`` really does break an order-dependent format.

    Without this the whole of E1 could report ``psi = 0`` for every
    representation and be measuring nothing at all.
    """
    backend = registry.get_repr_backend("graph6")
    metric = registry.get_metric("levenshtein")
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    base = backend.encode(graph)
    rng = random.Random(common.SEED)
    distances = [
        metric.distance(base, backend.encode(fixtures.shuffled_copy(graph, rng)))
        for _ in range(common.RELABELLINGS)
    ]
    assert any(d > 0.0 for d in distances)


@pytest.mark.parametrize("backend_name", ["adjacency", "graph6", "sparse6", "isalgraph_pruned"])
def test_shuffling_only_the_insertion_order_changes_nothing(backend_name: str) -> None:
    """Every pool backend depends on the labelling and not on insertion order.

    Finding 13 is about a harness that relabels with
    ``nx.relabel_nodes(copy=True)``, which preserves insertion order.  That
    matters only for a backend that *reads* insertion order, and after T-04's
    :func:`~isalgraph.competitors.backends.adjacency.normalised` the ``n^2``
    family reads ``sorted(graph.nodes())`` instead.  Measured here: with the
    labels held fixed and the insertion and edge orders shuffled, the encoding
    does not move once in :data:`common.RELABELLINGS` trials.

    Two consequences, and both are load-bearing:

    - Part A's label-only permutation sweep is not merely an upper bound on
      Part B for these backends; the two test the same dependence.
    - ``shuffled_copy`` is still the right relabeller, but for its *label*
      shuffle.  Its insertion-order shuffle is inert on the current pool, so
      an F3 harness built on ``relabel_nodes`` would fail here too -- the
      defect finding 13 names is real but is no longer the live one.
    """
    backend = registry.get_repr_backend(backend_name)
    metric = registry.get_metric("levenshtein")
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    base = backend.encode(graph)
    rng = random.Random(common.SEED)
    for _ in range(common.RELABELLINGS):
        order = list(graph.nodes())
        rng.shuffle(order)
        copy = nx.Graph()
        copy.add_nodes_from(order)
        edges = list(graph.edges())
        rng.shuffle(edges)
        copy.add_edges_from(edges)
        assert metric.distance(base, backend.encode(copy)) == 0.0


def test_build_cohort_is_a_pure_function_of_its_seed() -> None:
    """Every backend must see the same relabellings, so they are drawn once."""
    first = _fixture_cohort(CONNECTED_FIXTURES, relabellings=4, seed=common.SEED)
    second = _fixture_cohort(CONNECTED_FIXTURES, relabellings=4, seed=common.SEED)
    for a_row, b_row in zip(first.copies, second.copies, strict=True):
        for a, b in zip(a_row, b_row, strict=True):
            assert sorted(a.edges()) == sorted(b.edges())
            assert list(a.nodes()) == list(b.nodes())


# --------------------------------------------------------------------------
# Part B: psi
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _have("nauty_graph6"), reason="pynauty not installed")
def test_psi_is_exactly_zero_for_a_canonical_backend() -> None:
    """``psi = 0`` iff the representation is invariant on the cohort."""
    cohort = _fixture_cohort(CONNECTED_FIXTURES, relabellings=8, seed=common.SEED)
    row, per_graph = e1.cohort_psi("nauty_graph6", "levenshtein", cohort=cohort, resamples=64)
    assert row.psi == 0.0
    assert row.mean_self_distance == 0.0
    assert row.invariance_rate == 1.0
    assert row.n_graphs_all_relabellings_invariant == row.n_graphs
    assert row.psi_ci == (0.0, 0.0)
    assert set(per_graph.values()) == {0.0}
    # Zero events in N trials is an upper bound, never the rate 0.
    assert row.non_invariance_rule_of_three == pytest.approx(common.rule_of_three(row.n_self_pairs))


def test_psi_is_strictly_positive_for_graph6() -> None:
    """``graph6`` separates relabellings of one graph, so ``psi > 0``."""
    cohort = _fixture_cohort(CONNECTED_FIXTURES, relabellings=8, seed=common.SEED)
    row, per_graph = e1.cohort_psi("graph6", "levenshtein", cohort=cohort, resamples=64)
    assert row.psi is not None and row.psi > 0.0
    assert row.mean_self_distance is not None and row.mean_self_distance > 0.0
    assert row.invariance_rate is not None and row.invariance_rate < 1.0
    assert row.non_invariance_rule_of_three is None
    assert any(value > 0.0 for value in per_graph.values())
    assert row.psi_ci is not None
    low, high = row.psi_ci
    assert 0.0 <= low <= high


def test_psi_equals_the_ratio_of_the_two_reported_means() -> None:
    """The printed ``psi`` is the printed numerator over the printed denominator."""
    cohort = _fixture_cohort(CONNECTED_FIXTURES, relabellings=6, seed=common.SEED)
    row, per_graph = e1.cohort_psi("graph6", "levenshtein", cohort=cohort, resamples=32)
    assert row.psi == pytest.approx(row.mean_self_distance / row.mean_between_distance)
    # The per-graph statistic averages to the same number, which is what makes
    # the paired test in part C a test about psi rather than about something
    # adjacent to it.
    assert sum(per_graph.values()) / len(per_graph) == pytest.approx(row.psi)


def test_psi_denominator_excludes_isomorphic_pairs() -> None:
    """A duplicate graph in the draw contributes distance 0 and is excluded."""
    graph = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    duplicate = fixtures.shuffled_copy(graph, random.Random(7))
    other = fixtures.to_networkx(fixtures.K33)
    cohort = e1.build_cohort("dup", [graph, duplicate, other], relabellings=4, seed=common.SEED)

    row, _ = e1.cohort_psi("graph6", "levenshtein", cohort=cohort, resamples=32)
    assert row.n_isomorphic_pairs == 1
    assert row.n_between_pairs == 2
    assert row.isomorphism_verified

    # And the exclusion moves the number.  Under an invariant representation
    # the duplicated pair contributes exactly 0, so keeping it drags the
    # denominator down and inflates psi -- on a cohort where psi should be 0
    # that is invisible, which is why the check runs on the denominator.
    labels_ignoring_isomorphism = [0, 1, 2]
    kept, _ = e1.cohort_psi(
        "isalgraph_pruned",
        "levenshtein",
        cohort=cohort,
        iso_labels=labels_ignoring_isomorphism,
        resamples=32,
    )
    conditioned, _ = e1.cohort_psi("isalgraph_pruned", "levenshtein", cohort=cohort, resamples=32)
    assert kept.n_between_pairs == 3
    assert kept.mean_between_distance is not None
    assert conditioned.mean_between_distance is not None
    assert kept.mean_between_distance < conditioned.mean_between_distance


def test_isomorphism_classes_are_settled_by_an_exact_test() -> None:
    """``K_{3,3}`` and the prism share every cheap invariant and differ.

    Both are connected and 3-regular on six vertices, so ``(n, m, degree
    sequence)`` cannot separate them.  Only the VF2 verdict can, which is why
    the cheap key is a pre-filter and never the answer.
    """
    k33 = fixtures.to_networkx(fixtures.K33)
    prism = fixtures.to_networkx(fixtures.PRISM)
    assert k33.number_of_nodes() == prism.number_of_nodes()
    assert k33.number_of_edges() == prism.number_of_edges()
    assert sorted(d for _, d in k33.degree()) == sorted(d for _, d in prism.degree())

    labels = e1.isomorphism_classes([k33, prism, fixtures.shuffled_copy(k33, random.Random(3))])
    assert labels[0] != labels[1]
    assert labels[0] == labels[2]


def test_per_pair_and_per_graph_invariance_are_different_quantities() -> None:
    """A large automorphism group scores zeros without being invariant.

    ``K_{3,3}`` has ``|Aut| = 72`` of ``6! = 720`` labellings, so a random
    relabelling lands on an automorphism often enough that the per-pair rate
    is well above the per-graph rate.  Quoting one for the other is the error
    this pair of fields exists to prevent.
    """
    graph = fixtures.to_networkx(fixtures.K33)
    cohort = e1.build_cohort("k33", [graph], relabellings=40, seed=common.SEED)
    row, _ = e1.cohort_psi("adjacency", "levenshtein", cohort=cohort, resamples=32)
    assert row.n_graphs_all_relabellings_invariant == 0
    assert row.invariance_rate is not None and row.invariance_rate > 0.0
    assert row.graphs_invariant_ci is not None


def test_a_graph_the_backend_raises_on_is_skipped_not_counted_as_variant() -> None:
    """A ``SUITE1_ONLY`` refusal is a skip; it never lowers the invariance rate."""
    if not _have("agm_cam"):
        pytest.skip("agm_cam unavailable")
    small = fixtures.to_networkx(fixtures.RUNNING_EXAMPLE)
    large = nx.path_graph(30)
    cohort = e1.build_cohort("mixed", [small, large], relabellings=4, seed=common.SEED)
    row, _ = e1.cohort_psi("agm_cam", "levenshtein", cohort=cohort, resamples=16)
    assert row.n_skipped == 1
    assert row.n_graphs == 1
    assert row.invariance_rate == 1.0


def test_psi_needs_two_graphs_and_says_so_rather_than_dividing_by_zero() -> None:
    """One graph gives no between-pair, so ``psi`` is absent, not 0."""
    cohort = _fixture_cohort(("running_example",), relabellings=3, seed=common.SEED)
    row, per_graph = e1.cohort_psi("graph6", "levenshtein", cohort=cohort, resamples=16)
    assert row.psi is None
    assert row.mean_between_distance is None
    assert per_graph == {}


def test_psi_bootstrap_resamples_graphs_and_is_reproducible() -> None:
    """The interval is a pure function of ``(k, resamples, seed)``."""
    cohort = _fixture_cohort(CONNECTED_FIXTURES, relabellings=6, seed=common.SEED)
    first, _ = e1.cohort_psi("graph6", "levenshtein", cohort=cohort, resamples=128, seed=42)
    second, _ = e1.cohort_psi("graph6", "levenshtein", cohort=cohort, resamples=128, seed=42)
    third, _ = e1.cohort_psi("graph6", "levenshtein", cohort=cohort, resamples=128, seed=7)
    assert first.psi_ci == second.psi_ci
    assert first.psi_ci != third.psi_ci


# --------------------------------------------------------------------------
# Part C: the paired comparison
# --------------------------------------------------------------------------


def test_paired_comparisons_are_holm_corrected_over_the_whole_family() -> None:
    """Every adjusted p-value is at least its raw value, and the family is complete."""
    positions = list(range(20))
    per_graph = {
        "invariant": dict.fromkeys(positions, 0.0),
        "mild": {p: 0.1 + 0.01 * p for p in positions},
        "severe": {p: 0.8 + 0.01 * p for p in positions},
    }
    rows = e1.paired_comparisons("draw", per_graph)
    assert len(rows) == 3
    for row in rows:
        assert row.n_paired == 20
        if row.p_raw is not None:
            assert row.p_holm is not None and row.p_holm >= row.p_raw


def test_paired_comparison_of_two_invariant_backends_is_all_ties() -> None:
    """Two exactly invariant representations tie on every graph; ``p`` is absent.

    Reporting ``p = 1`` there would suggest a test was run on information that
    does not exist.  ``common.wilcoxon_paired`` returns ``None`` and this
    carries it through.
    """
    per_graph = {"a": dict.fromkeys(range(10), 0.0), "b": dict.fromkeys(range(10), 0.0)}
    rows = e1.paired_comparisons("draw", per_graph)
    assert len(rows) == 1
    assert rows[0].p_raw is None
    assert rows[0].p_holm is None
    assert rows[0].n_nonzero == 0


def test_paired_comparison_restricts_to_the_graphs_both_backends_encoded() -> None:
    """A ``SUITE1_ONLY`` backend pairs on its own graphs, not on padded zeros."""
    per_graph = {
        "wide": {p: 0.5 for p in range(10)},
        "narrow": {p: 0.1 for p in range(4)},
    }
    rows = e1.paired_comparisons("draw", per_graph)
    assert len(rows) == 1
    assert rows[0].n_paired == 4


def test_rank_biserial_direction_must_be_read_from_the_medians() -> None:
    """The effect size is a magnitude; the sign lives in the two medians.

    ``scipy.stats.wilcoxon`` returns ``min(W+, W-)`` under a two-sided
    alternative, so ``common.wilcoxon_paired``'s ``1 - 2W/total`` is
    non-negative whichever representation is worse.  The row therefore carries
    both medians, and a reader must use them rather than the sign of the
    effect size.
    """
    positions = list(range(12))
    forward = e1.paired_comparisons(
        "draw",
        {
            "a": dict.fromkeys(positions, 0.0),
            "b": {p: 0.5 + 0.01 * p for p in positions},
        },
    )[0]
    backward = e1.paired_comparisons(
        "draw",
        {
            "a": {p: 0.5 + 0.01 * p for p in positions},
            "b": dict.fromkeys(positions, 0.0),
        },
    )[0]
    assert forward.rank_biserial == backward.rank_biserial
    assert forward.median_psi_a < forward.median_psi_b
    assert backward.median_psi_a > backward.median_psi_b


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_the_grid_supplies_the_primary_distance_and_levenshtein_stands_in(tmp_path) -> None:  # noqa: ANN001
    """The three representations with no admitted distance are flagged, not hidden."""
    path = _grid_file(tmp_path, {"graph6": None, "nauty_graph6": "levenshtein"})
    resolved = e1._resolve_metrics(path, ["graph6", "nauty_graph6"])
    assert resolved["graph6"] == (e1.FALLBACK_METRIC, True)
    assert resolved["nauty_graph6"] == ("levenshtein", False)


def test_a_metric_override_marks_every_row_a_fallback(tmp_path) -> None:  # noqa: ANN001
    """The supplementary run cannot be mistaken for the grid's selection."""
    path = _grid_file(tmp_path, {"graph6": None, "nauty_graph6": "levenshtein"})
    resolved = e1._resolve_metrics(path, ["graph6", "nauty_graph6"], override="padded_hamming")
    assert resolved == {
        "graph6": ("padded_hamming", True),
        "nauty_graph6": ("padded_hamming", True),
    }


def test_psi_under_padded_hamming_is_larger_than_under_levenshtein() -> None:
    """The fallback the protocol fixed is the *conservative* one, measurably.

    Edit distance between two graphs of different order is dominated by the
    length difference, which inflates the denominator and deflates ``psi``.
    ``padded_hamming`` aligns on the positional frame instead and charges only
    cell disagreements.  Both are legitimate readings of the same failure; the
    protocol's choice understates it, so no result rests on picking the
    flattering metric.
    """
    graphs = [
        fixtures.to_networkx(fixtures.ALL_FIXTURES[name])
        for name in ("running_example", "running_example_minus_edge", "k33", "prism")
    ]
    cohort = e1.build_cohort("frames", graphs, relabellings=12, seed=common.SEED)
    edit, _ = e1.cohort_psi("adjacency", "levenshtein", cohort=cohort, resamples=64)
    positional, _ = e1.cohort_psi("adjacency", "padded_hamming", cohort=cohort, resamples=64)
    assert edit.psi is not None and positional.psi is not None
    assert positional.psi > edit.psi


def test_run_e1_part_a_only_needs_no_cohort(tmp_path) -> None:  # noqa: ANN001
    """Part A is a decision procedure over the atlas and touches no dataset."""
    path = _grid_file(tmp_path, {"graph6": None})
    record = e1.run_e1(path, backends=["graph6"], parts="A", max_n=4)
    assert record["parts"] == "A"
    assert record["metric_per_backend"]["graph6"]["fallback"] is True
    assert "cohort" not in record
    rows = record["exhaustive"]
    assert [row["n_nodes"] for row in rows] == [2, 3, 4]
    assert all(row["invariant_set_is_complete_graph"] for row in rows)


def test_cli_writes_a_record_with_the_frozen_header(tmp_path) -> None:  # noqa: ANN001
    """The CLI runs end to end and the header carries the frozen constants."""
    grid = _grid_file(tmp_path, {"adjacency": None})
    out = tmp_path / "e1.json"
    assert (
        e1.main(
            [
                "--grid",
                grid,
                "--out",
                str(out),
                "--parts",
                "A",
                "--backends",
                "adjacency",
                "--log-level",
                "WARNING",
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["experiment"] == "E1"
    assert payload["seed"] == common.SEED
    assert payload["relabellings"] == common.RELABELLINGS
    assert payload["resamples"] == common.RESAMPLES
    assert "exploratory" in payload["note"].lower()
    assert math.isfinite(payload["wall_seconds"])


@pytest.mark.integration
def test_the_pooled_draw_is_the_grid_s_s200() -> None:
    """Part B's pooled draw is re-derived from ``(ALL_DATASETS, 200, 42)``."""
    if not datasets.available_datasets():
        pytest.skip("cohorts not mounted")
    draws = e1._draws(datasets.available_datasets(), k=common.N_POOLED, seed=common.SEED)
    assert draws[0][0] == "pooled_S200"
    expected = datasets.pooled_stratified_sample(
        datasets.ALL_DATASETS, common.N_POOLED, seed=common.SEED
    )
    assert len(draws[0][1]) == len(expected)
    assert [g.number_of_nodes() for g in draws[0][1]] == [r.n_nodes for r in expected]
    assert [name for name, _ in draws[1:]] == list(datasets.available_datasets())


@pytest.mark.integration
def test_isomorphic_pairs_really_occur_in_a_real_draw() -> None:
    """The VF2 conditioning is not ceremony: real draws hold repeated graphs."""
    if "iam_letter_low" not in datasets.available_datasets():
        pytest.skip("iam_letter_low not mounted")
    cohort = datasets.load("iam_letter_low")
    graphs = [cohort.graphs[i] for i in cohort.sample(60, seed=common.SEED)]
    labels = e1.isomorphism_classes(graphs)
    assert len(set(labels)) < len(graphs)
