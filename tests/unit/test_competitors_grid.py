"""T-04a: the (representation x distance) grid and its sampling protocol.

**Every test in this file corresponds to a defect measured in the shipped
T-04 harness** (`.claude/notes/review/tasks/T-04a-design.md` §1.3), and each
one fails on the base commit `7e96f4a`.  That is the point: the four defects
each produced a plausible number and no error, so a test that only passes on
the repaired code is the only evidence that the repair was needed.

Defect -> test:

1. ``--sample stratified-200`` drew 1,889 graphs, ``k`` **per dataset**, and
   allocated proportionally within a dataset ->
   :func:`test_stratified_sample_is_deleted`,
   :func:`test_stratum_quotas_send_the_remainder_to_the_largest_strata`,
   :func:`test_pooled_stratified_sample_draws_exactly_k_balanced_over_strata`.
2. The ``size_null`` *metric* won selection for all eleven representations ->
   :func:`test_size_null_metric_would_win_on_f6_and_is_refused_anyway`.
3. Encode failures were discarded and F1 was computed over whatever encoded
   -> :func:`test_encode_failures_are_counted_per_suite_with_their_type`,
   :func:`test_f1_denominator_is_pairs_among_encodable_graphs`.
4. F3 ran on ``graphs[:50]`` -- the smallest graphs of a stratum-ordered
   sample -- encoded each copy twice, and reported no skip count ->
   :func:`test_f3_subsample_is_stratum_balanced`,
   :func:`test_f3_attempted_plus_skipped_is_the_sample_size`.

Graphs are synthetic wherever the assertion does not genuinely need the real
cohort: the cohorts live on an external drive, and a slow test is a test that
gets skipped.  The real-cohort tests carry ``@pytest.mark.integration`` and
skip cleanly when ``datasets.available_datasets()`` is empty.
"""

from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING

import pytest

from isalgraph.competitors import datasets, fixtures, grid
from isalgraph.competitors.registry import available_backends, available_metrics

if TYPE_CHECKING:
    import networkx as nx

pytestmark = pytest.mark.unit

#: The one backend used for the merit/candidacy tests.  Chosen because it has
#: no third-party dependency, so these assertions cannot be silently skipped
#: on a machine without pynauty or grakel.
CANONICAL_BACKEND = "isalgraph_canonical"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _paths(orders: range) -> list[nx.Graph]:
    """Path graphs of the given orders.

    ``fixtures.py`` carries seven graphs on 2, 6 and 7 nodes, which is not
    enough node-count spread for the F4 and ``size_null`` assertions; these
    are the smallest synthetic family that gives one graph per order.
    """
    import networkx as nx

    return [nx.path_graph(n) for n in orders]


def _fixture_graphs(names: tuple[str, ...]) -> list[nx.Graph]:
    return [fixtures.to_networkx(fixtures.ALL_FIXTURES[name]) for name in names]


def _synthetic_records(per_stratum: dict[int, int]) -> tuple[datasets.SampleRecord, ...]:
    """Records covering the requested strata, with no disk access.

    The node count is the stratum's lower bound, so a record round-trips
    through :func:`datasets.stratum_of` to the stratum it claims.
    """
    out: list[datasets.SampleRecord] = []
    for stratum, count in sorted(per_stratum.items()):
        low = datasets.STRATA[stratum][0]
        for index in range(count):
            out.append(
                datasets.SampleRecord(
                    dataset="iam_letter_low" if stratum % 2 == 0 else "mutagenicity",
                    index=index,
                    n_nodes=low,
                    stratum=stratum,
                    suite="suite1" if stratum % 2 == 0 else "suite2",
                )
            )
    return tuple(out)


def _merit_criteria_hold(cell: grid.Cell) -> bool:
    """§3.5's F1-F4 criteria, **ignoring candidacy**.

    This is the rule the shipped harness applied.  It is reproduced here so a
    test can assert that ``size_null`` satisfies it and is refused anyway.
    """
    if not cell.applicable or cell.f1_defined_frac != 1.0:
        return False
    if cell.f2_violations and any(cell.f2_violations.values()):
        return False
    if cell.f3_invariant is None:
        return False
    got, total = (int(x) for x in cell.f3_invariant.split("/"))
    if total == 0 or got < total:
        return False
    if cell.f4_zero_mass is None or cell.f4_zero_mass > 0.5:
        return False
    return cell.f4_coeff_variation is not None and cell.f4_coeff_variation >= 1e-6


requires_cohorts = pytest.mark.skipif(
    not datasets.available_datasets(),
    reason="no exported cohort under $ISALGRAPH_COHORT_ROOT",
)


# --------------------------------------------------------------------------
# defect 1 -- the sample is one pooled, stratum-balanced draw of exactly k
# --------------------------------------------------------------------------


def test_stratified_sample_is_deleted() -> None:
    """The per-dataset proportional sampler is gone, not merely unused.

    It returned 1,889 graphs for ``k = 200`` and reproduced each dataset's own
    size distribution.  Leaving it importable leaves a second, wrong protocol
    one keystroke away from a paper number.
    """
    assert not hasattr(datasets, "stratified_sample")


def test_stratum_quotas_send_the_remainder_to_the_largest_strata() -> None:
    """§3.1 and §3.2's frozen allocations, both of them."""
    assert datasets.stratum_quotas(200, 6) == (33, 33, 33, 33, 34, 34)
    assert datasets.stratum_quotas(50, 6) == (8, 8, 8, 8, 9, 9)
    assert sum(datasets.stratum_quotas(200, 6)) == 200
    assert sum(datasets.stratum_quotas(50, 6)) == 50
    assert datasets.stratum_quotas(6, 6) == (1, 1, 1, 1, 1, 1)
    assert datasets.stratum_quotas(5, 0) == ()


@pytest.mark.parametrize(
    ("n_nodes", "expected"),
    [
        (1, None),
        (2, 0),
        (5, 0),
        (6, 1),
        (9, 1),
        (10, 2),
        (12, 2),
        (13, 3),
        (20, 3),
        (21, 4),
        (40, 4),
        (41, 5),
        (98, 5),
    ],
)
def test_stratum_of_bins_on_the_first_match(n_nodes: int, expected: int | None) -> None:
    """Every boundary of the six frozen strata, plus the ``n < 2`` gap."""
    assert datasets.stratum_of(n_nodes) == expected


def test_stratified_subsample_is_balanced_and_a_pure_function_of_its_arguments() -> None:
    """The draw hits the quotas and does not depend on what ran before it."""
    pool = _synthetic_records(dict.fromkeys(range(6), 100))
    first = datasets.stratified_subsample(pool, 200, seed=42)
    second = datasets.stratified_subsample(pool, 200, seed=42)

    assert len(first) == 200
    assert first == second, "the draw must be a pure function of (records, k, seed)"
    counts = [sum(1 for r in first if r.stratum == s) for s in range(6)]
    assert counts == [33, 33, 33, 33, 34, 34]

    other_seed = datasets.stratified_subsample(pool, 200, seed=7)
    assert other_seed != first, "a different seed must give a different draw"


def test_stratified_subsample_reports_a_shortfall_without_redistributing_it() -> None:
    """A thin stratum contributes all it has; the shortfall is not made up.

    Redistributing would quietly re-weight the sample towards whichever strata
    happen to be large, which is the failure the proportional sampler had.
    """
    pool = _synthetic_records({0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 2})
    drawn = datasets.stratified_subsample(pool, 200, seed=42)
    counts = [sum(1 for r in drawn if r.stratum == s) for s in range(6)]
    assert counts == [33, 33, 33, 33, 34, 2]
    assert len(drawn) == 168


def test_stratified_subsample_is_sorted_by_stratum_then_dataset_then_index() -> None:
    pool = _synthetic_records(dict.fromkeys(range(6), 20))
    drawn = datasets.stratified_subsample(pool, 60, seed=42)
    rank = {name: i for i, name in enumerate(datasets.ALL_DATASETS)}
    keys = [(r.stratum, rank[r.dataset], r.index) for r in drawn]
    assert keys == sorted(keys)


@pytest.mark.integration
@requires_cohorts
def test_pooled_stratified_sample_draws_exactly_k_balanced_over_strata() -> None:
    """A1: ``S200`` is 200 graphs, reproducible from ``(ALL_DATASETS, 200, 42)``."""
    drawn = datasets.pooled_stratified_sample(datasets.ALL_DATASETS, 200, seed=42)
    assert len(drawn) == 200
    counts = [sum(1 for r in drawn if r.stratum == s) for s in range(6)]
    assert counts == [33, 33, 33, 33, 34, 34]

    again = datasets.pooled_stratified_sample(datasets.ALL_DATASETS, 200, seed=42)
    assert drawn == again

    # Every record addresses the graph it describes.
    for record in drawn:
        graph = datasets.load(record.dataset).graphs[record.index]
        assert graph.number_of_nodes() == record.n_nodes
        assert datasets.stratum_of(record.n_nodes) == record.stratum
        assert datasets.suite_of(record.dataset) == record.suite


# --------------------------------------------------------------------------
# defect 2 -- a metric that reads no part of the encoding is never primary
# --------------------------------------------------------------------------


def test_size_null_metric_would_win_on_f6_and_is_refused_anyway() -> None:
    """A3, the defect that mattered most.

    On this fixture ``size_null`` satisfies **every** merit criterion §3.4
    names -- defined on 100 % of pairs, no F2 violation, invariant on every
    attempted graph, non-degenerate -- and is by an order of magnitude the
    cheapest cell, so the shipped rule's ``min`` on ``(F6, name)`` names it.
    The repaired rule refuses it on ``consumes``, before any measurement.
    """
    graphs = _paths(range(2, 13))
    suites = ["suite1"] * len(graphs)
    cache = grid.encode_sample(CANONICAL_BACKEND, graphs, suites)
    f3cache = grid.encode_f3(CANONICAL_BACKEND, graphs, seed=42)
    cells = [
        grid.measure_cell(CANONICAL_BACKEND, metric, cache, f3cache, seed=42)
        for metric in ("levenshtein", "size_null")
    ]
    by_metric = {cell.metric: cell for cell in cells}

    # The premise: size_null passes on merit and is the cheapest.
    assert _merit_criteria_hold(by_metric["size_null"])
    assert _merit_criteria_hold(by_metric["levenshtein"])
    assert by_metric["size_null"].f6_ms_per_pair is not None
    assert by_metric["levenshtein"].f6_ms_per_pair is not None
    assert by_metric["size_null"].f6_ms_per_pair < by_metric["levenshtein"].f6_ms_per_pair

    # The rule: refused on `consumes`, with the reason CONTRACTS §4 fixes.
    assert by_metric["size_null"].candidate is False
    assert by_metric["size_null"].passes_selection is False
    assert by_metric["size_null"].excluded_because == grid.ORDER_EXCLUSION
    assert by_metric["levenshtein"].candidate is True
    assert select_metric(cells) == "levenshtein"


def select_metric(cells: list[grid.Cell]) -> str | None:
    """``select_primary`` for a single-backend cell list."""
    return grid.select_primary(cells)[cells[0].backend]


def test_no_non_candidate_metric_is_ever_selected_anywhere() -> None:
    """The rule holds for **every** registered representation, not just one.

    ``size_null`` and ``levenshtein_char`` both read something other than the
    representation -- the node count and the character rendering -- and
    §3.4's candidate set is ``{"symbols", "frame", "features"}``.
    """
    graphs = _fixture_graphs(fixtures.CONNECTED_FIXTURES)
    suites = ["suite1"] * len(graphs)
    cells, _ = grid.run_grid(graphs, suites, graphs, seed=42, backends=None, metrics=None)
    primary = grid.select_primary(cells)
    assert "size_null" not in set(primary.values())
    assert "levenshtein_char" not in set(primary.values())

    for cell in cells:
        metric_consumes_representation = cell.metric not in {"size_null", "levenshtein_char"}
        if not metric_consumes_representation:
            assert cell.candidate is False
            assert cell.passes_selection is False


def test_baseline_backend_keeps_its_existing_exclusion() -> None:
    """``Capability.BASELINE`` on the *backend* still means never primary."""
    graphs = _paths(range(2, 13))
    suites = ["suite1"] * len(graphs)
    cache = grid.encode_sample("size_null", graphs, suites)
    f3cache = grid.encode_f3("size_null", graphs, seed=42)
    cell = grid.measure_cell("size_null", "levenshtein", cache, f3cache, seed=42)
    assert cell.candidate is False
    assert cell.excluded_because == grid.BASELINE_EXCLUSION
    assert grid.select_primary([cell]) == {"size_null": None}


def test_selection_reason_names_the_failing_criterion_for_every_candidate() -> None:
    """A7: an empty selection is a printed absence, never a missing key."""
    graphs = _fixture_graphs(("k33", "prism", "running_example"))
    suites = ["suite1"] * len(graphs)
    cells, _ = grid.run_grid(graphs, suites, graphs, seed=42, backends=("adjacency",))
    primary = grid.select_primary(cells)
    reasons = grid.selection_reasons(cells, primary)
    assert set(reasons) == set(primary)
    assert all(reasons[backend] for backend in reasons)


# --------------------------------------------------------------------------
# defect 3 -- encode failures are counted, and F1 is over encodable pairs
# --------------------------------------------------------------------------


def test_encode_failures_are_counted_per_suite_with_their_type() -> None:
    """A4: F0 is a property of the representation and is split by suite.

    ``c4_plus_k3_disjoint`` is the documented asymmetry: IsalGraph raises on
    it, the ``n^2`` family does not.  The exception type is carried, since
    "9 graphs failed" and "9 graphs timed out" are different findings.
    """
    graphs = _fixture_graphs(("running_example", "c4_plus_k3_disjoint", "k33", "prism"))
    suites = ["suite1", "suite2", "suite1", "suite2"]
    cache = grid.encode_sample(CANONICAL_BACKEND, graphs, suites)

    assert cache.f0["overall"] == {
        "attempted": 4,
        "encodable": 3,
        "frac": 0.75,
        "errors": {"DisconnectedGraphError": 1},
    }
    assert cache.f0["suite1"] == {
        "attempted": 2,
        "encodable": 2,
        "frac": 1.0,
        "errors": {},
    }
    assert cache.f0["suite2"] == {
        "attempted": 2,
        "encodable": 1,
        "frac": 0.5,
        "errors": {"DisconnectedGraphError": 1},
    }
    assert len(cache.items) == 3 == len(cache.n_nodes)


def test_a_non_competitor_exception_is_still_counted_not_raised() -> None:
    """min-DFS raises a plain ``ValueError`` on a disconnected graph.

    Catching only ``CompetitorError`` would let it escape and abort the run,
    so F0 counts by ``type(exc).__name__`` over ``Exception``.
    """
    graphs = _fixture_graphs(("running_example", "c4_plus_k3_disjoint"))
    cache = grid.encode_sample("min_dfs", graphs, ["suite1", "suite1"])
    assert cache.f0["overall"]["encodable"] == 1
    assert sum(cache.f0["overall"]["errors"].values()) == 1


def test_all_suite_keys_are_present_even_when_empty() -> None:
    graphs = _fixture_graphs(("running_example", "k33"))
    cache = grid.encode_sample("adjacency", graphs, ["suite1", "suite1"])
    assert set(cache.f0) == {"overall", "suite1", "suite2"}
    assert cache.f0["suite2"]["attempted"] == 0


def test_f1_denominator_is_pairs_among_encodable_graphs() -> None:
    """F1 is a property of the **distance**, over the pairs that can exist.

    Three of four graphs encode, so the denominator is ``C(3,2) = 3`` and not
    ``C(4,2) = 6``.  The shipped harness recorded no denominator at all, so a
    reader could not tell the two apart.
    """
    graphs = _fixture_graphs(("running_example", "c4_plus_k3_disjoint", "k33", "prism"))
    suites = ["suite1"] * len(graphs)
    cache = grid.encode_sample(CANONICAL_BACKEND, graphs, suites)
    f3cache = grid.encode_f3(CANONICAL_BACKEND, graphs, seed=42)
    cell = grid.measure_cell(CANONICAL_BACKEND, "levenshtein", cache, f3cache, seed=42)
    assert cell.f1_n_pairs == 3
    assert cell.f1_defined_frac == 1.0


def test_encoding_is_shared_across_the_metrics_of_a_row() -> None:
    """One cache per backend, so a row cannot disagree about what encoded."""
    graphs = _fixture_graphs(("running_example", "c4_plus_k3_disjoint", "k33", "prism"))
    suites = ["suite1"] * len(graphs)
    cache = grid.encode_sample(CANONICAL_BACKEND, graphs, suites)
    f3cache = grid.encode_f3(CANONICAL_BACKEND, graphs, seed=42)
    denominators = {
        grid.measure_cell(CANONICAL_BACKEND, m, cache, f3cache, seed=42).f1_n_pairs
        for m in ("levenshtein", "hamming", "size_null")
    }
    assert denominators == {3}


# --------------------------------------------------------------------------
# defect 4 -- F3 on a stratified sub-sample, one encode per copy, skips counted
# --------------------------------------------------------------------------


def test_f3_subsample_is_stratum_balanced() -> None:
    """A5: ``S50`` is §3.2's ``[8, 8, 8, 8, 9, 9]``, not ``graphs[:50]``.

    On a stratum-ordered sample ``graphs[:50]`` is the whole of the smallest
    stratum plus part of the next, so F3 was measured on the small graphs
    only -- precisely where an order-dependent format is most likely to look
    invariant by luck.
    """
    pool = _synthetic_records(dict.fromkeys(range(6), 40))
    s200 = datasets.stratified_subsample(pool, 200, seed=42)
    s50 = datasets.stratified_subsample(s200, 50, seed=42)
    counts = [sum(1 for r in s50 if r.stratum == s) for s in range(6)]
    assert counts == [8, 8, 8, 8, 9, 9]
    assert set(s50).issubset(set(s200)), "S50 must be drawn FROM S200"


@pytest.mark.integration
@requires_cohorts
def test_f3_subsample_of_the_real_s200_is_stratum_balanced() -> None:
    s200 = datasets.pooled_stratified_sample(datasets.ALL_DATASETS, 200, seed=42)
    s50 = datasets.stratified_subsample(s200, 50, seed=42, order=datasets.ALL_DATASETS)
    counts = [sum(1 for r in s50 if r.stratum == s) for s in range(6)]
    assert counts == [8, 8, 8, 8, 9, 9]
    assert set(s50).issubset(set(s200))


def test_f3_attempted_plus_skipped_is_the_sample_size() -> None:
    """A5: a graph the backend raises on is *skipped*, never non-invariant.

    Without the skip count, ``0/50`` means both "invariant on nothing" and
    "never ran", which are opposite findings.
    """
    cycled = itertools.islice(itertools.cycle(fixtures.ALL_FIXTURES), grid.F3_GRAPHS)
    graphs = _fixture_graphs(tuple(cycled))
    assert len(graphs) == 50

    f3cache = grid.encode_f3(CANONICAL_BACKEND, graphs, seed=42)
    assert f3cache.skipped > 0, "the disjoint fixture must make IsalGraph raise"
    assert f3cache.attempted + f3cache.skipped == 50

    cache = grid.encode_sample(CANONICAL_BACKEND, graphs, ["suite1"] * 50)
    cell = grid.measure_cell(CANONICAL_BACKEND, "levenshtein", cache, f3cache, seed=42)
    assert cell.f3_skipped == f3cache.skipped
    attempted = int(cell.f3_invariant.split("/")[1]) if cell.f3_invariant else -1
    assert attempted == f3cache.attempted
    assert attempted + (cell.f3_skipped or 0) == 50


def test_f3_is_evaluated_on_one_encoding_per_copy_shared_by_every_metric() -> None:
    """The copies are encoded once per backend and reused across the row."""
    graphs = _fixture_graphs(("running_example", "k33", "prism"))
    f3cache = grid.encode_f3(CANONICAL_BACKEND, graphs, seed=42)
    assert f3cache.attempted == 3
    assert all(len(copies) == grid.F3_RELABELLINGS for _, copies in f3cache.entries)


def test_f3_relabelling_can_actually_fail_for_an_order_dependent_format() -> None:
    """An F3 harness that cannot fail is worthless.

    ``adjacency`` reads the incident node order, so a genuine relabelling
    must break it.  If this ever passes 3/3, the relabeller has stopped
    relabelling (finding 13).
    """
    graphs = _fixture_graphs(("running_example", "k33", "prism"))
    suites = ["suite1"] * len(graphs)
    cache = grid.encode_sample("adjacency", graphs, suites)
    f3cache = grid.encode_f3("adjacency", graphs, seed=42)
    cell = grid.measure_cell("adjacency", "levenshtein", cache, f3cache, seed=42)
    assert cell.f3_invariant is not None
    invariant, attempted = (int(x) for x in cell.f3_invariant.split("/"))
    assert attempted == 3
    assert invariant < attempted


# --------------------------------------------------------------------------
# the grid as a whole
# --------------------------------------------------------------------------


def test_every_registered_cell_is_measured_and_printed() -> None:
    """A2: all 66 cells attempted and present, failures included."""
    graphs = _fixture_graphs(fixtures.CONNECTED_FIXTURES)
    suites = ["suite1"] * len(graphs)
    backends = available_backends(include_baseline=True)
    metrics = available_metrics()
    cells, f0 = grid.run_grid(graphs, suites, graphs, seed=42)

    assert len(cells) == len(backends) * len(metrics)
    assert {(c.backend, c.metric) for c in cells} == {(b, m) for b in backends for m in metrics}
    assert set(f0) == set(backends)
    if len(backends) == 11 and len(metrics) == 6:
        assert len(cells) == 66

    undefined = next(c for c in cells if (c.backend, c.metric) == ("sparse6", "padded_hamming"))
    assert undefined.f1_defined_frac == 0.0
    assert undefined.passes_selection is False


def test_f6_advisory_flag_never_gates_selection() -> None:
    """§3.5: F6 is the tie-break, never a criterion.

    A cell over the advisory limit still passes if F1-F4 hold.
    """
    cell = grid.Cell(
        backend="x",
        metric="levenshtein",
        f1_defined_frac=1.0,
        f2_violations={"identity": 0, "symmetry": 0, "triangle": 0},
        f3_invariant="5/5",
        f4_zero_mass=0.0,
        f4_coeff_variation=0.5,
        f6_ms_per_pair=10.0,
        f6_over_advisory_limit=True,
    )
    grid._apply_selection_rule(cell, None)
    assert cell.passes_selection is True
    assert cell.excluded_because is None


def test_f6_large_pair_reading_is_restricted_to_both_graphs_at_least_21_nodes() -> None:
    """``f6_ms_per_pair_large`` is ``None`` when no pair qualifies."""
    small = _paths(range(2, 13))
    cache = grid.encode_sample("adjacency", small, ["suite1"] * len(small))
    f3cache = grid.encode_f3("adjacency", small[:3], seed=42)
    cell = grid.measure_cell("adjacency", "levenshtein", cache, f3cache, seed=42)
    assert cell.f6_ms_per_pair is not None
    assert cell.f6_ms_per_pair_large is None

    large = _paths(range(21, 32))
    cache = grid.encode_sample("adjacency", large, ["suite2"] * len(large))
    f3cache = grid.encode_f3("adjacency", large[:3], seed=42)
    cell = grid.measure_cell("adjacency", "levenshtein", cache, f3cache, seed=42)
    assert cell.f6_ms_per_pair_large is not None


def test_grid_run_is_reproducible_apart_from_the_timings() -> None:
    """Two runs at one seed give the same measurements."""
    graphs = _fixture_graphs(fixtures.CONNECTED_FIXTURES)
    suites = ["suite1"] * len(graphs)
    first, _ = grid.run_grid(graphs, suites, graphs, seed=42, backends=("adjacency", "graph6"))
    second, _ = grid.run_grid(graphs, suites, graphs, seed=42, backends=("adjacency", "graph6"))

    def stripped(cells: list[grid.Cell]) -> list[tuple[object, ...]]:
        return [
            (
                c.backend,
                c.metric,
                c.applicable,
                c.candidate,
                c.f1_defined_frac,
                c.f1_n_pairs,
                c.f2_violations,
                c.f3_invariant,
                c.f3_skipped,
                c.f4_zero_mass,
                c.f4_coeff_variation,
                c.passes_selection,
                c.excluded_because,
            )
            for c in cells
        ]

    assert stripped(first) == stripped(second)


def test_an_unavailable_backend_produces_cells_rather_than_an_exception() -> None:
    """A missing backend is a printed row, not a crash."""
    graphs = _fixture_graphs(("running_example", "k33"))
    cache = grid.encode_sample("no_such_backend", graphs, ["suite1", "suite1"])
    assert cache.available is False
    assert cache.f0["overall"]["attempted"] == 2
    assert cache.f0["overall"]["encodable"] == 0
    f3cache = grid.encode_f3("no_such_backend", graphs, seed=42)
    assert f3cache.skipped == 2
    cell = grid.measure_cell("no_such_backend", "levenshtein", cache, f3cache, seed=42)
    assert cell.applicable is False
    assert cell.candidate is False


def test_sample_block_matches_the_frozen_json_shape() -> None:
    """CONTRACTS §2's ``sample`` block, which the report indexes by name."""
    records = _synthetic_records({0: 3, 4: 2})
    block = grid.sample_block(
        "pooled_stratified",
        records,
        k=5,
        seed=42,
        names=datasets.ALL_DATASETS,
    )
    assert block["kind"] == "pooled_stratified"
    assert block["k"] == 5
    assert block["seed"] == 42
    assert block["per_stratum"] == {"0": 3, "1": 0, "2": 0, "3": 0, "4": 2, "5": 0}
    assert set(block["per_dataset"]) == set(datasets.ALL_DATASETS)
    assert block["strata"] == [list(bounds) for bounds in datasets.STRATA]
    assert block["n_min"] == 2
    assert block["n_max"] == 21
    assert len(block["records"]) == 5
    assert set(block["records"][0]) == {"dataset", "index", "n_nodes", "stratum", "suite"}


@pytest.mark.integration
@requires_cohorts
def test_dryrun_cli_writes_the_frozen_payload(tmp_path: object) -> None:
    """A2 and A7 end to end, on the smallest real cohort."""
    import pathlib

    assert isinstance(tmp_path, pathlib.Path)
    out = tmp_path / "grid.json"
    dataset = "iam_letter_low"
    if dataset not in datasets.available_datasets():
        pytest.skip(f"{dataset} not exported")
    code = grid.main(
        ["--sample", "dryrun-20", "--dataset", dataset, "--seed", "42", "--out", str(out)]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["protocol"] == "T-04a"
    assert set(payload) >= {
        "protocol",
        "seed",
        "n_graphs",
        "sample",
        "f3_sample",
        "backends",
        "metrics",
        "f0",
        "cells",
        "primary_distance",
        "selection_reason",
        "f5",
    }
    assert len(payload["cells"]) == len(payload["backends"]) * len(payload["metrics"])
    assert set(payload["primary_distance"]) == set(payload["backends"])
    assert set(payload["selection_reason"]) == set(payload["backends"])
    assert "size_null" not in set(payload["primary_distance"].values())
    for block in payload["f0"].values():
        assert set(block) == {"overall", "suite1", "suite2"}
    assert payload["f5"].startswith("NOT COMPUTED HERE")
