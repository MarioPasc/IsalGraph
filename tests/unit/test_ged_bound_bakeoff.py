"""Unit tests for ged_bound_bakeoff -- the T-27 GEDLIB bound bake-off harness.

The harness measures nine bound methods against 3.8 M certified exact-GED
values. Its failure mode is silence: GEDLIB returns ``0.00`` rather than raising
when the wrong accessor is called, and ``inf`` from a method that sets only one
end, so a whole matrix can fill with zeros and look like a result. These tests
exercise the guards that stand between that and a published number, and the
alignment assertions that stand between a misaligned ``graph_ids`` array and a
complete, plausible, entirely wrong bake-off.

They also pin the one rule that had to be *removed* from the design: a lower
bound of exactly zero on a pair with positive exact distance is legitimate, not
a defect, and must not raise.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from benchmarks.eval_setup.ged_bound_bakeoff import (
    BAKEOFF_CELLS,
    CELLS,
    COST_MODEL,
    DOMINATED_BY_BIPARTITE,
    BakeoffError,
    CellSpec,
    Corpus,
    all_zero_guard,
    assert_aligned,
    build_graphs,
    capability_probe,
    check_branch_equivalence,
    check_dominance,
    is_deterministic,
    label_graph,
    load_corpus,
    probe_pairs,
    read_bound,
    sample_pairs,
    summarise_cell,
    validity_refuted,
    write_failure,
    write_index,
)


def _gedlib_available() -> bool:
    """Report whether the GEDLIB bindings can be imported."""
    try:
        importlib.import_module("gklearn.gedlib.libraries_import")
        importlib.import_module("gklearn.gedlib.gedlibpy_gxl")
    except Exception:  # noqa: BLE001 -- any import failure means unavailable
        return False
    return True


requires_gedlib = pytest.mark.skipif(
    not _gedlib_available(),
    reason="GEDLIB bindings unavailable; export PYTHONPATH to the graphkit-learn checkout",
)


# --------------------------------------------------------------------------
# A synthetic dataset on disk, shaped exactly like the real artifacts
# --------------------------------------------------------------------------


def _true_ged(g1: nx.Graph, g2: nx.Graph) -> float:
    """Exact GED under the D6 unit cost model, from networkx.

    The fixture computes its ground truth rather than inventing it. An invented
    matrix would make every bound look like a validity violation, and the test
    would be measuring the fixture rather than the harness.
    """
    return float(
        nx.graph_edit_distance(
            g1,
            g2,
            node_subst_cost=lambda a, b: 0,
            node_del_cost=lambda a: 1,
            node_ins_cost=lambda a: 1,
            edge_subst_cost=lambda a, b: 0,
            edge_del_cost=lambda a: 1,
            edge_ins_cost=lambda a: 1,
        )
    )


def _write_dataset(
    root: Path,
    dataset: str = "toy",
    *,
    graph_ids: list[str] | None = None,
    truth_ids: list[str] | None = None,
    exported_node_counts: np.ndarray | None = None,
) -> Path:
    """Write a four-graph dataset in the layout ``load_corpus`` expects.

    Graph 0 is ``P4``, graph 1 ``C4``, graph 2 ``K3`` and graph 3 a single node,
    so node counts differ and the pair census is six. Pair ``(0, 3)`` is marked
    censored, which is what exercises the one-sided validity path.

    The ground truth is real: :func:`_true_ged` computes it with the same solver
    and the same cost model as the production census.
    """
    graphs = [nx.path_graph(4), nx.cycle_graph(4), nx.complete_graph(3), nx.empty_graph(1)]
    n = len(graphs)
    ids = graph_ids or [f"{dataset}_{k:04d}" for k in range(n)]

    n_nodes = np.array([g.number_of_nodes() for g in graphs], dtype=np.int32)
    exported_nodes = n_nodes if exported_node_counts is None else exported_node_counts
    n_edges = np.array([g.number_of_edges() for g in graphs], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(n_edges)]).astype(np.int64)
    edges = np.array(
        [[u for g in graphs for u, _ in g.edges()], [v for g in graphs for _, v in g.edges()]],
        dtype=np.int32,
    )

    exported = root / "exported"
    exported.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        exported / f"{dataset}.npz",
        graph_ids=np.array(ids),
        n_nodes=exported_nodes,
        n_edges=n_edges,
        edge_offsets=offsets,
        edges=edges,
    )

    ged = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            ged[i, j] = ged[j, i] = _true_ged(graphs[i], graphs[j])
    certified = np.ones((n, n), dtype=bool)
    lb = ged.copy()
    ub = ged.copy()
    # Pair (0, 3) is censored: exact becomes inf and the solver bracket widens
    # around the true value, which is what the one-sided M4 test consumes.
    true_03 = ged[0, 3]
    certified[0, 3] = certified[3, 0] = False
    lb[0, 3] = lb[3, 0] = true_03 - 2.0
    ub[0, 3] = ub[3, 0] = true_03 + 2.0
    ged[0, 3] = ged[3, 0] = np.inf

    computed = root / "source" / "GED_PRECOMPUTED" / "extended_merged_exact_ged" / "computed"
    computed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        computed / f"{dataset}.npz",
        ged_matrix=ged,
        lb_matrix=lb,
        ub_matrix=ub,
        certified_mask=certified,
        node_counts=n_nodes,
        edge_counts=n_edges,
        graph_ids=np.array(truth_ids or ids),
        metadata=np.asarray(json.dumps({"ged_method": "networkx"})),
    )

    lev_dir = root / "eval" / "levenshtein_matrices"
    lev_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("exhaustive", "greedy", "greedy_single"):
        matrix = np.arange(n * n, dtype=np.int32).reshape(n, n)
        np.savez_compressed(
            lev_dir / f"{dataset}_{variant}.npz",
            levenshtein_matrix=matrix,
            graph_ids=np.array(truth_ids or ids),
        )
    return root


@pytest.fixture
def toy_root(tmp_path: Path) -> Path:
    """A four-graph dataset written to a temporary data root."""
    return _write_dataset(tmp_path)


@pytest.fixture
def toy_corpus(toy_root: Path) -> Corpus:
    """The four-graph dataset loaded through the real loader."""
    return load_corpus(toy_root, "toy")


# --------------------------------------------------------------------------
# Method registry
# --------------------------------------------------------------------------


class TestRegistry:
    """The registry is what decides which accessor is read."""

    def test_cost_model_is_d6(self) -> None:
        assert list(COST_MODEL) == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    def test_twelve_cells_split_five_lower_seven_upper(self) -> None:
        lower = [c for c in BAKEOFF_CELLS if CELLS[c].end == "lower"]
        upper = [c for c in BAKEOFF_CELLS if CELLS[c].end == "upper"]
        assert len(BAKEOFF_CELLS) == 12
        assert set(lower) == {"BRANCH", "BRANCH_FAST", "BRANCH_TIGHT", "STAR", "HED"}
        assert set(upper) == {
            "BIPARTITE",
            "IPFP_MS",
            "REFINE_MS",
            "BP_BEAM_MS",
            "IPFP_DET",
            "REFINE_DET",
            "BP_BEAM_DET",
        }

    def test_a_cell_is_not_a_method(self) -> None:
        """Six cells share three GEDLIB methods under two configurations each."""
        assert CELLS["IPFP_MS"].method == CELLS["IPFP_DET"].method == "IPFP"
        assert CELLS["IPFP_MS"].options != CELLS["IPFP_DET"].options
        assert len({CELLS[c].method for c in BAKEOFF_CELLS}) == 9

    def test_every_cell_key_matches_its_own_name(self) -> None:
        for name, spec in CELLS.items():
            assert spec.cell == name

    def test_hed_is_a_lower_bound_with_optimal_edge_set_distances(self) -> None:
        """HED's default is identically zero under D6; OPTIMAL is what makes it useful."""
        spec = CELLS["HED"]
        assert spec.end == "lower"
        assert "--edge-set-distances OPTIMAL" in spec.options

    def test_no_campaign_option_string_carries_a_seed(self) -> None:
        """GEDLIB exposes no --seed; passing one raises Invalid option "seed"."""
        for spec in CELLS.values():
            assert "--seed" not in spec.options
            assert "--seed" not in spec.defaults

    def test_every_campaign_option_string_is_single_threaded(self) -> None:
        for spec in CELLS.values():
            assert "--threads 1" in spec.options

    def test_local_search_cells_are_flagged_randomised_at_defaults(self) -> None:
        for name in ("IPFP_MS", "REFINE_MS", "BP_BEAM_MS", "IPFP_DET", "REFINE_DET", "BP_BEAM_DET"):
            assert CELLS[name].randomised is True
        for name in ("BRANCH", "BRANCH_FAST", "BRANCH_TIGHT", "STAR", "BIPARTITE", "HED"):
            assert CELLS[name].randomised is False

    def test_multi_start_cells_ask_for_several_initial_solutions(self) -> None:
        for name in ("IPFP_MS", "REFINE_MS", "BP_BEAM_MS"):
            assert "--initial-solutions 10" in CELLS[name].options
            assert "--randomness PSEUDO" in CELLS[name].options

    def test_det_cells_start_from_the_bipartite_map(self) -> None:
        """This is what makes them provably dominated by the BIPARTITE cell."""
        for name in DOMINATED_BY_BIPARTITE:
            assert "--initialization-method BIPARTITE" in CELLS[name].options
            assert "--initial-solutions 1" in CELLS[name].options

    def test_a_pinned_cell_is_reported_deterministic_but_its_defaults_are_not(self) -> None:
        spec = CELLS["IPFP_MS"]
        assert is_deterministic(spec, spec.options) is True
        assert is_deterministic(spec, spec.defaults) is False
        assert is_deterministic(CELLS["BRANCH"], CELLS["BRANCH"].defaults) is True


# --------------------------------------------------------------------------
# Loading and alignment
# --------------------------------------------------------------------------


class TestAlignment:
    """A misaligned graph_ids array produces a complete and entirely wrong result."""

    def test_identical_ids_pass(self) -> None:
        ids = np.array(["a", "b", "c"])
        assert assert_aligned(ids, ids.copy(), dataset="toy", what="graph_ids") is None

    def test_permuted_ids_raise_even_though_the_sets_match(self) -> None:
        left = np.array(["a", "b", "c"])
        right = np.array(["a", "c", "b"])
        with pytest.raises(BakeoffError, match="same set in a different order"):
            assert_aligned(left, right, dataset="toy", what="graph_ids")

    def test_different_sets_raise(self) -> None:
        with pytest.raises(BakeoffError, match="different sets"):
            assert_aligned(
                np.array(["a", "b"]), np.array(["a", "z"]), dataset="toy", what="graph_ids"
            )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(BakeoffError, match="shapes differ"):
            assert_aligned(np.array(["a", "b"]), np.array(["a"]), dataset="toy", what="graph_ids")

    def test_load_corpus_raises_on_misaligned_graph_ids(self, tmp_path: Path) -> None:
        _write_dataset(
            tmp_path,
            graph_ids=["toy_0000", "toy_0001", "toy_0002", "toy_0003"],
            truth_ids=["toy_0001", "toy_0000", "toy_0002", "toy_0003"],
        )
        with pytest.raises(BakeoffError, match="graph_ids are not aligned"):
            load_corpus(tmp_path, "toy")

    def test_load_corpus_raises_on_node_count_disagreement(self, tmp_path: Path) -> None:
        _write_dataset(tmp_path, exported_node_counts=np.array([4, 4, 3, 2], dtype=np.int32))
        with pytest.raises(BakeoffError, match="n_nodes != node_counts"):
            load_corpus(tmp_path, "toy")


class TestGraphReconstruction:
    """CSR offsets and the edge array must not drift apart."""

    def test_graphs_rebuild_with_the_recorded_topology(self, toy_corpus: Corpus) -> None:
        assert [g.number_of_nodes() for g in toy_corpus.graphs] == [4, 4, 3, 1]
        assert [g.number_of_edges() for g in toy_corpus.graphs] == [3, 4, 3, 0]

    def test_every_node_and_edge_carries_a_string_label(self, toy_corpus: Corpus) -> None:
        """add_nx_graph rejects non-string attributes."""
        for graph in toy_corpus.graphs:
            for _, data in graph.nodes(data=True):
                assert isinstance(data["label"], str)
            for _, _, data in graph.edges(data=True):
                assert isinstance(data["label"], str)

    def test_inconsistent_edge_count_raises(self) -> None:
        exported = {
            "n_nodes": np.array([3], dtype=np.int32),
            "n_edges": np.array([5], dtype=np.int32),
            "edge_offsets": np.array([0, 2], dtype=np.int64),
            "edges": np.array([[0, 1], [1, 2]], dtype=np.int32),
        }
        with pytest.raises(BakeoffError, match="CSR offsets"):
            build_graphs(exported)


class TestPairCensus:
    """Every array in the wave is in triu order and is never compacted."""

    def test_index_length_is_n_choose_2(self, toy_corpus: Corpus) -> None:
        n = toy_corpus.n_graphs
        assert toy_corpus.n_pairs == n * (n - 1) // 2 == 6

    @pytest.mark.parametrize(
        ("n", "expected"),
        [(89, 3916), (769, 295296), (1180, 695610), (1253, 784378), (2059, 2118711)],
    )
    def test_census_counts_match_the_design_table(self, n: int, expected: int) -> None:
        assert n * (n - 1) // 2 == expected
        assert np.triu_indices(n, k=1)[0].size == expected

    def test_pair_order_is_row_major_upper_triangular(self, toy_corpus: Corpus) -> None:
        assert list(zip(toy_corpus.pair_i.tolist(), toy_corpus.pair_j.tolist(), strict=True)) == [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ]
        assert np.all(toy_corpus.pair_i < toy_corpus.pair_j)

    def test_censored_pairs_carry_infinite_exact_and_a_finite_bracket(
        self, toy_corpus: Corpus
    ) -> None:
        censored = ~toy_corpus.certified
        assert censored.sum() == 1
        assert np.isinf(toy_corpus.exact[censored]).all()
        assert np.isfinite(toy_corpus.exact_lb[censored]).all()
        assert np.isfinite(toy_corpus.exact_ub[censored]).all()

    def test_certified_bracket_collapses_onto_exact(self, toy_corpus: Corpus) -> None:
        certified = toy_corpus.certified
        assert np.array_equal(toy_corpus.exact_lb[certified], toy_corpus.exact[certified])
        assert np.array_equal(toy_corpus.exact_ub[certified], toy_corpus.exact[certified])


class TestIndexFile:
    """CONTRACTS section 2."""

    def test_index_carries_every_contracted_key_at_census_length(
        self, toy_root: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        path = write_index(toy_root, out, "toy")
        payload = np.load(path, allow_pickle=True)
        per_pair = (
            "pair_i",
            "pair_j",
            "exact",
            "exact_lb",
            "exact_ub",
            "certified",
            "n_max",
            "lev_exhaustive",
            "lev_greedy",
            "lev_greedy_single",
        )
        for key in per_pair:
            assert payload[key].shape == (6,), key
        for key in ("graph_ids", "node_counts", "edge_counts"):
            assert payload[key].shape == (4,), key
        assert payload["pair_i"].dtype == np.int32
        assert payload["exact"].dtype == np.float64
        assert payload["certified"].dtype == bool
        assert payload["lev_exhaustive"].dtype == np.int32

    def test_index_meta_is_json_with_null_method_fields(
        self, toy_root: Path, tmp_path: Path
    ) -> None:
        path = write_index(toy_root, tmp_path / "out", "toy")
        meta = json.loads(str(np.load(path, allow_pickle=True)["meta"]))
        assert meta["schema_version"] == 1
        assert meta["dataset"] == "toy"
        assert meta["n_pairs"] == 6
        assert meta["cost_model"] == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
        for key in ("cell", "method", "end", "options", "deterministic"):
            assert meta[key] is None, key

    def test_n_max_is_the_larger_node_count(self, toy_root: Path, tmp_path: Path) -> None:
        path = write_index(toy_root, tmp_path / "out", "toy")
        payload = np.load(path, allow_pickle=True)
        assert payload["n_max"].tolist() == [4, 4, 4, 4, 4, 3]


# --------------------------------------------------------------------------
# The guards
# --------------------------------------------------------------------------


class _FakeEnv:
    """A stand-in returning fixed values from each accessor."""

    def __init__(self, lower: float, upper: float) -> None:
        self._lower = lower
        self._upper = upper

    def get_lower_bound(self, a: int, b: int) -> float:
        return self._lower

    def get_upper_bound(self, a: int, b: int) -> float:
        return self._upper


class TestReadBound:
    """Guard one: every single read is checked."""

    def test_each_end_reads_only_its_own_accessor(self) -> None:
        env = _FakeEnv(lower=2.0, upper=7.0)
        assert read_bound(env, 0, 1, "lower", context="t") == 2.0
        assert read_bound(env, 0, 1, "upper", context="t") == 7.0

    def test_infinite_value_raises(self) -> None:
        """HED sets only the lower bound, so get_upper_bound() returns inf."""
        env = _FakeEnv(lower=1.0, upper=float("inf"))
        with pytest.raises(BakeoffError, match="infinite"):
            read_bound(env, 0, 1, "upper", context="t")

    def test_nan_raises(self) -> None:
        with pytest.raises(BakeoffError, match="NaN"):
            read_bound(_FakeEnv(lower=float("nan"), upper=1.0), 0, 1, "lower", context="t")

    def test_negative_raises(self) -> None:
        with pytest.raises(BakeoffError, match="negative"):
            read_bound(_FakeEnv(lower=-1.0, upper=1.0), 0, 1, "lower", context="t")

    def test_a_bare_zero_is_returned_not_raised(self) -> None:
        """A valid lower bound of zero is legitimate; only the guards above raise."""
        assert read_bound(_FakeEnv(lower=0.0, upper=3.0), 0, 1, "lower", context="t") == 0.0

    def test_unknown_end_raises(self) -> None:
        with pytest.raises(BakeoffError, match="unknown end"):
            read_bound(_FakeEnv(1.0, 1.0), 0, 1, "sideways", context="t")  # type: ignore[arg-type]


class TestAllZeroGuard:
    """Guard three: a whole column of zeros against positive truth is the trap."""

    def test_all_zero_against_positive_exact_raises(self) -> None:
        with pytest.raises(BakeoffError, match="accessor is wrong"):
            all_zero_guard(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]), context="toy/BIPARTITE")

    def test_all_zero_is_fine_when_every_exact_is_zero(self) -> None:
        all_zero_guard(np.zeros(3), np.zeros(3), context="toy/BRANCH")

    def test_all_zero_is_fine_when_every_exact_is_censored(self) -> None:
        all_zero_guard(np.zeros(3), np.full(3, np.inf), context="toy/BRANCH")

    def test_a_single_zero_among_positives_does_not_raise(self) -> None:
        """The rule this replaces: a loose lower bound legitimately returns zero."""
        all_zero_guard(np.array([0.0, 3.0, 5.0]), np.array([4.0, 3.0, 6.0]), context="toy/BRANCH")


class TestValidity:
    """M4, two-sided on certified pairs and one-sided on censored ones."""

    def test_lower_bound_refuted_only_above_the_solver_upper_bracket(self) -> None:
        values = np.array([3.0, 5.0, 5.001, 0.0])
        exact_lb = np.array([2.0, 2.0, 2.0, 2.0])
        exact_ub = np.array([5.0, 5.0, 5.0, 5.0])
        assert validity_refuted(values, exact_lb, exact_ub, "lower").tolist() == [
            False,
            False,
            True,
            False,
        ]

    def test_upper_bound_refuted_only_below_the_solver_lower_bracket(self) -> None:
        values = np.array([3.0, 2.0, 1.999, 99.0])
        exact_lb = np.array([2.0, 2.0, 2.0, 2.0])
        exact_ub = np.array([5.0, 5.0, 5.0, 5.0])
        assert validity_refuted(values, exact_lb, exact_ub, "upper").tolist() == [
            False,
            False,
            True,
            False,
        ]

    def test_on_a_certified_pair_the_one_sided_rule_is_the_two_sided_test(self) -> None:
        """Certified means exact_lb == exact_ub == exact, so one expression covers both."""
        exact = np.array([4.0, 4.0, 4.0])
        values = np.array([3.0, 4.0, 5.0])
        assert validity_refuted(values, exact, exact, "lower").tolist() == [False, False, True]
        assert validity_refuted(values, exact, exact, "upper").tolist() == [True, False, False]

    def test_a_censored_pair_refutes_a_lower_bound_only_past_its_upper_bracket(self) -> None:
        """The solver timed out with GED in [2, 7]; only LB > 7 is refuted."""
        lb = np.array([2.0, 2.0, 2.0])
        ub = np.array([7.0, 7.0, 7.0])
        assert validity_refuted(np.array([6.0, 7.0, 8.0]), lb, ub, "lower").tolist() == [
            False,
            False,
            True,
        ]
        assert validity_refuted(np.array([1.0, 2.0, 3.0]), lb, ub, "upper").tolist() == [
            True,
            False,
            False,
        ]

    def test_a_valid_bound_inside_the_censored_bracket_is_never_refuted(self) -> None:
        rng = np.random.default_rng(42)
        lb = np.full(200, 2.0)
        ub = np.full(200, 7.0)
        inside = rng.uniform(2.0, 7.0, size=200)
        assert not validity_refuted(inside, lb, ub, "lower").any()
        assert not validity_refuted(inside, lb, ub, "upper").any()


class TestUpperBoundSymmetry:
    """Section 3.6: every GEDLIB upper bound is direction-dependent."""

    def test_min_resolves_a_synthetic_asymmetry(self) -> None:
        forward = np.array([12.0, 5.0, 7.0, 3.0])
        reverse = np.array([14.0, 7.0, 5.0, 3.0])
        value = np.minimum(forward, reverse)
        assert value.tolist() == [12.0, 5.0, 5.0, 3.0]
        assert np.all(value <= forward)
        assert np.all(value <= reverse)

    def test_min_is_symmetric_under_swapping_the_orientations(self) -> None:
        forward = np.array([12.0, 5.0, 7.0])
        reverse = np.array([14.0, 7.0, 5.0])
        assert np.array_equal(np.minimum(forward, reverse), np.minimum(reverse, forward))

    def test_min_of_two_valid_upper_bounds_is_still_valid(self) -> None:
        exact = np.array([4.0, 4.0, 4.0])
        forward = np.array([9.0, 4.0, 6.0])
        reverse = np.array([5.0, 8.0, 4.0])
        value = np.minimum(forward, reverse)
        assert not validity_refuted(value, exact, exact, "upper").any()

    def test_the_disagreement_fraction_is_measurable(self) -> None:
        forward = np.array([1.0, 2.0, 3.0, 4.0])
        reverse = np.array([1.0, 3.0, 3.0, 9.0])
        assert float(np.mean(forward != reverse)) == 0.5


# --------------------------------------------------------------------------
# Sampling and summary
# --------------------------------------------------------------------------


class TestSampling:
    """Seeded samples must be reproducible and in canonical order."""

    def test_sample_is_seeded_and_reproducible(self) -> None:
        assert np.array_equal(sample_pairs(1000, 50, 42), sample_pairs(1000, 50, 42))

    def test_a_different_seed_gives_a_different_sample(self) -> None:
        assert not np.array_equal(sample_pairs(1000, 50, 42), sample_pairs(1000, 50, 7))

    def test_sample_is_sorted_and_without_repeats(self) -> None:
        sample = sample_pairs(1000, 50, 42)
        assert np.all(np.diff(sample) > 0)
        assert sample.size == 50

    def test_requesting_more_than_the_census_returns_the_census(self) -> None:
        assert np.array_equal(sample_pairs(10, 500, 42), np.arange(10))


class TestSummary:
    """Relative error is undefined where exact is zero -- design section 3.1."""

    def test_exact_zero_pairs_are_excluded_from_relative_error(self) -> None:
        corpus = Corpus(
            dataset="toy",
            graphs=[],
            graph_ids=np.array([]),
            node_counts=np.array([]),
            edge_counts=np.array([]),
            exact=np.array([0.0, 2.0, 4.0]),
            exact_lb=np.array([0.0, 2.0, 4.0]),
            exact_ub=np.array([0.0, 2.0, 4.0]),
            certified=np.array([True, True, True]),
            pair_i=np.array([0, 0, 1]),
            pair_j=np.array([1, 2, 2]),
        )
        values = np.array([0.0, 1.0, 2.0])
        summary = summarise_cell(values, corpus, CELLS["BRANCH"], np.zeros(3, dtype=bool))
        assert summary["n_m1_eligible"] == 2
        assert summary["mean_relative_error"] == pytest.approx(0.5)
        assert summary["n_zero_with_positive_exact"] == 0

    def test_a_legitimate_zero_lower_bound_is_counted_not_raised(self) -> None:
        corpus = Corpus(
            dataset="toy",
            graphs=[],
            graph_ids=np.array([]),
            node_counts=np.array([]),
            edge_counts=np.array([]),
            exact=np.array([4.0, 2.0]),
            exact_lb=np.array([4.0, 2.0]),
            exact_ub=np.array([4.0, 2.0]),
            certified=np.array([True, True]),
            pair_i=np.array([0, 0]),
            pair_j=np.array([1, 2]),
        )
        summary = summarise_cell(
            np.array([0.0, 2.0]), corpus, CELLS["BRANCH"], np.zeros(2, dtype=bool)
        )
        assert summary["n_zero_with_positive_exact"] == 1
        assert summary["n_refuted"] == 0


class TestFailureReport:
    """A failed cell writes JSON and never a partial .npz."""

    def test_failure_json_carries_the_contracted_keys(self, tmp_path: Path) -> None:
        path = write_failure(tmp_path, "toy", "IPFP_MS", BakeoffError("boom"), "--threads 1")
        payload = json.loads(path.read_text())
        assert set(payload) == {"dataset", "cell", "method", "reason", "traceback", "options"}
        assert payload["cell"] == "IPFP_MS"
        assert payload["method"] == "IPFP"
        assert "boom" in payload["reason"]
        assert not (tmp_path / "data" / "cells" / "toy__IPFP_MS.npz").exists()


# --------------------------------------------------------------------------
# GEDLIB itself
# --------------------------------------------------------------------------


@requires_gedlib
class TestCapabilityProbe:
    """Guard two: prove the accessor is live before the pair loop runs."""

    def test_probe_pairs_declare_their_true_distance(self) -> None:
        for name, g1, g2, exact in probe_pairs():
            measured = nx.graph_edit_distance(
                g1,
                g2,
                node_subst_cost=lambda a, b: 0,
                node_del_cost=lambda a: 1,
                node_ins_cost=lambda a: 1,
                edge_subst_cost=lambda a, b: 0,
                edge_del_cost=lambda a: 1,
                edge_ins_cost=lambda a: 1,
            )
            assert measured == pytest.approx(exact), name

    def test_probe_pairs_all_differ_in_degree_sequence(self) -> None:
        """A shared degree sequence is exactly what lets a valid bound return zero."""
        for name, g1, g2, _ in probe_pairs():
            d1 = sorted(d for _, d in g1.degree())
            d2 = sorted(d for _, d in g2.degree())
            assert d1 != d2, name

    @pytest.mark.parametrize("cell", sorted(CELLS))
    def test_every_configured_cell_passes_its_probe(self, cell: str) -> None:
        values = capability_probe(CELLS[cell])
        assert set(values) == {name for name, _, _, _ in probe_pairs()}
        assert all(v > 0.0 for v in values.values())

    def test_reading_an_upper_bound_method_as_a_lower_bound_raises(self) -> None:
        """The trap: get_lower_bound() on BIPARTITE returns 0.00 and does not raise."""
        wrong = CellSpec("WRONG", "BIPARTITE", "lower", "--threads 1")
        with pytest.raises(BakeoffError, match="wrong-accessor signature"):
            capability_probe(wrong)

    @pytest.mark.parametrize("method", ["IPFP", "REFINE", "BP_BEAM"])
    def test_reading_a_local_search_method_as_a_lower_bound_raises(self, method: str) -> None:
        with pytest.raises(BakeoffError, match="wrong-accessor signature"):
            capability_probe(CellSpec("WRONG", method, "lower", "--threads 1"))

    def test_reading_hed_as_an_upper_bound_raises_on_infinity(self) -> None:
        """hed.ipp sets only the lower bound, so the upper accessor returns inf."""
        with pytest.raises(BakeoffError, match="infinite"):
            capability_probe(CellSpec("WRONG", "HED", "upper", "--edge-set-distances OPTIMAL"))

    def test_hed_default_options_are_vacuous_under_d6(self) -> None:
        """Free edge substitution makes the default edge-set distance identically zero."""
        with pytest.raises(BakeoffError, match="wrong-accessor signature"):
            capability_probe(CellSpec("WRONG", "HED", "lower", ""))

    def test_gedlib_rejects_an_unknown_option_rather_than_dropping_it(self) -> None:
        with pytest.raises(Exception, match="Invalid option"):
            capability_probe(CellSpec("WRONG", "BRANCH", "lower", "--nonsense 1"))

    def test_gedlib_rejects_the_seed_option(self) -> None:
        """The design note's example pinned string named --seed, which does not exist."""
        with pytest.raises(Exception, match='Invalid option "seed"'):
            capability_probe(CellSpec("WRONG", "IPFP", "upper", "--seed 42"))


@requires_gedlib
class TestKnownValues:
    """Reproduce the recorded smoke test, and the zero that must not raise."""

    def test_p4_versus_c4_reproduces_the_recorded_bounds(self) -> None:
        values = capability_probe(CELLS["BRANCH_FAST"])
        assert values["P4_vs_C4"] == pytest.approx(1.0)

    def test_a_degree_preserving_non_isomorphic_pair_gives_a_zero_lower_bound(self) -> None:
        """C6 against two triangles: exact GED 4, every valid lower bound 0.

        This is the measurement that removed the design's "zero must raise"
        rule. Both graphs are 2-regular on six nodes with six edges, so under D6
        -- free node *and* edge substitution -- the identity-degree assignment
        costs nothing while the graphs are not isomorphic.
        """
        module = importlib.import_module("gklearn.gedlib.gedlibpy_gxl")
        c6 = nx.cycle_graph(6)
        two_triangles = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
        assert not nx.is_isomorphic(c6, two_triangles)
        assert sorted(d for _, d in c6.degree()) == sorted(d for _, d in two_triangles.degree())

        for method in ("BRANCH", "BRANCH_FAST", "BRANCH_TIGHT", "STAR"):
            env = module.GEDEnvGXL()
            a = env.add_nx_graph(label_graph(c6), "")
            b = env.add_nx_graph(label_graph(two_triangles), "")
            env.set_edit_cost("CONSTANT", edit_cost_constant=list(COST_MODEL))
            env.init(init_option="EAGER_WITHOUT_SHUFFLED_COPIES")
            env.set_method(method, "")
            env.init_method()
            env.run_method(a, b)
            value = read_bound(env, a, b, "lower", context=method)
            assert value == 0.0, f"{method} returned {value}"

        exact = nx.graph_edit_distance(
            c6,
            two_triangles,
            node_subst_cost=lambda x, y: 0,
            node_del_cost=lambda x: 1,
            node_ins_cost=lambda x: 1,
            edge_subst_cost=lambda x, y: 0,
            edge_del_cost=lambda x: 1,
            edge_ins_cost=lambda x: 1,
        )
        assert exact == pytest.approx(4.0)


@requires_gedlib
class TestCellEvaluation:
    """End to end on the four-graph dataset, against the contract."""

    def test_lower_bound_cell_omits_value_rev_entirely(
        self, toy_root: Path, tmp_path: Path
    ) -> None:
        from benchmarks.eval_setup.ged_bound_bakeoff import evaluate_cell

        summary = evaluate_cell(toy_root, tmp_path / "out", "toy", "BRANCH_FAST")
        payload = np.load(summary["path"], allow_pickle=True)
        assert "value_rev" not in payload
        assert np.array_equal(payload["value"], payload["value_fwd"])
        assert summary["n_violations"] == 0

    def test_upper_bound_cell_carries_both_orientations_and_their_min(
        self, toy_root: Path, tmp_path: Path
    ) -> None:
        from benchmarks.eval_setup.ged_bound_bakeoff import evaluate_cell

        summary = evaluate_cell(toy_root, tmp_path / "out", "toy", "BIPARTITE")
        payload = np.load(summary["path"], allow_pickle=True)
        assert "value_rev" in payload
        assert np.array_equal(
            payload["value"], np.minimum(payload["value_fwd"], payload["value_rev"])
        )

    def test_cell_meta_records_cell_method_end_and_options(
        self, toy_root: Path, tmp_path: Path
    ) -> None:
        from benchmarks.eval_setup.ged_bound_bakeoff import evaluate_cell

        summary = evaluate_cell(toy_root, tmp_path / "out", "toy", "HED")
        meta = json.loads(str(np.load(summary["path"], allow_pickle=True)["meta"]))
        assert meta["cell"] == "HED"
        assert meta["method"] == "HED"
        assert meta["end"] == "lower"
        assert "--edge-set-distances OPTIMAL" in meta["options"]
        assert meta["cost_model"] == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    def test_two_cells_of_one_method_are_distinguished_by_cell_not_method(
        self, toy_root: Path, tmp_path: Path
    ) -> None:
        from benchmarks.eval_setup.ged_bound_bakeoff import evaluate_cell

        out = tmp_path / "out"
        ms = evaluate_cell(toy_root, out, "toy", "IPFP_MS")
        det = evaluate_cell(toy_root, out, "toy", "IPFP_DET")
        assert ms["path"] != det["path"]
        for summary, cell in ((ms, "IPFP_MS"), (det, "IPFP_DET")):
            meta = json.loads(str(np.load(summary["path"], allow_pickle=True)["meta"]))
            assert meta["cell"] == cell
            assert meta["method"] == "IPFP"
            assert meta["deterministic"] is True

    @pytest.mark.parametrize("cell", sorted(CELLS))
    def test_every_cell_holds_its_bound_on_the_toy_dataset(
        self, toy_root: Path, tmp_path: Path, cell: str
    ) -> None:
        from benchmarks.eval_setup.ged_bound_bakeoff import evaluate_cell

        summary = evaluate_cell(toy_root, tmp_path / "out", "toy", cell)
        assert summary["n_violations"] == 0
        assert summary["n_refuted_certified"] == 0
        assert summary["n_refuted_censored"] == 0

    def test_the_parallel_path_reproduces_the_serial_one(
        self, toy_root: Path, tmp_path: Path
    ) -> None:
        from benchmarks.eval_setup.ged_bound_bakeoff import evaluate_cell

        serial = evaluate_cell(toy_root, tmp_path / "serial", "toy", "BRANCH", jobs=1)
        parallel = evaluate_cell(toy_root, tmp_path / "parallel", "toy", "BRANCH", jobs=2, chunk=2)
        left = np.load(serial["path"], allow_pickle=True)["value"]
        right = np.load(parallel["path"], allow_pickle=True)["value"]
        assert np.array_equal(left, right)


@requires_gedlib
class TestGates:
    """Two predictions the harness checks against itself, for free."""

    def test_branch_equals_branch_fast_end_to_end(self, toy_root: Path, tmp_path: Path) -> None:
        """Constant edge edit costs make the two provably equivalent."""
        from benchmarks.eval_setup.ged_bound_bakeoff import evaluate_cell, run_gates

        out = tmp_path / "out"
        for cell in ("BRANCH", "BRANCH_FAST"):
            evaluate_cell(toy_root, out, "toy", cell)
        report = run_gates(out, "toy")
        gate = report["p1_branch_equivalence"]
        assert gate["passes"] is True
        assert gate["n_equal"] == gate["n_pairs"]
        assert gate["max_abs_diff"] == 0.0

    def test_local_searches_never_exceed_their_bipartite_initialiser(
        self, toy_root: Path, tmp_path: Path
    ) -> None:
        from benchmarks.eval_setup.ged_bound_bakeoff import evaluate_cell, run_gates

        out = tmp_path / "out"
        for cell in ("BIPARTITE", *DOMINATED_BY_BIPARTITE):
            evaluate_cell(toy_root, out, "toy", cell)
        report = run_gates(out, "toy")
        for cell in DOMINATED_BY_BIPARTITE:
            assert report["dominance_vs_bipartite"][cell]["passes"] is True
            assert report["dominance_vs_bipartite"][cell]["n_violations"] == 0

    def test_gates_skip_rather_than_crash_when_a_cell_is_absent(self, tmp_path: Path) -> None:
        from benchmarks.eval_setup.ged_bound_bakeoff import run_gates

        report = run_gates(tmp_path / "empty", "toy")
        assert report["p1_branch_equivalence"].startswith("skipped")


class TestGateLogic:
    """The gate predicates, without GEDLIB."""

    def test_branch_equivalence_passes_on_identical_arrays(self) -> None:
        values = np.array([0.0, 1.0, 4.5])
        assert check_branch_equivalence(values, values.copy(), dataset="toy") is None

    def test_branch_equivalence_raises_at_zero_tolerance(self) -> None:
        """These are sums of integers under a unit cost model; no slack is allowed."""
        with pytest.raises(BakeoffError, match="P1 gate failed"):
            check_branch_equivalence(
                np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-12]), dataset="toy"
            )

    def test_branch_equivalence_raises_on_a_length_mismatch(self) -> None:
        with pytest.raises(BakeoffError, match="different lengths"):
            check_branch_equivalence(np.zeros(3), np.zeros(2), dataset="toy")

    def test_dominance_passes_when_the_local_search_only_improves(self) -> None:
        assert (
            check_dominance(
                np.array([3.0, 5.0, 5.0]),
                np.array([4.0, 5.0, 9.0]),
                dataset="toy",
                cell="REFINE_DET",
            )
            is None
        )

    def test_dominance_raises_when_the_local_search_is_worse(self) -> None:
        with pytest.raises(BakeoffError, match="dominance gate failed"):
            check_dominance(
                np.array([3.0, 6.0]), np.array([4.0, 5.0]), dataset="toy", cell="REFINE_DET"
            )
