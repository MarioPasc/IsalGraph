"""Unit tests for the T-27 GED bound bake-off analysis.

Follows ``tests/unit/test_ged_bounds.py``. Everything runs against the
synthetic fixture built from ``CONTRACTS.md``, so the suite never waits
on track A and never touches GEDLIB.

The properties under test are the ones a wrong answer would hide:
seed-42 bootstrap reproducibility, the pair set a graph-level resample
induces, one-sided validity on censored pairs, exclusion and counting of
``exact == 0`` pairs, and Holm correction against a hand-computed case.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from scipy import stats

from benchmarks.real_data.eval_setup import ged_bakeoff_analysis as bakeoff

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the synthetic fixture once for the module."""
    root = tmp_path_factory.mktemp("t27")
    bakeoff.build_synthetic_fixture(root)
    return root


@pytest.fixture(scope="module")
def bundles(fixture_root: Path) -> dict[str, bakeoff.DatasetBundle]:
    """Load every synthetic dataset."""
    return bakeoff.load_bundles(fixture_root, list(bakeoff.DATASETS))


@pytest.fixture(scope="module")
def linux_bundle(bundles: dict[str, bakeoff.DatasetBundle]) -> bakeoff.DatasetBundle:
    """The smallest bundle, for cheap per-test work."""
    return bundles["linux"]


# ---------------------------------------------------------------------------
# Rosters -- the Holm family size must come from the roster
# ---------------------------------------------------------------------------


def test_lower_end_carries_five_methods_including_hed() -> None:
    """HED is a lower bound once its edge-set distances are set to OPTIMAL."""
    assert bakeoff.methods_for_end("lower") == (
        "BRANCH",
        "BRANCH_FAST",
        "BRANCH_TIGHT",
        "STAR",
        "HED",
    )
    assert bakeoff.end_of_method("HED") == "lower"


def test_upper_end_competitors_are_the_multi_start_arm() -> None:
    """Only the ``_MS`` arm competes; ``_DET`` is measured but never selected."""
    assert bakeoff.methods_for_end("upper") == ("IPFP_MS", "REFINE_MS", "BIPARTITE", "BP_BEAM_MS")
    assert bakeoff.cells_for_end("upper") == (
        "IPFP_MS",
        "REFINE_MS",
        "BIPARTITE",
        "BP_BEAM_MS",
        "IPFP_DET",
        "REFINE_DET",
        "BP_BEAM_DET",
    )
    assert bakeoff.cells_for_end("lower") == bakeoff.methods_for_end("lower")
    for cell in bakeoff.UPPER_COMPANION_METHODS:
        assert bakeoff.end_of_method(cell) == "upper"


def test_the_grid_is_twelve_cells_per_dataset() -> None:
    """Five lower bounds and seven upper cells, 60 cells over five datasets."""
    per_dataset = len(bakeoff.cells_for_end("lower")) + len(bakeoff.cells_for_end("upper"))
    assert per_dataset == 12
    assert per_dataset * len(bakeoff.DATASETS) == 60


def test_unknown_method_raises() -> None:
    """A method in neither roster is an error, not a silent default."""
    with pytest.raises(bakeoff.BakeoffAnalysisError):
        bakeoff.end_of_method("ANCHOR_AWARE_GED")
    with pytest.raises(bakeoff.BakeoffAnalysisError):
        bakeoff.methods_for_end("middle")


def test_holm_family_size_is_derived_not_hardcoded(linux_bundle: bakeoff.DatasetBundle) -> None:
    """Five lower bounds give ten comparisons; four upper bounds give six."""
    lower = bakeoff.pairwise_significance("lower", "linux", linux_bundle.cells, linux_bundle.index)
    upper = bakeoff.pairwise_significance("upper", "linux", linux_bundle.cells, linux_bundle.index)
    assert lower["family_size"] == 10
    assert upper["family_size"] == 6


# ---------------------------------------------------------------------------
# Rank machinery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        np.array([3.0, 1.0, 2.0]),
        np.array([1.0, 1.0, 2.0, 2.0, 2.0, 5.0]),
        np.zeros(7),
        np.arange(50.0),
    ],
)
def test_midranks_match_scipy_rankdata(values: np.ndarray) -> None:
    """The counting-sort fast path must be exact, not approximate."""
    np.testing.assert_allclose(bakeoff.midranks(values), stats.rankdata(values))


def test_midranks_falls_back_for_non_integral_input() -> None:
    """Non-integral data leaves the fast path and still ranks correctly."""
    values = np.array([0.5, 0.25, 0.75, 0.25])
    np.testing.assert_allclose(bakeoff.midranks(values), stats.rankdata(values))


# ---------------------------------------------------------------------------
# factorize -- the precondition removal, exercised on HED's real granularity
# ---------------------------------------------------------------------------

#: The 8 distinct values HED returns with ``--edge-set-distances OPTIMAL``
#: over all 3,916 LINUX pairs: it charges each edge at both endpoints and
#: halves, so the LSAPE optimum lands on quarter-integers in [0, 1.75].
HED_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])


def test_factorize_is_a_strictly_monotone_dense_coding() -> None:
    """Codes must preserve order exactly, or every rank statistic is wrong."""
    values = np.array([1.75, 0.0, 0.5, 0.5, 0.25])
    codes = bakeoff.factorize(values)
    assert codes.tolist() == [3, 0, 2, 2, 1]
    assert codes.dtype == np.int64
    assert set(codes.tolist()) == set(range(4))
    order = np.argsort(values, kind="stable")
    assert np.all(np.diff(codes[order]) >= 0)


def test_factorized_hed_values_take_the_counting_sort_fast_path() -> None:
    """HED is quarter-integral, so raw values would leave the fast path.

    The fast path is detected by patching out the scipy fallback: if
    ``midranks`` reaches ``rankdata`` the call raises. Ranking the raw
    quarter-integers must fall back; ranking their codes must not.
    """
    rng = np.random.default_rng(0)
    values = rng.choice(HED_VALUES, size=5000)

    def _forbidden(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("midranks left the counting-sort fast path")

    codes = bakeoff.factorize(values)
    with mock.patch.object(bakeoff.stats, "rankdata", _forbidden):
        fast = bakeoff.midranks(codes)
    np.testing.assert_allclose(fast, stats.rankdata(values))

    with (
        mock.patch.object(bakeoff.stats, "rankdata", _forbidden),
        pytest.raises(AssertionError, match="fast path"),
    ):
        bakeoff.midranks(values)


def test_spearman_on_hed_granularity_matches_scipy_exactly() -> None:
    """Correlating codes must equal correlating values, not approximate it."""
    rng = np.random.default_rng(7)
    hed = rng.choice(HED_VALUES, size=4000)
    exact = np.floor(hed * 3) + rng.integers(0, 3, size=4000)
    via_codes = bakeoff.spearman_from_ranks(
        bakeoff.midranks(bakeoff.factorize(hed)),
        bakeoff.midranks(bakeoff.factorize(exact)),
    )
    np.testing.assert_allclose(via_codes, stats.spearmanr(hed, exact).statistic, rtol=1e-12)


def test_fixture_hed_cell_is_non_integral(bundles: dict[str, bakeoff.DatasetBundle]) -> None:
    """The fixture must reproduce HED's granularity, not generate integers.

    Had the fixture emitted whole edit operations, the quarter-integer
    slow path would never have surfaced in testing -- which is exactly
    how it was missed the first time.
    """
    values = bundles["linux"].cells["HED"].value
    assert not np.array_equal(values, np.floor(values))
    assert np.array_equal(values * 4, np.floor(values * 4))
    assert bakeoff.factorize(values).max() < values.size


def test_bootstrap_is_unaffected_by_a_non_integral_cell(
    linux_bundle: bakeoff.DatasetBundle,
) -> None:
    """HED must produce a finite rho like every other cell, not nan."""
    result = bakeoff.bootstrap_dataset(
        linux_bundle.index, [linux_bundle.cells["HED"]], replicates=32, seed=42
    )
    entry = result["statistics"]["rho_bound_exact::HED"]
    assert math.isfinite(entry["point"])
    assert entry["ci_low"] <= entry["point"] <= entry["ci_high"]


def test_midranks_handles_empty_input() -> None:
    """An empty replicate must not raise."""
    assert bakeoff.midranks(np.empty(0)).size == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spearman_matches_scipy(seed: int) -> None:
    """Spearman as Pearson-on-midranks is the definition, not a shortcut."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=400).astype(np.float64)
    y = (x + rng.integers(0, 4, size=400)).astype(np.float64)
    np.testing.assert_allclose(bakeoff.spearman(x, y), stats.spearmanr(x, y).statistic, rtol=1e-10)


def test_spearman_is_nan_for_a_constant_variable() -> None:
    """A constant variable has no rank variation and no correlation."""
    x = np.ones(10)
    y = np.arange(10.0)
    assert math.isnan(bakeoff.spearman(x, y))


# ---------------------------------------------------------------------------
# Wilcoxon and Holm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_wilcoxon_matches_scipy_asymptotic(seed: int) -> None:
    """Parity with scipy, whose result carries no rank-biserial effect size."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 10, size=250).astype(np.float64)
    y = rng.integers(0, 10, size=250).astype(np.float64)
    result = bakeoff.wilcoxon_signed_rank(x, y)
    reference = stats.wilcoxon(x, y, zero_method="wilcox", method="asymptotic", correction=False)
    np.testing.assert_allclose(result.statistic, float(reference.statistic))
    np.testing.assert_allclose(result.p_value, float(reference.pvalue), rtol=1e-9)


def test_wilcoxon_rank_biserial_is_plus_one_when_one_side_dominates() -> None:
    """Every difference positive means R- is empty and r_rb saturates."""
    x = np.array([2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.0, 1.0, 1.0])
    result = bakeoff.wilcoxon_signed_rank(x, y)
    assert result.rank_biserial == pytest.approx(1.0)
    assert result.n_zero == 0
    assert bakeoff.wilcoxon_signed_rank(y, x).rank_biserial == pytest.approx(-1.0)


def test_wilcoxon_all_ties_is_inert() -> None:
    """Identical samples give no effect, p = 1 and every pair discarded."""
    x = np.arange(6.0)
    result = bakeoff.wilcoxon_signed_rank(x, x)
    assert result.n_used == 0
    assert result.n_zero == 6
    assert result.p_value == 1.0
    assert result.rank_biserial == 0.0


def test_wilcoxon_rejects_mismatched_shapes() -> None:
    """Unequal lengths are a pairing bug and must raise."""
    with pytest.raises(bakeoff.BakeoffAnalysisError):
        bakeoff.wilcoxon_signed_rank(np.zeros(3), np.zeros(4))


def test_holm_correction_against_a_hand_computed_case() -> None:
    """Holm on [0.01, 0.04, 0.03] with m = 3, worked by hand.

    Sorted: 0.01, 0.03, 0.04. Step-down multipliers 3, 2, 1 give
    0.03, 0.06, 0.04; the running maximum enforces monotonicity, so the
    last becomes 0.06. Mapped back to input order: 0.03, 0.06, 0.06.
    """
    from benchmarks.real_data.eval_correlation.correlation_metrics import holm_bonferroni

    assert holm_bonferroni([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_applied_within_each_dataset_and_end(linux_bundle: bakeoff.DatasetBundle) -> None:
    """Adjusted p never falls below raw p, and the family is the end's."""
    payload = bakeoff.pairwise_significance(
        "lower", "linux", linux_bundle.cells, linux_bundle.index
    )
    evaluated = [c for c in payload["comparisons"] if c["status"] == "evaluated"]
    for comparison in evaluated:
        assert comparison["p_holm"] >= comparison["p_raw"] - 1e-12
        assert 0.0 <= comparison["p_holm"] <= 1.0
    assert "selection procedure, not a hypothesis test" in payload["statistical_status"]
    assert "not the basis of the selection" in payload["p_value_status"]


def test_branch_and_branch_fast_give_a_degenerate_test_kept_in_the_family(
    linux_bundle: bakeoff.DatasetBundle,
) -> None:
    """Provably equivalent cells measure nothing; the row stays in the family.

    Blumenthal et al., *VLDB Journal* §5.2.4: BRANCH and BRANCH-FAST are
    equivalent for constant edge edit costs, which is cost model D6. The
    paired difference vector is identically zero, so no p-value and no
    effect size are printed -- but the comparison is not dropped, because
    dropping a test on account of its outcome is the post-hoc adjustment
    the pre-registration exists to prevent.
    """
    payload = bakeoff.pairwise_significance(
        "lower", "linux", linux_bundle.cells, linux_bundle.index
    )
    assert payload["family_size"] == payload["family_size_nominal"] == 10
    degenerate = [c for c in payload["comparisons"] if c["status"] == "degenerate"]
    assert len(degenerate) == 1
    row = degenerate[0]
    assert {row["method_a"], row["method_b"]} == {"BRANCH", "BRANCH_FAST"}
    assert row["p_raw"] is None
    assert row["p_holm"] is None
    assert row["p_used_for_holm"] == 1.0
    assert row["rank_biserial"] == 0.0
    assert "provably" not in row["reason"] or "equivalent" in row["reason"]
    assert "all paired differences are exactly zero" in row["reason"]
    assert payload["n_degenerate"] == 1


def test_degenerate_wilcoxon_is_flagged_not_crashed() -> None:
    """An all-zero difference vector is where scipy raises; we must not."""
    x = np.arange(500.0)
    result = bakeoff.wilcoxon_signed_rank(x, x.copy())
    assert result.degenerate is True
    assert result.rank_biserial == 0.0
    non_degenerate = bakeoff.wilcoxon_signed_rank(x, x + 1.0)
    assert non_degenerate.degenerate is False


# ---------------------------------------------------------------------------
# Graph-level resampling -- the induced pair set
# ---------------------------------------------------------------------------


def test_pair_flat_index_inverts_triu_indices() -> None:
    """The closed-form inverse must agree with numpy for every pair."""
    for n in (2, 3, 7, 25):
        i, j = np.triu_indices(n, k=1)
        flat = bakeoff.pair_flat_index(n, i, j)
        np.testing.assert_array_equal(flat, np.arange(i.size))


def test_identity_resample_induces_every_pair_exactly_once() -> None:
    """Resampling each graph once reproduces the canonical pair order."""
    n = 12
    flat = bakeoff.induced_pairs(n, np.arange(n, dtype=np.int64))
    np.testing.assert_array_equal(np.sort(flat), np.arange(n * (n - 1) // 2))


def test_induced_pairs_drops_self_pairs_and_keeps_duplicates() -> None:
    """A duplicated graph contributes duplicated pairs, never a self-pair.

    Selection ``[0, 0, 1]`` gives slot pairs (0,1), (0,2), (1,2). The
    first is graph 0 against itself and has no observation, so it is
    dropped; the other two are both graph pair (0, 1) and both count.
    That duplication is the cluster bootstrap's variance mechanism.
    """
    flat = bakeoff.induced_pairs(3, np.array([0, 0, 1], dtype=np.int64))
    assert flat.tolist() == [0, 0]


def test_induced_pairs_of_a_constant_resample_is_empty() -> None:
    """Every slot holding the same graph induces no off-diagonal pair."""
    assert bakeoff.induced_pairs(5, np.zeros(5, dtype=np.int64)).size == 0


def test_replicate_selection_is_reproducible_under_seed_42() -> None:
    """Seed 42 must give byte-identical draws on every run and any order."""
    first = bakeoff.replicate_selection(100, 42, 7)
    second = bakeoff.replicate_selection(100, 42, 7)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, bakeoff.replicate_selection(100, 42, 8))
    assert not np.array_equal(first, bakeoff.replicate_selection(100, 43, 7))


def test_bootstrap_is_reproducible_and_order_independent(
    linux_bundle: bakeoff.DatasetBundle,
) -> None:
    """Two runs at seed 42 agree exactly, serial or parallel.

    Per-replicate ``SeedSequence([seed, replicate])`` seeding is what
    makes the parallel result identical to the serial one; a single
    shared generator would make the answer depend on chunking.
    """
    cells = list(linux_bundle.cells.values())
    serial = bakeoff.bootstrap_dataset(linux_bundle.index, cells, replicates=32, seed=42, jobs=1)
    parallel = bakeoff.bootstrap_dataset(linux_bundle.index, cells, replicates=32, seed=42, jobs=4)
    for key, entry in serial["statistics"].items():
        for field in ("point", "ci_low", "ci_high", "bootstrap_mean"):
            assert entry[field] == pytest.approx(parallel["statistics"][key][field], nan_ok=True)


def test_bootstrap_point_estimate_is_the_full_sample_value(
    linux_bundle: bakeoff.DatasetBundle,
) -> None:
    """The point estimate is observed, never the bootstrap mean."""
    cell = linux_bundle.cells["BRANCH_FAST"]
    result = bakeoff.bootstrap_dataset(linux_bundle.index, [cell], replicates=16, seed=42)
    index = linux_bundle.index
    mask = index.certified & (index.exact > 0)
    expected = float(
        (
            bakeoff.signed_error("lower", cell.value[mask], index.exact[mask]) / index.exact[mask]
        ).mean()
    )
    assert result["statistics"]["mean_rel_err::BRANCH_FAST"]["point"] == pytest.approx(expected)


def test_bootstrap_ci_brackets_the_point_estimate(linux_bundle: bakeoff.DatasetBundle) -> None:
    """A percentile interval on a smooth statistic must contain the estimate."""
    cells = list(linux_bundle.cells.values())
    result = bakeoff.bootstrap_dataset(linux_bundle.index, cells, replicates=200, seed=42)
    entry = result["statistics"]["rho_lev_exact"]
    assert entry["ci_low"] <= entry["point"] <= entry["ci_high"]
    assert entry["ci_low"] < entry["ci_high"]


def test_paired_differences_use_one_resample_per_replicate(
    linux_bundle: bakeoff.DatasetBundle,
) -> None:
    """D7: the difference is taken inside the replicate, not across runs."""
    cells = [linux_bundle.cells["BRANCH"], linux_bundle.cells["STAR"]]
    result = bakeoff.bootstrap_dataset(linux_bundle.index, cells, replicates=64, seed=42)
    stats_block = result["statistics"]
    diff = stats_block["diff_mean_rel_err::lower::BRANCH|STAR"]["point"]
    expected = (
        stats_block["mean_rel_err::BRANCH"]["point"] - stats_block["mean_rel_err::STAR"]["point"]
    )
    assert diff == pytest.approx(expected)


def test_bootstrap_widens_relative_to_a_pair_level_interval(
    linux_bundle: bakeoff.DatasetBundle,
) -> None:
    """The graph-level interval must be wider than a naive pair-level one.

    This is the whole point of D2. LINUX contributes 89 graphs, not
    3,916 independent observations, so resampling pairs would understate
    the uncertainty. If this ever inverts, the resampling unit has
    silently reverted to the pair.
    """
    index = linux_bundle.index
    cells = [linux_bundle.cells["BRANCH"]]
    graph_level = bakeoff.bootstrap_dataset(index, cells, replicates=300, seed=42)
    entry = graph_level["statistics"]["rho_lev_exact"]
    graph_width = entry["ci_high"] - entry["ci_low"]

    mask = index.certified
    exact = index.exact[mask]
    lev = index.lev["exhaustive"][mask]
    rng = np.random.default_rng(42)
    pair_level = [
        bakeoff.spearman(exact[draw], lev[draw])
        for draw in (rng.integers(0, exact.size, size=exact.size) for _ in range(300))
    ]
    lo, hi = np.percentile(pair_level, [2.5, 97.5])
    assert graph_width > float(hi - lo)


# ---------------------------------------------------------------------------
# M4 validity, including the one-sided censored regime
# ---------------------------------------------------------------------------


def _hand_index(
    exact: list[float],
    exact_lb: list[float],
    exact_ub: list[float],
    certified: list[bool],
) -> bakeoff.IndexData:
    """Build a three-graph index by hand, bypassing the fixture."""
    return bakeoff.IndexData(
        dataset="hand",
        n_graphs=3,
        exact=np.array(exact, dtype=np.float64),
        exact_lb=np.array(exact_lb, dtype=np.float64),
        exact_ub=np.array(exact_ub, dtype=np.float64),
        certified=np.array(certified, dtype=bool),
        n_max=np.array([4, 4, 4], dtype=np.int32),
        lev={"exhaustive": np.array([1.0, 2.0, 3.0])},
        node_counts=np.array([4, 4, 4], dtype=np.int32),
        meta={},
    )


def _hand_cell(method: str, end: str, value: list[float]) -> bakeoff.CellData:
    """Build a three-pair cell by hand."""
    arr = np.array(value, dtype=np.float64)
    return bakeoff.CellData(
        dataset="hand",
        method=method,
        end=end,
        value=arr,
        value_fwd=arr,
        value_rev=None if end == "lower" else arr,
        meta={},
    )


def test_lower_bound_validity_two_sided_on_certified_pairs() -> None:
    """LB > exact on a certified pair is the only two-sided refutation."""
    index = _hand_index([4.0, 4.0, 4.0], [4.0] * 3, [4.0] * 3, [True] * 3)
    result = bakeoff.compute_validity(_hand_cell("BRANCH", "lower", [4.0, 3.0, 5.0]), index)
    assert result.violations == 1
    assert result.n_two_sided == 3
    assert result.n_one_sided == 0
    assert result.examples[0]["pair_index"] == 2
    assert result.examples[0]["regime"] == "certified"


def test_zero_lower_bound_with_positive_exact_is_not_a_violation() -> None:
    """C6 against two disjoint triangles: exact 4, every BRANCH bound 0.

    Under cost model D6 node and edge substitution are both free, so any
    degree-preserving assignment costs nothing and two non-isomorphic
    graphs with the same degree sequence get a zero lower bound. The
    bound is correct; flagging it would halt the ticket on a
    non-defect.
    """
    index = _hand_index([4.0, 6.0, 2.0], [4.0, 6.0, 2.0], [4.0, 6.0, 2.0], [True] * 3)
    result = bakeoff.compute_validity(_hand_cell("BRANCH_FAST", "lower", [0.0, 0.0, 0.0]), index)
    assert result.violations == 0


def test_censored_lower_bound_is_refuted_only_above_the_solver_bracket() -> None:
    """One-sided on censored pairs: refuted iff LB > exact_ub (design §3.5)."""
    index = _hand_index(
        [math.inf, math.inf, math.inf],
        [2.0, 2.0, 2.0],
        [6.0, 6.0, 6.0],
        [False, False, False],
    )
    result = bakeoff.compute_validity(_hand_cell("STAR", "lower", [1.0, 6.0, 7.0]), index)
    assert result.violations == 1
    assert result.n_one_sided == 3
    assert result.n_two_sided == 0
    assert result.examples[0]["pair_index"] == 2
    assert result.examples[0]["regime"] == "censored"


def test_censored_upper_bound_is_refuted_only_below_the_solver_bracket() -> None:
    """One-sided on censored pairs: refuted iff UB < exact_lb."""
    index = _hand_index(
        [math.inf, math.inf, math.inf],
        [2.0, 2.0, 2.0],
        [6.0, 6.0, 6.0],
        [False, False, False],
    )
    result = bakeoff.compute_validity(_hand_cell("IPFP_MS", "upper", [2.0, 9.0, 1.0]), index)
    assert result.violations == 1
    assert result.examples[0]["pair_index"] == 2


def test_validity_mixes_both_regimes_over_all_pairs() -> None:
    """M4 spans every pair; M1/M2/M3/M5/M6 span certified pairs only."""
    index = _hand_index(
        [4.0, math.inf, 3.0],
        [4.0, 1.0, 3.0],
        [4.0, 8.0, 3.0],
        [True, False, True],
    )
    result = bakeoff.compute_validity(_hand_cell("BRANCH", "lower", [5.0, 9.0, 0.0]), index)
    assert result.n_checked == 3
    assert result.n_two_sided == 2
    assert result.n_one_sided == 1
    assert result.violations == 2


def test_every_synthetic_cell_is_valid(bundles: dict[str, bakeoff.DatasetBundle]) -> None:
    """The fixture must produce M4 = 0; a fixture violation would mask the check."""
    for bundle in bundles.values():
        for cell in bundle.cells.values():
            assert bakeoff.compute_validity(cell, bundle.index).violations == 0


def test_fixture_contains_zero_but_valid_lower_bounds(
    bundles: dict[str, bakeoff.DatasetBundle],
) -> None:
    """The fixture must exercise the legal zero-LB case, not just exact = 0."""
    bundle = bundles["linux"]
    cell = bundle.cells["BRANCH_FAST"]
    zero_but_positive = (cell.value == 0.0) & (bundle.index.exact > 0) & bundle.index.certified
    assert int(zero_but_positive.sum()) > 0


# ---------------------------------------------------------------------------
# Proven orderings -- gates, not findings
# ---------------------------------------------------------------------------


def test_proven_orderings_hold_on_the_fixture(
    bundles: dict[str, bakeoff.DatasetBundle],
) -> None:
    """BRANCH >= HED, the BIPARTITE-started searches, and BRANCH == BRANCH_FAST."""
    for dataset, bundle in bundles.items():
        report = bakeoff.check_proven_orderings(bundle.cells)
        assert report["violations"] == 0, dataset
        relations = {check["relation"] for check in report["checks"]}
        assert "BRANCH >= HED" in relations
        assert "BIPARTITE >= REFINE_DET" in relations
        assert "BIPARTITE >= BP_BEAM_DET" in relations
        assert "BRANCH == BRANCH_FAST" in relations


def test_branch_ge_hed_violation_is_caught() -> None:
    """A HED above BRANCH is a harness bug; the gate must see it."""
    cells = {
        "BRANCH": _hand_cell("BRANCH", "lower", [2.0, 2.0, 2.0]),
        "HED": _hand_cell("HED", "lower", [1.0, 2.0, 3.0]),
    }
    report = bakeoff.check_proven_orderings(cells)
    assert report["violations"] == 1
    check = next(c for c in report["checks"] if c["relation"] == "BRANCH >= HED")
    assert check["examples"] == [2]
    assert check["max_excess"] == pytest.approx(1.0)


def test_monotone_local_search_above_bipartite_is_caught() -> None:
    """REFINE_DET and BP_BEAM_DET only accept strict improvements."""
    cells = {
        "BIPARTITE": _hand_cell("BIPARTITE", "upper", [5.0, 5.0, 5.0]),
        "REFINE_DET": _hand_cell("REFINE_DET", "upper", [4.0, 5.0, 6.0]),
        "BP_BEAM_DET": _hand_cell("BP_BEAM_DET", "upper", [3.0, 3.0, 3.0]),
    }
    report = bakeoff.check_proven_orderings(cells)
    assert report["violations"] == 1
    check = next(c for c in report["checks"] if c["relation"] == "BIPARTITE >= REFINE_DET")
    assert check["violations"] == 1


def test_branch_equivalence_violation_is_caught() -> None:
    """A single disagreeing pair refutes a theorem and must be reported."""
    cells = {
        "BRANCH": _hand_cell("BRANCH", "lower", [1.0, 2.0, 3.0]),
        "BRANCH_FAST": _hand_cell("BRANCH_FAST", "lower", [1.0, 2.0, 4.0]),
    }
    report = bakeoff.check_proven_orderings(cells)
    check = next(c for c in report["checks"] if c["kind"] == "equivalence")
    assert check["violations"] == 1
    assert check["max_abs_difference"] == pytest.approx(1.0)


def test_proven_ordering_violations_halt_the_run(tmp_path: Path) -> None:
    """A gate failure counts into ``total_violations`` and sets the halt flag."""
    bakeoff.build_synthetic_fixture(tmp_path, bakeoff.FIXTURE_SPECS[:1])
    path = tmp_path / "data" / "cells" / "linux__HED.npz"
    with np.load(path) as payload:
        arrays: dict[str, Any] = dict(payload)
    arrays["value"] = arrays["value"] + 1000.0
    arrays["value_fwd"] = arrays["value"]
    np.savez_compressed(path, **arrays)

    written = bakeoff.run_analysis(tmp_path, datasets=["linux"], replicates=4, make_figures=False)
    validity = json.loads(written["validity"].read_text(encoding="utf-8"))
    assert validity["proven_ordering_violations"] > 0
    assert validity["halts_ticket"] is True


# ---------------------------------------------------------------------------
# The deterministic companion
# ---------------------------------------------------------------------------


def test_deterministic_companion_pairs_each_det_with_its_multi_start_twin(
    bundles: dict[str, bakeoff.DatasetBundle],
) -> None:
    """The ``_DET`` arm is reported beside ``_MS``, never inside the selection."""
    metrics = {
        dataset: {
            method: bakeoff.compute_cell_metrics(cell, bundle.index, Path(".")).payload
            for method, cell in bundle.cells.items()
        }
        for dataset, bundle in list(bundles.items())[:1]
    }
    companion = bakeoff.deterministic_companion(metrics)
    row = companion["per_dataset"]["linux"]["IPFP_DET"]
    assert row["multi_start_cell"] == "IPFP_MS"
    assert row["multi_start_advantage"] == pytest.approx(
        row["mean_relative_error_det"] - row["mean_relative_error_ms"]
    )
    assert "never in the selection" in companion["role"]


def test_det_cells_never_enter_the_selection_candidates(
    bundles: dict[str, bakeoff.DatasetBundle],
) -> None:
    """A companion must not be selectable, however tight it turns out."""
    bundle = bundles["linux"]
    metrics = {
        method: bakeoff.compute_cell_metrics(cell, bundle.index, Path(".")).payload
        for method, cell in bundle.cells.items()
    }
    candidates = bakeoff._candidates("upper", bundle, metrics)
    assert {c.method for c in candidates} == set(bakeoff.UPPER_METHODS)


# ---------------------------------------------------------------------------
# M1-M3 domains: exact == 0 excluded and counted
# ---------------------------------------------------------------------------


def test_exact_zero_pairs_are_excluded_from_m1_and_counted(
    bundles: dict[str, bakeoff.DatasetBundle],
) -> None:
    """Relative error is undefined at exact = 0; the count is reported."""
    bundle = bundles["iam_letter_low"]
    cell = bundle.cells["BRANCH"]
    payload = bakeoff.compute_cell_metrics(cell, bundle.index, Path(".")).payload
    n_zero = int(((bundle.index.exact == 0) & bundle.index.certified).sum())
    assert n_zero > 0
    assert payload["n_exact_zero_certified"] == n_zero
    assert payload["M1_relative_error"]["n_undefined_excluded"] == n_zero
    m1 = payload["M1_relative_error"]["exact_gt_zero"]
    assert m1["n"] == int(bundle.index.certified.sum()) - n_zero
    assert payload["M1_relative_error"]["domains_coincide"] is True
    assert math.isfinite(m1["mean"])


def test_m3_is_reported_twice_and_the_zero_domain_inflates_it(
    bundles: dict[str, bakeoff.DatasetBundle],
) -> None:
    """Every valid LB certifies for free at exact = 0, so M3 must inflate."""
    bundle = bundles["iam_letter_low"]
    payload = bakeoff.compute_cell_metrics(bundle.cells["STAR"], bundle.index, Path(".")).payload
    m3 = payload["M3_certification_rate"]
    assert m3["headline"] == "exact_gt_zero"
    assert m3["all_certified"] > m3["exact_gt_zero"]
    assert m3["well_defined"] is True


def test_m2_domains_differ_but_m1_domains_coincide(
    bundles: dict[str, bakeoff.DatasetBundle],
) -> None:
    """M2 has a defined value at exact = 0; M1 does not, so M1's two agree."""
    bundle = bundles["iam_letter_med"]
    payload = bakeoff.compute_cell_metrics(bundle.cells["IPFP_MS"], bundle.index, Path(".")).payload
    m2 = payload["M2_absolute_error"]
    assert m2["all_certified"]["n"] > m2["exact_gt_zero"]["n"]
    m1 = payload["M1_relative_error"]
    assert m1["all_certified"] == m1["exact_gt_zero"]


def test_metrics_use_certified_pairs_only(bundles: dict[str, bakeoff.DatasetBundle]) -> None:
    """Censored pairs enter M4 and nothing else (design §3.5)."""
    bundle = bundles["aids"]
    payload = bakeoff.compute_cell_metrics(bundle.cells["BRANCH"], bundle.index, Path(".")).payload
    n_certified = int(bundle.index.certified.sum())
    assert payload["M2_absolute_error"]["all_certified"]["n"] == n_certified
    assert payload["M5_rho_bound_exact"]["n"] == n_certified
    assert payload["M4_validity"]["n_checked"] == bundle.index.n_pairs
    assert payload["M4_validity"]["n_one_sided"] > 0


def test_signed_error_orientation() -> None:
    """Slack is non-negative for a valid bound at either end."""
    exact = np.array([5.0, 5.0])
    np.testing.assert_allclose(
        bakeoff.signed_error("lower", np.array([3.0, 5.0]), exact), [2.0, 0.0]
    )
    np.testing.assert_allclose(
        bakeoff.signed_error("upper", np.array([7.0, 5.0]), exact), [2.0, 0.0]
    )


def test_error_stats_on_empty_input_is_all_nan() -> None:
    """An empty domain reports nan, never zero, which would read as perfect."""
    summary = bakeoff.error_stats(np.empty(0))
    assert summary.n == 0
    assert math.isnan(summary.mean)


# ---------------------------------------------------------------------------
# M8 symmetry
# ---------------------------------------------------------------------------


def test_symmetry_is_evaluated_only_for_upper_bounds(
    bundles: dict[str, bakeoff.DatasetBundle],
) -> None:
    """LB cells carry one orientation; UB cells carry both (design §3.6)."""
    bundle = bundles["linux"]
    assert bakeoff.compute_symmetry(bundle.cells["BRANCH"])["evaluated"] == 0
    upper = bakeoff.compute_symmetry(bundle.cells["BIPARTITE"])
    assert upper["evaluated"] == 1
    assert 0.0 <= upper["frac_asymmetric"] <= 1.0
    assert upper["mean_gain_over_fwd"] >= 0.0


def test_upper_bound_value_must_be_the_min_of_both_orientations(
    fixture_root: Path, tmp_path: Path
) -> None:
    """A cell whose value is not min(fwd, rev) violates CONTRACTS §3."""
    index = bakeoff.load_index(fixture_root / "data" / "index" / "linux.npz")
    source = fixture_root / "data" / "cells" / "linux__IPFP_MS.npz"
    with np.load(source) as payload:
        arrays: dict[str, Any] = dict(payload)
    arrays["value"] = arrays["value_fwd"]
    arrays["value_rev"] = arrays["value_fwd"] - 1.0
    broken = tmp_path / "linux__IPFP_MS.npz"
    np.savez_compressed(broken, **arrays)
    with pytest.raises(bakeoff.BakeoffAnalysisError, match="min"):
        bakeoff.load_cell(broken, index)


# ---------------------------------------------------------------------------
# Loading contracts
# ---------------------------------------------------------------------------


def test_index_pair_order_is_asserted(fixture_root: Path, tmp_path: Path) -> None:
    """A shuffled pair order is the wave's silent-corruption risk."""
    source = fixture_root / "data" / "index" / "linux.npz"
    with np.load(source) as payload:
        arrays: dict[str, Any] = dict(payload)
    arrays["pair_i"] = arrays["pair_i"][::-1].copy()
    broken = tmp_path / "linux.npz"
    np.savez_compressed(broken, **arrays)
    with pytest.raises(bakeoff.BakeoffAnalysisError, match="triu_indices"):
        bakeoff.load_index(broken)


def test_certified_bracket_must_be_closed(fixture_root: Path, tmp_path: Path) -> None:
    """``exact_lb == exact_ub == exact`` is what "certified" means."""
    source = fixture_root / "data" / "index" / "linux.npz"
    with np.load(source) as payload:
        arrays: dict[str, Any] = dict(payload)
    arrays["exact_ub"] = arrays["exact_ub"] + 1.0
    broken = tmp_path / "linux.npz"
    np.savez_compressed(broken, **arrays)
    with pytest.raises(bakeoff.BakeoffAnalysisError, match="exact_ub"):
        bakeoff.load_index(broken)


def test_lower_bound_cell_must_not_carry_a_reverse_orientation(
    fixture_root: Path, tmp_path: Path
) -> None:
    """A ``value_rev`` on a lower bound means the harness ran the wrong end."""
    index = bakeoff.load_index(fixture_root / "data" / "index" / "linux.npz")
    source = fixture_root / "data" / "cells" / "linux__BRANCH.npz"
    with np.load(source) as payload:
        arrays: dict[str, Any] = dict(payload)
    arrays["value_rev"] = arrays["value"]
    broken = tmp_path / "linux__BRANCH.npz"
    np.savez_compressed(broken, **arrays)
    with pytest.raises(bakeoff.BakeoffAnalysisError, match="value_rev"):
        bakeoff.load_cell(broken, index)


def test_non_finite_bound_value_raises(fixture_root: Path, tmp_path: Path) -> None:
    """``HED`` returning inf must be caught, not averaged in."""
    index = bakeoff.load_index(fixture_root / "data" / "index" / "linux.npz")
    source = fixture_root / "data" / "cells" / "linux__HED.npz"
    with np.load(source) as payload:
        arrays: dict[str, Any] = dict(payload)
    arrays["value"] = arrays["value"].copy()
    arrays["value"][0] = np.inf
    broken = tmp_path / "linux__HED.npz"
    np.savez_compressed(broken, **arrays)
    with pytest.raises(bakeoff.BakeoffAnalysisError, match="non-finite"):
        bakeoff.load_cell(broken, index)


# ---------------------------------------------------------------------------
# Selection -- spec §5 verbatim plus the companions
# ---------------------------------------------------------------------------


def _candidate(method: str, rel: float, **kwargs: float | int | str | bool) -> bakeoff.Candidate:
    """Build a candidate with permissive defaults."""
    defaults: dict[str, Any] = {
        "mean_absolute_error": rel * 10.0,
        "violations": 0,
        "m3_well_defined": True,
        "cost_gate": "pass",
        "cost_us_per_pair": 50.0,
        "m6_abs_gap": 0.05,
    }
    defaults.update(kwargs)
    return bakeoff.Candidate(method=method, mean_relative_error=rel, **defaults)


def test_selection_minimises_mean_relative_error() -> None:
    """The frozen criterion is M1, not M2 and not rho."""
    result = bakeoff.select_on_dataset(
        [_candidate("BRANCH", 0.30), _candidate("BRANCH_FAST", 0.20), _candidate("STAR", 0.60)]
    )
    assert result["winner"] == "BRANCH_FAST"
    assert result["ranking"] == ["BRANCH_FAST", "BRANCH", "STAR"]
    assert result["margin_relative"] == pytest.approx(0.5)


def test_m4_violation_disqualifies() -> None:
    """A violated proven bound cannot win, however tight it looks."""
    result = bakeoff.select_on_dataset(
        [_candidate("BRANCH", 0.30), _candidate("STAR", 0.05, violations=3)]
    )
    assert result["winner"] == "BRANCH"
    assert result["excluded"] == [{"method": "STAR", "reason": "M4 violation"}]


def test_failed_cost_gate_disqualifies_but_unevaluated_does_not() -> None:
    """design §3.4: a method fails the gate only on probe evidence."""
    failed = bakeoff.select_on_dataset(
        [_candidate("BRANCH", 0.30), _candidate("BRANCH_TIGHT", 0.10, cost_gate="fail")]
    )
    assert failed["winner"] == "BRANCH"
    unevaluated = bakeoff.select_on_dataset(
        [_candidate("BRANCH", 0.30), _candidate("BRANCH_TIGHT", 0.10, cost_gate="unevaluated")]
    )
    assert unevaluated["winner"] == "BRANCH_TIGHT"


def test_tie_within_two_percent_breaks_on_cost_then_on_m6() -> None:
    """Ties break on M7 first, then M6; never on which method flatters rho."""
    by_cost = bakeoff.select_on_dataset(
        [
            _candidate("BRANCH", 0.200, cost_us_per_pair=90.0),
            _candidate("BRANCH_FAST", 0.201, cost_us_per_pair=20.0),
        ]
    )
    assert by_cost["winner"] == "BRANCH_FAST"
    assert by_cost["tie_group"] == ["BRANCH", "BRANCH_FAST"]
    assert by_cost["tie_break"] is not None

    by_m6 = bakeoff.select_on_dataset(
        [
            _candidate("BRANCH", 0.200, cost_us_per_pair=20.0, m6_abs_gap=0.30),
            _candidate("BRANCH_FAST", 0.201, cost_us_per_pair=20.0, m6_abs_gap=0.01),
        ]
    )
    assert by_m6["winner"] == "BRANCH_FAST"


def test_a_gap_beyond_two_percent_is_not_a_tie() -> None:
    """A 5 % gap is a win, so the cheaper method must not steal it."""
    result = bakeoff.select_on_dataset(
        [
            _candidate("BRANCH", 0.200, cost_us_per_pair=90.0),
            _candidate("BRANCH_FAST", 0.210, cost_us_per_pair=5.0),
        ]
    )
    assert result["winner"] == "BRANCH"
    assert result["tie_group"] == ["BRANCH"]


def test_global_primary_needs_four_of_five_datasets() -> None:
    """The frozen threshold is >= 4 of 5, applied verbatim."""
    winning = {
        d: [_candidate("BRANCH", 0.1 if d != "linux" else 0.4), _candidate("STAR", 0.2)]
        for d in bakeoff.DATASETS
    }
    result = bakeoff.select_end("lower", winning)
    assert result["frozen_rule"]["global_primary"] == "BRANCH"
    assert result["frozen_rule"]["win_counts"] == {"BRANCH": 4, "STAR": 1}

    split = {
        d: [_candidate("BRANCH", 0.1 if k < 3 else 0.4), _candidate("STAR", 0.2)]
        for k, d in enumerate(bakeoff.DATASETS)
    }
    fallback = bakeoff.select_end("lower", split)
    assert fallback["frozen_rule"]["global_primary"] is None
    assert "per dataset" in fallback["frozen_rule"]["branch"]


def test_corpus_collapsed_companion_treats_letter_as_one_unit() -> None:
    """design §3.2: Letter votes as the majority of its three levels."""
    winners = {
        "linux": "BRANCH",
        "aids": "BRANCH",
        "iam_letter_low": "BRANCH",
        "iam_letter_med": "BRANCH",
        "iam_letter_high": "STAR",
    }
    companion = bakeoff.collapse_to_corpora(winners)
    assert companion["corpus_winner"] == {"LINUX": "BRANCH", "AIDS": "BRANCH", "Letter": "BRANCH"}
    assert companion["global_primary"] == "BRANCH"


def test_corpus_companion_can_contradict_the_frozen_rule() -> None:
    """Three Letter votes carry the frozen rule but not the companion.

    A Letter-favouring method starts with three votes of five, so it can
    reach the frozen ">= 4 of 5" while winning only one of the three
    corpora. Detecting that disagreement is the companion's purpose.
    """
    winners = {
        "linux": "BRANCH",
        "aids": "STAR",
        "iam_letter_low": "STAR",
        "iam_letter_med": "STAR",
        "iam_letter_high": "STAR",
    }
    companion = bakeoff.collapse_to_corpora(winners)
    assert companion["corpus_winner"]["Letter"] == "STAR"
    assert companion["corpus_winner"]["LINUX"] == "BRANCH"
    assert companion["global_primary"] is None


def test_letter_has_no_vote_when_its_three_levels_disagree() -> None:
    """Three distinct level winners is no majority, so Letter abstains."""
    winners = {
        "iam_letter_low": "BRANCH",
        "iam_letter_med": "STAR",
        "iam_letter_high": "HED",
    }
    assert bakeoff.collapse_to_corpora(winners)["corpus_winner"]["Letter"] is None


def test_absolute_error_companion_is_reported_beside_the_frozen_rule() -> None:
    """design §3.1: disagreement is a finding, never an override."""
    candidates = {
        d: [
            _candidate("BRANCH", 0.30, mean_absolute_error=1.0),
            _candidate("STAR", 0.20, mean_absolute_error=9.0),
        ]
        for d in bakeoff.DATASETS
    }
    result = bakeoff.select_end("lower", candidates)
    assert result["frozen_rule"]["global_primary"] == "STAR"
    assert result["companion_absolute_error"]["per_dataset"]["linux"]["winner"] == "BRANCH"
    assert result["companion_absolute_error"]["all_agree"] is False
    assert result["companions_agree_with_frozen"] is False


# ---------------------------------------------------------------------------
# Friedman and the critical difference
# ---------------------------------------------------------------------------


def test_nemenyi_critical_difference_matches_demsar_table() -> None:
    """CD = q_0.05 sqrt(k(k+1)/(6N)); k = 4, N = 5 is the worked example."""
    assert bakeoff.nemenyi_critical_difference(4, 5) == pytest.approx(2.569 * math.sqrt(20 / 30.0))
    assert math.isnan(bakeoff.nemenyi_critical_difference(99, 5))


def test_rank_cliques_join_methods_within_the_critical_difference() -> None:
    """Joined methods are not separated; that is the diagram's only claim."""
    assert bakeoff.rank_cliques([1.0, 1.5, 3.5], 1.0) == ((0, 1),)
    assert bakeoff.rank_cliques([1.0, 2.0, 3.0], 10.0) == ((0, 1, 2),)
    assert bakeoff.rank_cliques([1.0, 5.0], 1.0) == ()


def test_friedman_reports_both_caveats_and_says_when_it_separates_nothing() -> None:
    """N = 5 and non-independence go in the output, plus a plain verdict."""
    scores = {d: {m: 0.5 for m in bakeoff.methods_for_end("upper")} for d in bakeoff.DATASETS}
    result = bakeoff.friedman_over_datasets("upper", scores)
    assert result["separates_any_pair"] is False
    assert "separates nothing" in result["separation_note"]
    assert "N = 5" in result["caveat_n5"]
    assert "3 + 1 + 1" in result["caveat_non_independence"]
    assert "selection procedure" in result["statistical_status"]


def test_friedman_is_not_evaluable_with_one_dataset() -> None:
    """An omnibus over one dataset is reported as such, not faked."""
    result = bakeoff.friedman_over_datasets(
        "upper", {"linux": {m: 0.5 for m in bakeoff.UPPER_METHODS}}
    )
    assert result["status"] == "not evaluable"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_run_analysis_writes_all_five_schema_valid_json(tmp_path: Path) -> None:
    """CONTRACTS §7: five files, every one carrying its provenance header."""
    bakeoff.build_synthetic_fixture(tmp_path, bakeoff.FIXTURE_SPECS[:2])
    written = bakeoff.run_analysis(
        tmp_path,
        datasets=["linux", "aids"],
        replicates=16,
        jobs=1,
        make_figures=False,
    )
    assert set(written) == {"metrics", "validity", "bootstrap", "significance", "selection"}
    for name, path in written.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["provenance"]["schema_version"] == bakeoff.SCHEMA_VERSION
        assert payload["provenance"]["wave"] == bakeoff.WAVE
        assert (
            "selection procedure, not a hypothesis test"
            in (payload["provenance"]["statistical_status"])
        )
        assert name in {"metrics", "validity", "bootstrap", "significance", "selection"}

    validity = json.loads(written["validity"].read_text(encoding="utf-8"))
    assert validity["total_violations"] == 0
    assert validity["halts_ticket"] is False
    assert len(validity["cells"]) == 2 * 12

    significance = json.loads(written["significance"].read_text(encoding="utf-8"))
    assert "not the basis of the selection" in significance["p_value_status"]
    assert significance["per_dataset"]["linux"]["lower"]["family_size"] == 10
    assert significance["per_dataset"]["linux"]["upper"]["family_size"] == 6

    bootstrap = json.loads(written["bootstrap"].read_text(encoding="utf-8"))
    assert "graph-level cluster bootstrap" in bootstrap["protocol"].lower()
    assert bootstrap["datasets"]["linux"]["seed"] == bakeoff.BOOTSTRAP_SEED


def test_run_analysis_refuses_an_empty_root(tmp_path: Path) -> None:
    """No data is an error, not an empty report."""
    with pytest.raises(bakeoff.BakeoffAnalysisError):
        bakeoff.run_analysis(tmp_path, datasets=["linux"], replicates=4, make_figures=False)


def test_write_json_serialises_non_finite_as_null(tmp_path: Path) -> None:
    """NaN is not valid JSON and several readers accept it silently."""
    path = bakeoff.write_json(tmp_path / "x.json", {"a": float("nan"), "b": np.float64(2.0)})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": None, "b": 2.0}
