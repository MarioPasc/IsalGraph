"""Tests for the pre-registered T-03 stage-1 pair sampler (CONTRACTS section 8).

The design is frozen before any production run, so these tests check conformance
to a specification rather than the quality of a heuristic. Four properties carry
the scientific claim and each has a test:

* **Determinism.** Two runs at seed 42 give byte-identical pair lists, so the
  reported stage-1 rho is reproducible from the design note alone.
* **Coverage.** Every graph in the population appears in at least one sampled
  pair. Stage 1's argument is that effective sample size is governed by the number
  of graphs, and a stage 1 built on a subset would concede the opposite.
* **A complete core block.** All C(K, 2) core pairs are present, which is what
  makes the D2 graph-level cluster bootstrap exact on the induced submatrix.
* **The stratum floor.** Every non-empty population stratum reaches
  ``min(f, |stratum|)``, and a stratum smaller than ``f`` is filled completely
  rather than over-drawn or skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.eval_setup.ged_pair_index import n_pairs, pairs_from_indices
from benchmarks.eval_setup.ged_sampling import (
    DEFAULT_F,
    DEFAULT_K,
    DEFAULT_Q,
    DEFAULT_SEED,
    GedSamplingError,
    _cell_to_pair,
    build_pair_strata,
    main,
    sampling_report,
    stage1_sample,
)

AIDS_N = 769


def _synthetic_population(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build node and edge counts resembling the filtered AIDS cohort.

    Node counts span the full 2..12 range the frozen size bins cover, and edge
    counts span connected (``n - 1``) to complete (``n(n-1)/2``), so all three size
    bins and all five density quintiles are populated.

    Args:
        n: Number of graphs.
        seed: RNG seed.

    Returns:
        ``(n_nodes, n_edges)``, both ``int64``.
    """
    rng = np.random.default_rng(seed)
    n_nodes = rng.integers(2, 13, size=n, dtype=np.int64)
    lo = n_nodes - 1
    hi = n_nodes * (n_nodes - 1) // 2
    n_edges = lo + (rng.random(n) * (hi - lo + 1)).astype(np.int64)
    return n_nodes, np.minimum(n_edges, hi)


def _write_contract_a(path: Path, n_nodes: np.ndarray, n_edges: np.ndarray) -> None:
    """Write a minimal CONTRACT A file carrying only the keys the sampler reads.

    Args:
        path: Destination ``.npz``.
        n_nodes: Node counts.
        n_edges: Edge counts.
    """
    n = int(n_nodes.size)
    np.savez_compressed(
        path,
        graph_ids=np.array([f"g{t:05d}" for t in range(n)], dtype=str),
        n_nodes=n_nodes.astype(np.int32),
        n_edges=n_edges.astype(np.int32),
        metadata=np.array(json.dumps({"dataset": "aids_synthetic"})),
    )


# --------------------------------------------------------------------------- #
# Strata
# --------------------------------------------------------------------------- #


def test_size_bins_cut_at_the_frozen_boundaries() -> None:
    """{2-5, 6-9, 10-12} are the three bins; the cut points are 6 and 10."""
    n_nodes = np.arange(2, 13, dtype=np.int64)
    n_edges = n_nodes - 1
    strata = build_pair_strata(n_nodes, n_edges)
    assert strata.size_bin.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]


def test_there_are_ninety_strata_six_size_cells_by_fifteen_density_cells() -> None:
    """The cross product is 6 x 15 = 90, empty cells included."""
    n_nodes, n_edges = _synthetic_population(200, seed=1)
    strata = build_pair_strata(n_nodes, n_edges)
    assert strata.n_strata == 90
    assert int(strata.population_counts.sum()) == n_pairs(200)
    assert strata.quantile_edges.size == 4
    assert set(np.unique(strata.density_bin).tolist()).issubset({0, 1, 2, 3, 4})


def test_the_top_quintile_stays_reachable_when_q80_equals_the_maximum_density() -> None:
    """A density equal to an edge goes into the UPPER bin, so quintile 4 never collapses.

    This is not a hypothetical corpus. After ``min_nodes=2`` the AIDS cohort contains
    ``n = 2`` graphs, and a connected two-node graph has exactly one edge, so its
    density is exactly ``1.0``. Whenever enough of those are present the 80th
    percentile equals the maximum, and under ``np.searchsorted``'s default
    ``side="left"`` every one of them would fall to bin 3 -- leaving the top quintile
    empty on the real data and silently removing every stratum that involves it from
    the top-up.

    Fixture: 40 graphs at density exactly 1.0 and 60 strictly below it, so
    ``q80 == max == 1.0`` by construction.
    """
    n_nodes = np.concatenate([np.full(40, 2), np.full(60, 6)]).astype(np.int64)
    # n=2 -> m=1 -> density 1.0; n=6 -> m in 5..10 -> density m/15 <= 0.667.
    n_edges = np.concatenate([np.full(40, 1), 5 + np.arange(60) % 6]).astype(np.int64)
    strata = build_pair_strata(n_nodes, n_edges)

    assert float(strata.quantile_edges[3]) == pytest.approx(1.0)
    assert float(strata.density.max()) == pytest.approx(1.0)
    assert float(strata.quantile_edges[3]) == pytest.approx(float(strata.density.max()))

    top = strata.density_bin == 4
    assert int(np.count_nonzero(top)) == 40, "every density-1.0 graph belongs to quintile 4"
    assert bool(np.all(strata.density_bin[:40] == 4)), "ties must fall consistently"
    assert bool(np.all(strata.density_bin[40:] < 4))

    # The strata that involve quintile 4 are non-empty in the population, so the
    # top-up will actually reach them.
    involving_top = [
        s
        for s in range(strata.n_strata)
        if 4 in _cell_to_pair(s % 15, 5) and strata.population_counts[s] > 0
    ]
    assert involving_top, "quintile 4 must appear in at least one non-empty stratum"
    assert int(strata.population_counts[involving_top].sum()) >= n_pairs(40)


def test_density_binning_is_stable_under_repetition() -> None:
    """The same population always yields the same bins; no order or tie dependence."""
    n_nodes, n_edges = _synthetic_population(300, seed=99)
    first = build_pair_strata(n_nodes, n_edges)
    second = build_pair_strata(n_nodes, n_edges)
    assert np.array_equal(first.density_bin, second.density_bin)
    assert np.array_equal(first.stratum, second.stratum)
    assert np.array_equal(first.quantile_edges, second.quantile_edges)


def test_strata_reject_node_counts_outside_the_covered_range() -> None:
    """A graph with 13 nodes has no size bin; silently lumping it in would be wrong."""
    with pytest.raises(GedSamplingError, match=r"\[2, 12\]"):
        build_pair_strata(np.array([2, 13]), np.array([1, 12]))


def test_strata_reject_a_population_smaller_than_two() -> None:
    """One graph has no pairs and cannot be stratified."""
    with pytest.raises(GedSamplingError):
        build_pair_strata(np.array([5]), np.array([4]))


# --------------------------------------------------------------------------- #
# The frozen design, at AIDS scale
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def aids_like() -> tuple[np.ndarray, np.ndarray]:
    """A 769-graph synthetic population standing in for the filtered AIDS cohort."""
    return _synthetic_population(AIDS_N, seed=20260812)


def test_stage1_covers_every_graph_in_the_population(
    aids_like: tuple[np.ndarray, np.ndarray],
) -> None:
    """All 769 graphs appear, which CONTRACTS section 8 requires explicitly."""
    sample = stage1_sample(*aids_like)
    assert sample.graphs_covered == AIDS_N
    i, j = pairs_from_indices(sample.pair_index, AIDS_N)
    assert np.unique(np.concatenate([i, j])).size == AIDS_N


def test_stage1_is_deterministic_under_a_fixed_seed(
    aids_like: tuple[np.ndarray, np.ndarray],
) -> None:
    """Same seed, same pair list -- including the halo and top-up draws."""
    a = stage1_sample(*aids_like, seed=DEFAULT_SEED)
    b = stage1_sample(*aids_like, seed=DEFAULT_SEED)
    assert np.array_equal(a.pair_index, b.pair_index)
    assert np.array_equal(a.core_graphs, b.core_graphs)
    assert np.array_equal(a.topup_pairs, b.topup_pairs)


def test_a_different_seed_gives_a_different_sample(
    aids_like: tuple[np.ndarray, np.ndarray],
) -> None:
    """The seed is doing real work; the design fixes it at 42 for that reason."""
    a = stage1_sample(*aids_like, seed=DEFAULT_SEED)
    b = stage1_sample(*aids_like, seed=DEFAULT_SEED + 1)
    assert not np.array_equal(a.core_graphs, b.core_graphs)


def test_the_pair_list_is_ascending_and_free_of_duplicates(
    aids_like: tuple[np.ndarray, np.ndarray],
) -> None:
    """A repeated pair would be computed twice and would skew every stratum count."""
    sample = stage1_sample(*aids_like)
    assert np.array_equal(sample.pair_index, np.sort(sample.pair_index))
    assert np.unique(sample.pair_index).size == sample.pair_index.size
    assert int(sample.pair_index.min()) >= 0
    assert int(sample.pair_index.max()) < n_pairs(AIDS_N)


def test_the_core_block_is_complete(aids_like: tuple[np.ndarray, np.ndarray]) -> None:
    """All C(180, 2) = 16,110 core pairs are present, with no holes.

    The D2 graph-level cluster bootstrap recomputes rho over the induced submatrix
    of a resampled set of core graphs; a hole would make that submatrix ragged.
    """
    sample = stage1_sample(*aids_like)
    assert sample.core_graphs.size == DEFAULT_K
    assert np.unique(sample.core_graphs).size == DEFAULT_K
    assert sample.core_pairs.size == DEFAULT_K * (DEFAULT_K - 1) // 2 == 16_110
    assert np.isin(sample.core_pairs, sample.pair_index).all()


def test_every_non_empty_stratum_reaches_its_floor(
    aids_like: tuple[np.ndarray, np.ndarray],
) -> None:
    """min(f, |stratum|) is met everywhere; empty strata are left alone."""
    sample = stage1_sample(*aids_like)
    pop = sample.strata.population_counts
    got = sample.sampled_counts
    for s in range(sample.strata.n_strata):
        if pop[s] == 0:
            assert got[s] == 0, "an empty population stratum cannot yield sampled pairs"
        else:
            assert got[s] >= min(DEFAULT_F, int(pop[s]))
            assert got[s] <= pop[s], "cannot sample more pairs than the stratum holds"


def test_the_halo_gives_every_non_core_graph_its_partners(
    aids_like: tuple[np.ndarray, np.ndarray],
) -> None:
    """Each of the 589 non-core graphs is in at least q pairs before the top-up."""
    sample = stage1_sample(*aids_like)
    in_core = np.zeros(AIDS_N, dtype=bool)
    in_core[sample.core_graphs] = True
    i, j = pairs_from_indices(sample.pair_index, AIDS_N)
    degree = np.bincount(np.concatenate([i, j]), minlength=AIDS_N)
    assert int(np.count_nonzero(~in_core)) == AIDS_N - DEFAULT_K == 589
    assert degree[~in_core].min() >= DEFAULT_Q


def test_total_pair_count_lands_in_the_designed_envelope(
    aids_like: tuple[np.ndarray, np.ndarray],
) -> None:
    """The design predicts roughly 22,500-24,500 pairs at K=180, q=10, f=30."""
    sample = stage1_sample(*aids_like)
    assert 20_000 <= sample.n_pairs_sampled <= 27_000
    assert sample.n_pairs_sampled < n_pairs(AIDS_N)


# --------------------------------------------------------------------------- #
# Small populations and edge cases
# --------------------------------------------------------------------------- #


def test_a_stratum_smaller_than_f_is_filled_completely_not_over_drawn() -> None:
    """When |stratum| < f the whole stratum is taken; nothing is invented."""
    n_nodes, n_edges = _synthetic_population(40, seed=7)
    sample = stage1_sample(n_nodes, n_edges, k_core=8, q_halo=2, f_topup=1000)
    pop = sample.strata.population_counts
    got = sample.sampled_counts
    small = [s for s in range(sample.strata.n_strata) if 0 < pop[s] < 1000]
    assert small, "the fixture must contain at least one under-sized stratum"
    for s in small:
        assert got[s] == pop[s], "an under-sized stratum is taken in full"
    assert sample.n_pairs_sampled == n_pairs(40)


def test_a_core_covering_the_whole_population_needs_no_halo() -> None:
    """K == N degenerates to the complete matrix and still satisfies coverage."""
    n_nodes, n_edges = _synthetic_population(12, seed=3)
    sample = stage1_sample(n_nodes, n_edges, k_core=12, q_halo=1, f_topup=0)
    assert sample.n_pairs_sampled == n_pairs(12)
    assert sample.graphs_covered == 12
    assert sample.halo_pairs.size == 0


def test_q_zero_fails_the_coverage_requirement_loudly() -> None:
    """Without a halo the non-core graphs are unreachable; that must raise, not warn."""
    n_nodes, n_edges = _synthetic_population(60, seed=5)
    with pytest.raises(GedSamplingError, match="covers"):
        stage1_sample(n_nodes, n_edges, k_core=5, q_halo=0, f_topup=0)


@pytest.mark.parametrize(
    ("k_core", "q_halo", "f_topup"), [(1, 10, 30), (100, 10, 30), (10, 60, 30), (10, 5, -1)]
)
def test_inadmissible_parameters_raise(k_core: int, q_halo: int, f_topup: int) -> None:
    """K, q and f are checked against the population before any RNG is consumed."""
    n_nodes, n_edges = _synthetic_population(50, seed=11)
    with pytest.raises(GedSamplingError):
        stage1_sample(n_nodes, n_edges, k_core=k_core, q_halo=q_halo, f_topup=f_topup)


# --------------------------------------------------------------------------- #
# Report and CLI
# --------------------------------------------------------------------------- #


def test_the_report_records_everything_the_contract_names(
    aids_like: tuple[np.ndarray, np.ndarray],
) -> None:
    """K, q, f, the seed, per-stratum counts, coverage and the total are all present."""
    sample = stage1_sample(*aids_like)
    rep = sampling_report(sample, dataset="aids")
    assert rep["K"] == DEFAULT_K
    assert rep["q"] == DEFAULT_Q
    assert rep["f"] == DEFAULT_F
    assert rep["seed"] == DEFAULT_SEED
    assert rep["graphs_covered"] == AIDS_N
    assert rep["graphs_covered_is_complete"] is True
    assert rep["n_sampled_pairs"] == sample.n_pairs_sampled
    assert rep["all_non_empty_strata_meet_floor"] is True
    assert len(rep["strata"]) == 90
    assert sum(int(r["population_pairs"]) for r in rep["strata"]) == n_pairs(AIDS_N)
    assert sum(int(r["sampled_pairs"]) for r in rep["strata"]) == sample.n_pairs_sampled
    json.dumps(rep)  # the report must be serialisable as written


def test_cli_writes_a_pair_list_and_a_report(tmp_path: Path) -> None:
    """End to end through the CLI, with the distinct-graph assertion enabled."""
    n_nodes, n_edges = _synthetic_population(AIDS_N, seed=20260812)
    src = tmp_path / "aids.npz"
    _write_contract_a(src, n_nodes, n_edges)
    pairs_out = tmp_path / "pair_list.npz"
    report_out = tmp_path / "sampling_report.json"
    code = main(
        [
            "--input",
            str(src),
            "--out-pairs",
            str(pairs_out),
            "--out-report",
            str(report_out),
            "--expect-graphs",
            str(AIDS_N),
        ]
    )
    assert code == 0
    with np.load(pairs_out) as data:
        listed = data["pair_index"]
    assert listed.dtype == np.int64
    assert np.array_equal(listed, np.sort(np.unique(listed)))
    rep = json.loads(report_out.read_text(encoding="utf-8"))
    assert rep["graphs_covered"] == AIDS_N
    assert rep["n_sampled_pairs"] == int(listed.size)


def test_cli_fails_when_the_population_is_not_the_expected_size(tmp_path: Path) -> None:
    """--expect-graphs is a cohort guard; a mismatch must stop the run."""
    n_nodes, n_edges = _synthetic_population(100, seed=2)
    src = tmp_path / "aids.npz"
    _write_contract_a(src, n_nodes, n_edges)
    code = main(
        [
            "--input",
            str(src),
            "--out-pairs",
            str(tmp_path / "p.npz"),
            "--out-report",
            str(tmp_path / "r.json"),
            "--expect-graphs",
            str(AIDS_N),
            "-K",
            "20",
        ]
    )
    assert code == 1
