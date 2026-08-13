"""Unit tests for approx_ged_sampling -- the frozen seed-42 subsample, CONTRACTS section 5.

The subsample is drawn **before** the run so the result cannot shape it. Two properties therefore
carry the whole design and both are tested against fabricated pools as well as the real cohort:

* the bin rule, exactly at the edges, because a right-open boundary read the wrong way moves
  thousands of pairs between strata and nothing raises;
* reproducibility from the seed alone, over array **content** rather than file bytes.

A third group covers the failure that the real export actually exposed: a truncated ``dataset_key``
column that silently merged three datasets under one label.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pytest

from benchmarks.eval_setup.approx_ged_sampling import (
    _KEY_DTYPE,
    BIN_EDGES,
    BIN_TABLE_NAME,
    DEFAULT_OUT_DIR,
    DOMINANCE_WARN_SHARE,
    MAX_PER_BIN,
    MAX_TOTAL_PAIRS,
    N_BINS,
    PAIR_LIST_SUBDIR,
    PROBE_NAME,
    PROBE_PAIR_LIST_SUBDIR,
    PROBE_TOTAL,
    SEED,
    SUBSAMPLE_NAME,
    DatasetPairs,
    SamplingError,
    Subsample,
    _check_dataset_keys,
    _overlap,
    allocate_evenly,
    bin_of,
    build_bin_table,
    build_metadata,
    build_pairs,
    build_pools,
    build_probe_metadata,
    content_digest,
    draw,
    draw_probe,
    pool_pair_index,
    read_node_counts,
    run,
    write_bin_table,
    write_probe,
    write_subsample,
)
from benchmarks.eval_setup.export_graphs_suite2 import (
    DEFAULT_EXPORT_DIR,
    SUITE2_DATASETS,
    TOTAL_EXPECTED_PAIRS,
)
from benchmarks.eval_setup.ged_pair_index import indices_of_pairs, pair_from_index

_EXPORT_PRESENT = Path(DEFAULT_EXPORT_DIR).is_dir()
requires_export = pytest.mark.skipif(
    not _EXPORT_PRESENT, reason=f"Suite-2 export absent: {DEFAULT_EXPORT_DIR}"
)


# --------------------------------------------------------------------------- #
# The bin rule
# --------------------------------------------------------------------------- #


def test_bin_edges_are_the_frozen_design() -> None:
    assert BIN_EDGES == (2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 99)
    assert N_BINS == 14
    assert SEED == 42
    assert MAX_PER_BIN == 2000
    assert MAX_TOTAL_PAIRS == 28000


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (2, 0),  # first edge, inclusive
        (3, 0),
        (4, 1),  # right-open: 4 opens bin 1, it does not close bin 0
        (5, 1),
        (6, 2),
        (11, 4),
        (12, 5),  # an edge, so the low end of its own bin
        (14, 5),
        (15, 6),
        (29, 8),
        (30, 9),  # an edge
        (39, 9),
        (40, 10),
        (79, 12),
        (80, 13),  # an edge; last bin is [80, 99)
        (98, 13),  # the largest graph in Suite 2
    ],
)
def test_bin_of_at_and_around_every_boundary(n: int, expected: int) -> None:
    """Right-open bins. ``n`` equal to an edge belongs to the bin that edge *opens*."""
    assert int(bin_of(np.asarray([n]))[0]) == expected


def test_bin_of_flags_values_outside_the_design() -> None:
    """Below the first edge yields -1, at or above the last yields N_BINS. Neither is a bin."""
    assert int(bin_of(np.asarray([1]))[0]) == -1
    assert int(bin_of(np.asarray([99]))[0]) == N_BINS
    assert int(bin_of(np.asarray([417]))[0]) == N_BINS


def test_build_pairs_rejects_a_graph_outside_the_frozen_bins() -> None:
    """A 99-node graph would fall outside the design; it must raise, not be silently omitted."""
    with pytest.raises(SamplingError, match="outside the 14 frozen bins"):
        build_pairs("linux", np.asarray([5, 99], dtype=np.int32))


def test_build_pairs_rejects_a_pair_of_single_node_graphs() -> None:
    """The guard is on ``max(n1, n2)``, so it fires only when *both* graphs are below the design."""
    with pytest.raises(SamplingError, match="outside the 14 frozen bins"):
        build_pairs("linux", np.asarray([1, 1], dtype=np.int32))


def test_a_lone_small_graph_is_invisible_at_pair_level() -> None:
    """``max(1, 5) = 5`` is a valid stratum, so ``build_pairs`` cannot detect a lone 1-node graph.

    Recorded rather than treated as a gap: ``filter_graphs(min_nodes=2)`` removes those graphs
    before the export, so the sampler never sees one. This test pins the boundary of what the guard
    covers, so nobody later mistakes it for a filter.
    """
    pool = build_pairs("linux", np.asarray([1, 5], dtype=np.int32))
    assert pool.n_max.tolist() == [5]
    assert int(pool.bin_index[0]) == 1


def test_build_pairs_enumerates_the_upper_triangle_in_order() -> None:
    pool = build_pairs("linux", np.asarray([2, 3, 4, 5], dtype=np.int32))
    assert pool.pair_i.tolist() == [0, 0, 0, 1, 1, 2]
    assert pool.pair_j.tolist() == [1, 2, 3, 2, 3, 3]
    assert bool((pool.pair_i < pool.pair_j).all())
    assert pool.n_max.tolist() == [3, 4, 5, 4, 5, 5]
    assert pool.pair_i.dtype == np.int32
    assert pool.bin_index.dtype == np.int8


def test_build_pairs_uses_max_not_min_or_mean() -> None:
    """The stratum is ``max(n1, n2)``; a pair of a 2-node and a 90-node graph is a large pair."""
    pool = build_pairs("linux", np.asarray([2, 90], dtype=np.int32))
    assert pool.n_max.tolist() == [90]
    assert int(pool.bin_index[0]) == 13


# --------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------- #


def _pool(key: str, node_counts: list[int]) -> DatasetPairs:
    return build_pairs(key, np.asarray(node_counts, dtype=np.int32))


def test_draw_takes_the_whole_bin_when_its_population_is_below_the_cap() -> None:
    """``min(2000, population)``: a small bin is taken entire, not padded and not resampled."""
    pool = _pool("linux", [2, 3, 4, 5, 6, 7])
    sample = draw([pool], seed=SEED)
    assert len(sample) == pool.pair_i.shape[0] == comb(6, 2)
    for b in range(N_BINS):
        assert sample.bin_drawn[b] == min(MAX_PER_BIN, sample.bin_population[b])


def test_draw_caps_a_large_bin_at_max_per_bin() -> None:
    pool = _pool("coil_del", [30] * 100)
    sample = draw([pool], seed=SEED, max_per_bin=7)
    assert sample.bin_population[9] == comb(100, 2) == 4950
    assert sample.bin_drawn[9] == 7
    assert len(sample) == 7


def test_draw_never_exceeds_the_design_ceiling() -> None:
    pool = _pool("coil_del", list(range(2, 99)))
    sample = draw([pool], seed=SEED)
    assert len(sample) <= MAX_TOTAL_PAIRS


def test_draw_is_reproducible_from_the_seed_alone() -> None:
    pools = [_pool("linux", list(range(2, 40))), _pool("grec", list(range(2, 30)))]
    first = draw(pools, seed=SEED, max_per_bin=5)
    second = draw(pools, seed=SEED, max_per_bin=5)
    assert content_digest(first) == content_digest(second)
    assert first.pair_index.tolist() == second.pair_index.tolist()


def test_a_different_seed_gives_a_different_draw() -> None:
    """Guards against a draw that only looks reproducible because it ignores the RNG."""
    pools = [_pool("coil_del", list(range(2, 60)))]
    assert content_digest(draw(pools, seed=42, max_per_bin=5)) != content_digest(
        draw(pools, seed=43, max_per_bin=5)
    )


def test_draw_emits_pairs_with_i_less_than_j_and_valid_indices() -> None:
    pool = _pool("grec", list(range(2, 40)))
    sample = draw([pool], seed=SEED, max_per_bin=11)
    assert bool((sample.pair_i < sample.pair_j).all())
    assert bool((sample.pair_i >= 0).all())
    assert bool((sample.pair_j < pool.n_graphs).all())


def test_pair_index_inverts_to_the_emitted_i_and_j() -> None:
    """``pair_index`` is what the runner consumes; it must name the same pair as ``(i, j)``."""
    pool = _pool("grec", list(range(2, 25)))
    sample = draw([pool], seed=SEED, max_per_bin=9)
    for k, i, j in zip(sample.pair_index, sample.pair_i, sample.pair_j, strict=True):
        assert pair_from_index(int(k), pool.n_graphs) == (int(i), int(j))


def test_draw_reports_the_bin_population_per_dataset() -> None:
    pools = [_pool("linux", [3, 3, 3]), _pool("grec", [30, 30])]
    sample = draw(pools, seed=SEED)
    assert sample.bin_population_by_dataset["linux"][0] == 3
    assert sample.bin_population_by_dataset["grec"][9] == 1
    assert sample.bin_population[0] == 3
    assert sample.bin_population[9] == 1


def test_draw_output_is_sorted_by_dataset_then_pair_index() -> None:
    """Emitted order must be pool order, carrying no information about the draw."""
    pools = [_pool("linux", list(range(2, 20))), _pool("grec", list(range(2, 20)))]
    sample = draw(pools, seed=SEED, max_per_bin=6)
    for key in ("linux", "grec"):
        selected = sample.pair_index[sample.dataset_key == key]
        assert selected.tolist() == sorted(selected.tolist())
    order = [list(SUITE2_DATASETS).index(str(k)) for k in sample.dataset_key]
    assert order == sorted(order)


def test_draw_samples_without_replacement() -> None:
    pool = _pool("coil_del", list(range(2, 50)))
    sample = draw([pool], seed=SEED, max_per_bin=13)
    pairs = list(zip(sample.pair_i.tolist(), sample.pair_j.tolist(), strict=True))
    assert len(pairs) == len(set(pairs))


# --------------------------------------------------------------------------- #
# The dataset_key column -- the truncation the real export exposed
# --------------------------------------------------------------------------- #


def test_key_dtype_is_wide_enough_for_every_registry_key() -> None:
    """``np.full(size, key, dtype=np.str_)`` yields ``<U1`` and truncates. Measured, not assumed."""
    width = int(_KEY_DTYPE.removeprefix("<U"))
    assert width == max(len(key) for key in SUITE2_DATASETS)
    assert width >= len("iam_letter_high")


def test_draw_emits_whole_dataset_keys() -> None:
    pools = [_pool("iam_letter_high", [3, 4, 5]), _pool("iam_letter_low", [3, 4])]
    sample = draw(pools, seed=SEED)
    assert set(map(str, sample.dataset_key)) == {"iam_letter_high", "iam_letter_low"}


def test_check_dataset_keys_rejects_a_truncated_column() -> None:
    """The exact corruption seen on the real export: every key cut to its first character."""
    truncated = np.full(5, "iam_letter_low", dtype=np.str_)
    assert truncated.dtype == np.dtype("<U1")
    with pytest.raises(SamplingError, match="truncation"):
        _check_dataset_keys(truncated)


def test_check_dataset_keys_accepts_registry_keys() -> None:
    _check_dataset_keys(np.asarray(["linux", "coil_del"], dtype=_KEY_DTYPE))


# --------------------------------------------------------------------------- #
# Metadata and writing
# --------------------------------------------------------------------------- #


def test_metadata_records_the_frozen_design_and_the_realised_counts(tmp_path: Path) -> None:
    pools = [_pool("linux", list(range(2, 30)))]
    sample = draw(pools, seed=SEED, max_per_bin=4)
    meta = build_metadata(sample, tmp_path, SEED, 4)

    assert meta["bin_edges"] == list(BIN_EDGES)
    assert meta["seed"] == 42
    assert meta["max_per_bin"] == 4
    assert meta["n_pairs"] == len(sample)
    assert set(meta["n_per_bin"]) == {str(b) for b in range(N_BINS)}
    assert set(meta["bin_population"]) == {str(b) for b in range(N_BINS)}
    assert "linux" in meta["bin_population_by_dataset"]
    assert "size-stratified" in str(meta["stratification"])
    assert meta["content_sha256"] == content_digest(sample)
    json.dumps(meta)  # must be serialisable as written


def test_write_subsample_emits_a_runner_consumable_pair_list(tmp_path: Path) -> None:
    """One ``pair_index`` array per dataset, ascending -- the format ``--pair-list`` requires."""
    pools = [_pool("linux", list(range(2, 20))), _pool("grec", list(range(2, 20)))]
    sample = draw(pools, seed=SEED, max_per_bin=5)
    meta = build_metadata(sample, tmp_path, SEED, 5)
    pooled, written = write_subsample(sample, meta, tmp_path)

    assert pooled.name == SUBSAMPLE_NAME
    assert {p.stem for p in written} == {"linux", "grec"}
    total = 0
    for path in written:
        with np.load(path, allow_pickle=False) as handle:
            assert "pair_index" in handle
            listed = handle["pair_index"]
            assert listed.dtype == np.int64
            assert listed.tolist() == sorted(listed.tolist())
            assert len(set(listed.tolist())) == listed.size
            total += listed.size
    assert total == len(sample)
    assert (tmp_path / PAIR_LIST_SUBDIR).is_dir()


def test_written_pooled_file_round_trips_without_pickle(tmp_path: Path) -> None:
    pools = [_pool("protein", list(range(2, 25)))]
    sample = draw(pools, seed=SEED, max_per_bin=6)
    pooled, _ = write_subsample(sample, build_metadata(sample, tmp_path, SEED, 6), tmp_path)

    with np.load(pooled, allow_pickle=False) as handle:
        assert set(handle.keys()) == {
            "dataset_key",
            "pair_i",
            "pair_j",
            "n_max",
            "bin_index",
            "pair_index",
            "metadata",
        }
        assert handle["pair_i"].dtype == np.int32
        assert handle["pair_j"].dtype == np.int32
        assert handle["n_max"].dtype == np.int32
        assert handle["bin_index"].dtype == np.int8
        assert handle["pair_index"].dtype == np.int64
        assert [str(x) for x in handle["dataset_key"]] == ["protein"] * len(sample)


def test_content_digest_separates_two_different_samples() -> None:
    pool = _pool("grec", list(range(2, 30)))
    a = draw([pool], seed=1, max_per_bin=5)
    b = draw([pool], seed=2, max_per_bin=5)
    assert content_digest(a) != content_digest(b)


def test_content_digest_is_stable_for_an_empty_sample() -> None:
    empty = Subsample(
        dataset_key=np.empty(0, dtype=_KEY_DTYPE),
        pair_i=np.empty(0, np.int32),
        pair_j=np.empty(0, np.int32),
        n_max=np.empty(0, np.int32),
        bin_index=np.empty(0, np.int8),
        pair_index=np.empty(0, np.int64),
    )
    assert len(empty) == 0
    assert content_digest(empty) == content_digest(empty)


# --------------------------------------------------------------------------- #
# The probe pair list -- equal per bin, then equal per dataset
# --------------------------------------------------------------------------- #


def test_pool_position_equals_pair_index() -> None:
    """``triu_indices`` order is ``ged_pair_index`` order, so position *is* the linear index.

    :func:`pool_pair_index` relies on this to keep the exclusion mask O(1) in memory. Asserted here
    rather than assumed, because if it were false every probe pair would name the wrong graph pair
    and nothing would raise.
    """
    for n in (2, 3, 5, 17, 64):
        pool = _pool("linux", list(range(2, 2 + n)))
        expected = indices_of_pairs(pool.pair_i.astype(np.int64), pool.pair_j.astype(np.int64), n)
        assert pool_pair_index(pool).tolist() == expected.tolist()


@pytest.mark.parametrize(
    ("total", "caps", "expected"),
    [
        (3000, [10**9] * 14, [215] * 4 + [214] * 10),  # the real allocation
        (10, [100, 100], [5, 5]),  # exact split
        (11, [100, 100], [6, 5]),  # remainder to the lowest index
        (10, [2, 100], [2, 8]),  # a capped slot returns its excess
        (10, [1, 1, 100], [1, 1, 8]),  # two capped slots
        (5, [1, 1, 1], [1, 1, 1]),  # capacity below demand: allocate what exists
        (0, [10, 10], [0, 0]),
        (7, [3, 3, 3], [3, 2, 2]),
    ],
)
def test_allocate_evenly(total: int, caps: list[int], expected: list[int]) -> None:
    assert allocate_evenly(total, caps) == expected


def test_allocate_evenly_never_exceeds_a_cap_or_the_total() -> None:
    caps = [7, 0, 3, 40, 1]
    alloc = allocate_evenly(25, caps)
    assert all(a <= c for a, c in zip(alloc, caps, strict=True))
    assert sum(alloc) == min(25, sum(caps))


def test_probe_draws_the_requested_total() -> None:
    pools = [_pool("coil_del", list(range(2, 99))), _pool("linux", [3, 4, 5])]
    probe = draw_probe(pools, seed=SEED, total=140)
    assert len(probe) == 140


def _pool_spanning_every_bin(key: str, per_bin: int = 25) -> DatasetPairs:
    """A pool with enough pairs in every one of the 14 bins to absorb an equal allocation."""
    return _pool(key, [BIN_EDGES[b] for b in range(N_BINS) for _ in range(per_bin)])


def test_probe_spreads_equally_across_bins_not_proportionally() -> None:
    """The allocation must not follow bin population, which is the whole point of the design.

    Bin populations here span 300 to 8,425 -- the top bin absorbs every cross-bin pair -- and the
    draw must still come out flat.
    """
    pools = [_pool_spanning_every_bin("coil_del")]
    probe = draw_probe(pools, seed=SEED, total=PROBE_TOTAL)
    drawn = [probe.bin_drawn[b] for b in range(N_BINS)]
    populations = [probe.bin_population[b] for b in range(N_BINS)]

    assert sum(drawn) == PROBE_TOTAL
    assert max(drawn) - min(drawn) <= 1, drawn
    assert max(populations) > 20 * min(populations), populations


def test_probe_redistributes_from_a_bin_too_small_to_take_its_share() -> None:
    """Where a bin cannot absorb its equal share, the surplus goes to bins that can.

    On the real cohort every bin holds far more than 215 candidates so this never fires, but a
    smaller cohort would hit it and must still return the full requested total.
    """
    pools = [_pool("coil_del", list(range(2, 99)))]
    probe = draw_probe(pools, seed=SEED, total=PROBE_TOTAL)
    drawn = [probe.bin_drawn[b] for b in range(N_BINS)]
    populations = [probe.bin_population[b] for b in range(N_BINS)]

    assert sum(drawn) == PROBE_TOTAL
    # Every bin that could not take 214 was drained entirely, not partially filled.
    starved = [b for b in range(N_BINS) if populations[b] < PROBE_TOTAL // N_BINS]
    assert starved, "fixture must contain at least one under-capacity bin"
    for b in starved:
        assert drawn[b] == populations[b]
    assert max(drawn) > PROBE_TOTAL // N_BINS, "surplus must land somewhere"


def test_probe_represents_every_dataset_it_is_given() -> None:
    """A dataset rare in every bin must still appear; the shortfall redistribution guarantees it."""
    pools = [
        _pool("coil_del", list(range(2, 99))),
        _pool("linux", [3, 4]),
        _pool("protein", [50, 51, 52]),
    ]
    probe = draw_probe(pools, seed=SEED, total=PROBE_TOTAL)
    assert set(map(str, probe.dataset_key)) == {"coil_del", "linux", "protein"}


def test_probe_is_reproducible_and_seed_sensitive() -> None:
    pools = [_pool("grec", list(range(2, 60))), _pool("protein", list(range(2, 40)))]
    a = draw_probe(pools, seed=SEED, total=200)
    b = draw_probe(pools, seed=SEED, total=200)
    c = draw_probe(pools, seed=SEED + 1, total=200)
    assert content_digest(a) == content_digest(b)
    assert content_digest(a) != content_digest(c)


def test_probe_excludes_the_subsample_pairs() -> None:
    """Disjointness is the point: a probe timing must never share a pair with an IPFP_MS run."""
    pools = [_pool("grec", list(range(2, 40)))]
    sample = draw(pools, seed=SEED, max_per_bin=20)
    probe = draw_probe(pools, exclude=sample, seed=SEED, total=100)
    assert _overlap(sample, probe) == 0


def test_probe_without_exclusion_may_overlap() -> None:
    """Establishes that the disjointness above is produced by ``exclude``, not by luck."""
    pools = [_pool("grec", list(range(2, 20)))]
    sample = draw(pools, seed=SEED, max_per_bin=60)
    unfiltered = draw_probe(pools, exclude=None, seed=SEED, total=60)
    filtered = draw_probe(pools, exclude=sample, seed=SEED, total=60)
    assert _overlap(sample, filtered) == 0
    assert content_digest(unfiltered) != content_digest(filtered)


def test_probe_emits_valid_pairs_and_strata() -> None:
    pools = [_pool("mutagenicity", list(range(2, 99)))]
    probe = draw_probe(pools, seed=SEED, total=300)
    assert bool((probe.pair_i < probe.pair_j).all())
    assert bool((probe.bin_index >= 0).all())
    assert bool((probe.bin_index < N_BINS).all())
    for k, i, j in zip(probe.pair_index, probe.pair_i, probe.pair_j, strict=True):
        assert pair_from_index(int(k), pools[0].n_graphs) == (int(i), int(j))


def test_probe_metadata_states_the_allocation_rule() -> None:
    """The rule and its rationale must travel with the file, not live only in a work log."""
    pools = [_pool("grec", list(range(2, 40)))]
    probe = draw_probe(pools, seed=SEED, total=100)
    meta = build_probe_metadata(probe, "somewhere", SEED, 100, disjoint=True)
    assert "equal per bin" in str(meta["allocation_rule"])
    assert meta["not_a_cohort_estimate"] is True
    assert meta["disjoint_from_subsample"] is True
    assert meta["seed"] == 42
    assert meta["bin_edges"] == list(BIN_EDGES)
    assert sum(meta["n_per_dataset"].values()) == len(probe)
    json.dumps(meta)


def test_write_probe_emits_the_same_conventions_as_the_subsample(tmp_path: Path) -> None:
    pools = [_pool("grec", list(range(2, 30))), _pool("protein", list(range(2, 30)))]
    probe = draw_probe(pools, seed=SEED, total=120)
    meta = build_probe_metadata(probe, tmp_path, SEED, 120, disjoint=False)
    pooled, written = write_probe(probe, meta, tmp_path)

    assert pooled.name == PROBE_NAME
    assert (tmp_path / PROBE_PAIR_LIST_SUBDIR).is_dir()
    with np.load(pooled, allow_pickle=False) as handle:
        assert set(handle.keys()) == {
            "dataset_key",
            "pair_i",
            "pair_j",
            "n_max",
            "bin_index",
            "pair_index",
            "metadata",
        }
    total = 0
    for path in written:
        with np.load(path, allow_pickle=False) as handle:
            listed = handle["pair_index"]
            assert listed.dtype == np.int64
            assert listed.tolist() == sorted(listed.tolist())
            total += listed.size
    assert total == len(probe)


# --------------------------------------------------------------------------- #
# The bin table
# --------------------------------------------------------------------------- #


def test_bin_table_has_the_schema_the_launcher_codes_against() -> None:
    pools = [_pool("grec", list(range(2, 40))), _pool("protein", list(range(2, 60)))]
    table = build_bin_table(pools)
    assert (
        table["bin_edges"]
        == list(BIN_EDGES)
        == [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 99]
    )
    assert len(table["totals"]) == N_BINS
    for counts in table["datasets"].values():
        assert len(counts) == N_BINS


def test_bin_table_totals_are_the_column_sums() -> None:
    pools = [_pool("grec", list(range(2, 40))), _pool("protein", list(range(2, 60)))]
    table = build_bin_table(pools)
    for b in range(N_BINS):
        assert table["totals"][b] == sum(c[b] for c in table["datasets"].values())
    assert sum(table["totals"]) == sum(p.pair_i.shape[0] for p in pools)


def test_bin_table_flags_a_single_dataset_bin() -> None:
    """A bin one dataset dominates must say so, so no figure quotes it as a size effect."""
    pools = [_pool("mutagenicity", [90, 91, 92, 93]), _pool("protein", [3, 4])]
    table = build_bin_table(pools)
    top = table["dominance"][13]
    assert top["dominant_dataset"] == "mutagenicity"
    assert top["dominant_share"] == 1.0
    assert top["single_dataset"] is True
    assert "mutagenicity" in top["caveat"]
    assert table["metadata"]["single_dataset_bins"]


def test_bin_table_does_not_flag_a_shared_bin() -> None:
    pools = [_pool("mutagenicity", [90, 91]), _pool("protein", [90, 91])]
    table = build_bin_table(pools)
    top = table["dominance"][13]
    assert top["n_datasets_present"] == 2
    assert top["dominant_share"] < DOMINANCE_WARN_SHARE
    assert top["single_dataset"] is False
    assert "caveat" not in top


def test_bin_table_handles_an_empty_bin() -> None:
    pools = [_pool("linux", [3, 3, 3])]
    table = build_bin_table(pools)
    assert table["dominance"][13]["total"] == 0
    assert table["dominance"][13]["dominant_dataset"] is None
    assert table["dominance"][13]["single_dataset"] is False


def test_write_bin_table_round_trips(tmp_path: Path) -> None:
    pools = [_pool("grec", list(range(2, 40)))]
    table = build_bin_table(pools)
    path = write_bin_table(table, tmp_path / BIN_TABLE_NAME)
    reloaded = json.loads(path.read_text())
    assert reloaded["bin_edges"] == list(BIN_EDGES)
    assert reloaded["totals"] == table["totals"]
    assert reloaded["datasets"] == table["datasets"]


# --------------------------------------------------------------------------- #
# Real export
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@requires_export
def test_pool_holds_exactly_the_locked_pair_count() -> None:
    total = 0
    for key in SUITE2_DATASETS:
        n_nodes = read_node_counts(DEFAULT_EXPORT_DIR, key)
        total += comb(int(n_nodes.shape[0]), 2)
    assert total == TOTAL_EXPECTED_PAIRS == 21710892


@pytest.mark.integration
@requires_export
def test_real_draw_hits_the_ceiling_and_reproduces() -> None:
    """All 14 bins exceed 2,000 on the real cohort, so the draw is exactly 28,000 pairs."""
    first = run(DEFAULT_EXPORT_DIR)
    second = run(DEFAULT_EXPORT_DIR)
    assert len(first) == MAX_TOTAL_PAIRS == 28000
    assert content_digest(first) == content_digest(second)
    for b in range(N_BINS):
        assert first.bin_population[b] > MAX_PER_BIN
        assert first.bin_drawn[b] == MAX_PER_BIN
    assert sum(first.bin_population.values()) == TOTAL_EXPECTED_PAIRS


@pytest.mark.integration
@requires_export
def test_real_draw_indexes_into_the_exported_graph_order() -> None:
    """Every emitted pair must be a valid index into its own dataset, with the recorded stratum."""
    sample = run(DEFAULT_EXPORT_DIR)
    for key in SUITE2_DATASETS:
        mask = sample.dataset_key == key
        if not bool(mask.any()):
            continue
        n_nodes = read_node_counts(DEFAULT_EXPORT_DIR, key)
        i = sample.pair_i[mask]
        j = sample.pair_j[mask]
        assert bool((i < j).all())
        assert bool((j < n_nodes.shape[0]).all())
        expected = np.maximum(n_nodes[i], n_nodes[j])
        assert bool((sample.n_max[mask] == expected).all())
        assert bool((sample.bin_index[mask] == bin_of(expected)).all())


@pytest.mark.integration
@requires_export
def test_real_probe_is_3000_pairs_over_all_ten_datasets_and_all_fourteen_bins() -> None:
    """The probe must span both axes, or the cost curve it feeds is fitted on a corner."""
    pools = build_pools(DEFAULT_EXPORT_DIR)
    sample = draw(pools, seed=SEED, max_per_bin=MAX_PER_BIN)
    probe = draw_probe(pools, exclude=sample, seed=SEED, total=PROBE_TOTAL)

    assert len(probe) == PROBE_TOTAL == 3000
    assert set(map(str, probe.dataset_key)) == set(SUITE2_DATASETS)
    drawn = [probe.bin_drawn[b] for b in range(N_BINS)]
    assert sum(drawn) == 3000
    assert sorted(drawn) == [214] * 10 + [215] * 4
    assert _overlap(sample, probe) == 0
    assert content_digest(draw_probe(pools, exclude=sample, seed=SEED, total=PROBE_TOTAL)) == (
        content_digest(probe)
    )


@pytest.mark.integration
@requires_export
@pytest.mark.skipif(
    not (Path(DEFAULT_EXPORT_DIR) / PROBE_NAME).is_file(),
    reason=f"probe absent: {DEFAULT_EXPORT_DIR}",
)
def test_written_probe_indexes_into_the_exported_graph_order() -> None:
    with np.load(Path(DEFAULT_EXPORT_DIR) / PROBE_NAME, allow_pickle=False) as handle:
        keys = np.asarray([str(x) for x in handle["dataset_key"]])
        pair_i = handle["pair_i"]
        pair_j = handle["pair_j"]
        n_max = handle["n_max"]
        stored = json.loads(str(handle["metadata"].item()))

    assert stored["seed"] == 42
    assert stored["n_pairs"] == 3000
    assert stored["not_a_cohort_estimate"] is True
    assert set(keys) == set(SUITE2_DATASETS)
    for key in SUITE2_DATASETS:
        mask = keys == key
        counts = read_node_counts(DEFAULT_EXPORT_DIR, key)
        i, j = pair_i[mask], pair_j[mask]
        assert bool((i < j).all())
        assert bool((j < counts.shape[0]).all())
        assert bool((n_max[mask] == np.maximum(counts[i], counts[j])).all())


@pytest.mark.integration
@requires_export
@pytest.mark.skipif(
    not (Path(DEFAULT_EXPORT_DIR) / BIN_TABLE_NAME).is_file(),
    reason=f"bin table absent: {DEFAULT_EXPORT_DIR}",
)
def test_written_bin_table_matches_the_cohort() -> None:
    """The launcher sizes jobs from this file; its totals must be the locked cohort."""
    table = json.loads((Path(DEFAULT_EXPORT_DIR) / BIN_TABLE_NAME).read_text())
    assert table["bin_edges"] == [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 99]
    assert len(table["totals"]) == 14
    assert sum(table["totals"]) == TOTAL_EXPECTED_PAIRS == 21710892
    assert set(table["datasets"]) == set(SUITE2_DATASETS)
    for key, counts in table["datasets"].items():
        assert len(counts) == 14
        assert sum(counts) == comb(int(read_node_counts(DEFAULT_EXPORT_DIR, key).shape[0]), 2)


@pytest.mark.integration
@requires_export
@pytest.mark.skipif(
    not (Path(DEFAULT_EXPORT_DIR) / BIN_TABLE_NAME).is_file(),
    reason=f"bin table absent: {DEFAULT_EXPORT_DIR}",
)
def test_top_bin_is_disclosed_as_a_single_dataset_statement() -> None:
    """Bin [80, 99) is 97.1 % Mutagenicity. The file must say so, not just the work log."""
    table = json.loads((Path(DEFAULT_EXPORT_DIR) / BIN_TABLE_NAME).read_text())
    top = table["dominance"][13]
    assert top["range"] == [80, 99]
    assert top["dominant_dataset"] == "mutagenicity"
    assert top["dominant_share"] > 0.97
    assert top["single_dataset"] is True
    assert "mutagenicity" in top["caveat"]
    assert any("bin 13" in w for w in table["metadata"]["single_dataset_bins"])

    # [60, 80) is dominated but shared, so it is reported without the single-dataset flag.
    second = table["dominance"][12]
    assert second["dominant_dataset"] == "mutagenicity"
    assert 0.6 < second["dominant_share"] < DOMINANCE_WARN_SHARE
    assert second["single_dataset"] is False


@pytest.mark.integration
@requires_export
@pytest.mark.skipif(
    not (Path(DEFAULT_OUT_DIR) / SUBSAMPLE_NAME).is_file(),
    reason=f"subsample absent: {DEFAULT_OUT_DIR}",
)
def test_written_subsample_matches_a_fresh_draw() -> None:
    """The file on disk must be the seed-42 draw, not a stale artifact of an earlier design."""
    with np.load(Path(DEFAULT_OUT_DIR) / SUBSAMPLE_NAME, allow_pickle=False) as handle:
        stored = json.loads(str(handle["metadata"].item()))
        assert handle["pair_i"].shape[0] == 28000
        assert set(map(str, handle["dataset_key"])) <= set(SUITE2_DATASETS)
    assert stored["seed"] == 42
    assert stored["bin_edges"] == list(BIN_EDGES)
    assert stored["content_sha256"] == content_digest(run(DEFAULT_EXPORT_DIR))
