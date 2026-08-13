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
    DEFAULT_OUT_DIR,
    MAX_PER_BIN,
    MAX_TOTAL_PAIRS,
    N_BINS,
    PAIR_LIST_SUBDIR,
    SEED,
    SUBSAMPLE_NAME,
    DatasetPairs,
    SamplingError,
    Subsample,
    _check_dataset_keys,
    bin_of,
    build_metadata,
    build_pairs,
    content_digest,
    draw,
    read_node_counts,
    run,
    write_subsample,
)
from benchmarks.eval_setup.export_graphs_suite2 import (
    DEFAULT_EXPORT_DIR,
    SUITE2_DATASETS,
    TOTAL_EXPECTED_PAIRS,
)
from benchmarks.eval_setup.ged_pair_index import pair_from_index

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
