"""Property tests for the linear upper-triangle pair index (T-03, CONTRACTS section 6).

The inverse is the single place in the exact-GED pipeline where a silent error is
possible. An off-by-one in ``pair_from_index`` would transpose GED values between
pairs while every merged matrix stayed symmetric, finite and plausible, so no
downstream assertion would fire and roughly a thousand core-hours of cluster time
would produce a wrong journal table.

The suite therefore proves the inverse three ways and cross-checks the three
against each other:

* exhaustive round-trip over **every** ``k`` for every ``N`` in ``2..200``;
* 10**5 random ``k`` for each production cohort size, ``N`` in
  ``{769, 1180, 1253, 2059}``;
* agreement between the ``isqrt`` scalar path, the ``float64``-plus-correction
  vectorised path, and the floating-point-free ``searchsorted`` path.

Chunk splitting is tested for the property SCBI's two-hour job floor actually
needs: contiguous, non-overlapping, exhaustive ranges whose sizes differ by at
most one, with the remainder on the low-numbered chunks rather than in a ragged
tail.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.eval_setup.ged_pair_index import (
    GedPairIndexError,
    index_of_pair,
    indices_of_pairs,
    n_pairs,
    pair_from_index,
    pairs_from_indices,
    pairs_from_indices_searchsorted,
    row_start,
    row_starts,
    split_chunk,
    split_range,
    split_sequence,
)

PRODUCTION_SIZES = (769, 1180, 1253, 2059)


# --------------------------------------------------------------------------- #
# Cardinalities and the forward map
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, 0),
        (1, 0),
        (2, 1),
        (89, 3916),
        (769, 295296),
        (1180, 695610),
        (1253, 784378),
        (2059, 2118711),
    ],
)
def test_n_pairs_matches_the_locked_cohort_counts(n: int, expected: int) -> None:
    """C(n, 2) reproduces every pair count in the locked cohort table."""
    assert n_pairs(n) == expected


def test_n_pairs_rejects_a_negative_count() -> None:
    """A negative graph count is a programming error, not a zero-pair corpus."""
    with pytest.raises(GedPairIndexError):
        n_pairs(-1)


def test_row_start_equals_the_cumulative_row_widths() -> None:
    """R(i) is the running total of row widths, which is what makes rows contiguous."""
    for n in (2, 3, 7, 40, 769):
        running = 0
        for i in range(n):
            assert row_start(i, n) == running
            running += n - i - 1
        assert row_start(n - 1, n) == n_pairs(n)


def test_row_starts_array_agrees_with_the_scalar_form() -> None:
    """The vectorised row-start table matches the scalar formula elementwise."""
    for n in (2, 5, 200, 2059):
        expected = np.array([row_start(i, n) for i in range(n)], dtype=np.int64)
        assert np.array_equal(row_starts(n), expected)


def test_forward_map_enumerates_the_triangle_in_order() -> None:
    """Sweeping (i, j) in row-major order yields 0, 1, 2, ... without gaps."""
    for n in (2, 3, 8, 33):
        seen = [index_of_pair(i, j, n) for i in range(n) for j in range(i + 1, n)]
        assert seen == list(range(n_pairs(n)))


def test_forward_map_matches_numpy_triu_indices() -> None:
    """np.triu_indices order is the linear index order; downstream code relies on it."""
    for n in (2, 9, 64, 300):
        i, j = np.triu_indices(n, k=1)
        assert np.array_equal(indices_of_pairs(i, j, n), np.arange(n_pairs(n), dtype=np.int64))


@pytest.mark.parametrize(("i", "j", "n"), [(0, 0, 5), (3, 1, 5), (-1, 2, 5), (0, 5, 5), (1, 2, 2)])
def test_forward_map_rejects_pairs_outside_the_strict_upper_triangle(
    i: int, j: int, n: int
) -> None:
    """Only 0 <= i < j < n is a pair; anything else is a caller bug."""
    with pytest.raises(GedPairIndexError):
        index_of_pair(i, j, n)


# --------------------------------------------------------------------------- #
# The inverse -- exhaustive
# --------------------------------------------------------------------------- #


def test_scalar_inverse_round_trips_every_index_for_n_up_to_200() -> None:
    """Every k for every N in 2..200 inverts to a pair that maps back to k.

    1,333,300 indices in total: C(201, 3), the exhaustive sweep the contract asks
    for.
    """
    checked = 0
    for n in range(2, 201):
        total = n_pairs(n)
        for k in range(total):
            i, j = pair_from_index(k, n)
            assert 0 <= i < j < n
            assert index_of_pair(i, j, n) == k
            checked += 1
    assert checked == 1_333_300


def test_vectorised_inverse_round_trips_every_index_for_n_up_to_200() -> None:
    """The float64-seeded vectorised inverse is exhaustively exact over the same range."""
    for n in range(2, 201):
        k = np.arange(n_pairs(n), dtype=np.int64)
        i, j = pairs_from_indices(k, n)
        assert np.array_equal(indices_of_pairs(i, j, n), k)


def test_searchsorted_inverse_round_trips_every_index_for_n_up_to_200() -> None:
    """The floating-point-free inverse is exhaustively exact over the same range."""
    for n in range(2, 201):
        k = np.arange(n_pairs(n), dtype=np.int64)
        i, j = pairs_from_indices_searchsorted(k, n)
        assert np.array_equal(indices_of_pairs(i, j, n), k)


def test_the_three_inverses_agree_exhaustively_for_small_n() -> None:
    """Three independent methods give the same answer, so agreement is evidence."""
    for n in range(2, 121):
        k = np.arange(n_pairs(n), dtype=np.int64)
        iv, jv = pairs_from_indices(k, n)
        iss, jss = pairs_from_indices_searchsorted(k, n)
        assert np.array_equal(iv, iss)
        assert np.array_equal(jv, jss)
        scalar = np.array([pair_from_index(int(x), n) for x in k], dtype=np.int64)
        assert np.array_equal(scalar[:, 0], iv)
        assert np.array_equal(scalar[:, 1], jv)


# --------------------------------------------------------------------------- #
# The inverse -- production cohort sizes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", PRODUCTION_SIZES)
def test_inverse_round_trips_100k_random_indices_at_production_sizes(n: int) -> None:
    """10**5 random k per cohort size round-trip through all three inverses."""
    rng = np.random.default_rng(42)
    k = rng.integers(0, n_pairs(n), size=100_000, dtype=np.int64)
    iv, jv = pairs_from_indices(k, n)
    assert np.array_equal(indices_of_pairs(iv, jv, n), k)
    iss, jss = pairs_from_indices_searchsorted(k, n)
    assert np.array_equal(iv, iss)
    assert np.array_equal(jv, jss)
    for pos in rng.choice(k.size, size=500, replace=False):
        assert pair_from_index(int(k[pos]), n) == (int(iv[pos]), int(jv[pos]))


@pytest.mark.parametrize("n", PRODUCTION_SIZES)
def test_inverse_is_exact_at_the_boundaries_of_the_index_space(n: int) -> None:
    """The first, last and every row boundary invert correctly -- the off-by-one sites."""
    total = n_pairs(n)
    assert pair_from_index(0, n) == (0, 1)
    assert pair_from_index(total - 1, n) == (n - 2, n - 1)
    edges = []
    for i in range(n - 1):
        edges.append(row_start(i, n))
        edges.append(row_start(i + 1, n) - 1)
    k = np.asarray(edges, dtype=np.int64)
    iv, jv = pairs_from_indices(k, n)
    assert np.array_equal(indices_of_pairs(iv, jv, n), k)
    assert int(iv[0]) == 0
    assert int(jv[1]) == n - 1


@pytest.mark.parametrize("n", [2, 3, 769, 2059])
def test_inverse_rejects_indices_outside_the_space(n: int) -> None:
    """Out-of-range indices raise rather than clamp; a clamp would be silent corruption."""
    total = n_pairs(n)
    for bad in (-1, total, total + 1):
        with pytest.raises(GedPairIndexError):
            pair_from_index(bad, n)
        with pytest.raises(GedPairIndexError):
            pairs_from_indices(np.array([bad], dtype=np.int64), n)
        with pytest.raises(GedPairIndexError):
            pairs_from_indices_searchsorted(np.array([bad], dtype=np.int64), n)


def test_inverse_handles_an_empty_index_array() -> None:
    """An empty chunk is legal and must not raise."""
    for fn in (pairs_from_indices, pairs_from_indices_searchsorted):
        i, j = fn(np.empty(0, dtype=np.int64), 769)
        assert i.size == 0
        assert j.size == 0


# --------------------------------------------------------------------------- #
# Chunk splitting
# --------------------------------------------------------------------------- #


def _assert_partition(total: int, n_chunks: int) -> list[tuple[int, int]]:
    """Assert the split is a contiguous, exhaustive, non-overlapping partition.

    Args:
        total: Size of the space.
        n_chunks: Number of chunks.

    Returns:
        The list of ``(start, stop)`` ranges.
    """
    ranges = [split_range(total, n_chunks, t) for t in range(n_chunks)]
    assert ranges[0].start == 0
    assert ranges[-1].stop == total
    for prev, cur in zip(ranges, ranges[1:], strict=False):
        assert cur.start == prev.stop, "chunks must be contiguous"
    assert sum(len(r) for r in ranges) == total
    sizes = [len(r) for r in ranges]
    assert max(sizes) - min(sizes) <= 1, "sizes must differ by at most one"
    return [(r.start, r.stop) for r in ranges]


@pytest.mark.parametrize("n", [2, 3, 5, 17, 89, 200, 769])
@pytest.mark.parametrize("n_chunks", [1, 2, 3, 7, 24, 64, 1000])
def test_chunks_partition_the_index_space(n: int, n_chunks: int) -> None:
    """Every (N, n_chunks) combination yields a clean partition of [0, C(N,2))."""
    total = n_pairs(n)
    _assert_partition(total, n_chunks)
    covered = np.zeros(total, dtype=bool)
    for t in range(n_chunks):
        rng = split_chunk(n, n_chunks, t)
        assert not covered[rng.start : rng.stop].any(), "chunks must not overlap"
        covered[rng.start : rng.stop] = True
    assert covered.all(), "chunks must be exhaustive"


def test_remainder_goes_to_the_low_numbered_chunks_not_a_ragged_tail() -> None:
    """The remainder is spread over the first chunks; the last chunk is never short.

    A fixed block size with a ragged tail is what produces the sub-two-hour task
    SCBI's job floor forbids, so this is a scheduling requirement, not a nicety.
    """
    sizes = [len(split_range(100, 7, t)) for t in range(7)]
    assert sizes == [15, 15, 14, 14, 14, 14, 14]
    assert sum(sizes) == 100
    # Non-increasing: no chunk is smaller than a later one.
    assert all(a >= b for a, b in zip(sizes, sizes[1:], strict=False))
    ragged = [len(split_range(n_pairs(769), 24, t)) for t in range(24)]
    assert max(ragged) - min(ragged) <= 1
    assert sum(ragged) == n_pairs(769)


def test_more_chunks_than_pairs_yields_empty_chunks_rather_than_an_error() -> None:
    """A launcher that overshoots produces no-op tasks, not a partial index space."""
    total = n_pairs(3)  # 3 pairs
    ranges = _assert_partition(total, 10)
    assert [stop - start for start, stop in ranges] == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert split_range(total, 10, 9).is_empty


def test_single_chunk_owns_everything() -> None:
    """n_chunks == 1 is the whole space in one range."""
    rng = split_chunk(769, 1, 0)
    assert (rng.start, rng.stop) == (0, n_pairs(769))
    assert len(rng) == n_pairs(769)


def test_zero_sized_space_splits_without_error() -> None:
    """A one-graph corpus has no pairs; splitting it must still be well defined."""
    for t in range(3):
        assert split_chunk(1, 3, t).is_empty


@pytest.mark.parametrize(
    ("total", "n_chunks", "index"), [(10, 0, 0), (10, -1, 0), (10, 3, 3), (10, 3, -1), (-1, 3, 0)]
)
def test_split_rejects_inadmissible_specs(total: int, n_chunks: int, index: int) -> None:
    """A bad chunk spec raises; a silently wrong chunk would lose pairs."""
    with pytest.raises(GedPairIndexError):
        split_range(total, n_chunks, index)


def test_pair_list_splits_by_the_same_rule() -> None:
    """--pair-list splits the sampled list, not the triangle, and stays exhaustive."""
    pairs = np.arange(0, 2000, 3, dtype=np.int64)
    pieces = [split_sequence(pairs, 7, t) for t in range(7)]
    assert np.array_equal(np.concatenate(pieces), pairs)
    sizes = [p.size for p in pieces]
    assert max(sizes) - min(sizes) <= 1
    assert all(a >= b for a, b in zip(sizes, sizes[1:], strict=False))
