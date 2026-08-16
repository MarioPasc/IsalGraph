"""Contiguous row bands, and the proof that a set of them tiles ``[0, G)``.

**Row bands, not slices of a linear pair index.**  The consumer indexes a
distance matrix by ``(i, j)``; a band therefore merges by concatenation along
axis 0 and a missing band is a missing block of rows, which is detectable in
one pass.  A linear-index sharding (``ged_exact_runner``'s scheme, correct for
its own sparse pair sets) would require re-scattering into a square matrix and
would make a partial merge look like a matrix full of zeros.

A band is computed over **all** ``G`` columns, not over its strict-upper part.
Work per band is then proportional to band height, so equal-height bands are
also equal-cost, and the merge cannot mis-assemble a triangle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from benchmarks.eval_distance.schema import ShardError


@dataclass(frozen=True, slots=True)
class RowBand:
    """Rows ``[start, stop)`` of a ``(G, G)`` matrix.

    Attributes:
        index: chunk index, ``0 <= index < n_chunks``.
        start: first row, inclusive.
        stop: last row, exclusive.  May equal *start* for an empty tail band.
    """

    index: int
    start: int
    stop: int

    @property
    def height(self) -> int:
        """Number of rows in this band."""
        return self.stop - self.start


def split_bands(n_graphs: int, n_chunks: int) -> tuple[RowBand, ...]:
    """Partition ``[0, n_graphs)`` into *n_chunks* contiguous bands.

    The first ``n_graphs % n_chunks`` bands carry one extra row, so heights
    differ by at most one.  When ``n_chunks > n_graphs`` the tail bands are
    empty rather than absent: a launcher that fans out a fixed array size must
    still get a well-formed shard from every task, and an empty band merges
    to nothing.

    Args:
        n_graphs: cohort size.
        n_chunks: number of shards.

    Returns:
        Bands in ascending order, one per chunk index.

    Raises:
        ShardError: when *n_chunks* is not positive or *n_graphs* is negative.
    """
    if n_chunks < 1:
        raise ShardError(f"n_chunks must be >= 1, got {n_chunks}")
    if n_graphs < 0:
        raise ShardError(f"n_graphs must be >= 0, got {n_graphs}")
    base, remainder = divmod(n_graphs, n_chunks)
    bands: list[RowBand] = []
    cursor = 0
    for index in range(n_chunks):
        height = base + (1 if index < remainder else 0)
        bands.append(RowBand(index=index, start=cursor, stop=cursor + height))
        cursor += height
    return tuple(bands)


def band_for(n_graphs: int, n_chunks: int, chunk_index: int) -> RowBand:
    """Return the single band this task owns.

    Raises:
        ShardError: when *chunk_index* is outside ``[0, n_chunks)``.
    """
    if not 0 <= chunk_index < n_chunks:
        raise ShardError(f"chunk_index {chunk_index} outside [0, {n_chunks})")
    return split_bands(n_graphs, n_chunks)[chunk_index]


def verify_tiling(bands: Sequence[RowBand], n_graphs: int) -> None:
    """Assert that *bands* tile ``[0, n_graphs)`` with no gap and no overlap.

    Args:
        bands: bands in any order.
        n_graphs: the interval that must be covered exactly.

    Raises:
        ShardError: on a gap, an overlap, an inverted band, or a band that
            runs past *n_graphs*.  Each message names the offending rows,
            because "the merge produced a partial matrix" is the failure this
            function exists to make impossible.
    """
    ordered = sorted(bands, key=lambda band: (band.start, band.stop))
    cursor = 0
    for band in ordered:
        if band.stop < band.start:
            raise ShardError(f"band {band.index} is inverted: [{band.start}, {band.stop})")
        if band.height == 0:
            continue
        if band.start < cursor:
            raise ShardError(
                f"bands overlap: rows [{band.start}, {cursor}) are claimed twice "
                f"(band {band.index} starts at {band.start}, coverage already reached {cursor})"
            )
        if band.start > cursor:
            raise ShardError(
                f"bands leave a gap: rows [{cursor}, {band.start}) are covered by no shard"
            )
        cursor = band.stop
    if cursor != n_graphs:
        raise ShardError(
            f"bands cover rows [0, {cursor}) but the cohort has {n_graphs} graphs; "
            f"rows [{cursor}, {n_graphs}) are missing"
        )


__all__ = ["RowBand", "band_for", "split_bands", "verify_tiling"]
