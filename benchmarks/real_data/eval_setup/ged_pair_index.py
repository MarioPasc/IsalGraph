"""Linear upper-triangle pair indexing for the exact-GED programme (T-03).

Every pair of distinct graphs ``(i, j)`` with ``0 <= i < j < N`` is addressed by a
single integer ``k``, so that a job array can own a contiguous half-open range of
``k`` and shards merge deterministically by index.

The map is frozen in ``.claude/notes/2026-08-12-exact-ged/CONTRACTS.md`` §6::

    k(i, j) = i*N - i*(i+1)//2 + (j - i - 1),    0 <= i < j < N,   0 <= k < C(N, 2)

Writing ``R(i) = i*N - i*(i+1)//2`` for the start of row ``i``, the identity
``R(i) = sum_{a<i} (N - a - 1)`` holds, so rows are laid out contiguously in
increasing ``i`` and, within a row, in increasing ``j``.

Correctness of the inverse is the load-bearing property of this module. An
off-by-one in ``pair_from_index`` would transpose GED values between pairs while
leaving every merged matrix symmetric, finite and entirely plausible, so nothing
downstream would catch it. Three independent inverses are therefore provided and
cross-checked against one another in ``tests/unit/test_ged_pair_index.py``:

1. :func:`pair_from_index` -- scalar, integer-exact via :func:`math.isqrt`, with a
   correction loop that terminates only when ``R(i) <= k < R(i+1)``.
2. :func:`pairs_from_indices` -- vectorised, ``float64`` ``sqrt`` seed plus a
   vectorised correction loop, followed by an *unconditional* elementwise
   re-derivation of ``k`` that raises on any mismatch.
3. :func:`pairs_from_indices_searchsorted` -- vectorised, no floating point at
   all; materialises the row-start table and bisects it.

Chunk splitting spreads the remainder over the low-numbered chunks rather than
leaving a ragged tail, because SCBI's two-hour job floor (design note §3) forbids
a short remainder task.

The module is stdlib + numpy only and imports nothing from this repository, so it
is safe to import inside a worker process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ChunkRange",
    "GedPairIndexError",
    "index_of_pair",
    "indices_of_pairs",
    "n_pairs",
    "pair_from_index",
    "pairs_from_indices",
    "pairs_from_indices_searchsorted",
    "row_start",
    "row_starts",
    "split_chunk",
    "split_range",
    "split_sequence",
]


class GedPairIndexError(Exception):
    """Raised when a pair index, a graph count or a chunk spec is out of range."""


# --------------------------------------------------------------------------- #
# Cardinalities and row offsets
# --------------------------------------------------------------------------- #


def n_pairs(n: int) -> int:
    """Return ``C(n, 2)``, the number of unordered pairs of distinct graphs.

    Args:
        n: Number of graphs. Must be non-negative.

    Returns:
        ``n * (n - 1) // 2``, computed in Python integers so it is exact for any
        ``n`` (no ``float`` intermediate, no overflow).

    Raises:
        GedPairIndexError: If ``n`` is negative.
    """
    if n < 0:
        raise GedPairIndexError(f"n must be non-negative, got {n}")
    return n * (n - 1) // 2


def row_start(i: int, n: int) -> int:
    """Return ``R(i) = i*n - i*(i+1)//2``, the linear index of pair ``(i, i+1)``.

    Args:
        i: Row (smaller endpoint). ``0 <= i <= n``.
        n: Number of graphs.

    Returns:
        The linear index at which row ``i`` starts. ``R(n-1) == R(n) == C(n, 2)``.
    """
    return i * n - i * (i + 1) // 2


def row_starts(n: int) -> np.ndarray:
    """Return ``R(0..n-1)`` as an ``int64`` array, for bisection-based inversion.

    Args:
        n: Number of graphs. Must be at least 1.

    Returns:
        Array of shape ``(n,)``, strictly increasing on ``[0, n-1)``.

    Raises:
        GedPairIndexError: If ``n < 1``.
    """
    if n < 1:
        raise GedPairIndexError(f"n must be at least 1, got {n}")
    i = np.arange(n, dtype=np.int64)
    return i * np.int64(n) - i * (i + np.int64(1)) // np.int64(2)


# --------------------------------------------------------------------------- #
# Forward map
# --------------------------------------------------------------------------- #


def index_of_pair(i: int, j: int, n: int) -> int:
    """Return the linear upper-triangle index ``k`` of the pair ``(i, j)``.

    Args:
        i: Smaller endpoint, ``0 <= i < j``.
        j: Larger endpoint, ``i < j < n``.
        n: Number of graphs.

    Returns:
        ``k = i*n - i*(i+1)//2 + (j - i - 1)`` in ``[0, C(n, 2))``.

    Raises:
        GedPairIndexError: If ``(i, j)`` is not a valid strictly-upper pair.
    """
    if not (0 <= i < j < n):
        raise GedPairIndexError(f"require 0 <= i < j < n, got i={i}, j={j}, n={n}")
    return row_start(i, n) + (j - i - 1)


def indices_of_pairs(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    """Vectorised forward map.

    Args:
        i: Integer array of smaller endpoints.
        j: Integer array of larger endpoints, elementwise greater than ``i``.
        n: Number of graphs.

    Returns:
        ``int64`` array of linear indices, same shape as the inputs.

    Raises:
        GedPairIndexError: If any element violates ``0 <= i < j < n``.
    """
    ii = np.asarray(i, dtype=np.int64)
    jj = np.asarray(j, dtype=np.int64)
    if ii.shape != jj.shape:
        raise GedPairIndexError(f"i and j must have the same shape, got {ii.shape} and {jj.shape}")
    if ii.size and not (
        bool(np.all(ii >= 0)) and bool(np.all(ii < jj)) and bool(np.all(jj < np.int64(n)))
    ):
        raise GedPairIndexError(f"some pair violates 0 <= i < j < n for n={n}")
    return ii * np.int64(n) - ii * (ii + np.int64(1)) // np.int64(2) + (jj - ii - np.int64(1))


# --------------------------------------------------------------------------- #
# Inverse map -- scalar, integer-exact
# --------------------------------------------------------------------------- #


def pair_from_index(k: int, n: int) -> tuple[int, int]:
    """Invert the linear upper-triangle index. Integer-exact, no floating point.

    ``R(i) <= k`` rearranges to ``i**2 - i*(2n - 1) + 2k >= 0``, whose smaller root
    is ``((2n - 1) - sqrt((2n - 1)**2 - 8k)) / 2``. The seed uses
    :func:`math.isqrt`, which is exact, so the seed is off by at most one; the
    correction loop then closes on the unique ``i`` with ``R(i) <= k < R(i+1)``
    and is what makes the result independent of how good the seed was.

    Args:
        k: Linear index, ``0 <= k < C(n, 2)``.
        n: Number of graphs.

    Returns:
        ``(i, j)`` with ``0 <= i < j < n`` and ``index_of_pair(i, j, n) == k``.

    Raises:
        GedPairIndexError: If ``k`` is out of range, or -- defensively -- if the
            correction loop fails to converge.
    """
    total = n_pairs(n)
    if not (0 <= k < total):
        raise GedPairIndexError(f"k must satisfy 0 <= k < C({n}, 2) = {total}, got {k}")

    b = 2 * n - 1
    disc = b * b - 8 * k
    # disc >= (2n-1)**2 - 8*(C(n,2) - 1) = (2n-1)**2 - 4n(n-1) + 8 = 9 > 0, always.
    i = (b - math.isqrt(disc)) // 2
    if i < 0:
        i = 0
    elif i > n - 2:
        i = n - 2

    # Close on R(i) <= k < R(i+1). At most a couple of steps from the isqrt seed;
    # the generous bound exists so a wrong seed raises instead of looping forever.
    for _ in range(64):
        if row_start(i, n) > k:
            i -= 1
        elif row_start(i + 1, n) <= k:
            i += 1
        else:
            j = k - row_start(i, n) + i + 1
            if not (0 <= i < j < n):
                raise GedPairIndexError(f"inverse produced an invalid pair ({i}, {j}) for k={k}")
            return int(i), int(j)
    raise GedPairIndexError(f"inverse failed to converge for k={k}, n={n}")


# --------------------------------------------------------------------------- #
# Inverse map -- vectorised
# --------------------------------------------------------------------------- #


def pairs_from_indices(k: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised inverse: ``float64`` seed, correction loop, verified forward map.

    The seed is the same closed form as :func:`pair_from_index` but evaluated in
    ``float64``. For ``n = 2059`` the discriminant peaks near ``1.7e7``, far inside
    the exact-integer range of ``float64``, so a correctly-rounded ``sqrt`` puts the
    seed within one of the truth; the correction loop absorbs that, and the closing
    re-derivation of ``k`` from ``(i, j)`` makes an undetected error impossible
    rather than merely unlikely.

    Args:
        k: Integer array of linear indices, each in ``[0, C(n, 2))``.
        n: Number of graphs.

    Returns:
        ``(i, j)``, both ``int64`` arrays shaped like ``k``.

    Raises:
        GedPairIndexError: If any index is out of range, if the correction loop
            does not converge, or if the verification pass finds any
            ``index_of_pair(i, j, n) != k``.
    """
    kk = np.asarray(k, dtype=np.int64)
    total = n_pairs(n)
    if kk.size and (int(kk.min()) < 0 or int(kk.max()) >= total):
        raise GedPairIndexError(f"some index falls outside [0, C({n}, 2) = {total})")
    if kk.size == 0:
        empty = np.empty(kk.shape, dtype=np.int64)
        return empty, empty.copy()

    b = np.float64(2 * n - 1)
    disc = np.maximum(b * b - 8.0 * kk.astype(np.float64), 0.0)
    i = np.floor((b - np.sqrt(disc)) / 2.0).astype(np.int64)
    np.clip(i, 0, n - 2, out=i)

    n64 = np.int64(n)
    for _ in range(64):
        start = i * n64 - i * (i + np.int64(1)) // np.int64(2)
        nxt = (i + np.int64(1)) * n64 - (i + np.int64(1)) * (i + np.int64(2)) // np.int64(2)
        too_high = start > kk
        too_low = nxt <= kk
        if not (bool(too_high.any()) or bool(too_low.any())):
            j = kk - start + i + np.int64(1)
            _verify_inverse(kk, i, j, n)
            return i, j
        i = i - too_high.astype(np.int64) + too_low.astype(np.int64)
        np.clip(i, 0, n - 2, out=i)
    raise GedPairIndexError(f"vectorised inverse failed to converge for n={n}")


def pairs_from_indices_searchsorted(k: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised inverse by bisection of the row-start table. No floating point.

    Independent of :func:`pairs_from_indices` in method as well as in code, which
    is what makes the agreement between them in the test suite evidence rather
    than a tautology. Costs ``O(n)`` memory, so it is the cross-check rather than
    the production path.

    Args:
        k: Integer array of linear indices, each in ``[0, C(n, 2))``.
        n: Number of graphs.

    Returns:
        ``(i, j)``, both ``int64`` arrays shaped like ``k``.

    Raises:
        GedPairIndexError: If any index is out of range or verification fails.
    """
    kk = np.asarray(k, dtype=np.int64)
    total = n_pairs(n)
    if kk.size and (int(kk.min()) < 0 or int(kk.max()) >= total):
        raise GedPairIndexError(f"some index falls outside [0, C({n}, 2) = {total})")
    if kk.size == 0:
        empty = np.empty(kk.shape, dtype=np.int64)
        return empty, empty.copy()

    starts = row_starts(n)
    i = (np.searchsorted(starts, kk, side="right") - 1).astype(np.int64)
    np.clip(i, 0, n - 2, out=i)
    j = kk - starts[i] + i + np.int64(1)
    _verify_inverse(kk, i, j, n)
    return i, j


def _verify_inverse(k: np.ndarray, i: np.ndarray, j: np.ndarray, n: int) -> None:
    """Re-derive ``k`` from ``(i, j)`` and raise on any mismatch.

    Always on. The cost is a handful of integer operations per pair against
    seconds of solver time, and the failure this guards against is silent.

    Args:
        k: The indices that were inverted.
        i: Recovered smaller endpoints.
        j: Recovered larger endpoints.
        n: Number of graphs.

    Raises:
        GedPairIndexError: If any recovered pair is out of range or does not map
            back to its own index.
    """
    if not (bool(np.all(i >= 0)) and bool(np.all(i < j)) and bool(np.all(j < np.int64(n)))):
        raise GedPairIndexError(f"inverse produced a pair outside 0 <= i < j < n for n={n}")
    back = i * np.int64(n) - i * (i + np.int64(1)) // np.int64(2) + (j - i - np.int64(1))
    bad = back != k
    if bool(bad.any()):
        first = int(np.flatnonzero(bad)[0])
        raise GedPairIndexError(
            f"inverse round-trip failed at position {first}: "
            f"k={int(k[first])} -> (i={int(i[first])}, j={int(j[first])}) -> {int(back[first])}"
        )


# --------------------------------------------------------------------------- #
# Chunk splitting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ChunkRange:
    """One task's contiguous half-open range ``[start, stop)`` of positions.

    Attributes:
        index: Chunk index ``t``, ``0 <= t < n_chunks``.
        n_chunks: Total number of chunks the space was split into.
        start: First position owned by this chunk.
        stop: One past the last position owned by this chunk.
    """

    index: int
    n_chunks: int
    start: int
    stop: int

    def __len__(self) -> int:
        """Return the number of positions in the range."""
        return self.stop - self.start

    @property
    def is_empty(self) -> bool:
        """Whether the chunk owns no positions."""
        return self.stop <= self.start


def split_range(total: int, n_chunks: int, index: int) -> ChunkRange:
    """Split ``[0, total)`` into ``n_chunks`` near-equal parts; return part ``index``.

    Sizes differ by at most one and the remainder goes to the **low-numbered**
    chunks. A fixed block size with a ragged tail is deliberately not used: the
    short tail task is exactly what SCBI's two-hour job floor forbids (design note
    §3). When ``n_chunks > total`` the surplus chunks are empty rather than
    erroring, so a launcher that overshoots produces no-op tasks instead of a
    partially covered index space.

    Args:
        total: Size of the space to split. Must be non-negative.
        n_chunks: Number of chunks. Must be at least 1.
        index: Which chunk to return, ``0 <= index < n_chunks``.

    Returns:
        The :class:`ChunkRange` owned by ``index``.

    Raises:
        GedPairIndexError: If ``total`` is negative, ``n_chunks < 1``, or ``index``
            is out of range.
    """
    if total < 0:
        raise GedPairIndexError(f"total must be non-negative, got {total}")
    if n_chunks < 1:
        raise GedPairIndexError(f"n_chunks must be at least 1, got {n_chunks}")
    if not (0 <= index < n_chunks):
        raise GedPairIndexError(f"require 0 <= index < n_chunks, got {index} of {n_chunks}")
    base, rem = divmod(total, n_chunks)
    start = index * base + min(index, rem)
    stop = start + base + (1 if index < rem else 0)
    return ChunkRange(index=index, n_chunks=n_chunks, start=start, stop=stop)


def split_chunk(n: int, n_chunks: int, index: int) -> ChunkRange:
    """Split the whole upper triangle of an ``n``-graph corpus into chunks.

    Args:
        n: Number of graphs.
        n_chunks: Number of chunks.
        index: Which chunk to return.

    Returns:
        The :class:`ChunkRange` of linear pair indices owned by ``index``.

    Raises:
        GedPairIndexError: As :func:`split_range`.
    """
    return split_range(n_pairs(n), n_chunks, index)


def split_sequence(pairs: np.ndarray, n_chunks: int, index: int) -> np.ndarray:
    """Take chunk ``index`` of an explicit pair list, by the same even-split rule.

    Used by ``--pair-list``: stage 1 computes a sampled subset, and the array of
    sampled indices -- not the full triangle -- is what gets split across tasks.

    Args:
        pairs: ``int64`` array of linear pair indices.
        n_chunks: Number of chunks.
        index: Which chunk to return.

    Returns:
        A view of ``pairs`` holding this chunk's positions.

    Raises:
        GedPairIndexError: As :func:`split_range`.
    """
    arr = np.asarray(pairs, dtype=np.int64)
    rng = split_range(int(arr.size), n_chunks, index)
    return arr[rng.start : rng.stop]
