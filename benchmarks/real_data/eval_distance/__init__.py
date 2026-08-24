"""Sharded pairwise-distance driver for the T-06 recomputation.

Claim B needs one distance matrix per ``(suite, dataset, representation,
metric)``.  Suite 2 is 21,710,892 unordered pairs over 16,370 graphs, with
COIL-DEL contributing 7,603,050 and Mutagenicity 8,158,780.  The pre-existing
:func:`benchmarks.eval_setup.levenshtein_computer.compute_levenshtein_matrix`
is a dense Python double loop with no chunking, no sharding and no resume; it
is kept here as a **differential oracle** and is never called by this package.

Four entry points, all writing the CONTRACTS §4 schema::

    python -m benchmarks.eval_distance.distance_runner   # one row band
    python -m benchmarks.eval_distance.distance_merge    # bands -> dense
    python -m benchmarks.eval_distance.size_null         # |n_i - n_j| baseline

plus :mod:`benchmarks.eval_distance.masks`, the helpers the statistics track
uses to read a dense matrix.

**Sharding is by contiguous row bands, never by a linear pair index.**  The
consumer indexes by ``(i, j)``, so a band merges by concatenation and the
tiling of ``[0, G)`` is checkable in one pass.  A band is computed over all
``G`` columns rather than over its strict-upper part alone: the redundancy is
a factor of two on a code path that runs at 6-33 M pairs/s, and it buys a
merge that cannot mis-assemble a triangle.
"""

from __future__ import annotations

__all__: list[str] = []
