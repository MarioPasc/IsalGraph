"""Assemble row-band shards into the dense CONTRACTS §4 file.

**Refuses to proceed on an incomplete shard set.**  A merge that quietly emits
a partial matrix is the failure mode this module exists to prevent: the gap
would arrive downstream as a block of zeros or NaNs, and a correlation
computed over it is a number nobody can tell is wrong.  Three checks, in
order:

1. every chunk index in ``[0, n_chunks)`` is present, ``n_chunks`` read from
   the shards themselves rather than from a caller-supplied hint;
2. the shards agree on cohort size, ``graph_ids`` and the identity triple;
3. the row bands tile ``[0, G)`` exactly -- no gap, no overlap.

The assembled matrix then goes through the same structural gate the single
chunk runner applies.

Usage::

    python -m benchmarks.eval_distance.distance_merge \\
        --shard-dir distances/suite2 \\
        --basename linux__isalgraph_pruned__levenshtein \\
        --out distances/suite2 [--expect-chunks 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmarks.eval_distance.bands import RowBand, verify_tiling
from benchmarks.eval_distance.gates import assert_dense, degenerate_zero_fraction
from benchmarks.eval_distance.schema import (
    DistanceShard,
    SchemaError,
    ShardError,
    load_shard,
    write_dense,
)

logger = logging.getLogger(__name__)


def discover_shards(shard_dir: Path, basename: str) -> list[Path]:
    """List the shard files for *basename*, ascending by chunk index.

    Raises:
        ShardError: when the directory holds none.
    """
    found = sorted(shard_dir.glob(f"{basename}.shard*.npz"))
    if not found:
        raise ShardError(f"no shard matches {basename}.shard*.npz under {shard_dir}")
    return found


def _chunk_bookkeeping(shards: Sequence[DistanceShard]) -> tuple[dict[int, DistanceShard], int]:
    """Index shards by their declared chunk index and recover ``n_chunks``.

    Raises:
        ShardError: on a duplicate index, a missing declaration, or shards
            that disagree about how many chunks there are.
    """
    by_index: dict[int, DistanceShard] = {}
    declared: set[int] = set()
    for shard in shards:
        index = shard.metadata.get("chunk_index")
        total = shard.metadata.get("n_chunks")
        if not isinstance(index, int) or not isinstance(total, int):
            raise ShardError(
                f"{shard.path} declares chunk_index={index!r} n_chunks={total!r}; both must be "
                f"integers or a missing shard cannot be told from a renamed one"
            )
        if index in by_index:
            raise ShardError(
                f"chunk index {index} appears twice: {by_index[index].path}, {shard.path}"
            )
        by_index[index] = shard
        declared.add(total)
    if len(declared) != 1:
        raise ShardError(f"shards disagree on n_chunks: {sorted(declared)}")
    return by_index, declared.pop()


def _require_complete(by_index: dict[int, DistanceShard], n_chunks: int) -> None:
    """Raise unless every chunk index is present.

    Raises:
        ShardError: naming the missing indices.
    """
    missing = sorted(set(range(n_chunks)) - set(by_index))
    if missing:
        raise ShardError(
            f"shard set is incomplete: {len(missing)} of {n_chunks} chunks are missing "
            f"({missing[:16]}{'...' if len(missing) > 16 else ''}). Refusing to emit a partial "
            f"matrix; rerun the missing chunks"
        )
    stray = sorted(index for index in by_index if index >= n_chunks)
    if stray:
        raise ShardError(f"chunk indices {stray} are outside [0, {n_chunks})")


def _require_agreement(shards: Sequence[DistanceShard]) -> DistanceShard:
    """Check that shards describe the same cohort and return the reference.

    Raises:
        ShardError: on a cohort-size, ``graph_ids`` or identity disagreement.
    """
    reference = shards[0]
    for shard in shards[1:]:
        if shard.n_graphs != reference.n_graphs:
            raise ShardError(
                f"{shard.path} covers {shard.n_graphs} graphs, {reference.path} covers "
                f"{reference.n_graphs}"
            )
        if not np.array_equal(shard.graph_ids, reference.graph_ids):
            raise ShardError(f"{shard.path} and {reference.path} carry different graph_ids")
        for key in ("suite", "dataset", "representation", "metric"):
            if shard.metadata.get(key) != reference.metadata.get(key):
                raise ShardError(
                    f"{shard.path} is {key}={shard.metadata.get(key)!r} but {reference.path} "
                    f"is {key}={reference.metadata.get(key)!r}; these are different matrices"
                )
    return reference


def merge_shards(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, DistanceShard]:
    """Assemble *paths* into a dense matrix, refusing an incomplete set.

    Args:
        paths: shard files, in any order.

    Returns:
        ``(distance_matrix, defined_mask, reference_shard)``.

    Raises:
        ShardError: on an incomplete, overlapping or inconsistent set.
    """
    shards = [load_shard(path) for path in paths]
    by_index, n_chunks = _chunk_bookkeeping(shards)
    _require_complete(by_index, n_chunks)
    reference = _require_agreement(shards)
    bands = [
        RowBand(index=index, start=shard.row_start, stop=shard.row_stop)
        for index, shard in sorted(by_index.items())
    ]
    verify_tiling(bands, reference.n_graphs)
    n = reference.n_graphs
    distance = np.full((n, n), np.nan, dtype=np.float64)
    defined = np.zeros((n, n), dtype=bool)
    for index, shard in sorted(by_index.items()):
        height = shard.row_stop - shard.row_start
        if shard.distance_band.shape != (height, n):
            raise ShardError(
                f"{shard.path} (chunk {index}) holds a {shard.distance_band.shape} band for "
                f"rows [{shard.row_start}, {shard.row_stop}) over {n} columns"
            )
        distance[shard.row_start : shard.row_stop, :] = shard.distance_band
        defined[shard.row_start : shard.row_stop, :] = shard.defined_band
    return distance, defined, reference


def run(shard_dir: Path, basename: str, out_dir: Path, expect_chunks: int | None = None) -> Path:
    """Merge one shard set and write the dense file.

    Args:
        shard_dir: where the shards are.
        basename: ``{dataset}__{representation}__{metric}``.
        out_dir: destination directory.
        expect_chunks: optional cross-check against the shards' own count.

    Returns:
        The dense file that was written.

    Raises:
        ShardError: when the set is incomplete or *expect_chunks* disagrees.
    """
    paths = discover_shards(shard_dir, basename)
    distance, defined, reference = merge_shards(paths)
    if expect_chunks is not None and len(paths) != expect_chunks:
        raise ShardError(f"expected {expect_chunks} shards, found {len(paths)} under {shard_dir}")
    report = assert_dense(distance, defined)
    degenerate_zero_fraction(report)
    metadata = dict(reference.metadata)
    metadata.update(
        {
            "chunk_index": None,
            "row_start": 0,
            "row_stop": reference.n_graphs,
            "merged_from": [path.name for path in paths],
            "merge_max_asymmetry": report.max_asymmetry,
            "merge_offdiag_zero_fraction": report.offdiag_zero_fraction,
            "merge_n_undefined_cells": report.n_undefined,
        }
    )
    out = out_dir / f"{basename}.npz"
    write_dense(
        out,
        distance_matrix=distance,
        graph_ids=reference.graph_ids,
        node_counts=reference.node_counts,
        defined_mask=defined,
        metadata=metadata,
    )
    logger.info("merged %d shards into %s", len(paths), out)
    return out


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="distance_merge",
        description="Merge row-band shards into a dense CONTRACTS §4 distance matrix.",
    )
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--basename", required=True, help="{dataset}__{representation}__{metric}")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--expect-chunks", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 when the shard set is incomplete or inconsistent.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        out = run(args.shard_dir, args.basename, args.out, args.expect_chunks)
    except (ShardError, SchemaError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    print(json.dumps({"dense": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
