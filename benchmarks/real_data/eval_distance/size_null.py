"""``|n_i - n_j|`` -- the baseline every printed rho has to carry.

Count the nodes, subtract.  No representation, no encoder, no budget.  Against
T-03's certified exact GED it scores rho = 0.71-0.93 on the five Suite-1
datasets, and **IsalGraph clears it on only one of the five**.  A correlation
printed without this column beside it has an unanswerable objection waiting.

Emitted once per dataset, not once per representation: ``node_counts`` is
carried identically in every encodings file for a given cohort, so any of them
is a valid source.  The output name drops the representation --
``{dataset}__size_null.npz`` -- and the schema is CONTRACTS §4 like any other
distance file, so the statistics track reads it through the same loader.

Usage::

    python -m benchmarks.eval_distance.size_null \\
        --encodings encodings/suite2/linux__isalgraph_pruned.npz \\
        --out distances/suite2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from benchmarks.eval_distance.gates import assert_dense
from benchmarks.eval_distance.schema import (
    SchemaError,
    build_metadata,
    load_encodings,
    write_dense,
)

logger = logging.getLogger(__name__)


def size_null_matrix(node_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(|n_i - n_j|, all-true mask)``.

    Args:
        node_counts: ``(G,)`` integer node counts.

    Returns:
        ``float64 (G, G)`` distances and a ``bool (G, G)`` mask that is true
        everywhere: a node count exists for every graph in the cohort,
        including one whose encoder failed, so this baseline is defined on
        every pair by construction.  That is the point of it.
    """
    counts = np.asarray(node_counts, dtype=np.int64)
    matrix = np.abs(counts[:, None] - counts[None, :]).astype(np.float64)
    return matrix, np.ones(matrix.shape, dtype=bool)


def run(encodings: Path, out_dir: Path, suite: str | None = None) -> Path:
    """Write ``{dataset}__size_null.npz`` for the cohort behind *encodings*.

    Args:
        encodings: any CONTRACTS §3 file for the dataset; only ``graph_ids``
            and ``node_counts`` are read.
        out_dir: destination directory.
        suite: override when the input metadata carries none.

    Returns:
        The file that was written.
    """
    source = load_encodings(encodings)
    dataset = str(source.metadata.get("dataset") or encodings.stem.partition("__")[0])
    resolved_suite = suite or str(source.metadata.get("suite") or "unknown")
    matrix, mask = size_null_matrix(source.node_counts)
    assert_dense(matrix, mask)
    metadata = build_metadata(
        suite=resolved_suite,
        dataset=dataset,
        representation="size_null",
        metric="size_null",
        n_graphs=source.n_graphs,
        notes="trivial baseline |n_i - n_j|; no representation is computed",
        extra={"node_counts_source": str(encodings)},
    )
    out = out_dir / f"{dataset}__size_null.npz"
    write_dense(
        out,
        distance_matrix=matrix,
        graph_ids=source.graph_ids,
        node_counts=source.node_counts,
        defined_mask=mask,
        metadata=metadata,
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="size_null",
        description="Emit the |n_i - n_j| baseline distance matrix for one dataset.",
    )
    parser.add_argument("--encodings", required=True, type=Path, help="any CONTRACTS §3 .npz")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--suite", default=None, choices=("suite1", "suite2"))
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 on a schema fault.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        out = run(args.encodings, args.out, args.suite)
    except SchemaError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    print(json.dumps({"size_null": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
