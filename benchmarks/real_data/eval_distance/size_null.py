"""``|n_i - n_j|`` -- the baseline every printed rho has to carry.

Count the nodes, subtract.  No representation, no encoder, no budget.  Against
T-03's certified exact GED it scores rho = 0.71-0.93 on the five Suite-1
datasets, and **IsalGraph clears it on only one of the five**.  A correlation
printed without this column beside it has an unanswerable objection waiting.

**Emitted once per (representation, dataset), restricted to that
representation's own encodable pair set.**  This supersedes CONTRACTS §4's
original "emitted once per dataset (not per representation)", which was wrong,
and the file name gains the representation accordingly:
``{dataset}__{representation}__size_null.npz``.  It thereby stops being a
special case and becomes an ordinary ``{dataset}__{representation}__{metric}``
file with ``metric = "size_null"``.

**Why the restriction is not cosmetic.**  Censoring is not independent of size,
and it differs by representation, so an unrestricted null is computed over
pairs the arm was never evaluated on.  Measured by T-04a on Mutagenicity:
``isalgraph_pruned`` loses 14 of 200 graphs and **every lost graph is larger
than every kept one** (mean 75.8 nodes, max 97, against 25.4 and 48), which
collapses ``sd(|dn|)`` from 16.4 to 8.0.  The whole-cohort null is 0.7538; the
restricted null is **0.6363** -- while the arm itself does not move at all
(0.8322 either way).  ``min_dfs`` loses the same *count* but a **different 14
graphs**, and its restricted null is **0.6817**.  One null per dataset would
therefore have been wrong for at least one of them, and comparing an arm
against a null measured on a different pair set is not a comparison.

This is **D14's censoring bias appearing inside the BASELINE rather than inside
the arm**, which is a distinct failure mode from the one D14 was written for.

**Both views survive in one file.**  ``distance_matrix`` is the full
``|n_i - n_j|`` over every graph in cohort order, because a node count exists
for every graph including one whose encoder failed.  ``defined_mask`` carries
the restriction.  A consumer that honours ``defined_mask`` -- which the
statistics track does on every matrix -- gets the restricted null; one that
ignores it recovers the whole-cohort null.  The contrast between the two is
reportable without a second run.

Usage::

    python -m benchmarks.eval_distance.size_null \\
        --encodings encodings/suite2/mutagenicity__isalgraph_pruned.npz \\
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
from benchmarks.eval_distance.masks import encodable_mask, pair_mask
from benchmarks.eval_distance.schema import (
    SchemaError,
    build_metadata,
    load_encodings,
    write_dense,
)

logger = logging.getLogger(__name__)


def size_null_matrix(
    node_counts: np.ndarray, encodable: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(|n_i - n_j|, defined_mask)``.

    Args:
        node_counts: ``(G,)`` integer node counts.
        encodable: optional ``(G,)`` row mask from :func:`encodable_mask`.
            When given, ``defined_mask`` is restricted to pairs whose **both**
            ends the representation encoded.  When ``None`` the mask is true
            everywhere, which is the whole-cohort null and is correct only for
            a representation that lost no graph.

    Returns:
        ``float64 (G, G)`` distances -- always the full node-count difference,
        since a node count exists for every graph in the cohort -- and a
        ``bool (G, G)`` mask carrying the restriction.
    """
    counts = np.asarray(node_counts, dtype=np.int64)
    matrix = np.abs(counts[:, None] - counts[None, :]).astype(np.float64)
    if encodable is None:
        return matrix, np.ones(matrix.shape, dtype=bool)
    return matrix, pair_mask(encodable)


def run(encodings: Path, out_dir: Path, suite: str | None = None) -> Path:
    """Write ``{dataset}__{representation}__size_null.npz`` for one arm.

    Args:
        encodings: the CONTRACTS §3 file **for the representation the null is
            being restricted to**.  ``graph_ids``, ``node_counts``, ``status``
            and ``length`` are read; the encodings themselves are not.
        out_dir: destination directory.
        suite: override when the input metadata carries none.

    Returns:
        The file that was written.
    """
    source = load_encodings(encodings)
    dataset = str(source.metadata.get("dataset") or encodings.stem.partition("__")[0])
    representation = str(source.metadata.get("representation") or encodings.stem.partition("__")[2])
    if not representation:
        raise SchemaError(f"{encodings}: cannot determine the representation to restrict to")
    resolved_suite = suite or str(source.metadata.get("suite") or "unknown")

    encodable = encodable_mask(source.status, source.length)
    matrix, mask = size_null_matrix(source.node_counts, encodable)
    assert_dense(matrix, mask)

    n_encodable = int(encodable.sum())
    n_excluded = int(source.n_graphs - n_encodable)
    metadata = build_metadata(
        suite=resolved_suite,
        dataset=dataset,
        representation=representation,
        metric="size_null",
        n_graphs=source.n_graphs,
        notes=(
            "trivial baseline |n_i - n_j|, restricted to the pair set "
            f"{representation} itself encoded; no representation is computed"
        ),
        extra={
            "node_counts_source": str(encodings),
            "restricted_to_representation": representation,
            "n_encodable": n_encodable,
            "n_excluded": n_excluded,
            "unrestricted_recoverable": True,
        },
    )
    if n_excluded:
        logger.info(
            "%s/%s: null restricted to %d of %d graphs (%d excluded)",
            dataset,
            representation,
            n_encodable,
            source.n_graphs,
            n_excluded,
        )
    out = out_dir / f"{dataset}__{representation}__size_null.npz"
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
        description=(
            "Emit the |n_i - n_j| baseline for one (representation, dataset), "
            "restricted to that representation's own encodable pair set."
        ),
    )
    parser.add_argument(
        "--encodings",
        required=True,
        type=Path,
        help="the CONTRACTS §3 .npz of the representation to restrict to",
    )
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
