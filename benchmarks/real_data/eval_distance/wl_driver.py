"""The WL subtree comparator's distance matrix -- its own driver, by necessity.

``wl_subtree`` is one of preregistration §4.1's **seven** Claim-B comparators,
but it is a :class:`~isalgraph.competitors.base.VectorBackend`, not a
``ReprBackend``: it has no ``encode()``, so there is no CONTRACTS §3 encodings
file for it and the encodings -> distance pipeline **structurally cannot feed
it**.  Hence this driver.  It reads the cohort directly and emits a CONTRACTS
§4 file the statistics track loads through the same loader as every other
distance matrix.

**This is the one sanctioned exception to "the distance track never opens a
cohort file".**  It is an exception because WL consumes graphs rather than
strings, not because opening cohorts is convenient.  To keep the join exact
anyway, ``--reference-encodings`` is **required**: ``graph_ids`` are checked
element-wise against a representation that did go through the encoding track,
so a cohort-order drift fails here rather than silently misaligning a rho.

**h = 2, and it is frozen.**  ``h`` is taken from the backend's own
``WL_ROUNDS`` and is not a CLI argument, because the failure this guards
against is tuning ``h`` on a correlation with GED.  There is **no grakel
off-by-one**: ``grakel(n_iter=k) == ours(h=k)`` exactly (corrected
2026-08-15), so ``h = 2`` is ``n_iter = 2``.  The nearby
``eval_setup/wl_kernel_computer.py`` defaulted to ``n_iter = 5`` until
2026-08-23, three refinement rounds past frozen.

**Fitting is per dataset, never per batch.**  ``VectorBackend.fit`` builds the
colour vocabulary, and a vocabulary built on a subset yields different
distances -- which would make the matrix depend on batching order.  This driver
therefore fits on the whole dataset before computing any row band, and a shard
is a slice of the *output*, never of the fit.

**Every pair is defined.**  WL needs no budget and cannot fail on a graph, so
``defined_mask`` is true everywhere and no graph is ever lost.  That also means
this representation contributes nothing to ``c``.  The kernel distance is a
**pseudometric**: non-isomorphic graphs WL cannot separate receive exactly
0.0, which is a property of WL and not a fault -- the degeneracy gate exists to
catch the different failure of an *unfilled buffer*.

Usage::

    python -m benchmarks.eval_distance.wl_driver \\
        --cohort .../exported_suite2/linux.npz \\
        --reference-encodings encodings/suite2/linux__isalgraph_pruned.npz \\
        --out distances/suite2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.eval_distance.gates import assert_dense, check_dense, degenerate_zero_fraction
from benchmarks.eval_distance.schema import (
    SchemaError,
    build_metadata,
    load_encodings,
    write_dense,
)

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

REPRESENTATION = "wl_subtree"
METRIC = "kernel"


def cohort_graphs(cohort: Path) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild every graph in a CSR cohort archive.

    Args:
        cohort: the exported ``{dataset}.npz`` with ``n_nodes``,
            ``edge_offsets``, ``edges`` and ``graph_ids``.

    Returns:
        ``(graphs, graph_ids, node_counts, edge_counts)``.

    Raises:
        SchemaError: when a required array is absent.
    """
    import networkx as nx

    with np.load(cohort, allow_pickle=False) as handle:
        missing = {"n_nodes", "edge_offsets", "edges", "graph_ids"} - set(handle.files)
        if missing:
            raise SchemaError(f"{cohort}: not a CSR cohort archive, missing={sorted(missing)}")
        node_counts = handle["n_nodes"]
        offsets = handle["edge_offsets"]
        edges = handle["edges"]
        graph_ids = np.asarray(handle["graph_ids"], dtype="<U16")

    graphs: list[nx.Graph] = []
    edge_counts = np.zeros(node_counts.shape[0], dtype=np.int32)
    for i in range(int(node_counts.shape[0])):
        graph = nx.Graph()
        graph.add_nodes_from(range(int(node_counts[i])))
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        graph.add_edges_from(zip(edges[0, lo:hi].tolist(), edges[1, lo:hi].tolist(), strict=True))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graphs.append(graph)
        edge_counts[i] = graph.number_of_edges()
    return graphs, graph_ids, np.asarray(node_counts, dtype=np.int32), edge_counts


def feature_table(graphs: list[Any]) -> tuple[np.ndarray, int]:
    """Fit WL on the whole dataset and return the dense count matrix.

    Fitting happens once, on every graph, before any row is computed:
    ``VectorBackend.fit`` builds the colour vocabulary and a vocabulary built
    on a subset produces different distances.

    Args:
        graphs: every graph in the dataset, in cohort order.

    Returns:
        ``(counts, h)`` where ``counts`` is ``float64 (G, V)`` over the fitted
        vocabulary and ``h`` is the frozen round count actually used.
    """
    from isalgraph.competitors.backends.wl import WL_ROUNDS, WLSubtree

    backend = WLSubtree(h=WL_ROUNDS, normalize=False)
    backend.fit(graphs)
    vocabulary = backend.vocabulary
    index = {colour: j for j, colour in enumerate(vocabulary)}
    counts = np.zeros((len(graphs), len(vocabulary)), dtype=np.float64)
    for i, graph in enumerate(graphs):
        for colour, count in backend.features(graph).items():
            position = index.get(colour)
            if position is not None:
                counts[i, position] = float(count)
    return counts, int(WL_ROUNDS)


def kernel_distance_matrix(counts: np.ndarray) -> np.ndarray:
    """Return the kernel-induced distance ``sqrt(K_ii + K_jj - 2 K_ij)``.

    ``K`` is the unnormalised linear kernel on the WL count vectors, which is
    the WL subtree kernel at the fitted number of rounds.  Normalisation is
    deliberately not applied: it would divide out the graph-size signal GED
    depends on.

    Args:
        counts: ``float64 (G, V)`` feature counts.

    Returns:
        ``float64 (G, G)``, symmetric with an exactly zero diagonal.
    """
    gram = counts @ counts.T
    diagonal = np.diagonal(gram)
    squared = diagonal[:, None] + diagonal[None, :] - 2.0 * gram
    # Clamp the negative values floating-point error puts just below zero; the
    # kernel is PSD so the true values cannot be negative.
    np.maximum(squared, 0.0, out=squared)
    matrix = np.sqrt(squared)
    # Force exact symmetry and an exact zero diagonal: sqrt of a clamped
    # difference is symmetric to within an ulp, and CONTRACTS §4 wants exact.
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _check_join(graph_ids: np.ndarray, reference: Path) -> None:
    """Fail unless *graph_ids* matches the reference encodings element-wise."""
    other = load_encodings(reference)
    if other.graph_ids.shape != graph_ids.shape:
        raise SchemaError(
            f"{reference}: {other.graph_ids.shape[0]} graph_ids against the cohort's "
            f"{graph_ids.shape[0]}"
        )
    if not np.array_equal(other.graph_ids, graph_ids):
        first = int(np.argmax(other.graph_ids != graph_ids))
        raise SchemaError(
            f"{reference}: graph_ids diverge from the cohort at position {first} "
            f"({other.graph_ids[first]!r} against {graph_ids[first]!r})"
        )


def run(cohort: Path, reference_encodings: Path, out_dir: Path, suite: str | None = None) -> Path:
    """Write ``{dataset}__wl_subtree__kernel.npz`` for one dataset.

    Args:
        cohort: the exported CSR archive for the dataset.
        reference_encodings: a CONTRACTS §3 file for the same dataset, used
            only to verify the ``graph_ids`` join.
        out_dir: destination directory.
        suite: override when the reference metadata carries none.

    Returns:
        The file that was written.
    """
    graphs, graph_ids, node_counts, _ = cohort_graphs(cohort)
    _check_join(graph_ids, reference_encodings)
    reference = load_encodings(reference_encodings)
    dataset = str(reference.metadata.get("dataset") or cohort.stem)
    resolved_suite = suite or str(reference.metadata.get("suite") or "unknown")

    counts, rounds = feature_table(graphs)
    matrix = kernel_distance_matrix(counts)
    # WL needs no budget and cannot fail on a graph, so every pair is defined.
    mask = np.ones(matrix.shape, dtype=bool)
    assert_dense(matrix, mask)
    report = check_dense(matrix, mask)
    degenerate_zero_fraction(report)

    metadata = build_metadata(
        suite=resolved_suite,
        dataset=dataset,
        representation=REPRESENTATION,
        metric=METRIC,
        n_graphs=int(graph_ids.shape[0]),
        notes=(
            f"WL subtree kernel distance at the frozen h = {rounds} "
            "(grakel n_iter = h; there is no off-by-one); unnormalised; fitted "
            "once on the whole dataset, never per batch"
        ),
        extra={
            "h": rounds,
            "vocabulary_size": int(counts.shape[1]),
            "normalize": False,
            "cohort_source": str(cohort),
            "graph_ids_verified_against": str(reference_encodings),
            "pseudometric": True,
            "bit_count": None,
            "bit_count_reason": (
                "no bit count: a feature-vector cost would measure the choice of "
                "container, not the encoding"
            ),
        },
    )
    out = out_dir / f"{dataset}__{REPRESENTATION}__{METRIC}.npz"
    write_dense(
        out,
        distance_matrix=matrix,
        graph_ids=graph_ids,
        node_counts=node_counts,
        defined_mask=mask,
        metadata=metadata,
    )
    logger.info(
        "%s: %d graphs, vocabulary %d, h = %d -> %s",
        dataset,
        graph_ids.shape[0],
        counts.shape[1],
        rounds,
        out,
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="wl_driver",
        description=(
            "Emit the WL subtree kernel distance matrix for one dataset. "
            "h is frozen at the backend's WL_ROUNDS and is deliberately not "
            "a command-line argument."
        ),
    )
    parser.add_argument("--cohort", required=True, type=Path, help="exported CSR {dataset}.npz")
    parser.add_argument(
        "--reference-encodings",
        required=True,
        type=Path,
        help="a CONTRACTS §3 file for the same dataset; verifies the graph_ids join",
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
        out = run(args.cohort, args.reference_encodings, args.out, args.suite)
    except SchemaError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    print(json.dumps({"wl_subtree": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
