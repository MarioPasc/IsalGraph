"""Main CLI orchestrator for the evaluation infrastructure setup.

Chains dataset loading, filtering, GED computation, dual canonical
string computation, Levenshtein matrix computation, method comparison,
and validation into a single pipeline.

Usage:
    python -m benchmarks.eval_setup.eval_setup \
        --data-root data/eval \
        --source-dir /path/to/source \
        --n-max 12 --n-workers 4 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import networkx as nx
import numpy as np

from benchmarks.eval_setup.canonical_computer import (
    CanonicalResult,
    compute_all_canonical,
    save_canonical_strings,
)
from benchmarks.eval_setup.dataset_filter import (
    FilterResult,
    build_filtering_report,
    extract_ged_submatrix,
    filter_graphs,
)
from benchmarks.eval_setup.ged_computer import (
    compute_all_pairs_ged,
    save_ged_matrix,
)
from benchmarks.eval_setup.graphedx_loader import load_graphedx_dataset
from benchmarks.eval_setup.iam_letter_loader import load_iam_letter
from benchmarks.eval_setup.levenshtein_computer import (
    compute_levenshtein_matrix,
    cross_validate_levenshtein,
    save_levenshtein_matrix,
)
from benchmarks.eval_setup.method_comparator import (
    compare_methods,
    save_method_comparison,
)
from benchmarks.eval_setup.validator import (
    save_validation_report,
    validate_all,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "data/eval"
DEFAULT_SOURCE_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph/data/source"
DEFAULT_N_MAX = 12
DEFAULT_SEED = 42
DEFAULT_TIMEOUT = 600

ALL_DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
IAM_DATASETS = {"iam_letter_low": "LOW", "iam_letter_med": "MED", "iam_letter_high": "HIGH"}
GRAPHEDX_DATASETS = {"linux": "LINUX", "aids": "AIDS"}


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def _load_dataset(
    dataset_name: str,
    source_dir: str,
) -> tuple[list[nx.Graph], list[str], list[str], np.ndarray | None]:
    """Load a dataset and return (graphs, graph_ids, labels, ged_matrix_or_None).

    Args:
        dataset_name: One of ALL_DATASETS.
        source_dir: Path to the source data directory.

    Returns:
        (graphs, graph_ids, labels, ged_matrix) tuple.
        ged_matrix is None for IAM Letter (computed later).
    """
    if dataset_name in IAM_DATASETS:
        level = IAM_DATASETS[dataset_name]
        letter_dir = os.path.join(source_dir, "Letter")
        ds = load_iam_letter(letter_dir, level)
        return ds.graphs, ds.graph_ids, ds.labels, None

    if dataset_name in GRAPHEDX_DATASETS:
        gx_name = GRAPHEDX_DATASETS[dataset_name]
        result = load_graphedx_dataset(gx_name, source_dir)
        labels = [""] * len(result.graphs)
        return result.graphs, result.graph_ids, labels, result.ged_matrix

    raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------


def _process_dataset(
    dataset_name: str,
    graphs: list[nx.Graph],
    graph_ids: list[str],
    labels: list[str],
    raw_ged_matrix: np.ndarray | None,
    data_root: str,
    n_max: int,
    n_workers: int,
    seed: int,
    timeout_per_graph: float,
    skip_ged: bool,
    skip_canonical: bool,
    skip_levenshtein: bool,
) -> FilterResult:
    """Run the full pipeline for one dataset.

    Returns:
        FilterResult for the dataset.
    """
    logger.info("=" * 60)
    logger.info("Processing dataset: %s (%d raw graphs)", dataset_name, len(graphs))
    logger.info("=" * 60)

    # ---- Step 0: Filter ----
    filter_result = filter_graphs(graphs, graph_ids, n_max)
    kept_graphs = [graphs[i] for i in filter_result.kept_indices]
    kept_ids = [graph_ids[i] for i in filter_result.kept_indices]
    kept_labels = [labels[i] for i in filter_result.kept_indices]
    n_kept = len(kept_graphs)

    node_counts = [g.number_of_nodes() for g in kept_graphs]
    edge_counts = [g.number_of_edges() for g in kept_graphs]

    # Save graph metadata
    metadata_dir = os.path.join(data_root, "graph_metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, f"{dataset_name}.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "n_graphs": n_kept,
                "graph_ids": kept_ids,
                "labels": kept_labels,
                "node_counts": node_counts,
                "edge_counts": edge_counts,
            },
            f,
            indent=2,
        )

    if n_kept == 0:
        logger.warning("No graphs remaining after filtering for %s", dataset_name)
        return filter_result

    # ---- Step 1: GED ----
    ged_dir = os.path.join(data_root, "ged_matrices")
    ged_path = os.path.join(ged_dir, f"{dataset_name}.npz")

    if not skip_ged:
        if dataset_name in IAM_DATASETS:
            checkpoint = os.path.join(ged_dir, f"{dataset_name}_checkpoint.npz")
            ged_matrix = compute_all_pairs_ged(
                kept_graphs,
                kept_ids,
                n_workers=n_workers,
                checkpoint_path=checkpoint,
                timeout_per_pair=60.0,
            )
        elif dataset_name in GRAPHEDX_DATASETS:
            if raw_ged_matrix is None:
                raise ValueError(f"No precomputed GED matrix for {dataset_name}")
            ged_matrix = extract_ged_submatrix(raw_ged_matrix, filter_result.kept_indices)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        ged_source = "networkx" if dataset_name in IAM_DATASETS else "graphedx_jain2024"
        ged_method = "exact_a_star" if dataset_name in IAM_DATASETS else "precomputed_bai2019"

        save_ged_matrix(
            ged_matrix=ged_matrix,
            graph_ids=kept_ids,
            labels=kept_labels,
            node_counts=node_counts,
            edge_counts=edge_counts,
            output_path=ged_path,
            dataset_name=dataset_name,
            ged_method=ged_method,
            ged_cost_function="uniform_topology_only",
            source=ged_source,
            n_max_filter=n_max,
            n_dropped=filter_result.n_raw - filter_result.n_kept,
        )

    # ---- Step 2: Dual canonical strings ----
    canonical_dir = os.path.join(data_root, "canonical_strings")
    os.makedirs(canonical_dir, exist_ok=True)

    canonical_results: list[CanonicalResult] = []
    if not skip_canonical:
        checkpoint = os.path.join(canonical_dir, f"{dataset_name}_checkpoint.json")
        canonical_results = compute_all_canonical(
            kept_graphs,
            kept_ids,
            n_workers=n_workers,
            timeout_s=timeout_per_graph,
            checkpoint_path=checkpoint,
        )

        for method in ["exhaustive", "greedy"]:
            out_path = os.path.join(canonical_dir, f"{dataset_name}_{method}.json")
            save_canonical_strings(canonical_results, method, dataset_name, n_max, out_path)

    # ---- Step 3: Levenshtein matrices ----
    lev_dir = os.path.join(data_root, "levenshtein_matrices")
    os.makedirs(lev_dir, exist_ok=True)

    if not skip_levenshtein and canonical_results:
        # Cross-validate Levenshtein implementation
        exhaustive_strings = [r.exhaustive_string for r in canonical_results]
        greedy_strings = [r.greedy_string for r in canonical_results]

        valid_exhaust = [s for s in exhaustive_strings if s is not None]
        if valid_exhaust:
            cv_result = cross_validate_levenshtein(valid_exhaust)
            logger.info("Levenshtein cross-validation: %s", cv_result.get("status", "unknown"))

        for method, strings in [("exhaustive", exhaustive_strings), ("greedy", greedy_strings)]:
            lev_matrix = compute_levenshtein_matrix(strings, kept_ids, method)
            lev_path = os.path.join(lev_dir, f"{dataset_name}_{method}.npz")
            save_levenshtein_matrix(lev_matrix, kept_ids, method, lev_path)

    # ---- Step 4: Method comparison ----
    comparison_dir = os.path.join(data_root, "method_comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    if canonical_results and not skip_levenshtein and not skip_ged:
        # Load the matrices we just saved
        lev_exhaust_path = os.path.join(lev_dir, f"{dataset_name}_exhaustive.npz")
        lev_greedy_path = os.path.join(lev_dir, f"{dataset_name}_greedy.npz")

        if (
            os.path.exists(lev_exhaust_path)
            and os.path.exists(lev_greedy_path)
            and os.path.exists(ged_path)
        ):
            lev_exhaust = np.load(lev_exhaust_path)["levenshtein_matrix"]
            lev_greedy = np.load(lev_greedy_path)["levenshtein_matrix"]
            ged_mat = np.load(ged_path, allow_pickle=True)["ged_matrix"]

            comparison = compare_methods(
                canonical_results, lev_exhaust, lev_greedy, ged_mat, kept_ids, dataset_name
            )
            comp_path = os.path.join(comparison_dir, f"{dataset_name}_comparison.json")
            save_method_comparison(comparison, comp_path)

    return filter_result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    data_root: str,
    source_dir: str,
    datasets: list[str],
    n_max: int,
    n_workers: int,
    seed: int,
    timeout_per_graph: float,
    skip_ged: bool,
    skip_canonical: bool,
    skip_levenshtein: bool,
    validate_only: bool,
    max_graphs_per_dataset: int | None = None,
) -> None:
    """Run the full evaluation setup pipeline.

    Args:
        data_root: Root output directory for evaluation artifacts.
        source_dir: Path to source data (Letter/, LINUX/, AIDS/).
        datasets: List of dataset names to process.
        n_max: Maximum node count filter.
        n_workers: Number of parallel workers.
        seed: Random seed.
        timeout_per_graph: Per-graph canonical timeout.
        skip_ged: Skip GED computation.
        skip_canonical: Skip canonical computation.
        skip_levenshtein: Skip Levenshtein computation.
        validate_only: Only run validation.
        max_graphs_per_dataset: Limit graphs per dataset (for testing).
    """
    os.makedirs(data_root, exist_ok=True)
    np.random.seed(seed)

    logger.info("Evaluation Setup Pipeline")
    logger.info("  data_root: %s", data_root)
    logger.info("  source_dir: %s", source_dir)
    logger.info("  datasets: %s", datasets)
    logger.info("  n_max: %d", n_max)
    logger.info("  n_workers: %d", n_workers)
    logger.info("  seed: %d", seed)
    logger.info("  timeout_per_graph: %.0fs", timeout_per_graph)
    if max_graphs_per_dataset:
        logger.info("  max_graphs_per_dataset: %d (TESTING MODE)", max_graphs_per_dataset)

    if validate_only:
        logger.info("Running validation only...")
        checks = validate_all(data_root)
        report_path = os.path.join(data_root, "validation_report.json")
        save_validation_report(checks, report_path)
        n_failed = sum(1 for c in checks if not c.passed and c.severity == "error")
        if n_failed > 0:
            logger.error("%d validation checks FAILED", n_failed)
            sys.exit(1)
        return

    # Process each dataset
    t0 = time.perf_counter()
    filter_results: dict[str, FilterResult] = {}

    for dataset_name in datasets:
        try:
            graphs, graph_ids, labels, raw_ged = _load_dataset(dataset_name, source_dir)

            # Limit for testing
            if max_graphs_per_dataset and len(graphs) > max_graphs_per_dataset:
                graphs = graphs[:max_graphs_per_dataset]
                graph_ids = graph_ids[:max_graphs_per_dataset]
                labels = labels[:max_graphs_per_dataset]
                # Truncate GED matrix to match
                if raw_ged is not None:
                    raw_ged = raw_ged[:max_graphs_per_dataset, :max_graphs_per_dataset]

            fr = _process_dataset(
                dataset_name=dataset_name,
                graphs=graphs,
                graph_ids=graph_ids,
                labels=labels,
                raw_ged_matrix=raw_ged,
                data_root=data_root,
                n_max=n_max,
                n_workers=n_workers,
                seed=seed,
                timeout_per_graph=timeout_per_graph,
                skip_ged=skip_ged,
                skip_canonical=skip_canonical,
                skip_levenshtein=skip_levenshtein,
            )
            filter_results[dataset_name] = fr

        except Exception:
            logger.exception("Failed to process dataset: %s", dataset_name)
            continue

    # ---- Save filtering report ----
    if filter_results:
        report = build_filtering_report(filter_results, n_max)
        report_path = os.path.join(data_root, "filtering_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved filtering report to %s", report_path)

    # ---- Validation ----
    logger.info("Running validation...")
    checks = validate_all(data_root)
    report_path = os.path.join(data_root, "validation_report.json")
    save_validation_report(checks, report_path)

    elapsed = time.perf_counter() - t0
    n_failed = sum(1 for c in checks if not c.passed and c.severity == "error")
    logger.info("Pipeline complete in %.1fs. Validation: %d failures.", elapsed, n_failed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluation infrastructure setup (dual-mode canonical strings)."
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-dir", type=str, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--n-max", type=int, default=DEFAULT_N_MAX)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--mode",
        choices=["local", "picasso"],
        default="local",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset names, or 'all'.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-ged", action="store_true")
    parser.add_argument("--skip-canonical", action="store_true")
    parser.add_argument("--skip-levenshtein", action="store_true")
    parser.add_argument("--timeout-per-graph", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Limit graphs per dataset (for smoke testing).",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse datasets
    if args.datasets == "all":
        datasets = list(ALL_DATASETS)
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]
        for d in datasets:
            if d not in ALL_DATASETS:
                logger.error("Unknown dataset: %s. Choose from %s", d, ALL_DATASETS)
                sys.exit(1)

    # Picasso defaults
    if args.mode == "picasso" and args.n_workers == 1:
        args.n_workers = 64

    run_pipeline(
        data_root=args.data_root,
        source_dir=args.source_dir,
        datasets=datasets,
        n_max=args.n_max,
        n_workers=args.n_workers,
        seed=args.seed,
        timeout_per_graph=args.timeout_per_graph,
        skip_ged=args.skip_ged,
        skip_canonical=args.skip_canonical,
        skip_levenshtein=args.skip_levenshtein,
        validate_only=args.validate_only,
        max_graphs_per_dataset=args.max_graphs,
    )


if __name__ == "__main__":
    main()
