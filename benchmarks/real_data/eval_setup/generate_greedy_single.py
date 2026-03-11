# ruff: noqa: E402
"""Generate Greedy-rnd(v₀) data: greedy encoding from a single random starting node.

For each graph in each dataset, picks a uniformly random starting node
(seeded for reproducibility) and runs the greedy G2S algorithm once.
Saves canonical strings JSON and Levenshtein matrices NPZ in the same
format as the existing greedy/exhaustive data.

Usage:
    python -m benchmarks.eval_setup.generate_greedy_single \
        --source-dir /media/mpascual/Sandisk2TB/research/isalgraph/data/source \
        --data-root /media/mpascual/Sandisk2TB/research/isalgraph/data/eval \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

ALL_DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
IAM_DATASETS = {"iam_letter_low": "LOW", "iam_letter_med": "MED", "iam_letter_high": "HIGH"}
GRAPHEDX_DATASETS = {"linux": "LINUX", "aids": "AIDS"}


def _load_dataset(dataset_name: str, source_dir: str):
    """Load a dataset and return (graphs, graph_ids)."""

    from benchmarks.eval_setup.graphedx_loader import load_graphedx_dataset
    from benchmarks.eval_setup.iam_letter_loader import load_iam_letter

    if dataset_name in IAM_DATASETS:
        level = IAM_DATASETS[dataset_name]
        letter_dir = os.path.join(source_dir, "Letter")
        ds = load_iam_letter(letter_dir, level)
        return ds.graphs, ds.graph_ids

    if dataset_name in GRAPHEDX_DATASETS:
        gx_name = GRAPHEDX_DATASETS[dataset_name]
        result = load_graphedx_dataset(gx_name, source_dir)
        return result.graphs, result.graph_ids

    raise ValueError(f"Unknown dataset: {dataset_name}")


def _filter_by_existing_ids(
    graphs: list,
    graph_ids: list[str],
    data_root: str,
    dataset_name: str,
) -> tuple[list, list[str]]:
    """Filter graphs to only those present in the existing greedy data."""
    existing_path = os.path.join(data_root, "canonical_strings", f"{dataset_name}_greedy.json")
    if not os.path.isfile(existing_path):
        logger.warning("No existing greedy data for %s, using all graphs", dataset_name)
        return graphs, graph_ids

    with open(existing_path, encoding="utf-8") as f:
        existing_data = json.load(f)

    existing_ids = set(existing_data["strings"].keys())
    filtered_graphs = []
    filtered_ids = []
    for g, gid in zip(graphs, graph_ids, strict=True):
        if gid in existing_ids:
            filtered_graphs.append(g)
            filtered_ids.append(gid)

    logger.info(
        "Filtered %s: %d -> %d graphs (matching existing greedy data)",
        dataset_name,
        len(graphs),
        len(filtered_ids),
    )
    return filtered_graphs, filtered_ids


def _encode_single_random(
    graphs: list,
    graph_ids: list[str],
    seed: int,
) -> tuple[list[str | None], list[float]]:
    """Run greedy G2S from a single random starting node per graph.

    Delegates to greedy_single_computer.compute_greedy_single.
    """
    from benchmarks.eval_setup.greedy_single_computer import compute_greedy_single

    return compute_greedy_single(graphs, graph_ids, seed)


def _save_strings_json(
    graph_ids: list[str],
    strings: list[str | None],
    times: list[float],
    dataset_name: str,
    output_path: str,
) -> None:
    """Save greedy-single strings. Delegates to greedy_single_computer."""
    from benchmarks.eval_setup.greedy_single_computer import save_greedy_single_strings

    save_greedy_single_strings(graph_ids, strings, times, dataset_name, 12, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Greedy-rnd(v₀) data.")
    parser.add_argument(
        "--source-dir",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/data/source",
        help="Source data directory.",
    )
    parser.add_argument(
        "--data-root",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/data/eval",
        help="Eval data root (for existing greedy data and output).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=ALL_DATASETS,
        help="Datasets to process.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from benchmarks.eval_setup.levenshtein_computer import (
        compute_levenshtein_matrix,
        save_levenshtein_matrix,
    )

    for dataset_name in args.datasets:
        logger.info("=== Processing %s ===", dataset_name)

        # Load source graphs
        graphs, graph_ids = _load_dataset(dataset_name, args.source_dir)

        # Filter to only graphs in existing eval data
        graphs, graph_ids = _filter_by_existing_ids(graphs, graph_ids, args.data_root, dataset_name)

        # Encode from random v₀
        strings, times = _encode_single_random(graphs, graph_ids, args.seed)

        # Save strings JSON
        strings_path = os.path.join(
            args.data_root, "canonical_strings", f"{dataset_name}_greedy_single.json"
        )
        _save_strings_json(graph_ids, strings, times, dataset_name, strings_path)

        # Compute and save Levenshtein matrix
        lev_matrix = compute_levenshtein_matrix(
            strings, graph_ids, "greedy_single", use_c_extension=True
        )
        lev_path = os.path.join(
            args.data_root, "levenshtein_matrices", f"{dataset_name}_greedy_single.npz"
        )
        save_levenshtein_matrix(lev_matrix, graph_ids, "greedy_single", lev_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
