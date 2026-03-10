# ruff: noqa: N803
"""Main CLI for generating all embedding figures (H2.1, H2.2, H2.3).

Generates population figures, individual examples, and tables for
MDS embedding quality, Procrustes alignment, and Shepard fidelity.
"""

from __future__ import annotations

import argparse
import logging
import os

from benchmarks.eval_visualizations.embedding_loader import load_embedding_data
from benchmarks.eval_visualizations.individual_figures.embedding_examples import (
    generate_all_embedding_individual,
)
from benchmarks.eval_visualizations.population_figures.embedding_figures import (
    generate_all_embedding_population,
)
from benchmarks.eval_visualizations.result_loader import load_all_results
from benchmarks.plotting_styles import apply_ieee_style

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for embedding figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate all embedding figures for H2.1-H2.3.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory for eval data (ged_matrices/, levenshtein_matrices/, etc.).",
    )
    parser.add_argument(
        "--embedding-dir",
        required=True,
        help="Root of eval_embedding output (contains stats/, raw/).",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/embedding",
        help="Output directory.",
    )
    parser.add_argument(
        "--population",
        action="store_true",
        help="Generate population figures and tables.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Generate individual example figures.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate everything (population + individual).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()

    os.makedirs(args.output_dir, exist_ok=True)

    do_pop = args.population or args.all
    do_ind = args.individual or args.all

    if not do_pop and not do_ind:
        logger.warning("No output requested. Use --population, --individual, or --all.")
        return

    # Load data
    results = load_all_results(args.data_root)
    emb = load_embedding_data(args.embedding_dir)

    if do_pop:
        logger.info("Generating embedding population figures and tables...")
        generate_all_embedding_population(emb, args.output_dir)

    if do_ind:
        logger.info("Generating embedding individual example figures...")
        generate_all_embedding_individual(results, emb, args.embedding_dir, args.output_dir)

    logger.info("All embedding figures saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
