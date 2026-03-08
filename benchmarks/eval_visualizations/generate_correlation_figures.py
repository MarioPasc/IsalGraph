# ruff: noqa: N803
"""Main CLI for generating all correlation figures (5.3 + 5.4).

Generates H1.1-H1.5 population figures, individual examples, and tables.
"""

from __future__ import annotations

import argparse
import logging
import os

from benchmarks.eval_visualizations.individual_figures.correlation_examples import (
    generate_all_core_individual,
)
from benchmarks.eval_visualizations.individual_figures.derived_examples import (
    generate_all_derived_individual,
)
from benchmarks.eval_visualizations.population_figures.correlation_figures import (
    generate_all_core_population,
)
from benchmarks.eval_visualizations.population_figures.derived_figures import (
    generate_all_derived_population,
)
from benchmarks.eval_visualizations.result_loader import load_all_results
from benchmarks.plotting_styles import apply_ieee_style

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for correlation figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate all correlation figures for H1.1-H1.5.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory for eval data (ged_matrices/, levenshtein_matrices/, etc.).",
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root directory for source graph data (Letter/, LINUX/, AIDS/).",
    )
    parser.add_argument(
        "--stats-dir",
        required=True,
        help="Directory with correlation stats JSONs.",
    )
    parser.add_argument(
        "--method-comparison-dir",
        default=None,
        help="Directory with method comparison JSONs (for H1.5 individual).",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/correlation",
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

    # Determine method comparison dir
    mc_dir = args.method_comparison_dir
    if mc_dir is None:
        mc_dir = os.path.join(args.data_root, "method_comparison")

    do_pop = args.population or args.all
    do_ind = args.individual or args.all

    if not do_pop and not do_ind:
        logger.warning("No output requested. Use --population, --individual, or --all.")
        return

    # Load results
    results = load_all_results(args.data_root, args.stats_dir)

    if do_pop:
        logger.info("Generating population figures and tables...")
        generate_all_core_population(results, args.stats_dir, args.output_dir)
        generate_all_derived_population(results, args.stats_dir, args.output_dir)

    if do_ind:
        logger.info("Generating individual example figures...")
        generate_all_core_individual(results, args.source_root, args.output_dir)
        generate_all_derived_individual(results, args.source_root, mc_dir, args.output_dir)

    logger.info("All correlation figures saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
