"""Generate all paper figures from a completed pipeline run.

Step 4 orchestrator: reads config.yaml from the run directory, locates all
intermediate outputs, and calls the existing visualization functions to produce
the final paper figures in ``<run_dir>/figures/``.

Usage:
    python experiments/paper_pipeline/generate_figures.py --run-dir runs/<run_id>
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(run_dir: str) -> dict:
    """Load the frozen config.yaml from a run directory."""
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def copy_figures(
    src_dir: str,
    dst_dir: str,
    basenames: list[str],
    extensions: tuple[str, ...] = (".pdf", ".png"),
) -> list[str]:
    """Copy figures from intermediate directory to final figures directory.

    Searches for each basename with any of the given extensions.
    Returns list of copied destination paths.
    """
    copied = []
    for name in basenames:
        found = False
        for ext in extensions:
            src = os.path.join(src_dir, f"{name}{ext}")
            if os.path.isfile(src):
                dst = os.path.join(dst_dir, f"{name}{ext}")
                shutil.copy2(src, dst)
                copied.append(dst)
                found = True
                logger.info("  Copied: %s -> %s", src, dst)
        if not found:
            # Try glob for any file matching the basename
            matches = glob.glob(os.path.join(src_dir, f"{name}.*"))
            if matches:
                for src in matches:
                    ext = os.path.splitext(src)[1]
                    if ext in extensions:
                        dst = os.path.join(dst_dir, os.path.basename(src))
                        shutil.copy2(src, dst)
                        copied.append(dst)
                        logger.info("  Copied: %s -> %s", src, dst)
                        found = True
            if not found:
                logger.warning("  NOT FOUND: %s in %s", name, src_dir)
    return copied


def check_dir_exists(path: str, label: str) -> bool:
    """Check that a directory exists and log a warning if not."""
    if not os.path.isdir(path):
        logger.warning("Missing %s directory: %s (skipping dependent figures)", label, path)
        return False
    return True


def report_generated_files(figures_dir: str) -> None:
    """List all files in the figures directory with sizes."""
    logger.info("=" * 60)
    logger.info("PAPER OUTPUTS:")
    logger.info("=" * 60)

    all_files = []
    for root, _dirs, files in os.walk(figures_dir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, figures_dir)
            size_kb = os.path.getsize(fpath) / 1024
            all_files.append((rel, size_kb))

    # Show top-level files first (the paper outputs)
    top_level = [(r, s) for r, s in all_files if os.sep not in r]
    intermediate = [(r, s) for r, s in all_files if os.sep in r]

    if top_level:
        logger.info("Final figures:")
        for rel, size_kb in sorted(top_level):
            logger.info("  %-50s  %7.1f KB", rel, size_kb)

    if intermediate:
        logger.info("Intermediate files: %d", len(intermediate))

    logger.info("Total: %d files", len(all_files))
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------


def gen_composite_method_tradeoff(
    data_root: str,
    corr_stats_dir: str,
    comp_dir: str,
    figures_dir: str,
) -> None:
    """Generate composite_method_tradeoff_v2.pdf."""
    logger.info("[1/8] Generating composite_method_tradeoff_v2...")

    if not check_dir_exists(data_root, "data_root"):
        return
    if not check_dir_exists(corr_stats_dir, "correlation/stats"):
        return
    if not check_dir_exists(comp_dir, "computational"):
        return

    from benchmarks.eval_visualizations.composite_method_tradeoff import (
        generate_composite_method_tradeoff_v2,
    )

    path = generate_composite_method_tradeoff_v2(
        data_root,
        corr_stats_dir,
        comp_dir,
        figures_dir,
    )
    logger.info("  -> %s", path)


def gen_aggregated_density_heatmap(
    data_root: str,
    corr_stats_dir: str,
    figures_dir: str,
) -> None:
    """Generate fig_aggregated_density_correlation.pdf."""
    logger.info("[2/8] Generating fig_aggregated_density_correlation...")

    if not check_dir_exists(data_root, "data_root"):
        return

    from benchmarks.eval_visualizations.population_figures.central_heatmap import (
        generate_aggregated_density_heatmap,
    )
    from benchmarks.eval_visualizations.result_loader import load_all_results

    correlation_dir = corr_stats_dir if os.path.isdir(corr_stats_dir) else None
    results = load_all_results(data_root, correlation_dir)
    path = generate_aggregated_density_heatmap(results, figures_dir)
    logger.info("  -> %s", path)


def gen_algorithm_overview(algo_dir: str, figures_dir: str) -> None:
    """Copy algorithm overview figures from Step 3a output."""
    logger.info("[3/8] Copying algorithm overview figures...")

    if not check_dir_exists(algo_dir, "algorithm intermediate"):
        return

    copy_figures(
        algo_dir,
        figures_dir,
        ["fig_algorithm_overview", "fig_algorithm_overview_full"],
    )


def gen_empirical_complexity(encoding_raw_dir: str, figures_dir: str) -> None:
    """Generate fig_empirical_complexity.pdf."""
    logger.info("[5/8] Generating fig_empirical_complexity...")

    if not check_dir_exists(encoding_raw_dir, "encoding/raw"):
        return

    from benchmarks.eval_visualizations.fig_empirical_complexity import (
        generate_empirical_complexity,
    )

    path = generate_empirical_complexity(encoding_raw_dir, figures_dir)
    logger.info("  -> %s", path)


def gen_neighborhood_topology(topo_dir: str, figures_dir: str) -> None:
    """Copy neighbourhood topology figure from Step 3b output."""
    logger.info("[6/8] Copying neighbourhood topology figure...")

    if not check_dir_exists(topo_dir, "topology intermediate"):
        return

    copy_figures(topo_dir, figures_dir, ["fig_neighborhood_topology"])


def gen_shortest_path_comparison(figures_dir: str) -> None:
    """Generate fig_shortest_path_comparison.pdf (standalone, no precomputed data)."""
    logger.info("[4/8] Generating fig_shortest_path_comparison...")

    from benchmarks.eval_visualizations.illustrative.shortest_path_comparison import (
        generate_shortest_path_comparison,
    )

    path = generate_shortest_path_comparison(figures_dir)
    logger.info("  -> %s", path)


def gen_message_length_figure(
    msg_raw_dir: str,
    msg_stats_dir: str,
    figures_dir: str,
) -> None:
    """Generate fig_message_length_scatter.pdf and table_message_length_summary.tex."""
    logger.info("[8/9] Generating message length figures...")

    if not check_dir_exists(msg_raw_dir, "message_length/raw"):
        return

    from benchmarks.eval_visualizations.fig_message_length import (
        generate_message_length_table,
        generate_ratio_figure,
        generate_scatter_figure,
    )

    generate_scatter_figure(msg_raw_dir, figures_dir)
    generate_ratio_figure(msg_raw_dir, figures_dir)
    if os.path.isdir(msg_stats_dir):
        generate_message_length_table(msg_stats_dir, figures_dir)


def gen_performance_table(
    data_root: str,
    corr_stats_dir: str,
    figures_dir: str,
) -> None:
    """Generate table_performance_summary.tex."""
    logger.info("[9/9] Generating table_performance_summary...")

    if not check_dir_exists(data_root, "data_root"):
        return
    if not check_dir_exists(corr_stats_dir, "correlation/stats"):
        return

    from benchmarks.eval_visualizations.table_performance_summary import (
        generate_performance_table,
    )

    path = generate_performance_table(data_root, corr_stats_dir, figures_dir)
    logger.info("  -> %s", path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def generate_all(run_dir: str) -> None:
    """Generate all paper figures from a completed pipeline run."""
    config = load_config(run_dir)
    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Apply IEEE style before any matplotlib imports
    from benchmarks.plotting_styles import apply_ieee_style

    apply_ieee_style()

    # Derived paths
    data_root = os.path.join(run_dir, "data")
    corr_stats_dir = os.path.join(run_dir, "correlation", "stats")
    comp_dir = os.path.join(run_dir, "computational")
    encoding_raw_dir = os.path.join(run_dir, "encoding", "raw")
    msg_raw_dir = os.path.join(run_dir, "message_length", "raw")
    msg_stats_dir = os.path.join(run_dir, "message_length", "stats")
    algo_dir = os.path.join(run_dir, "figures", "_intermediate", "algorithm")
    topo_dir = os.path.join(run_dir, "figures", "_intermediate", "topology")

    steps = config.get("steps", {})
    errors: list[str] = []

    # 1. composite_method_tradeoff_v2
    if steps.get("eval_correlation", {}).get("enabled") and steps.get("eval_computational", {}).get(
        "enabled"
    ):
        try:
            gen_composite_method_tradeoff(data_root, corr_stats_dir, comp_dir, figures_dir)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            errors.append(f"composite_method_tradeoff_v2: {e}")

    # 2. fig_aggregated_density_correlation
    if steps.get("eval_correlation", {}).get("enabled"):
        try:
            gen_aggregated_density_heatmap(data_root, corr_stats_dir, figures_dir)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            errors.append(f"fig_aggregated_density_correlation: {e}")

    # 3 & 4. Algorithm overview (from Step 3a)
    if steps.get("algorithm_figures", {}).get("enabled"):
        try:
            gen_algorithm_overview(algo_dir, figures_dir)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            errors.append(f"algorithm_figures: {e}")

    # 4b. fig_shortest_path_comparison (standalone, no deps)
    try:
        gen_shortest_path_comparison(figures_dir)
    except Exception as e:
        logger.error("  FAILED: %s", e)
        errors.append(f"fig_shortest_path_comparison: {e}")

    # 5. fig_empirical_complexity
    if steps.get("eval_encoding", {}).get("enabled"):
        try:
            gen_empirical_complexity(encoding_raw_dir, figures_dir)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            errors.append(f"fig_empirical_complexity: {e}")

    # 6. Message length figures (from Step 2d)
    if steps.get("eval_message_length", {}).get("enabled"):
        try:
            gen_message_length_figure(msg_raw_dir, msg_stats_dir, figures_dir)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            errors.append(f"fig_message_length: {e}")

    # 7. Neighbourhood topology (from Step 3b)
    if steps.get("topology_figs", {}).get("enabled"):
        try:
            gen_neighborhood_topology(topo_dir, figures_dir)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            errors.append(f"fig_neighborhood_topology: {e}")

    # 7. table_performance_summary
    if steps.get("eval_correlation", {}).get("enabled"):
        try:
            gen_performance_table(data_root, corr_stats_dir, figures_dir)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            errors.append(f"table_performance_summary: {e}")

    # Summary
    report_generated_files(figures_dir)

    if errors:
        logger.warning("Completed with %d error(s):", len(errors))
        for err in errors:
            logger.warning("  - %s", err)
        sys.exit(1)
    else:
        logger.info("All paper outputs generated successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all paper figures from a completed pipeline run.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run directory (contains config.yaml and intermediate data).",
    )
    args = parser.parse_args()
    generate_all(args.run_dir)


if __name__ == "__main__":
    main()
