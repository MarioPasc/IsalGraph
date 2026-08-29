"""CLI entry point for building all T-28 reference distance matrices.

For each of the 15 (suite, dataset) cells, builds five reference NPZ files:
``spectral``, ``spectral_comb``, ``spectral_adj``, ``spectral_esd``, and
``wl``.  Cells are processed serially, one at a time, with memory freed between
them.  Outputs go to::

    {out_root}/{suite}/{dataset}__{refkey}.npz

Usage::

    python benchmarks/real_data/eval_reference_metrics/build_references.py \\
        --archive-root /home/mpascual/research/data/isalgraph_archive \\
        --out-root /home/mpascual/research/data/isalgraph_archive/data/source/T28/references \\
        [--suite suite1|suite2|all] \\
        [--dataset DATASET] \\
        [--log-level DEBUG|INFO|WARNING]

All 15 cells take roughly 2-20 minutes total depending on hardware.  The
largest cells (mutagenicity, coil_del) are processed last so that partial
outputs from earlier cells are not blocked.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from isalgraph.competitors.references.build import REF_KEYS, build_cell

# ---------------------------------------------------------------------------
# Dataset inventory
# ---------------------------------------------------------------------------

SUITE1_DATASETS: tuple[str, ...] = (
    "aids",
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
)

SUITE2_DATASETS: tuple[str, ...] = (
    "aids_graphedx",
    "aids_iam",
    "coil_del",
    "grec",
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "mutagenicity",
    "protein",
)

ALL_CELLS: tuple[tuple[str, str], ...] = (
    tuple(("suite1", d) for d in SUITE1_DATASETS)
    + tuple(("suite2", d) for d in SUITE2_DATASETS)  # type: ignore[assignment]
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build T-28 reference distance matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/home/mpascual/research/data/isalgraph_archive"),
        help="Root of the IsalGraph archive.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/home/mpascual/research/data/isalgraph_archive/data/source/T28/references"
        ),
        help="Output root for reference NPZ files.",
    )
    parser.add_argument(
        "--suite",
        choices=["suite1", "suite2", "all"],
        default="all",
        help="Which suite(s) to process.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process only this dataset (within the selected suite).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _select_cells(
    suite: str,
    dataset: str | None,
) -> list[tuple[str, str]]:
    """Return the subset of cells to process."""
    if suite == "all":
        cells: list[tuple[str, str]] = list(ALL_CELLS)
    elif suite == "suite1":
        cells = [("suite1", d) for d in SUITE1_DATASETS]
    else:
        cells = [("suite2", d) for d in SUITE2_DATASETS]

    if dataset is not None:
        cells = [(s, d) for s, d in cells if d == dataset]
        if not cells:
            raise ValueError(
                f"dataset {dataset!r} not found in suite={suite!r}; "
                f"available suite1={SUITE1_DATASETS}, suite2={SUITE2_DATASETS}"
            )
    return cells


def _print_status_table(
    results: dict[tuple[str, str], dict[str, bool]],
) -> None:
    """Print a 15×5 build status table to stdout."""
    col_w = max(len(k) for k in REF_KEYS) + 2
    header = f"{'cell':<30}" + "".join(f"{k:>{col_w}}" for k in REF_KEYS)
    print(header)
    print("-" * len(header))
    all_ok = True
    for (suite, dataset), status in sorted(results.items()):
        row = f"{suite}/{dataset:<25}"
        for k in REF_KEYS:
            ok = status.get(k, False)
            if not ok:
                all_ok = False
            row += f"{'OK' if ok else 'FAIL':>{col_w}}"
        print(row)
    print()
    if all_ok:
        print("All cells succeeded.")
    else:
        print("SOME CELLS FAILED — check the log above.")


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        0 on full success, 1 if any cell/key failed.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    try:
        cells = _select_cells(args.suite, args.dataset)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Building %d cell(s) → %s", len(cells), args.out_root
    )

    results: dict[tuple[str, str], dict[str, bool]] = {}
    any_failure = False

    for suite, dataset in cells:
        logger.info("=== %s/%s ===", suite, dataset)
        status = build_cell(
            suite, dataset, args.archive_root, args.out_root
        )
        results[(suite, dataset)] = status
        failed = [k for k, ok in status.items() if not ok]
        if failed:
            any_failure = True
            logger.warning("%s/%s: FAILED keys: %s", suite, dataset, failed)
        else:
            logger.info("%s/%s: all 5 keys OK", suite, dataset)

    print()
    _print_status_table(results)

    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
