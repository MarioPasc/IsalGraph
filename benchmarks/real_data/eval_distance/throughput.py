"""Sizing harness for the distance driver.  **Not a published timing.**

Three agents share this workstation, so every concurrent measurement is
contaminated.  What this module produces is an order-of-magnitude input to the
orchestrator's shard-count decision, and it is labelled as such wherever it is
quoted.  It is a module rather than a scratch script so the number can be
reproduced instead of remembered.

It times the three implementations that were candidates for the inner loop, on
a real encodings file, and reports the string-length distribution alongside --
throughput on a symbol sequence is meaningless without it, since Levenshtein
is ``O(L_a L_b / 64)`` in the bit-parallel implementation rapidfuzz uses.

Usage::

    python -m benchmarks.eval_distance.throughput \\
        --encodings encodings/suite2/linux__isalgraph_pruned.npz [--jobs 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from benchmarks.eval_distance.bands import RowBand
from benchmarks.eval_distance.distance_runner import (
    rebuild_encodings,
    resolve_symbol_separator,
)
from benchmarks.eval_distance.schema import load_encodings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Timing:
    """One implementation's measurement.

    Attributes:
        implementation: what was timed.
        seconds: wall clock for the whole cell block.
        cells: ordered ``(i, j)`` cells evaluated, including the diagonal.
        cells_per_second: *cells* over *seconds*.
    """

    implementation: str
    seconds: float
    cells: int
    cells_per_second: float


def _time(name: str, cells: int, work: Callable[[], object]) -> Timing:
    """Run *work* once and report its throughput."""
    start = time.perf_counter()
    work()
    elapsed = time.perf_counter() - start
    return Timing(
        implementation=name,
        seconds=elapsed,
        cells=cells,
        cells_per_second=cells / elapsed if elapsed > 0 else float("inf"),
    )


def _cdist_work(symbols: Sequence[object], jobs: int) -> Callable[[], object]:
    """The chosen implementation: one C call over the whole block."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    def run() -> object:
        return process.cdist(
            symbols, symbols, scorer=Levenshtein.distance, dtype=np.int64, workers=jobs
        )

    return run


def _metric_loop_work(encodings: Sequence[object]) -> Callable[[], object]:
    """Rejected: a Python double loop through the metric protocol."""
    from isalgraph.competitors import get_metric

    metric = get_metric("levenshtein")

    def run() -> object:
        n = len(encodings)
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += metric.distance(encodings[i], encodings[j])
        return total

    return run


def _core_engine_work(texts: Sequence[str]) -> Callable[[], object]:
    """Rejected: the C++ engine's single-pair ``levenshtein``."""
    from isalgraph.core.backends import levenshtein

    def run() -> object:
        n = len(texts)
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += levenshtein(texts[i], texts[j])
        return total

    return run


def measure(encodings_path: Path, jobs: int = 1, limit: int = 200) -> dict[str, object]:
    """Time the three candidate implementations on *encodings_path*.

    Args:
        encodings_path: a CONTRACTS §3 file.
        jobs: threads for the cdist path.
        limit: graphs to take; keep it small, this is not a campaign.

    Returns:
        A JSON-serialisable record: the length distribution, the three
        timings, and the engine provenance.
    """
    source = load_encodings(encodings_path)
    representation = str(source.metadata.get("representation") or "unknown")
    separator = resolve_symbol_separator(representation, source.metadata, None)
    rebuilt = rebuild_encodings(source, representation, separator)
    take = min(limit, source.n_graphs)
    band = RowBand(index=0, start=0, stop=take)
    symbols = rebuilt.symbols[: band.stop]
    texts = rebuilt.texts[: band.stop]
    objects = rebuilt.encodings[: band.stop]
    lengths = rebuilt.lengths[: band.stop]
    ordered = take * take
    unordered = take * (take - 1) // 2
    timings = [
        _time("rapidfuzz.process.cdist", ordered, _cdist_work(symbols, jobs)),
        _time("python loop through DistanceMetric", unordered, _metric_loop_work(objects)),
        _time("isalgraph.core.backends.levenshtein", unordered, _core_engine_work(texts)),
    ]
    return {
        "encodings": str(encodings_path),
        "dataset": source.metadata.get("dataset"),
        "representation": representation,
        "n_graphs_timed": take,
        "symbol_length_min": int(lengths.min()) if take else 0,
        "symbol_length_median": float(np.median(lengths)) if take else 0.0,
        "symbol_length_max": int(lengths.max()) if take else 0,
        "jobs": jobs,
        "isalgraph_engine": source.metadata.get("isalgraph_engine"),
        "timings": [asdict(timing) for timing in timings],
        "caveat": (
            "sizing input for the orchestrator only; three agents share this workstation "
            "and every concurrent measurement is contaminated"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="throughput",
        description="Time the candidate distance inner loops. Sizing only, never a result.",
    )
    parser.add_argument("--encodings", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 always; a failure surfaces as an exception.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(measure(args.encodings, args.jobs, args.limit), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
