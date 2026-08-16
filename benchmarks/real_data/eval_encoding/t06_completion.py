"""Per-``(representation, dataset)`` completion rates, the pre-registration's ``c`` input.

``CONTRACTS.md`` §2 fixes ``COMPUTABILITY_THRESHOLD = 0.99`` **per (representation,
dataset)**, never per representation: ``agm_cam`` completes on 100 % of the three
Letter datasets and LINUX while failing on most of GREC, AIDS-IAM, COIL-DEL,
Protein and Mutagenicity, so an all-or-nothing gate would delete tests that can
actually be run.

**This module reports the rate. It does not decide ``c``, and must not.** The
orchestrator applies the rule. What is emitted is the raw fraction with its
numerator and denominator, so the rule can be re-applied later without rerunning
anything.

Failures are split by family rather than pooled. A wall-clock failure (the 300 s
budget ran out) and an internal-cap failure (AGM's branch-and-bound ceiling,
min-DFS's projection ceiling, both frozen in T-04) are different facts about a
representation, and a rate that conflates them cannot be interpreted. A scope
refusal is a third thing again: ``agm_cam`` and ``isalgraph_canonical`` decline
above their declared node ceiling rather than producing a column conditioned on
the graphs that happened to finish.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.real_data.eval_encoding.t06_encode_worker import error_family

LOGGER = logging.getLogger(__name__)

#: ``CONTRACTS.md`` §2. Reported alongside the rate for convenience; the gate is
#: applied by the orchestrator, not here.
COMPUTABILITY_THRESHOLD = 0.99

FAMILIES: tuple[str, ...] = ("wall_clock", "internal_cap", "scope", "unavailable", "other")


@dataclass(frozen=True, slots=True)
class CompletionRow:
    """One ``(representation, dataset)`` completion measurement.

    Attributes:
        suite: ``"suite1"`` or ``"suite2"``.
        dataset: Dataset key.
        representation: Backend name.
        n_graphs: Denominator -- every graph in the cohort.
        n_completed: Numerator -- graphs that encoded within the budget under
            the backend's frozen configuration, i.e. ``status == "ok"``.
        rate: ``n_completed / n_graphs``.
        n_censored: Graphs retained under D14 with the greedy-min string. These
            are **not** completions; they are the reason the cohort is not
            thinned.
        n_wall_clock: Failures where the 300 s budget ran out.
        n_internal_cap: Failures at a backend's own frozen ceiling.
        n_scope: Refusals above a declared node ceiling.
        n_unavailable: Failures from a missing optional dependency.
        n_other: Everything else, which should be zero and is worth reading.
        error_kinds: Exception class name -> count, for diagnosis.
        meets_threshold: Whether ``rate >= 0.99``. Reported for convenience;
            the orchestrator, not this module, decides ``c``.
    """

    suite: str
    dataset: str
    representation: str
    n_graphs: int
    n_completed: int
    rate: float
    n_censored: int
    n_wall_clock: int
    n_internal_cap: int
    n_scope: int
    n_unavailable: int
    n_other: int
    error_kinds: dict[str, int]
    meets_threshold: bool


def _parse_name(path: Path) -> tuple[str, str]:
    """Split ``{dataset}__{representation}.npz`` into its two parts.

    Args:
        path: The encodings file.

    Returns:
        ``(dataset, representation)``.

    Raises:
        ValueError: If the basename does not carry the double underscore.
    """
    dataset, sep, representation = path.stem.partition("__")
    if not sep:
        raise ValueError(f"{path.name} is not a '{{dataset}}__{{representation}}.npz' file")
    return dataset, representation


def measure(path: Path) -> CompletionRow:
    """Compute the completion row for one encodings file.

    Args:
        path: An encodings ``.npz``.

    Returns:
        The row.
    """
    dataset, representation = _parse_name(path)
    with np.load(path, allow_pickle=False) as handle:
        status = handle["status"]
        error_kind = handle["error_kind"]
        suite = json.loads(str(handle["metadata"]))["suite"]

    families = Counter(
        error_family(str(kind))
        for kind, state in zip(error_kind, status, strict=True)
        if state == "error"
    )
    n_graphs = int(status.shape[0])
    n_completed = int((status == "ok").sum())
    return CompletionRow(
        suite=suite,
        dataset=dataset,
        representation=representation,
        n_graphs=n_graphs,
        n_completed=n_completed,
        rate=n_completed / n_graphs if n_graphs else 0.0,
        n_censored=int((status == "censored").sum()),
        n_wall_clock=families["wall_clock"],
        n_internal_cap=families["internal_cap"],
        n_scope=families["scope"],
        n_unavailable=families["unavailable"],
        n_other=families["other"],
        error_kinds=dict(Counter(str(kind) for kind in error_kind if str(kind)).most_common()),
        meets_threshold=(n_completed / n_graphs if n_graphs else 0.0) >= COMPUTABILITY_THRESHOLD,
    )


def collect(encodings_dir: Path) -> list[CompletionRow]:
    """Measure every encodings file under *encodings_dir*.

    Args:
        encodings_dir: The ``encodings/`` tree, or one suite inside it.

    Returns:
        Rows sorted by ``(suite, representation, dataset)``.
    """
    rows = [measure(path) for path in sorted(encodings_dir.rglob("*.npz"))]
    return sorted(rows, key=lambda row: (row.suite, row.representation, row.dataset))


def build_report(rows: Sequence[CompletionRow]) -> dict[str, Any]:
    """Assemble ``completion_rates.json``.

    Args:
        rows: The measured rows.

    Returns:
        The report payload.
    """
    return {
        "schema_version": "t06.1",
        "ticket": "T-06",
        "computability_threshold": COMPUTABILITY_THRESHOLD,
        "threshold_scope": "per (representation, dataset)",
        "decides_c": False,
        "note": (
            "This file reports the raw completion fraction with its numerator and "
            "denominator, and splits failures into wall-clock and internal-cap "
            "families. The orchestrator applies the c rule; this module does not."
        ),
        "rows": [asdict(row) for row in rows],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-(representation, dataset) completion rates.")
    parser.add_argument("--encodings", required=True, type=Path, help="the encodings/ tree")
    parser.add_argument("--out", required=True, type=Path, help="completion_rates.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    rows = collect(args.encodings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build_report(rows), indent=2, sort_keys=True))
    for row in rows:
        LOGGER.info(
            "%s/%-16s %-20s %5d/%-5d = %.4f  wall=%d cap=%d scope=%d censored=%d",
            row.suite,
            row.dataset,
            row.representation,
            row.n_completed,
            row.n_graphs,
            row.rate,
            row.n_wall_clock,
            row.n_internal_cap,
            row.n_scope,
            row.n_censored,
        )
    LOGGER.info("wrote %s (%d rows)", args.out, len(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
