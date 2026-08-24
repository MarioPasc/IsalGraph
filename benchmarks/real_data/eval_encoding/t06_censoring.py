"""D14 censoring, per dataset and per size stratum, as one auditable object.

Acceptance criterion A3 names ``censoring.json``. The measurement itself was
made by the encoding campaign at the frozen 300 s per-graph budget with a killed
subprocess, and its numbers are recoverable by joining ``manifest.json`` to
``completion_rates.json``. That is not the same as an artifact a reviewer can
open, which is what A3 asks for, so this module reads the encoding ``.npz``
files and emits the accounting directly.

**A censored graph is retained, not dropped** (D14 / F-4). It keeps its
greedy-min fallback string, so it still produces an encoding and still enters
every downstream analysis. The censoring rate is therefore a *reported result*,
never an exclusion, and the complete-case arm sits beside the primary one so the
selection D14 exists to expose is visible rather than argued about.

**The stratification is by size, and that is a limitation stated rather than
worked around.** A3 asks for the rate per *symmetry* stratum, on the premise
that the graphs which exhaust the budget are those with the largest automorphism
groups. No artifact this ticket holds carries ``|Aut|`` --- not the encoding
files, not ``eval/graph_metadata/``, which carries only labels, node counts and
edge counts --- so a symmetry stratum here would be invented. The node-count
stratum is emitted in its place, it is informative in the same direction, and
the gap is recorded in the file itself.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from benchmarks.real_data.eval_encoding.t06_claim_a import clopper_pearson

LOGGER: Final = logging.getLogger(__name__)

#: The arm D14 governs. Only the reference arm censors; a comparator that failed
#: is an ``error``, which is a different disposition.
REFERENCE_ARM: Final[str] = "isalgraph_pruned"

#: The frozen per-graph budget (F-3), enforced by a killed subprocess.
ENCODE_BUDGET_S: Final[float] = 300.0

#: Node-count band edges. Chosen once, before the rates were read, so the bands
#: cannot have been drawn around the answer.
SIZE_BANDS: Final[tuple[int, ...]] = (0, 10, 20, 30, 40, 60, 80, 10**9)

#: Why the symmetry stratum A3 names is absent.
SYMMETRY_GAP: Final[str] = (
    "A3 asks for the censoring rate per SYMMETRY stratum. No artifact this ticket holds "
    "carries |Aut|: the encoding .npz files carry graph_ids, node_counts, edge_counts, "
    "status, seconds and the bit counts, and eval/graph_metadata/ carries labels, node "
    "counts and edge counts. Computing |Aut| would be a separate campaign over both "
    "cohorts. The node-count stratum below is emitted in its place and the substitution is "
    "recorded here rather than passed off as the stratum that was asked for."
)


class CensoringError(Exception):
    """Raised when the censoring accounting cannot be assembled."""


@dataclass(frozen=True)
class DatasetCensoring:
    """One dataset's D14 accounting.

    Attributes:
        suite: Suite key.
        dataset: Dataset key.
        n_graphs: Graphs in the cohort.
        n_ok: Graphs encoded within budget.
        n_censored: Graphs that exhausted the budget and kept a fallback string.
        n_error: Graphs with no encoding at all.
        rate: ``n_censored / n_graphs``.
        ci_low: Clopper-Pearson lower limit on that rate.
        ci_high: Clopper-Pearson upper limit.
        invariant_holds: Whether every censored row carries ``fallback_used``,
            a non-empty encoding and a non-negative length --- the D14 invariant.
        median_seconds: Median solver seconds over the cohort.
        total_seconds: Solver seconds over the cohort.
        max_seconds: Largest solver time seen.
        censored_node_counts: ``(min, median, max)`` over censored graphs, or
            ``None`` when none censored.
        kept_node_counts: The same over graphs that did not censor.
    """

    suite: str
    dataset: str
    n_graphs: int
    n_ok: int
    n_censored: int
    n_error: int
    rate: float
    ci_low: float
    ci_high: float
    invariant_holds: bool
    median_seconds: float
    total_seconds: float
    max_seconds: float
    censored_node_counts: tuple[int, int, int] | None
    kept_node_counts: tuple[int, int, int] | None


def _band(count: int) -> str:
    """Return the node-count band label for *count*."""
    for low, high in zip(SIZE_BANDS, SIZE_BANDS[1:], strict=False):
        if low < count <= high:
            return f"{low + 1}-{high}" if high < 10**9 else f"{low + 1}+"
    return "unclassified"


def _triple(values: np.ndarray) -> tuple[int, int, int] | None:
    """Return ``(min, median, max)`` or ``None`` for an empty selection."""
    if values.size == 0:
        return None
    return int(values.min()), int(np.median(values)), int(values.max())


def measure(path: Path, suite: str, dataset: str) -> DatasetCensoring:
    """Read one encoding file and return its D14 accounting.

    Args:
        path: The reference arm's ``.npz``.
        suite: Suite key.
        dataset: Dataset key.

    Returns:
        The accounting.
    """
    with np.load(path, allow_pickle=True) as handle:
        status = np.asarray(handle["status"]).astype(str)
        node_counts = np.asarray(handle["node_counts"], dtype=np.int64)
        seconds = np.asarray(handle["seconds"], dtype=np.float64)
        fallback = np.asarray(handle["fallback_used"], dtype=bool)
        encoding = np.asarray(handle["encoding"]).astype(str)
        length = np.asarray(handle["length"], dtype=np.int64)

    censored = status == "censored"
    n = int(status.size)
    n_censored = int(censored.sum())
    low, high = clopper_pearson(n_censored, n)
    invariant = bool(
        censored.sum() == 0
        or (
            fallback[censored].all()
            and (encoding[censored] != "").all()
            and (length[censored] >= 0).all()
        )
    )
    return DatasetCensoring(
        suite=suite,
        dataset=dataset,
        n_graphs=n,
        n_ok=int((status == "ok").sum()),
        n_censored=n_censored,
        n_error=int((status == "error").sum()),
        rate=n_censored / n if n else 0.0,
        ci_low=low,
        ci_high=high,
        invariant_holds=invariant,
        median_seconds=float(np.median(seconds)),
        total_seconds=float(seconds.sum()),
        max_seconds=float(seconds.max()) if seconds.size else 0.0,
        censored_node_counts=_triple(node_counts[censored]),
        kept_node_counts=_triple(node_counts[~censored]),
    )


def size_strata(path: Path) -> list[dict[str, Any]]:
    """Return the censoring rate per node-count band for one dataset.

    Args:
        path: The reference arm's ``.npz``.

    Returns:
        One record per non-empty band.
    """
    with np.load(path, allow_pickle=True) as handle:
        status = np.asarray(handle["status"]).astype(str)
        node_counts = np.asarray(handle["node_counts"], dtype=np.int64)
    labels = np.array([_band(int(c)) for c in node_counts])
    records: list[dict[str, Any]] = []
    for label in sorted(set(labels), key=lambda s: int(s.split("-")[0].rstrip("+"))):
        selection = labels == label
        total = int(selection.sum())
        censored = int((selection & (status == "censored")).sum())
        records.append(
            {
                "band": label,
                "n_graphs": total,
                "n_censored": censored,
                "rate": censored / total if total else 0.0,
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encodings", type=Path, required=True, help="the encodings/ tree")
    ap.add_argument("--out", type=Path, required=True, help="censoring.json")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        0 on success, 1 when nothing was measured.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows: list[DatasetCensoring] = []
    strata: dict[str, list[dict[str, Any]]] = {}
    for suite in ("suite1", "suite2"):
        for path in sorted((args.encodings / suite).glob(f"*__{REFERENCE_ARM}.npz")):
            dataset = path.name.split("__")[0]
            rows.append(measure(path, suite, dataset))
            strata[f"{suite}/{dataset}"] = size_strata(path)
            LOGGER.info(
                "%s/%-16s %5d graphs  censored %4d (%.2f %%)  invariant=%s",
                suite,
                dataset,
                rows[-1].n_graphs,
                rows[-1].n_censored,
                100.0 * rows[-1].rate,
                rows[-1].invariant_holds,
            )

    if not rows:
        LOGGER.error("no reference-arm encodings under %s", args.encodings)
        return 1

    by_suite = {
        suite: {
            "n_graphs": sum(r.n_graphs for r in rows if r.suite == suite),
            "n_censored": sum(r.n_censored for r in rows if r.suite == suite),
        }
        for suite in ("suite1", "suite2")
    }
    for block in by_suite.values():
        block["rate"] = block["n_censored"] / block["n_graphs"] if block["n_graphs"] else 0.0

    payload = {
        "schema_version": "t06.censoring.1",
        "ticket": "T-06",
        "acceptance_criterion": "A3",
        "reference_arm": REFERENCE_ARM,
        "encode_budget_s": ENCODE_BUDGET_S,
        "enforcement": "killed subprocess, never signal.setitimer (F-3)",
        "disposition": (
            "A censored graph is RETAINED with its greedy-min fallback string and flagged "
            "(D14 / F-4). It is never dropped, so the rate is a reported result and not an "
            "exclusion; the complete-case arm is reported beside the primary one."
        ),
        "symmetry_stratum": SYMMETRY_GAP,
        "reporting_rule": (
            "No cohort-level censoring rate may be quoted without naming Mutagenicity: the "
            "rate is not a property of the cohort, it is one dataset's property diluted by "
            "nine others."
        ),
        "totals": by_suite,
        "invariant_holds_everywhere": all(r.invariant_holds for r in rows),
        "n_rows": len(rows),
        "rows": [asdict(r) for r in rows],
        "size_strata": strata,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
