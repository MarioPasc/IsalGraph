"""Re-derive the cohort table: both suites, retained and discarded, from tracked code.

T-01. The Suite-2 half of ``.claude/notes/review/plan/data.md`` section 1 -- 19,670
graphs, 40,024,242 pairs, ``n_max = 98`` -- and every discard ratio in sections 3
and 5 came from scripts that no longer exist and never entered git. This module
replaces them. **What it measures becomes the table**; a disagreement with the
recorded value is a finding, not something to reconcile away.

The precedent is T-25: six figures were re-derived from a lost script and none
reproduced, all six in the flattering direction.

What it produces
----------------
For each dataset, in both suites:

* retained ``n`` mean / median / max, ``m`` mean, density, and ``C(kept, 2)``;
* **discarded** ``n`` mean / median / max, ``m`` mean and density, **broken down
  by discard reason** -- which is what ``data.md`` section 3 disclosure 1 promises to
  print and what ``export_graphs.py`` never emitted;
* the discarded-to-retained size ratio, the size-bias figure the paper quotes.

Suite 1 (``n_max = 12``) is a **self-check**: it must reproduce the counts
``export_graphs.py`` asserts. A mismatch means this module is wrong, not that the
locked cohort is.

Run::

    python -m benchmarks.real_data.eval_setup.cohort_audit

Reference:
    Riesen, K. & Bunke, H. (2008). IAM Graph Database Repository for Graph Based
    Pattern Recognition and Machine Learning. SSPR/SPR 2008, LNCS 5342, 287-297.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from math import comb
from pathlib import Path
from typing import Any

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_filter import FilterResult, filter_graphs  # noqa: E402
from iam_gxl_loader import (  # noqa: E402
    IAM_DATASETS,
    REJECTED_KEYS,
    Enumeration,
    load_iam_gxl,
)

logger = logging.getLogger(__name__)

#: The submitted filter, unchanged. ``data.md`` section 1.
FILTER_MIN_NODES = 2
FILTER_REQUIRE_CONNECTED = True

#: Suite 1's ceiling, and Suite 2's absence of one. ``filter_graphs`` requires an
#: int, so "no ceiling" is a sentinel far above the largest IAM graph (417 nodes).
SUITE1_N_MAX = 12
NO_N_MAX = 10**9

DEFAULT_SOURCE_DIR = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source"
DEFAULT_OUT_DIR = "results/cohort_audit"

#: Suite 1: the five datasets the submitted manuscript reports, under ``n <= 12``.
SUITE1_KEYS: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "aids_graphedx",
)

#: Suite 2: the same five without a ceiling, plus the five added for scale.
SUITE2_KEYS: tuple[str, ...] = SUITE1_KEYS + (
    "grec",
    "aids_iam",
    "coil_del",
    "mutagenicity",
    "protein",
)

#: ``export_graphs.py``'s asserted Suite-1 cohort, duplicated here **on purpose**:
#: importing it would make the check circular if that module were ever edited.
#: These are the numbers the submitted manuscript prints.
SUITE1_EXPECTED: dict[str, tuple[int, int]] = {
    "iam_letter_low": (1180, 695610),
    "iam_letter_med": (1253, 784378),
    "iam_letter_high": (2059, 2118711),
    "linux": (89, 3916),
    "aids_graphedx": (769, 295296),
}
SUITE1_EXPECTED_TOTAL_PAIRS = 3897911


class CohortAuditError(Exception):
    """Raised when a dataset cannot be loaded or a locked count does not reproduce."""


@dataclass
class GraphStats:
    """Size and density summary for a set of graphs.

    ``density_mean`` is the mean of per-graph ``2m / (n(n-1))``; ``density_aggregate``
    applies the same formula to the mean ``n`` and ``m``. They differ, and the
    cohort table has always reported the former -- both are emitted so that no
    future reader has to guess which convention a printed number used.
    """

    count: int = 0
    n_mean: float = 0.0
    n_median: float = 0.0
    n_min: int = 0
    n_max: int = 0
    m_mean: float = 0.0
    density_mean: float = 0.0
    density_aggregate: float = 0.0

    @classmethod
    def from_graphs(cls, graphs: list[nx.Graph]) -> GraphStats:
        """Summarise a graph list. An empty list yields an all-zero record."""
        if not graphs:
            return cls()

        node_counts = [g.number_of_nodes() for g in graphs]
        edge_counts = [g.number_of_edges() for g in graphs]
        # Density is undefined below two nodes; those graphs are excluded from the
        # density mean and counted nowhere else, so the mean is over n >= 2.
        densities = [
            2.0 * m / (n * (n - 1)) for n, m in zip(node_counts, edge_counts, strict=True) if n >= 2
        ]

        n_mean = statistics.fmean(node_counts)
        m_mean = statistics.fmean(edge_counts)
        aggregate = 2.0 * m_mean / (n_mean * (n_mean - 1)) if n_mean > 1 else 0.0

        return cls(
            count=len(graphs),
            n_mean=n_mean,
            n_median=statistics.median(node_counts),
            n_min=min(node_counts),
            n_max=max(node_counts),
            m_mean=m_mean,
            density_mean=statistics.fmean(densities) if densities else 0.0,
            density_aggregate=aggregate,
        )


@dataclass
class CohortRow:
    """One dataset in one suite, retained and discarded side by side."""

    key: str
    suite: str
    source: str
    n_raw: int
    n_kept: int
    keep_pct: float
    n_pairs: int
    retained: GraphStats = field(default_factory=GraphStats)
    discarded: GraphStats = field(default_factory=GraphStats)
    discarded_by_reason: dict[str, GraphStats] = field(default_factory=dict)
    size_bias: float = 0.0
    n_files_on_disk: int = 0
    n_files_indexed: int = 0
    node_attributes: list[str] = field(default_factory=list)
    edge_attributes: list[str] = field(default_factory=list)
    parse_failures: list[str] = field(default_factory=list)


def _reason_bucket(reason: str) -> str:
    """Collapse ``filter_graphs``'s parametrised reason string to its category."""
    if reason.startswith("trivial"):
        return "trivial"
    if reason.startswith("too_large"):
        return "too_large"
    if reason.startswith("disconnected"):
        return "disconnected"
    return "other"


def audit(
    key: str,
    suite: str,
    source: str,
    graphs: list[nx.Graph],
    graph_ids: list[str],
    n_max: int,
) -> CohortRow:
    """Filter one dataset and summarise both sides of the split.

    Parameters
    ----------
    key, suite, source : str
        Cohort key, ``'suite1'`` or ``'suite2'``, and the loader family.
    graphs, graph_ids : list
        The raw dataset, splits already merged (decision 3).
    n_max : int
        ``SUITE1_N_MAX`` or :data:`NO_N_MAX`.

    Returns
    -------
    CohortRow
        Retained and discarded statistics, and the size-bias ratio.
    """
    result: FilterResult = filter_graphs(
        graphs,
        graph_ids,
        n_max=n_max,
        require_connected=FILTER_REQUIRE_CONNECTED,
        min_nodes=FILTER_MIN_NODES,
    )

    kept = [graphs[i] for i in result.kept_indices]
    dropped = [graphs[i] for i in result.dropped_indices]

    by_reason: dict[str, list[nx.Graph]] = {}
    for i in result.dropped_indices:
        by_reason.setdefault(_reason_bucket(result.dropped_reasons[i]), []).append(graphs[i])

    retained = GraphStats.from_graphs(kept)
    discarded = GraphStats.from_graphs(dropped)

    return CohortRow(
        key=key,
        suite=suite,
        source=source,
        n_raw=result.n_raw,
        n_kept=result.n_kept,
        keep_pct=100.0 * result.n_kept / result.n_raw if result.n_raw else 0.0,
        n_pairs=comb(result.n_kept, 2),
        retained=retained,
        discarded=discarded,
        discarded_by_reason={
            reason: GraphStats.from_graphs(gs) for reason, gs in sorted(by_reason.items())
        },
        size_bias=discarded.n_mean / retained.n_mean if retained.n_mean else 0.0,
    )


def _load_graphedx(source_dir: Path, name: str) -> tuple[list[nx.Graph], list[str]]:
    """Load one GraphEdX dataset, splits merged. Needs ``torch``."""
    from data_roots import resolve_graphedx_root
    from graphedx_loader import load_graphedx_dataset

    dataset = load_graphedx_dataset(name, str(resolve_graphedx_root(source_dir)))
    return dataset.graphs, dataset.graph_ids


def load_dataset(
    source_dir: Path, key: str, enumeration: Enumeration = "cxl"
) -> tuple[list[nx.Graph], list[str], dict[str, Any]]:
    """Load one cohort dataset by key, splits merged.

    Returns
    -------
    list[nx.Graph], list[str], dict
        Graphs, ids, and loader provenance (file counts, observed attribute
        names, parse failures). The provenance dict is empty for GraphEdX, whose
        ``.pt`` files carry no equivalent.
    """
    if key == "linux":
        graphs, ids = _load_graphedx(source_dir, "LINUX")
        return graphs, ids, {"source": "graphedx"}
    if key == "aids_graphedx":
        graphs, ids = _load_graphedx(source_dir, "AIDS")
        return graphs, ids, {"source": "graphedx"}
    if key in IAM_DATASETS:
        from data_roots import resolve_iam_root

        dataset = load_iam_gxl(str(resolve_iam_root(source_dir)), key, enumeration=enumeration)
        return (
            dataset.graphs,
            dataset.graph_ids,
            {
                "source": "iam_gxl",
                "enumeration": dataset.enumeration,
                "n_files_on_disk": dataset.n_files_on_disk,
                "n_files_indexed": dataset.n_files_indexed,
                "split_sizes": dataset.split_sizes,
                "node_attributes": dataset.node_attributes,
                "edge_attributes": dataset.edge_attributes,
                "parse_failures": dataset.parse_failures,
            },
        )
    raise CohortAuditError(f"unknown cohort key {key!r}")


def _attach_provenance(row: CohortRow, provenance: dict[str, Any]) -> CohortRow:
    """Copy loader provenance onto a finished row."""
    row.n_files_on_disk = int(provenance.get("n_files_on_disk", 0))
    row.n_files_indexed = int(provenance.get("n_files_indexed", 0))
    row.node_attributes = list(provenance.get("node_attributes", []))
    row.edge_attributes = list(provenance.get("edge_attributes", []))
    row.parse_failures = list(provenance.get("parse_failures", []))
    return row


def run_suite(source_dir: Path, suite: str, enumeration: Enumeration = "cxl") -> list[CohortRow]:
    """Audit every dataset in one suite."""
    keys = SUITE1_KEYS if suite == "suite1" else SUITE2_KEYS
    n_max = SUITE1_N_MAX if suite == "suite1" else NO_N_MAX

    rows: list[CohortRow] = []
    for key in keys:
        graphs, ids, provenance = load_dataset(source_dir, key, enumeration=enumeration)
        row = audit(key, suite, str(provenance.get("source", "")), graphs, ids, n_max)
        rows.append(_attach_provenance(row, provenance))
    return rows


def check_suite1(rows: list[CohortRow]) -> list[str]:
    """Compare Suite 1 against the counts ``export_graphs.py`` asserts.

    Returns
    -------
    list[str]
        One line per mismatch; empty when the locked cohort reproduces.
    """
    problems: list[str] = []
    for row in rows:
        expected_kept, expected_pairs = SUITE1_EXPECTED[row.key]
        if row.n_kept != expected_kept:
            problems.append(f"{row.key}: kept {row.n_kept}, locked {expected_kept}")
        if row.n_pairs != expected_pairs:
            problems.append(f"{row.key}: pairs {row.n_pairs}, locked {expected_pairs}")

    total = sum(r.n_pairs for r in rows)
    if total != SUITE1_EXPECTED_TOTAL_PAIRS:
        problems.append(f"total pairs {total}, locked {SUITE1_EXPECTED_TOTAL_PAIRS}")
    return problems


def _fmt_row(row: CohortRow) -> str:
    r, d = row.retained, row.discarded
    return (
        f"| {row.key} | {row.n_raw:,} | **{row.n_kept:,}** | {row.keep_pct:.1f} % | "
        f"{r.n_mean:.2f} | {r.n_median:.0f} | {r.n_max} | {r.m_mean:.2f} | "
        f"{r.density_mean:.3f} | {row.n_pairs:,} | "
        f"{d.count:,} | {d.n_mean:.2f} | {d.n_max if d.count else 0} | "
        f"{row.size_bias:.2f}x |"
    )


def to_markdown(rows: list[CohortRow], title: str) -> str:
    """Render one suite as the cohort table, retained and discarded side by side."""
    header = (
        "| Dataset | raw | **kept** | keep % | n̄ | ñ | n max | m̄ | density | **pairs** "
        "| disc. | disc. n̄ | disc. n max | bias |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    body = "\n".join(_fmt_row(r) for r in rows)
    total_graphs = sum(r.n_kept for r in rows)
    total_pairs = sum(r.n_pairs for r in rows)
    n_max = max((r.retained.n_max for r in rows), default=0)
    total = (
        f"| **Total** | | **{total_graphs:,}** | | | | **{n_max}** | | | "
        f"**{total_pairs:,}** | | | | |"
    )
    return f"### {title}\n\n{header}\n{body}\n{total}\n"


def main(argv: list[str] | None = None) -> int:
    """Run both suites, write JSON and markdown, and check the Suite-1 lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE_DIR, help="data/source root")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument(
        "--enumeration",
        choices=("cxl", "directory"),
        default="cxl",
        help="how to enumerate IAM datasets; the two differ only on COIL-DEL",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    source_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite1 = run_suite(source_dir, "suite1", args.enumeration)
    suite2 = run_suite(source_dir, "suite2", args.enumeration)

    # The rejected datasets are audited under Suite 2's filter so their retention
    # and size bias are comparable to the cohort they were dropped in favour of.
    rejected: list[CohortRow] = []
    for key in REJECTED_KEYS:
        graphs, ids, provenance = load_dataset(source_dir, key, enumeration=args.enumeration)
        row = audit(key, "rejected", str(provenance.get("source", "")), graphs, ids, NO_N_MAX)
        rejected.append(_attach_provenance(row, provenance))

    for name, rows in (("suite1", suite1), ("suite2", suite2), ("rejected", rejected)):
        (out_dir / f"{name}.json").write_text(
            json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
        )

    (out_dir / "cohort_table.md").write_text(
        to_markdown(suite1, f"Suite 1 -- exact GED, n <= {SUITE1_N_MAX}")
        + "\n"
        + to_markdown(suite2, "Suite 2 -- proven bracket, no n_max")
        + "\n"
        + to_markdown(rejected, "Rejected from the cohort (audited, never used)"),
        encoding="utf-8",
    )

    problems = check_suite1(suite1)
    if problems:
        for line in problems:
            logger.error("SUITE-1 LOCK BROKEN: %s", line)
        logger.error(
            "Suite 1 does not reproduce export_graphs.py. This module is wrong, "
            "or the locked cohort is -- do not print either until it is resolved."
        )
        return 1

    logger.info(
        "Suite 1 reproduces the locked cohort exactly (%d pairs).", SUITE1_EXPECTED_TOTAL_PAIRS
    )
    logger.info(
        "Suite 2: %d graphs, %d pairs, n_max %d.",
        sum(r.n_kept for r in suite2),
        sum(r.n_pairs for r in suite2),
        max(r.retained.n_max for r in suite2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
