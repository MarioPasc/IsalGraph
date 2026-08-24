"""D14 censoring, per dataset and per size and symmetry stratum.

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

**Both stratifications are measured.** A3 asks for the rate per *symmetry*
stratum, on the premise that the graphs which exhaust the budget are those with
the largest automorphism groups. ``|Aut|`` is computed here with ``pynauty``'s
``autgrp`` over the exported CSR cohort --- measured at **0.118 ms per graph**,
worst case 0.9 ms at ``n = 98``, so roughly **3 s for all 21,720 graphs**. It was
briefly assumed to be a separate campaign; it is not, and the assumption was
replaced by the measurement. The node-count stratum is kept beside it because
the two answer different questions and the whole point is to tell size apart
from symmetry.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

#: log10 |Aut| band edges. Chosen before the rates were read.
AUT_BANDS: Final[tuple[float, ...]] = (-0.5, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9)

#: How the symmetry stratum is obtained, recorded so the cost is not re-guessed.
SYMMETRY_METHOD: Final[str] = (
    "|Aut| computed with pynauty 2.8.8.1 autgrp over the exported CSR cohort, joined to the "
    "encoding record on graph_ids. Measured cost 0.118 ms per graph, worst case 0.9 ms at "
    "n = 98, about 3 s for all 21,720 graphs -- so this is a stratification, not a campaign. "
    "Edge direction is ignored: the cohort is undirected and the canonical string does not "
    "encode directedness in any case."
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


def _aut_band(aut: float) -> str:
    """Return the log10 |Aut| band label.

    Args:
        aut: Automorphism-group order.

    Returns:
        A band label; ``|Aut| = 1`` (asymmetric) gets its own band because it is
        the qualitatively different case, not merely the smallest one.
    """
    if aut <= 1.0:
        return "1 (asymmetric)"
    exponent = math.log10(aut)
    for low, high in zip(AUT_BANDS, AUT_BANDS[1:], strict=False):
        if low < exponent <= high:
            return f"1e{low:g}-1e{high:g}" if high < 1e9 else f">1e{low:g}"
    return "unclassified"


def automorphism_orders(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(graph_ids, |Aut|, n_orbits)`` for one exported CSR cohort.

    Args:
        path: The dataset's exported ``.npz``.

    Returns:
        Ids in file order, the automorphism-group order per graph as a float
        (the groups reach 1e14, beyond exact int64 comfort but well inside
        float64 for a banding), and the orbit count.

    Raises:
        CensoringError: If ``pynauty`` is not importable.
    """
    try:
        import pynauty
    except ImportError as exc:  # pragma: no cover - environment guard
        raise CensoringError("pynauty is required for the symmetry stratum") from exc

    with np.load(path, allow_pickle=True) as handle:
        ids = np.asarray(handle["graph_ids"]).astype(str)
        n_nodes = np.asarray(handle["n_nodes"], dtype=np.int64)
        offsets = np.asarray(handle["edge_offsets"], dtype=np.int64)
        edges = np.asarray(handle["edges"], dtype=np.int64)

    orders = np.empty(ids.size, dtype=np.float64)
    orbits = np.empty(ids.size, dtype=np.int64)
    for i in range(ids.size):
        adjacency: dict[int, list[int]] = {}
        for a, b in zip(
            edges[0, offsets[i] : offsets[i + 1]],
            edges[1, offsets[i] : offsets[i + 1]],
            strict=True,
        ):
            adjacency.setdefault(int(a), []).append(int(b))
        graph = pynauty.Graph(int(n_nodes[i]), directed=False, adjacency_dict=adjacency)
        _, mantissa, exponent, _, n_orbits = pynauty.autgrp(graph)
        orders[i] = float(mantissa) * (10.0 ** int(exponent))
        orbits[i] = int(n_orbits)
    return ids, orders, orbits


def symmetry_strata(
    encoding_path: Path, exported_path: Path
) -> list[dict[str, Any]] | None:
    """Return the censoring rate per ``|Aut|`` band for one dataset.

    Args:
        encoding_path: The reference arm's ``.npz``.
        exported_path: The dataset's exported CSR ``.npz``.

    Returns:
        One record per non-empty band, or ``None`` when the cohort export is
        absent.
    """
    if not exported_path.exists():
        return None
    with np.load(encoding_path, allow_pickle=True) as handle:
        enc_ids = np.asarray(handle["graph_ids"]).astype(str)
        status = np.asarray(handle["status"]).astype(str)

    ids, orders, orbits = automorphism_orders(exported_path)
    position = {gid: i for i, gid in enumerate(ids)}
    if not set(enc_ids) <= set(position):
        raise CensoringError(
            f"{encoding_path.name}: graph_ids do not join onto {exported_path.name}"
        )
    # Join on graph_ids, never positionally (F-12).
    order = np.array([position[gid] for gid in enc_ids])
    aut = orders[order]
    labels = np.array([_aut_band(float(a)) for a in aut])

    records: list[dict[str, Any]] = []
    for label in sorted(set(labels), key=lambda s: (s != "1 (asymmetric)", s)):
        selection = labels == label
        total = int(selection.sum())
        censored = int((selection & (status == "censored")).sum())
        records.append(
            {
                "band": label,
                "n_graphs": total,
                "n_censored": censored,
                "rate": censored / total if total else 0.0,
                "median_aut": float(np.median(aut[selection])),
                "median_orbits": float(np.median(orbits[order][selection])),
            }
        )
    return records


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
    ap.add_argument(
        "--exported",
        type=Path,
        default=None,
        help="the exported/ tree root; suite1 reads exported/, suite2 exported_suite2/",
    )
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
    symmetry: dict[str, list[dict[str, Any]]] = {}
    for suite in ("suite1", "suite2"):
        exported = None
        if args.exported is not None:
            exported = args.exported / ("exported" if suite == "suite1" else "exported_suite2")
        for path in sorted((args.encodings / suite).glob(f"*__{REFERENCE_ARM}.npz")):
            dataset = path.name.split("__")[0]
            rows.append(measure(path, suite, dataset))
            strata[f"{suite}/{dataset}"] = size_strata(path)
            if exported is not None:
                bands = symmetry_strata(path, exported / f"{dataset}.npz")
                if bands is not None:
                    symmetry[f"{suite}/{dataset}"] = bands
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
        "symmetry_stratum_method": SYMMETRY_METHOD,
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
        "symmetry_strata": symmetry,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
