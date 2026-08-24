"""Assemble, from the T-06 artifact tree, the inputs each F2 row consumes.

This module does loading and alignment only. It computes no statistic, so the
statistical layer in :mod:`t06_f2` can be read without the file-format detail
and this file can be read without the inference.

Three rules it enforces, each because the alternative produces a number that
reads as evidence and is not.

**Every join is on ``graph_ids`` (F-12).** Suite-1 ``aids`` is 769 graphs and
Suite-2 ``aids_graphedx`` is 819; the first is a verified strict subset of the
second. A positional join between them is silently wrong on 50 rows.

**A representation's primary distance is discovered, not asserted.** The
selection was made by ``grid.py`` on F1-F4 with cost as tie-break and blind to
correlation with GED. Re-declaring it here as a literal would let this module
disagree with the selection without anything failing, so the metric is read off
the artifact tree and the read is required to be unambiguous.

**The size null is per ``(representation, dataset)``** (``CONTRACTS.md`` 4.1).
Censoring is not independent of size and differs by representation, so one null
per dataset is computed over pairs some arm was never evaluated on. Each arm
carries its own.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from benchmarks.real_data.eval_stats.family import ReductionInputs
from benchmarks.real_data.eval_stats.t06_gates import APPROX_ALIAS, _subset_on_ids

LOGGER: Final = logging.getLogger(__name__)

#: The reference arm, frozen by measurement in the design note (F-1).
REFERENCE_ARM: Final[str] = "isalgraph_pruned"

#: The two bit conventions F-5 requires to be reported together.
BIT_CONVENTIONS: Final[tuple[str, str]] = ("entropy_bits", "realised_bits")

#: Statuses that contribute to the primary Claim-A arm. A censored graph is
#: retained with its greedy-min string (D14/F-4), so it does produce an
#: encoding and does contribute; only ``error`` rows are absent.
PRIMARY_STATUSES: Final[tuple[str, ...]] = ("ok", "censored", "fallback")

#: Suffix of the per-``(representation, dataset)`` trivial baseline.
SIZE_NULL_METRIC: Final[str] = "size_null"


class F2InputError(Exception):
    """Raised when an F2 input cannot be assembled from the artifact tree."""


@dataclass(frozen=True)
class ArmEncodings:
    """One representation's encoding record for one cohort.

    Attributes:
        representation: Backend name.
        graph_ids: Cohort ids, in file order.
        node_counts: Nodes per graph.
        edge_counts: Edges per graph.
        status: ``ok``, ``censored`` or ``error`` per graph.
        bits: Bit counts keyed by convention name.
        fallback_used: Whether the D14 greedy-min string was substituted.
    """

    representation: str
    graph_ids: npt.NDArray[Any]
    node_counts: npt.NDArray[Any]
    edge_counts: npt.NDArray[Any]
    status: npt.NDArray[Any]
    bits: dict[str, npt.NDArray[Any]]
    fallback_used: npt.NDArray[Any]

    def usable(self, arm: str) -> npt.NDArray[Any]:
        """Return the mask of graphs contributing to *arm*.

        Args:
            arm: ``"primary"`` or ``"complete_case"``.

        Returns:
            Boolean mask over :attr:`graph_ids`.
        """
        if arm == "complete_case":
            return np.asarray(self.status == "ok")
        return np.isin(self.status, PRIMARY_STATUSES)


@dataclass(frozen=True)
class ArmMatrices:
    """One representation's distance matrix and its own size null.

    Attributes:
        representation: Backend name.
        metric: The primary distance selected for it, e.g. ``levenshtein``.
        distance: Square distance matrix.
        defined: The pairs this representation actually encoded.
        size_null: ``|n_i - n_j|`` on the same index set, unrestricted.
        graph_ids: Ids in matrix order.
        node_counts: Nodes per graph, in matrix order.
    """

    representation: str
    metric: str
    distance: npt.NDArray[Any]
    defined: npt.NDArray[Any]
    size_null: npt.NDArray[Any]
    graph_ids: npt.NDArray[Any]
    node_counts: npt.NDArray[Any]


def primary_metric(distances: Path, suite: str, dataset: str, representation: str) -> str | None:
    """Return the primary distance selected for *representation* on *dataset*.

    Read off the artifact tree rather than declared, so this module cannot
    disagree with ``grid.py``'s F5-blind selection without failing.

    Args:
        distances: The ``distances/`` tree.
        suite: Suite key.
        dataset: Dataset key.
        representation: Backend name.

    Returns:
        The metric name, or ``None`` when the cell was never computed.

    Raises:
        F2InputError: If more than one non-null metric file exists, which would
            mean the selection is ambiguous rather than absent.
    """
    prefix = f"{dataset}__{representation}__"
    found = [
        path.name[len(prefix) : -len(".npz")]
        for path in sorted((distances / suite).glob(f"{prefix}*.npz"))
    ]
    metrics = [name for name in found if name != SIZE_NULL_METRIC]
    if not metrics:
        return None
    if len(metrics) > 1:
        raise F2InputError(
            f"{suite}/{dataset}/{representation} carries {len(metrics)} distances {metrics}; "
            "the primary selection must be unambiguous"
        )
    return metrics[0]


def load_arm(
    distances: Path,
    suite: str,
    dataset: str,
    representation: str,
    *,
    target_ids: npt.NDArray[Any] | None = None,
) -> ArmMatrices | None:
    """Load one representation's distance matrix and size null, aligned.

    Args:
        distances: The ``distances/`` tree.
        suite: Suite key.
        dataset: Dataset key.
        representation: Backend name.
        target_ids: Ids to align onto, or ``None`` to keep the file's own.

    Returns:
        The arm, or ``None`` when the cell was never computed.

    Raises:
        F2InputError: If the size null is missing, which would leave a printed
            rho without the baseline F-11 requires beside it.
    """
    metric = primary_metric(distances, suite, dataset, representation)
    if metric is None:
        return None
    root = distances / suite
    with np.load(root / f"{dataset}__{representation}__{metric}.npz", allow_pickle=True) as z:
        ids = np.asarray(z["graph_ids"]).astype(str)
        distance = np.asarray(z["distance_matrix"], dtype=np.float64)
        defined = np.asarray(z["defined_mask"], dtype=bool)
        node_counts = np.asarray(z["node_counts"], dtype=np.int64)

    null_path = root / f"{dataset}__{representation}__{SIZE_NULL_METRIC}.npz"
    if not null_path.exists():
        raise F2InputError(
            f"{suite}/{dataset}/{representation} has a distance but no size null; "
            "F-11 requires the trivial baseline beside every printed rho"
        )
    with np.load(null_path, allow_pickle=True) as z:
        null_ids = np.asarray(z["graph_ids"]).astype(str)
        size_null = _subset_on_ids(
            np.asarray(z["distance_matrix"], dtype=np.float64), null_ids, ids
        )

    if target_ids is not None:
        distance = _subset_on_ids(distance, ids, target_ids)
        defined = _subset_on_ids(defined.astype(np.float64), ids, target_ids) > 0.5
        size_null = _subset_on_ids(size_null, ids, target_ids)
        order = {gid: j for j, gid in enumerate(ids)}
        node_counts = node_counts[np.array([order[g] for g in target_ids])]
        ids = np.asarray(target_ids)

    return ArmMatrices(
        representation=representation,
        metric=metric,
        distance=distance,
        defined=defined,
        size_null=size_null,
        graph_ids=ids,
        node_counts=node_counts,
    )


def load_references(
    suite: str,
    dataset: str,
    target_ids: npt.NDArray[Any],
    ged_root: Path,
    approx_root: Path,
) -> dict[str, npt.NDArray[Any]]:
    """Load the ground-truth GED matrices for one dataset, aligned.

    Suite 1 carries ``exact``. Suite 2 carries ``lb`` and ``ub`` as two separate
    records, never averaged and never interpolated (F-10). Both live in the one
    ``LB/`` archive.

    Args:
        suite: Suite key.
        dataset: Dataset key.
        target_ids: Ids to align onto.
        ged_root: Suite-1 exact matrices.
        approx_root: The ``APPROX_GED`` root.

    Returns:
        Named reference matrices; may be empty when none exists.
    """
    references: dict[str, npt.NDArray[Any]] = {}
    approx_path = approx_root / "LB" / f"{APPROX_ALIAS.get(dataset, dataset)}.npz"
    if suite == "suite2" and approx_path.exists():
        with np.load(approx_path, allow_pickle=True) as z:
            src = np.asarray(z["graph_ids"]).astype(str)
            for key, field in (("lb", "lb_matrix"), ("ub", "ub_matrix")):
                references[key] = _subset_on_ids(
                    np.asarray(z[field], dtype=np.float64), src, target_ids
                )
    if suite == "suite1":
        exact_path = ged_root / f"{dataset}.npz"
        if exact_path.exists():
            with np.load(exact_path, allow_pickle=True) as z:
                src = np.asarray(z["graph_ids"]).astype(str)
                references["exact"] = _subset_on_ids(
                    np.asarray(z["ged_matrix"], dtype=np.float64), src, target_ids
                )
    return references


def load_encodings(
    encodings: Path, suite: str, dataset: str, representation: str
) -> ArmEncodings | None:
    """Load one representation's encoding record.

    Args:
        encodings: The ``encodings/`` tree.
        suite: Suite key.
        dataset: Dataset key.
        representation: Backend name.

    Returns:
        The record, or ``None`` when the file does not exist.
    """
    path = encodings / suite / f"{dataset}__{representation}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return ArmEncodings(
            representation=representation,
            graph_ids=np.asarray(z["graph_ids"]).astype(str),
            node_counts=np.asarray(z["node_counts"], dtype=np.int64),
            edge_counts=np.asarray(z["edge_counts"], dtype=np.int64),
            status=np.asarray(z["status"]).astype(str),
            bits={name: np.asarray(z[name], dtype=np.float64) for name in BIT_CONVENTIONS},
            fallback_used=np.asarray(z["fallback_used"], dtype=bool),
        )


def paired_bits(
    reference: ArmEncodings, competitor: ArmEncodings, convention: str, arm: str
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Return bit counts for the graphs both arms encoded, joined on ids.

    A per-dataset difference between two columns computed over different graph
    subsets is not a difference, so the join is explicit and the intersection
    is taken before any statistic.

    Args:
        reference: The IsalGraph arm.
        competitor: The comparator.
        convention: ``entropy_bits`` or ``realised_bits``.
        arm: ``"primary"`` or ``"complete_case"``.

    Returns:
        ``(reference_bits, competitor_bits)``, equal length, same graphs.
    """
    ref_pos = {gid: i for i, gid in enumerate(reference.graph_ids)}
    com_pos = {gid: i for i, gid in enumerate(competitor.graph_ids)}
    ref_ok, com_ok = reference.usable(arm), competitor.usable(arm)
    keep = [
        (ref_pos[gid], com_pos[gid])
        for gid in np.intersect1d(reference.graph_ids, competitor.graph_ids)
        if ref_ok[ref_pos[gid]] and com_ok[com_pos[gid]]
    ]
    if not keep:
        return np.empty(0), np.empty(0)
    ref_bits = reference.bits[convention][np.array([i for i, _ in keep])]
    com_bits = competitor.bits[convention][np.array([j for _, j in keep])]
    finite = np.isfinite(ref_bits) & np.isfinite(com_bits)
    return ref_bits[finite], com_bits[finite]


def edge_counts_for(
    encodings: Path, suite: str, dataset: str, target_ids: npt.NDArray[Any]
) -> npt.NDArray[Any] | None:
    """Return edge counts aligned to *target_ids*, for the MRM density term.

    The distance archives carry ``node_counts`` but not ``edge_counts``, so the
    density predictor is joined from the reference arm's encoding record.

    Args:
        encodings: The ``encodings/`` tree.
        suite: Suite key.
        dataset: Dataset key.
        target_ids: Ids to align onto.

    Returns:
        Edge counts, or ``None`` when the reference encoding is absent.

    Raises:
        F2InputError: If a target id is absent from the encoding record.
    """
    record = load_encodings(encodings, suite, dataset, REFERENCE_ARM)
    if record is None:
        return None
    position = {gid: i for i, gid in enumerate(record.graph_ids)}
    missing = [gid for gid in target_ids if gid not in position]
    if missing:
        raise F2InputError(
            f"{suite}/{dataset}: {len(missing)} matrix ids absent from the encoding record, "
            f"e.g. {missing[:3]}"
        )
    return record.edge_counts[np.array([position[gid] for gid in target_ids])]


def noncomputable_triples(
    completion_rates: Path, *, threshold: float = 0.99
) -> tuple[frozenset[tuple[str, str, str]], list[dict[str, Any]]]:
    """Return ``c``'s source triples and the rows behind them.

    ``preregistration.md`` 5.1: a representation is computable on a dataset iff
    it produces an encoding for at least *threshold* of that dataset's graphs
    within the frozen 300 s per-graph budget. Consequence 2 of the same section
    exempts the IsalGraph reference arm from ``c`` entirely: D14 governs it, its
    censoring rate is a reported result rather than an exclusion, and a censored
    graph is retained with its greedy-min string, so it *did* produce an
    encoding.

    Args:
        completion_rates: ``completion_rates.json`` from the encoding campaign.
        threshold: The frozen computability criterion.

    Returns:
        ``(triples, rows)`` where *rows* are the failing records, including the
        exempted reference-arm rows, so the exemption is visible rather than
        silent.
    """
    payload = json.loads(completion_rates.read_text())
    failing = [row for row in payload["rows"] if float(row["rate"]) < threshold]
    triples = frozenset(
        (row["suite"], row["dataset"], row["representation"])
        for row in failing
        if row["representation"] != REFERENCE_ARM
    )
    for row in failing:
        if row["representation"] == REFERENCE_ARM:
            LOGGER.info(
                "%s/%s/%s at %.4f is below the threshold but is EXEMPT from c "
                "(preregistration.md 5.1 consequence 2: D14 governs the reference arm)",
                row["suite"],
                row["dataset"],
                row["representation"],
                float(row["rate"]),
            )
    return triples, failing


def build_reduction_inputs(
    completion_rates: Path,
    *,
    excluded_representations: frozenset[str],
    f0_demotes_approximate: bool,
    uninformative_datasets: frozenset[str] = frozenset(),
) -> tuple[ReductionInputs, list[dict[str, Any]]]:
    """Build the frozen reduction terms for F2.

    Under F0's majority branch ``d`` is **not applied at all**
    (``preregistration.md`` 5.3): F1 tests the bracket within the approximate
    regime, and once that regime is descriptive F1 removes nothing from a family
    that no longer contains its rows. F1's ``d`` is still reported beside F0; it
    simply does not reduce the denominator.

    Args:
        completion_rates: ``completion_rates.json``.
        excluded_representations: ``k``.
        f0_demotes_approximate: F0's majority branch.
        uninformative_datasets: ``d``, ignored when the branch fired.

    Returns:
        ``(inputs, failing_rows)``.

    Raises:
        F2InputError: If the branch fired and ``d`` was nevertheless supplied
            as if it were to be applied.
    """
    triples, failing = noncomputable_triples(completion_rates)
    if f0_demotes_approximate and uninformative_datasets:
        raise F2InputError(
            "F0's majority branch fired, so d is not applied (preregistration.md 5.3); "
            "supplying d here would charge the same cells twice"
        )
    return (
        ReductionInputs(
            excluded_representations=excluded_representations,
            uninformative_datasets=uninformative_datasets,
            noncomputable=triples,
            f0_demotes_approximate=f0_demotes_approximate,
        ),
        failing,
    )
