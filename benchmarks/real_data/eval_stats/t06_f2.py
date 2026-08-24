"""Run F2, the primary family, over the enumerated admissible cell set.

F0's majority branch fired on 4 of 5 Suite-1 datasets under the conservative
reading of ``GED_approx`` (design note 18.7), so the 81 approximate-regime cells
--- B1a 70, B2 1, B3a 10 --- leave the confirmatory family, ``d`` is not applied
at all, and ``k`` removes only its 5 B1e cells per representation. The family is
``101 - 5 k - c`` cells. **The enumeration in :func:`family.admissible_cells` is
the definition and the closed form is a printed check**; where they disagree the
enumeration wins and the discrepancy is reported.

Four rulings frozen before this module ran (design note 18.8):

**A1's bit convention is settled by an intersection-union test.** F-5 requires
both conventions reported and names no primary. Claim A read against a
two-convention report is conjunctive --- fewer bits under *both* --- and the IUT
for a conjunction takes ``p = max`` of the components. It is a level-alpha
procedure, it is conservative for BH because ``max(p1, p2) >= p1, p2``, and it
needs no primary to be named after the data exist. Both marginals are printed
and per-cell discordance is flagged rather than absorbed.

**Descriptive is not unmeasured.** The 81 demoted cells are computed at full
cohort. Since F0 fired, descriptive is the only form the large-``n`` story can
take, so those are the numbers the response letter quotes.

**The confirmatory view is ``all_pairs``; ``equal_n`` is descriptive** (design
note 18.9). ``preregistration`` section 4 names no pair view, and three things
settle it: A9's pair-accounting ladder carries no equal-``n`` rung, section 4.2's
"per Suite-1 dataset" is unqualified, and F0 and F1 have already run on the full
defined pair set. The ``equal_n`` arm is still computed, because inside such a
stratum the size null is identically constant so raw rho *is* the structural
signal --- but it is reported under a descriptive key with a **locally** adjusted
q, never as a third BH column beside the family's own two. Three columns side by
side is the shape a reader takes for three families.

**The MRM runs on ``all_pairs`` only.** D4 regresses GED on Levenshtein while
controlling for ``|delta n|``, which is identically zero inside an equal-``n``
view; the design matrix is rank-deficient there and the coefficient it would
report is not ``beta1``.

The bootstrap resamples **graphs, never pairs** (A8). Nothing in this module's
import closure reaches ``correlation_metrics.bootstrap_correlation``, and a test
asserts it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from benchmarks.real_data.eval_stats.association import (
    CorrelationSpec,
    DifferenceSpec,
    MrmResult,
    PairVariables,
    bootstrap_associations,
    delta_density_matrix,
    delta_n_matrix,
    mrm,
)
from benchmarks.real_data.eval_stats.family import (
    SUITE1,
    SUITE2,
    Cell,
    ReductionInputs,
    admissible_cells,
    run_f2,
)
from benchmarks.real_data.eval_stats.multiplicity import (
    Regime,
    benjamini_hochberg,
    friedman_omnibus,
    wilcoxon_holm_posthoc,
)
from benchmarks.real_data.eval_stats.multiplicity import _wilcoxon as wilcoxon_pair
from benchmarks.real_data.eval_stats.resampling import FDR_Q, SEED, bootstrap_tier
from benchmarks.real_data.eval_stats.t06_f2_inputs import (
    BIT_CONVENTIONS,
    REFERENCE_ARM,
    ArmMatrices,
    build_reduction_inputs,
    edge_counts_for,
    load_arm,
    load_encodings,
    load_references,
    paired_bits,
)

LOGGER: Final = logging.getLogger(__name__)

#: ``k``, from ``competitors.md`` 3.4 --- no candidate distance passes F1 at
#: 100 %, F2, F3 and F4. Frozen by T-04a, and F5-blind by construction.
EXCLUDED_REPRESENTATIONS: Final[frozenset[str]] = frozenset({"adjacency", "graph6", "sparse6"})

#: The Claim-B comparators surviving ``k``, in the frozen spelling.
FAMILY_COMPARATORS: Final[tuple[str, ...]] = ("nauty_graph6", "agm_cam", "min_dfs", "wl_subtree")

#: Computed and reported, but outside the frozen family: ``preregistration``
#: 4.1 names plain ``sparse6``, and this is the nauty-canonicalised variant that
#: T-04 added afterwards. Section 15.3 makes it one of the three representations
#: that beat the reference arm while failing nothing, so omitting it would flatter
#: the comparison.
DESCRIPTIVE_COMPARATORS: Final[tuple[str, ...]] = ("sparse6_nauty",)

#: The six Claim-A serialisations (``preregistration`` 4.1). A bit count needs
#: no distance, so ``k`` does not reach these.
CLAIM_A_COMPARATORS: Final[tuple[str, ...]] = (
    "graph6",
    "sparse6",
    "nauty_graph6",
    "adjacency",
    "agm_cam",
    "min_dfs",
)

#: Representations with no message length. Their A1 cell carries a reason, never
#: a number: a feature-vector "bit cost" measures the vectoriser, not the graph.
BIT_COUNT_UNDEFINED: Final[tuple[str, ...]] = ("wl_subtree", "size_null")

#: The two pair views. ``all_pairs`` is what F0 and F1 ran on.
VIEWS: Final[tuple[str, str]] = ("all_pairs", "equal_n")

#: Why the size null carries no value inside an equal-``n`` stratum.
EQUAL_N_NULL_REASON: Final[str] = (
    "the size_null distance |n_i - n_j| is identically zero over the pairs of this view, "
    "so the rank correlation has no denominator; raw rho IS the structural signal here "
    "(design note 16.3)"
)

#: Why D4 is not run inside an equal-``n`` stratum.
EQUAL_N_MRM_REASON: Final[str] = (
    "D4 controls for |delta n|, which is identically zero inside an equal-n stratum, so the "
    "design matrix is rank-deficient and the reported coefficient would not be beta1"
)


class F2DriverError(Exception):
    """Raised when an F2 row cannot be assembled or run."""


@dataclass
class RhoRecord:
    """One printed rho, with everything F-11 requires beside it.

    Attributes:
        suite: Suite key.
        dataset: Dataset key.
        representation: Backend name.
        metric: Its primary distance.
        reference: ``exact``, ``lb`` or ``ub``.
        view: ``all_pairs`` or ``equal_n``.
        rho: Point estimate with its graph-level bootstrap interval.
        tau_b: Kendall tau-b on the full sample.
        n_pairs: Pairs behind the estimate.
        n_graphs: Graphs behind the resample --- the resampling unit.
        null_rho: The per-``(representation, dataset)`` size null, or ``None``.
        null_undefined_reason: Why the null is absent, when it is.
        excess_over_null: Paired ``rho - null_rho`` on the shared resample.
        difference_vs_reference_arm: Paired ``rho(IsalGraph) - rho(this)``.
        p_value: Bootstrap p-value of that difference.
        reference_arm_rho_on_shared_pairs: The arm's rho on **this comparator's**
            pair set, which is what the paired difference is taken against. It
            differs from the arm's own reported rho whenever the comparator
            covers fewer pairs, and printing both is what makes the difference
            auditable rather than merely reported.
        row: The F2 row this record serves, or ``None`` when descriptive.
        in_family: Whether it carries a confirmatory cell.
        regime: ``exact`` or ``approximate``.
        tier: The frozen D15 effort behind the resample.
    """

    suite: str
    dataset: str
    representation: str
    metric: str
    reference: str
    view: str
    rho: dict[str, Any]
    tau_b: float
    n_pairs: int
    n_graphs: int
    null_rho: dict[str, Any] | None = None
    null_undefined_reason: str | None = None
    excess_over_null: dict[str, Any] | None = None
    difference_vs_reference_arm: dict[str, Any] | None = None
    p_value: float | None = None
    reference_arm_rho_on_shared_pairs: float | None = None
    row: str | None = None
    in_family: bool = False
    regime: str = "exact"
    tier: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "suite": self.suite,
            "dataset": self.dataset,
            "representation": self.representation,
            "metric": self.metric,
            "reference": self.reference,
            "view": self.view,
            "rho": self.rho,
            "tau_b": self.tau_b,
            "n_pairs": self.n_pairs,
            "n_graphs": self.n_graphs,
            "size_null": self.null_rho,
            "size_null_undefined_reason": self.null_undefined_reason,
            "excess_over_size_null": self.excess_over_null,
            "difference_vs_reference_arm": self.difference_vs_reference_arm,
            "p_value": self.p_value,
            "reference_arm_rho_on_shared_pairs": self.reference_arm_rho_on_shared_pairs,
            "row": self.row,
            "in_family": self.in_family,
            "regime": self.regime,
            "tier": self.tier,
        }


def _mask_digest(mask: npt.NDArray[Any]) -> str:
    """Return a stable digest of a boolean mask, for grouping equal masks."""
    return hashlib.blake2b(
        np.ascontiguousarray(mask, dtype=bool).tobytes(), digest_size=8
    ).hexdigest()


def equal_n_mask(node_counts: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return the ``n_i == n_j`` pair mask.

    Not materialised as an artifact (``CONTRACTS.md`` 4.1); the consumer derives
    it, which is what this is.

    Args:
        node_counts: Nodes per graph, in matrix order.

    Returns:
        Square boolean mask.
    """
    counts = np.asarray(node_counts).reshape(-1)
    return np.asarray(counts[:, None] == counts[None, :])


def _view_mask(view: str, node_counts: npt.NDArray[Any]) -> npt.NDArray[Any] | None:
    """Return the extra pair restriction a view imposes, or ``None``."""
    if view == "all_pairs":
        return None
    if view == "equal_n":
        return equal_n_mask(node_counts)
    raise F2DriverError(f"unknown view {view!r}")


@dataclass(frozen=True)
class CorrelationGroup:
    """Comparators sharing one validity mask, so one resample serves them all.

    Grouping is by the byte identity of a comparator's ``defined_mask``: equal
    masks give an equal intersection with the reference arm, hence an equal
    valid pair set and an equal resample. Comparators whose coverage differs ---
    ``agm_cam`` at 6 % of Protein, ``min_dfs`` on Mutagenicity --- fall into
    their own group rather than dragging every other arm down to their pairs.

    Attributes:
        digest: The shared mask digest.
        comparators: Arms in the group.
    """

    digest: str
    comparators: tuple[ArmMatrices, ...]


def _group_by_mask(arms: Sequence[ArmMatrices]) -> list[CorrelationGroup]:
    """Group comparator arms by the identity of their defined masks."""
    buckets: dict[str, list[ArmMatrices]] = {}
    for arm in arms:
        buckets.setdefault(_mask_digest(arm.defined), []).append(arm)
    return [CorrelationGroup(digest, tuple(group)) for digest, group in buckets.items()]


def _select(datasets: Sequence[str], only: frozenset[str] | None) -> tuple[str, ...]:
    """Return *datasets* restricted to *only*, preserving the frozen order.

    Args:
        datasets: The suite's frozen dataset order.
        only: Names to keep, or ``None`` for all of them.

    Returns:
        The retained datasets.
    """
    if not only:
        return tuple(datasets)
    return tuple(name for name in datasets if name in only)


def _reference_regime(reference: str) -> str:
    """Return the regime label a reference matrix belongs to."""
    return "exact" if reference == "exact" else "approximate"


def run_correlation_group(
    *,
    suite: str,
    dataset: str,
    view: str,
    arm: ArmMatrices,
    group: CorrelationGroup,
    references: dict[str, npt.NDArray[Any]],
    replicates: int | None,
    emit_arm: bool = False,
) -> list[RhoRecord]:
    """Bootstrap one group of comparators against every reference, paired.

    Every correlation in the group --- the reference arm, each comparator, and
    each of their size nulls --- is estimated on **one** graph-level resample,
    which is what makes ``rho(IsalGraph) - rho(comparator)`` a paired difference
    by construction rather than by a matching step afterwards (D7). This is the
    instrument design note 16.4 asks for and marginal CI overlap only
    approximates.

    Args:
        suite: Suite key.
        dataset: Dataset key.
        view: ``all_pairs`` or ``equal_n``.
        arm: The IsalGraph reference arm.
        group: Comparators sharing a validity mask.
        references: Ground-truth matrices, ``exact`` or ``lb``/``ub``.
        replicates: Override for the frozen tier effort; ``None`` in production.
        emit_arm: Whether to emit the reference arm's own record from this
            group. **Only one group may do so.** A group's valid pair set is the
            intersection of the arm's mask with its comparators', so the arm's
            rho differs between groups: on Protein the ``agm_cam`` group holds
            595 pairs against the others' 161,596, and an arm record emitted
            there describes the arm on ``agm_cam``'s 6 % of the data. The caller
            emits the arm once, from its own unrestricted pairs.

    Returns:
        One record per (comparator, reference), plus the arm's when *emit_arm*.
    """
    matrices: dict[str, npt.NDArray[Any]] = {"__arm__": arm.distance}
    defined: dict[str, npt.NDArray[Any]] = {"__arm__": arm.defined}
    for index, comparator in enumerate(group.comparators):
        matrices[f"c{index}"] = comparator.distance
        defined[f"c{index}"] = comparator.defined
    matrices.update(references)

    nulls_defined = view != "equal_n"
    if nulls_defined:
        matrices["__arm_null__"] = arm.size_null
        for index, comparator in enumerate(group.comparators):
            matrices[f"n{index}"] = comparator.size_null

    restriction = _view_mask(view, arm.node_counts)
    if restriction is not None:
        defined["__view__"] = restriction

    variables = PairVariables.from_matrices(matrices, defined=defined)
    if variables.n_pairs < 32:
        LOGGER.warning(
            "%s/%s/%s: %d usable pairs, below the floor; skipped",
            suite,
            dataset,
            view,
            variables.n_pairs,
        )
        return []

    specs: list[CorrelationSpec] = []
    differences: list[DifferenceSpec] = []
    for reference in references:
        specs.append(CorrelationSpec(f"arm@{reference}", "__arm__", reference))
        if nulls_defined:
            specs.append(CorrelationSpec(f"armnull@{reference}", "__arm_null__", reference))
            differences.append(
                DifferenceSpec(f"armexcess@{reference}", f"arm@{reference}", f"armnull@{reference}")
            )
        for index in range(len(group.comparators)):
            specs.append(CorrelationSpec(f"c{index}@{reference}", f"c{index}", reference))
            differences.append(
                DifferenceSpec(
                    f"diff{index}@{reference}", f"arm@{reference}", f"c{index}@{reference}"
                )
            )
            if nulls_defined:
                specs.append(CorrelationSpec(f"n{index}@{reference}", f"n{index}", reference))
                differences.append(
                    DifferenceSpec(
                        f"excess{index}@{reference}",
                        f"c{index}@{reference}",
                        f"n{index}@{reference}",
                    )
                )

    tier = bootstrap_tier(dataset, suite)
    results, diffs = bootstrap_associations(
        variables, specs, tier, differences=differences, seed=SEED, replicates=replicates
    )

    records: list[RhoRecord] = []
    for reference in references:
        if emit_arm:
            records.append(
                _arm_record(
                    suite=suite,
                    dataset=dataset,
                    view=view,
                    reference=reference,
                    arm=arm,
                    variables=variables,
                    results=results,
                    diffs=diffs,
                    tier=tier.as_dict(),
                    nulls_defined=nulls_defined,
                    digest=group.digest,
                )
            )
        for index, comparator in enumerate(group.comparators):
            records.append(
                _comparator_record(
                    suite=suite,
                    dataset=dataset,
                    view=view,
                    reference=reference,
                    comparator=comparator,
                    index=index,
                    variables=variables,
                    results=results,
                    diffs=diffs,
                    tier=tier.as_dict(),
                    nulls_defined=nulls_defined,
                )
            )
    return records


def _arm_record(
    *,
    suite: str,
    dataset: str,
    view: str,
    reference: str,
    arm: ArmMatrices,
    variables: PairVariables,
    results: dict[str, Any],
    diffs: dict[str, Any],
    tier: dict[str, Any],
    nulls_defined: bool,
    digest: str,
) -> RhoRecord:
    """Build the reference arm's record for one reference matrix."""
    outcome = results[f"arm@{reference}"]
    null = results.get(f"armnull@{reference}")
    excess = diffs.get(f"armexcess@{reference}")
    return RhoRecord(
        suite=suite,
        dataset=dataset,
        representation=arm.representation,
        metric=arm.metric,
        reference=reference,
        view=view,
        rho=outcome.rho.as_dict(),
        tau_b=outcome.tau_b,
        n_pairs=outcome.n_pairs,
        n_graphs=variables.n_graphs,
        null_rho=None if null is None else null.rho.as_dict(),
        null_undefined_reason=None if nulls_defined else EQUAL_N_NULL_REASON,
        excess_over_null=None if excess is None else excess.interval.as_dict(),
        regime=_reference_regime(reference),
        tier={**tier, "mask_group": digest},
    )


def _comparator_record(
    *,
    suite: str,
    dataset: str,
    view: str,
    reference: str,
    comparator: ArmMatrices,
    index: int,
    variables: PairVariables,
    results: dict[str, Any],
    diffs: dict[str, Any],
    tier: dict[str, Any],
    nulls_defined: bool,
) -> RhoRecord:
    """Build one comparator's record for one reference matrix."""
    outcome = results[f"c{index}@{reference}"]
    null = results.get(f"n{index}@{reference}")
    excess = diffs.get(f"excess{index}@{reference}")
    difference = diffs[f"diff{index}@{reference}"]
    shared_arm = results[f"arm@{reference}"]
    in_family = comparator.representation in FAMILY_COMPARATORS
    row = None
    if in_family:
        row = "B1e" if reference == "exact" else "B1a"
    return RhoRecord(
        suite=suite,
        dataset=dataset,
        representation=comparator.representation,
        metric=comparator.metric,
        reference=reference,
        view=view,
        rho=outcome.rho.as_dict(),
        tau_b=outcome.tau_b,
        n_pairs=outcome.n_pairs,
        n_graphs=variables.n_graphs,
        null_rho=None if null is None else null.rho.as_dict(),
        null_undefined_reason=None if nulls_defined else EQUAL_N_NULL_REASON,
        excess_over_null=None if excess is None else excess.interval.as_dict(),
        difference_vs_reference_arm=difference.interval.as_dict(),
        p_value=difference.p_value,
        reference_arm_rho_on_shared_pairs=shared_arm.rho.point,
        row=row,
        in_family=in_family,
        regime=_reference_regime(reference),
        tier=tier,
    )


def run_b_rows(
    *,
    distances: Path,
    ged_root: Path,
    approx_root: Path,
    suite: str,
    views: Sequence[str],
    replicates: int | None,
    only: frozenset[str] | None = None,
) -> list[RhoRecord]:
    """Run every B-row correlation for one suite, in both views.

    Args:
        distances: The ``distances/`` tree.
        ged_root: Suite-1 exact matrices.
        approx_root: The ``APPROX_GED`` root.
        suite: Suite key.
        views: Pair views to compute.
        replicates: Override for the frozen tier effort.

    Returns:
        Every rho record for the suite.
    """
    datasets = _select(SUITE1 if suite == "suite1" else SUITE2, only)
    wanted = (*FAMILY_COMPARATORS, *DESCRIPTIVE_COMPARATORS)
    records: list[RhoRecord] = []
    for dataset in datasets:
        arm = load_arm(distances, suite, dataset, REFERENCE_ARM)
        if arm is None:
            LOGGER.warning("%s/%s: no reference arm, B rows skipped", suite, dataset)
            continue
        references = load_references(suite, dataset, arm.graph_ids, ged_root, approx_root)
        if not references:
            LOGGER.warning("%s/%s: no ground-truth GED, B rows skipped", suite, dataset)
            continue
        comparators = [
            loaded
            for name in wanted
            if (loaded := load_arm(distances, suite, dataset, name, target_ids=arm.graph_ids))
        ]
        groups = _group_by_mask(comparators)
        for view in views:
            # The arm's OWN record, on its own unrestricted pairs. Emitted from a
            # comparator-free group because every real group intersects the arm's
            # mask with its comparators' -- on Protein the agm_cam group holds 595
            # pairs against 161,596, and an arm rho taken from there describes the
            # arm on 6 % of the data while looking like the headline number.
            records.extend(
                run_correlation_group(
                    suite=suite,
                    dataset=dataset,
                    view=view,
                    arm=arm,
                    group=CorrelationGroup("arm_only", ()),
                    references=references,
                    replicates=replicates,
                    emit_arm=True,
                )
            )
            for group in groups:
                started = time.monotonic()
                records.extend(
                    run_correlation_group(
                        suite=suite,
                        dataset=dataset,
                        view=view,
                        arm=arm,
                        group=group,
                        references=references,
                        replicates=replicates,
                    )
                )
                LOGGER.info(
                    "%s/%-16s %-9s group[%s] %d arms in %6.1f s",
                    suite,
                    dataset,
                    view,
                    group.digest[:6],
                    len(group.comparators),
                    time.monotonic() - started,
                )
    return records


def _size_beta(fit: MrmResult) -> float:
    """Return the standardised size coefficient beside which beta1 must travel.

    Args:
        fit: A D4 fit.

    Returns:
        The ``delta_n`` coefficient, or ``nan`` when the predictor is absent.
    """
    betas = dict(zip(fit.predictors, fit.betas, strict=False))
    return float(betas.get("delta_n", float("nan")))


def run_mrm_rows(
    *,
    distances: Path,
    encodings: Path,
    ged_root: Path,
    approx_root: Path,
    suite: str,
    replicates: int | None,
    permutations: int | None,
    only: frozenset[str] | None = None,
) -> dict[tuple[str, str], MrmResult]:
    """Run D4's MRM per dataset, on ``all_pairs`` only.

    Args:
        distances: The ``distances/`` tree.
        encodings: The ``encodings/`` tree, for the density predictor.
        ged_root: Suite-1 exact matrices.
        approx_root: The ``APPROX_GED`` root.
        suite: Suite key.
        replicates: Override for the frozen tier effort.
        permutations: Override for the frozen permutation count.

    Returns:
        The fits, keyed by ``(dataset, reference)``.
    """
    datasets = _select(SUITE1 if suite == "suite1" else SUITE2, only)
    fits: dict[tuple[str, str], MrmResult] = {}
    for dataset in datasets:
        arm = load_arm(distances, suite, dataset, REFERENCE_ARM)
        if arm is None:
            continue
        references = load_references(suite, dataset, arm.graph_ids, ged_root, approx_root)
        edges = edge_counts_for(encodings, suite, dataset, arm.graph_ids)
        if not references or edges is None:
            LOGGER.warning("%s/%s: MRM skipped, no reference or no edge counts", suite, dataset)
            continue
        levenshtein = np.array(arm.distance, dtype=np.float64, copy=True)
        levenshtein[~arm.defined] = np.inf
        predictors = {
            "levenshtein": levenshtein,
            "delta_n": delta_n_matrix(arm.node_counts),
            "delta_density": delta_density_matrix(arm.node_counts, edges),
        }
        tier = bootstrap_tier(dataset, suite)
        for reference, matrix in references.items():
            started = time.monotonic()
            fits[dataset, reference] = mrm(
                matrix,
                predictors,
                tier,
                seed=SEED,
                replicates=replicates,
                n_permutations=permutations,
            )
            LOGGER.info(
                # beta1 never travels without beta_size -- INCLUDING here. This
                # log line is the narrowest common point for every consumer that
                # reads a log, and it was emitting a bare beta1 that briefly read
                # as a clean win when a log-sourced set reached the summary.
                # Guarding each reader is four fixes; guarding the producer is one.
                "%s/%-16s MRM@%-5s beta1=%+.4f beta_size=%+.4f p=%.5f in %6.1f s",
                suite,
                dataset,
                reference,
                fits[dataset, reference].beta1,
                _size_beta(fits[dataset, reference]),
                fits[dataset, reference].beta1_permutation_p,
                time.monotonic() - started,
            )
    return fits


@dataclass
class ClaimARecord:
    """One A1 cell: the intersection-union test over both bit conventions.

    Attributes:
        dataset: Dataset key.
        representation: The comparator.
        arm: ``primary`` or ``complete_case``.
        n_graphs: Graphs both arms encoded.
        marginal_p: Raw Wilcoxon p per convention.
        marginal_statistic: Wilcoxon statistic per convention.
        median_difference: Median ``competitor - reference`` per convention;
            positive means IsalGraph is shorter.
        fraction_isalgraph_shorter: Per convention.
        iut_p: ``max`` of the marginals --- the reported p-value.
        discordant: Whether the two conventions disagree in direction.
    """

    dataset: str
    representation: str
    arm: str
    n_graphs: int
    marginal_p: dict[str, float]
    marginal_statistic: dict[str, float]
    median_difference: dict[str, float]
    fraction_isalgraph_shorter: dict[str, float]
    iut_p: float
    discordant: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "dataset": self.dataset,
            "representation": self.representation,
            "arm": self.arm,
            "n_graphs": self.n_graphs,
            "marginal_p": self.marginal_p,
            "marginal_statistic": self.marginal_statistic,
            "median_difference": self.median_difference,
            "fraction_isalgraph_shorter": self.fraction_isalgraph_shorter,
            "iut_p": self.iut_p,
            "conventions_discordant": self.discordant,
            "test": "Wilcoxon signed-rank per convention; intersection-union p = max",
        }


def run_claim_a(
    *,
    encodings: Path,
    suite: str,
    arm: str = "primary",
    only: frozenset[str] | None = None,
) -> tuple[list[ClaimARecord], dict[str, dict[str, float]]]:
    """Run A1 for every ``(dataset, comparator)`` and collect A2's blocks.

    Args:
        encodings: The ``encodings/`` tree.
        suite: Suite key.
        arm: ``primary`` (D14 graphs retained with their fallback string) or
            ``complete_case``.

    Returns:
        ``(records, medians)`` where *medians* maps dataset to the median bit
        count per representation, the block scores A2 ranks.
    """
    datasets = _select(SUITE2 if suite == "suite2" else SUITE1, only)
    records: list[ClaimARecord] = []
    medians: dict[str, dict[str, float]] = {}
    for dataset in datasets:
        reference = load_encodings(encodings, suite, dataset, REFERENCE_ARM)
        if reference is None:
            LOGGER.warning("%s/%s: no reference encoding, A1 skipped", suite, dataset)
            continue
        block: dict[str, float] = {}
        usable = reference.usable(arm)
        block[REFERENCE_ARM] = float(np.median(reference.bits["entropy_bits"][usable]))
        for name in CLAIM_A_COMPARATORS:
            competitor = load_encodings(encodings, suite, dataset, name)
            if competitor is None:
                continue
            record = _claim_a_cell(reference, competitor, dataset, name, arm)
            if record is None:
                continue
            records.append(record)
            competitor_usable = competitor.usable(arm)
            if competitor_usable.any():
                block[name] = float(np.median(competitor.bits["entropy_bits"][competitor_usable]))
        medians[dataset] = block
    return records, medians


def _claim_a_cell(
    reference: Any, competitor: Any, dataset: str, representation: str, arm: str
) -> ClaimARecord | None:
    """Run one A1 cell's intersection-union test."""
    marginal_p: dict[str, float] = {}
    statistic: dict[str, float] = {}
    median: dict[str, float] = {}
    fraction: dict[str, float] = {}
    n_graphs = 0
    for convention in BIT_CONVENTIONS:
        ref_bits, com_bits = paired_bits(reference, competitor, convention, arm)
        if ref_bits.size == 0:
            return None
        n_graphs = int(ref_bits.size)
        stat, p_value = wilcoxon_pair(com_bits, ref_bits)
        statistic[convention] = stat
        marginal_p[convention] = p_value
        median[convention] = float(np.median(com_bits - ref_bits))
        fraction[convention] = float(np.mean(ref_bits < com_bits))
    directions = {float(np.sign(value)) for value in median.values()}
    return ClaimARecord(
        dataset=dataset,
        representation=representation,
        arm=arm,
        n_graphs=n_graphs,
        marginal_p=marginal_p,
        marginal_statistic=statistic,
        median_difference=median,
        fraction_isalgraph_shorter=fraction,
        iut_p=max(marginal_p.values()),
        discordant=len(directions) > 1,
    )


def _complete_blocks(
    medians: dict[str, dict[str, float]],
) -> tuple[npt.NDArray[Any], tuple[str, ...], tuple[str, ...]]:
    """Return the complete block design A2 and B2 need, and what it dropped.

    Friedman is a complete-block test. ``c`` leaves ``agm_cam`` and ``min_dfs``
    without an encoding on some datasets, so a method missing anywhere is
    dropped rather than the datasets that carry it: dropping methods keeps ten
    blocks, dropping datasets would leave four.

    Args:
        medians: Dataset to representation to score.

    Returns:
        ``(scores, methods, dropped)``.
    """
    datasets = tuple(medians)
    everywhere = [
        name
        for name in {name for block in medians.values() for name in block}
        if all(name in medians[dataset] for dataset in datasets)
    ]
    methods = tuple(sorted(everywhere))
    dropped = tuple(sorted({name for block in medians.values() for name in block} - set(methods)))
    scores = np.array(
        [[medians[dataset][name] for name in methods] for dataset in datasets], dtype=np.float64
    )
    return scores, methods, dropped


def _metadata(out_dir: Path, extra: dict[str, Any]) -> dict[str, Any]:
    """Return the wave's mandatory metadata block (``CONTRACTS.md`` 5)."""
    import isalgraph

    def _rev(path: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):  # pragma: no cover - provenance only
            return "unknown"

    return {
        "schema_version": "t06.1",
        "ticket": "T-06",
        "wave": "2026-08-16-t06-recompute",
        "generated_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "isalgraph_engine": isalgraph.engine(),
        "isalgraph_build_hash": isalgraph.build_info()["build_hash"],
        "code_commit": _rev(str(Path(__file__).resolve().parents[3])),
        "src_commit": _rev("/home/mpascual/research/code/IsalGraph"),
        "encode_budget_s": 300.0,
        "out_dir": str(out_dir),
        **extra,
    }


def assemble_p_values(
    claim_a: Sequence[dict[str, Any]],
    rho_records: Sequence[dict[str, Any]],
    mrm_fits: dict[str, dict[str, Any]],
    inputs: ReductionInputs,
    *,
    view: str,
) -> dict[Cell, float]:
    """Map each admissible F2 cell to its p-value.

    A cell with no p-value is reported as missing and **stays in the BH
    denominator**: shrinking ``N_actual`` further than the data force is the
    anti-conservative direction.

    Args:
        claim_a: A1 records, as emitted dicts.
        rho_records: Every rho record, as emitted dicts.
        mrm_fits: MRM fits keyed ``"{dataset}@{reference}"``.
        inputs: The frozen reduction terms.
        view: Which pair view supplies the B-row p-values.

    Returns:
        The p-value per admissible cell.
    """
    admissible = set(admissible_cells(inputs).admissible)
    values: dict[Cell, float] = {}
    for record in claim_a:
        cell = Cell("A1", "suite2", record["dataset"], record["representation"])
        if cell in admissible and record["arm"] == "primary":
            values[cell] = float(record["iut_p"])
    for record in rho_records:
        if record["row"] != "B1e" or record["view"] != view or record["p_value"] is None:
            continue
        cell = Cell("B1e", "suite1", record["dataset"], record["representation"])
        if cell in admissible:
            values[cell] = float(record["p_value"])
    for key, fit in mrm_fits.items():
        dataset, _, reference = key.rpartition("@")
        if reference != "exact":
            continue
        cell = Cell("B3e", "suite1", dataset, None)
        if cell in admissible:
            values[cell] = float(fit["beta1_permutation_p"])
    return values


def campaign_status(partial_dirs: Sequence[Path], logs: Path | None) -> dict[str, Any]:
    """Report progress counted from **artifacts**, never from progress signals.

    Three times in one session an observer counted a log line, a running
    process or a partially written set and reported a cell as complete when it
    was not: a shard's log carried a finished ``MRM@`` line while its partial
    was still unwritten, and the count that resulted was both plausible and
    wrong. The handoff's rule already covered it --- *confirm a run against its
    output file, never against a process list* --- so the failure was not an
    unknown rule but an unmechanised one.

    This is the mechanism. A cell is **landed** iff its partial exists on disk.
    Anything visible only in a log is reported under ``in_flight`` and can never
    be added to the landed count by accident, because the two never share a
    field.

    **Without a log root the in-flight state is unknowable, and this says so
    rather than guessing.** An earlier version reported every unlanded cell as
    ``not started`` when ``logs`` was ``None`` --- which is a claim the function
    cannot support, and a dangerous one: "not started" on a shard that is 90 %
    through its MRM reads as a dead shard, and the correct response to a dead
    shard is to relaunch it. That is the over-accepting-consumer defect
    reappearing inside the tool written to prevent it.

    Args:
        partial_dirs: Directories of shard partials.
        logs: Shard-log directory, or ``None``.

    Returns:
        ``landed``, ``in_flight`` and the expected total.
    """
    landed: set[str] = set()
    for directory in partial_dirs:
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            landed.add(path.stem)

    expected = {f"suite1__{d}" for d in SUITE1} | {f"suite2__{d}" for d in SUITE2}
    in_flight: dict[str, str] = {}
    logs_readable = logs is not None and logs.is_dir() and any(logs.glob("f2_suite*.log"))
    if not logs_readable:
        LOGGER.warning(
            "no readable shard logs at %s: the in-flight state of the %d unlanded cells is "
            "UNKNOWN, not 'not started'",
            logs,
            len(expected - landed),
        )
    if logs_readable and logs is not None:
        for path in sorted(logs.glob("f2_suite*.log")):
            stem = path.stem[len("f2_") :]
            if stem in landed:
                continue
            text = path.read_text(errors="replace")
            groups = text.count("group[")
            mrms = text.count("MRM@")
            in_flight[stem] = f"{groups} bootstrap group(s), {mrms} MRM(s) reported, NOT landed"

    return {
        "landed": sorted(landed),
        "n_landed": len(landed),
        "n_expected": len(expected),
        "missing": sorted(expected - landed),
        "in_flight": in_flight,
        "in_flight_state_known": logs_readable,
        "not_started": sorted(expected - landed - set(in_flight)) if logs_readable else [],
        "unknown_state": [] if logs_readable else sorted(expected - landed),
        "rule": (
            "A cell is landed iff its partial exists. A log line is a progress signal, not "
            "an artifact, and is never counted as completion. Without a readable log root "
            "the in-flight state is reported as UNKNOWN, never as 'not started'."
        ),
    }


def provenance_markdown(out_dir: Path, family: Path | None) -> str:
    """Render the archive's ``PROVENANCE.md``.

    **Generated, never hand-written.** A provenance page maintained by hand
    drifts from the run it claims to describe, and a reader six months out has
    no way to tell. Every value here is read from the artifact or the live
    environment at the moment of archiving.

    Args:
        out_dir: Where the report tree is being written.
        family: ``family_F2.json``, for the reduction terms.

    Returns:
        Markdown.
    """
    meta = _metadata(out_dir, {})
    card: dict[str, Any] = {}
    if family is not None and family.exists():
        card = json.loads(family.read_text()).get("cardinality", {})

    lines = [
        "# T-06 — provenance",
        "",
        "Generated at archive time from the artifacts and the live environment. "
        "Nothing here is transcribed by hand.",
        "",
        "| | |",
        "|---|---|",
        f"| generated | `{meta['generated_utc']}` |",
        f"| code_commit | `{meta['code_commit']}` |",
        f"| src_commit | `{meta['src_commit']}` |",
        f"| engine | `{meta['isalgraph_engine']}` |",
        f"| build_hash | `{meta['isalgraph_build_hash']}` |",
        f"| seed | `{meta['seed']}` |",
        f"| encode_budget_s | `{meta['encode_budget_s']}` (F-3, killed subprocess) |",
        "",
        "## The confirmatory family",
        "",
        "| term | value | source |",
        "|---|---|---|",
        f"| `N_actual` | **{card.get('n_actual', 'n/a')}** | enumeration of the admissible "
        "cell set — **this is the definition** |",
        f"| closed form | {card.get('closed_form', 'n/a')} | "
        f"`{card.get('closed_form_expression', 'n/a')}`, printed as a check only |",
        f"| discrepancy | {card.get('discrepancy', 'n/a')} | enumeration minus closed form; "
        "the enumeration wins on disagreement |",
        f"| `k` | {card.get('k', 'n/a')} | representations with no admissible distance "
        "(`competitors.md` 3.4), F5-blind |",
        f"| `d` | {card.get('d', 'n/a')} | **not applied** — F0's majority branch fired, so "
        "F1 removes nothing from a family that no longer contains its rows |",
        f"| `c` | {card.get('c', 'n/a')} | cells whose representation falls below 99 % "
        "completion at the 300 s budget |",
        f"| `N_max` | {card.get('n_max', 'n/a')} | `preregistration.md` 4.2 |",
        "",
        "## Cohorts",
        "",
        "| | graphs | pairs |",
        "|---|---:|---:|",
        "| Suite 1 (`n <= 12`, exact GED) | 5,350 | 3,897,911 |",
        "| Suite 2 (bracketed GED) | 16,370 | 21,710,892 |",
        "",
        "Both counts are gated in `data/gates/gate_T06_reproduction.json`; the per-dataset "
        "pair accounting is in `data/ladder.json` and `data/ladder_suite1.json`, whose "
        "`analysed` rung equals the `n_pairs` behind every rho in `data/rho_table.json`.",
        "",
        "## Reading order",
        "",
        "`REPORT.md` first — it opens with a five-line answer. `T-06-FRAMING.md` records the "
        "framings that were **measured and rejected**, which is the part that saves the next "
        "person work. `data/` holds the analysis JSONs; the encodings and distance matrices "
        "they were computed from stay under `data/source/T06/` and are deliberately not "
        "copied here.",
        "",
    ]
    return "\n".join(lines)


def collinearity_report(distances: Path, encodings: Path) -> dict[str, Any]:
    """Measure predictor collinearity for D4, per dataset.

    D4 regresses GED on Levenshtein, ``|delta n|`` and ``|delta density|``
    **simultaneously**, and reads the comparison between the first two
    coefficients as the ticket's central finding. That reading requires the two
    to be separately identifiable. Where they are not, the split between them is
    arbitrary within a wide equivalence class and no amount of significance
    rescues it --- a fit can have R^2 = 0.998 and still not support "size
    explains 0.91 of it and the string 0.10".

    The signature is not always visible in the coefficients. ``coil_del``'s
    upper-bound fit announces itself (beta_lev = +1.49 with beta_size negative),
    but ``aids_iam``'s lower-bound fit looks perfectly ordinary at +0.10 against
    +0.91 --- and has VIF 15.3. Only the design matrix shows it, so it is
    measured here rather than eyeballed.

    Args:
        distances: The ``distances/`` tree.
        encodings: The ``encodings/`` tree, for the density predictor.

    Returns:
        Per ``suite/dataset``: the predictor correlation, the VIFs, and whether
        the fit is identifiable at the conventional VIF > 10 threshold.
    """
    rows: dict[str, Any] = {}
    for suite, datasets in (("suite1", SUITE1), ("suite2", SUITE2)):
        for dataset in datasets:
            arm = load_arm(distances, suite, dataset, REFERENCE_ARM)
            edges = edge_counts_for(encodings, suite, dataset, arm.graph_ids) if arm else None
            if arm is None or edges is None:
                continue
            upper = np.triu_indices(arm.graph_ids.size, 1)
            lev = arm.distance[upper]
            keep = arm.defined[upper] & np.isfinite(lev) & (lev >= 0.0)
            if int(keep.sum()) < 1000:
                continue
            design = np.column_stack(
                [
                    lev[keep],
                    delta_n_matrix(arm.node_counts)[upper][keep],
                    delta_density_matrix(arm.node_counts, edges)[upper][keep],
                ]
            )
            names = ("levenshtein", "delta_n", "delta_density")
            vifs: dict[str, float] = {}
            for j, name in enumerate(names):
                others = [k for k in range(3) if k != j]
                matrix = np.column_stack([np.ones(design.shape[0]), design[:, others]])
                beta, *_ = np.linalg.lstsq(matrix, design[:, j], rcond=None)
                residual = design[:, j] - matrix @ beta
                total = ((design[:, j] - design[:, j].mean()) ** 2).sum()
                r_squared = 1.0 - (residual @ residual) / total if total > 0 else 1.0
                vifs[name] = float(1.0 / max(1.0 - r_squared, 1e-12))
            worst = max(vifs.values())
            rows[f"{suite}/{dataset}"] = {
                "r_lev_delta_n": float(np.corrcoef(design[:, 0], design[:, 1])[0, 1]),
                "vif": vifs,
                "max_vif": worst,
                "identifiable": bool(worst <= 10.0),
                "n_pairs": int(keep.sum()),
            }
            LOGGER.info(
                "%s/%-16s r(lev,dn)=%+.4f  max VIF=%6.2f  %s",
                suite,
                dataset,
                rows[f"{suite}/{dataset}"]["r_lev_delta_n"],
                worst,
                "identifiable" if worst <= 10.0 else "NOT IDENTIFIABLE",
            )
    return {
        "schema_version": "t06.collinearity.1",
        "threshold": 10.0,
        "note": (
            "D4 compares beta_levenshtein against beta_delta_n. Where VIF exceeds 10 the two "
            "are not separately identifiable and that comparison is not supported, however "
            "high R^2 or however small the p-value."
        ),
        "datasets": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description="Run F2, the primary family.")
    ap.add_argument("--distances", type=Path, default=None)
    ap.add_argument("--encodings", type=Path, default=None)
    ap.add_argument("--completion-rates", type=Path, default=None)
    ap.add_argument("--ged-root", type=Path, default=None)
    ap.add_argument("--approx-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument(
        "--datasets",
        default="",
        help="comma-separated dataset filter; empty means every dataset of the named suites",
    )
    ap.add_argument(
        "--suites",
        default="suite1,suite2",
        help="suite2 supplies A1/A2 and the descriptive B1a/B2/B3a rows",
    )
    ap.add_argument(
        "--replicates", type=int, default=None, help="smoke override; None in production"
    )
    ap.add_argument("--permutations", type=int, default=None, help="smoke override")
    ap.add_argument("--q", type=float, default=FDR_Q)
    ap.add_argument(
        "--emit-partial",
        type=Path,
        default=None,
        help="write this shard's raw records here and stop, for a sharded run",
    )
    ap.add_argument(
        "--collinearity",
        type=Path,
        default=None,
        help="measure D4 predictor collinearity, write it here and stop",
    )
    ap.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="write PROVENANCE.md here and stop",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="report campaign progress counted from partials, never from log lines",
    )
    ap.add_argument("--logs", type=Path, default=None, help="shard logs, for the status report")
    ap.add_argument(
        "--merge-partials",
        type=Path,
        default=None,
        help="assemble the family from every partial JSON in this directory",
    )
    ap.add_argument(
        "--ged-zero-probes",
        default="suite1/iam_letter_low,suite2/iam_letter_low,suite2/protein",
        help=(
            "suite/dataset pairs for the GED = 0 tie sensitivity; two contrasting datasets, "
            "one where the zero block is large and one where it is negligible"
        ),
    )
    return ap


def descriptive_equal_n(rho_records: Sequence[dict[str, Any]], q: float) -> dict[str, Any]:
    """Summarise the equal-``n`` arm as a descriptive block, not a BH column.

    The confirmatory view is ``all_pairs`` (design note 18.9). This arm is
    reported with point estimates, graph-level intervals and a **locally**
    adjusted q over its own records only. It is deliberately not named ``bh_*``
    and not placed beside the family's BH columns: three columns side by side is
    the shape a reader takes for three families, and this revision cannot carry
    that reading.

    Args:
        rho_records: Every rho record; the equal-``n`` ones are selected here.
        q: The false discovery rate used for the local adjustment.

    Returns:
        The descriptive block.
    """
    selected = [
        record
        for record in rho_records
        if record["view"] == "equal_n" and record["p_value"] is not None and record["in_family"]
    ]
    local = benjamini_hochberg(
        [float(record["p_value"]) for record in selected],
        family="descriptive equal_n arm, locally adjusted",
        q=q,
    )
    return {
        "status": "DESCRIPTIVE. Not a confirmatory family and not a BH column of F2.",
        "why": (
            "preregistration section 4 names no pair view; design note 18.9 rules the "
            "confirmatory view is all_pairs on three grounds, the strongest being that A9's "
            "pair-accounting ladder carries no equal-n rung. This arm is reported because in "
            "an equal-n stratum the size null is identically constant, so raw rho is the "
            "structural signal (design note 16.3)."
        ),
        "size_null": EQUAL_N_NULL_REASON,
        "n_records": len(selected),
        "local_fdr_adjustment": local.as_dict(),
        "records": [
            {
                key: record[key]
                for key in (
                    "suite",
                    "dataset",
                    "representation",
                    "reference",
                    "rho",
                    "difference_vs_reference_arm",
                    "p_value",
                )
            }
            for record in selected
        ],
    }


def ged_zero_sensitivity(
    *,
    distances: Path,
    ged_root: Path,
    approx_root: Path,
    probes: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Measure what the legitimate GED = 0 pairs do to rho, rather than assert it.

    Those pairs are kept: GED is legitimately 0 for isomorphic graphs, 28.05 %
    of IAM Letter LOW pairs are certified exact at 0, and filtering them would
    truncate the response at its most informative end (``CONTRACTS.md`` 4.1,
    trap 9). But they are a large block of ties and Spearman is tie-sensitive,
    so the size of the effect is measured on two datasets chosen to contrast:
    one where the block is large and one where it is negligible.

    Args:
        distances: The ``distances/`` tree.
        ged_root: Suite-1 exact matrices.
        approx_root: The ``APPROX_GED`` root.
        probes: ``(suite, dataset)`` pairs to probe.

    Returns:
        One record per ``(suite, dataset, reference)``.
    """
    from benchmarks.real_data.eval_stats.association import condensed, spearman

    records: list[dict[str, Any]] = []
    for suite, dataset in probes:
        arm = load_arm(distances, suite, dataset, REFERENCE_ARM)
        if arm is None:
            continue
        references = load_references(suite, dataset, arm.graph_ids, ged_root, approx_root)
        distance = condensed(arm.distance)
        defined = condensed(arm.defined.astype(np.float64)) > 0.5
        for name, matrix in references.items():
            reference = condensed(matrix)
            usable = defined & np.isfinite(reference) & (reference >= 0.0) & np.isfinite(distance)
            positive = usable & (reference > 0.0)
            n_zero = int(np.count_nonzero(usable & ~positive))
            with_zeros = spearman(distance[usable], reference[usable])
            without = (
                spearman(distance[positive], reference[positive])
                if positive.any()
                else float("nan")
            )
            records.append(
                {
                    "suite": suite,
                    "dataset": dataset,
                    "reference": name,
                    "n_pairs_analysed": int(np.count_nonzero(usable)),
                    "n_pairs_ged_zero": n_zero,
                    "fraction_ged_zero": n_zero / max(int(np.count_nonzero(usable)), 1),
                    "rho_with_zeros": with_zeros,
                    "rho_without_zeros": without,
                    "delta": without - with_zeros,
                }
            )
            LOGGER.info(
                "GED=0 sensitivity %s/%-16s @%-5s zeros=%6.2f%% rho %+.4f -> %+.4f (delta %+.4f)",
                suite,
                dataset,
                name,
                100.0 * records[-1]["fraction_ged_zero"],
                with_zeros,
                without,
                records[-1]["delta"],
            )
    return records


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Run every measurement this shard owns and return its raw records.

    Kept separate from assembly so a production run can be sharded per dataset:
    each shard writes its own partial, a failure costs one dataset rather than
    the whole campaign, and the shards run concurrently.

    Args:
        args: Parsed arguments.

    Returns:
        The shard's records, JSON-serialisable.
    """
    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    suites = tuple(s.strip() for s in args.suites.split(",") if s.strip())
    only = frozenset(d.strip() for d in args.datasets.split(",") if d.strip()) or None

    claim_a: list[ClaimARecord] = []
    medians: dict[str, dict[str, float]] = {}
    if "suite2" in suites:
        for arm in ("primary", "complete_case"):
            records, block = run_claim_a(
                encodings=args.encodings, suite="suite2", arm=arm, only=only
            )
            claim_a.extend(records)
            if arm == "primary":
                medians = block

    rho_records: list[RhoRecord] = []
    mrm_fits: dict[str, dict[str, Any]] = {}
    for suite in suites:
        rho_records.extend(
            run_b_rows(
                distances=args.distances,
                ged_root=args.ged_root,
                approx_root=args.approx_root,
                suite=suite,
                views=views,
                replicates=args.replicates,
                only=only,
            )
        )
        for (dataset, reference), fit in run_mrm_rows(
            distances=args.distances,
            encodings=args.encodings,
            ged_root=args.ged_root,
            approx_root=args.approx_root,
            suite=suite,
            replicates=args.replicates,
            permutations=args.permutations,
            only=only,
        ).items():
            mrm_fits[f"{dataset}@{reference}"] = fit.as_dict() | {"suite": suite}

    ged_zero = ged_zero_sensitivity(
        distances=args.distances,
        ged_root=args.ged_root,
        approx_root=args.approx_root,
        probes=tuple(
            (suite, dataset)
            for suite, dataset in (
                pair.split("/", 1) for pair in args.ged_zero_probes.split(",") if pair.strip()
            )
            if suite in suites and (only is None or dataset in only)
        ),
    )
    return {
        "views": list(views),
        "suites": list(suites),
        "datasets": sorted(only) if only else [],
        "a1_cells": [record.as_dict() for record in claim_a],
        "claim_a_medians": medians,
        "rho_rows": [record.as_dict() for record in rho_records],
        "mrm": mrm_fits,
        "ged_zero": ged_zero,
    }


def dedup_rho_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per identifying key, choosing the widest pair set.

    Partials written before the reference arm was emitted once per cell carry
    **several** arm records under the same key --- one per mask group, each on
    that group's intersection. Keeping "the first" makes the surviving value
    depend on emission order, which is how a 595-pair record can stand in for a
    161,596-pair one without anything failing.

    The widest pair set is the unambiguous choice: for the arm it is its own
    unrestricted pairs, and for a comparator every candidate is identical
    anyway, so the rule is a no-op on well-formed input.

    Args:
        rows: Rho rows, possibly with duplicates.

    Returns:
        One row per ``(suite, dataset, representation, reference, view)``.
    """
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row["suite"], row["dataset"], row["representation"], row["reference"], row["view"])
        incumbent = best.get(key)
        if incumbent is None or int(row["n_pairs"]) > int(incumbent["n_pairs"]):
            best[key] = row
    dropped = len(rows) - len(best)
    if dropped:
        LOGGER.info(
            "dropped %d duplicate rho rows, keeping the widest pair set for each cell", dropped
        )
    return list(best.values())


def merge_partials(directory: Path) -> dict[str, Any]:
    """Combine every shard's partial into one record set.

    Args:
        directory: Directory holding ``*.json`` partials.

    Returns:
        The merged records.

    Raises:
        F2DriverError: If the directory holds no partial.
    """
    partials = sorted(directory.glob("*.json"))
    if not partials:
        raise F2DriverError(f"no partials under {directory}")
    merged: dict[str, Any] = {
        "views": [],
        "suites": [],
        "datasets": [],
        "a1_cells": [],
        "claim_a_medians": {},
        "rho_rows": [],
        "mrm": {},
        "ged_zero": [],
        "shards": [],
    }
    for path in partials:
        shard = json.loads(path.read_text())
        merged["a1_cells"].extend(shard["a1_cells"])
        merged["rho_rows"].extend(shard["rho_rows"])
        merged["ged_zero"].extend(shard["ged_zero"])
        merged["mrm"].update(shard["mrm"])
        merged["claim_a_medians"].update(shard["claim_a_medians"])
        for key in ("views", "suites", "datasets"):
            merged[key] = sorted(set(merged[key]) | set(shard[key]))
        merged["shards"].append(path.name)
    merged["rho_rows"] = dedup_rho_rows(merged["rho_rows"])
    LOGGER.info(
        "merged %d partials: %d a1 cells, %d rho rows, %d mrm fits",
        len(partials),
        len(merged["a1_cells"]),
        len(merged["rho_rows"]),
        len(merged["mrm"]),
    )
    return merged


def assemble(collected: dict[str, Any], args: argparse.Namespace) -> int:
    """Turn the collected records into ``family_F2.json`` and ``rho_table.json``.

    Args:
        collected: Records from :func:`collect` or :func:`merge_partials`.
        args: Parsed arguments.

    Returns:
        0 on success.
    """
    inputs, failing = build_reduction_inputs(
        args.completion_rates,
        excluded_representations=EXCLUDED_REPRESENTATIONS,
        f0_demotes_approximate=True,
    )
    rho_rows = collected["rho_rows"]
    medians = collected["claim_a_medians"]

    omnibus: dict[str, tuple[npt.NDArray[Any], Sequence[str], bool]] = {}
    posthoc: dict[str, Any] = {}
    dropped_methods: tuple[str, ...] = ()
    a2_p: float | None = None
    if medians:
        scores, methods, dropped_methods = _complete_blocks(medians)
        omnibus["A2"] = (scores, methods, True)
        posthoc["A2"] = wilcoxon_holm_posthoc(scores, methods).as_dict()
        # A2 is an admissible cell, so it needs a p-value in the BH denominator.
        # run_f2 runs the omnibus itself for reporting; the same test is computed
        # here because the cell's p-value has to exist before BH is applied.
        a2 = friedman_omnibus(scores, methods, Regime.APPROXIMATE, lower_is_better=True)
        a2_p = a2.p_value if a2.ran else None

    p_values = assemble_p_values(
        collected["a1_cells"], rho_rows, collected["mrm"], inputs, view="all_pairs"
    )
    if a2_p is not None:
        p_values[Cell("A2", None, None, None)] = a2_p

    result = run_f2(p_values, inputs, q=args.q, omnibus_scores=omnibus)
    meta = _metadata(
        args.out_dir,
        {
            "views": collected["views"],
            "suites": collected["suites"],
            "shards": collected.get("shards", []),
        },
    )

    payload: dict[str, Any] = {
        "family": "F2",
        "metadata": meta,
        "primary_view": "all_pairs",
        "view_ruling": (
            "design note 18.9: preregistration section 4 names no pair view, and the "
            "confirmatory view is all_pairs on three grounds -- A9's pair-accounting ladder "
            "carries no equal-n rung; section 4.2's 'per Suite-1 dataset' is unqualified, and "
            "the unmarked reading of rho over a dataset is over its pairs; and F0 and F1 have "
            "already run on the full defined pair set, so a family whose gates and rows used "
            "different pair sets would be incoherent."
        ),
        "descriptive_arms": {
            "equal_n_view": descriptive_equal_n(rho_rows, args.q),
            "ged_zero_pairs": {
                "status": "SENSITIVITY. The pairs are kept; this measures what keeping them costs.",
                "why": (
                    "GED is legitimately 0 for isomorphic graphs and filtering those pairs would "
                    "truncate the response at its most informative end (CONTRACTS.md 4.1, "
                    "trap 9). They are nevertheless a large block of ties and Spearman is "
                    "tie-sensitive, so the effect is measured on two contrasting datasets rather "
                    "than assumed small."
                ),
                "records": collected["ged_zero"],
            },
        },
        "claim_a_test": "intersection-union over both bit conventions; p = max (design note 18.8)",
        "claim_a_bit_count_undefined": list(BIT_COUNT_UNDEFINED),
        "a2_methods_dropped_for_complete_blocks": list(dropped_methods),
        "posthoc_holm_not_in_bh": posthoc,
        "mrm_view": "all_pairs",
        "mrm_equal_n_reason": EQUAL_N_MRM_REASON,
        "completion_rows_below_threshold": failing,
        "a1_cells": collected["a1_cells"],
        "mrm": collected["mrm"],
        **result.as_dict(),
    }
    (args.out_dir / "family_F2.json").write_text(json.dumps(payload, indent=2, default=str))
    (args.out_dir / "rho_table.json").write_text(
        json.dumps(
            {
                "metadata": meta,
                "resampling_unit": "graph",
                "n_rows": len(rho_rows),
                "rows": rho_rows,
            },
            indent=2,
            default=str,
        )
    )

    print(
        f"\nF2: N_actual={result.cardinality.n_actual} "
        f"closed_form={result.cardinality.closed_form} "
        f"discrepancy={result.cardinality.discrepancy:+d}"
    )
    print(f"    cells with a p-value: {len(result.cells)} of {result.cardinality.n_actual}")
    print(f"    BH over N_actual : {result.bh_primary.n_rejected} rejected at q={args.q}")
    print(f"    BH over N_max=182: {result.bh_sensitivity.n_rejected} rejected at q={args.q}")
    print(f"    rho records: {len(rho_rows)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        0 on success.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.collinearity is not None:
        report = collinearity_report(args.distances, args.encodings)
        args.collinearity.parent.mkdir(parents=True, exist_ok=True)
        args.collinearity.write_text(json.dumps(report, indent=2))
        bad = [k for k, v in report["datasets"].items() if not v["identifiable"]]
        total = len(report["datasets"])
        print(f"wrote {args.collinearity}: {len(bad)} of {total} NOT identifiable")
        for name in bad:
            print(f"  NOT IDENTIFIABLE {name}  max VIF {report['datasets'][name]['max_vif']:.2f}")
        return 0

    if args.provenance is not None:
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        family = args.out_dir / "family_F2.json"
        args.provenance.write_text(provenance_markdown(args.out_dir, family))
        print(f"wrote {args.provenance}")
        return 0

    if args.status:
        # Default the log root to the campaign's conventional location so a bare
        # --status is complete; the UNKNOWN path still covers a wrong one.
        log_root = args.logs if args.logs is not None else args.out_dir.parent / "logs"
        report = campaign_status(
            [args.out_dir / "f2_partials", args.out_dir / "f2_partials_early"], log_root
        )
        print(f"LANDED {report['n_landed']} of {report['n_expected']} cells (partials on disk)")
        for name in report["landed"]:
            print(f"  landed    {name}")
        for name, detail in sorted(report["in_flight"].items()):
            print(f"  IN FLIGHT {name:<30} {detail}")
        for name in report["not_started"]:
            print(f"  not started {name}")
        if not report["in_flight_state_known"]:
            print(
                f"  !!! in-flight state UNKNOWN for {len(report['unknown_state'])} cells "
                "-- no readable shard logs; pass --logs to see it"
            )
            for name in report["unknown_state"]:
                print(f"  UNKNOWN   {name}")
        print(f"\n{report['rule']}")
        return 0

    if args.merge_partials is not None:
        return assemble(merge_partials(args.merge_partials), args)

    inputs, _ = build_reduction_inputs(
        args.completion_rates,
        excluded_representations=EXCLUDED_REPRESENTATIONS,
        f0_demotes_approximate=True,
    )
    card = admissible_cells(inputs)
    LOGGER.info(
        "N_actual=%d closed_form=%d discrepancy=%+d (k=%d d=%d c=%d)",
        card.n_actual,
        card.closed_form,
        card.discrepancy,
        card.k,
        card.d,
        card.c,
    )

    collected = collect(args)
    if args.emit_partial is not None:
        args.emit_partial.parent.mkdir(parents=True, exist_ok=True)
        args.emit_partial.write_text(json.dumps(collected, indent=2, default=str))
        print(
            f"partial: {len(collected['rho_rows'])} rho rows, "
            f"{len(collected['a1_cells'])} a1 cells, {len(collected['mrm'])} mrm fits "
            f"-> {args.emit_partial}"
        )
        return 0
    return assemble(collected, args)


if __name__ == "__main__":
    raise SystemExit(main())
