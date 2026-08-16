"""D8 / D9 --- across-dataset comparison and multiplicity control.

D8 compares methods across datasets with a Friedman omnibus, a pairwise
Wilcoxon signed-rank post-hoc under Holm, and a critical-difference diagram
(Demsar, *JMLR* 7:1-30, 2006). Two rules constrain it and both are enforced
here rather than left to the caller:

* **The exact and approximate regimes are never mixed in one omnibus.**
* **The exact regime gets no omnibus at all.** It has five datasets; Friedman
  at ``N = 5`` separates almost nothing, and an underpowered figure dressed as
  a result is worse than no figure (``statistics.md`` section 4, locked).
  :func:`friedman_omnibus` refuses the exact regime by construction and returns
  the reason instead.

D9 applies Benjamini-Hochberg at ``q = 0.05`` **within each declared family
separately**, over ``N_actual``, and emits the BH-over-``N_max`` sensitivity
column beside it. The Wilcoxon-Holm post-hoc is deliberately **not** a BH
member: Holm already controls the FWER within the post-hoc set, and nesting it
inside BH would correct twice.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from benchmarks.real_data.eval_correlation.correlation_metrics import holm_bonferroni
from benchmarks.real_data.eval_stats.resampling import FDR_Q, FloatArray

LOGGER = logging.getLogger(__name__)

__all__ = [
    "BenjaminiHochbergResult",
    "CriticalDifference",
    "FriedmanResult",
    "MultiplicityError",
    "PostHocResult",
    "Regime",
    "benjamini_hochberg",
    "critical_difference",
    "fcr_adjusted_level",
    "friedman_omnibus",
    "wilcoxon_holm_posthoc",
]


class MultiplicityError(Exception):
    """Raised when a multiplicity request violates a locked rule."""


class Regime(Enum):
    """Which GED regime a comparison sits in.

    Attributes:
        EXACT: Suite 1, five datasets, exact GED. **Descriptive only** --- no
            omnibus and no critical-difference diagram.
        APPROXIMATE: Suite 2, ten datasets, bracketed GED. Carries D8's
            omnibus and CD diagram.
    """

    EXACT = "exact"
    APPROXIMATE = "approximate"


# ---------------------------------------------------------------------------
# D9 --- Benjamini-Hochberg
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenjaminiHochbergResult:
    """BH-adjusted p-values for one declared family.

    Attributes:
        family: The family name, e.g. ``"F0"``, ``"F1"``, ``"F2"``.
        p_values: Raw p-values in input order.
        adjusted: BH-adjusted p-values in input order, over ``m``.
        rejected: Whether each test is rejected at ``q``.
        m: The denominator actually used; ``N_actual`` for the primary column.
        q: The false discovery rate.
        n_rejected: Number of rejections.
    """

    family: str
    p_values: tuple[float, ...]
    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]
    m: int
    q: float

    @property
    def n_rejected(self) -> int:
        """Number of rejected hypotheses."""
        return int(sum(self.rejected))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "family": self.family,
            "m": self.m,
            "q": self.q,
            "n_tests": len(self.p_values),
            "n_rejected": self.n_rejected,
            "p_values": list(self.p_values),
            "adjusted": list(self.adjusted),
            "rejected": list(self.rejected),
        }


def benjamini_hochberg(
    p_values: Sequence[float],
    *,
    family: str = "unnamed",
    m: int | None = None,
    q: float = FDR_Q,
) -> BenjaminiHochbergResult:
    """Return BH step-up adjusted p-values (Benjamini & Hochberg, 1995).

    The adjusted value of the ``i``-th smallest p-value is
    ``min_{j >= i} ( m / j * p_(j) )``, clipped at 1. The running minimum from
    the largest rank downwards is what enforces monotonicity, and omitting it
    is where naive implementations break: without it an adjusted value can
    exceed one computed from a larger raw p-value.

    Args:
        p_values: Raw p-values. ``nan`` entries are carried through unadjusted
            and never rejected.
        family: Family name recorded on the result.
        m: Denominator. Defaults to the number of finite p-values, which is
            ``N_actual`` when the caller passes exactly the admissible family.
            Pass ``N_max`` to obtain the pre-registered sensitivity column.
        q: False discovery rate; 0.05 as pre-registered.

    Returns:
        The adjusted p-values and rejection flags in input order.

    Raises:
        MultiplicityError: If *q* is outside ``(0, 1)`` or *m* is below the
            number of finite p-values, which would be an invalid denominator.
    """
    if not 0.0 < q < 1.0:
        raise MultiplicityError(f"q must lie in (0, 1), got {q}")
    raw = np.asarray(list(p_values), dtype=np.float64)
    finite = np.isfinite(raw)
    n_finite = int(np.count_nonzero(finite))
    denominator = n_finite if m is None else int(m)
    if denominator < n_finite:
        raise MultiplicityError(
            f"BH denominator {denominator} is below the {n_finite} finite p-values supplied"
        )

    adjusted = np.full(raw.shape, np.nan, dtype=np.float64)
    if n_finite:
        idx = np.flatnonzero(finite)
        order = idx[np.argsort(raw[idx], kind="stable")]
        ranks = np.arange(1, n_finite + 1, dtype=np.float64)
        scaled = raw[order] * denominator / ranks
        monotone = np.minimum.accumulate(scaled[::-1])[::-1]
        adjusted[order] = np.minimum(monotone, 1.0)

    rejected = np.zeros(raw.shape, dtype=bool)
    rejected[finite] = adjusted[finite] <= q
    return BenjaminiHochbergResult(
        family=family,
        p_values=tuple(float(v) for v in raw),
        adjusted=tuple(float(v) for v in adjusted),
        rejected=tuple(bool(v) for v in rejected),
        m=denominator,
        q=q,
    )


def fcr_adjusted_level(n_rejected: int, m: int, q: float = FDR_Q) -> float:
    """Return the coverage of a BH-adjusted (FCR) confidence interval.

    The pre-registration's gate rules say "its BH-adjusted CI excludes 0". A
    percentile interval carries no multiplicity control on its own; the
    standard object that does is the false-coverage-statement-rate interval of
    Benjamini & Yekutieli (*JASA* 100(469):71-81, 2005): after BH selects ``R``
    of ``m`` hypotheses, intervals for the selected parameters are built at
    coverage ``1 - R q / m``, which controls the FCR at ``q``.

    Args:
        n_rejected: ``R``, the number of BH rejections in the family.
        m: The BH denominator.
        q: The false discovery rate.

    Returns:
        Coverage in ``(0, 1)``. With no rejection the level degenerates to 1,
        so it is floored at ``1 - q`` --- an unselected parameter gets the
        ordinary marginal interval.

    Raises:
        MultiplicityError: If *m* is not positive.
    """
    if m <= 0:
        raise MultiplicityError(f"the BH denominator must be positive, got {m}")
    if n_rejected <= 0:
        return 1.0 - q
    return float(1.0 - min(1.0, n_rejected * q / m))


# ---------------------------------------------------------------------------
# D8 --- Friedman, Wilcoxon-Holm, critical difference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FriedmanResult:
    """A Friedman omnibus, or the recorded refusal to run one.

    Attributes:
        methods: Method names in column order.
        statistic: Friedman chi-square, ``nan`` when refused.
        p_value: Omnibus p-value, ``nan`` when refused.
        average_ranks: Mean rank per method, lower is better.
        n_datasets: Blocks behind the test.
        ran: Whether the omnibus was computed.
        refusal_reason: Why it was not, when ``ran`` is ``False``.
    """

    methods: tuple[str, ...]
    statistic: float
    p_value: float
    average_ranks: tuple[float, ...]
    n_datasets: int
    ran: bool
    refusal_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "methods": list(self.methods),
            "statistic": self.statistic,
            "p_value": self.p_value,
            "average_ranks": list(self.average_ranks),
            "n_datasets": self.n_datasets,
            "ran": self.ran,
            "refusal_reason": self.refusal_reason,
        }


def _rank_blocks(scores: npt.NDArray[Any], lower_is_better: bool) -> FloatArray:
    """Rank methods within each dataset block, 1 = best."""
    array = np.asarray(scores, dtype=np.float64)
    signed = array if lower_is_better else -array
    return np.asarray(
        np.vstack([stats.rankdata(row) for row in signed]),
        dtype=np.float64,
    )


def friedman_omnibus(
    scores: npt.NDArray[Any],
    methods: Sequence[str],
    regime: Regime,
    *,
    lower_is_better: bool = True,
) -> FriedmanResult:
    """Run D8's Friedman omnibus, refusing the exact regime.

    ``statistics.md`` section 4 locks the omnibus and critical-difference
    diagram to the ten-dataset approximate regime. The exact regime is reported
    descriptively --- per-dataset rho with graph-level bootstrap intervals and
    D7 paired differences --- and the reason is stated in the text. This
    function encodes that refusal rather than trusting a caller to remember it.

    Args:
        scores: ``(n_datasets, n_methods)`` matrix of per-dataset scores.
        methods: Method names, one per column.
        regime: :attr:`Regime.EXACT` is refused; :attr:`Regime.APPROXIMATE` runs.
        lower_is_better: ``True`` for bit counts, ``False`` for correlations.

    Returns:
        The omnibus, or a refusal carrying the average ranks and the reason.

    Raises:
        MultiplicityError: If *scores* is not two-dimensional or its width
            disagrees with *methods*.
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 2:
        raise MultiplicityError(f"scores must be 2-D, got shape {array.shape}")
    if array.shape[1] != len(methods):
        raise MultiplicityError(
            f"{array.shape[1]} score columns against {len(methods)} method names"
        )
    ranks = _rank_blocks(array, lower_is_better)
    average = tuple(float(v) for v in ranks.mean(axis=0))
    names = tuple(methods)

    if regime is Regime.EXACT:
        reason = (
            "statistics.md section 4: the exact regime has five datasets; Friedman at N = 5 "
            "separates almost nothing, so the omnibus and CD diagram are reported for the "
            "ten-dataset approximate regime only and the exact regime is descriptive."
        )
        LOGGER.info("refusing a Friedman omnibus on the exact regime: %s", reason)
        return FriedmanResult(
            names, float("nan"), float("nan"), average, array.shape[0], False, reason
        )

    if array.shape[0] < 2 or array.shape[1] < 3:
        reason = f"Friedman needs >= 2 blocks and >= 3 methods, got {array.shape}"
        return FriedmanResult(
            names, float("nan"), float("nan"), average, array.shape[0], False, reason
        )

    outcome = stats.friedmanchisquare(*[array[:, j] for j in range(array.shape[1])])
    return FriedmanResult(
        methods=names,
        statistic=float(outcome.statistic),
        p_value=float(outcome.pvalue),
        average_ranks=average,
        n_datasets=int(array.shape[0]),
        ran=True,
    )


@dataclass(frozen=True)
class PostHocResult:
    """The Wilcoxon-Holm post-hoc under D8.

    **Not a BH family member.** Holm already controls the FWER within this set;
    nesting it inside BH would correct twice (``preregistration.md`` section 4.2).

    Attributes:
        pairs: Method-name pairs in comparison order.
        statistics: Wilcoxon statistic per pair.
        p_values: Raw two-sided p-values per pair.
        holm_adjusted: Holm-adjusted p-values per pair.
        counted_in_bh: Always ``False``; carried so the reporting layer cannot
            silently fold this into the BH family.
    """

    pairs: tuple[tuple[str, str], ...]
    statistics: tuple[float, ...]
    p_values: tuple[float, ...]
    holm_adjusted: tuple[float, ...]
    counted_in_bh: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "pairs": [list(pair) for pair in self.pairs],
            "statistics": list(self.statistics),
            "p_values": list(self.p_values),
            "holm_adjusted": list(self.holm_adjusted),
            "counted_in_bh": self.counted_in_bh,
        }


def wilcoxon_holm_posthoc(
    scores: npt.NDArray[Any],
    methods: Sequence[str],
) -> PostHocResult:
    """Run every pairwise Wilcoxon signed-rank test under Holm.

    Args:
        scores: ``(n_datasets, n_methods)`` matrix of per-dataset scores.
        methods: Method names, one per column.

    Returns:
        The pairwise comparisons with Holm-adjusted p-values.

    Raises:
        MultiplicityError: If fewer than two methods are supplied.
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.shape[1] != len(methods) or len(methods) < 2:
        raise MultiplicityError(f"need >= 2 named methods, got {len(methods)}")
    pairs: list[tuple[str, str]] = []
    statistics: list[float] = []
    raw: list[float] = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            pairs.append((methods[i], methods[j]))
            statistic, p_value = _wilcoxon(array[:, i], array[:, j])
            statistics.append(statistic)
            raw.append(p_value)
    adjusted = holm_bonferroni(raw)
    return PostHocResult(
        pairs=tuple(pairs),
        statistics=tuple(statistics),
        p_values=tuple(raw),
        holm_adjusted=tuple(float(v) for v in adjusted),
    )


def _wilcoxon(a: FloatArray, b: FloatArray) -> tuple[float, float]:
    """Return the Wilcoxon signed-rank statistic and p-value, ties allowed."""
    difference = a - b
    if not np.any(difference != 0.0):
        return float("nan"), 1.0
    outcome = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(outcome.statistic), float(outcome.pvalue)


#: Nemenyi critical values at alpha = 0.05, indexed by the number of methods
#: (Demsar 2006, Table 5: the studentised range statistic divided by sqrt(2)).
_NEMENYI_Q05: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
}

#: The same at alpha = 0.10.
_NEMENYI_Q10: dict[int, float] = {
    2: 1.645,
    3: 2.052,
    4: 2.291,
    5: 2.459,
    6: 2.589,
    7: 2.693,
    8: 2.780,
    9: 2.855,
    10: 2.920,
    11: 2.978,
    12: 3.030,
    13: 3.077,
    14: 3.120,
    15: 3.159,
}


@dataclass(frozen=True)
class CriticalDifference:
    """The data behind a critical-difference diagram.

    Attributes:
        methods: Method names ordered by average rank, best first.
        average_ranks: Average ranks in the same order.
        cd: The Nemenyi critical difference.
        alpha: The significance level behind ``cd``.
        n_datasets: Blocks behind the ranks.
        cliques: Groups of methods no further apart than ``cd``, as index
            tuples into *methods*.
    """

    methods: tuple[str, ...]
    average_ranks: tuple[float, ...]
    cd: float
    alpha: float
    n_datasets: int
    cliques: tuple[tuple[int, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "methods": list(self.methods),
            "average_ranks": list(self.average_ranks),
            "critical_difference": self.cd,
            "alpha": self.alpha,
            "n_datasets": self.n_datasets,
            "cliques": [list(c) for c in self.cliques],
        }


def critical_difference(
    average_ranks: Sequence[float],
    methods: Sequence[str],
    n_datasets: int,
    *,
    alpha: float = 0.05,
) -> CriticalDifference:
    """Return the Nemenyi critical difference and its cliques.

    ``CD = q_alpha * sqrt( k (k + 1) / (6 N) )`` with ``k`` methods over ``N``
    datasets (Demsar 2006, section 3.2.2).

    Args:
        average_ranks: Average rank per method.
        methods: Method names, same order.
        n_datasets: ``N``.
        alpha: 0.05 or 0.10; the only two tabulated levels.

    Returns:
        The diagram data, methods sorted best-first.

    Raises:
        MultiplicityError: If the method count is outside the table or *alpha*
            is not tabulated.
    """
    table = {0.05: _NEMENYI_Q05, 0.10: _NEMENYI_Q10}.get(alpha)
    if table is None:
        raise MultiplicityError(f"alpha must be 0.05 or 0.10, got {alpha}")
    k = len(methods)
    if k not in table:
        raise MultiplicityError(f"no tabulated Nemenyi value for {k} methods")
    if n_datasets <= 0:
        raise MultiplicityError(f"n_datasets must be positive, got {n_datasets}")

    cd = float(table[k] * np.sqrt(k * (k + 1) / (6.0 * n_datasets)))
    order = np.argsort(np.asarray(average_ranks, dtype=np.float64), kind="stable")
    sorted_names = tuple(methods[int(i)] for i in order)
    sorted_ranks = tuple(float(average_ranks[int(i)]) for i in order)

    cliques: list[tuple[int, ...]] = []
    for start in range(k):
        stop = start
        while stop + 1 < k and sorted_ranks[stop + 1] - sorted_ranks[start] <= cd:
            stop += 1
        if stop > start:
            candidate = tuple(range(start, stop + 1))
            if not any(set(candidate) <= set(existing) for existing in cliques):
                cliques.append(candidate)
    return CriticalDifference(
        methods=sorted_names,
        average_ranks=sorted_ranks,
        cd=cd,
        alpha=alpha,
        n_datasets=int(n_datasets),
        cliques=tuple(cliques),
    )
