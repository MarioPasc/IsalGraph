"""D2 / D7 / D15 --- the graph-level cluster bootstrap and its frozen effort.

The resampling unit is the **graph**, never the pair. That single change is the
answer to R3.5c: LINUX contributes 89 graphs, not 3,916 independent
observations, and an interval computed from the pair list understates the
uncertainty by roughly the square root of the average cluster size.

What is reused rather than rewritten
------------------------------------
Two graph-level bootstraps already exist in this repository and both are
correct. This module factors a single entry point over their shared mechanism
instead of adding a third implementation:

* ``ged_bakeoff_analysis.replicate_selection``, ``.induced_pairs`` and
  ``.pair_flat_index`` are imported directly. The seeding rule
  ``SeedSequence([seed, replicate])`` is the repository convention and makes a
  replicate's draw independent of loop order.
* ``approx_ged_analysis._bootstrap_slopes_subsampled``'s tier-3 mechanism ---
  draw the graph resample first, then draw slot pairs uniformly without
  replacement from the ``N (N - 1) / 2`` slot pairs and drop self-pairs --- is
  reimplemented here over pair indices rather than over a regression fit,
  reusing the same ``SeedSequence([seed, replicate, 1])`` substream so the two
  paths draw identical subsamples.

What is **not** reused: ``approx_ged_analysis``'s ``np.bincount`` /
``np.einsum`` weighted-sums identity. That identity computes sums of a matrix
over the induced pair multiset without materialising it, which is exact for an
OLS slope and inapplicable to Spearman: ranks are not a sum, and D15 requires
re-ranking inside every replicate.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from benchmarks.real_data.eval_setup.ged_bakeoff_analysis import (
    induced_pairs,
    pair_flat_index,
    replicate_selection,
)
from benchmarks.real_data.eval_setup.ged_pair_index import pairs_from_indices_searchsorted

LOGGER = logging.getLogger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

#: CONTRACTS.md section 2. Everywhere, no exceptions.
SEED: int = 42

#: Percentile bootstrap interval, two-sided.
CI_LEVEL: float = 0.95

#: D9's family-wise false discovery rate.
FDR_Q: float = 0.05

__all__ = [
    "CI_LEVEL",
    "FDR_Q",
    "SEED",
    "BootstrapTier",
    "PercentileInterval",
    "bootstrap_p_value",
    "bootstrap_tier",
    "cluster_bootstrap",
    "difference_samples",
    "induced_pairs",
    "pair_flat_index",
    "percentile_interval",
    "replicate_pair_indices",
    "replicate_selection",
]


class ResamplingError(Exception):
    """Raised when a resampling request is inconsistent with its inputs."""


@dataclass(frozen=True)
class BootstrapTier:
    """The D15 resampling effort assigned to one dataset.

    Attributes:
        tier: 1, 2 or 3, as ``statistics.md`` section 5 assigns it.
        replicates: Bootstrap replicate count.
        permutations: Mantel permutation count.
        subsample: Induced pairs drawn uniformly without replacement inside
            each replicate, or ``None`` for every induced pair.
    """

    tier: int
    replicates: int
    permutations: int
    subsample: int | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description, for the mandatory reporting."""
        return {
            "tier": self.tier,
            "replicates": self.replicates,
            "permutations": self.permutations,
            "within_replicate_pairs": "all" if self.subsample is None else self.subsample,
            "seed": SEED,
            "resampling_unit": "graph",
        }


#: The frozen D15 tier index. CONTRACTS.md section 2 states it is **looked up,
#: never recomputed at run time**, so this is a literal table and not a
#: function of the observed pair count.
D15_TIERS: dict[str, int] = {
    "linux": 1,
    "protein": 1,
    "grec": 1,
    "aids_graphedx": 1,
    "iam_letter_low": 1,
    "iam_letter_med": 1,
    "aids_iam": 2,
    "iam_letter_high": 2,
    "coil_del": 3,
    "mutagenicity": 3,
}

#: Effort per tier: (replicates, permutations, within-replicate pair budget).
_TIER_EFFORT: dict[int, tuple[int, int, int | None]] = {
    1: (2000, 9999, None),
    2: (2000, 4999, None),
    3: (1000, 1999, 2_000_000),
}

#: Suite 1 applies ``n_max = 12`` and no Suite-1 dataset is subsampled
#: (``statistics.md`` section 5). Only IAM Letter HIGH reaches tier 2 there.
_SUITE1_TIERS: dict[str, int] = {
    "iam_letter_low": 1,
    "iam_letter_med": 1,
    "iam_letter_high": 2,
    "linux": 1,
    "aids": 1,
}


def bootstrap_tier(dataset_key: str, suite: str = "suite2") -> BootstrapTier:
    """Return the frozen D15 tier for a dataset.

    The assignment is a lookup. ``statistics.md`` section 5's frozen table is
    "the assignment T-06 runs; it is not recomputed at execution time", so a
    dataset absent from the table is an error rather than a default.

    Args:
        dataset_key: A dataset key as spelled in CONTRACTS.md section 2.
        suite: ``"suite1"`` or ``"suite2"``. Suite 1 has its own assignment
            because ``n_max = 12`` changes the pair counts.

    Returns:
        The replicate count, permutation count and within-replicate pair budget.

    Raises:
        ResamplingError: If the dataset or suite is unknown.
    """
    table = _SUITE1_TIERS if suite == "suite1" else D15_TIERS
    if suite not in {"suite1", "suite2"}:
        raise ResamplingError(f"unknown suite {suite!r}")
    if dataset_key not in table:
        raise ResamplingError(f"no frozen D15 tier for {suite} dataset {dataset_key!r}")
    tier = table[dataset_key]
    replicates, permutations, subsample = _TIER_EFFORT[tier]
    return BootstrapTier(
        tier=tier,
        replicates=replicates,
        permutations=permutations,
        subsample=subsample,
    )


def replicate_pair_indices(
    n_graphs: int,
    seed: int,
    replicate: int,
    subsample: int | None = None,
) -> IntArray:
    """Return the canonical pair indices contributed by one replicate.

    **The graph resample is drawn first, unconditionally.** When *subsample* is
    given, it applies to the induced pairs *inside* that replicate and never to
    the graph list --- D15 rule 1, and the reason D2's answer to R3.5c survives
    tier 3.

    Args:
        n_graphs: Number of graphs in the dataset.
        seed: Master seed; 42 in production.
        replicate: Zero-based replicate number.
        subsample: Induced pairs to draw, or ``None`` for all of them.

    Returns:
        Flat indices into the canonical upper-triangular pair order, with
        repetitions. The duplication is the cluster bootstrap's variance
        mechanism and must not be de-duplicated away.
    """
    selection = replicate_selection(n_graphs, seed, replicate)
    if subsample is None:
        return np.asarray(induced_pairs(n_graphs, selection), dtype=np.int64)

    total_slots = n_graphs * (n_graphs - 1) // 2
    draw = min(int(subsample), total_slots)
    rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 1]))
    flat_slots = rng.choice(total_slots, size=draw, replace=False, shuffle=False)
    slot_i, slot_j = pairs_from_indices_searchsorted(flat_slots, n_graphs)
    a = selection[slot_i]
    b = selection[slot_j]
    keep = a != b
    lo = np.minimum(a[keep], b[keep])
    hi = np.maximum(a[keep], b[keep])
    return np.asarray(pair_flat_index(n_graphs, lo, hi), dtype=np.int64)


#: A statistic evaluated on one replicate's pair index vector.
StatisticFn: TypeAlias = Callable[[IntArray], Mapping[str, float]]


def cluster_bootstrap(
    n_graphs: int,
    statistic: StatisticFn,
    tier: BootstrapTier,
    *,
    valid: BoolArray | None = None,
    seed: int = SEED,
    replicates: int | None = None,
) -> dict[str, FloatArray]:
    """Run the D2 graph-level cluster bootstrap.

    Every statistic *statistic* returns is evaluated on the **same** resample,
    which is what makes the D7 paired difference correct by construction rather
    than by a matching step afterwards.

    Args:
        n_graphs: Number of graphs; the resampling unit.
        statistic: Callable mapping one replicate's flat pair indices to a
            mapping of statistic name to value. ``nan`` is permitted and is
            dropped from the percentile interval.
        tier: The frozen D15 effort for this dataset.
        valid: Optional length ``n (n - 1) / 2`` mask over the canonical pair
            order. Pairs where it is ``False`` are dropped from every replicate.
        seed: Master seed; 42 in production.
        replicates: Overrides ``tier.replicates``. Tests use it; production
            does not.

    Returns:
        One float array per statistic name, of length ``replicates``.

    Raises:
        ResamplingError: If ``n_graphs`` is below 2 or *valid* is misshaped.
    """
    if n_graphs < 2:
        raise ResamplingError(f"a cluster bootstrap needs at least 2 graphs, got {n_graphs}")
    n_pairs = n_graphs * (n_graphs - 1) // 2
    if valid is not None and valid.shape != (n_pairs,):
        raise ResamplingError(f"valid mask must have shape ({n_pairs},), got {valid.shape}")

    count = int(tier.replicates if replicates is None else replicates)
    collected: dict[str, list[float]] = {}
    for replicate in range(count):
        flat = replicate_pair_indices(n_graphs, seed, replicate, tier.subsample)
        if valid is not None:
            flat = flat[valid[flat]]
        for key, value in statistic(flat).items():
            collected.setdefault(key, []).append(float(value))
    return {key: np.asarray(values, dtype=np.float64) for key, values in collected.items()}


def difference_samples(
    samples: Mapping[str, FloatArray],
    left: str,
    right: str,
) -> FloatArray:
    """Return the D7 paired difference of two bootstrap distributions.

    Both arguments must come from one :func:`cluster_bootstrap` call so that
    replicate ``r`` of each refers to the same graph resample. This is
    deliberately **not** Hotelling-Williams or Steiger: those are the textbook
    tools for dependent correlations sharing a variable, and both assume
    independent observations --- exactly the error R3.5c identified.

    Args:
        samples: Output of :func:`cluster_bootstrap`.
        left: Name of the minuend statistic.
        right: Name of the subtrahend statistic.

    Returns:
        The elementwise difference, one value per replicate.

    Raises:
        ResamplingError: If either name is absent or the lengths differ.
    """
    if left not in samples or right not in samples:
        missing = sorted({left, right} - set(samples))
        raise ResamplingError(f"statistic(s) {missing} absent from the bootstrap output")
    a = samples[left]
    b = samples[right]
    if a.shape != b.shape:
        raise ResamplingError(
            "paired differences require one shared resample; "
            f"got {a.shape} and {b.shape} replicates"
        )
    return a - b


@dataclass(frozen=True)
class PercentileInterval:
    """A percentile bootstrap interval and the distribution behind it.

    Attributes:
        point: The full-sample estimate. Never the bootstrap mean --- the
            resample carries uncertainty, not the estimate.
        ci_low: Lower percentile bound.
        ci_high: Upper percentile bound.
        level: Two-sided coverage, e.g. 0.95.
        bootstrap_mean: Mean of the finite replicates.
        bootstrap_sd: Standard deviation of the finite replicates.
        n_finite: Replicates that produced a finite value.
        n_replicates: Replicates attempted.
    """

    point: float
    ci_low: float
    ci_high: float
    level: float
    bootstrap_mean: float
    bootstrap_sd: float
    n_finite: int
    n_replicates: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies strictly on one side of zero."""
        if not (math.isfinite(self.ci_low) and math.isfinite(self.ci_high)):
            return False
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def width(self) -> float:
        """Interval width, ``nan`` when either bound is not finite."""
        return self.ci_high - self.ci_low

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "level": self.level,
            "bootstrap_mean": self.bootstrap_mean,
            "bootstrap_sd": self.bootstrap_sd,
            "n_finite": self.n_finite,
            "n_replicates": self.n_replicates,
            "excludes_zero": self.excludes_zero,
        }


def percentile_interval(
    samples: Sequence[float] | FloatArray,
    point: float,
    level: float = CI_LEVEL,
) -> PercentileInterval:
    """Return the percentile interval of a bootstrap distribution.

    Args:
        samples: Replicate values; non-finite entries are dropped.
        point: The full-sample estimate.
        level: Two-sided coverage. D9's FCR adjustment supplies a level below
            0.95 for the tests BH selects (Benjamini & Yekutieli, *JASA*
            100(469):71-81, 2005).

    Returns:
        The interval, with ``nan`` bounds when no replicate was finite.

    Raises:
        ResamplingError: If *level* is outside ``(0, 1)``.
    """
    if not 0.0 < level < 1.0:
        raise ResamplingError(f"coverage must lie in (0, 1), got {level}")
    array = np.asarray(samples, dtype=np.float64)
    finite = array[np.isfinite(array)]
    n_replicates = int(array.size)
    if finite.size == 0:
        nan = float("nan")
        return PercentileInterval(point, nan, nan, level, nan, nan, 0, n_replicates)
    alpha = 1.0 - level
    lo, hi = (
        float(v) for v in np.percentile(finite, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    )
    return PercentileInterval(
        point=float(point),
        ci_low=lo,
        ci_high=hi,
        level=level,
        bootstrap_mean=float(finite.mean()),
        bootstrap_sd=float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        n_finite=int(finite.size),
        n_replicates=n_replicates,
    )


def bootstrap_p_value(samples: Sequence[float] | FloatArray) -> float:
    """Return a two-sided bootstrap p-value for the null that the mean is zero.

    Uses the inversion of the percentile interval with the ``(count + 1) /
    (R + 1)`` continuity correction, so a distribution entirely on one side of
    zero returns ``2 / (R + 1)`` rather than 0. D10 keeps inference on the
    intervals; this p-value exists so BH can be applied to the family.

    Args:
        samples: Replicate values; non-finite entries are dropped.

    Returns:
        A p-value in ``(0, 1]``, or ``nan`` when no replicate was finite.
    """
    array = np.asarray(samples, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    n = float(finite.size)
    below = (1.0 + float(np.count_nonzero(finite <= 0.0))) / (n + 1.0)
    above = (1.0 + float(np.count_nonzero(finite >= 0.0))) / (n + 1.0)
    return float(min(1.0, 2.0 * min(below, above)))
