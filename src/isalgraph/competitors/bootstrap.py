"""Graph-level percentile bootstrap for a Spearman rho over graph *pairs*.

**Why the pair level is the wrong level.**  A rho computed over the
``k(k-1)/2`` pairs induced by ``k`` graphs looks as if it had ``k(k-1)/2``
independent observations, and a pair-level bootstrap -- resampling pairs --
inherits that fiction.  It does not: every pair shares a graph with
``2(k-2)`` other pairs, so the effective sample size is governed by the
**graphs**.  The measured consequence on this cohort is that rho moved by up
to 0.07 between two independent 200-graph draws on AIDS, which no pair-level
interval of any width would have predicted.

So this module resamples **graph indices** with replacement, rebuilds the
pair set the resampled graphs induce, and recomputes rho on it.  That is the
dyadic (Snijders and Borgatti, *Connections* 22(2):161-170, 1999) bootstrap:
the resampling unit is the unit that was actually sampled.

**Self-pairs are dropped, deliberately.**  A resample draws graphs with
replacement, so the same graph can occupy two slots.  Pairing it with itself
would contribute a point at distance 0 and GED 0 that was never in the
observed pair set, biasing rho upward exactly where the distribution is
tightest.  :func:`graph_bootstrap_ci` drops those slots by construction: the
pair lookup table's diagonal is never written, so a duplicated slot resolves
to "absent" rather than to a fabricated zero.

**One resample matrix per (dataset, view), shared by every representation.**
:func:`make_resample_index` returns the draws as an object the caller passes
to every representation's :func:`graph_bootstrap_ci` call, so differences
between two representations on one dataset are *paired* rather than
independently noisy.  Generating the draws inside the rho function instead
would make every representation see a different set of graphs, and the
comparison the paper leads with -- IsalGraph against the size null -- would
be the comparison most corrupted by it.

Randomness is ``random.Random(seed)``, matching the wave contract, not
``numpy.random``.  Determinism is asserted by a test.

Standard library plus ``numpy``/``scipy`` only, and both arrive inside
function bodies so that importing this module costs nothing.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy.typing as npt

#: Frozen protocol values (T-04a design note 3.8, decision D2).
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.05

#: Fewer surviving pairs than this and a resample contributes no rho.  Three
#: is the smallest pair count on which a rank correlation is not degenerate.
MIN_PAIRS = 3


class BootstrapError(ValueError):
    """Raised when a bootstrap is asked for something structurally impossible."""


@dataclass(frozen=True, slots=True)
class ResampleIndex:
    """Graph-index draws, generated once and reused across representations.

    Attributes:
        n_graphs: number of graphs in the observed draw.  Indices in
            :attr:`draws` and in every ``pair_index`` are **positions into
            that draw**, ``0 .. n_graphs - 1`` -- never cohort indices.
        resamples: number of bootstrap replicates.
        seed: the ``random.Random`` seed that produced :attr:`draws`.
        draws: ``(resamples, n_graphs)`` array of resampled positions.
    """

    n_graphs: int
    resamples: int
    seed: int
    draws: npt.NDArray[Any]


def make_resample_index(
    n_graphs: int,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> ResampleIndex:
    """Draw the shared resample matrix for one (dataset, view).

    Args:
        n_graphs: size of the observed graph draw.
        resamples: number of bootstrap replicates.
        seed: ``random.Random`` seed.  The matrix is a pure function of
            ``(n_graphs, resamples, seed)``.

    Returns:
        The :class:`ResampleIndex` to hand to every representation.

    Raises:
        BootstrapError: if *n_graphs* or *resamples* is not positive.
    """
    import numpy as np

    if n_graphs <= 0:
        raise BootstrapError(f"n_graphs must be positive, got {n_graphs}")
    if resamples <= 0:
        raise BootstrapError(f"resamples must be positive, got {resamples}")

    rng = random.Random(seed)
    population = range(n_graphs)
    draws = np.empty((resamples, n_graphs), dtype=np.int32)
    for r in range(resamples):
        draws[r] = rng.choices(population, k=n_graphs)
    return ResampleIndex(n_graphs=n_graphs, resamples=resamples, seed=seed, draws=draws)


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Point-estimate Spearman rho and its p-value, via ``scipy.stats``."""
    from scipy.stats import spearmanr

    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _rho(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
    """Spearman rho only, without the p-value.

    ``scipy.stats.spearmanr`` recomputes a t-distribution tail on every call;
    over 2,000 replicates that dominates the cost and none of it is read.
    Ranks with average tie handling followed by a Pearson correlation is the
    same statistic.
    """
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denominator == 0.0:
        return float("nan")  # a constant column: rho is undefined, not zero
    return float((rx * ry).sum() / denominator)


def graph_bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    pair_index: Sequence[tuple[int, int]],
    index: ResampleIndex,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_pairs: int = MIN_PAIRS,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for ``spearman(x, y)``, resampling graphs.

    Args:
        x: one value per observed pair -- the representation's distance.
        y: one value per observed pair -- the ground-truth GED.
        pair_index: ``(a, b)`` positions into the observed graph draw, one
            entry per element of *x* and *y*, with ``a != b``.
        index: the shared :class:`ResampleIndex`.  **The same object must be
            passed for every representation within one dataset and view**;
            that is what makes their intervals paired.
        alpha: two-sided miss rate.  ``0.05`` gives the 2.5/97.5 percentiles.
        min_pairs: a replicate inducing fewer pairs than this contributes no
            rho.

    Returns:
        ``(low, high)``, or ``None`` when fewer than two replicates produced
        a defined rho.  ``None`` is a printed absence; a fabricated interval
        computed from one replicate is not.

    Raises:
        BootstrapError: if the inputs disagree in length, or a pair names a
            graph outside ``0 .. index.n_graphs - 1``.
    """
    import numpy as np

    if len(x) != len(y) or len(x) != len(pair_index):
        raise BootstrapError(
            f"x, y and pair_index must agree in length; got {len(x)}, {len(y)}, {len(pair_index)}"
        )
    if len(pair_index) < min_pairs:
        return None

    n = index.n_graphs
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)

    # -1 marks "this graph pair is not in the observed set".  The diagonal is
    # never written, so a graph drawn into two slots yields no self-pair.
    lookup = np.full((n, n), -1, dtype=np.int32)
    for k, (a, b) in enumerate(pair_index):
        if not (0 <= a < n and 0 <= b < n):
            raise BootstrapError(f"pair ({a}, {b}) is outside 0..{n - 1}")
        lookup[a, b] = k
        lookup[b, a] = k

    upper_i, upper_j = np.triu_indices(n, 1)
    rhos: list[float] = []
    for r in range(index.resamples):
        drawn = index.draws[r]
        selected = lookup[drawn[upper_i], drawn[upper_j]]
        selected = selected[selected >= 0]
        if selected.size < min_pairs:
            continue
        value = _rho(xs[selected], ys[selected])
        if math.isfinite(value):
            rhos.append(value)

    if len(rhos) < 2:
        return None
    low, high = np.percentile(np.asarray(rhos), [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "MIN_PAIRS",
    "BootstrapError",
    "ResampleIndex",
    "graph_bootstrap_ci",
    "make_resample_index",
    "spearman",
]
