"""Shared frozen constants and exact statistics for E1-E4.

Written by the orchestrator **before** any experiment ran, so that no track
chooses its own interval, its own resampling unit or its own relabelling
count.  Every value here is fixed by
``.claude/notes/review/tasks/T-04a-admissibility-protocol.md`` §1.

Two rules this module exists to enforce mechanically:

- **Proportions get exact intervals, never Wald.**  Every proportion measured
  in this annex sits at or near a boundary -- invariance rates are 0 or 1,
  collision rates are 0 -- where the normal approximation is invalid and can
  leave the unit interval entirely.  :func:`clopper_pearson` is the only
  interval for a count in this package.
- **Zero events is not "0".**  ``0/N`` is reported as an upper bound by the
  rule of three, :func:`rule_of_three`.  Printing 0 asserts impossibility from
  a finite sample.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

# --------------------------------------------------------------------------
# Frozen protocol constants.  Changing one requires a dated changelog entry in
# the protocol note; they are collected here so that is a one-line diff.
# --------------------------------------------------------------------------

#: Relabellings per graph in E1 (protocol §2).
RELABELLINGS: int = 50
#: The single seed for every draw, relabelling stream and resample in E1-E4.
SEED: int = 42
#: Per-dataset draw size (protocol §2, §3).
N_PER_DATASET: int = 200
#: Pooled stratified draw size -- identical to the T-04a grid's ``S200``.
N_POOLED: int = 200
#: Graph-level bootstrap resamples (D2).
RESAMPLES: int = 2000
#: Largest node count for the exhaustive triple sweep in E3 (142 graphs).
EXHAUSTIVE_N_TRIPLES: int = 6
#: Largest node count for the exhaustive invariance sweep in E1 (995 graphs).
EXHAUSTIVE_N_INVARIANCE: int = 7

#: Where the self-contained deliverable is assembled.
DELIVERABLE_ROOT = (
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/experiments/metric-admissibility"
)


class AdmissibilityError(RuntimeError):
    """Raised when an experiment's precondition fails.

    Distinct from a *result*: a class-III representation colliding, or a
    Levenshtein distance violating the triangle inequality, is a defect in our
    code rather than a property of the method, and the protocol escalates it
    instead of reporting it.
    """


# --------------------------------------------------------------------------
# Exact interval estimation
# --------------------------------------------------------------------------


def clopper_pearson(k: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval for ``k`` successes in ``n`` trials.

    Clopper & Pearson, *Biometrika* 26(4):404-413, 1934.  Inverts the binomial
    test rather than approximating it, so it is valid at ``k = 0`` and
    ``k = n`` where the Wald interval collapses to a point or leaves ``[0,1]``.

    Args:
        k: Number of successes.
        n: Number of trials.
        alpha: Two-sided error rate.

    Returns:
        ``(lo, hi)``, the conservative two-sided interval.  ``(0.0, 0.0)`` when
        ``n == 0``.

    Raises:
        ValueError: If ``k`` is negative or exceeds ``n``.
    """
    from scipy.stats import beta

    if n == 0:
        return (0.0, 0.0)
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, n={n}]")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def rule_of_three(n: int, *, confidence: float = 0.95) -> float:
    """One-sided upper bound on a rate given **zero** events in ``n`` trials.

    ``3/n`` at 95 %.  Reporting an observed ``0/n`` as the rate ``0`` asserts
    impossibility from a finite sample; this is what the sample actually
    licenses.

    Args:
        n: Number of trials, all of which produced no event.
        confidence: Coverage; the numerator is ``-ln(1 - confidence)``.

    Returns:
        The upper bound, or ``1.0`` when ``n == 0``.
    """
    if n <= 0:
        return 1.0
    return min(1.0, -math.log(1.0 - confidence) / n)


# --------------------------------------------------------------------------
# Paired between-representation comparison (protocol D-A5)
# --------------------------------------------------------------------------


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, order preserved.

    Holm, *Scand. J. Statist.* 6(2):65-70, 1979.  Controls the FWER within the
    pairwise family without assuming independence.  This family is exploratory
    and is deliberately outside ``preregistration.md``'s frozen family, so it
    changes neither ``N_max`` nor ``N_actual``.

    Args:
        pvalues: Raw p-values.

    Returns:
        Adjusted p-values in the input order, each clipped to ``[0, 1]`` and
        made monotone non-decreasing along the sorted sequence.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    out = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        out[idx] = min(1.0, running)
    return out


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict[str, float | int | None]:
    """Wilcoxon signed-rank on paired samples, with a rank-biserial effect size.

    Every representation sees the same graphs and the same relabellings (D7),
    so the comparison is paired by construction and an unpaired test would
    discard that pairing.

    Args:
        a: First sample.
        b: Second sample, same length and pairing as *a*.

    Returns:
        ``statistic``, ``p``, ``n_nonzero``, ``rank_biserial`` and the two
        medians.  ``p`` is ``None`` when every pair is tied, which is the
        expected outcome when two representations are both exactly invariant.

    Raises:
        ValueError: If the samples differ in length.
    """
    from scipy.stats import rankdata, wilcoxon

    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} and {len(b)}")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    nonzero = [d for d in diffs if d != 0.0]
    if not nonzero:
        return {"statistic": 0.0, "p": None, "n_nonzero": 0, "rank_biserial": 0.0}
    result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")

    # scipy's two-sided `statistic` is min(W+, W-), which carries NO direction:
    # deriving the effect size from it yields an unsigned magnitude that reads
    # as though it were signed.  Rebuild W+ and W- from the signed ranks so the
    # sign means what a reader will assume it means.
    ranks = rankdata([abs(d) for d in nonzero])
    w_pos = float(sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0))
    w_neg = float(sum(r for r, d in zip(ranks, nonzero, strict=True) if d < 0))
    total = w_pos + w_neg
    return {
        "statistic": float(result.statistic),
        "p": float(result.pvalue),
        "n_nonzero": len(nonzero),
        # Matched-pairs rank-biserial, SIGNED: (W+ - W-) / (W+ + W-), in [-1, 1].
        # Positive means `a` exceeds `b`.
        "rank_biserial": float((w_pos - w_neg) / total) if total else 0.0,
        "median_a": float(statistics.median(a)),
        "median_b": float(statistics.median(b)),
    }


# --------------------------------------------------------------------------
# Exhaustive graph families
# --------------------------------------------------------------------------


def connected_atlas(max_n: int) -> list[nx.Graph]:
    """Every connected graph on ``2..max_n`` nodes, one per isomorphism class.

    From ``networkx.graph_atlas_g``, which is complete and non-redundant up to
    seven nodes.  Counts are OEIS A001349 -- 1, 2, 6, 21, 112, 853 for
    ``n = 2..7`` -- and a caller asserting that is asserting the atlas is
    intact rather than trusting it.

    Args:
        max_n: Largest node count, at most 7.

    Returns:
        The graphs, ascending in node count.

    Raises:
        ValueError: If ``max_n`` exceeds the atlas's range.
    """
    import networkx as nx

    if max_n > 7:
        raise ValueError(f"graph_atlas_g stops at n = 7; asked for {max_n}")
    return [
        g for g in nx.graph_atlas_g() if 2 <= g.number_of_nodes() <= max_n and nx.is_connected(g)
    ]


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def primary_distances(grid_path: str) -> dict[str, str | None]:
    """The selected primary distance per representation, from a grid JSON.

    Args:
        grid_path: Path to ``grid_200.json``.

    Returns:
        Mapping from backend name to metric name, ``None`` where the grid
        admitted no distance.

    Raises:
        AdmissibilityError: If the file carries no ``primary_distance`` block.
    """
    with open(grid_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    block = payload.get("primary_distance")
    if not isinstance(block, dict):
        raise AdmissibilityError(f"{grid_path} has no primary_distance block")
    return {str(k): (str(v) if v is not None else None) for k, v in block.items()}


def write_result(path: str, experiment: str, payload: dict[str, Any]) -> None:
    """Write one experiment's result with a standard provenance header.

    Args:
        path: Destination ``.json``.  Parent directories are created.
        experiment: ``"E1"``…``"E4"``.
        payload: The experiment's own record.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = {
        "experiment": experiment,
        "protocol": ".claude/notes/review/tasks/T-04a-admissibility-protocol.md",
        "seed": SEED,
        "relabellings": RELABELLINGS,
        "resamples": RESAMPLES,
        "note": (
            "Exploratory. Outside preregistration.md's frozen confirmatory "
            "family; changes neither N_max nor N_actual (decision 23)."
        ),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({**header, **payload}, handle, indent=2, sort_keys=True)


__all__ = [
    "DELIVERABLE_ROOT",
    "EXHAUSTIVE_N_INVARIANCE",
    "EXHAUSTIVE_N_TRIPLES",
    "N_POOLED",
    "N_PER_DATASET",
    "RELABELLINGS",
    "RESAMPLES",
    "SEED",
    "AdmissibilityError",
    "clopper_pearson",
    "connected_atlas",
    "holm",
    "primary_distances",
    "rule_of_three",
    "wilcoxon_paired",
    "write_result",
]
