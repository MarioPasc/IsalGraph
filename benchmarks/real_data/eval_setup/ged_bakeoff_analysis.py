"""Aggregation, bootstrap, significance and selection for the T-27 bake-off.

Consumes the per-cell measurements written by
``ged_bound_bakeoff.py`` (contract:
``.claude/notes/2026-08-13-t27-bakeoff/CONTRACTS.md`` §2, §3, §5) and
produces the five analysis JSON files of §7 plus the figures of §8.

What this module is for
-----------------------
The manuscript reports a proven bracket ``LB <= GED <= UB`` above twelve
nodes. ``BRANCH_FAST`` was named the lower bound on 400 LINUX pairs and
``IPFP`` was named the upper bound on no measurement at all. T-27
re-selects both ends against 3,836,827 certified exact GED values, and
this module is the selection.

Statistical protocol -- locked, not negotiable here
---------------------------------------------------
The resampling unit is the **graph**, never the pair
(``statistics.md`` D2). Pair-level asymptotics is the error Reviewer 3
identified, and repeating it in a more sophisticated form is exactly
what D7 forbids. Consequences, all of them deliberate:

* Confidence intervals are wide. LINUX contributes 89 graphs, not 3,916
  independent observations. That is the correct outcome and it is
  reported.
* A p-value over 2,030,043 dyadically dependent pairs is not evidence.
  Wilcoxon returns a vanishing p for a difference of no practical size,
  and dependence makes the nominal p anticonservative on top of that.
  Effect sizes lead (D10): the matched-pairs rank-biserial correlation
  and the graph-level bootstrap CI on the difference in mean error are
  the primary evidence, and every emitted JSON says so.
* Everything here is a **selection procedure, not a hypothesis test, and
  is outside the pre-registered confirmatory family**.

The frozen rules of ``T-27-design.md`` §3.1-§3.3, §3.5-§3.8 and
``T-27-spec.md`` §5 are implemented verbatim, not improved.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import platform
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from benchmarks.real_data.eval_correlation.correlation_metrics import holm_bonferroni

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
WAVE = "2026-08-13-t27-bakeoff"

#: statistics.md D2 / design §3.7. Not configurable in production runs;
#: the CLI exposes an override only so tests can run in milliseconds.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
CI_PERCENTILES = (2.5, 97.5)

#: ``HED`` joins the lower end after track A resolved its vacuous 0.00:
#: under cost model D6 edge substitution is free, and only
#: ``--edge-set-distances OPTIMAL`` makes the accessor return a
#: non-degenerate bound. Verified on four hand-built pairs, all valid.
#: It is expected to lose -- ``approx_ged.md`` §2 records the published
#: dominance ``BED >= HED`` -- and a loose HED confirms that ordering at
#: 3.8 M-pair scale, which is a result rather than a defect.
LOWER_METHODS: tuple[str, ...] = ("BRANCH", "BRANCH_FAST", "BRANCH_TIGHT", "STAR", "HED")
UPPER_METHODS: tuple[str, ...] = ("IPFP", "REFINE", "BIPARTITE", "BP_BEAM")
ENDS: tuple[str, ...] = ("lower", "upper")


def methods_for_end(end: str) -> tuple[str, ...]:
    """Return the method roster for a bracket end.

    The roster is the single source of the Holm family size: the lower
    end runs ten pairwise comparisons (five methods) and the upper end
    six (four methods). Nothing downstream may hard-code either number.

    Args:
        end: ``"lower"`` or ``"upper"``.

    Returns:
        The method names for that end.

    Raises:
        BakeoffAnalysisError: If *end* is neither.
    """
    if end == "lower":
        return LOWER_METHODS
    if end == "upper":
        return UPPER_METHODS
    raise BakeoffAnalysisError(f"end must be 'lower' or 'upper', got {end!r}")


def end_of_method(method: str) -> str:
    """Return the bracket end a GEDLIB method belongs to.

    Args:
        method: GEDLIB method name, upper case.

    Returns:
        ``"lower"`` or ``"upper"``.

    Raises:
        BakeoffAnalysisError: If the method is in neither roster.
    """
    if method in LOWER_METHODS:
        return "lower"
    if method in UPPER_METHODS:
        return "upper"
    raise BakeoffAnalysisError(f"unknown method {method!r}")


DATASETS: tuple[str, ...] = (
    "linux",
    "aids",
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
)

#: design §3.2 -- the five datasets are three corpora. IAM Letter
#: LOW/MED/HIGH share a 15-class letter alphabet and one generator, so
#: the frozen ">= 4 of 5" vote is really 3 + 1 + 1.
CORPUS_OF: dict[str, str] = {
    "linux": "LINUX",
    "aids": "AIDS",
    "iam_letter_low": "Letter",
    "iam_letter_med": "Letter",
    "iam_letter_high": "Letter",
}

#: spec §5: ties within 2 % relative break on M7 cost, then on M6.
TIE_TOLERANCE = 0.02

#: spec §5 cost gate, in microseconds per pair, evaluated on the design
#: §3.4 probe at n-bar ~ 30 -- never on the bake-off corpus, which is
#: n <= 12 by construction and cannot evaluate it.
COST_GATE_US_PER_PAIR = 1000.0

#: Studentised range statistic divided by sqrt(2), alpha = 0.05, for the
#: Nemenyi post-hoc test. Demsar, JMLR 7:1-30, 2006, Table 5.
NEMENYI_Q05: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}

SELECTION_STATUS = (
    "This is a selection procedure, not a hypothesis test, and it is outside "
    "the pre-registered confirmatory family."
)

PVALUE_STATUS = (
    "The p-value is reported for completeness and is not the basis of the "
    "selection. With millions of dyadically dependent pairs a Wilcoxon test "
    "returns a vanishing p-value for a difference of no practical size, and "
    "the dyadic dependence makes the nominal p anticonservative on top of "
    "that. The primary evidence is the matched-pairs rank-biserial effect "
    "size and the graph-level bootstrap CI on the difference in mean error "
    "(statistics.md D10: effect sizes lead, p-values support)."
)

BOOTSTRAP_STATUS = (
    "Graph-level cluster bootstrap (statistics.md D2): graphs are resampled "
    "with replacement and every statistic is recomputed over the induced "
    "pair submatrix. Effective sample size is governed by the number of "
    "graphs, not pairs -- LINUX contributes 89 graphs, not 3,916 independent "
    "observations. Intervals are wider than pair-level intervals would be; "
    "that is the correct outcome and it is reported. Method comparisons "
    "within a dataset use the same resample for both methods and the "
    "difference is taken per replicate (D7)."
)

MAX_PROCESSES = 8


class BakeoffAnalysisError(Exception):
    """Raised when a contract violation makes the analysis unsafe to continue."""


# ---------------------------------------------------------------------------
# Rank statistics
# ---------------------------------------------------------------------------


def midranks(values: np.ndarray) -> np.ndarray:
    """Return average (mid) ranks of *values*, ties averaged.

    Uses a counting-sort fast path when the input is integral with a
    small range, which every quantity here is: GED, its bounds and the
    Levenshtein distance are all small non-negative integers. That turns
    the per-replicate cost from ``O(p log p)`` to ``O(p + K)``, which is
    what makes a 2,000-replicate bootstrap over 2.1 M induced pairs
    affordable. Falls back to :func:`scipy.stats.rankdata` otherwise.

    Args:
        values: One-dimensional array of observations.

    Returns:
        Float array of 1-based average ranks, same shape as *values*.
    """
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.all() and np.array_equal(values, np.floor(values)):
        vmin = float(values.min())
        vmax = float(values.max())
        span = vmax - vmin
        if span < 1 << 20:
            shifted = (values - vmin).astype(np.int64)
            counts = np.bincount(shifted, minlength=int(span) + 1)
            cum = np.cumsum(counts)
            strictly_less = cum - counts
            mid = strictly_less + (counts + 1) / 2.0
            return np.asarray(mid[shifted], dtype=np.float64)
    return np.asarray(stats.rankdata(values), dtype=np.float64)


def spearman_from_ranks(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """Return Spearman rho given pre-computed midranks.

    Spearman with ties is by definition the Pearson correlation of the
    midranks, so this is exact rather than an approximation.

    Args:
        rank_a: Midranks of the first variable.
        rank_b: Midranks of the second variable.

    Returns:
        Spearman rho, or ``nan`` when either variable is constant.
    """
    if rank_a.size < 2:
        return float("nan")
    a = rank_a - rank_a.mean()
    b = rank_b - rank_b.mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    if denom <= 0.0:
        return float("nan")
    return float((a @ b) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman rho between *x* and *y*.

    Args:
        x: First variable.
        y: Second variable.

    Returns:
        Spearman rho, or ``nan`` when either variable is constant.
    """
    return spearman_from_ranks(midranks(x), midranks(y))


@dataclass(frozen=True)
class WilcoxonResult:
    """Outcome of a matched-pairs Wilcoxon signed-rank test.

    Attributes:
        statistic: The smaller of the two signed rank sums.
        p_value: Two-sided asymptotic p-value.
        rank_biserial: Matched-pairs rank-biserial correlation,
            ``(R+ - R-) / (R+ + R-)``. This is the effect size that
            leads; the p-value supports it.
        n_used: Non-zero differences entering the test.
        n_zero: Discarded zero differences (Wilcoxon's own method).
        z: The normal deviate behind *p_value*.
    """

    statistic: float
    p_value: float
    rank_biserial: float
    n_used: int
    n_zero: int
    z: float


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> WilcoxonResult:
    """Wilcoxon signed-rank test with the rank-biserial effect size.

    Implemented directly rather than through :func:`scipy.stats.wilcoxon`
    for one reason: the rank-biserial correlation needs ``R+`` and ``R-``
    from the *same* midranks that produced the statistic, and scipy does
    not return them. Parity with scipy's asymptotic mode (no continuity
    correction, ``zero_method="wilcox"``) is asserted in the unit tests.

    Args:
        x: First paired sample.
        y: Second paired sample, same length.

    Returns:
        The test outcome.

    Raises:
        BakeoffAnalysisError: If the samples differ in length.
    """
    if x.shape != y.shape:
        raise BakeoffAnalysisError(f"paired samples differ in shape: {x.shape} vs {y.shape}")
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    nonzero = diff[diff != 0.0]
    n_zero = int(diff.size - nonzero.size)
    n = int(nonzero.size)
    if n == 0:
        return WilcoxonResult(0.0, 1.0, 0.0, 0, n_zero, 0.0)

    ranks = midranks(np.abs(nonzero))
    r_plus = float(ranks[nonzero > 0].sum())
    r_minus = float(ranks[nonzero < 0].sum())
    total = r_plus + r_minus
    rank_biserial = (r_plus - r_minus) / total if total > 0 else 0.0

    mean = n * (n + 1) / 4.0
    _, tie_counts = np.unique(ranks, return_counts=True)
    tie_correction = float(((tie_counts**3) - tie_counts).sum()) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    if var <= 0.0:
        return WilcoxonResult(min(r_plus, r_minus), 1.0, rank_biserial, n, n_zero, 0.0)
    z = (r_plus - mean) / math.sqrt(var)
    p = 2.0 * float(stats.norm.sf(abs(z)))
    return WilcoxonResult(
        statistic=min(r_plus, r_minus),
        p_value=min(1.0, p),
        rank_biserial=rank_biserial,
        n_used=n,
        n_zero=n_zero,
        z=float(z),
    )


def nemenyi_critical_difference(k: int, n_datasets: int) -> float:
    """Return the Nemenyi critical difference at alpha = 0.05.

    Args:
        k: Number of methods compared.
        n_datasets: Number of datasets in the omnibus.

    Returns:
        ``q_0.05 * sqrt(k (k + 1) / (6 N))``; ``nan`` outside the table.
    """
    q = NEMENYI_Q05.get(k)
    if q is None or n_datasets <= 0:
        return float("nan")
    return q * math.sqrt(k * (k + 1) / (6.0 * n_datasets))


def rank_cliques(average_ranks: Sequence[float], cd: float) -> tuple[tuple[int, ...], ...]:
    """Return maximal groups of methods not separated by *cd*.

    Args:
        average_ranks: Average rank per method, in method order.
        cd: The critical difference.

    Returns:
        Tuples of method indices, each a maximal set whose rank spread is
        below *cd*. Single-method groups are dropped.
    """
    if not average_ranks or not math.isfinite(cd):
        return ()
    order = sorted(range(len(average_ranks)), key=lambda k: average_ranks[k])
    groups: list[tuple[int, ...]] = []
    for start in range(len(order)):
        end = start
        while (
            end + 1 < len(order)
            and average_ranks[order[end + 1]] - average_ranks[order[start]] < cd
        ):
            end += 1
        if end > start:
            groups.append(tuple(order[start : end + 1]))
    maximal = [g for g in groups if not any(set(g) < set(h) for h in groups)]
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for g in maximal:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return tuple(out)


# ---------------------------------------------------------------------------
# Loading -- CONTRACTS §2 and §3
# ---------------------------------------------------------------------------

LEV_VARIANTS: tuple[str, ...] = ("exhaustive", "greedy", "greedy_single")
PRIMARY_LEV = "exhaustive"


@dataclass
class IndexData:
    """One dataset's ground truth, in the canonical pair order.

    Attributes:
        dataset: Dataset key.
        n_graphs: Number of graphs.
        exact: Ground-truth GED; ``inf`` where the solver was censored.
        exact_lb: Solver lower bracket.
        exact_ub: Solver upper bracket.
        certified: Whether the exact solver closed the bracket.
        n_max: ``max(n_i, n_j)`` per pair.
        lev: Levenshtein distance per encoder variant.
        node_counts: Nodes per graph.
        meta: The index file's ``meta`` JSON.
    """

    dataset: str
    n_graphs: int
    exact: np.ndarray
    exact_lb: np.ndarray
    exact_ub: np.ndarray
    certified: np.ndarray
    n_max: np.ndarray
    lev: dict[str, np.ndarray]
    node_counts: np.ndarray
    meta: dict[str, Any]

    @property
    def n_pairs(self) -> int:
        """Number of pairs, ``n (n - 1) / 2``."""
        return int(self.exact.size)


@dataclass
class CellData:
    """One (dataset x method) cell.

    Attributes:
        dataset: Dataset key.
        method: GEDLIB method name.
        end: ``"lower"`` or ``"upper"``.
        value: The reported bound, in canonical pair order.
        value_fwd: Orientation ``(i, j)``.
        value_rev: Orientation ``(j, i)``; ``None`` for lower bounds.
        meta: The cell file's ``meta`` JSON.
    """

    dataset: str
    method: str
    end: str
    value: np.ndarray
    value_fwd: np.ndarray
    value_rev: np.ndarray | None
    meta: dict[str, Any]


def _meta_of(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Decode the zero-dimensional ``meta`` string array of an npz file."""
    if "meta" not in payload:
        return {}
    raw = payload["meta"]
    text = str(raw.item()) if hasattr(raw, "item") else str(raw)
    try:
        decoded: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - malformed input
        raise BakeoffAnalysisError(f"meta is not valid JSON: {exc}") from exc
    return decoded


def load_index(path: Path) -> IndexData:
    """Load and validate ``$OUT/data/index/{ds}.npz``.

    The contract calls the pair order "the spine of the whole wave" and
    the graph-id agreement "the single silent-corruption risk"; both are
    asserted here rather than trusted.

    Args:
        path: Path to the index npz.

    Returns:
        The validated index.

    Raises:
        BakeoffAnalysisError: On any contract violation.
    """
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        required = {
            "pair_i",
            "pair_j",
            "exact",
            "exact_lb",
            "exact_ub",
            "certified",
            "n_max",
            "graph_ids",
            "node_counts",
            "edge_counts",
        }
        missing = required - keys
        if missing:
            raise BakeoffAnalysisError(f"{path.name}: missing keys {sorted(missing)}")
        meta = _meta_of(payload)
        n_graphs = int(payload["graph_ids"].size)
        pair_i = np.asarray(payload["pair_i"])
        pair_j = np.asarray(payload["pair_j"])
        expected_i, expected_j = np.triu_indices(n_graphs, k=1)
        if pair_i.size != expected_i.size:
            raise BakeoffAnalysisError(
                f"{path.name}: {pair_i.size} pairs for {n_graphs} graphs, "
                f"expected {expected_i.size}"
            )
        if not (
            np.array_equal(pair_i, expected_i.astype(pair_i.dtype))
            and np.array_equal(pair_j, expected_j.astype(pair_j.dtype))
        ):
            raise BakeoffAnalysisError(f"{path.name}: pair order is not np.triu_indices(n, k=1)")

        exact = np.asarray(payload["exact"], dtype=np.float64)
        exact_lb = np.asarray(payload["exact_lb"], dtype=np.float64)
        exact_ub = np.asarray(payload["exact_ub"], dtype=np.float64)
        certified = np.asarray(payload["certified"], dtype=bool)
        n_max = np.asarray(payload["n_max"], dtype=np.int32)
        node_counts = np.asarray(payload["node_counts"], dtype=np.int32)
        lev = {
            variant: np.asarray(payload[f"lev_{variant}"], dtype=np.float64)
            for variant in LEV_VARIANTS
            if f"lev_{variant}" in keys
        }

    if not np.isfinite(exact[certified]).all():
        raise BakeoffAnalysisError(f"{path.name}: a certified pair carries a non-finite exact GED")
    if not np.allclose(exact_lb[certified], exact[certified]):
        raise BakeoffAnalysisError(f"{path.name}: exact_lb != exact on certified pairs")
    if not np.allclose(exact_ub[certified], exact[certified]):
        raise BakeoffAnalysisError(f"{path.name}: exact_ub != exact on certified pairs")
    if np.isfinite(exact[~certified]).any():
        raise BakeoffAnalysisError(f"{path.name}: a censored pair carries a finite exact GED")
    if (exact_lb[~certified] > exact_ub[~certified]).any():
        raise BakeoffAnalysisError(f"{path.name}: inverted solver bracket on a censored pair")
    if PRIMARY_LEV not in lev:
        raise BakeoffAnalysisError(f"{path.name}: lev_{PRIMARY_LEV} is required for M6")

    dataset = str(meta.get("dataset") or path.stem)
    return IndexData(
        dataset=dataset,
        n_graphs=n_graphs,
        exact=exact,
        exact_lb=exact_lb,
        exact_ub=exact_ub,
        certified=certified,
        n_max=n_max,
        lev=lev,
        node_counts=node_counts,
        meta=meta,
    )


def load_cell(path: Path, index: IndexData) -> CellData:
    """Load and validate one ``$OUT/data/cells/{ds}__{METHOD}.npz``.

    A lower-bound value of exactly ``0.0`` on a pair with ``exact > 0``
    is **legal** and is not treated as an error. Under cost model D6
    node and edge substitutions are free, so any degree-preserving
    assignment costs nothing: C6 against two disjoint triangles has exact
    GED 4 and every BRANCH-family bound returns 0.00, correctly. The
    accessor trap the contract warns about is caught in track A's
    harness, at the point where GEDLIB is actually called.

    Args:
        path: Path to the cell npz.
        index: The dataset's index, for length and end checks.

    Returns:
        The validated cell.

    Raises:
        BakeoffAnalysisError: On any contract violation.
    """
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        if "value" not in keys:
            raise BakeoffAnalysisError(f"{path.name}: no 'value' array")
        meta = _meta_of(payload)
        value = np.asarray(payload["value"], dtype=np.float64)
        value_fwd = np.asarray(payload.get("value_fwd", payload["value"]), dtype=np.float64)
        value_rev = (
            np.asarray(payload["value_rev"], dtype=np.float64) if "value_rev" in keys else None
        )

    stem = path.stem
    method = str(meta.get("method") or (stem.split("__", 1)[1] if "__" in stem else stem))
    end = str(meta.get("end") or end_of_method(method))
    if end != end_of_method(method):
        raise BakeoffAnalysisError(f"{path.name}: meta end {end!r} disagrees with the roster")
    if value.size != index.n_pairs:
        raise BakeoffAnalysisError(f"{path.name}: {value.size} values for {index.n_pairs} pairs")
    if not np.isfinite(value).all():
        raise BakeoffAnalysisError(f"{path.name}: non-finite bound value")
    if (value < 0).any():
        raise BakeoffAnalysisError(f"{path.name}: negative bound value")
    if end == "lower" and value_rev is not None:
        raise BakeoffAnalysisError(f"{path.name}: a lower-bound cell must omit value_rev")
    if end == "upper":
        if value_rev is None:
            raise BakeoffAnalysisError(f"{path.name}: an upper-bound cell must carry value_rev")
        if not np.allclose(value, np.minimum(value_fwd, value_rev)):
            raise BakeoffAnalysisError(
                f"{path.name}: value is not min(value_fwd, value_rev) (CONTRACTS §3)"
            )
    return CellData(
        dataset=index.dataset,
        method=method,
        end=end,
        value=value,
        value_fwd=value_fwd,
        value_rev=value_rev,
        meta=meta,
    )


def discover_cells(root: Path, dataset: str) -> list[Path]:
    """Return the cell files present for *dataset*, sorted by method name.

    Args:
        root: The report root holding ``data/cells``.
        dataset: Dataset key.

    Returns:
        Paths to the dataset's cell npz files.
    """
    return sorted((root / "data" / "cells").glob(f"{dataset}__*.npz"))


# ---------------------------------------------------------------------------
# M1-M8
# ---------------------------------------------------------------------------


def signed_error(end: str, value: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """Return the slack of a bound against the exact GED.

    Non-negative when the bound is valid: ``exact - LB`` for a lower
    bound, ``UB - exact`` for an upper bound. A negative entry is an M4
    violation.

    Args:
        end: ``"lower"`` or ``"upper"``.
        value: The reported bound.
        exact: The ground-truth GED.

    Returns:
        The slack array.
    """
    return exact - value if end == "lower" else value - exact


@dataclass(frozen=True)
class ErrorStats:
    """Distributional summary of one error array.

    Attributes:
        n: Observations behind the summary.
        mean: Arithmetic mean.
        median: Median.
        q25: First quartile.
        q75: Third quartile.
        iqr: ``q75 - q25``.
        p95: 95th percentile.
    """

    n: int
    mean: float
    median: float
    q25: float
    q75: float
    iqr: float
    p95: float

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-ready mapping."""
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "iqr": self.iqr,
            "p95": self.p95,
        }


def error_stats(values: np.ndarray) -> ErrorStats:
    """Summarise an error array.

    Args:
        values: Error observations.

    Returns:
        Mean, median, quartiles, IQR and the 95th percentile; all ``nan``
        for an empty input.
    """
    n = int(values.size)
    if n == 0:
        nan = float("nan")
        return ErrorStats(0, nan, nan, nan, nan, nan, nan)
    q25, med, q75, p95 = (float(v) for v in np.percentile(values, [25.0, 50.0, 75.0, 95.0]))
    return ErrorStats(
        n=n,
        mean=float(values.mean()),
        median=med,
        q25=q25,
        q75=q75,
        iqr=q75 - q25,
        p95=p95,
    )


@dataclass(frozen=True)
class ValidityResult:
    """M4 over every pair, two-sided on certified and one-sided on censored.

    Attributes:
        n_checked: All pairs in the dataset.
        n_two_sided: Certified pairs, where ``exact`` is known exactly.
        n_one_sided: Censored pairs, where only the solver bracket is known.
        violations: Refuted pairs, over both regimes.
        examples: Up to ten refuted pairs, for triage.
    """

    n_checked: int
    n_two_sided: int
    n_one_sided: int
    violations: int
    examples: tuple[dict[str, float | int | str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping."""
        return {
            "n_checked": self.n_checked,
            "n_two_sided": self.n_two_sided,
            "n_one_sided": self.n_one_sided,
            "violations": self.violations,
            "examples": list(self.examples),
        }


def compute_validity(cell: CellData, index: IndexData, *, max_examples: int = 10) -> ValidityResult:
    """Evaluate M4 on all pairs (design §3.5).

    Certified pairs get the two-sided test against the exact value.
    Censored pairs get the one-sided test the solver bracket still
    licenses: a lower bound is refuted iff ``LB > exact_ub``, an upper
    bound iff ``UB < exact_lb``. That recovers 61,084 extra validity
    checks at no extra compute, on precisely the pairs where a bound is
    most likely to break.

    ``LB == 0`` on a pair with ``exact > 0`` is **not** a violation.
    Under D6 both substitutions are free, so any degree-preserving
    assignment costs nothing and a zero lower bound is correct for two
    non-isomorphic graphs sharing a degree sequence.

    Args:
        cell: The cell to check.
        index: The dataset index.
        max_examples: How many refuted pairs to record.

    Returns:
        The M4 result.
    """
    certified = index.certified
    censored = ~certified
    if cell.end == "lower":
        bad_two = certified & (cell.value > index.exact)
        bad_one = censored & (cell.value > index.exact_ub)
    else:
        bad_two = certified & (cell.value < index.exact)
        bad_one = censored & (cell.value < index.exact_lb)
    bad = bad_two | bad_one

    pair_i, pair_j = np.triu_indices(index.n_graphs, k=1)
    examples: list[dict[str, float | int | str]] = []
    for flat in np.flatnonzero(bad)[:max_examples]:
        examples.append(
            {
                "pair_index": int(flat),
                "graph_i": int(pair_i[flat]),
                "graph_j": int(pair_j[flat]),
                "value": float(cell.value[flat]),
                "exact": float(index.exact[flat]),
                "exact_lb": float(index.exact_lb[flat]),
                "exact_ub": float(index.exact_ub[flat]),
                "regime": "certified" if certified[flat] else "censored",
            }
        )
    return ValidityResult(
        n_checked=index.n_pairs,
        n_two_sided=int(certified.sum()),
        n_one_sided=int(censored.sum()),
        violations=int(bad.sum()),
        examples=tuple(examples),
    )


def compute_symmetry(cell: CellData) -> dict[str, float | int]:
    """Evaluate M8 from the two orientations (design §3.6).

    Args:
        cell: The cell to check.

    Returns:
        Disagreement fraction and the mean gain the ``min`` buys over a
        single orientation. Lower-bound cells report
        ``"evaluated": 0`` -- they are run in one orientation, and their
        symmetry spot-check belongs to track A's harness.
    """
    if cell.value_rev is None:
        return {"evaluated": 0, "frac_asymmetric": float("nan"), "mean_gain_over_fwd": 0.0}
    diff = cell.value_fwd - cell.value_rev
    return {
        "evaluated": 1,
        "n_pairs": int(cell.value.size),
        "frac_asymmetric": float((diff != 0).mean()),
        "mean_gain_over_fwd": float((cell.value_fwd - cell.value).mean()),
        "mean_abs_spread": float(np.abs(diff).mean()),
        "max_abs_spread": float(np.abs(diff).max()),
    }


def load_timing(root: Path, dataset: str, method: str) -> dict[str, Any]:
    """Read M7 from track A's timing JSON; never measure it here.

    Args:
        root: Report root.
        dataset: Dataset key.
        method: Method name.

    Returns:
        ``{"dataset": ..., "probe": ..., "gate": ...}``. The gate is
        ``"unevaluated"`` when the design §3.4 probe at n-bar ~ 30 is
        absent -- never ``"pass"``, because the bake-off corpus is
        ``n <= 12`` by construction and cannot evaluate it.
    """
    timing_dir = root / "data" / "timing"
    per_dataset: dict[str, Any] | None = None
    probe: dict[str, Any] | None = None
    ds_path = timing_dir / f"{dataset}__{method}.json"
    probe_path = timing_dir / f"probe_n30__{method}.json"
    if ds_path.is_file():
        per_dataset = json.loads(ds_path.read_text(encoding="utf-8"))
    if probe_path.is_file():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe is None:
        gate = "unevaluated"
    else:
        rate = float(probe.get("us_per_pair_mean", float("nan")))
        gate = "pass" if rate < COST_GATE_US_PER_PAIR else "fail"
    return {
        "dataset_timing": per_dataset,
        "probe_n30": probe,
        "gate": gate,
        "gate_threshold_us_per_pair": COST_GATE_US_PER_PAIR,
    }


@dataclass
class CellMetrics:
    """M1-M8 for one cell, over both §3.3 domains."""

    dataset: str
    method: str
    end: str
    payload: dict[str, Any]


def compute_cell_metrics(
    cell: CellData,
    index: IndexData,
    root: Path,
    *,
    lev_variant: str = PRIMARY_LEV,
) -> CellMetrics:
    """Compute M1-M8 for one cell.

    M1, M2 and M3 are each reported twice (design §3.3): over all
    certified pairs, and over ``exact > 0`` only, the second being the
    headline. On Letter LOW 15.5 % of pairs have exact GED 0, where
    relative error is undefined and every valid lower bound certifies for
    free.

    M1's two domains coincide by construction -- relative error is
    undefined at ``exact = 0``, so those pairs leave both. The count of
    excluded pairs is reported so the coincidence is visible rather than
    implied. M2 and M3 genuinely differ between the domains.

    M1/M2/M3/M5/M6 use certified pairs only (§3.5); M4 uses all pairs.

    Args:
        cell: The cell.
        index: The dataset index.
        root: Report root, for the M7 timing JSON.
        lev_variant: Encoder variant behind M6.

    Returns:
        The metric payload.
    """
    certified = index.certified
    exact_c = index.exact[certified]
    value_c = cell.value[certified]
    abs_err = signed_error(cell.end, value_c, exact_c)
    positive = exact_c > 0
    n_undefined = int((~positive).sum())

    rel_err = abs_err[positive] / exact_c[positive]
    m1 = error_stats(rel_err)
    m2_all = error_stats(abs_err)
    m2_pos = error_stats(abs_err[positive])
    certifies = abs_err == 0
    m3_all = float(certifies.mean()) if certifies.size else float("nan")
    m3_pos = float(certifies[positive].mean()) if positive.any() else float("nan")

    lev_c = index.lev[lev_variant][certified]
    rank_exact = midranks(exact_c)
    rank_value = midranks(value_c)
    rank_lev = midranks(lev_c)
    rho_bound_exact = spearman_from_ranks(rank_value, rank_exact)
    rho_lev_bound = spearman_from_ranks(rank_lev, rank_value)
    rho_lev_exact = spearman_from_ranks(rank_lev, rank_exact)

    payload: dict[str, Any] = {
        "dataset": cell.dataset,
        "method": cell.method,
        "end": cell.end,
        "n_pairs": index.n_pairs,
        "n_certified": int(certified.sum()),
        "n_censored": int((~certified).sum()),
        "n_exact_zero_certified": n_undefined,
        "M1_relative_error": {
            "all_certified": m1.as_dict(),
            "exact_gt_zero": m1.as_dict(),
            "n_undefined_excluded": n_undefined,
            "domains_coincide": True,
            "note": (
                "Relative error is undefined at exact = 0, so those pairs leave "
                "both domains and the two reports are identical by construction. "
                "The count is reported so the coincidence is visible."
            ),
        },
        "M2_absolute_error": {
            "all_certified": m2_all.as_dict(),
            "exact_gt_zero": m2_pos.as_dict(),
        },
        "M3_certification_rate": {
            "all_certified": m3_all,
            "exact_gt_zero": m3_pos,
            "headline": "exact_gt_zero",
            "well_defined": bool(positive.any()),
        },
        "M4_validity": compute_validity(cell, index).as_dict(),
        "M5_rho_bound_exact": {"point": rho_bound_exact, "n": int(certified.sum())},
        "M6_rho_lev": {
            "variant": lev_variant,
            "rho_lev_bound": rho_lev_bound,
            "rho_lev_exact_anchor": rho_lev_exact,
            "gap": rho_lev_bound - rho_lev_exact,
            "abs_gap": abs(rho_lev_bound - rho_lev_exact),
        },
        "M7_cost": load_timing(root, cell.dataset, cell.method),
        "M8_symmetry": compute_symmetry(cell),
    }
    return CellMetrics(cell.dataset, cell.method, cell.end, payload)


def error_vs_n(
    cell: CellData,
    index: IndexData,
    *,
    min_count: int = 1,
) -> dict[str, list[float] | list[int]]:
    """Bin relative error by ``max(n1, n2)`` for panel (a).

    Only certified pairs with ``exact > 0`` enter, matching the §3.5 and
    §3.3 domains of M1.

    Args:
        cell: The cell.
        index: The dataset index.
        min_count: Bins holding fewer pairs are dropped.

    Returns:
        Parallel lists ``n_values``, ``mean``, ``q25``, ``q75``, ``counts``.
    """
    mask = index.certified & (index.exact > 0)
    exact = index.exact[mask]
    rel = signed_error(cell.end, cell.value[mask], exact) / exact
    sizes = index.n_max[mask]

    n_values: list[int] = []
    means: list[float] = []
    q25: list[float] = []
    q75: list[float] = []
    counts: list[int] = []
    for n in np.unique(sizes):
        bucket = rel[sizes == n]
        if bucket.size < min_count:
            continue
        lo, hi = (float(v) for v in np.percentile(bucket, [25.0, 75.0]))
        n_values.append(int(n))
        means.append(float(bucket.mean()))
        q25.append(lo)
        q75.append(hi)
        counts.append(int(bucket.size))
    return {
        "n_values": n_values,
        "mean": means,
        "q25": q25,
        "q75": q75,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Graph-level cluster bootstrap -- statistics.md D2 / D7, design §3.7
# ---------------------------------------------------------------------------


def pair_flat_index(n_graphs: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map graph pairs ``(lo, hi)`` with ``lo < hi`` to canonical pair indices.

    Inverts ``np.triu_indices(n, k=1)`` in closed form, which is what
    makes a graph-level resample cheap: the induced pair submatrix is a
    gather, not a search.

    Args:
        n_graphs: Number of graphs in the dataset.
        lo: Smaller graph index of each pair.
        hi: Larger graph index of each pair.

    Returns:
        Flat indices into the canonical pair order.
    """
    lo64 = lo.astype(np.int64)
    hi64 = hi.astype(np.int64)
    return lo64 * n_graphs - (lo64 * (lo64 + 1)) // 2 + (hi64 - lo64 - 1)


def induced_pairs(n_graphs: int, selection: np.ndarray) -> np.ndarray:
    """Return the canonical pair indices induced by a graph resample.

    Slots holding the same original graph induce a self-pair, which has
    no observation in an upper-triangular matrix and is dropped. Every
    other unordered slot pair contributes, duplicates included: that
    duplication is the cluster bootstrap's variance mechanism and must
    not be de-duplicated away.

    Args:
        n_graphs: Number of graphs in the dataset.
        selection: Resampled graph indices, length ``n_graphs``.

    Returns:
        Flat pair indices, with repetitions.
    """
    tri_i, tri_j = np.triu_indices(selection.size, k=1)
    a = selection[tri_i]
    b = selection[tri_j]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keep = lo != hi
    return pair_flat_index(n_graphs, lo[keep], hi[keep])


def replicate_selection(n_graphs: int, seed: int, replicate: int) -> np.ndarray:
    """Draw the graph resample for one replicate.

    Seeding is per replicate via ``SeedSequence([seed, replicate])``, so
    the draw does not depend on loop order and the result is identical
    whether the bootstrap runs serially or across processes.

    Args:
        n_graphs: Number of graphs.
        seed: Master seed (42, frozen).
        replicate: Zero-based replicate number.

    Returns:
        ``n_graphs`` graph indices drawn with replacement.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
    return rng.integers(0, n_graphs, size=n_graphs, dtype=np.int64)


#: Fork-inherited payload for the worker processes. Set by
#: :func:`bootstrap_dataset` before the pool is created, so the large
#: arrays are shared copy-on-write rather than pickled per task.
_BOOT_STATE: dict[str, Any] = {}


def _replicate_statistics(state: Mapping[str, Any], replicate: int) -> dict[str, float]:
    """Compute every bootstrap statistic on one graph resample.

    All statistics share one resample, so the D7 paired differences are
    correct by construction rather than by a matching step afterwards.

    Args:
        state: The bootstrap payload.
        replicate: Zero-based replicate number.

    Returns:
        Mapping from statistic key to value.
    """
    n_graphs = int(state["n_graphs"])
    selection = replicate_selection(n_graphs, int(state["seed"]), replicate)
    flat = induced_pairs(n_graphs, selection)
    certified: np.ndarray = state["certified"]
    return _statistics_on_index(state, flat[certified[flat]])


def _worker_chunk(bounds: tuple[int, int]) -> list[dict[str, float]]:
    """Run a contiguous block of replicates inside a worker process."""
    start, stop = bounds
    return [_replicate_statistics(_BOOT_STATE, r) for r in range(start, stop)]


def _percentile_ci(samples: Sequence[float]) -> dict[str, float | int]:
    """Return the percentile CI of a bootstrap distribution.

    Args:
        samples: Replicate values, possibly containing ``nan``.

    Returns:
        ``ci_low``, ``ci_high``, ``bootstrap_mean``, ``bootstrap_sd`` and
        the number of finite replicates behind them.
    """
    arr = np.asarray([s for s in samples if math.isfinite(s)], dtype=np.float64)
    if arr.size == 0:
        nan = float("nan")
        return {"ci_low": nan, "ci_high": nan, "bootstrap_mean": nan, "bootstrap_sd": nan, "n": 0}
    lo, hi = (float(v) for v in np.percentile(arr, list(CI_PERCENTILES)))
    return {
        "ci_low": lo,
        "ci_high": hi,
        "bootstrap_mean": float(arr.mean()),
        "bootstrap_sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def bootstrap_dataset(
    index: IndexData,
    cells: Sequence[CellData],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    jobs: int = 1,
    lev_variant: str = PRIMARY_LEV,
) -> dict[str, Any]:
    """Run the graph-level bootstrap for one dataset.

    Args:
        index: The dataset index.
        cells: Every cell of that dataset.
        replicates: Replicate count; 2,000 in production (D2).
        seed: Master seed; 42 in production.
        jobs: Worker processes, capped at :data:`MAX_PROCESSES`.
        lev_variant: Encoder variant behind M6.

    Returns:
        Point estimates, percentile CIs and the D7 paired differences.
    """
    state = {
        "n_graphs": index.n_graphs,
        "seed": seed,
        "certified": index.certified,
        "exact": index.exact,
        "lev": index.lev[lev_variant],
        "values": {c.method: c.value for c in cells},
        "ends": {c.method: c.end for c in cells},
        "rosters": {
            end: tuple(m for m in methods_for_end(end) if any(c.method == m for c in cells))
            for end in ENDS
        },
    }

    started = time.perf_counter()
    n_jobs = max(1, min(int(jobs), MAX_PROCESSES))
    if n_jobs == 1 or replicates < 2 * n_jobs:
        samples = [_replicate_statistics(state, r) for r in range(replicates)]
    else:
        _BOOT_STATE.clear()
        _BOOT_STATE.update(state)
        edges = np.linspace(0, replicates, n_jobs + 1).astype(int)
        chunks = [(int(edges[k]), int(edges[k + 1])) for k in range(n_jobs)]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            samples = [row for block in pool.map(_worker_chunk, chunks) for row in block]
        _BOOT_STATE.clear()
    elapsed = time.perf_counter() - started

    collected: dict[str, list[float]] = {}
    for row in samples:
        for key, value in row.items():
            collected.setdefault(key, []).append(value)

    point = _replicate_point_estimates(index, cells, lev_variant)
    result: dict[str, Any] = {
        "dataset": index.dataset,
        "n_graphs": index.n_graphs,
        "n_pairs": index.n_pairs,
        "replicates": replicates,
        "seed": seed,
        "lev_variant": lev_variant,
        "wall_seconds": elapsed,
        "jobs": n_jobs,
        "statistics": {},
    }
    for key, values in sorted(collected.items()):
        entry = _percentile_ci(values)
        entry["point"] = point.get(key, float("nan"))
        result["statistics"][key] = entry
    return result


def _replicate_point_estimates(
    index: IndexData,
    cells: Sequence[CellData],
    lev_variant: str,
) -> dict[str, float]:
    """Compute every bootstrap statistic on the observed sample.

    The reported point estimate is the full-sample value, never the mean
    of the bootstrap distribution: the resample is for uncertainty, not
    for the estimate.

    Args:
        index: The dataset index.
        cells: Every cell of the dataset.
        lev_variant: Encoder variant behind M6.

    Returns:
        Mapping from statistic key to observed value.
    """
    state = {
        "n_graphs": index.n_graphs,
        "seed": 0,
        "certified": index.certified,
        "exact": index.exact,
        "lev": index.lev[lev_variant],
        "values": {c.method: c.value for c in cells},
        "ends": {c.method: c.end for c in cells},
        "rosters": {
            end: tuple(m for m in methods_for_end(end) if any(c.method == m for c in cells))
            for end in ENDS
        },
    }
    identity = np.arange(index.n_graphs, dtype=np.int64)
    flat = induced_pairs(index.n_graphs, identity)
    return _statistics_on_index(state, flat[index.certified[flat]])


def _statistics_on_index(state: Mapping[str, Any], idx: np.ndarray) -> dict[str, float]:
    """Evaluate the bootstrap statistic set on an explicit pair index set."""
    out: dict[str, float] = {}
    if idx.size < 2:
        return out
    exact = state["exact"][idx]
    lev = state["lev"][idx]
    rank_exact = midranks(exact)
    rank_lev = midranks(lev)
    rho_lev_exact = spearman_from_ranks(rank_lev, rank_exact)
    out["rho_lev_exact"] = rho_lev_exact
    positive = exact > 0
    exact_pos = exact[positive]
    rel: dict[str, float] = {}
    absolute: dict[str, float] = {}
    for method, values in state["values"].items():
        value = values[idx]
        rank_value = midranks(value)
        out[f"rho_bound_exact::{method}"] = spearman_from_ranks(rank_value, rank_exact)
        rho_lev_bound = spearman_from_ranks(rank_lev, rank_value)
        out[f"rho_lev_bound::{method}"] = rho_lev_bound
        out[f"d_rho_lev::{method}"] = rho_lev_bound - rho_lev_exact
        err = signed_error(state["ends"][method], value, exact)
        absolute[method] = float(err.mean())
        rel[method] = float((err[positive] / exact_pos).mean()) if exact_pos.size else float("nan")
        out[f"mean_abs_err::{method}"] = absolute[method]
        out[f"mean_rel_err::{method}"] = rel[method]
    for end, methods in state["rosters"].items():
        present = [m for m in methods if m in rel]
        for a_idx, method_a in enumerate(present):
            for method_b in present[a_idx + 1 :]:
                key = f"{end}::{method_a}|{method_b}"
                out[f"diff_mean_rel_err::{key}"] = rel[method_a] - rel[method_b]
                out[f"diff_mean_abs_err::{key}"] = absolute[method_a] - absolute[method_b]
    return out


# ---------------------------------------------------------------------------
# Significance -- design §3.8. A selection procedure, not a hypothesis test.
# ---------------------------------------------------------------------------


def relative_errors(cell: CellData, index: IndexData) -> np.ndarray:
    """Return per-pair relative error over certified pairs with ``exact > 0``.

    Args:
        cell: The cell.
        index: The dataset index.

    Returns:
        Relative error, in canonical pair order restricted to the mask.
    """
    mask = index.certified & (index.exact > 0)
    exact = index.exact[mask]
    return signed_error(cell.end, cell.value[mask], exact) / exact


def pairwise_significance(
    end: str,
    dataset: str,
    cells: Mapping[str, CellData],
    index: IndexData,
    bootstrap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Wilcoxon signed-rank with Holm correction inside one (dataset, end).

    The family size is ``C(k, 2)`` for the ``k`` methods present at that
    end -- ten for the five lower bounds, six for the four upper bounds.
    It is derived from the roster, never hard-coded.

    Args:
        end: ``"lower"`` or ``"upper"``.
        dataset: Dataset key.
        cells: Cells of that dataset, keyed by method.
        index: The dataset index.
        bootstrap: The dataset's bootstrap result, for the CI on the
            difference in mean error that carries the evidence.

    Returns:
        One entry per comparison, plus the family size and the status
        text that keeps a reader from reading a p-value as a claim.
    """
    methods = [m for m in methods_for_end(end) if m in cells]
    errors = {m: relative_errors(cells[m], index) for m in methods}
    comparisons: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for a_idx, method_a in enumerate(methods):
        for method_b in methods[a_idx + 1 :]:
            result = wilcoxon_signed_rank(errors[method_a], errors[method_b])
            key = f"diff_mean_rel_err::{end}::{method_a}|{method_b}"
            ci = (bootstrap or {}).get("statistics", {}).get(key, {})
            comparisons.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_pairs": int(errors[method_a].size),
                    "mean_error_a": float(errors[method_a].mean()),
                    "mean_error_b": float(errors[method_b].mean()),
                    "rank_biserial": result.rank_biserial,
                    "wilcoxon_statistic": result.statistic,
                    "z": result.z,
                    "p_raw": result.p_value,
                    "n_used": result.n_used,
                    "n_tied": result.n_zero,
                    "bootstrap_diff_point": ci.get("point"),
                    "bootstrap_diff_ci_low": ci.get("ci_low"),
                    "bootstrap_diff_ci_high": ci.get("ci_high"),
                    "ci_excludes_zero": (
                        None if not ci else bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0)
                    ),
                }
            )
            raw_p.append(result.p_value)

    for comparison, adjusted in zip(comparisons, holm_bonferroni(raw_p), strict=True):
        comparison["p_holm"] = adjusted
    return {
        "dataset": dataset,
        "end": end,
        "methods": methods,
        "family_size": len(comparisons),
        "correction": "holm-bonferroni within (dataset, end)",
        "primary_evidence": "rank_biserial + bootstrap_diff_ci",
        "p_value_status": PVALUE_STATUS,
        "statistical_status": SELECTION_STATUS,
        "comparisons": comparisons,
    }


def friedman_over_datasets(
    end: str,
    scores: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Friedman omnibus and Nemenyi critical difference over the datasets.

    Args:
        end: ``"lower"`` or ``"upper"``.
        scores: ``{dataset: {method: mean relative error}}``. Lower is
            better, so rank 1 is the smallest error.

    Returns:
        Average ranks, the omnibus p-value, the critical difference and
        the cliques it fails to separate, with both caveats attached.
    """
    datasets = sorted(scores)
    methods = [m for m in methods_for_end(end) if all(m in scores[d] for d in datasets)]
    if len(methods) < 2 or len(datasets) < 2:
        return {
            "end": end,
            "datasets": datasets,
            "methods": methods,
            "status": "not evaluable",
            "statistical_status": SELECTION_STATUS,
        }

    matrix = np.array([[scores[d][m] for m in methods] for d in datasets], dtype=np.float64)
    ranks = np.vstack([stats.rankdata(row) for row in matrix])
    average_ranks = ranks.mean(axis=0)
    chi2, p_value = stats.friedmanchisquare(*[matrix[:, k] for k in range(len(methods))])
    cd = nemenyi_critical_difference(len(methods), len(datasets))
    cliques = rank_cliques([float(r) for r in average_ranks], cd)
    separates = any(
        abs(average_ranks[a] - average_ranks[b]) >= cd
        for a in range(len(methods))
        for b in range(a + 1, len(methods))
    )
    return {
        "end": end,
        "datasets": datasets,
        "methods": methods,
        "average_ranks": {m: float(r) for m, r in zip(methods, average_ranks, strict=True)},
        "friedman_chi2": float(chi2),
        "friedman_p": float(p_value),
        "critical_difference": cd,
        "cliques": [[methods[k] for k in clique] for clique in cliques],
        "separates_any_pair": bool(separates),
        "separation_note": (
            "The critical difference separates at least one pair."
            if separates
            else "The critical difference separates nothing: every pairwise rank gap "
            "is below CD, so the diagram supports no ordering claim."
        ),
        "caveat_n5": (
            "N = 5 datasets. Friedman is conservative here and the critical "
            "difference is wide (statistics.md §4), which is why the exact regime "
            "gets no CD diagram in the main analysis."
        ),
        "caveat_non_independence": (
            "The five datasets are not independent: IAM Letter LOW/MED/HIGH are one "
            "15-class corpus at three distortion levels, so the omnibus sees 3 + 1 + 1, "
            "not 5 (design §3.2)."
        ),
        "statistical_status": SELECTION_STATUS,
    }


# ---------------------------------------------------------------------------
# Selection -- spec §5 verbatim, with the §3.1 and §3.2 companions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One method's evidence on one dataset, for the frozen rule.

    Attributes:
        method: Method name.
        mean_relative_error: The frozen ranking criterion (M1).
        mean_absolute_error: The §3.1 companion criterion (M2).
        violations: M4; a non-zero value disqualifies.
        m3_well_defined: Whether M3 has a defined headline value.
        cost_gate: ``"pass"``, ``"fail"`` or ``"unevaluated"``.
        cost_us_per_pair: M7 at the dataset's own n-bar, for tie-breaking.
        m6_abs_gap: ``|rho(Lev, bound) - rho(Lev, exact)|``, the second
            tie-break.
    """

    method: str
    mean_relative_error: float
    mean_absolute_error: float
    violations: int
    m3_well_defined: bool
    cost_gate: str
    cost_us_per_pair: float
    m6_abs_gap: float


def _eligible(candidate: Candidate) -> tuple[bool, str]:
    """Return eligibility under the frozen rule's side conditions."""
    if candidate.violations != 0:
        return False, "M4 violation"
    if not candidate.m3_well_defined:
        return False, "M3 undefined"
    if candidate.cost_gate == "fail":
        return False, "M7 gate failed on the n=30 probe"
    if not math.isfinite(candidate.mean_relative_error):
        return False, "mean relative error undefined"
    return True, ""


def select_on_dataset(
    candidates: Sequence[Candidate],
    *,
    criterion: str = "relative",
) -> dict[str, Any]:
    """Apply the frozen §5 rule to one dataset.

    The primary is the eligible method minimising mean relative error.
    Methods within :data:`TIE_TOLERANCE` relative of the best form a tie
    group, broken on M7 cost and then on M6 agreement with
    ``rho(Lev, exact)``. Ties never break on which method flatters
    IsalGraph's rho.

    Args:
        candidates: The dataset's candidates.
        criterion: ``"relative"`` for the frozen rule, ``"absolute"`` for
            the §3.1 companion.

    Returns:
        Winner, ranking, margin and the tie-break trace.
    """
    eligible: list[Candidate] = []
    excluded: list[dict[str, str]] = []
    for candidate in candidates:
        ok, reason = _eligible(candidate)
        if ok:
            eligible.append(candidate)
        else:
            excluded.append({"method": candidate.method, "reason": reason})

    def score(candidate: Candidate) -> float:
        return (
            candidate.mean_relative_error
            if criterion == "relative"
            else candidate.mean_absolute_error
        )

    if not eligible:
        return {
            "winner": None,
            "criterion": criterion,
            "ranking": [],
            "excluded": excluded,
            "margin_relative": None,
            "tie_group": [],
            "tie_break": None,
        }

    ordered = sorted(eligible, key=score)
    best = score(ordered[0])
    tie_group = [
        c for c in ordered if math.isfinite(best) and score(c) <= best * (1 + TIE_TOLERANCE)
    ]
    tie_break: str | None = None
    winner = ordered[0]
    if len(tie_group) > 1:
        by_cost = sorted(
            tie_group,
            key=lambda c: (
                c.cost_us_per_pair if math.isfinite(c.cost_us_per_pair) else math.inf,
                c.m6_abs_gap if math.isfinite(c.m6_abs_gap) else math.inf,
                c.method,
            ),
        )
        winner = by_cost[0]
        tie_break = "M7 cost, then M6 agreement with rho(Lev, exact)"

    runner_up = next((c for c in ordered if c.method != winner.method), None)
    margin = None
    if runner_up is not None and math.isfinite(score(winner)) and score(winner) > 0:
        margin = (score(runner_up) - score(winner)) / score(winner)
    return {
        "winner": winner.method,
        "criterion": criterion,
        "ranking": [c.method for c in ordered],
        "scores": {c.method: score(c) for c in ordered},
        "excluded": excluded,
        "margin_relative": margin,
        "tie_group": [c.method for c in tie_group],
        "tie_break": tie_break,
    }


def collapse_to_corpora(per_dataset_winner: Mapping[str, str | None]) -> dict[str, Any]:
    """Apply the design §3.2 corpus-collapsed companion vote.

    Three units -- Letter, LINUX, AIDS. Letter's vote is the majority of
    its three distortion levels; with three distinct level winners it
    has no vote. A global primary is declared only if one method wins
    all three corpora.

    Args:
        per_dataset_winner: The frozen rule's winner per dataset.

    Returns:
        The corpus winners and the companion global primary.
    """
    by_corpus: dict[str, list[str]] = {}
    for dataset, winner in per_dataset_winner.items():
        if winner is None:
            continue
        by_corpus.setdefault(CORPUS_OF.get(dataset, dataset), []).append(winner)

    corpus_winner: dict[str, str | None] = {}
    for corpus, winners in by_corpus.items():
        counts: dict[str, int] = {}
        for w in winners:
            counts[w] = counts.get(w, 0) + 1
        best = max(counts.values())
        leaders = sorted(m for m, c in counts.items() if c == best)
        needed = len(winners) / 2.0
        corpus_winner[corpus] = leaders[0] if len(leaders) == 1 and best > needed else None

    voted = [w for w in corpus_winner.values() if w is not None]
    unanimous = (
        voted[0]
        if len(corpus_winner) > 0 and len(voted) == len(corpus_winner) and len(set(voted)) == 1
        else None
    )
    return {
        "corpus_of": dict(CORPUS_OF),
        "corpus_winner": corpus_winner,
        "global_primary": unanimous,
        "rule": "global primary iff one method wins all three corpora",
        "rationale": (
            "IAM Letter LOW/MED/HIGH are one 15-class corpus at three distortion "
            "levels, so the frozen five-dataset vote is really 3 + 1 + 1 and a "
            "Letter-favouring method starts with three votes of five (design §3.2)."
        ),
    }


def select_end(
    end: str,
    candidates_by_dataset: Mapping[str, Sequence[Candidate]],
) -> dict[str, Any]:
    """Run the frozen rule and both companions for one bracket end.

    Args:
        end: ``"lower"`` or ``"upper"``.
        candidates_by_dataset: Candidates keyed by dataset.

    Returns:
        The frozen outcome, the §3.1 absolute-error companion, the §3.2
        corpus-collapsed companion, and whether they agree.
    """
    frozen = {d: select_on_dataset(c) for d, c in candidates_by_dataset.items()}
    absolute = {
        d: select_on_dataset(c, criterion="absolute") for d, c in candidates_by_dataset.items()
    }

    winners = {d: r["winner"] for d, r in frozen.items()}
    counts: dict[str, int] = {}
    for winner in winners.values():
        if winner is not None:
            counts[winner] = counts.get(winner, 0) + 1
    n_datasets = len(frozen)
    threshold = 4 if n_datasets == 5 else math.ceil(0.8 * n_datasets)
    global_primary = next(
        (m for m, c in sorted(counts.items()) if c >= threshold),
        None,
    )
    branch = (
        f"global primary declared: {global_primary} wins {counts.get(global_primary, 0)} "
        f"of {n_datasets} datasets"
        if global_primary is not None
        else f"no method wins >= {threshold} of {n_datasets}; primary is declared per dataset"
    )

    rankings_agree = {d: frozen[d]["ranking"] == absolute[d]["ranking"] for d in sorted(frozen)}
    winners_agree = {d: frozen[d]["winner"] == absolute[d]["winner"] for d in sorted(frozen)}
    companion = collapse_to_corpora(winners)
    return {
        "end": end,
        "frozen_rule": {
            "text": (
                "Per end, the primary method is the one minimising mean relative "
                "error (M1) on that dataset, subject to M4 = 0, M3 well-defined and "
                "M7 < 1 ms/pair at n-bar = 30. A single global primary is declared "
                "only if the same method wins on >= 4 of the 5 datasets. Ties "
                "(within 2 % relative) break on M7, then on M6. Ties never break on "
                "which method flatters IsalGraph's rho."
            ),
            "per_dataset": frozen,
            "win_counts": counts,
            "threshold": threshold,
            "global_primary": global_primary,
            "branch": branch,
        },
        "companion_absolute_error": {
            "per_dataset": absolute,
            "rankings_agree": rankings_agree,
            "winners_agree": winners_agree,
            "all_agree": all(rankings_agree.values()),
            "note": (
                "design §3.1: mean absolute error is ranked in parallel. A "
                "disagreement is a finding about the metric, never grounds to "
                "override the frozen rule."
            ),
        },
        "companion_corpus_collapsed": companion,
        "companions_agree_with_frozen": bool(
            all(winners_agree.values()) and companion["global_primary"] == global_primary
        ),
        "statistical_status": SELECTION_STATUS,
    }


# ---------------------------------------------------------------------------
# Synthetic fixture -- CONTRACTS §2/§3/§4/§5, so track B never waits on track A
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureSpec:
    """Shape of one synthetic dataset.

    Attributes:
        dataset: Dataset key.
        n_graphs: Graph count; the pair count follows.
        censored_frac: Fraction of pairs the exact solver failed to close.
        zero_frac: Fraction of certified pairs with exact GED 0.
    """

    dataset: str
    n_graphs: int
    censored_frac: float
    zero_frac: float


#: Census fractions of design §1, at a size a unit test can run.
FIXTURE_SPECS: tuple[FixtureSpec, ...] = (
    FixtureSpec("linux", 40, 0.0117, 0.0),
    FixtureSpec("aids", 46, 0.2067, 0.0),
    FixtureSpec("iam_letter_low", 44, 0.0, 0.1552),
    FixtureSpec("iam_letter_med", 42, 0.0, 0.1404),
    FixtureSpec("iam_letter_high", 48, 0.0, 0.0419),
)

#: The real graph counts, for the bootstrap timing projection only.
FULL_CENSUS_SPECS: tuple[FixtureSpec, ...] = (
    FixtureSpec("linux", 89, 0.0117, 0.0),
    FixtureSpec("aids", 769, 0.2067, 0.0),
    FixtureSpec("iam_letter_low", 1180, 0.0, 0.1552),
    FixtureSpec("iam_letter_med", 1253, 0.0, 0.1404),
    FixtureSpec("iam_letter_high", 2059, 0.0, 0.0419),
)

#: Looseness of each synthetic bound. The lower-bound factors multiply the
#: exact value; ``HED`` is deliberately the loosest, matching the published
#: dominance ``BED >= HED`` that track A measured at 0.44 against BRANCH's
#: 4.09 on LINUX. The upper-bound entries are Poisson means for the slack.
LB_TIGHTNESS: dict[str, float] = {
    "BRANCH_TIGHT": 0.92,
    "BRANCH": 0.85,
    "BRANCH_FAST": 0.80,
    "STAR": 0.55,
    "HED": 0.30,
}
UB_SLACK: dict[str, float] = {
    "IPFP": 0.35,
    "REFINE": 0.50,
    "BP_BEAM": 0.90,
    "BIPARTITE": 1.60,
}

#: Fraction of lower-bound entries forced to a valid zero. Under cost model
#: D6 both substitutions are free, so any degree-preserving assignment costs
#: nothing: C6 against two disjoint triangles has exact GED 4 and every
#: BRANCH-family bound correctly returns 0. A fixture that never produced
#: this case would not exercise the real distribution.
ZERO_LB_FRACTION = 0.12


def _fixture_meta(spec: FixtureSpec, n_pairs: int, **extra: Any) -> str:
    """Return the CONTRACTS §4 ``meta`` JSON for a synthetic file."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "wave": WAVE,
        "dataset": spec.dataset,
        "n_graphs": spec.n_graphs,
        "n_pairs": n_pairs,
        "method": None,
        "end": None,
        "options": None,
        "deterministic": None,
        "cost_model": [1, 1, 0, 1, 1, 0],
        "gedlib_commit": "synthetic",
        "code_commit": "synthetic",
        "host": platform.node(),
        "wall_seconds": 0.0,
        "created_utc": datetime.now(UTC).isoformat(),
        "synthetic": True,
    }
    payload.update(extra)
    return json.dumps(payload, sort_keys=True)


def build_synthetic_fixture(
    root: Path,
    specs: Iterable[FixtureSpec] = FIXTURE_SPECS,
    *,
    seed: int = 7,
    write_timing: bool = True,
) -> list[str]:
    """Write a contract-shaped synthetic fixture under *root*.

    Every invariant the loaders assert is honoured: canonical pair order,
    ``exact_lb == exact_ub == exact`` on certified pairs, ``inf`` exact on
    censored ones, and ``value == min(value_fwd, value_rev)`` on upper
    bounds. Every synthetic bound is valid, so M4 must be 0 everywhere;
    a fixture that produced a violation would mask the real check.

    Args:
        root: Report root; ``data/index``, ``data/cells`` and
            ``data/timing`` are created beneath it.
        specs: Dataset shapes to generate.
        seed: Generator seed.
        write_timing: Whether to emit the M7 timing and probe JSON.

    Returns:
        The dataset keys written.
    """
    index_dir = root / "data" / "index"
    cells_dir = root / "data" / "cells"
    timing_dir = root / "data" / "timing"
    for directory in (index_dir, cells_dir, timing_dir):
        directory.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for spec in specs:
        rng = np.random.default_rng([seed, abs(hash(spec.dataset)) % (2**31)])
        n = spec.n_graphs
        pair_i, pair_j = np.triu_indices(n, k=1)
        n_pairs = int(pair_i.size)
        node_counts = rng.integers(2, 13, size=n).astype(np.int32)
        edge_counts = (node_counts + rng.integers(0, 4, size=n)).astype(np.int32)
        n_max = np.maximum(node_counts[pair_i], node_counts[pair_j]).astype(np.int32)

        # Hidden truth: always finite, used to keep censored bounds valid.
        size_gap = np.abs(node_counts[pair_i].astype(np.int64) - node_counts[pair_j])
        truth = (size_gap + 1 + rng.poisson(2.0, size=n_pairs)).astype(np.float64)
        zero = rng.random(n_pairs) < spec.zero_frac
        truth[zero] = 0.0

        censored = rng.random(n_pairs) < spec.censored_frac
        censored &= ~zero
        certified = ~censored
        exact = np.where(certified, truth, np.inf)
        exact_lb = np.where(certified, truth, np.floor(truth * 0.6))
        exact_ub = np.where(certified, truth, np.ceil(truth * 1.4) + 1.0)

        lev = {
            "exhaustive": np.rint(truth * 1.4 + rng.normal(0.0, 1.0, n_pairs)).clip(min=0),
            "greedy": np.rint(truth * 1.5 + rng.normal(0.0, 1.4, n_pairs)).clip(min=0),
            "greedy_single": np.rint(truth * 1.6 + rng.normal(0.0, 1.8, n_pairs)).clip(min=0),
        }

        np.savez_compressed(
            index_dir / f"{spec.dataset}.npz",
            pair_i=pair_i.astype(np.int32),
            pair_j=pair_j.astype(np.int32),
            exact=exact,
            exact_lb=exact_lb,
            exact_ub=exact_ub,
            certified=certified,
            n_max=n_max,
            lev_exhaustive=lev["exhaustive"].astype(np.int32),
            lev_greedy=lev["greedy"].astype(np.int32),
            lev_greedy_single=lev["greedy_single"].astype(np.int32),
            graph_ids=np.array([f"{spec.dataset}_{k:05d}" for k in range(n)]),
            node_counts=node_counts,
            edge_counts=edge_counts,
            meta=np.array(_fixture_meta(spec, n_pairs)),
        )

        for method, factor in LB_TIGHTNESS.items():
            value = np.floor(truth * factor)
            value[rng.random(n_pairs) < ZERO_LB_FRACTION] = 0.0
            value = np.minimum(value, truth)  # valid by construction
            np.savez_compressed(
                cells_dir / f"{spec.dataset}__{method}.npz",
                value=value,
                value_fwd=value,
                meta=np.array(
                    _fixture_meta(
                        spec, n_pairs, method=method, end="lower", options="", deterministic=True
                    )
                ),
            )
        for method, slack in UB_SLACK.items():
            fwd = truth + rng.poisson(slack, size=n_pairs)
            rev = truth + rng.poisson(slack, size=n_pairs)
            value = np.minimum(fwd, rev)
            np.savez_compressed(
                cells_dir / f"{spec.dataset}__{method}.npz",
                value=value,
                value_fwd=fwd,
                value_rev=rev,
                meta=np.array(
                    _fixture_meta(
                        spec, n_pairs, method=method, end="upper", options="", deterministic=False
                    )
                ),
            )

        if write_timing:
            for method in (*LB_TIGHTNESS, *UB_SLACK):
                rate = 20.0 + 40.0 * (
                    list(LB_TIGHTNESS).index(method) if method in LB_TIGHTNESS else 4
                )
                (timing_dir / f"{spec.dataset}__{method}.json").write_text(
                    json.dumps(
                        {
                            "dataset": spec.dataset,
                            "method": method,
                            "options": "",
                            "n_pairs_timed": min(2000, n_pairs),
                            "seed": BOOTSTRAP_SEED,
                            "n_bar": float(node_counts.mean()),
                            "us_per_pair_mean": rate,
                            "us_per_pair_median": rate * 0.95,
                            "us_per_pair_p95": rate * 1.9,
                            "clock": "time.process_time",
                            "parallel": False,
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                (timing_dir / f"probe_n30__{method}.json").write_text(
                    json.dumps(
                        {
                            "dataset": "probe_n30",
                            "method": method,
                            "options": "",
                            "n_pairs_timed": 2000,
                            "seed": BOOTSTRAP_SEED,
                            "n_bar": 30.0,
                            "us_per_pair_mean": rate * 6.0,
                            "us_per_pair_median": rate * 5.7,
                            "us_per_pair_p95": rate * 11.0,
                            "clock": "time.process_time",
                            "parallel": False,
                            "source": "iam_gxl:GREC+Protein",
                            "n_range": [25, 35],
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
        written.append(spec.dataset)
    return written


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _code_commit() -> str:
    """Return the repository HEAD, or ``"unknown"`` outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git
        return "unknown"


def _provenance(**extra: Any) -> dict[str, Any]:
    """Return the provenance header stamped into every analysis JSON."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "wave": WAVE,
        "produced_by": "benchmarks.real_data.eval_setup.ged_bakeoff_analysis",
        "code_commit": _code_commit(),
        "host": platform.node(),
        "created_utc": datetime.now(UTC).isoformat(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "statistical_status": SELECTION_STATUS,
    }
    payload.update(extra)
    return payload


def _jsonable(obj: Any) -> Any:  # noqa: ANN401 -- recursive JSON coercion
    """Coerce numpy scalars and arrays into JSON-serialisable Python."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write *payload* as indented, key-sorted JSON.

    Non-finite floats become ``null`` rather than the ``NaN`` literal,
    which is not valid JSON and which several readers reject silently.

    Args:
        path: Destination path.
        payload: The mapping to write.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass
class DatasetBundle:
    """One dataset's loaded index and cells."""

    index: IndexData
    cells: dict[str, CellData] = field(default_factory=dict)


def load_bundles(root: Path, datasets: Sequence[str]) -> dict[str, DatasetBundle]:
    """Load every dataset index with its cells.

    Args:
        root: Report root.
        datasets: Dataset keys to load; missing indices are skipped.

    Returns:
        Bundles keyed by dataset.

    Raises:
        BakeoffAnalysisError: If no dataset could be loaded.
    """
    bundles: dict[str, DatasetBundle] = {}
    for dataset in datasets:
        index_path = root / "data" / "index" / f"{dataset}.npz"
        if not index_path.is_file():
            LOGGER.warning("no index for %s at %s; skipping", dataset, index_path)
            continue
        index = load_index(index_path)
        bundle = DatasetBundle(index=index)
        for cell_path in discover_cells(root, dataset):
            cell = load_cell(cell_path, index)
            bundle.cells[cell.method] = cell
        if not bundle.cells:
            LOGGER.warning("no cells for %s; skipping", dataset)
            continue
        bundles[dataset] = bundle
    if not bundles:
        raise BakeoffAnalysisError(f"no loadable dataset under {root}")
    return bundles


def _candidates(
    end: str,
    bundle: DatasetBundle,
    metrics: Mapping[str, Mapping[str, Any]],
) -> list[Candidate]:
    """Assemble the selection candidates for one (dataset, end)."""
    out: list[Candidate] = []
    for method in methods_for_end(end):
        payload = metrics.get(method)
        if payload is None:
            continue
        timing = payload["M7_cost"]
        per_dataset = timing.get("dataset_timing") or {}
        out.append(
            Candidate(
                method=method,
                mean_relative_error=float(payload["M1_relative_error"]["exact_gt_zero"]["mean"]),
                mean_absolute_error=float(payload["M2_absolute_error"]["exact_gt_zero"]["mean"]),
                violations=int(payload["M4_validity"]["violations"]),
                m3_well_defined=bool(payload["M3_certification_rate"]["well_defined"]),
                cost_gate=str(timing["gate"]),
                cost_us_per_pair=float(per_dataset.get("us_per_pair_mean", float("nan"))),
                m6_abs_gap=float(payload["M6_rho_lev"]["abs_gap"]),
            )
        )
    return out


def run_analysis(
    root: Path,
    *,
    datasets: Sequence[str] = DATASETS,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    jobs: int = 1,
    lev_variant: str = PRIMARY_LEV,
    make_figures: bool = True,
) -> dict[str, Path]:
    """Run the whole T-27 analysis and write CONTRACTS §7 and §8.

    Args:
        root: Report root holding ``data/index`` and ``data/cells``.
        datasets: Dataset keys to analyse.
        replicates: Bootstrap replicates; 2,000 in production.
        seed: Bootstrap master seed; 42 in production.
        jobs: Worker processes, capped at :data:`MAX_PROCESSES`.
        lev_variant: Encoder variant behind M6.
        make_figures: Whether to render the figures.

    Returns:
        Mapping from artefact name to written path.
    """
    bundles = load_bundles(root, datasets)
    analysis_dir = root / "data" / "analysis"
    written: dict[str, Path] = {}

    metrics: dict[str, dict[str, Any]] = {}
    curves: dict[str, dict[str, Any]] = {}
    validity: dict[str, Any] = {}
    for dataset, bundle in bundles.items():
        metrics[dataset] = {}
        curves[dataset] = {}
        for method, cell in sorted(bundle.cells.items()):
            payload = compute_cell_metrics(
                cell, bundle.index, root, lev_variant=lev_variant
            ).payload
            metrics[dataset][method] = payload
            curves[dataset][method] = error_vs_n(cell, bundle.index)
            validity[f"{dataset}__{method}"] = payload["M4_validity"]

    total_violations = sum(int(v["violations"]) for v in validity.values())
    written["validity"] = write_json(
        analysis_dir / "validity.json",
        {
            "provenance": _provenance(analysis="M4 validity"),
            "domain": (
                "All pairs: two-sided on certified pairs, one-sided on censored ones "
                "(design §3.5). A lower bound is refuted iff LB > exact_ub, an upper "
                "bound iff UB < exact_lb. LB == 0 with exact > 0 is legal under cost "
                "model D6 and is never flagged."
            ),
            "total_violations": total_violations,
            "halts_ticket": total_violations > 0,
            "cells": validity,
        },
    )

    written["metrics"] = write_json(
        analysis_dir / "metrics.json",
        {
            "provenance": _provenance(analysis="M1-M8", lev_variant=lev_variant),
            "domain_note": (
                "M1, M2 and M3 are each reported over all certified pairs and over "
                "exact > 0 only; the second is the headline (design §3.3). "
                "M1/M2/M3/M5/M6 use certified pairs only (§3.5)."
            ),
            "cells": metrics,
            "error_vs_n": curves,
        },
    )

    bootstrap: dict[str, Any] = {}
    for dataset, bundle in bundles.items():
        LOGGER.info("bootstrap: %s (%d graphs)", dataset, bundle.index.n_graphs)
        bootstrap[dataset] = bootstrap_dataset(
            bundle.index,
            list(bundle.cells.values()),
            replicates=replicates,
            seed=seed,
            jobs=jobs,
            lev_variant=lev_variant,
        )
    written["bootstrap"] = write_json(
        analysis_dir / "bootstrap.json",
        {
            "provenance": _provenance(analysis="M5/M6 bootstrap", replicates=replicates),
            "protocol": BOOTSTRAP_STATUS,
            "datasets": bootstrap,
        },
    )

    significance: dict[str, Any] = {"per_dataset": {}, "friedman": {}}
    for dataset, bundle in bundles.items():
        significance["per_dataset"][dataset] = {
            end: pairwise_significance(end, dataset, bundle.cells, bundle.index, bootstrap[dataset])
            for end in ENDS
        }
    for end in ENDS:
        scores = {
            dataset: {
                method: float(payload["M1_relative_error"]["exact_gt_zero"]["mean"])
                for method, payload in metrics[dataset].items()
                if payload["end"] == end
            }
            for dataset in bundles
        }
        significance["friedman"][end] = friedman_over_datasets(end, scores)
    written["significance"] = write_json(
        analysis_dir / "significance.json",
        {
            "provenance": _provenance(analysis="Wilcoxon/Holm + Friedman"),
            "p_value_status": PVALUE_STATUS,
            **significance,
        },
    )

    selection = {
        end: select_end(end, {d: _candidates(end, b, metrics[d]) for d, b in bundles.items()})
        for end in ENDS
    }
    written["selection"] = write_json(
        analysis_dir / "selection.json",
        {
            "provenance": _provenance(analysis="frozen selection rule"),
            "tie_tolerance": TIE_TOLERANCE,
            "ends": selection,
        },
    )

    if make_figures:
        written.update(
            render_figures(
                root,
                metrics=metrics,
                curves=curves,
                bootstrap=bootstrap,
                selection=selection,
                friedman=significance["friedman"],
            )
        )
    return written


# ---------------------------------------------------------------------------
# Figures -- CONTRACTS §8, rendered through isalgraph.viz
# ---------------------------------------------------------------------------


def build_panels(
    end: str,
    metrics: Mapping[str, Mapping[str, Any]],
    curves: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> Any:  # noqa: ANN401 -- returns viz.BakeoffPanels, imported lazily
    """Assemble the panel data for one bracket end.

    Args:
        end: ``"lower"`` or ``"upper"``.
        metrics: Per-dataset, per-method metric payloads.
        curves: Per-dataset, per-method ``error_vs_n`` bins.
        bootstrap: Per-dataset bootstrap results.
        selection: The selection outcome, for the winner marker.

    Returns:
        A :class:`isalgraph.viz.bound_bakeoff_view.BakeoffPanels`.
    """
    from isalgraph.viz.bound_bakeoff_view import (
        BakeoffPanels,
        DatasetCurves,
        ErrorCurve,
        ForestEntry,
    )

    datasets = [d for d in DATASETS if d in metrics] + [
        d for d in sorted(metrics) if d not in DATASETS
    ]
    methods = tuple(m for m in methods_for_end(end) if any(m in metrics[d] for d in datasets))
    winners = selection.get(end, {}).get("frozen_rule", {}).get("per_dataset", {})

    dataset_curves: list[Any] = []
    forest: list[Any] = []
    for dataset in datasets:
        bins = curves.get(dataset, {})
        dataset_curves.append(
            DatasetCurves(
                dataset=dataset,
                curves=tuple(
                    ErrorCurve(
                        method=method,
                        n_values=tuple(bins[method]["n_values"]),
                        mean=tuple(bins[method]["mean"]),
                        q25=tuple(bins[method]["q25"]),
                        q75=tuple(bins[method]["q75"]),
                        counts=tuple(bins[method]["counts"]),
                    )
                    for method in methods
                    if method in bins
                ),
            )
        )
        winner = winners.get(dataset, {}).get("winner")
        stats_block = bootstrap.get(dataset, {}).get("statistics", {})
        for method in methods:
            payload = metrics[dataset].get(method)
            if payload is None:
                continue
            entry = stats_block.get(f"mean_rel_err::{method}", {})
            point = float(payload["M1_relative_error"]["exact_gt_zero"]["mean"])
            forest.append(
                ForestEntry(
                    dataset=dataset,
                    method=method,
                    mean=point,
                    ci_low=float(entry.get("ci_low", point)),
                    ci_high=float(entry.get("ci_high", point)),
                    winner=(method == winner),
                )
            )
    return BakeoffPanels(
        end=end,
        dataset_curves=tuple(dataset_curves),
        forest=tuple(forest),
        methods=methods,
    )


def build_critical_difference(end: str, friedman: Mapping[str, Any]) -> Any:  # noqa: ANN401
    """Assemble the CD-diagram payload for one end.

    Args:
        end: ``"lower"`` or ``"upper"``.
        friedman: The end's :func:`friedman_over_datasets` result.

    Returns:
        A :class:`isalgraph.viz.bound_bakeoff_view.CriticalDifference`,
        or ``None`` when the omnibus was not evaluable.
    """
    from isalgraph.viz.bound_bakeoff_view import CriticalDifference

    if "average_ranks" not in friedman:
        return None
    methods = tuple(friedman["methods"])
    ranks = tuple(float(friedman["average_ranks"][m]) for m in methods)
    cliques = tuple(
        tuple(methods.index(m) for m in clique) for clique in friedman.get("cliques", [])
    )
    return CriticalDifference(
        end=end,
        methods=methods,
        average_ranks=ranks,
        cd=float(friedman["critical_difference"]),
        n_datasets=len(friedman["datasets"]),
        friedman_p=float(friedman["friedman_p"]),
        cliques=cliques,
    )


def render_figures(
    root: Path,
    *,
    metrics: Mapping[str, Mapping[str, Any]],
    curves: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    selection: Mapping[str, Any],
    friedman: Mapping[str, Any],
) -> dict[str, Path]:
    """Render the CONTRACTS §8 figures plus the design §3.8 CD diagrams.

    The CD diagram is not in CONTRACTS §8, which names two figures per
    end; it is added because design §3.8 requires it beside the
    Wilcoxon/Holm table, and it is written to separate files so the two
    contracted figures keep exactly the panels they were promised.

    Args:
        root: Report root; figures go to ``figures/``.
        metrics: Per-dataset metric payloads.
        curves: Per-dataset ``error_vs_n`` bins.
        bootstrap: Per-dataset bootstrap results.
        selection: The selection outcome.
        friedman: Per-end Friedman results.

    Returns:
        Mapping from artefact name to the PDF path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from isalgraph.viz.bound_bakeoff_view import (
        bound_bakeoff_figure,
        critical_difference_figure,
    )
    from isalgraph.viz.style import save_figure

    out: dict[str, Path] = {}
    figures_dir = root / "figures"
    for end in ENDS:
        panels = build_panels(end, metrics, curves, bootstrap, selection)
        fig = bound_bakeoff_figure(panels)
        paths = save_figure(fig, figures_dir / f"T27_{end}_bound")
        plt.close(fig)
        out[f"figure_{end}"] = paths[0]

        cd = build_critical_difference(end, friedman.get(end, {}))
        if cd is not None:
            fig_cd = critical_difference_figure(cd)
            cd_paths = save_figure(fig_cd, figures_dir / f"T27_{end}_cd")
            plt.close(fig_cd)
            out[f"figure_{end}_cd"] = cd_paths[0]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="ged_bakeoff_analysis",
        description=(
            "Aggregate the T-27 GED bound bake-off: M1-M8, graph-level bootstrap, "
            "Wilcoxon/Holm, Friedman and the frozen selection rule."
        ),
    )
    parser.add_argument("--out", type=Path, required=True, help="report root")
    parser.add_argument(
        "--datasets", nargs="*", default=list(DATASETS), help="dataset keys to analyse"
    )
    parser.add_argument(
        "--replicates", type=int, default=BOOTSTRAP_REPLICATES, help="bootstrap replicates"
    )
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED, help="bootstrap master seed")
    parser.add_argument(
        "--jobs", type=int, default=1, help=f"worker processes (capped at {MAX_PROCESSES})"
    )
    parser.add_argument(
        "--lev-variant", default=PRIMARY_LEV, choices=list(LEV_VARIANTS), help="encoder variant"
    )
    parser.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    parser.add_argument(
        "--make-fixture",
        action="store_true",
        help="write a synthetic fixture under --out before analysing",
    )
    parser.add_argument(
        "--fixture-full-census",
        action="store_true",
        help="use the real graph counts for the fixture (timing projection only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.make_fixture:
        specs = FULL_CENSUS_SPECS if args.fixture_full_census else FIXTURE_SPECS
        specs = tuple(s for s in specs if s.dataset in set(args.datasets))
        LOGGER.info("writing synthetic fixture for %s", [s.dataset for s in specs])
        build_synthetic_fixture(args.out, specs)
    written = run_analysis(
        args.out,
        datasets=args.datasets,
        replicates=args.replicates,
        seed=args.seed,
        jobs=args.jobs,
        lev_variant=args.lev_variant,
        make_figures=not args.no_figures,
    )
    for name, path in sorted(written.items()):
        LOGGER.info("%s -> %s", name, path)
    return 0


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "DATASETS",
    "FIXTURE_SPECS",
    "FULL_CENSUS_SPECS",
    "LOWER_METHODS",
    "MAX_PROCESSES",
    "TIE_TOLERANCE",
    "UPPER_METHODS",
    "BakeoffAnalysisError",
    "Candidate",
    "CellData",
    "CellMetrics",
    "DatasetBundle",
    "ErrorStats",
    "FixtureSpec",
    "IndexData",
    "ValidityResult",
    "WilcoxonResult",
    "bootstrap_dataset",
    "build_critical_difference",
    "build_panels",
    "build_synthetic_fixture",
    "collapse_to_corpora",
    "compute_cell_metrics",
    "compute_symmetry",
    "compute_validity",
    "discover_cells",
    "end_of_method",
    "error_stats",
    "error_vs_n",
    "friedman_over_datasets",
    "induced_pairs",
    "load_bundles",
    "load_cell",
    "load_index",
    "main",
    "methods_for_end",
    "midranks",
    "nemenyi_critical_difference",
    "pair_flat_index",
    "pairwise_significance",
    "rank_cliques",
    "relative_errors",
    "render_figures",
    "replicate_selection",
    "run_analysis",
    "select_end",
    "select_on_dataset",
    "signed_error",
    "spearman",
    "spearman_from_ranks",
    "wilcoxon_signed_rank",
    "write_json",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
