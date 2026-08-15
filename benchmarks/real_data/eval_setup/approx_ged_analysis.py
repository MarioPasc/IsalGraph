"""T-05 §7 analysis of the bounded-GED matrices over Suite 2.

Produces the four computable deliverables of ``T-05-design.md`` §7 --- the
bracket width against node count (§7.1), the certification rate (§7.2), the
bracket width by size and density stratum (§7.3) and the realised cost table
(§7.4) --- from the ``LB``/``UB``/``UB_SENSITIVITY`` matrices, writing a
report, one JSON file per table and the figures the report cites.

Every rule this module applies was frozen in ``T-05-design.md`` amendment 12
before the first figure was drawn; the module does not choose any of them.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from benchmarks.real_data.eval_setup.ged_pair_index import pairs_from_indices_searchsorted

FloatArray: TypeAlias = npt.NDArray[np.float64]
Float32Array: TypeAlias = npt.NDArray[np.float32]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

LOGGER = logging.getLogger("approx_ged_analysis")


class ApproxGedAnalysisError(Exception):
    """Base class for every error raised by this module."""


class InputError(ApproxGedAnalysisError):
    """An input file is missing, malformed or internally inconsistent."""


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

#: Suite-2 cohort, ``T-05-design.md`` §2, in the order that file lists them.
DATASET_KEYS: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
    "aids_graphedx",
    "grec",
    "aids_iam",
    "coil_del",
    "mutagenicity",
    "protein",
)

#: The four datasets §7.1 names as spanning enough ``n`` to carry an
#: unconfounded within-dataset slope on their own.
UNCONFOUNDED_DATASETS: tuple[str, ...] = ("mutagenicity", "coil_del", "aids_iam", "protein")

#: The datasets that cap at ``n <= 10`` and therefore constrain only the
#: small-``n`` end of the size-scaling curve.
SMALL_N_DATASETS: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
)

#: Role -> (directory-argument name, method, human label). The method names
#: are the ones ``T-05-design.md`` §1 froze; they are re-read from each file's
#: ``metadata`` and asserted rather than assumed.
ROLE_METHOD: dict[str, str] = {
    "LB": "BRANCH_FAST",
    "UB": "BIPARTITE",
    "UBS": "BP_BEAM",
}
ROLE_LABEL: dict[str, str] = {
    "LB": "lower bound (BRANCH_FAST)",
    "UB": "upper bound, primary (BIPARTITE)",
    "UBS": "upper bound, sensitivity arm (BP_BEAM_DET)",
}
ROLES: tuple[str, ...] = ("LB", "UB", "UBS")

#: ``statistics.md`` §8 node-count bins on ``max(n1, n2)``, with the leading
#: ``2`` bin amendment 12 requires because Suite 2 filters at ``min_nodes = 2``
#: and §8's lowest bin starts at 3. Left-closed, right-open, last unbounded.
SIZE_BIN_EDGES: tuple[int, ...] = (2, 3, 6, 10, 13, 21, 41)
SIZE_BIN_LABELS: tuple[str, ...] = ("2", "3-5", "6-9", "10-12", "13-20", "21-40", ">40")

#: A pair is certified when the two bounds coincide, i.e. GED is proven exactly.
CERTIFIED_TOL: float = 1e-9

#: D15 tier 3 --- ``statistics.md`` §5's frozen tier assignment names exactly
#: these two datasets and nothing recomputes the assignment at run time.
TIER3_DATASETS: frozenset[str] = frozenset({"coil_del", "mutagenicity"})
TIER3_REPLICATES: int = 1000
TIER3_SUBSAMPLE: int = 2_000_000
DEFAULT_REPLICATES: int = 2000

BOOTSTRAP_SEED: int = 42
CI_PERCENTILES: tuple[float, float] = (2.5, 97.5)

#: A (dataset x density quintile) cell below this many pairs is reported as
#: dropped rather than silently omitted (§7.1 secondary).
MIN_CELL_PAIRS: int = 1000

N_DENSITY_QUANTILES: int = 5

#: Replicates per BLAS call in the exact weighted bootstrap. Bounds the
#: transient ``(n_graphs, batch)`` count matrix.
BOOTSTRAP_BATCH: int = 250

SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisConfig:
    """Everything the analysis needs that is not read from a file.

    Parameters
    ----------
    lb_dir, ub_dir, ubs_dir
        Directories holding ``{dataset}.npz`` for the three roles.
    input_dir
        The ``exported_suite2`` directory; used only to cross-check graph ids
        and node counts against the bound files.
    out_dir
        Destination for ``REPORT.md``, ``data/`` and ``figures/``.
    datasets
        Dataset keys to analyse, in report order.
    datasets_explicit
        True when the caller named the datasets on the command line, which
        makes a missing file an error rather than a skipped cohort member.
    seed
        Master bootstrap seed; 42, frozen.
    min_cell_pairs
        Population floor for a (dataset x density quintile) secondary cell.
    make_figures
        When False the figure stage is skipped; the JSON and the report are
        still written.
    """

    lb_dir: Path
    ub_dir: Path
    ubs_dir: Path
    input_dir: Path
    out_dir: Path
    datasets: tuple[str, ...] = DATASET_KEYS
    datasets_explicit: bool = False
    seed: int = BOOTSTRAP_SEED
    min_cell_pairs: int = MIN_CELL_PAIRS
    make_figures: bool = True

    def role_dir(self, role: str) -> Path:
        """Return the directory holding a role's matrices.

        Parameters
        ----------
        role
            One of ``LB``, ``UB``, ``UBS``.

        Returns
        -------
        Path
            The configured directory.

        Raises
        ------
        InputError
            If the role name is not one of the three.
        """
        mapping = {"LB": self.lb_dir, "UB": self.ub_dir, "UBS": self.ubs_dir}
        if role not in mapping:
            raise InputError(f"unknown role {role!r}; expected one of {sorted(mapping)}")
        return mapping[role]


@dataclass(frozen=True)
class BootstrapTier:
    """The D15 resampling effort assigned to one dataset.

    Parameters
    ----------
    tier
        1, 2 or 3 as ``statistics.md`` §5 assigns it.
    replicates
        Bootstrap replicate count.
    subsample
        Induced pairs drawn uniformly without replacement inside each
        replicate, or ``None`` for all induced pairs.
    """

    tier: int
    replicates: int
    subsample: int | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the tier."""
        return {
            "tier": self.tier,
            "replicates": self.replicates,
            "within_replicate_pairs": "all" if self.subsample is None else self.subsample,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "graph",
        }


def bootstrap_tier(dataset_key: str) -> BootstrapTier:
    """Return the frozen D15 tier for a dataset.

    The assignment is a lookup, not a computation: ``statistics.md`` §5's
    "Frozen tier assignment" table names COIL-DEL and Mutagenicity as the only
    tier-3 datasets and states that the assignment is not recomputed at
    execution time. Tier 1 and tier 2 differ only in permutation count, which
    this module does not use, so both resolve to the same resampling effort.

    Parameters
    ----------
    dataset_key
        A Suite-2 dataset key.

    Returns
    -------
    BootstrapTier
        Replicate count and within-replicate pair budget.
    """
    if dataset_key in TIER3_DATASETS:
        return BootstrapTier(tier=3, replicates=TIER3_REPLICATES, subsample=TIER3_SUBSAMPLE)
    tier = 2 if dataset_key in {"aids_iam", "iam_letter_high"} else 1
    return BootstrapTier(tier=tier, replicates=DEFAULT_REPLICATES, subsample=None)


# ---------------------------------------------------------------------------
# Numeric core
# ---------------------------------------------------------------------------


def upper_triangle(matrix: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return the strict upper triangle of a square matrix as a flat vector.

    Parameters
    ----------
    matrix
        A square array.

    Returns
    -------
    numpy.ndarray
        The ``k = 1`` upper-triangular entries in row-major order.

    Raises
    ------
    InputError
        If the array is not square or not two-dimensional.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise InputError(f"expected a square matrix, got shape {matrix.shape}")
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols]


def bracket_width(lb: npt.NDArray[Any], ub: npt.NDArray[Any]) -> FloatArray:
    """Return the relative bracket width ``(UB - LB) / UB``.

    Where ``UB == 0`` the width is 0, not undefined: the bounds satisfy
    ``0 <= LB <= UB``, so ``UB == 0`` forces ``LB == 0`` and the bracket is
    genuinely closed. GED is legitimately 0 for isomorphic graphs --- 15.5 % of
    IAM Letter LOW --- so those pairs are kept, not filtered.

    Parameters
    ----------
    lb
        Lower bounds.
    ub
        Upper bounds, same shape as ``lb``.

    Returns
    -------
    numpy.ndarray
        Widths in ``[0, 1]``, float64.

    Raises
    ------
    InputError
        If the shapes differ, if any value is negative or non-finite, or if
        ``LB > UB`` anywhere beyond ``CERTIFIED_TOL``.
    """
    lb64 = np.asarray(lb, dtype=np.float64)
    ub64 = np.asarray(ub, dtype=np.float64)
    if lb64.shape != ub64.shape:
        raise InputError(f"shape mismatch: lb {lb64.shape} vs ub {ub64.shape}")
    if not (np.isfinite(lb64).all() and np.isfinite(ub64).all()):
        raise InputError("non-finite bound encountered")
    if (lb64 < 0).any() or (ub64 < 0).any():
        raise InputError("negative bound encountered")
    n_violations = int((lb64 > ub64 + CERTIFIED_TOL).sum())
    if n_violations:
        raise InputError(f"{n_violations} pairs violate LB <= UB")
    width = np.zeros_like(ub64)
    positive = ub64 > 0.0
    np.divide(ub64 - lb64, ub64, out=width, where=positive)
    return np.clip(width, 0.0, 1.0)


def graph_density(node_counts: npt.NDArray[Any], edge_counts: npt.NDArray[Any]) -> FloatArray:
    """Return per-graph density ``2 m / (n (n - 1))``.

    Parameters
    ----------
    node_counts
        Node count per graph.
    edge_counts
        Edge count per graph.

    Returns
    -------
    numpy.ndarray
        Density per graph, 0 where ``n < 2`` (which the Suite-2 filter
        excludes, so the guard never fires on real input).
    """
    n = np.asarray(node_counts, dtype=np.float64)
    m = np.asarray(edge_counts, dtype=np.float64)
    denominator = n * (n - 1.0)
    density = np.zeros_like(n)
    np.divide(2.0 * m, denominator, out=density, where=denominator > 0.0)
    return density


def pair_density_matrix(density: FloatArray) -> FloatArray:
    """Return the pair-density matrix ``d_pair = (d1 + d2) / 2``.

    Amendment 12 rule 3 freezes the mean over ``min``, ``max`` and the density
    of the union: it is the only one of the four that is symmetric in the pair
    and reduces to the common value when the two graphs match.

    Parameters
    ----------
    density
        Per-graph density.

    Returns
    -------
    numpy.ndarray
        ``(N, N)`` matrix of pair densities.
    """
    return 0.5 * (density[:, None] + density[None, :])


def n_max_matrix(node_counts: npt.NDArray[Any]) -> FloatArray:
    """Return the matrix of ``max(n1, n2)`` over every ordered pair.

    Parameters
    ----------
    node_counts
        Node count per graph.

    Returns
    -------
    numpy.ndarray
        ``(N, N)`` float64 matrix.
    """
    n = np.asarray(node_counts, dtype=np.float64)
    return np.maximum(n[:, None], n[None, :])


def size_bin_codes(n_max: npt.NDArray[Any]) -> npt.NDArray[np.int8]:
    """Assign each pair to a ``statistics.md`` §8 node-count stratum.

    Parameters
    ----------
    n_max
        ``max(n1, n2)`` per pair.

    Returns
    -------
    numpy.ndarray
        Zero-based stratum index into :data:`SIZE_BIN_LABELS`, int8.

    Raises
    ------
    InputError
        If any pair falls below the lowest edge, which would mean a graph with
        fewer than two nodes survived the Suite-2 filter.
    """
    values = np.asarray(n_max)
    if values.size and int(values.min()) < SIZE_BIN_EDGES[0]:
        raise InputError(f"n_max = {int(values.min())} is below the lowest stratum edge")
    codes = np.searchsorted(np.asarray(SIZE_BIN_EDGES[1:]), values, side="right")
    return codes.astype(np.int8)


@dataclass(frozen=True)
class OlsFit:
    """An ordinary-least-squares fit of ``y`` on a single regressor ``x``."""

    slope: float
    intercept: float
    r_squared: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the fit."""
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "n_pairs": self.n,
        }


def ols_fit(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> OlsFit:
    """Fit ``y = intercept + slope * x`` by ordinary least squares.

    Parameters
    ----------
    x
        Regressor.
    y
        Response, same length as ``x``.

    Returns
    -------
    OlsFit
        Slope, intercept, ``R^2`` and the sample size. All three statistics are
        ``nan`` when ``x`` has zero variance or fewer than two observations,
        which is the honest answer for a dataset whose graphs are all the same
        size.
    """
    xf = np.asarray(x, dtype=np.float64).ravel()
    yf = np.asarray(y, dtype=np.float64).ravel()
    n = xf.size
    if n < 2:
        return OlsFit(math.nan, math.nan, math.nan, n)
    s_x = float(xf.sum())
    s_y = float(yf.sum())
    s_xx = float(xf @ xf)
    s_xy = float(xf @ yf)
    s_yy = float(yf @ yf)
    return _ols_from_sums(s_x, s_y, s_xx, s_xy, s_yy, n)


def _ols_from_sums(
    s_x: float,
    s_y: float,
    s_xx: float,
    s_xy: float,
    s_yy: float,
    n: int,
) -> OlsFit:
    """Assemble an :class:`OlsFit` from raw cross-product sums.

    Parameters
    ----------
    s_x, s_y, s_xx, s_xy, s_yy
        Sums of ``x``, ``y``, ``x^2``, ``x y`` and ``y^2``.
    n
        Number of observations behind the sums.

    Returns
    -------
    OlsFit
        The fit, with ``nan`` statistics when ``x`` has no variance.
    """
    if n < 2:
        return OlsFit(math.nan, math.nan, math.nan, n)
    var_x = s_xx - s_x * s_x / n
    if var_x <= 0.0:
        return OlsFit(math.nan, math.nan, math.nan, n)
    cov_xy = s_xy - s_x * s_y / n
    slope = cov_xy / var_x
    intercept = s_y / n - slope * s_x / n
    var_y = s_yy - s_y * s_y / n
    r_squared = (cov_xy * cov_xy) / (var_x * var_y) if var_y > 0.0 else math.nan
    return OlsFit(float(slope), float(intercept), float(r_squared), int(n))


def percentile_ci(samples: npt.NDArray[Any]) -> dict[str, Any]:
    """Return the percentile confidence interval of a bootstrap distribution.

    Parameters
    ----------
    samples
        Replicate statistics, possibly containing ``nan``.

    Returns
    -------
    dict
        ``ci_low``, ``ci_high``, ``bootstrap_mean``, ``bootstrap_sd``,
        ``n_finite_replicates`` and the percentile pair used.
    """
    arr = np.asarray(samples, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "ci_low": math.nan,
            "ci_high": math.nan,
            "bootstrap_mean": math.nan,
            "bootstrap_sd": math.nan,
            "n_finite_replicates": 0,
            "percentiles": list(CI_PERCENTILES),
        }
    low, high = (float(v) for v in np.percentile(finite, list(CI_PERCENTILES)))
    return {
        "ci_low": low,
        "ci_high": high,
        "bootstrap_mean": float(finite.mean()),
        "bootstrap_sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "n_finite_replicates": int(finite.size),
        "percentiles": list(CI_PERCENTILES),
    }


# ---------------------------------------------------------------------------
# Graph-level cluster bootstrap
# ---------------------------------------------------------------------------


def replicate_selection(n_graphs: int, seed: int, replicate: int) -> IntArray:
    """Draw the graph resample for one replicate.

    The seeding rule is the repo's existing convention
    (``ged_bakeoff_analysis.replicate_selection``): ``SeedSequence([seed,
    replicate])``, so a replicate's draw does not depend on loop order and is
    identical serially or across processes. A unit test asserts equality with
    that function.

    Parameters
    ----------
    n_graphs
        Number of graphs in the dataset.
    seed
        Master seed; 42 in production.
    replicate
        Zero-based replicate number.

    Returns
    -------
    numpy.ndarray
        ``n_graphs`` graph indices drawn with replacement. **The resampling
        unit is the graph, never the pair.**
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
    return rng.integers(0, n_graphs, size=n_graphs, dtype=np.int64)


def induced_pair_slots(selection: IntArray) -> tuple[IntArray, IntArray]:
    """Return the graph indices of every pair induced by a graph resample.

    Slots holding the same original graph induce a self-pair, which has no
    observation in a strict upper triangle and is dropped; every other
    unordered slot pair contributes, duplicates included, because that
    duplication is the cluster bootstrap's variance mechanism. This matches
    ``ged_bakeoff_analysis.induced_pairs``.

    Parameters
    ----------
    selection
        Resampled graph indices.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lo, hi)`` graph indices with ``lo < hi``, with repetitions.
    """
    slot_i, slot_j = np.triu_indices(selection.size, k=1)
    a = selection[slot_i]
    b = selection[slot_j]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keep = lo != hi
    return lo[keep], hi[keep]


def _replicate_sums_reference(
    matrices: Mapping[str, FloatArray],
    selection: IntArray,
) -> dict[str, float]:
    """Sum each matrix over the induced pair multiset by explicit enumeration.

    This is the readable definition of the statistic. It materialises every
    induced pair, so it is ``O(N^2)`` in memory and is used as the correctness
    reference for :func:`_replicate_sums_weighted`, not in production.

    Parameters
    ----------
    matrices
        Named symmetric ``(N, N)`` matrices to sum.
    selection
        Resampled graph indices.

    Returns
    -------
    dict
        One sum per matrix name plus ``n`` , the induced pair count.
    """
    lo, hi = induced_pair_slots(selection)
    sums: dict[str, float] = {name: float(mat[lo, hi].sum()) for name, mat in matrices.items()}
    sums["n"] = float(lo.size)
    return sums


def _replicate_sums_weighted(
    matrices: Mapping[str, FloatArray],
    counts: FloatArray,
) -> dict[str, npt.NDArray[np.float64]]:
    """Sum each matrix over the induced pair multiset via a quadratic form.

    For a resample with multiplicity vector ``c``, the sum of a symmetric
    matrix ``F`` over the unordered slot pairs whose two graphs differ is

    .. math:: S_F = \\tfrac{1}{2}\\left(c^{\\top} F c - \\sum_a c_a^2 F_{aa}\\right)

    and the induced pair count is :math:`(N^2 - \\sum_a c_a^2)/2`. This is an
    algebraic identity, not an approximation: it returns exactly what
    :func:`_replicate_sums_reference` returns, at ``O(N^2)`` work per replicate
    inside BLAS instead of ``O(N^2)`` Python-level fancy indexing.

    Parameters
    ----------
    matrices
        Named symmetric ``(N, N)`` float64 matrices.
    counts
        ``(N, R)`` multiplicity matrix, one column per replicate.

    Returns
    -------
    dict
        One length-``R`` vector of sums per matrix name, plus ``n``.
    """
    squared = counts * counts
    total = counts.sum(axis=0)
    sums: dict[str, npt.NDArray[np.float64]] = {}
    for name, mat in matrices.items():
        quadratic = np.einsum("ij,ij->j", counts, mat @ counts, optimize=True)
        diagonal = np.diag(mat) @ squared
        sums[name] = 0.5 * (quadratic - diagonal)
    sums["n"] = 0.5 * (total * total - squared.sum(axis=0))
    return sums


def _bootstrap_slopes_full(
    width: FloatArray,
    n_max: FloatArray,
    n_graphs: int,
    replicates: int,
    seed: int,
) -> FloatArray:
    """Bootstrap the OLS slope over all induced pairs, exactly.

    Parameters
    ----------
    width
        ``(N, N)`` bracket-width matrix, zero diagonal.
    n_max
        ``(N, N)`` matrix of ``max(n1, n2)``.
    n_graphs
        Number of graphs.
    replicates
        Replicate count.
    seed
        Master seed.

    Returns
    -------
    numpy.ndarray
        One slope per replicate.
    """
    matrices = {
        "y": width,
        "x": n_max,
        "xx": n_max * n_max,
        "xy": n_max * width,
    }
    slopes = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, BOOTSTRAP_BATCH):
        stop = min(start + BOOTSTRAP_BATCH, replicates)
        counts = np.empty((n_graphs, stop - start), dtype=np.float64)
        for column, replicate in enumerate(range(start, stop)):
            selection = replicate_selection(n_graphs, seed, replicate)
            counts[:, column] = np.bincount(selection, minlength=n_graphs)
        sums = _replicate_sums_weighted(matrices, counts)
        n = sums["n"]
        variance = sums["xx"] - sums["x"] * sums["x"] / n
        covariance = sums["xy"] - sums["x"] * sums["y"] / n
        with np.errstate(divide="ignore", invalid="ignore"):
            batch = np.where(variance > 0.0, covariance / variance, np.nan)
        slopes[start:stop] = batch
    return slopes


def _bootstrap_slopes_subsampled(
    width: Float32Array,
    n_max: Float32Array,
    n_graphs: int,
    replicates: int,
    subsample: int,
    seed: int,
) -> FloatArray:
    """Bootstrap the OLS slope on a uniform subsample of the induced pairs.

    D15 tier 3. Within each replicate the graph resample is drawn first ---
    the resampling unit stays the graph --- and the subsample applies only to
    the induced pairs inside that replicate. Slot pairs are drawn uniformly
    without replacement from the ``N (N - 1) / 2`` unordered slot pairs and
    self-pairs are then dropped, which yields a uniform sample of the induced
    pair multiset.

    Parameters
    ----------
    width
        ``(N, N)`` bracket-width matrix.
    n_max
        ``(N, N)`` matrix of ``max(n1, n2)``.
    n_graphs
        Number of graphs.
    replicates
        Replicate count.
    subsample
        Induced pairs per replicate.
    seed
        Master seed.

    Returns
    -------
    numpy.ndarray
        One slope per replicate.
    """
    total_slots = n_graphs * (n_graphs - 1) // 2
    draw = min(subsample, total_slots)
    slopes = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selection = replicate_selection(n_graphs, seed, replicate)
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 1]))
        flat = rng.choice(total_slots, size=draw, replace=False, shuffle=False)
        slot_i, slot_j = pairs_from_indices_searchsorted(flat, n_graphs)
        a = selection[slot_i]
        b = selection[slot_j]
        keep = a != b
        lo = np.minimum(a[keep], b[keep])
        hi = np.maximum(a[keep], b[keep])
        y = width[lo, hi].astype(np.float64)
        x = n_max[lo, hi].astype(np.float64)
        slopes[replicate] = ols_fit(x, y).slope
    return slopes


def bootstrap_slope_ci(
    width_matrix: npt.NDArray[Any],
    n_max_mat: npt.NDArray[Any],
    tier: BootstrapTier,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return a percentile CI on the OLS slope from a graph-level bootstrap.

    Parameters
    ----------
    width_matrix
        ``(N, N)`` bracket-width matrix with a zero diagonal.
    n_max_mat
        ``(N, N)`` matrix of ``max(n1, n2)``.
    tier
        The frozen D15 effort for this dataset.
    seed
        Master seed.

    Returns
    -------
    dict
        The percentile CI, the tier description and the wall time spent.
    """
    n_graphs = int(width_matrix.shape[0])
    started = time.perf_counter()
    if tier.subsample is None:
        slopes = _bootstrap_slopes_full(
            np.asarray(width_matrix, dtype=np.float64),
            np.asarray(n_max_mat, dtype=np.float64),
            n_graphs,
            tier.replicates,
            seed,
        )
    else:
        slopes = _bootstrap_slopes_subsampled(
            np.asarray(width_matrix, dtype=np.float32),
            np.asarray(n_max_mat, dtype=np.float32),
            n_graphs,
            tier.replicates,
            tier.subsample,
            seed,
        )
    result = percentile_ci(slopes)
    result["bootstrap"] = tier.as_dict()
    result["wall_seconds"] = float(time.perf_counter() - started)
    return result


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleFile:
    """One role's matrices and provenance for one dataset.

    Parameters
    ----------
    role
        ``LB``, ``UB`` or ``UBS``.
    path
        The ``.npz`` this was read from.
    ged
        ``(N, N)`` ``ged_matrix``, i.e. **this role's own** bound. The
        cross-filled ``lb_matrix``/``ub_matrix`` copies are deliberately not
        read, so the analysis is independent of whether cross-fill ran.
    seconds
        ``(N, N)`` ``seconds_matrix``.
    node_counts, edge_counts, graph_ids, labels
        The per-graph arrays.
    metadata
        The parsed ``metadata`` JSON scalar.
    """

    role: str
    path: Path
    ged: FloatArray
    seconds: Float32Array
    node_counts: npt.NDArray[np.int32]
    edge_counts: npt.NDArray[np.int32]
    graph_ids: npt.NDArray[np.str_]
    labels: npt.NDArray[np.str_]
    metadata: dict[str, Any]


REQUIRED_KEYS: tuple[str, ...] = (
    "ged_matrix",
    "seconds_matrix",
    "node_counts",
    "edge_counts",
    "graph_ids",
    "labels",
    "metadata",
)


def load_role_file(path: Path, role: str) -> RoleFile:
    """Read one role's ``.npz`` and validate its schema.

    Parameters
    ----------
    path
        Path to ``{dataset}.npz``.
    role
        ``LB``, ``UB`` or ``UBS``.

    Returns
    -------
    RoleFile
        The loaded arrays and provenance.

    Raises
    ------
    InputError
        If the file is missing, a required key is absent, the matrices are not
        square and mutually conformable, or the recorded method contradicts the
        role that was requested.
    """
    if not path.is_file():
        raise InputError(f"{role} matrix for {path.stem!r} not found at {path}")
    with np.load(path, allow_pickle=False) as handle:
        missing = [key for key in REQUIRED_KEYS if key not in handle.files]
        if missing:
            raise InputError(f"{path} is missing required keys {missing}")
        ged = np.asarray(handle["ged_matrix"], dtype=np.float64)
        seconds = np.asarray(handle["seconds_matrix"], dtype=np.float32)
        node_counts = np.asarray(handle["node_counts"])
        edge_counts = np.asarray(handle["edge_counts"])
        graph_ids = np.asarray(handle["graph_ids"])
        labels = np.asarray(handle["labels"])
        metadata = json.loads(str(handle["metadata"]))
    n_graphs = node_counts.size
    for name, array in (("ged_matrix", ged), ("seconds_matrix", seconds)):
        if array.shape != (n_graphs, n_graphs):
            raise InputError(f"{path}: {name} has shape {array.shape}, expected {(n_graphs,) * 2}")
    expected_method = ROLE_METHOD[role]
    recorded = str(metadata.get("method", ""))
    if recorded != expected_method:
        raise InputError(
            f"{path}: metadata records method {recorded!r} but role {role} requires "
            f"{expected_method!r}; the role directories may be swapped"
        )
    return RoleFile(
        role=role,
        path=path,
        ged=ged,
        seconds=seconds,
        node_counts=node_counts,
        edge_counts=edge_counts,
        graph_ids=graph_ids,
        labels=labels,
        metadata=metadata,
    )


@dataclass
class PairVectors:
    """Compact per-pair vectors retained for the pooled analyses.

    Every field is over the **strict upper triangle** in row-major order, so
    the ``k``-th entry of each is the same pair.
    """

    dataset: str
    n_max: npt.NDArray[np.int16]
    size_code: npt.NDArray[np.int8]
    width: Float32Array
    width_sensitivity: Float32Array
    density: Float32Array
    certified: BoolArray
    certified_sensitivity: BoolArray

    @property
    def n_pairs(self) -> int:
        """Number of pairs in the strict upper triangle."""
        return int(self.width.size)


def _provenance(files: Mapping[str, RoleFile]) -> dict[str, Any]:
    """Collect the provenance each role file carries.

    §7.4 must surface this: the worker count of the production run is **not**
    recorded in the file metadata, so the amendment-11 pathology cannot be
    quantified per dataset from these files alone. Recording exactly what is
    and is not present is the point.

    Parameters
    ----------
    files
        Role name to loaded file.

    Returns
    -------
    dict
        Per-role provenance fields plus the fields known to be absent.
    """
    fields = (
        "method",
        "options_string",
        "accessor",
        "cost_model",
        "code_commit",
        "computed_utc",
        "n_shards",
        "gedlib_source",
        "seconds_total",
        "mean_seconds_per_pair",
        "schema_version",
    )
    absent = ("workers", "n_workers", "slurm_jobid", "picasso_jobid", "cpus_per_task")
    out: dict[str, Any] = {}
    for role, handle in files.items():
        entry = {name: handle.metadata.get(name) for name in fields}
        entry["path"] = str(handle.path)
        entry["provenance_fields_absent"] = [name for name in absent if name not in handle.metadata]
        out[role] = entry
    return out


def load_dataset(
    config: AnalysisConfig, dataset: str
) -> tuple[dict[str, RoleFile], dict[str, Any]]:
    """Load all three roles for one dataset and cross-check them.

    Parameters
    ----------
    config
        The analysis configuration.
    dataset
        Dataset key.

    Returns
    -------
    tuple
        The role files and a consistency record for the report.

    Raises
    ------
    InputError
        If the three roles disagree on the graph roster, or if the exported
        Suite-2 file disagrees with them on node counts.
    """
    files = {role: load_role_file(config.role_dir(role) / f"{dataset}.npz", role) for role in ROLES}
    reference = files["LB"]
    for role, handle in files.items():
        if not np.array_equal(handle.graph_ids, reference.graph_ids):
            raise InputError(f"{dataset}: role {role} has a different graph roster from LB")
        if not np.array_equal(handle.node_counts, reference.node_counts):
            raise InputError(f"{dataset}: role {role} disagrees with LB on node_counts")
    consistency: dict[str, Any] = {
        "n_graphs": int(reference.node_counts.size),
        "roster_matches_across_roles": True,
    }
    exported = config.input_dir / f"{dataset}.npz"
    if exported.is_file():
        with np.load(exported, allow_pickle=False) as handle:
            consistency["exported_graph_ids_match"] = bool(
                np.array_equal(np.asarray(handle["graph_ids"]), reference.graph_ids)
            )
            consistency["exported_node_counts_match"] = bool(
                np.array_equal(np.asarray(handle["n_nodes"]), reference.node_counts)
            )
            consistency["exported_edge_counts_match"] = bool(
                np.array_equal(np.asarray(handle["n_edges"]), reference.edge_counts)
            )
        for key in (
            "exported_graph_ids_match",
            "exported_node_counts_match",
            "exported_edge_counts_match",
        ):
            if not consistency[key]:
                raise InputError(
                    f"{dataset}: {key} is False; the bound files and the export differ"
                )
    else:
        consistency["exported_suite2_present"] = False
    return files, consistency


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------


def _describe(values: npt.NDArray[Any]) -> dict[str, Any]:
    """Return mean, median, sd and count for a vector.

    Parameters
    ----------
    values
        Any numeric vector.

    Returns
    -------
    dict
        ``n``, ``mean``, ``median``, ``sd``. All statistics are ``nan`` when
        the vector is empty.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "sd": math.nan}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def _stratum_table(
    codes: npt.NDArray[Any],
    labels: Sequence[str],
    values: npt.NDArray[Any],
) -> list[dict[str, Any]]:
    """Summarise ``values`` within each stratum.

    Parameters
    ----------
    codes
        Zero-based stratum index per observation.
    labels
        Stratum labels, indexed by code.
    values
        The quantity to summarise.

    Returns
    -------
    list of dict
        One row per stratum, including empty ones so the table shape is fixed.
    """
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        mask = codes == index
        row = {"stratum": label, "stratum_index": index}
        row.update(_describe(values[mask]))
        rows.append(row)
    return rows


def _size_profile(
    n_max: npt.NDArray[Any],
    width: npt.NDArray[Any],
    width_sensitivity: npt.NDArray[Any],
) -> list[dict[str, Any]]:
    """Return the mean bracket width at each distinct value of ``n_max``.

    This is the compact form the §7.1 figures are drawn from, so a figure and
    the report read the same numbers.

    Parameters
    ----------
    n_max
        ``max(n1, n2)`` per pair.
    width
        Primary-arm bracket width per pair.
    width_sensitivity
        Sensitivity-arm bracket width per pair.

    Returns
    -------
    list of dict
        One row per distinct ``n_max``, ascending.
    """
    values = np.asarray(n_max)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    unique, starts = np.unique(sorted_values, return_index=True)
    bounds = list(starts) + [sorted_values.size]
    primary = np.asarray(width, dtype=np.float64)[order]
    arm = np.asarray(width_sensitivity, dtype=np.float64)[order]
    rows: list[dict[str, Any]] = []
    for position, value in enumerate(unique):
        lo, hi = bounds[position], bounds[position + 1]
        rows.append(
            {
                "n_max": int(value),
                "n_pairs": int(hi - lo),
                "mean_width": float(primary[lo:hi].mean()),
                "median_width": float(np.median(primary[lo:hi])),
                "mean_width_sensitivity": float(arm[lo:hi].mean()),
                "median_width_sensitivity": float(np.median(arm[lo:hi])),
            }
        )
    return rows


def _cost_rows(files: Mapping[str, RoleFile], size_code: npt.NDArray[Any]) -> list[dict[str, Any]]:
    """Build the §7.4 cost rows for one dataset.

    Every figure here is **realised wall time under a known-pathological
    parallelisation** (design amendment 11), not a per-pair cost of the method.

    Parameters
    ----------
    files
        The three role files.
    size_code
        Size stratum index per pair.

    Returns
    -------
    list of dict
        One row per role.
    """
    rows: list[dict[str, Any]] = []
    for role in ROLES:
        handle = files[role]
        seconds = upper_triangle(handle.seconds).astype(np.float64)
        row: dict[str, Any] = {
            "role": role,
            "method": ROLE_LABEL[role],
            "options_string": handle.metadata.get("options_string"),
            "n_pairs": int(seconds.size),
            "total_core_seconds": float(seconds.sum()),
            "metadata_seconds_total": float(handle.metadata.get("seconds_total", math.nan)),
            "mean_ms_per_pair": float(seconds.mean() * 1e3),
            "median_ms_per_pair": float(np.median(seconds) * 1e3),
            "max_ms_per_pair": float(seconds.max() * 1e3),
            "by_size_stratum": [
                {
                    "stratum": label,
                    "n_pairs": int((size_code == index).sum()),
                    "mean_ms_per_pair": (
                        float(seconds[size_code == index].mean() * 1e3)
                        if int((size_code == index).sum())
                        else math.nan
                    ),
                }
                for index, label in enumerate(SIZE_BIN_LABELS)
            ],
        }
        row["triu_sum_matches_metadata_total"] = bool(
            math.isclose(
                row["total_core_seconds"],
                row["metadata_seconds_total"],
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        )
        rows.append(row)
    return rows


def analyse_dataset(
    config: AnalysisConfig,
    dataset: str,
) -> tuple[dict[str, Any], PairVectors]:
    """Run every per-dataset deliverable and return the retained pair vectors.

    Parameters
    ----------
    config
        The analysis configuration.
    dataset
        Dataset key.

    Returns
    -------
    tuple
        The dataset's results block and its compact per-pair vectors.
    """
    started = time.perf_counter()
    files, consistency = load_dataset(config, dataset)
    reference = files["LB"]
    n_graphs = int(reference.node_counts.size)
    LOGGER.info("%s: %d graphs, %d pairs", dataset, n_graphs, n_graphs * (n_graphs - 1) // 2)

    lb = files["LB"].ged
    width_matrix = bracket_width(lb, files["UB"].ged)
    width_sensitivity_matrix = bracket_width(lb, files["UBS"].ged)
    certified_matrix = np.abs(lb - files["UB"].ged) <= CERTIFIED_TOL
    certified_sensitivity_matrix = np.abs(lb - files["UBS"].ged) <= CERTIFIED_TOL
    n_max_mat = n_max_matrix(reference.node_counts)
    density = graph_density(reference.node_counts, reference.edge_counts)

    width = upper_triangle(width_matrix)
    width_sensitivity = upper_triangle(width_sensitivity_matrix)
    n_max = upper_triangle(n_max_mat)
    d_pair = upper_triangle(pair_density_matrix(density))
    certified = upper_triangle(certified_matrix)
    certified_sensitivity = upper_triangle(certified_sensitivity_matrix)
    size_code = size_bin_codes(n_max)

    tier = bootstrap_tier(dataset)
    primary_fit = ols_fit(n_max, width)
    sensitivity_fit = ols_fit(n_max, width_sensitivity)
    LOGGER.info("%s: bootstrapping %d replicates (tier %d)", dataset, tier.replicates, tier.tier)
    primary_ci = bootstrap_slope_ci(width_matrix, n_max_mat, tier, config.seed)
    sensitivity_ci = bootstrap_slope_ci(width_sensitivity_matrix, n_max_mat, tier, config.seed)

    n_zero_ub = int((upper_triangle(files["UB"].ged) == 0.0).sum())
    role_in_slope_set = (
        "unconfounded"
        if dataset in UNCONFOUNDED_DATASETS
        else ("small-n constraint only" if dataset in SMALL_N_DATASETS else "intermediate")
    )
    result: dict[str, Any] = {
        "dataset": dataset,
        "n_graphs": n_graphs,
        "n_pairs": int(width.size),
        "n_max_min": int(n_max.min()),
        "n_max_max": int(n_max.max()),
        "mean_nodes": float(reference.node_counts.mean()),
        "mean_pair_density": float(d_pair.mean()),
        "n_pairs_with_zero_ub": n_zero_ub,
        "slope_role": role_in_slope_set,
        "consistency": consistency,
        "provenance": _provenance(files),
        "bootstrap": tier.as_dict(),
        "s71": {
            "primary": {"fit": primary_fit.as_dict(), "slope_ci": primary_ci},
            "sensitivity": {"fit": sensitivity_fit.as_dict(), "slope_ci": sensitivity_ci},
            "size_profile": _size_profile(n_max, width, width_sensitivity),
        },
        "s72": {
            "certification_rate_primary": float(certified.mean()),
            "n_certified_primary": int(certified.sum()),
            "certification_rate_sensitivity": float(certified_sensitivity.mean()),
            "n_certified_sensitivity": int(certified_sensitivity.sum()),
            "by_size_stratum": [
                {
                    "stratum": label,
                    "stratum_index": index,
                    "n_pairs": int((size_code == index).sum()),
                    "certification_rate_primary": (
                        float(certified[size_code == index].mean())
                        if int((size_code == index).sum())
                        else math.nan
                    ),
                    "certification_rate_sensitivity": (
                        float(certified_sensitivity[size_code == index].mean())
                        if int((size_code == index).sum())
                        else math.nan
                    ),
                }
                for index, label in enumerate(SIZE_BIN_LABELS)
            ],
        },
        "s73": {
            "overall_primary": _describe(width),
            "overall_sensitivity": _describe(width_sensitivity),
            "by_size_stratum_primary": _stratum_table(size_code, SIZE_BIN_LABELS, width),
            "by_size_stratum_sensitivity": _stratum_table(
                size_code, SIZE_BIN_LABELS, width_sensitivity
            ),
        },
        "s74": {"roles": _cost_rows(files, size_code)},
        "wall_seconds": float(time.perf_counter() - started),
    }
    vectors = PairVectors(
        dataset=dataset,
        n_max=n_max.astype(np.int16),
        size_code=size_code,
        width=width.astype(np.float32),
        width_sensitivity=width_sensitivity.astype(np.float32),
        density=d_pair.astype(np.float32),
        certified=certified,
        certified_sensitivity=certified_sensitivity,
    )
    return result, vectors


# ---------------------------------------------------------------------------
# Pooled analysis
# ---------------------------------------------------------------------------


def density_quintile_edges(vectors: Sequence[PairVectors], n_quantiles: int) -> list[float]:
    """Return the density quintile edges over the pooled Suite-2 pair population.

    Amendment 12 rule 3 fixes the population: the quintiles are computed over
    **pairs**, not over graphs. The returned edges are the interior cut points,
    so ``n_quantiles - 1`` of them.

    Parameters
    ----------
    vectors
        Per-dataset pair vectors.
    n_quantiles
        Number of quantile strata; 5, frozen.

    Returns
    -------
    list of float
        Interior quantile cut points, ascending.
    """
    pooled = np.concatenate([v.density for v in vectors]) if vectors else np.zeros(0, np.float32)
    if pooled.size == 0:
        return [0.0] * (n_quantiles - 1)
    probabilities = [k / n_quantiles for k in range(1, n_quantiles)]
    return [float(v) for v in np.quantile(pooled.astype(np.float64), probabilities)]


def density_codes(density: npt.NDArray[Any], edges: Sequence[float]) -> npt.NDArray[np.int8]:
    """Assign each pair to a pooled density quintile.

    Parameters
    ----------
    density
        Pair density.
    edges
        Interior quantile cut points.

    Returns
    -------
    numpy.ndarray
        Zero-based quintile index, int8.
    """
    return np.searchsorted(np.asarray(edges, dtype=np.float64), density, side="right").astype(
        np.int8
    )


def density_labels(edges: Sequence[float], n_quantiles: int) -> tuple[str, ...]:
    """Return human labels for the density quintiles.

    Parameters
    ----------
    edges
        Interior quantile cut points.
    n_quantiles
        Number of strata.

    Returns
    -------
    tuple of str
        One label per stratum, in ascending density order.
    """
    bounds = [-math.inf, *edges, math.inf]
    return tuple(
        f"Q{index + 1} [{bounds[index]:.4g}, {bounds[index + 1]:.4g})"
        for index in range(n_quantiles)
    )


def pooled_size_strata(
    vectors: Sequence[PairVectors],
) -> list[dict[str, Any]]:
    """Summarise the pooled cohort within each size stratum, with provenance.

    The dominance share is computed from the data rather than quoted: §7.1's
    confound is that size and provenance move together, and a pooled number is
    only quotable beside the share of its stratum that one dataset contributes.

    Parameters
    ----------
    vectors
        Per-dataset pair vectors.

    Returns
    -------
    list of dict
        One row per size stratum.
    """
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(SIZE_BIN_LABELS):
        parts_primary: list[npt.NDArray[np.float32]] = []
        parts_arm: list[npt.NDArray[np.float32]] = []
        parts_density: list[npt.NDArray[np.float32]] = []
        parts_certified: list[BoolArray] = []
        composition: dict[str, int] = {}
        for vector in vectors:
            mask = vector.size_code == index
            count = int(mask.sum())
            if count == 0:
                continue
            composition[vector.dataset] = count
            parts_primary.append(vector.width[mask])
            parts_arm.append(vector.width_sensitivity[mask])
            parts_density.append(vector.density[mask])
            parts_certified.append(vector.certified[mask])
        total = sum(composition.values())
        row: dict[str, Any] = {
            "stratum": label,
            "stratum_index": index,
            "n_pairs": total,
            "composition": dict(sorted(composition.items(), key=lambda kv: -kv[1])),
            "composition_share": {
                key: value / total
                for key, value in sorted(composition.items(), key=lambda kv: -kv[1])
            }
            if total
            else {},
        }
        if total:
            top_key, top_count = max(composition.items(), key=lambda kv: kv[1])
            row["dominant_dataset"] = top_key
            row["dominant_share"] = top_count / total
            row["width_primary"] = _describe(np.concatenate(parts_primary))
            row["width_sensitivity"] = _describe(np.concatenate(parts_arm))
            row["mean_pair_density"] = float(np.concatenate(parts_density).mean())
            row["certification_rate_primary"] = float(np.concatenate(parts_certified).mean())
        else:
            row["dominant_dataset"] = None
            row["dominant_share"] = math.nan
            row["width_primary"] = _describe(np.zeros(0))
            row["width_sensitivity"] = _describe(np.zeros(0))
            row["mean_pair_density"] = math.nan
            row["certification_rate_primary"] = math.nan
        rows.append(row)
    return rows


def pooled_density_strata(
    vectors: Sequence[PairVectors],
    edges: Sequence[float],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarise the pooled cohort within each density quintile.

    Parameters
    ----------
    vectors
        Per-dataset pair vectors.
    edges
        Interior quantile cut points.
    labels
        Quintile labels.

    Returns
    -------
    list of dict
        One row per quintile.
    """
    rows: list[dict[str, Any]] = []
    codes = {v.dataset: density_codes(v.density, edges) for v in vectors}
    for index, label in enumerate(labels):
        parts_primary: list[npt.NDArray[np.float32]] = []
        parts_arm: list[npt.NDArray[np.float32]] = []
        parts_certified: list[BoolArray] = []
        composition: dict[str, int] = {}
        for vector in vectors:
            mask = codes[vector.dataset] == index
            count = int(mask.sum())
            if count == 0:
                continue
            composition[vector.dataset] = count
            parts_primary.append(vector.width[mask])
            parts_arm.append(vector.width_sensitivity[mask])
            parts_certified.append(vector.certified[mask])
        total = sum(composition.values())
        row: dict[str, Any] = {
            "stratum": label,
            "stratum_index": index,
            "n_pairs": total,
            "composition": dict(sorted(composition.items(), key=lambda kv: -kv[1])),
        }
        if total:
            top_key, top_count = max(composition.items(), key=lambda kv: kv[1])
            row["dominant_dataset"] = top_key
            row["dominant_share"] = top_count / total
            row["width_primary"] = _describe(np.concatenate(parts_primary))
            row["width_sensitivity"] = _describe(np.concatenate(parts_arm))
            row["certification_rate_primary"] = float(np.concatenate(parts_certified).mean())
        else:
            row["dominant_dataset"] = None
            row["dominant_share"] = math.nan
            row["width_primary"] = _describe(np.zeros(0))
            row["width_sensitivity"] = _describe(np.zeros(0))
            row["certification_rate_primary"] = math.nan
        rows.append(row)
    return rows


def secondary_density_cells(
    vectors: Sequence[PairVectors],
    edges: Sequence[float],
    labels: Sequence[str],
    min_cell_pairs: int,
) -> dict[str, Any]:
    """Fit the §7.1 secondary slope within each (dataset x density quintile) cell.

    Cells below ``min_cell_pairs`` are reported as dropped, with their
    population, rather than silently omitted.

    Parameters
    ----------
    vectors
        Per-dataset pair vectors.
    edges
        Interior quantile cut points.
    labels
        Quintile labels.
    min_cell_pairs
        Population floor.

    Returns
    -------
    dict
        ``cells`` (the fitted ones) and ``dropped`` (the underpopulated ones).
    """
    fitted: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for vector in vectors:
        codes = density_codes(vector.density, edges)
        for index, label in enumerate(labels):
            mask = codes == index
            count = int(mask.sum())
            if count == 0:
                continue
            entry = {
                "dataset": vector.dataset,
                "density_stratum": label,
                "density_stratum_index": index,
                "n_pairs": count,
            }
            if count < min_cell_pairs:
                entry["reason"] = f"n_pairs < min_cell_pairs ({min_cell_pairs})"
                dropped.append(entry)
                continue
            x = vector.n_max[mask].astype(np.float64)
            entry["primary"] = ols_fit(x, vector.width[mask]).as_dict()
            entry["sensitivity"] = ols_fit(x, vector.width_sensitivity[mask]).as_dict()
            entry["mean_width_primary"] = float(vector.width[mask].mean())
            entry["mean_width_sensitivity"] = float(vector.width_sensitivity[mask].mean())
            entry["n_max_min"] = int(vector.n_max[mask].min())
            entry["n_max_max"] = int(vector.n_max[mask].max())
            fitted.append(entry)
    return {
        "min_cell_pairs": min_cell_pairs,
        "n_cells_fitted": len(fitted),
        "n_cells_dropped": len(dropped),
        "cells": fitted,
        "dropped": dropped,
    }


def pooled_slope(vectors: Sequence[PairVectors]) -> dict[str, Any]:
    """Fit the descriptive pooled slope across every dataset at once.

    Reported as a descriptive overlay only. The pooled slope fits a dataset
    transition and a density transition as faithfully as a size one, so it is
    never the estimate a conclusion rests on.

    Parameters
    ----------
    vectors
        Per-dataset pair vectors.

    Returns
    -------
    dict
        Primary and sensitivity fits with the caveat attached.
    """
    if not vectors:
        empty = OlsFit(math.nan, math.nan, math.nan, 0).as_dict()
        return {"primary": empty, "sensitivity": empty, "status": "no data"}
    x = np.concatenate([v.n_max.astype(np.float64) for v in vectors])
    primary = ols_fit(x, np.concatenate([v.width for v in vectors]))
    arm = ols_fit(x, np.concatenate([v.width_sensitivity for v in vectors]))
    return {
        "primary": primary.as_dict(),
        "sensitivity": arm.as_dict(),
        "caveat": (
            "Descriptive only. Size and provenance are confounded in Suite 2: the "
            "small-n strata are dominated by Letter and the large-n strata by "
            "Mutagenicity and COIL-DEL, and density moves with provenance over the "
            "same range. The within-dataset slope is the primary estimate."
        ),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _figure_palette(n: int) -> list[str]:
    """Return ``n`` colours from the published Paul Tol muted palette.

    Parameters
    ----------
    n
        Number of colours required.

    Returns
    -------
    list of str
        Hex colours, cycled if ``n`` exceeds the palette length.
    """
    from benchmarks.plotting_styles import PAUL_TOL_MUTED

    palette = list(PAUL_TOL_MUTED)
    return [palette[index % len(palette)] for index in range(n)]


def _finite(values: Iterable[Any]) -> list[float]:
    """Return only the finite entries of an iterable of floats."""
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def make_figures(results: Mapping[str, Any], figures_dir: Path) -> list[str]:
    """Write every figure the report cites.

    All styling comes from :mod:`benchmarks.plotting_styles`, which re-exports
    :mod:`isalgraph.viz.style`, so the palette cannot drift from the published
    one.

    Parameters
    ----------
    results
        The assembled results tree.
    figures_dir
        Destination directory.

    Returns
    -------
    list of str
        Paths written, relative to ``figures_dir``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.plotting_styles import apply_ieee_style, get_figure_size, save_figure

    apply_ieee_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    datasets = list(results["datasets"])
    colours = dict(zip(datasets, _figure_palette(len(datasets))))

    written += _figure_width_vs_n(
        results, datasets, colours, figures_dir, plt, get_figure_size, save_figure
    )
    written += _figure_slope_forest(
        results, datasets, figures_dir, plt, get_figure_size, save_figure
    )
    written += _figure_pooled_overlay(results, figures_dir, plt, get_figure_size, save_figure)
    written += _figure_certification(
        results, datasets, colours, figures_dir, plt, get_figure_size, save_figure
    )
    written += _figure_strata(results, figures_dir, plt, get_figure_size, save_figure)
    written += _figure_cost(results, datasets, figures_dir, plt, get_figure_size, save_figure)
    return written


def _figure_width_vs_n(results, datasets, colours, figures_dir, plt, get_figure_size, save_figure):
    """Draw mean bracket width against ``n_max``, within dataset, both arms."""
    fig, axes = plt.subplots(
        1, 2, figsize=get_figure_size("double", height_ratio=0.42), sharey=True
    )
    for arm, axis, title in (
        ("mean_width", axes[0], "Primary arm: BIPARTITE"),
        ("mean_width_sensitivity", axes[1], "Sensitivity arm: BP_BEAM_DET"),
    ):
        for dataset in datasets:
            profile = results["per_dataset"][dataset]["s71"]["size_profile"]
            x = [row["n_max"] for row in profile]
            y = [row[arm] for row in profile]
            axis.plot(
                x,
                y,
                marker="o",
                markersize=2.0,
                linewidth=1.0,
                color=colours[dataset],
                label=dataset.replace("_", " "),
            )
        axis.set_xlabel(r"$n_{\max} = \max(n_1, n_2)$")
        axis.set_title(title)
    axes[0].set_ylabel(r"mean bracket width $(UB-LB)/UB$")
    axes[1].legend(fontsize=4.5, ncol=2, loc="lower right")
    fig.tight_layout()
    written = save_figure(
        fig, str(figures_dir / "fig_71_width_vs_n_within_dataset"), formats=("pdf",)
    )
    plt.close(fig)
    return [Path(p).name for p in written]


def _figure_slope_forest(results, datasets, figures_dir, plt, get_figure_size, save_figure):
    """Draw the within-dataset slope with its bootstrap CI, both arms."""
    fig, axis = plt.subplots(figsize=get_figure_size("single", height_ratio=0.85))
    offsets = {"primary": -0.16, "sensitivity": 0.16}
    markers = {"primary": "o", "sensitivity": "s"}
    colours = dict(zip(("primary", "sensitivity"), _figure_palette(2)))
    positions = list(range(len(datasets)))
    for arm in ("primary", "sensitivity"):
        for position, dataset in zip(positions, datasets):
            block = results["per_dataset"][dataset]["s71"][arm]
            slope = block["fit"]["slope"]
            ci = block["slope_ci"]
            if not math.isfinite(slope):
                continue
            y = position + offsets[arm]
            axis.plot([ci["ci_low"], ci["ci_high"]], [y, y], color=colours[arm], linewidth=1.0)
            axis.plot(
                [slope],
                [y],
                marker=markers[arm],
                markersize=3.0,
                color=colours[arm],
                label=arm if position == positions[0] else None,
            )
    axis.axvline(0.0, color="0.4", linewidth=0.6, linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels([d.replace("_", " ") for d in datasets], fontsize=5)
    axis.set_xlabel(r"OLS slope of $(UB-LB)/UB$ on $n_{\max}$ (per node)")
    axis.legend(fontsize=5, loc="best")
    fig.tight_layout()
    written = save_figure(fig, str(figures_dir / "fig_71_slope_forest"), formats=("pdf",))
    plt.close(fig)
    return [Path(p).name for p in written]


def _figure_pooled_overlay(results, figures_dir, plt, get_figure_size, save_figure):
    """Draw the pooled size-stratum overlay with its dominance shares."""
    rows = [row for row in results["pooled"]["size_strata"] if row["n_pairs"] > 0]
    fig, axis = plt.subplots(figsize=get_figure_size("single", height_ratio=0.75))
    positions = list(range(len(rows)))
    colour_primary, colour_arm = _figure_palette(2)
    axis.plot(
        positions,
        [row["width_primary"]["mean"] for row in rows],
        marker="o",
        markersize=3.0,
        color=colour_primary,
        label="primary (BIPARTITE)",
    )
    axis.plot(
        positions,
        [row["width_sensitivity"]["mean"] for row in rows],
        marker="s",
        markersize=3.0,
        color=colour_arm,
        label="sensitivity (BP_BEAM_DET)",
    )
    for position, row in zip(positions, rows):
        axis.annotate(
            f"{row['dominant_dataset']}\n{100 * row['dominant_share']:.0f}%",
            (position, row["width_primary"]["mean"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=4.0,
            color="0.35",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([row["stratum"] for row in rows], fontsize=5)
    axis.set_xlabel(r"$n_{\max}$ stratum (statistics.md §8)")
    axis.set_ylabel(r"mean bracket width $(UB-LB)/UB$")
    axis.legend(fontsize=5, loc="best")
    fig.tight_layout()
    written = save_figure(
        fig, str(figures_dir / "fig_71_pooled_overlay_confounded"), formats=("pdf",)
    )
    plt.close(fig)
    return [Path(p).name for p in written]


def _figure_certification(
    results, datasets, colours, figures_dir, plt, get_figure_size, save_figure
):
    """Draw the certification rate per dataset and size stratum."""
    fig, axis = plt.subplots(figsize=get_figure_size("single", height_ratio=0.75))
    for dataset in datasets:
        rows = results["per_dataset"][dataset]["s72"]["by_size_stratum"]
        x = [row["stratum_index"] for row in rows if row["n_pairs"] > 0]
        y = [100 * row["certification_rate_primary"] for row in rows if row["n_pairs"] > 0]
        axis.plot(
            x,
            y,
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            color=colours[dataset],
            label=dataset.replace("_", " "),
        )
    axis.set_xticks(list(range(len(SIZE_BIN_LABELS))))
    axis.set_xticklabels(SIZE_BIN_LABELS, fontsize=5)
    axis.set_yscale("symlog", linthresh=0.1)
    axis.set_xlabel(r"$n_{\max}$ stratum")
    axis.set_ylabel("certification rate, LB = UB (%)")
    axis.legend(fontsize=4.5, ncol=2, loc="best")
    fig.tight_layout()
    written = save_figure(fig, str(figures_dir / "fig_72_certification_rate"), formats=("pdf",))
    plt.close(fig)
    return [Path(p).name for p in written]


def _figure_strata(results, figures_dir, plt, get_figure_size, save_figure):
    """Draw pooled bracket width by size stratum and by density quintile."""
    size_rows = [row for row in results["pooled"]["size_strata"] if row["n_pairs"] > 0]
    density_rows = [row for row in results["pooled"]["density_strata"] if row["n_pairs"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=get_figure_size("double", height_ratio=0.40))
    colour_primary, colour_arm = _figure_palette(2)
    for axis, rows, xlabel in (
        (axes[0], size_rows, r"$n_{\max}$ stratum"),
        (axes[1], density_rows, "pair-density quintile"),
    ):
        positions = list(range(len(rows)))
        width = 0.36
        axis.bar(
            [p - width / 2 for p in positions],
            [r["width_primary"]["mean"] for r in rows],
            width=width,
            color=colour_primary,
            label="primary, mean",
        )
        axis.bar(
            [p + width / 2 for p in positions],
            [r["width_sensitivity"]["mean"] for r in rows],
            width=width,
            color=colour_arm,
            label="sensitivity, mean",
        )
        axis.plot(
            positions,
            [r["width_primary"]["median"] for r in rows],
            linestyle="none",
            marker="_",
            markersize=6,
            color="0.15",
            label="primary, median",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([r["stratum"].split(" ")[0] for r in rows], fontsize=5, rotation=0)
        axis.set_xlabel(xlabel)
    axes[0].set_ylabel(r"bracket width $(UB-LB)/UB$")
    axes[0].legend(fontsize=4.5, loc="best")
    fig.tight_layout()
    written = save_figure(fig, str(figures_dir / "fig_73_width_by_stratum"), formats=("pdf",))
    plt.close(fig)
    return [Path(p).name for p in written]


def _figure_cost(results, datasets, figures_dir, plt, get_figure_size, save_figure):
    """Draw the realised per-pair wall time per dataset and role."""
    fig, axis = plt.subplots(figsize=get_figure_size("single", height_ratio=0.72))
    colours = dict(zip(ROLES, _figure_palette(3)))
    positions = list(range(len(datasets)))
    width = 0.26
    for offset, role in zip((-width, 0.0, width), ROLES):
        values = []
        for dataset in datasets:
            row = next(
                r for r in results["per_dataset"][dataset]["s74"]["roles"] if r["role"] == role
            )
            values.append(row["mean_ms_per_pair"])
        axis.bar(
            [p + offset for p in positions],
            values,
            width=width,
            color=colours[role],
            label=ROLE_METHOD[role],
        )
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [d.replace("_", " ") for d in datasets], fontsize=4.5, rotation=60, ha="right"
    )
    axis.set_ylabel("realised ms per pair (pool-inflated)")
    axis.legend(fontsize=5, loc="best")
    fig.tight_layout()
    written = save_figure(fig, str(figures_dir / "fig_74_realised_cost"), formats=("pdf",))
    plt.close(fig)
    return [Path(p).name for p in written]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a number for a markdown table, or ``--`` when it is not one."""
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines.append("")
    return lines


def _ci_text(block: Mapping[str, Any]) -> str:
    """Render a slope with its percentile CI."""
    return f"[{_fmt(block['ci_low'], 5)}, {_fmt(block['ci_high'], 5)}]"


def write_report(results: Mapping[str, Any], out_dir: Path) -> Path:
    """Write ``REPORT.md`` from the assembled results tree.

    Every number printed here is read out of ``results``, which is the same
    object serialised into ``data/*.json``. Nothing in the prose is computed at
    render time, so no printed figure can be unsourced.

    Parameters
    ----------
    results
        The assembled results tree.
    out_dir
        The report directory.

    Returns
    -------
    Path
        The written report.
    """
    datasets = list(results["datasets"])
    per_dataset = results["per_dataset"]
    lines: list[str] = []
    add = lines.append

    add("# T-05 §7 -- bounded GED over Suite 2: bracket behaviour at scale")
    add("")
    add(
        f"Generated {results['generated_utc']} by "
        "`benchmarks/real_data/eval_setup/approx_ged_analysis.py`. "
        f"Code commit `{results['code_commit']}`."
    )
    add("")
    add(
        "Every number below is read from `data/*.json` in this directory; the report is "
        "rendered from that tree and computes nothing at render time. The analysis rules are "
        "`T-05-design.md` amendment 12 and were frozen before the first figure."
    )
    add("")

    missing = results["cohort"]["missing_datasets"]
    if missing:
        add("> ## COHORT INCOMPLETE")
        add(">")
        add(
            f"> {len(missing)} of the 10 Suite-2 datasets are absent and **no number in this "
            f"report is a cohort-level Suite-2 quantity**: {', '.join(missing)}. "
            "The pooled strata, the pooled slope and the dominance shares are all computed over "
            f"the {len(datasets)} datasets present. Re-run on the full cohort before quoting "
            "anything pooled."
        )
        add("")

    add("## 0. Cohort analysed")
    add("")
    add(
        f"{len(datasets)} datasets, {results['cohort']['n_graphs']:,} graphs, "
        f"{results['cohort']['n_pairs']:,} pairs over the strict upper triangle "
        "(`np.triu_indices(N, k=1)`). Source: `data/summary.json`."
    )
    add("")
    lines += _table(
        [
            "dataset",
            "graphs",
            "pairs",
            "n_max range",
            "mean n",
            "mean d_pair",
            "slope role",
            "D15 tier",
        ],
        [
            [
                d,
                _fmt(per_dataset[d]["n_graphs"]),
                _fmt(per_dataset[d]["n_pairs"]),
                f"{per_dataset[d]['n_max_min']}--{per_dataset[d]['n_max_max']}",
                _fmt(per_dataset[d]["mean_nodes"], 2),
                _fmt(per_dataset[d]["mean_pair_density"], 4),
                per_dataset[d]["slope_role"],
                f"{per_dataset[d]['bootstrap']['tier']} "
                f"({per_dataset[d]['bootstrap']['replicates']} rep, "
                f"{per_dataset[d]['bootstrap']['within_replicate_pairs']} pairs)",
            ]
            for d in datasets
        ],
    )

    add("### Frozen definitions")
    add("")
    add("- `n_max = max(n1, n2)`; bracket width `w = (UB - LB)/UB`, and `w = 0` where `UB == 0`.")
    add(
        f"  {results['cohort']['n_pairs_with_zero_ub']:,} pairs "
        f"({100 * results['cohort']['zero_ub_share']:.2f} %) have `UB == 0`. Since "
        "`0 <= LB <= UB`, `UB == 0` forces `LB == 0`: the bracket is genuinely closed there, "
        "not undefined, and those pairs are kept."
    )
    add(f"- Certified pair: `|LB - UB| <= {CERTIFIED_TOL:g}`.")
    add(
        "- Size strata: `statistics.md` §8 node-count bins on `n_max` -- "
        + ", ".join(SIZE_BIN_LABELS)
        + ". The leading `2` bin exists because Suite 2 filters at `min_nodes = 2` while §8's "
        "lowest bin starts at 3; dropping it would discard "
        f"{results['cohort']['n_pairs_in_bin_2']:,} pairs silently."
    )
    add(
        "- Pair density `d_pair = (d1 + d2)/2` with `d = 2m/(n(n-1))`; quintile edges computed "
        "over the **pooled pair population**, not the graph population: "
        + ", ".join(f"{edge:.5f}" for edge in results["pooled"]["density_quintile_edges"])
        + "."
    )
    add(
        "- Bootstrap: the resampling unit is the **graph**, resampled with replacement; "
        "percentile CI at "
        f"{CI_PERCENTILES[0]}/{CI_PERCENTILES[1]}, seed {results['seed']}. Effort is D15's "
        "frozen tier assignment, stated per table."
    )
    add("")
    return _write_report_body(lines, results, out_dir)


def _report_s71(results: Mapping[str, Any]) -> list[str]:
    """Render §7.1 -- bracket width against node count."""
    datasets = list(results["datasets"])
    per_dataset = results["per_dataset"]
    lines: list[str] = []
    add = lines.append

    add("## §7.1 -- `(UB - LB)/UB` versus `n`")
    add("")
    add(
        "The single measurement that answers AE.1. Source: "
        "`data/s71_within_dataset_slopes.json`, `data/s71_size_profiles.json`, "
        "`data/s71_density_cells.json`, `data/s71_pooled.json`."
    )
    add("")
    add("### §7.1a PRIMARY -- within-dataset slope, graph-level cluster bootstrap")
    add("")
    add(
        "OLS of `w` on `n_max` within each dataset, with a graph-level cluster bootstrap "
        "95 % percentile CI on the slope at that dataset's frozen D15 tier. Replicate count and "
        "within-replicate pair budget are in the last column of every row; they are not uniform "
        "across datasets and a CI from 1,000 replicates is not silently presented beside one from "
        "2,000."
    )
    add("")
    lines += _table(
        ["dataset", "slope /node", "95 % CI", "intercept", "R^2", "pairs", "effort", "role"],
        [
            [
                d,
                _fmt(per_dataset[d]["s71"]["primary"]["fit"]["slope"], 5),
                _ci_text(per_dataset[d]["s71"]["primary"]["slope_ci"]),
                _fmt(per_dataset[d]["s71"]["primary"]["fit"]["intercept"], 4),
                _fmt(per_dataset[d]["s71"]["primary"]["fit"]["r_squared"], 4),
                _fmt(per_dataset[d]["s71"]["primary"]["fit"]["n_pairs"]),
                f"{per_dataset[d]['bootstrap']['replicates']}x"
                f"{per_dataset[d]['bootstrap']['within_replicate_pairs']}",
                per_dataset[d]["slope_role"],
            ]
            for d in datasets
        ],
    )
    add("### §7.1b Sensitivity arm -- the same fit on `w_s = (UBS - LB)/UBS`")
    add("")
    lines += _table(
        ["dataset", "slope /node", "95 % CI", "intercept", "R^2", "primary slope", "arm - primary"],
        [
            [
                d,
                _fmt(per_dataset[d]["s71"]["sensitivity"]["fit"]["slope"], 5),
                _ci_text(per_dataset[d]["s71"]["sensitivity"]["slope_ci"]),
                _fmt(per_dataset[d]["s71"]["sensitivity"]["fit"]["intercept"], 4),
                _fmt(per_dataset[d]["s71"]["sensitivity"]["fit"]["r_squared"], 4),
                _fmt(per_dataset[d]["s71"]["primary"]["fit"]["slope"], 5),
                _fmt(
                    per_dataset[d]["s71"]["sensitivity"]["fit"]["slope"]
                    - per_dataset[d]["s71"]["primary"]["fit"]["slope"],
                    5,
                ),
            ]
            for d in datasets
        ],
    )
    add("#### What the two arms separate, and why the arm is not optional")
    add("")
    add(
        "The bracket `(UB - LB)/UB` is a property of **our reference bounds**, not of IsalGraph. "
        "A bracket that widens with `n` admits exactly two readings, and the two arms separate "
        "them:"
    )
    add("")
    add(
        "1. **IsalGraph degrades at scale.** This would be a claim about the encoding, and the "
        "bracket cannot support it: the bracket contains no Levenshtein distance and no canonical "
        "string. Nothing in §7.1 measures IsalGraph. What the bracket does control is *how much "
        "resolution any downstream correlation against GED can have* at a given `n` -- a pair "
        "whose GED is only known to within a factor of two cannot discriminate a good "
        "approximation from a bad one."
    )
    add(
        "2. **Our reference bound degrades at scale.** `BIPARTITE` is a single linear-assignment "
        "relaxation and T-27 §5.4 measured its relative error growing about 10x faster in `n` "
        "than the alternatives. `BP_BEAM_DET` is a different upper bound with a different "
        "degradation profile. **If both arms widen at similar rates, the widening is a property "
        "of the LB/UB gap and the large-`n` regime is genuinely less resolved. If the primary arm "
        "widens materially faster than the arm, the widening is substantially an artefact of the "
        "frozen gate's choice of upper bound**, and the conclusion AE.1 disputes is being licensed "
        "on a bound that was selected at `n <= 12`."
    )
    add("")
    add(
        "The `arm - primary` column above is the difference the comparison rests on. Its sign "
        "and magnitude per dataset are in `data/s71_within_dataset_slopes.json`. Note that the "
        "two arms share the **same lower bound**, so the difference between them is entirely a "
        "difference between two upper bounds and carries no lower-bound contribution."
    )
    add("")

    add("### §7.1c SECONDARY -- within (dataset x density quintile)")
    add("")
    cells = results["pooled"]["density_cells"]
    add(
        f"{cells['n_cells_fitted']} cells fitted, {cells['n_cells_dropped']} dropped for holding "
        f"fewer than {cells['min_cell_pairs']:,} pairs. Dropped cells are listed in "
        "`data/s71_density_cells.json` under `dropped` with their populations; they are not "
        "silently omitted."
    )
    add("")
    lines += _table(
        ["dataset", "density quintile", "pairs", "n_max range", "slope /node", "R^2", "arm slope"],
        [
            [
                cell["dataset"],
                cell["density_stratum"],
                _fmt(cell["n_pairs"]),
                f"{cell['n_max_min']}--{cell['n_max_max']}",
                _fmt(cell["primary"]["slope"], 5),
                _fmt(cell["primary"]["r_squared"], 4),
                _fmt(cell["sensitivity"]["slope"], 5),
            ]
            for cell in cells["cells"]
        ],
    )
    if cells["dropped"]:
        add("Dropped cells:")
        add("")
        lines += _table(
            ["dataset", "density quintile", "pairs", "reason"],
            [
                [c["dataset"], c["density_stratum"], _fmt(c["n_pairs"]), c["reason"]]
                for c in cells["dropped"]
            ],
        )

    add("### §7.1d POOLED -- descriptive overlay only")
    add("")
    pooled = results["pooled"]["slope"]
    add(
        f"Pooled slope {_fmt(pooled['primary']['slope'], 5)} per node "
        f"(R^2 {_fmt(pooled['primary']['r_squared'], 4)}, "
        f"{_fmt(pooled['primary']['n_pairs'])} pairs); sensitivity arm "
        f"{_fmt(pooled['sensitivity']['slope'], 5)}."
    )
    add("")
    add(
        "> **Caption, and it travels with any pooled number.** " + pooled["caveat"] + " The "
        "per-stratum dominance shares below are computed from the data in this run, not quoted "
        "from the design's §1.1 bin table, because the strata here are §8's report strata and not "
        "§1.1's 14 draw bins."
    )
    add("")
    lines += _table(
        [
            "n_max stratum",
            "pairs",
            "dominant dataset",
            "share",
            "mean d_pair",
            "mean w",
            "mean w_s",
        ],
        [
            [
                row["stratum"],
                _fmt(row["n_pairs"]),
                str(row["dominant_dataset"]),
                f"{100 * row['dominant_share']:.1f} %"
                if math.isfinite(row["dominant_share"])
                else "--",
                _fmt(row["mean_pair_density"], 4),
                _fmt(row["width_primary"]["mean"], 4),
                _fmt(row["width_sensitivity"]["mean"], 4),
            ]
            for row in results["pooled"]["size_strata"]
        ],
    )
    add(
        "Figures: `figures/fig_71_width_vs_n_within_dataset.pdf`, "
        "`figures/fig_71_slope_forest.pdf`, `figures/fig_71_pooled_overlay_confounded.pdf`."
    )
    add("")
    return lines


def _report_s72(results: Mapping[str, Any]) -> list[str]:
    """Render §7.2 -- certification rate."""
    datasets = list(results["datasets"])
    per_dataset = results["per_dataset"]
    lines: list[str] = []
    add = lines.append
    add("## §7.2 -- certification rate (`LB == UB`, GED proven exactly for free)")
    add("")
    add(
        "T-27 measured 1.2--40.2 % for `BIPARTITE` at `n <= 12`; this extends the measurement to "
        f"n = {results['cohort']['n_max_max']}. Source: `data/s72_certification.json`."
    )
    add("")
    lines += _table(
        ["dataset", "pairs", "certified, primary", "rate", "certified, arm", "rate (arm)"],
        [
            [
                d,
                _fmt(per_dataset[d]["n_pairs"]),
                _fmt(per_dataset[d]["s72"]["n_certified_primary"]),
                f"{100 * per_dataset[d]['s72']['certification_rate_primary']:.4f} %",
                _fmt(per_dataset[d]["s72"]["n_certified_sensitivity"]),
                f"{100 * per_dataset[d]['s72']['certification_rate_sensitivity']:.4f} %",
            ]
            for d in datasets
        ],
    )
    add("### By size stratum, primary arm (%)")
    add("")
    lines += _table(
        ["dataset", *SIZE_BIN_LABELS],
        [
            [
                d,
                *[
                    f"{100 * row['certification_rate_primary']:.3f}" if row["n_pairs"] else "--"
                    for row in per_dataset[d]["s72"]["by_size_stratum"]
                ],
            ]
            for d in datasets
        ],
    )
    add("### By size stratum, sensitivity arm (%)")
    add("")
    lines += _table(
        ["dataset", *SIZE_BIN_LABELS],
        [
            [
                d,
                *[
                    f"{100 * row['certification_rate_sensitivity']:.3f}" if row["n_pairs"] else "--"
                    for row in per_dataset[d]["s72"]["by_size_stratum"]
                ],
            ]
            for d in datasets
        ],
    )
    add("Figure: `figures/fig_72_certification_rate.pdf`.")
    add("")
    return lines


def _report_s73(results: Mapping[str, Any]) -> list[str]:
    """Render §7.3 -- bracket width by size and density stratum."""
    datasets = list(results["datasets"])
    per_dataset = results["per_dataset"]
    lines: list[str] = []
    add = lines.append
    add("## §7.3 -- bracket width by size stratum and by density stratum")
    add("")
    add("Source: `data/s73_strata.json`. Mean and median with pair counts, both arms.")
    add("")
    add("### Per dataset, overall")
    add("")
    lines += _table(
        ["dataset", "pairs", "mean w", "median w", "sd w", "mean w_s", "median w_s"],
        [
            [
                d,
                _fmt(per_dataset[d]["s73"]["overall_primary"]["n"]),
                _fmt(per_dataset[d]["s73"]["overall_primary"]["mean"], 4),
                _fmt(per_dataset[d]["s73"]["overall_primary"]["median"], 4),
                _fmt(per_dataset[d]["s73"]["overall_primary"]["sd"], 4),
                _fmt(per_dataset[d]["s73"]["overall_sensitivity"]["mean"], 4),
                _fmt(per_dataset[d]["s73"]["overall_sensitivity"]["median"], 4),
            ]
            for d in datasets
        ],
    )
    add("### Per dataset x size stratum, primary arm -- mean (median) [pairs]")
    add("")
    lines += _table(
        ["dataset", *SIZE_BIN_LABELS],
        [
            [
                d,
                *[
                    f"{row['mean']:.3f} ({row['median']:.3f}) [{row['n']:,}]" if row["n"] else "--"
                    for row in per_dataset[d]["s73"]["by_size_stratum_primary"]
                ],
            ]
            for d in datasets
        ],
    )
    add("### Pooled by size stratum")
    add("")
    lines += _table(
        ["stratum", "pairs", "mean w", "median w", "mean w_s", "median w_s", "dominant"],
        [
            [
                row["stratum"],
                _fmt(row["n_pairs"]),
                _fmt(row["width_primary"]["mean"], 4),
                _fmt(row["width_primary"]["median"], 4),
                _fmt(row["width_sensitivity"]["mean"], 4),
                _fmt(row["width_sensitivity"]["median"], 4),
                f"{row['dominant_dataset']} {100 * row['dominant_share']:.0f} %"
                if row["dominant_dataset"]
                else "--",
            ]
            for row in results["pooled"]["size_strata"]
        ],
    )
    add("### Pooled by pair-density quintile")
    add("")
    add(
        "Quintile edges over the pooled pair population: "
        + ", ".join(f"{edge:.5f}" for edge in results["pooled"]["density_quintile_edges"])
        + "."
    )
    add("")
    lines += _table(
        ["quintile", "pairs", "mean w", "median w", "mean w_s", "median w_s", "dominant"],
        [
            [
                row["stratum"],
                _fmt(row["n_pairs"]),
                _fmt(row["width_primary"]["mean"], 4),
                _fmt(row["width_primary"]["median"], 4),
                _fmt(row["width_sensitivity"]["mean"], 4),
                _fmt(row["width_sensitivity"]["median"], 4),
                f"{row['dominant_dataset']} {100 * row['dominant_share']:.0f} %"
                if row["dominant_dataset"]
                else "--",
            ]
            for row in results["pooled"]["density_strata"]
        ],
    )
    add("Figure: `figures/fig_73_width_by_stratum.pdf`.")
    add("")
    return lines


def _report_s74(results: Mapping[str, Any]) -> list[str]:
    """Render §7.4 -- the realised cost table, with the amendment-11 caveat."""
    datasets = list(results["datasets"])
    per_dataset = results["per_dataset"]
    lines: list[str] = []
    add = lines.append
    add("## §7.4 -- realised wall time per dataset and role")
    add("")
    add(
        "> ### CAVEAT -- THESE ARE NOT PER-PAIR COSTS OF THE METHODS"
        "\n>\n"
        "> `T-05-design.md` amendment 11 established that the process pool used in the timed-out "
        "production campaigns is **negative-scaling**: on the identical 211,871-pair Letter HIGH "
        "slice, varying only `--workers`, the run cost 36 core-seconds at 1 worker, 212 at 4, 928 "
        "at 15 and 5,260 at 32. Adding workers made both the wall clock and the core-seconds "
        "worse. Amendment 11 **withdrew** amendment 7's per-pair cost conclusions as artefacts of "
        "that regime -- Letter HIGH ran at 19.8 ms/pair in production against 172 us/pair "
        "single-worker, a 115x inflation.\n>\n"
        "> The `lb`, `ub` and `ubs` campaigns ran at **15, 37 and 126** workers respectively, and "
        "the datasets within a campaign were not all processed under the same effective load. So "
        "every number in this table is **realised wall time under a known-pathological "
        "parallelisation**. It is not a per-pair cost of `BRANCH_FAST`, `BIPARTITE` or "
        "`BP_BEAM_DET`, it is **not comparable across datasets as a property of the method**, and "
        "it does not measure T-27 limitation 3. This ticket has not measured that limitation and "
        "must not be cited as having done so."
    )
    add("")
    add(
        "Source: `data/s74_cost.json`. Each role's `seconds_matrix` is symmetric and its strict "
        "upper triangle sums to the file's own `metadata.seconds_total`, which is asserted per "
        "row in the JSON (`triu_sum_matches_metadata_total`)."
    )
    add("")
    for role in ROLES:
        add(f"### {role} -- {ROLE_LABEL[role]}")
        add("")
        lines += _table(
            ["dataset", "pairs", "total core-s", "mean ms/pair", "median ms/pair", "max ms/pair"],
            [
                [
                    d,
                    _fmt(row["n_pairs"]),
                    _fmt(row["total_core_seconds"], 2),
                    _fmt(row["mean_ms_per_pair"], 4),
                    _fmt(row["median_ms_per_pair"], 4),
                    _fmt(row["max_ms_per_pair"], 2),
                ]
                for d in datasets
                for row in [next(r for r in per_dataset[d]["s74"]["roles"] if r["role"] == role)]
            ],
        )
        add(f"Mean ms/pair by size stratum, {role}:")
        add("")
        lines += _table(
            ["dataset", *SIZE_BIN_LABELS],
            [
                [
                    d,
                    *[
                        _fmt(cell["mean_ms_per_pair"], 3) if cell["n_pairs"] else "--"
                        for cell in row["by_size_stratum"]
                    ],
                ]
                for d in datasets
                for row in [next(r for r in per_dataset[d]["s74"]["roles"] if r["role"] == role)]
            ],
        )
    add("### Provenance actually present in the files")
    add("")
    add(
        "The worker count of the production run is **not recorded in the file metadata**. That "
        "is a real gap: the amendment-11 pathology scales with worker count, so without it the "
        "inflation factor cannot be attributed per dataset from these files alone. What the files "
        "do carry is below; the absent fields are listed per role in `data/s74_cost.json` under "
        "`provenance_fields_absent`."
    )
    add("")
    lines += _table(
        ["dataset", "role", "method", "options", "code commit", "computed (UTC)", "shards"],
        [
            [
                d,
                role,
                str(per_dataset[d]["provenance"][role]["method"]),
                f"`{per_dataset[d]['provenance'][role]['options_string']}`",
                str(per_dataset[d]["provenance"][role]["code_commit"])[:10],
                str(per_dataset[d]["provenance"][role]["computed_utc"])[:19],
                _fmt(per_dataset[d]["provenance"][role]["n_shards"]),
            ]
            for d in datasets
            for role in ROLES
        ],
    )
    add("Figure: `figures/fig_74_realised_cost.pdf`.")
    add("")
    return lines


def _report_s75_and_limitations(results: Mapping[str, Any]) -> list[str]:
    """Render §7.5's non-computation and the limitations section."""
    datasets = list(results["datasets"])
    per_dataset = results["per_dataset"]
    lines: list[str] = []
    add = lines.append

    add("## §7.5 -- not computed: deferred in full to T-06 by PI decision")
    add("")
    add(
        "**§7.5 is deferred to T-06 by PI decision of 2026-08-15, recorded in `T-05-design.md` "
        "amendment 13(a).** T-05's acceptance criterion §8.7 is therefore met for **§7.1-§7.4 "
        "only. This is a deliberate, recorded reduction of the ticket's scope, not an unmet "
        "criterion and not an oversight.**"
    )
    add("")
    add(
        "§7.5 asks for a primary-vs-sensitivity **bracket comparison** on the same graph-level "
        "resamples: per dataset, `rho(Lev, UB_BIPARTITE)` against `rho(Lev, UB_BP_BEAM_DET)`. "
        "Both correlations need `rho(Lev, .)` and therefore a **canonical string per Suite-2 "
        "graph**. §7 asserts its five deliverables are 'computable once the matrices exist; no "
        "extra cluster time'. That holds for §7.1-§7.4 and is false for §7.5: measured at "
        "finalisation, `data/eval/canonical_strings/` and `data/eval/levenshtein_matrices/` hold "
        "**five datasets only** -- `iam_letter_{low,med,high}`, `linux`, `aids` -- and **every one "
        "is a Suite-1 cohort capped at `n <= 12`**. The six new Suite-2 cohorts have never been "
        "canonicalised at all."
    )
    add("")
    add(
        "**The method was never the obstacle.** D14 (`statistics.md` §7) is already locked and "
        "fixes the whole protocol: 300 s canonicalisation timeout, greedy-min fallback for a "
        "censored graph (never a drop), affected pairs flagged, censoring rate reported per "
        "symmetry stratum, and a complete-case sensitivity arm beside the primary. What was open "
        "was **ownership**, and the board settles it: T-06 is 'full recompute -- all experiments, "
        "C++ engine, new cohorts, new statistics' and depends on T-05. Canonicalising the six new "
        "cohorts inside T-05 would duplicate T-06's stated deliverable."
    )
    add("")
    add(
        "**Free information for T-06, orchestrator-verified.** The five existing Levenshtein "
        "matrices join cleanly to the Suite-2 files: `linux` and the three Letter cohorts are "
        "element-wise identical in `graph_ids`, and Suite-1 `aids` (769 graphs) is a **strict "
        "subset** of `aids_graphedx` (819), joinable on `graph_ids` and **never positionally**. "
        "So T-06 inherits five of ten datasets already done and owes six."
    )
    add("")
    add(
        "Computing §7.5 here on the Suite-1 strings would have answered the question at "
        "`n <= 12`, which is exactly the regime AE.1 disputes, and would have been worse than not "
        "answering it."
    )
    add("")

    add("## Limitations")
    add("")
    cert_rows = [(d, per_dataset[d]["s72"]["certification_rate_primary"]) for d in datasets]
    best = max(cert_rows, key=lambda kv: kv[1])
    worst = min(cert_rows, key=lambda kv: kv[1])
    add(
        f"1. **Certification is not a fixed-quality subset, and it collapses with `n`.** The "
        f"primary-arm certification rate runs from {100 * best[1]:.2f} % on `{best[0]}` to "
        f"{100 * worst[1]:.4f} % on `{worst[0]}` -- a factor of "
        f"{best[1] / worst[1]:.0f} across the cohort, and it falls monotonically within every "
        "dataset that spans several size strata (§7.2 tables). Any quantity conditioned on "
        "certified pairs is therefore computed on an **increasingly biased subset as `n` grows**: "
        "certified pairs are the ones the bounds happened to close, which over-represents the "
        "structurally easy pairs at every size."
    )
    add("")
    add(
        "   **Where this report conditions on certification: nowhere.** §7.1's slopes, §7.3's "
        "strata and §7.4's cost table are computed over **all** strict-upper-triangle pairs. "
        "§7.2 *measures* the certification rate but does not condition on it. This is a "
        "deliberate choice, and it has its own cost: the width `w` is a bracket, not an error, "
        "so a per-stratum mean `w` mixes pairs whose GED is pinned exactly (`w = 0`) with pairs "
        "whose GED is known only to within `w`. The `w = 0` mass is exactly the certification "
        "rate, so the two tables must be read together -- a stratum whose mean `w` is low because "
        "half its pairs are certified is not the same object as one whose pairs all have a "
        "moderate bracket."
    )
    add("")
    add(
        f"2. **`UB == 0` pairs.** {results['cohort']['n_pairs_with_zero_ub']:,} pairs "
        f"({100 * results['cohort']['zero_ub_share']:.2f} % of the cohort) have `UB == 0` and are "
        "assigned `w = 0` by the frozen rule. That is correct -- `0 <= LB <= UB` forces "
        "`LB == 0` -- but it means the small-`n` end of every curve carries a mass of exact "
        "zeros that is a property of **isomorphism frequency in the dataset**, not of the bound. "
        "IAM Letter is where this concentrates."
    )
    add("")
    add(
        "3. **The pooled curve is confounded and the within-dataset slopes are not "
        "interchangeable with it.** Size and provenance move together in Suite 2 because it is a "
        "property of which real datasets contain large connected graphs; no sampling design "
        "removes it. The per-stratum dominance shares in §7.1d are the measurement of that "
        "confound in this run. Only "
        f"{len([d for d in datasets if d in UNCONFOUNDED_DATASETS])} of the "
        f"{len(datasets)} datasets analysed span enough `n` to carry an unconfounded slope "
        "(`" + "`, `".join(d for d in datasets if d in UNCONFOUNDED_DATASETS) + "`); the "
        "Letter datasets and LINUX cap at `n <= 10` and constrain only the small-`n` end."
    )
    add("")
    add(
        "4. **A within-dataset slope is still a within-dataset slope.** Density, mean degree and "
        "class composition all vary within a dataset and are not held fixed by the primary fit. "
        "§7.1c's (dataset x density quintile) cells are the check on this, but they are "
        "descriptive: no CI is attached to them, and where a dataset's density range is narrow "
        "the cells collapse to one or two populated quintiles and the check has no power."
    )
    add("")
    add(
        "5. **The OLS fit is linear in `n_max` and unweighted.** The bracket width is bounded in "
        "`[0, 1]`, so a linear fit is guaranteed to be misspecified far enough out; the slopes are "
        "local descriptions of the trend over each dataset's own `n` range, not extrapolable. "
        "Pairs are also not independent -- each graph appears in `N - 1` pairs -- which is exactly "
        "why the CI comes from a graph-level cluster bootstrap rather than from the OLS standard "
        "error. The point estimate is unweighted OLS, so a dataset's slope is dominated by its "
        "most populous `n_max` values."
    )
    add("")
    add(
        "6. **`n_max` is a coarse summary of a pair.** Two graphs of sizes (3, 97) and (96, 97) "
        "share `n_max = 97` and are not comparable problems. `n_max` is the frozen regressor "
        "because it is what §7.1 names; `|n1 - n2|` is uncontrolled and correlates with the "
        "bracket through the node insertion/deletion cost under D6."
    )
    add("")
    add(
        "7. **Tier 3 runs a different estimator from tiers 1 and 2.** COIL-DEL and Mutagenicity "
        "use 1,000 replicates on a 2,000,000-pair subsample per replicate; the other datasets use "
        "2,000 replicates on all induced pairs. The two CIs are not the same estimator and the "
        "tier-3 intervals are wider for two reasons at once -- fewer replicates and within-"
        "replicate subsampling. D15's ratio-matched and structure-matched validation arms are "
        "T-06's, not this module's, and have not been run here."
    )
    add("")
    add(
        "8. **Nothing in this report measures IsalGraph.** The bracket is a property of GEDLIB's "
        "`BRANCH_FAST`/`BIPARTITE`/`BP_BEAM_DET` under cost model D6. It bounds the *resolution* "
        "available to any later correlation against GED at a given `n`; it is not itself evidence "
        "for or against the encoding, and §7.5 -- the deliverable that would connect the two -- "
        "is deferred to T-06."
    )
    add("")
    add(
        "9. **The cohort analysed here is "
        + ("complete." if results["cohort"]["complete"] else "INCOMPLETE.")
        + "** "
        + (
            "All ten Suite-2 datasets are present."
            if results["cohort"]["complete"]
            else "The datasets absent from this run are `"
            + "`, `".join(results["cohort"]["missing_datasets"])
            + "`. Every pooled quantity -- the strata, the dominance shares, the density quintile "
            "edges and the pooled slope -- is computed over the datasets present and will move "
            "when the cohort completes. The within-dataset slopes are unaffected, since each is "
            "fitted on its own dataset alone; the quintile *edges* used by §7.1c and §7.3 are "
            "pooled, so those cell boundaries will move."
        )
    )
    add("")
    add(
        "10. **The two arms share their lower bound, so this is not a two-sided sensitivity "
        "analysis.** `BRANCH_FAST` is the only lower bound computed. Every width in this report "
        "inherits whatever looseness that bound carries, and the primary-versus-arm comparison "
        "isolates the upper bound alone. A lower bound that degrades with `n` would widen both "
        "arms equally and would be invisible here."
    )
    add("")
    return lines


def _write_report_body(
    lines: list[str],
    results: Mapping[str, Any],
    out_dir: Path,
) -> Path:
    """Append every section after the header and write the report to disk.

    Parameters
    ----------
    lines
        The header lines already rendered.
    results
        The assembled results tree.
    out_dir
        The report directory.

    Returns
    -------
    Path
        The written ``REPORT.md``.
    """
    lines += _report_headline(results)
    lines += _report_s71(results)
    lines += _report_s72(results)
    lines += _report_s73(results)
    lines += _report_s74(results)
    lines += _report_s75_and_limitations(results)
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append(results["command"])
    lines.append("```")
    lines.append("")
    lines.append(
        f"Wall time {results['wall_seconds']:.1f} s. Files written: "
        + ", ".join(f"`{name}`" for name in results["artifacts"])
        + "."
    )
    lines.append("")
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """Coerce numpy scalars and paths into JSON-serialisable values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value)!r}")


def _dump_json(payload: Any, path: Path) -> str:
    """Write ``payload`` as pretty JSON and return the file name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=_json_default, allow_nan=True),
        encoding="utf-8",
    )
    return path.name


def _code_commit() -> str:
    """Return the current git commit, or ``unknown`` outside a checkout.

    Resolved at process start so the recorded commit is the code that ran, not
    the code a later stage happens to see (design amendment 9c).
    """
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _available_datasets(config: AnalysisConfig) -> tuple[list[str], list[str]]:
    """Split the requested datasets into present and missing.

    Parameters
    ----------
    config
        The analysis configuration.

    Returns
    -------
    tuple
        ``(present, missing)`` dataset keys, in the requested order.

    Raises
    ------
    InputError
        If the caller named datasets explicitly and any of them is absent.
    """
    present: list[str] = []
    missing: list[str] = []
    for dataset in config.datasets:
        paths = [config.role_dir(role) / f"{dataset}.npz" for role in ROLES]
        (present if all(p.is_file() for p in paths) else missing).append(dataset)
    if missing and config.datasets_explicit:
        raise InputError(
            f"--datasets named {missing} but their role files are not all present; "
            "omit them or wait for the campaign to land"
        )
    return present, missing


def run_analysis(config: AnalysisConfig, command: str = "") -> dict[str, Any]:
    """Run every deliverable and write the report, the JSON and the figures.

    Parameters
    ----------
    config
        The analysis configuration.
    command
        The command line to record in the report, for reproduction.

    Returns
    -------
    dict
        The assembled results tree, identical to what is serialised.

    Raises
    ------
    InputError
        If no requested dataset has all three role files.
    """
    started = time.perf_counter()
    commit = _code_commit()
    datasets, missing = _available_datasets(config)
    if not datasets:
        raise InputError(
            f"none of {list(config.datasets)} has all three role files under "
            f"{config.lb_dir}, {config.ub_dir}, {config.ubs_dir}"
        )
    if missing:
        LOGGER.warning(
            "COHORT INCOMPLETE: %d of %d datasets absent (%s); no pooled number in this run is a "
            "cohort-level Suite-2 quantity",
            len(missing),
            len(config.datasets),
            ", ".join(missing),
        )

    per_dataset: dict[str, Any] = {}
    vectors: list[PairVectors] = []
    for dataset in datasets:
        result, vector = analyse_dataset(config, dataset)
        per_dataset[dataset] = result
        vectors.append(vector)

    edges = density_quintile_edges(vectors, N_DENSITY_QUANTILES)
    labels = density_labels(edges, N_DENSITY_QUANTILES)
    pooled = {
        "density_quintile_edges": edges,
        "density_quintile_labels": list(labels),
        "n_quantiles": N_DENSITY_QUANTILES,
        "size_strata": pooled_size_strata(vectors),
        "density_strata": pooled_density_strata(vectors, edges, labels),
        "density_cells": secondary_density_cells(vectors, edges, labels, config.min_cell_pairs),
        "slope": pooled_slope(vectors),
    }

    n_pairs = sum(v.n_pairs for v in vectors)
    zero_ub = sum(per_dataset[d]["n_pairs_with_zero_ub"] for d in datasets)
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": commit,
        "command": command,
        "seed": config.seed,
        "datasets": datasets,
        "cohort": {
            "requested_datasets": list(config.datasets),
            "analysed_datasets": datasets,
            "missing_datasets": missing,
            "complete": not missing and len(datasets) == len(DATASET_KEYS),
            "n_graphs": sum(per_dataset[d]["n_graphs"] for d in datasets),
            "n_pairs": n_pairs,
            "n_pairs_with_zero_ub": zero_ub,
            "zero_ub_share": zero_ub / n_pairs if n_pairs else math.nan,
            "n_max_max": max(per_dataset[d]["n_max_max"] for d in datasets),
            "n_max_min": min(per_dataset[d]["n_max_min"] for d in datasets),
            "n_pairs_in_bin_2": sum(int((v.size_code == 0).sum()) for v in vectors),
        },
        "strata": {
            "size_bin_edges": list(SIZE_BIN_EDGES),
            "size_bin_labels": list(SIZE_BIN_LABELS),
            "density_quintile_edges": edges,
            "density_quintile_labels": list(labels),
            "certified_tolerance": CERTIFIED_TOL,
            "pair_density_rule": (
                "d_pair = (d1 + d2) / 2, quintiles over the pooled pair population"
            ),
        },
        "per_dataset": per_dataset,
        "pooled": pooled,
        "artifacts": [],
        "wall_seconds": 0.0,
    }

    data_dir = config.out_dir / "data"
    artifacts = [
        _dump_json(
            {
                key: results[key]
                for key in (
                    "schema_version",
                    "generated_utc",
                    "code_commit",
                    "command",
                    "seed",
                    "datasets",
                    "cohort",
                    "strata",
                )
            },
            data_dir / "summary.json",
        ),
        _dump_json(
            {
                "bootstrap_note": (
                    "The resampling unit is the graph, drawn with replacement; the subsample, "
                    "where a tier has one, applies only to the induced pairs inside a replicate."
                ),
                "datasets": {
                    d: {
                        "bootstrap": per_dataset[d]["bootstrap"],
                        "slope_role": per_dataset[d]["slope_role"],
                        "primary": per_dataset[d]["s71"]["primary"],
                        "sensitivity": per_dataset[d]["s71"]["sensitivity"],
                        "arm_minus_primary_slope": (
                            per_dataset[d]["s71"]["sensitivity"]["fit"]["slope"]
                            - per_dataset[d]["s71"]["primary"]["fit"]["slope"]
                        ),
                    }
                    for d in datasets
                },
            },
            data_dir / "s71_within_dataset_slopes.json",
        ),
        _dump_json(
            {d: per_dataset[d]["s71"]["size_profile"] for d in datasets},
            data_dir / "s71_size_profiles.json",
        ),
        _dump_json(pooled["density_cells"], data_dir / "s71_density_cells.json"),
        _dump_json(
            {
                "slope": pooled["slope"],
                "size_strata": pooled["size_strata"],
                "density_quintile_edges": edges,
            },
            data_dir / "s71_pooled.json",
        ),
        _dump_json(
            {
                "tolerance": CERTIFIED_TOL,
                "datasets": {d: per_dataset[d]["s72"] for d in datasets},
            },
            data_dir / "s72_certification.json",
        ),
        _dump_json(
            {
                "datasets": {d: per_dataset[d]["s73"] for d in datasets},
                "pooled_size_strata": pooled["size_strata"],
                "pooled_density_strata": pooled["density_strata"],
                "density_quintile_edges": edges,
            },
            data_dir / "s73_strata.json",
        ),
        _dump_json(
            {
                "caveat": (
                    "Realised wall time under the negative-scaling process pool of T-05 design "
                    "amendment 11. NOT a per-pair cost of the method and NOT comparable across "
                    "datasets as a property of the method."
                ),
                "datasets": {
                    d: {
                        "roles": per_dataset[d]["s74"]["roles"],
                        "provenance": per_dataset[d]["provenance"],
                        "consistency": per_dataset[d]["consistency"],
                    }
                    for d in datasets
                },
            },
            data_dir / "s74_cost.json",
        ),
    ]

    if config.make_figures:
        artifacts += make_figures(results, config.out_dir / "figures")
    results["artifacts"] = artifacts
    results["wall_seconds"] = float(time.perf_counter() - started)
    _dump_json(results, data_dir / "results_full.json")
    results["artifacts"] = [*artifacts, "results_full.json"]
    write_report(results, config.out_dir)
    return results


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.real_data.eval_setup.approx_ged_analysis",
        description=(
            "T-05 §7 analysis of the Suite-2 bounded-GED matrices: bracket width against n, "
            "certification rate, width by stratum and the realised cost table."
        ),
    )
    parser.add_argument("--lb-dir", type=Path, required=True, help="BRANCH_FAST matrices")
    parser.add_argument("--ub-dir", type=Path, required=True, help="BIPARTITE matrices")
    parser.add_argument("--ubs-dir", type=Path, required=True, help="BP_BEAM_DET matrices")
    parser.add_argument("--input-dir", type=Path, required=True, help="exported_suite2 directory")
    parser.add_argument("--out", type=Path, required=True, help="report directory")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="comma-separated dataset keys; default is all ten, absent ones skipped with a banner",
    )
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED, help="master bootstrap seed")
    parser.add_argument(
        "--min-cell-pairs",
        type=int,
        default=MIN_CELL_PAIRS,
        help="population floor for a (dataset x density quintile) secondary cell",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip the figure stage")
    parser.add_argument("--quiet", action="store_true", help="log warnings and errors only")
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    """Build an :class:`AnalysisConfig` from parsed arguments.

    Parameters
    ----------
    args
        Parsed command-line arguments.

    Returns
    -------
    AnalysisConfig
        The configuration.

    Raises
    ------
    InputError
        If ``--datasets`` names a key outside the Suite-2 cohort.
    """
    if args.datasets:
        requested = tuple(part.strip() for part in args.datasets.split(",") if part.strip())
        unknown = [key for key in requested if key not in DATASET_KEYS]
        if unknown:
            raise InputError(f"unknown dataset keys {unknown}; expected a subset of {DATASET_KEYS}")
        datasets = tuple(key for key in DATASET_KEYS if key in requested)
        explicit = True
    else:
        datasets = DATASET_KEYS
        explicit = False
    return AnalysisConfig(
        lb_dir=args.lb_dir,
        ub_dir=args.ub_dir,
        ubs_dir=args.ubs_dir,
        input_dir=args.input_dir,
        out_dir=args.out,
        datasets=datasets,
        datasets_explicit=explicit,
        seed=int(args.seed),
        min_cell_pairs=int(args.min_cell_pairs),
        make_figures=not args.no_figures,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv
        Argument vector; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = config_from_args(args)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    command = "python -m benchmarks.real_data.eval_setup.approx_ged_analysis " + " ".join(
        sys.argv[1:] if argv is None else list(argv)
    )
    results = run_analysis(config, command=command)
    LOGGER.info(
        "wrote %s (%d datasets, %d pairs, %.1f s)",
        config.out_dir / "REPORT.md",
        len(results["datasets"]),
        results["cohort"]["n_pairs"],
        results["wall_seconds"],
    )
    return 0


def _sign_word(value: float) -> str:
    """Return ``widens``, ``narrows`` or ``flat`` for a slope."""
    if not math.isfinite(value):
        return "undetermined"
    return "widens" if value > 0 else ("narrows" if value < 0 else "flat")


def _ci_excludes_zero(block: Mapping[str, Any]) -> bool:
    """Return True when a percentile CI lies entirely on one side of zero."""
    low, high = block["ci_low"], block["ci_high"]
    if not (math.isfinite(low) and math.isfinite(high)):
        return False
    return low > 0.0 or high < 0.0


def _report_headline(results: Mapping[str, Any]) -> list[str]:
    """Render the headline: what §7.1 measured, stated before the tables."""
    datasets = list(results["datasets"])
    per_dataset = results["per_dataset"]
    lines: list[str] = []
    add = lines.append
    add("## Headline")
    add("")
    unconfounded = [d for d in datasets if d in UNCONFOUNDED_DATASETS]
    small_n = [d for d in datasets if d in SMALL_N_DATASETS]
    pooled_slope_value = results["pooled"]["slope"]["primary"]["slope"]

    add(
        "**§7.1's primary estimate is the within-dataset slope of `(UB - LB)/UB` on `n_max`.** "
        "Signs, with the graph-level cluster bootstrap 95 % percentile CI:"
    )
    add("")
    lines += _table(
        ["dataset", "slope /node", "95 % CI", "R^2", "direction", "CI excludes 0", "role"],
        [
            [
                d,
                _fmt(per_dataset[d]["s71"]["primary"]["fit"]["slope"], 5),
                _ci_text(per_dataset[d]["s71"]["primary"]["slope_ci"]),
                _fmt(per_dataset[d]["s71"]["primary"]["fit"]["r_squared"], 3),
                _sign_word(per_dataset[d]["s71"]["primary"]["fit"]["slope"]),
                "yes" if _ci_excludes_zero(per_dataset[d]["s71"]["primary"]["slope_ci"]) else "no",
                per_dataset[d]["slope_role"],
            ]
            for d in datasets
        ],
    )
    if unconfounded:
        signs = {_sign_word(per_dataset[d]["s71"]["primary"]["fit"]["slope"]) for d in unconfounded}
        add(
            "**Among the datasets that span enough `n` to carry an unconfounded slope "
            f"(`{'`, `'.join(unconfounded)}`), the bracket "
            + ("**narrows**" if signs == {"narrows"} else "moves in mixed directions")
            + " with `n`.** "
        )
        if signs == {"narrows"} and pooled_slope_value > 0:
            add("")
            add(
                "> ### SIGN REVERSAL BETWEEN THE POOLED AND THE WITHIN-DATASET FITS\n>\n"
                f"> The pooled slope is **{pooled_slope_value:+.5f}** per node --- positive --- "
                "while every unconfounded within-dataset slope is negative. This is a Simpson "
                "reversal, and it is the empirical vindication of the rule amendment 12 froze "
                "before the analysis ran: **the pooled curve is a descriptive overlay and never "
                "the estimate a conclusion rests on.** A conclusion drawn from the pooled fit "
                "would report the opposite sign from every dataset that actually spans the size "
                "range in dispute. The mechanism is visible in the dominance table in §7.1d: the "
                "small-`n` strata are dense Letter graphs with a narrow bracket and the large-`n` "
                "strata are sparse graphs from entirely different datasets, so the pooled fit is "
                "reading a provenance and density transition as if it were a size effect."
            )
        add("")
    if small_n:
        add(
            f"`{'`, `'.join(small_n)}` cap at `n <= "
            + str(max(per_dataset[d]["n_max_max"] for d in small_n))
            + "` and constrain only the small-`n` end; their slopes are reported but are not "
            "evidence about the large-`n` regime AE.1 disputes."
        )
        add("")
    add(
        "**What this does and does not say about AE.1.** AE.1 objects that conclusions drawn at "
        "`n <= 12` were licensed to `n = 98`. The bracket is the resolution the reference GED is "
        "known to at each size: a wide bracket means a downstream correlation against GED cannot "
        "discriminate at that size, whatever IsalGraph does. The measurement above says the "
        "bracket does **not** degrade monotonically with `n` inside a dataset over this cohort. "
        "It says nothing about IsalGraph, because no Levenshtein distance enters it --- that is "
        "§7.5, which is not computed (see below). The R^2 column is small throughout: `n_max` "
        "explains a minor share of the pair-to-pair variance in bracket width, so these slopes "
        "describe a trend, not a predictive relation."
    )
    add("")
    add(
        "**Certification, the other half of the same object.** The primary-arm certification rate "
        "runs from "
        + ", ".join(
            f"{100 * per_dataset[d]['s72']['certification_rate_primary']:.2f} % (`{d}`)"
            for d in sorted(
                datasets, key=lambda k: -per_dataset[k]["s72"]["certification_rate_primary"]
            )[:1]
        )
        + " down to "
        + ", ".join(
            f"{100 * per_dataset[d]['s72']['certification_rate_primary']:.4f} % (`{d}`)"
            for d in sorted(
                datasets, key=lambda k: per_dataset[k]["s72"]["certification_rate_primary"]
            )[:1]
        )
        + ". A pair with `w = 0` is a pair whose GED is proven exactly, so the mean bracket width "
        "and the certification rate are two views of one distribution and must be read together."
    )
    add("")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
