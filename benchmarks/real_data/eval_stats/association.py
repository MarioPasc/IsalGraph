"""D1 / D3 / D4 --- association measures, Mantel inference and the MRM.

D1 makes Spearman rho the primary association measure with Kendall tau-b beside
it as a tie-robustness check. D3 takes every p-value from the Mantel
permutation test, which permutes graph labels jointly on rows and columns and
therefore respects the dyadic dependence R3.5c identified. D4 promotes the
multiple regression on distance matrices to a confirmatory analysis, because it
is the analysis that can refute the paper's central result: if the standardised
partial coefficient on Levenshtein collapses once ``|delta n|`` is in the model,
the reported correlation was largely size agreement and Claim B must be
restated.

Spearman requires **re-ranking inside every replicate** (D15). Ranks computed
once on the full matrix cannot be reused, because the induced pair multiset
differs per replicate. The cost is contained by factorising each variable once
into dense integer codes --- ranks are invariant under a strictly monotone map
--- which puts every per-replicate ranking on the counting-sort path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy import stats

from benchmarks.real_data.eval_correlation.correlation_metrics import MantelResult, mantel_test
from benchmarks.real_data.eval_setup.ged_bakeoff_analysis import (
    factorize,
    midranks,
    spearman_from_ranks,
)
from benchmarks.real_data.eval_stats.resampling import (
    CI_LEVEL,
    SEED,
    BoolArray,
    BootstrapTier,
    FloatArray,
    IntArray,
    PercentileInterval,
    bootstrap_p_value,
    cluster_bootstrap,
    difference_samples,
    percentile_interval,
)

LOGGER = logging.getLogger(__name__)

Method: TypeAlias = str

__all__ = [
    "AssociationError",
    "AssociationResult",
    "CorrelationSpec",
    "DifferenceResult",
    "DifferenceSpec",
    "MrmResult",
    "PairVariables",
    "bootstrap_associations",
    "condensed",
    "delta_density_matrix",
    "delta_n_matrix",
    "kendall_tau_b",
    "mantel",
    "mrm",
    "partial_mantel",
    "spearman",
]


class AssociationError(Exception):
    """Raised when an association request is inconsistent with its inputs."""


# ---------------------------------------------------------------------------
# Condensed pair vectors
# ---------------------------------------------------------------------------


def condensed(matrix: npt.NDArray[Any]) -> FloatArray:
    """Return the strict upper triangle of a square matrix as a flat vector.

    Args:
        matrix: A square ``(G, G)`` matrix.

    Returns:
        Its ``G (G - 1) / 2`` strict-upper-triangular entries in canonical
        order, as float64.

    Raises:
        AssociationError: If the matrix is not square.
    """
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise AssociationError(f"expected a square matrix, got shape {array.shape}")
    idx = np.triu_indices(array.shape[0], k=1)
    return np.asarray(array[idx], dtype=np.float64)


def delta_n_matrix(node_counts: npt.NDArray[Any]) -> FloatArray:
    """Return the ``|n_i - n_j|`` matrix, D4's first control."""
    counts = np.asarray(node_counts, dtype=np.float64)
    return np.abs(counts[:, None] - counts[None, :])


def graph_density(node_counts: npt.NDArray[Any], edge_counts: npt.NDArray[Any]) -> FloatArray:
    """Return ``2 m / (n (n - 1))`` per graph, with ``n < 2`` mapped to zero."""
    n = np.asarray(node_counts, dtype=np.float64)
    m = np.asarray(edge_counts, dtype=np.float64)
    denom = n * (n - 1.0)
    return np.divide(2.0 * m, denom, out=np.zeros_like(n), where=denom > 0.0)


def delta_density_matrix(
    node_counts: npt.NDArray[Any],
    edge_counts: npt.NDArray[Any],
) -> FloatArray:
    """Return the ``|density_i - density_j|`` matrix, D4's second control."""
    density = graph_density(node_counts, edge_counts)
    return np.abs(density[:, None] - density[None, :])


@dataclass(frozen=True)
class PairVariables:
    """Condensed pair vectors for one dataset, plus the mask of usable pairs.

    Every vector has the full ``G (G - 1) / 2`` length so that a replicate's
    flat pair indices can index it directly. Unusable pairs stay in place and
    are excluded through *valid*, never by compaction --- compaction would
    break the correspondence between a pair index and a graph pair, on which
    the graph-level resample depends.

    Attributes:
        n_graphs: Number of graphs.
        valid: Length ``G (G - 1) / 2`` mask; ``False`` where any variable is
            unusable for that pair.
        values: Raw condensed vectors, keyed by variable name.
        codes: Order-preserving dense integer codes of the same vectors.
    """

    n_graphs: int
    valid: BoolArray
    values: dict[str, FloatArray]
    codes: dict[str, IntArray] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        """Number of pairs passing *valid*."""
        return int(np.count_nonzero(self.valid))

    @classmethod
    def from_matrices(
        cls,
        matrices: Mapping[str, npt.NDArray[Any]],
        *,
        defined: Mapping[str, npt.NDArray[Any]] | None = None,
        require_non_negative: bool = True,
    ) -> PairVariables:
        """Build condensed vectors and the shared validity mask.

        Censored entries in the T-05 matrices carry ``inf``, never ``nan``, and
        a legitimately zero GED is common --- 28.05 % of IAM Letter LOW pairs
        are certified exact at 0 --- so the mask rejects non-finite and
        negative values and **never** rejects zero.

        Args:
            matrices: Named square ``(G, G)`` matrices. All must share a shape.
            defined: Optional per-name boolean ``(G, G)`` masks, as the T-06
                distance schema's ``defined_mask``.
            require_non_negative: Reject negative entries. Distances and GED
                are non-negative; a signed variable would set this ``False``.

        Returns:
            The condensed variables.

        Raises:
            AssociationError: If *matrices* is empty or shapes disagree.
        """
        if not matrices:
            raise AssociationError("at least one matrix is required")
        shapes = {np.asarray(m).shape for m in matrices.values()}
        if len(shapes) != 1:
            raise AssociationError(f"matrices disagree on shape: {sorted(shapes)}")
        n_graphs = int(next(iter(shapes))[0])

        values: dict[str, FloatArray] = {}
        valid = np.ones(n_graphs * (n_graphs - 1) // 2, dtype=bool)
        for name, matrix in matrices.items():
            vector = condensed(matrix)
            values[name] = vector
            ok = np.isfinite(vector)
            if require_non_negative:
                ok &= vector >= 0.0
            valid &= ok
        for name, mask in (defined or {}).items():
            valid &= condensed(np.asarray(mask, dtype=np.float64)) > 0.5
            LOGGER.debug("applied defined_mask for %s", name)

        codes = {name: factorize(vector) for name, vector in values.items()}
        return cls(n_graphs=n_graphs, valid=valid, values=values, codes=codes)

    def require(self, *names: str) -> None:
        """Raise if any named variable is absent.

        Args:
            *names: Variable names.

        Raises:
            AssociationError: If a name is missing.
        """
        missing = [name for name in names if name not in self.values]
        if missing:
            raise AssociationError(f"variables {missing} absent; have {sorted(self.values)}")


# ---------------------------------------------------------------------------
# D1 --- point estimates
# ---------------------------------------------------------------------------


def spearman(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
    """Return Spearman rho, ties handled as midranks.

    Args:
        x: First variable.
        y: Second variable.

    Returns:
        Spearman rho, or ``nan`` when either variable is constant.
    """
    return spearman_from_ranks(midranks(np.asarray(x)), midranks(np.asarray(y)))


def kendall_tau_b(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
    """Return Kendall tau-b, D1's tie-robustness check.

    Tau-b is reported as a point estimate beside rho. It is **not** bootstrapped
    by default: its ``O(p log p)`` cost carries a far larger constant than a
    counting-sort ranking, and D1 assigns it the role of a check rather than of
    the primary measure.

    Args:
        x: First variable.
        y: Second variable.

    Returns:
        Kendall tau-b, or ``nan`` when either variable is constant.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    return float(stats.kendalltau(a, b, variant="b").statistic)


# ---------------------------------------------------------------------------
# D2 + D7 over correlations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationSpec:
    """One correlation to estimate on every replicate.

    Attributes:
        name: Key under which the statistic is reported.
        x: Variable name in :class:`PairVariables`.
        y: Variable name in :class:`PairVariables`.
    """

    name: str
    x: str
    y: str


@dataclass(frozen=True)
class DifferenceSpec:
    """A D7 paired difference of two correlations on the same resample.

    Attributes:
        name: Key under which the difference is reported.
        left: Name of a :class:`CorrelationSpec`.
        right: Name of a :class:`CorrelationSpec`.
    """

    name: str
    left: str
    right: str


@dataclass(frozen=True)
class AssociationResult:
    """A bootstrapped correlation.

    Attributes:
        name: The spec name.
        rho: Full-sample Spearman rho with its percentile interval.
        tau_b: Full-sample Kendall tau-b, ``nan`` when not requested.
        n_pairs: Pairs behind the full-sample estimate.
    """

    name: str
    rho: PercentileInterval
    tau_b: float
    n_pairs: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "name": self.name,
            "spearman": self.rho.as_dict(),
            "kendall_tau_b": self.tau_b,
            "n_pairs": self.n_pairs,
        }


@dataclass(frozen=True)
class DifferenceResult:
    """A D7 paired difference of correlations.

    Attributes:
        name: The spec name.
        left: Name of the minuend correlation.
        right: Name of the subtrahend correlation.
        interval: Percentile interval of the difference.
        p_value: Two-sided bootstrap p-value against a zero difference.
    """

    name: str
    left: str
    right: str
    interval: PercentileInterval
    p_value: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "name": self.name,
            "left": self.left,
            "right": self.right,
            "difference": self.interval.as_dict(),
            "p_value": self.p_value,
        }


def _statistic_factory(
    variables: PairVariables,
    specs: Sequence[CorrelationSpec],
) -> Any:
    """Build the per-replicate Spearman evaluator.

    Args:
        variables: The condensed variables.
        specs: The correlations to evaluate.

    Returns:
        A callable from flat pair indices to a mapping of spec name to rho.
    """
    codes = variables.codes

    def statistic(flat: IntArray) -> dict[str, float]:
        ranks: dict[str, FloatArray] = {}
        needed = {name for spec in specs for name in (spec.x, spec.y)}
        for name in needed:
            ranks[name] = midranks(codes[name][flat])
        return {spec.name: spearman_from_ranks(ranks[spec.x], ranks[spec.y]) for spec in specs}

    return statistic


def bootstrap_associations(
    variables: PairVariables,
    specs: Sequence[CorrelationSpec],
    tier: BootstrapTier,
    *,
    differences: Sequence[DifferenceSpec] = (),
    seed: int = SEED,
    replicates: int | None = None,
    level: float = CI_LEVEL,
    kendall: bool = True,
) -> tuple[dict[str, AssociationResult], dict[str, DifferenceResult]]:
    """Estimate correlations and their D7 differences on one set of resamples.

    Args:
        variables: Condensed pair variables for the dataset.
        specs: Correlations to estimate.
        tier: The frozen D15 effort.
        differences: D7 paired differences over *specs*.
        seed: Master seed; 42 in production.
        replicates: Overrides ``tier.replicates``; tests use it.
        level: Interval coverage. D9's FCR adjustment supplies a lower level.
        kendall: Compute the full-sample Kendall tau-b beside each rho.

    Returns:
        The correlations and the differences, keyed by spec name.

    Raises:
        AssociationError: If a spec names an absent variable or an absent
            correlation.
    """
    for spec in specs:
        variables.require(spec.x, spec.y)
    known = {spec.name for spec in specs}
    for diff in differences:
        if diff.left not in known or diff.right not in known:
            raise AssociationError(f"difference {diff.name!r} names an undeclared correlation")

    full = np.flatnonzero(variables.valid).astype(np.int64)
    statistic = _statistic_factory(variables, specs)
    point = statistic(full)
    samples = cluster_bootstrap(
        variables.n_graphs,
        statistic,
        tier,
        valid=variables.valid,
        seed=seed,
        replicates=replicates,
    )

    results: dict[str, AssociationResult] = {}
    for spec in specs:
        tau = float("nan")
        if kendall:
            tau = kendall_tau_b(
                variables.values[spec.x][full],
                variables.values[spec.y][full],
            )
        results[spec.name] = AssociationResult(
            name=spec.name,
            rho=percentile_interval(samples[spec.name], point[spec.name], level),
            tau_b=tau,
            n_pairs=int(full.size),
        )

    diff_results: dict[str, DifferenceResult] = {}
    for diff in differences:
        delta = difference_samples(samples, diff.left, diff.right)
        observed = point[diff.left] - point[diff.right]
        diff_results[diff.name] = DifferenceResult(
            name=diff.name,
            left=diff.left,
            right=diff.right,
            interval=percentile_interval(delta, observed, level),
            p_value=bootstrap_p_value(delta),
        )
    return results, diff_results


# ---------------------------------------------------------------------------
# D3 --- Mantel
# ---------------------------------------------------------------------------


def mantel(
    d1: npt.NDArray[Any],
    d2: npt.NDArray[Any],
    *,
    method: Method = "spearman",
    n_permutations: int = 9999,
    seed: int = SEED,
) -> MantelResult:
    """Run the Mantel permutation test, D3's source of every p-value.

    A thin wrapper over the repository's existing
    ``correlation_metrics.mantel_test``, which already permutes graph labels
    jointly on rows and columns and is therefore the correct null for dyadic
    data. It is kept rather than rewritten; only the reporting is new (E10:
    the function existed and had never been reported).

    Args:
        d1: First square distance matrix.
        d2: Second square distance matrix.
        method: ``"spearman"`` or ``"pearson"``.
        n_permutations: Permutation count; the D15 tier supplies it.
        seed: Master seed; 42 in production.

    Returns:
        The observed correlation and its permutation p-value.
    """
    return mantel_test(
        np.asarray(d1, dtype=np.float64),
        np.asarray(d2, dtype=np.float64),
        method=method,
        n_permutations=n_permutations,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# D4 --- multiple regression on distance matrices
# ---------------------------------------------------------------------------


def _standardise(matrix: FloatArray) -> FloatArray:
    """Z-score each column, leaving a constant column at zero."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, ddof=0, keepdims=True)
    return np.divide(centred, scale, out=np.zeros_like(centred), where=scale > 0.0)


def _standardised_betas(design: FloatArray, response: FloatArray) -> FloatArray:
    """Return standardised OLS coefficients, one per design column."""
    x = _standardise(design)
    y = _standardise(response.reshape(-1, 1)).ravel()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return np.asarray(beta, dtype=np.float64)


@dataclass(frozen=True)
class MrmResult:
    """D4's multiple regression on distance matrices.

    ``GED ~ beta1 * Lev + beta2 * |delta n| + beta3 * |delta density|`` with
    every variable standardised, so the coefficients are directly comparable.

    Interpretation is fixed in advance (``statistics.md`` section 6): beta1
    remaining large means the association is structural and Claim B stands as
    stated; beta1 collapsing means the correlation was largely size agreement
    and **Claim B must be restated**.

    Attributes:
        predictors: Predictor names in coefficient order.
        betas: Standardised coefficients.
        beta1: The coefficient on the first predictor, by convention Levenshtein.
        beta1_interval: Graph-level bootstrap percentile interval on ``beta1``.
        beta1_permutation_p: Two-sided permutation p-value for ``beta1 = 0``.
        r_squared: Coefficient of determination of the standardised fit.
        n_pairs: Pairs behind the fit.
        n_permutations: Permutations behind the p-value.
    """

    predictors: tuple[str, ...]
    betas: tuple[float, ...]
    beta1: float
    beta1_interval: PercentileInterval
    beta1_permutation_p: float
    r_squared: float
    n_pairs: int
    n_permutations: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "predictors": list(self.predictors),
            "standardised_betas": list(self.betas),
            "beta1": self.beta1,
            "beta1_interval": self.beta1_interval.as_dict(),
            "beta1_permutation_p": self.beta1_permutation_p,
            "r_squared": self.r_squared,
            "n_pairs": self.n_pairs,
            "n_permutations": self.n_permutations,
        }


def mrm(
    response: npt.NDArray[Any],
    predictors: Mapping[str, npt.NDArray[Any]],
    tier: BootstrapTier,
    *,
    seed: int = SEED,
    replicates: int | None = None,
    n_permutations: int | None = None,
    level: float = CI_LEVEL,
) -> MrmResult:
    """Fit D4's multiple regression on distance matrices.

    Inference is doubled deliberately. The **permutation** p-value is what D4
    specifies: graph labels of the response matrix are permuted jointly on rows
    and columns (Legendre, Lapointe & Casgrain, *Evolution* 48(5):1487-1499,
    1994), which is the Mantel null carried into a regression. The **interval**
    is the D2 graph-level bootstrap, because D2 states that all uncertainty
    comes from it and a permutation distribution is a null, not an interval.

    Args:
        response: Square ``(G, G)`` response matrix, by convention GED.
        predictors: Ordered mapping of predictor name to square matrix. The
            first entry carries ``beta1`` and is by convention Levenshtein.
        tier: The frozen D15 effort.
        seed: Master seed; 42 in production.
        replicates: Overrides ``tier.replicates``; tests use it.
        n_permutations: Overrides ``tier.permutations``; tests use it.
        level: Interval coverage.

    Returns:
        The standardised fit with both forms of inference.

    Raises:
        AssociationError: If fewer than one predictor is given.
    """
    names = tuple(predictors)
    if not names:
        raise AssociationError("the MRM needs at least one predictor")

    matrices: dict[str, npt.NDArray[Any]] = {"__response__": response}
    matrices.update(predictors)
    variables = PairVariables.from_matrices(matrices)
    full = np.flatnonzero(variables.valid).astype(np.int64)

    def design_of(flat: IntArray) -> FloatArray:
        return np.column_stack([variables.values[name][flat] for name in names])

    betas = _standardised_betas(design_of(full), variables.values["__response__"][full])
    fitted = _standardise(design_of(full)) @ betas
    target = _standardise(variables.values["__response__"][full].reshape(-1, 1)).ravel()
    residual = target - fitted
    total = float(target @ target)
    r_squared = float(1.0 - (residual @ residual) / total) if total > 0.0 else float("nan")

    def statistic(flat: IntArray) -> dict[str, float]:
        if flat.size <= len(names) + 1:
            return {"beta1": float("nan")}
        fit = _standardised_betas(design_of(flat), variables.values["__response__"][flat])
        return {"beta1": float(fit[0])}

    samples = cluster_bootstrap(
        variables.n_graphs,
        statistic,
        tier,
        valid=variables.valid,
        seed=seed,
        replicates=replicates,
    )
    interval = percentile_interval(samples["beta1"], float(betas[0]), level)

    permutations = int(tier.permutations if n_permutations is None else n_permutations)
    p_value = _mrm_permutation_p(
        response=np.asarray(response, dtype=np.float64),
        variables=variables,
        names=names,
        observed=float(betas[0]),
        n_permutations=permutations,
        seed=seed,
    )
    return MrmResult(
        predictors=names,
        betas=tuple(float(b) for b in betas),
        beta1=float(betas[0]),
        beta1_interval=interval,
        beta1_permutation_p=p_value,
        r_squared=r_squared,
        n_pairs=int(full.size),
        n_permutations=permutations,
    )


def _mrm_permutation_p(
    *,
    response: FloatArray,
    variables: PairVariables,
    names: tuple[str, ...],
    observed: float,
    n_permutations: int,
    seed: int,
) -> float:
    """Return the two-sided permutation p-value for ``beta1``.

    Graph labels of the response matrix are permuted jointly on rows and
    columns; the validity mask stays pinned to pair positions, matching what
    ``mantel_test`` does.
    """
    if n_permutations <= 0:
        return float("nan")
    n = int(response.shape[0])
    triu_i, triu_j = np.triu_indices(n, k=1)
    valid = variables.valid
    design = np.column_stack([variables.values[name][valid] for name in names])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        sigma = rng.permutation(n)
        permuted = response[sigma[triu_i], sigma[triu_j]][valid]
        if not np.isfinite(permuted).all():
            continue
        beta = _standardised_betas(design, permuted)
        extreme += int(abs(float(beta[0])) >= abs(observed))
    return float((extreme + 1) / (n_permutations + 1))


def partial_mantel(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    control: npt.NDArray[Any],
    *,
    n_permutations: int = 9999,
    seed: int = SEED,
) -> dict[str, Any]:
    """Partial Mantel of *x* and *y* controlling for *control*.

    D4 asks for this beside the MRM because it is the form reviewers recognise
    (Smouse, Long & Sokal, *Systematic Zoology* 35(4):627-632, 1986). Vectors
    are rank-transformed first, so the partial correlation is the Spearman
    flavour D1 makes primary.

    Args:
        x: First square matrix, by convention Levenshtein.
        y: Second square matrix, by convention GED.
        control: The square matrix partialled out, by convention ``|delta n|``.
        n_permutations: Permutations of *x*'s graph labels.
        seed: Master seed; 42 in production.

    Returns:
        ``r_partial``, ``p_value``, ``n_permutations`` and ``n_pairs``.
    """
    variables = PairVariables.from_matrices({"x": x, "y": y, "z": control})
    valid = variables.valid
    n = variables.n_graphs
    triu_i, triu_j = np.triu_indices(n, k=1)
    ry = midranks(variables.values["y"][valid])
    rz = midranks(variables.values["z"][valid])
    x_full = np.asarray(x, dtype=np.float64)

    observed = _partial_r(midranks(variables.values["x"][valid]), ry, rz)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        sigma = rng.permutation(n)
        permuted = x_full[sigma[triu_i], sigma[triu_j]][valid]
        candidate = _partial_r(midranks(permuted), ry, rz)
        extreme += int(abs(candidate) >= abs(observed))
    return {
        "r_partial": observed,
        "p_value": float((extreme + 1) / (n_permutations + 1)),
        "n_permutations": int(n_permutations),
        "n_pairs": int(np.count_nonzero(valid)),
    }


def _partial_r(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
    """First-order partial correlation of *a* and *b* given *c*."""
    r_ab = _pearson(a, b)
    r_ac = _pearson(a, c)
    r_bc = _pearson(b, c)
    denom = float(np.sqrt(max(0.0, (1.0 - r_ac**2) * (1.0 - r_bc**2))))
    if denom <= 0.0:
        return float("nan")
    return float((r_ab - r_ac * r_bc) / denom)


def _pearson(a: FloatArray, b: FloatArray) -> float:
    """Pearson correlation, ``nan`` when either input is constant."""
    da = a - a.mean()
    db = b - b.mean()
    denom = float(np.sqrt(float(da @ da) * float(db @ db)))
    if denom <= 0.0:
        return float("nan")
    return float((da @ db) / denom)
