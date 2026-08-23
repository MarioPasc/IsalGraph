"""``python -m isalgraph.competitors.admissibility.e4_trap`` -- E4.

**Quantify the trap that F5-blindness exists to prevent.**

``competitors.md`` §3's header asserts, without measuring it, that the raw
adjacency matrix scores rho = 0.75-0.87 against exact GED while failing F3.
If that is true then the representation that looks *best* on correlation with
GED is precisely the one whose distance is **not a well-defined function on
isomorphism classes** -- and that, not a preference, is why the frozen
selection rule of ``competitors.md`` §3.4 is blind to rho (signed decision
24).  This module turns the assertion into a measurement with an interval.

**D-A6 is frozen and applies whichever way this falls.**  A positive
difference whose CI excludes 0 is the headline justification for decision 24.
A difference that is not positive leaves the F5-blindness argument resting on
principle alone, and the paper says so.  Neither outcome was anticipated more
than the other and the code treats them identically.

**E4 never feeds back into the selection.**  ``grid.py`` computed and froze
``primary_distance`` before this module existed and has no import path to a
GED value; that ordering is the whole defence.  This module *consumes* the
frozen selection and adds the three representations the grid excluded --
``adjacency``, ``graph6``, ``sparse6`` -- under ``levenshtein``, which is the
point of the experiment rather than a violation of it.

Three design points that decide whether the number means anything.

**The comparison is paired at the resample.**  Both arms are handed the *same*
:class:`~isalgraph.competitors.bootstrap.ResampleIndex`, and the difference
``rho_A - rho_B`` is formed **inside** each replicate.  Two independently
resampled intervals could be overlaid but not subtracted: their difference
would carry the sum of two variances instead of the variance of a difference,
which on positively correlated arms is far too wide.  The magnitude of that
mistake is reported per comparison as ``ci_unpaired``, computed against a
deliberately different seed, so the reader can see what the pairing bought.

**The paired arms share a pair set.**  A graph one backend refuses to encode
is absent from that arm only; subtracting two rhos computed over different
pair sets would attribute a cohort difference to a representation difference.
The difference is therefore computed over the intersection, and both the
common and the marginal pair counts reach the record.

**The bootstrap p-value cannot be 0.**  It is the achieved significance level
with the observed replicate included on both tails,
``2 * min((#{delta <= 0} + 1), (#{delta >= 0} + 1)) / (R + 1)``, so 2,000
replicates report at best ``p ~ 1e-3`` -- which is what 2,000 replicates can
resolve.  Holm-corrected across the five Suite-1 datasets (D-A5).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import TYPE_CHECKING, Any

from isalgraph.competitors import datasets, f5
from isalgraph.competitors.admissibility import common
from isalgraph.competitors.bootstrap import (
    DEFAULT_ALPHA,
    MIN_PAIRS,
    ResampleIndex,
    _rho,
    make_resample_index,
    spearman,
)

if TYPE_CHECKING:
    import numpy.typing as npt

#: The three representations ``competitors.md`` §3.4 excluded, and the
#: distance E4 reads them under.  They have no *admissible* distance -- that
#: is the finding E1 characterises -- so a distance must be named here rather
#: than taken from the grid, and ``levenshtein`` is the one every other
#: sequence representation in the pool was selected onto.
EXCLUDED_UNDER: dict[str, str] = {
    "adjacency": "levenshtein",
    "graph6": "levenshtein",
    "sparse6": "levenshtein",
}

#: The arm every excluded representation is tested against: the manuscript's
#: own representation, under its selected primary distance.
REFERENCE_ARM = "isalgraph_pruned"

#: Presentational thresholds for the rho-versus-psi reading, declared here so
#: that no cut point is chosen after seeing a number.  They label the cells of
#: protocol §5's table; **the measured rho and psi are the result**, and the
#: label is a reading aid that carries no inference.
PSI_INVARIANT_MAX = 0.05
RHO_HIGH_MIN = 0.50

#: The two views, taken from F5 so the tables are commensurable.
VIEWS: tuple[str, ...] = f5.VIEWS


def representations(grid_path: str) -> tuple[dict[str, str], dict[str, bool]]:
    """Every representation E4 measures, and whether the grid admitted it.

    Args:
        grid_path: path to ``grid_200.json``.

    Returns:
        ``(under, admitted)`` -- the distance each representation is read
        under, and ``True`` for the ones the grid selected a primary distance
        for.  ``size_null`` is excluded: it is a descriptive baseline, not a
        representation, and F5 already reports it.
    """
    primary = common.primary_distances(grid_path)
    under: dict[str, str] = {}
    admitted: dict[str, bool] = {}
    for name, metric in sorted(primary.items()):
        if name == f5.SIZE_NULL or metric is None:
            continue
        under[name] = metric
        admitted[name] = True
    for name, metric_name in EXCLUDED_UNDER.items():
        under[name] = metric_name
        admitted[name] = False
    return under, admitted


# --------------------------------------------------------------------------
# The paired bootstrap
# --------------------------------------------------------------------------


def replicate_rhos(
    x: list[float],
    y: list[float],
    pair_index: list[tuple[int, int]],
    index: ResampleIndex,
    *,
    min_pairs: int = MIN_PAIRS,
) -> npt.NDArray[Any]:
    """Spearman rho **per bootstrap replicate**, aligned to the replicate id.

    :func:`~isalgraph.competitors.bootstrap.graph_bootstrap_ci` returns only
    the finished interval, so it cannot express a paired difference: the two
    arms have to be subtracted replicate by replicate, before any percentile
    is taken.  This returns the vector that makes that possible, and a test
    asserts its percentiles reproduce ``graph_bootstrap_ci`` exactly, so the
    marginal intervals here and in the F5 table are the same statistic.

    ``bootstrap._rho`` is reused rather than reimplemented for the same
    reason: a second rank correlation would be a second definition.

    Args:
        x: the representation's distance, one value per observed pair.
        y: the reference GED, one value per observed pair.
        pair_index: ``(a, b)`` positions into the graph draw, one per pair.
        index: the shared resample matrix.  **The same object must reach both
            arms of a comparison** -- that is what makes it paired.
        min_pairs: a replicate inducing fewer pairs than this yields ``nan``.

    Returns:
        A ``(resamples,)`` array; ``nan`` marks a replicate that induced too
        few pairs or a constant column.  ``nan`` rather than a dropped entry,
        because the alignment between the two arms is what is being preserved.

    Raises:
        ValueError: if the inputs disagree in length.
    """
    import numpy as np

    if not (len(x) == len(y) == len(pair_index)):
        raise ValueError(
            f"x, y and pair_index must agree in length; got {len(x)}, {len(y)}, {len(pair_index)}"
        )

    out = np.full(index.resamples, np.nan, dtype=np.float64)
    if len(pair_index) < min_pairs:
        return out

    n = index.n_graphs
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)

    # The diagonal is never written, so a graph drawn into two slots pairs
    # with itself and resolves to "absent" rather than to a fabricated
    # (distance 0, GED 0) point that was never in the observed set.
    lookup = np.full((n, n), -1, dtype=np.int64)
    for k, (a, b) in enumerate(pair_index):
        lookup[a, b] = k
        lookup[b, a] = k

    upper_i, upper_j = np.triu_indices(n, 1)
    for r in range(index.resamples):
        drawn = index.draws[r]
        selected = lookup[drawn[upper_i], drawn[upper_j]]
        selected = selected[selected >= 0]
        if selected.size < min_pairs:
            continue
        value = _rho(xs[selected], ys[selected])
        if math.isfinite(value):
            out[r] = value
    return out


def percentile_ci(values: npt.NDArray[Any], *, alpha: float = DEFAULT_ALPHA) -> list[float] | None:
    """Two-sided percentile interval over the finite entries of *values*.

    Args:
        values: replicate statistics, possibly with ``nan`` entries.
        alpha: two-sided miss rate.

    Returns:
        ``[low, high]``, or ``None`` when fewer than two replicates are
        finite.  ``None`` is a printed absence; an interval from one replicate
        is not an interval.
    """
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    low, high = np.percentile(finite, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return [float(low), float(high)]


def bootstrap_p(values: npt.NDArray[Any]) -> float | None:
    """Two-sided achieved significance level for ``H0: delta = 0``.

    The observed replicate is counted on both tails, so the value can never be
    exactly 0 -- 2,000 replicates resolve to ``p >= 1/2001``, and reporting
    ``0`` would claim a resolution the resampling does not have.

    Args:
        values: the replicate differences.

    Returns:
        The p-value, or ``None`` when fewer than two replicates are finite.
    """
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    n = int(finite.size)
    below = int((finite <= 0.0).sum())
    above = int((finite >= 0.0).sum())
    return float(min(1.0, 2.0 * min(below + 1, above + 1) / (n + 1)))


# --------------------------------------------------------------------------
# One dataset, one view
# --------------------------------------------------------------------------


def arm_series(
    metric_name: str,
    draw: f5.Draw,
    encoded: dict[int, Any],
    pairs: list[tuple[int, int]],
    geds: list[float],
) -> tuple[list[float], list[float], list[tuple[int, int]], list[tuple[int, int]], int]:
    """One representation's distances over the pairs it can actually score.

    Args:
        metric_name: registry key of the distance.
        draw: the graph draw, for the cohort-index-to-position map.
        encoded: cohort index -> encoding, missing where the backend refused.
        pairs: candidate cohort-index pairs.
        geds: the reference GED, aligned to *pairs*.

    Returns:
        ``(distances, values, positions, kept_pairs, n_undefined)``.
        ``kept_pairs`` carries the cohort indices, which is what lets two arms
        be intersected before they are subtracted.
    """
    from isalgraph.competitors.registry import get_metric

    metric = get_metric(metric_name)
    distances: list[float] = []
    values: list[float] = []
    positions: list[tuple[int, int]] = []
    kept: list[tuple[int, int]] = []
    undefined = 0
    for (a, b), ged in zip(pairs, geds, strict=True):
        if a not in encoded or b not in encoded:
            continue
        if not metric.is_defined(encoded[a], encoded[b]):
            undefined += 1
            continue
        distances.append(float(metric.distance(encoded[a], encoded[b])))
        values.append(ged)
        positions.append((draw.position[a], draw.position[b]))
        kept.append((a, b))
    return distances, values, positions, kept, undefined


def rho_cell(
    distances: list[float],
    values: list[float],
    positions: list[tuple[int, int]],
    index: ResampleIndex,
    metric_name: str,
    n_undefined: int,
) -> dict[str, Any]:
    """Point rho, its p-value and its graph-level interval, or a stated absence.

    Args:
        distances: the representation's distance per pair.
        values: the reference GED per pair.
        positions: draw positions per pair.
        index: the shared resample matrix.
        metric_name: printed into the record.
        n_undefined: pairs the metric declared undefined.

    Returns:
        The cell record.  ``rho`` is ``None`` with a ``reason`` whenever the
        rank correlation has no denominator, which is a printed absence rather
        than a ``NaN`` that a strict JSON reader would reject or a lenient one
        would read as a number.
    """
    if len(distances) < 3:
        return {
            "metric": metric_name,
            "rho": None,
            "p": None,
            "ci": None,
            "n_pairs": len(distances),
            "n_undefined": n_undefined,
            "zero_frac": None,
            "reason": "fewer than three defined pairs",
        }
    if len(set(distances)) < 2 or len(set(values)) < 2:
        side = "the distance" if len(set(distances)) < 2 else "the reference GED"
        return {
            "metric": metric_name,
            "rho": None,
            "p": None,
            "ci": None,
            "n_pairs": len(distances),
            "n_undefined": n_undefined,
            "zero_frac": sum(d == 0.0 for d in distances) / len(distances),
            "reason": f"Spearman undefined: {side} is constant over this view",
        }

    rho, p = spearman(distances, values)
    return {
        "metric": metric_name,
        "rho": float(rho),
        "p": float(p),
        "ci": percentile_ci(replicate_rhos(distances, values, positions, index)),
        "n_pairs": len(distances),
        "n_undefined": n_undefined,
        "zero_frac": sum(d == 0.0 for d in distances) / len(distances),
        "reason": None,
    }


def paired_comparison(
    challenger: str,
    reference: str,
    series: dict[
        str, tuple[list[float], list[float], list[tuple[int, int]], list[tuple[int, int]]]
    ],
    index: ResampleIndex,
    *,
    unpaired_seed: int,
) -> dict[str, Any]:
    """``rho_challenger - rho_reference`` under the **shared** resample index.

    Args:
        challenger: the excluded representation's name.
        reference: the admissible arm's name, normally
            :data:`REFERENCE_ARM`.
        series: name -> ``(distances, geds, positions, cohort pairs)``.
        index: the shared resample matrix.  Both arms get this object.
        unpaired_seed: seed for the deliberately *independent* second index
            used to compute the contrast interval.  It must differ from
            ``index.seed``; that is what makes the contrast a contrast.

    Returns:
        The comparison record, or a stated absence when the two arms share
        fewer than :data:`~isalgraph.competitors.bootstrap.MIN_PAIRS` pairs.

    Raises:
        ValueError: if *unpaired_seed* equals the shared index's seed, which
            would silently turn the contrast into a second paired interval.
    """
    import numpy as np

    if unpaired_seed == index.seed:
        raise ValueError(
            f"the unpaired contrast needs a seed different from the shared "
            f"index's ({index.seed}); an identical seed reproduces the paired "
            f"interval and the contrast would silently measure nothing"
        )

    a_dist, a_ged, a_pos, a_pairs = series[challenger]
    b_dist, b_ged, b_pos, b_pairs = series[reference]

    common_pairs = sorted(set(a_pairs) & set(b_pairs))
    if len(common_pairs) < MIN_PAIRS:
        return {
            "challenger": challenger,
            "reference": reference,
            "n_pairs_common": len(common_pairs),
            "difference": None,
            "ci": None,
            "ci_unpaired": None,
            "p": None,
            "reason": (
                f"the two arms share {len(common_pairs)} pairs, below the "
                f"{MIN_PAIRS}-pair floor; a difference over disjoint pair sets "
                f"would attribute a cohort difference to a representation difference"
            ),
        }

    def restrict(
        dist: list[float],
        ged: list[float],
        pos: list[tuple[int, int]],
        pairs: list[tuple[int, int]],
    ) -> tuple[list[float], list[float], list[tuple[int, int]]]:
        table = {pair: k for k, pair in enumerate(pairs)}
        picked = [table[pair] for pair in common_pairs]
        return [dist[k] for k in picked], [ged[k] for k in picked], [pos[k] for k in picked]

    ad, ag, ap = restrict(a_dist, a_ged, a_pos, a_pairs)
    bd, bg, bp = restrict(b_dist, b_ged, b_pos, b_pairs)
    # The GED column is the same reference read over the same pairs, so a
    # mismatch here means the two arms were not aligned and every subsequent
    # number would be a comparison of different pairs wearing one label.
    if ag != bg or ap != bp:
        raise ValueError(
            f"{challenger} and {reference} disagree on the reference GED or the draw "
            f"positions of their common pairs; the arms are misaligned"
        )

    rho_a, _ = spearman(ad, ag)
    rho_b, _ = spearman(bd, bg)

    reps_a = replicate_rhos(ad, ag, ap, index)
    reps_b = replicate_rhos(bd, bg, bp, index)
    delta = reps_a - reps_b

    loose = make_resample_index(index.n_graphs, resamples=index.resamples, seed=unpaired_seed)
    delta_unpaired = reps_a - replicate_rhos(bd, bg, bp, loose)

    return {
        "challenger": challenger,
        "reference": reference,
        "n_pairs_common": len(common_pairs),
        "n_pairs_challenger": len(a_pairs),
        "n_pairs_reference": len(b_pairs),
        "rho_challenger": float(rho_a),
        "rho_reference": float(rho_b),
        "difference": float(rho_a - rho_b),
        "ci": percentile_ci(delta),
        "ci_unpaired": percentile_ci(delta_unpaired),
        "n_replicates": int(np.isfinite(delta).sum()),
        "p": bootstrap_p(delta),
        "reason": None,
    }


def reading(rho: float | None, psi: float | None) -> str:
    """Protocol §5's joint reading of one (rho, psi) cell.

    Args:
        rho: correlation with exact GED, or ``None`` when undefined.
        psi: the separation ratio from E1, or ``None`` when E1 was not
            supplied.

    Returns:
        The reading, using the thresholds declared in
        :data:`PSI_INVARIANT_MAX` and :data:`RHO_HIGH_MIN`.
    """
    if rho is None:
        return "rho undefined"
    if psi is None:
        return "psi absent (E1 not supplied)"
    invariant = psi <= PSI_INVARIANT_MAX
    high = rho >= RHO_HIGH_MIN
    if high and invariant:
        return "a good, well-defined graph distance"
    if high and not invariant:
        return "THE TRAP: correlates with GED and is not a function on isomorphism classes"
    if invariant:
        return "well-defined and weak"
    return "neither well-defined nor strong"


# --------------------------------------------------------------------------
# The E1 join
# --------------------------------------------------------------------------

#: Leaf keys :func:`load_psi` accepts as the separation ratio.
PSI_KEYS: frozenset[str] = frozenset({"psi", "separation_ratio", "psi_point", "separation"})

#: Keys :func:`load_psi` unwraps when the ratio arrives as a nested record.
POINT_KEYS: tuple[str, ...] = ("point", "estimate", "value", "psi", "mean")


def load_psi(path: str) -> dict[tuple[str, str], float]:
    """Separation ratios from E1's payload, keyed by ``(dataset, backend)``.

    E1 is written by another track and its schema is not frozen, so this reads
    it **structurally**: it walks the payload keeping the path, and records
    any numeric leaf under a key in :data:`PSI_KEYS` whose path also names a
    known dataset and a known backend.  A ratio found outside a dataset scope
    is filed under ``"pooled"``.

    Reading the shape rather than a fixed path is deliberate.  The alternative
    -- hard-coding one path -- fails **silently** against a different nesting,
    producing an empty psi column that reads as "E1 found nothing" rather than
    as "E4 could not parse E1".  Anything this cannot find is reported as
    absent by :func:`reading`, and E4 still reports rho.

    Args:
        path: E1's result JSON.

    Returns:
        ``(dataset, backend) -> psi``.  Empty when the file names none.

    Raises:
        AdmissibilityError: if the file is not readable JSON.  A missing
            ``--e1`` is handled by the caller and is not an error.
    """
    from isalgraph.competitors.registry import available_backends

    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise common.AdmissibilityError(f"cannot read E1 payload at {path}: {exc}") from exc

    known_datasets = set(datasets.ALL_DATASETS)
    known_backends = set(available_backends(include_baseline=True))
    out: dict[tuple[str, str], float] = {}

    def visit(node: object, path_keys: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                text = str(key)
                if text in PSI_KEYS:
                    number = _as_number(value)
                    if number is not None:
                        _file_psi(out, path_keys, number, known_datasets, known_backends)
                        continue
                visit(value, (*path_keys, text))
        elif isinstance(node, list):
            for item in node:
                visit(item, path_keys)

    visit(payload, ())
    return out


def _as_number(value: object) -> float | None:
    """A float from a bare number or from a small record wrapping one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        for key in POINT_KEYS:
            if key in value:
                inner = value[key]
                if isinstance(inner, (int, float)) and not isinstance(inner, bool):
                    return float(inner) if math.isfinite(float(inner)) else None
    return None


def _file_psi(
    out: dict[tuple[str, str], float],
    path_keys: tuple[str, ...],
    value: float,
    known_datasets: set[str],
    known_backends: set[str],
) -> None:
    """Record *value* under the dataset and backend named on *path_keys*."""
    dataset = next((k for k in reversed(path_keys) if k in known_datasets), "pooled")
    backend = next((k for k in reversed(path_keys) if k in known_backends), None)
    if backend is not None:
        out[(dataset, backend)] = value


def psi_for(
    table: dict[tuple[str, str], float], dataset: str, backend: str
) -> tuple[float | None, str]:
    """The separation ratio for one cell, preferring the per-dataset value.

    Args:
        table: :func:`load_psi`'s output.
        dataset: cohort name.
        backend: representation name.

    Returns:
        ``(psi, scope)`` -- the value and where it came from, so a pooled
        fallback is never printed as if it were per-dataset.
    """
    if (dataset, backend) in table:
        return table[(dataset, backend)], dataset
    if ("pooled", backend) in table:
        return table[("pooled", backend)], "pooled"
    return None, "absent"


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run(
    grid_path: str,
    *,
    names: tuple[str, ...] = datasets.SUITE1,
    e1_path: str | None = None,
    n_graphs: int = common.N_PER_DATASET,
    seed: int = common.SEED,
    resamples: int = common.RESAMPLES,
) -> dict[str, Any]:
    """Run E4 over *names*, which must be Suite-1 cohorts.

    Args:
        grid_path: ``grid_200.json``; supplies the frozen primary distances.
        names: datasets.  Exact GED exists for Suite 1 only.
        e1_path: E1's payload, for the psi column.  ``None`` degrades to rho
            alone with psi marked absent.
        n_graphs: per-dataset draw.
        seed: the single seed for the draw and the resample matrix.
        resamples: bootstrap replicates.

    Returns:
        The payload.

    Raises:
        common.AdmissibilityError: if a requested dataset is outside Suite 1,
            where there is no certified exact GED to correlate against.
    """
    outside = [name for name in names if name not in datasets.SUITE1]
    if outside:
        raise common.AdmissibilityError(
            f"E4 correlates against certified exact GED, which exists for Suite 1 only; "
            f"{outside} are outside it"
        )

    under, admitted = representations(grid_path)
    psi_table = load_psi(e1_path) if e1_path else {}

    payload: dict[str, Any] = {
        "protocol_section": "5",
        "grid": os.path.abspath(grid_path),
        "e1": os.path.abspath(e1_path) if e1_path else None,
        "psi_present": bool(psi_table),
        "note": (
            "REPORTED AFTER SELECTION, NEVER FED BACK INTO IT. grid.py froze "
            "primary_distance before this module existed and has no import path to a "
            "GED value; D-A6 fixes what is reported in either direction."
        ),
        "reference_arm": REFERENCE_ARM,
        "excluded_under": dict(EXCLUDED_UNDER),
        "read_under": under,
        "grid_admitted": admitted,
        "thresholds": {"psi_invariant_max": PSI_INVARIANT_MAX, "rho_high_min": RHO_HIGH_MIN},
        "n_graphs": n_graphs,
        "resamples": resamples,
        "results": {},
        "comparisons": {},
    }

    per_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in names:
        record, comparisons = _dataset_record(
            dataset, under, psi_table, n_graphs=n_graphs, seed=seed, resamples=resamples
        )
        payload["results"][dataset] = record
        per_dataset[dataset] = comparisons

    payload["comparisons"] = _holm_across_datasets(per_dataset, names)
    return payload


def _dataset_record(
    dataset: str,
    under: dict[str, str],
    psi_table: dict[tuple[str, str], float],
    *,
    n_graphs: int,
    seed: int,
    resamples: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """One dataset: both views, every representation, every comparison."""
    draw = f5.draw_for(dataset, n_graphs, seed)
    pairs, geds = f5.reference_pairs(draw, "exact")
    index = make_resample_index(len(draw.indices), resamples=resamples, seed=seed)

    encodings: dict[str, dict[int, Any]] = {}
    unencodable: dict[str, int] = {}
    errors: dict[str, dict[str, int]] = {}
    for name in sorted(under):
        encoded, failed, failures = f5.encode_draw(name, draw)
        encodings[name] = encoded
        unencodable[name] = failed
        if failures:
            errors[name] = failures

    views: dict[str, Any] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for view in VIEWS:
        mask = f5._view_mask(draw, pairs, view)
        view_pairs = [pair for pair, keep in zip(pairs, mask, strict=True) if keep]
        view_geds = [ged for ged, keep in zip(geds, mask, strict=True) if keep]

        series: dict[
            str, tuple[list[float], list[float], list[tuple[int, int]], list[tuple[int, int]]]
        ] = {}
        row: dict[str, Any] = {}
        for name in sorted(under):
            dist, values, pos, kept, undefined = arm_series(
                under[name], draw, encodings[name], view_pairs, view_geds
            )
            series[name] = (dist, values, pos, kept)
            cell = rho_cell(dist, values, pos, index, under[name], undefined)
            psi, scope = psi_for(psi_table, dataset, name)
            cell["psi"] = psi
            cell["psi_scope"] = scope
            cell["reading"] = reading(cell["rho"], psi)
            row[name] = cell
        views[view] = row

        for challenger in sorted(EXCLUDED_UNDER):
            if challenger not in series or REFERENCE_ARM not in series:
                continue
            key = f"{challenger}_vs_{REFERENCE_ARM}::{view}"
            comparisons[key] = paired_comparison(
                challenger, REFERENCE_ARM, series, index, unpaired_seed=seed + 1
            )

    record = {
        "dataset": dataset,
        "suite": draw.suite,
        "reference": "exact",
        "n_graphs": len(draw.indices),
        "n_certified_pairs": len(pairs),
        "n_unencodable": unencodable,
        "encode_errors": errors,
        "views": views,
    }
    return record, comparisons


def _holm_across_datasets(
    per_dataset: dict[str, dict[str, dict[str, Any]]], names: tuple[str, ...]
) -> dict[str, Any]:
    """Group the comparisons by ``(challenger, view)`` and Holm-correct within.

    The family is the five Suite-1 datasets for one comparison, per D-A5.
    Correcting across the challengers too would treat three readings of one
    phenomenon as three independent questions.

    Args:
        per_dataset: dataset -> comparison key -> record.
        names: dataset order.

    Returns:
        Comparison key -> ``{"datasets": {...}, "n_tested": int}`` with
        ``p_holm`` written into each dataset's record.
    """
    keys = sorted({key for records in per_dataset.values() for key in records})
    out: dict[str, Any] = {}
    for key in keys:
        rows = {name: per_dataset[name][key] for name in names if key in per_dataset[name]}
        tested = [name for name, row in rows.items() if row.get("p") is not None]
        adjusted = common.holm([float(rows[name]["p"]) for name in tested])
        for name, value in zip(tested, adjusted, strict=True):
            rows[name]["p_holm"] = value
        for row in rows.values():
            row.setdefault("p_holm", None)
        out[key] = {"n_tested": len(tested), "datasets": rows}
    return out


def _print_summary(payload: dict[str, Any]) -> None:
    for dataset, record in payload["results"].items():
        for view, row in record["views"].items():
            print(f"\n=== {dataset} [{view}]  rho vs psi ===")
            for name in sorted(row):
                cell = row[name]
                admitted = "   " if payload["grid_admitted"].get(name) else "EXC"
                if cell["rho"] is None:
                    print(f"  {admitted} {name:22s}    ---   {cell['reason']}")
                    continue
                psi = cell["psi"]
                psi_text = " psi=n/a " if psi is None else f" psi={psi:6.3f}"
                ci = cell["ci"]
                span = "" if ci is None else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
                print(
                    f"  {admitted} {name:22s} rho={cell['rho']:7.4f}{span}{psi_text}"
                    f"  {cell['reading']}"
                )

    for key, block in payload["comparisons"].items():
        print(f"\n=== {key}  (paired graph-level bootstrap, Holm over datasets) ===")
        for dataset, row in block["datasets"].items():
            if row["difference"] is None:
                print(f"  {dataset:18s} ---  {row['reason']}")
                continue
            ci = row["ci"]
            loose = row["ci_unpaired"]
            width = "" if ci is None else f" width={ci[1] - ci[0]:.4f}"
            loose_width = "" if loose is None else f" (unpaired {loose[1] - loose[0]:.4f})"
            holm = row.get("p_holm")
            print(
                f"  {dataset:18s} d={row['difference']:+.4f}"
                f"  CI [{'n/a' if ci is None else f'{ci[0]:+.4f}, {ci[1]:+.4f}'}]"
                f"{width}{loose_width}"
                f"  p={row['p']:.4f} p_holm={'n/a' if holm is None else f'{holm:.4f}'}"
            )


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: command line, or ``None`` for ``sys.argv``.

    Returns:
        ``0``.
    """
    parser = argparse.ArgumentParser(prog="python -m isalgraph.competitors.admissibility.e4_trap")
    parser.add_argument("--grid", required=True, help="grid_200.json from competitors.grid")
    parser.add_argument(
        "--e1",
        default="",
        help="E1's result JSON, for the psi column. Optional: without it E4 reports "
        "rho alone and marks psi absent",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--n-graphs", type=int, default=common.N_PER_DATASET)
    parser.add_argument("--resamples", type=int, default=common.RESAMPLES)
    parser.add_argument(
        "--datasets",
        default="",
        help="comma-separated subset of Suite 1; the default is all five",
    )
    args = parser.parse_args(argv)

    if args.datasets.strip():
        names = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    else:
        names = datasets.SUITE1

    started = time.perf_counter()
    payload = run(
        args.grid,
        names=names,
        e1_path=args.e1.strip() or None,
        n_graphs=args.n_graphs,
        seed=args.seed,
        resamples=args.resamples,
    )
    payload["wall_seconds"] = time.perf_counter() - started

    _print_summary(payload)
    common.write_result(args.out, "E4", payload)
    print(f"\nwrote {args.out} in {payload['wall_seconds']:.1f} s")
    return 0


__all__ = [
    "EXCLUDED_UNDER",
    "PSI_INVARIANT_MAX",
    "REFERENCE_ARM",
    "RHO_HIGH_MIN",
    "VIEWS",
    "arm_series",
    "bootstrap_p",
    "load_psi",
    "main",
    "paired_comparison",
    "percentile_ci",
    "psi_for",
    "reading",
    "replicate_rhos",
    "representations",
    "rho_cell",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
