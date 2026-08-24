"""Claim A: IsalGraph encodes a graph in fewer bits than competing serialisations.

Answers **AE.4a**, which the Area Editor raised under a requirement modal, plus
R3.6a. The input is the encoding campaign's ``.npz`` files; the output is
``claim_a.json``.

Four rules this module enforces, each because the alternative produces a number
that reads as evidence and is not.

**Both bit conventions, always together.** The entropy bound
(``L log2 |Sigma|``, the like-for-like measure of encoding efficiency) and the
realised bytes (what a practitioner stores) can disagree in direction, so
quoting one alone is a choice made after seeing the result.

**Never a mean without dispersion.** Encoding-length distributions are
right-skewed -- a Suite-2 dataset spans ``n = 2`` to ``n = 98`` -- so the median
leads and the mean is reported beside it with the interquartile range and the
standard deviation.

**Paired, on the intersection.** A per-dataset mean difference between two
columns computed over different graph subsets is not a difference. Every
comparison joins on ``graph_ids`` and keeps only the graphs both arms encoded.

**Two arms, both reportable.** The *primary* arm includes the D14 graphs that
entered with their greedy-min string; the *complete-case* arm keeps only
``status == "ok"``. The censored graphs are exactly the ones with the largest
automorphism groups, so the difference between the two arms is the selection
bias D14 exists to expose. ``wl_subtree`` and ``size_null`` appear with an
explicit ``BitCountUndefined`` reason rather than a fabricated zero.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

#: The reference arm, frozen by the orchestrator by measurement.
REFERENCE_REPRESENTATION = "isalgraph_pruned"

#: The six serialisations Claim A compares against, ``CONTRACTS.md`` §2.
CLAIM_A_SERIALISATIONS: tuple[str, ...] = (
    "graph6",
    "sparse6",
    "nauty_graph6",
    "adjacency",
    "agm_cam",
    "min_dfs",
)

#: Representations with no message length. Their cells carry a reason, never a
#: number: a feature-vector or node-count "bit cost" would measure the choice of
#: container rather than the encoding.
UNDEFINED_REPRESENTATIONS: tuple[str, ...] = ("wl_subtree", "size_null")

BIT_CONVENTIONS: tuple[str, ...] = ("entropy_bits", "realised_bits")

#: Statuses whose encoding is usable in the primary arm. ``censored`` is
#: included on purpose: D14 retains those graphs rather than thinning the cohort.
PRIMARY_STATUSES: tuple[str, ...] = ("ok", "censored", "fallback")

ALPHA = 0.05


class ClaimAError(RuntimeError):
    """Raised when the inputs cannot support a Claim A table."""


@dataclass(frozen=True, slots=True)
class Dispersion:
    """A location-and-spread summary of a bit-count distribution.

    Attributes:
        n: Graphs contributing.
        median: The headline statistic; the distributions are right-skewed.
        mean: Reported beside the median, never instead of it.
        std: Sample standard deviation.
        q1: 25th percentile.
        q3: 75th percentile.
        minimum: Smallest value.
        maximum: Largest value.
    """

    n: int
    median: float
    mean: float
    std: float
    q1: float
    q3: float
    minimum: float
    maximum: float


def summarise(values: np.ndarray) -> Dispersion | None:
    """Summarise a bit-count vector, or return ``None`` when it is empty.

    Args:
        values: Finite bit counts.

    Returns:
        The summary, or ``None`` if nothing is finite.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return Dispersion(
        n=int(finite.size),
        median=float(np.median(finite)),
        mean=float(np.mean(finite)),
        std=float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        q1=float(np.percentile(finite, 25)),
        q3=float(np.percentile(finite, 75)),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
    )


def clopper_pearson(successes: int, trials: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact binomial confidence interval for a proportion.

    Clopper & Pearson, *The use of confidence or fiducial limits illustrated in
    the case of the binomial*, Biometrika 26(4):404-413, 1934. Exact rather than
    normal-approximate because several cells sit at or near a proportion of 1,
    where a Wald interval extends past 1 and is not defined.

    Args:
        successes: Number of successes.
        trials: Number of trials.
        alpha: Two-sided error rate.

    Returns:
        ``(lower, upper)``. Degenerate ends are clamped: the lower limit is 0
        at zero successes and the upper limit is 1 at complete success.

    Raises:
        ClaimAError: If *trials* is zero or *successes* is out of range.
    """
    from scipy.stats import beta

    if trials <= 0 or not 0 <= successes <= trials:
        raise ClaimAError(f"cannot form an interval for {successes}/{trials}")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return lower, upper


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """IsalGraph against one competitor on one dataset, one bit convention.

    Attributes:
        representation: The competitor.
        convention: ``entropy_bits`` or ``realised_bits``.
        arm: ``primary`` or ``complete_case``.
        n_pairs: Graphs both arms encoded.
        median_difference: Median of ``competitor - reference``. Positive means
            IsalGraph is shorter.
        mean_difference: Mean of the same, with ``std_difference`` beside it.
        std_difference: Standard deviation of the paired differences.
        n_isalgraph_shorter: Graphs where the reference is strictly shorter.
        fraction_isalgraph_shorter: The same as a proportion.
        ci_lower: Clopper-Pearson lower limit for that proportion.
        ci_upper: Clopper-Pearson upper limit.
        n_ties: Graphs where the two are equal, reported because a tie is
            neither a win nor a loss and pooling it either way is a choice.
    """

    representation: str
    convention: str
    arm: str
    n_pairs: int
    median_difference: float
    mean_difference: float
    std_difference: float
    n_isalgraph_shorter: int
    fraction_isalgraph_shorter: float
    ci_lower: float
    ci_upper: float
    n_ties: int


def _load(path: Path) -> dict[str, np.ndarray]:
    """Read one encodings file into a plain dict."""
    with np.load(path, allow_pickle=False) as handle:
        return {name: handle[name] for name in handle.files}


def _usable(arrays: dict[str, np.ndarray], arm: str) -> np.ndarray:
    """Boolean mask of graphs contributing to *arm*.

    Args:
        arrays: One encodings file.
        arm: ``"primary"`` or ``"complete_case"``.

    Returns:
        The mask.
    """
    status = arrays["status"]
    if arm == "complete_case":
        return status == "ok"
    return np.isin(status, PRIMARY_STATUSES)


def _aligned(
    reference: dict[str, np.ndarray],
    competitor: dict[str, np.ndarray],
    convention: str,
    arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit counts for the graphs both files encoded, joined on ``graph_ids``.

    Args:
        reference: The reference arm's arrays.
        competitor: The competitor's arrays.
        convention: ``entropy_bits`` or ``realised_bits``.
        arm: ``"primary"`` or ``"complete_case"``.

    Returns:
        ``(reference_bits, competitor_bits)``, same length, same graphs.
    """
    shared = np.intersect1d(reference["graph_ids"], competitor["graph_ids"])
    ref_pos = {gid: i for i, gid in enumerate(reference["graph_ids"])}
    com_pos = {gid: i for i, gid in enumerate(competitor["graph_ids"])}
    ref_ok, com_ok = _usable(reference, arm), _usable(competitor, arm)
    keep = [
        (ref_pos[gid], com_pos[gid])
        for gid in shared
        if ref_ok[ref_pos[gid]] and com_ok[com_pos[gid]]
    ]
    if not keep:
        return np.empty(0), np.empty(0)
    ref_idx = np.array([i for i, _ in keep])
    com_idx = np.array([j for _, j in keep])
    ref_bits = reference[convention][ref_idx]
    com_bits = competitor[convention][com_idx]
    finite = np.isfinite(ref_bits) & np.isfinite(com_bits)
    return ref_bits[finite], com_bits[finite]


def compare(
    reference: dict[str, np.ndarray],
    competitor: dict[str, np.ndarray],
    representation: str,
    convention: str,
    arm: str,
) -> PairedComparison | None:
    """Paired comparison of IsalGraph against one competitor.

    Args:
        reference: The reference arm's arrays.
        competitor: The competitor's arrays.
        representation: The competitor's name.
        convention: ``entropy_bits`` or ``realised_bits``.
        arm: ``"primary"`` or ``"complete_case"``.

    Returns:
        The comparison, or ``None`` when no graph is shared.
    """
    ref_bits, com_bits = _aligned(reference, competitor, convention, arm)
    if ref_bits.size == 0:
        return None
    difference = com_bits - ref_bits
    shorter = int((ref_bits < com_bits).sum())
    lower, upper = clopper_pearson(shorter, int(ref_bits.size))
    return PairedComparison(
        representation=representation,
        convention=convention,
        arm=arm,
        n_pairs=int(ref_bits.size),
        median_difference=float(np.median(difference)),
        mean_difference=float(np.mean(difference)),
        std_difference=float(np.std(difference, ddof=1)) if difference.size > 1 else 0.0,
        n_isalgraph_shorter=shorter,
        fraction_isalgraph_shorter=shorter / ref_bits.size,
        ci_lower=lower,
        ci_upper=upper,
        n_ties=int((ref_bits == com_bits).sum()),
    )


def _marginals(arrays: dict[str, np.ndarray], representation: str) -> dict[str, Any]:
    """Per-representation location and spread, or the undefined reason.

    Args:
        arrays: One encodings file.
        representation: The backend name.

    Returns:
        A dict keyed by arm and convention, or a ``reason`` entry.
    """
    if representation in UNDEFINED_REPRESENTATIONS:
        return {
            "entropy_bits": None,
            "realised_bits": None,
            "reason": "BitCountUndefined",
            "detail": (
                "no message length: a feature-vector or node-count bit cost would "
                "measure the container, not the encoding"
            ),
        }
    out: dict[str, Any] = {}
    for arm in ("primary", "complete_case"):
        mask = _usable(arrays, arm)
        out[arm] = {
            convention: (
                asdict(summary)
                if (summary := summarise(arrays[convention][mask])) is not None
                else None
            )
            for convention in BIT_CONVENTIONS
        }
    return out


def _dataset_files(encodings_dir: Path, suite: str) -> dict[str, dict[str, Path]]:
    """Index the suite's files as ``{dataset: {representation: path}}``."""
    index: dict[str, dict[str, Path]] = {}
    for path in sorted((encodings_dir / suite).glob("*.npz")):
        dataset, sep, representation = path.stem.partition("__")
        if sep:
            index.setdefault(dataset, {})[representation] = path
    return index


def build_report(encodings_dir: Path, suite: str) -> dict[str, Any]:
    """Assemble ``claim_a.json`` for one suite.

    Args:
        encodings_dir: The ``encodings/`` tree.
        suite: ``"suite1"`` or ``"suite2"``.

    Returns:
        The report payload.

    Raises:
        ClaimAError: If the suite holds no files.
    """
    index = _dataset_files(encodings_dir, suite)
    if not index:
        raise ClaimAError(f"no encodings found under {encodings_dir / suite}")
    datasets = {dataset: _dataset_report(files) for dataset, files in sorted(index.items())}
    return {
        "schema_version": "t06.1",
        "ticket": "T-06",
        "suite": suite,
        "reference_representation": REFERENCE_REPRESENTATION,
        "alpha": ALPHA,
        "ci_method": "Clopper-Pearson exact binomial",
        "arms": {
            "primary": "includes D14 graphs retained with their greedy-min string",
            "complete_case": "status == 'ok' only; the sensitivity arm",
        },
        "datasets": datasets,
    }


def _dataset_report(files: dict[str, Path]) -> dict[str, Any]:
    """Marginals for every representation plus paired comparisons for the six."""
    arrays = {name: _load(path) for name, path in files.items()}
    report: dict[str, Any] = {
        "representations": {
            name: _marginals(values, name) for name, values in sorted(arrays.items())
        },
        "paired": [],
    }
    reference = arrays.get(REFERENCE_REPRESENTATION)
    if reference is None:
        report["paired_skipped"] = f"{REFERENCE_REPRESENTATION} absent for this dataset"
        return report
    report["paired"] = [
        asdict(result)
        for name in CLAIM_A_SERIALISATIONS
        if name in arrays
        for convention in BIT_CONVENTIONS
        for arm in ("primary", "complete_case")
        if (result := compare(reference, arrays[name], name, convention, arm)) is not None
    ]
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate encodings into Claim A.")
    parser.add_argument("--encodings", required=True, type=Path, help="the encodings/ tree")
    parser.add_argument("--suite", required=True, choices=("suite1", "suite2"))
    parser.add_argument("--out", required=True, type=Path, help="claim_a.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    report = build_report(args.encodings, args.suite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    LOGGER.info("wrote %s (%d datasets)", args.out, len(report["datasets"]))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
