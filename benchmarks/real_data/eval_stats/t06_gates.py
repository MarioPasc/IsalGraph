"""Run F0 (calibration) and F1 (bracket), the two pre-registered gates.

F0, per Suite-1 dataset: ``rho(Lev, GED_exact) - rho(Lev, GED_approx)``.
F1, per Suite-2 dataset: ``rho(Lev, GED_LB) - rho(Lev, GED_UB)``.

Both fire on the pre-declared rule --- the FCR-adjusted interval excludes 0
**and** ``|point| > 0.05`` --- and both are evaluated on **one shared graph-level
resample per dataset** (D7), which is what makes the paired difference correct
by construction rather than by a matching step afterwards.

**F0's ``GED_approx`` is ambiguous in the pre-registration.** Section 2 names one
approximation and specifies five tests, but the large-``n`` regime reports a
*bracket* with two bounds and refuses to interpolate a midpoint. Rather than
invent a primary, this module computes the gate against **both** bounds and
takes the **conservative** one --- the larger ``|point|``, i.e. whichever bound
makes the approximation look *less* like a validated stand-in. The family stays
at five tests, and both readings are recorded so the choice is auditable. Flagged
to the PI as a pre-registration ambiguity resolved conservatively, not as a
measurement decision.

Suite-1 ``aids`` is 769 graphs and has no approximate matrix of its own; it is a
strict subset of ``aids_graphedx``'s 819. The subset is taken **by graph_ids**
(F-12), never positionally.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from benchmarks.real_data.eval_stats.association import (
    CorrelationSpec,
    PairVariables,
    _statistic_factory,
)
from benchmarks.real_data.eval_stats.family import GateInput, run_f0, run_f1
from benchmarks.real_data.eval_stats.resampling import (
    SEED,
    bootstrap_tier,
    cluster_bootstrap,
    difference_samples,
)

LOGGER: Final = logging.getLogger(__name__)

REFERENCE_ARM: Final[str] = "isalgraph_pruned"

#: Suite-1 datasets whose approximate matrices live under a different key.
APPROX_ALIAS: Final[dict[str, str]] = {"aids": "aids_graphedx"}


class GateDriverError(Exception):
    """Raised when a gate input cannot be assembled."""


@dataclass
class DatasetMatrices:
    """Aligned matrices for one dataset.

    Attributes:
        suite: Suite key.
        dataset: Dataset key.
        graph_ids: Cohort ids, in the distance matrix's order.
        matrices: Named square matrices aligned to *graph_ids*.
        defined: The reference arm's ``defined_mask``.
    """

    suite: str
    dataset: str
    graph_ids: npt.NDArray[Any]
    matrices: dict[str, npt.NDArray[Any]]
    defined: npt.NDArray[Any]


def _subset_on_ids(
    matrix: npt.NDArray[Any], source_ids: npt.NDArray[Any], target_ids: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """Reorder and subset *matrix* onto *target_ids*.

    Args:
        matrix: Square matrix indexed by *source_ids*.
        source_ids: Ids of *matrix*, in its own order.
        target_ids: Ids wanted, in the wanted order.

    Returns:
        The square submatrix.

    Raises:
        GateDriverError: If any target id is absent from *source_ids*.
    """
    position = {gid: j for j, gid in enumerate(source_ids)}
    missing = [g for g in target_ids if g not in position]
    if missing:
        raise GateDriverError(
            f"{len(missing)} graph_ids absent from the reference, e.g. {missing[:3]}"
        )
    perm = np.array([position[g] for g in target_ids])
    return matrix[np.ix_(perm, perm)]


def load_matrices(
    suite: str, dataset: str, distances: Path, ged_root: Path, approx_root: Path
) -> DatasetMatrices | None:
    """Assemble every matrix one gate needs, aligned on the arm's graph_ids.

    Args:
        suite: Suite key.
        dataset: Dataset key.
        distances: The ``distances/`` tree.
        ged_root: Suite-1 exact matrices.
        approx_root: ``APPROX_GED`` root.

    Returns:
        The aligned matrices, or ``None`` when an input is missing.
    """
    arm = distances / suite / f"{dataset}__{REFERENCE_ARM}__levenshtein.npz"
    if not arm.exists():
        LOGGER.warning("%s/%s: no reference arm", suite, dataset)
        return None
    with np.load(arm, allow_pickle=True) as z:
        ids = np.asarray(z["graph_ids"]).astype(str)
        matrices: dict[str, npt.NDArray[Any]] = {
            "lev": np.asarray(z["distance_matrix"], dtype=np.float64)
        }
        defined = np.asarray(z["defined_mask"], dtype=bool)

    approx_path = approx_root / "LB" / f"{APPROX_ALIAS.get(dataset, dataset)}.npz"
    if approx_path.exists():
        with np.load(approx_path, allow_pickle=True) as z:
            src = np.asarray(z["graph_ids"]).astype(str)
            matrices["lb"] = _subset_on_ids(np.asarray(z["lb_matrix"], dtype=np.float64), src, ids)
            matrices["ub"] = _subset_on_ids(np.asarray(z["ub_matrix"], dtype=np.float64), src, ids)

    if suite == "suite1":
        exact_path = ged_root / f"{dataset}.npz"
        if not exact_path.exists():
            LOGGER.warning("%s/%s: no exact GED", suite, dataset)
            return None
        with np.load(exact_path, allow_pickle=True) as z:
            src = np.asarray(z["graph_ids"]).astype(str)
            matrices["exact"] = _subset_on_ids(
                np.asarray(z["ged_matrix"], dtype=np.float64), src, ids
            )

    return DatasetMatrices(
        suite=suite, dataset=dataset, graph_ids=ids, matrices=matrices, defined=defined
    )


def _paired_difference(
    data: DatasetMatrices, left: tuple[str, str], right: tuple[str, str]
) -> tuple[float, npt.NDArray[Any], int]:
    """Bootstrap one paired rho difference on a shared graph-level resample.

    Mirrors what :func:`association.bootstrap_associations` does internally, but
    returns the raw replicate array, which :class:`family.GateInput` requires and
    that function does not expose.

    Args:
        data: The dataset's aligned matrices.
        left: ``(x, y)`` variable names for the left correlation.
        right: ``(x, y)`` variable names for the right correlation.

    Returns:
        ``(point, samples, n_pairs)``.
    """
    names = sorted({*left, *right})
    variables = PairVariables.from_matrices(
        {n: data.matrices[n] for n in names}, defined={"lev": data.defined}
    )
    specs = (
        CorrelationSpec("left", left[0], left[1]),
        CorrelationSpec("right", right[0], right[1]),
    )
    statistic = _statistic_factory(variables, specs)
    full = np.flatnonzero(variables.valid).astype(np.int64)
    point_map = statistic(full)
    tier = bootstrap_tier(data.dataset, data.suite)
    samples = cluster_bootstrap(
        variables.n_graphs, statistic, tier, valid=variables.valid, seed=SEED
    )
    delta = difference_samples(samples, "left", "right")
    return float(point_map["left"] - point_map["right"]), delta, int(full.size)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=Path, required=True)
    ap.add_argument("--ged-root", type=Path, required=True)
    ap.add_argument("--approx-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap


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

    from benchmarks.real_data.eval_stats.family import SUITE1, SUITE2

    # ---- F0, on both bounds, conservative reading primary --------------------
    f0_inputs: list[GateInput] = []
    f0_detail: list[dict[str, Any]] = []
    for dataset in SUITE1:
        data = load_matrices("suite1", dataset, args.distances, args.ged_root, args.approx_root)
        if data is None or "exact" not in data.matrices:
            continue
        per_bound: dict[str, tuple[float, npt.NDArray[Any], int]] = {}
        for bound in ("lb", "ub"):
            if bound not in data.matrices:
                continue
            per_bound[bound] = _paired_difference(data, ("lev", "exact"), ("lev", bound))
        if not per_bound:
            LOGGER.warning("suite1/%s: no approximate bound, F0 skipped", dataset)
            continue
        worst = max(per_bound, key=lambda b: abs(per_bound[b][0]))
        point, samples, n_pairs = per_bound[worst]
        f0_inputs.append(GateInput(dataset=dataset, point=point, samples=samples))
        f0_detail.append(
            {
                "dataset": dataset,
                "conservative_bound": worst,
                "n_pairs": n_pairs,
                "points": {b: per_bound[b][0] for b in per_bound},
            }
        )
        LOGGER.info(
            "F0 %-16s point=%+.4f (worst of %s) n_pairs=%d",
            dataset,
            point,
            {b: round(per_bound[b][0], 4) for b in per_bound},
            n_pairs,
        )

    f0 = run_f0(f0_inputs)
    (args.out_dir / "family_F0.json").write_text(
        json.dumps(
            {
                "family": "F0",
                "note": f0.note,
                "ci_level": f0.ci_level,
                "failing_datasets": list(f0.failing_datasets),
                "ambiguity_resolution": (
                    "preregistration section 2 names one GED_approx but the large-n regime "
                    "reports a two-bound bracket; the gate is computed against both and the "
                    "larger |point| is taken, which makes the approximation look LESS like a "
                    "validated stand-in. Conservative. Both readings recorded."
                ),
                "per_dataset": f0_detail,
                "outcomes": [o.as_dict() for o in f0.outcomes],
            },
            indent=2,
            default=str,
        )
    )

    # ---- F1, the bracket gate; its failures are d ----------------------------
    f1_inputs: list[GateInput] = []
    for dataset in SUITE2:
        data = load_matrices("suite2", dataset, args.distances, args.ged_root, args.approx_root)
        if data is None or "lb" not in data.matrices:
            continue
        point, samples, n_pairs = _paired_difference(data, ("lev", "lb"), ("lev", "ub"))
        f1_inputs.append(GateInput(dataset=dataset, point=point, samples=samples))
        LOGGER.info("F1 %-16s point=%+.4f n_pairs=%d", dataset, point, n_pairs)

    f1 = run_f1(f1_inputs)
    (args.out_dir / "family_F1.json").write_text(
        json.dumps(
            {
                "family": "F1",
                "note": f1.note,
                "ci_level": f1.ci_level,
                "failing_datasets": list(f1.failing_datasets),
                "d": len(f1.failing_datasets),
                "outcomes": [o.as_dict() for o in f1.outcomes],
            },
            indent=2,
            default=str,
        )
    )

    print(
        f"\nF0: {len(f0.failing_datasets)} of {len(f0.outcomes)} datasets fire "
        f"-> {sorted(f0.failing_datasets)}"
    )
    print(f"    {f0.note}")
    print(
        f"F1: d = {len(f1.failing_datasets)} of {len(f1.outcomes)} -> {sorted(f1.failing_datasets)}"
    )
    print(f"    {f1.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
