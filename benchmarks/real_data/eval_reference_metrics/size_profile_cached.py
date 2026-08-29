"""Equal-``n`` size profile computed from the CACHED distance matrices.

Same output schema as :mod:`benchmarks.real_data.eval_size_profile.size_profile`,
same statistics -- ``MIN_PAIRS``, ``N_BOOTSTRAP``, ``SEED`` and the graph-level
``_bootstrap_ci`` are imported from it rather than restated, so the two cannot
drift -- but the representation distances are **read** from the T-06 cache
instead of being recomputed from the encodings.

That is the whole premise of T-28. The encoding-driven path re-derives every
Levenshtein block per stratum and measured 2 of ~120 (dataset, representation)
units in fifteen minutes on this workstation; the matrices it was recomputing
were already on disk. Reading them makes the same table in minutes.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy import stats

from benchmarks.real_data.eval_size_profile.size_profile import (
    MIN_PAIRS,
    N_BOOTSTRAP,
    PRIMARY_DISTANCE,
    SEED,
    StratumRow,
    _bootstrap_ci,
)

LOGGER: Final = logging.getLogger(__name__)

SUITES: Final[dict[str, tuple[str, ...]]] = {
    "suite1": ("aids", "iam_letter_low", "iam_letter_med", "iam_letter_high", "linux"),
    "suite2": (
        "aids_graphedx",
        "aids_iam",
        "coil_del",
        "grec",
        "iam_letter_low",
        "iam_letter_med",
        "iam_letter_high",
        "linux",
        "mutagenicity",
        "protein",
    ),
}


def _align(
    matrix: npt.NDArray[Any], src: npt.NDArray[Any], target: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """Reorder a square matrix from *src* id order onto *target* id order."""
    pos = {gid: i for i, gid in enumerate(src)}
    index = np.array([pos[g] for g in target], dtype=np.intp)
    return matrix[np.ix_(index, index)]


def _load_square(path: Path, field: str, target: npt.NDArray[Any] | None) -> Any:
    """Load one square matrix and its ids, optionally aligned onto *target*."""
    with np.load(path, allow_pickle=True) as z:
        ids = np.asarray(z["graph_ids"]).astype(str)
        matrix = np.asarray(z[field], dtype=np.float64)
        mask = np.asarray(z["defined_mask"], dtype=bool) if "defined_mask" in z else None
        counts = np.asarray(z["node_counts"], dtype=np.int64) if "node_counts" in z else None
    if target is not None and list(ids) != list(target):
        matrix = _align(matrix, ids, target)
        if mask is not None:
            mask = _align(mask, ids, target)
        ids = target
    return matrix, ids, mask, counts


def _references(
    archive: Path, suite: str, dataset: str, ids: npt.NDArray[Any], t28_root: Path | None
) -> dict[str, npt.NDArray[Any]]:
    """Return every reference matrix for one cell, aligned onto *ids*."""
    out: dict[str, npt.NDArray[Any]] = {}
    if suite == "suite1":
        path = archive / "data/eval/ged_matrices" / f"{dataset}.npz"
        if path.exists():
            out["exact"] = _load_square(path, "ged_matrix", ids)[0]
    else:
        path = archive / "data/source/APPROX_GED/LB" / f"{dataset}.npz"
        if path.exists():
            for key, field in (("lb", "lb_matrix"), ("ub", "ub_matrix")):
                with np.load(path, allow_pickle=True) as z:
                    if field not in z:
                        continue
                out[key] = _load_square(path, field, ids)[0]
    if t28_root is not None:
        for path in sorted((t28_root / suite).glob(f"{dataset}__*.npz")):
            out[path.stem.split("__", 1)[1]] = _load_square(path, "distance_matrix", ids)[0]
    return out


def profile_cell(
    archive: Path,
    suite: str,
    dataset: str,
    t28_root: Path | None,
    keep: frozenset[str] | None = None,
    bootstrap: bool = True,
) -> list[StratumRow]:
    """Return one row per (representation, reference, n) for one cell.

    Args:
        archive: Artifact archive root.
        suite: Suite key.
        dataset: Dataset key.
        t28_root: T-28 reference tree, or None.
        keep: Restrict to these references. The graph-level bootstrap runs
            ``N_BOOTSTRAP`` replicates per (representation, reference, stratum),
            so every extra reference is a full multiple of the cost; a figure
            that needs one reference should ask for one.
        bootstrap: Compute the per-stratum interval. **The figures do not read
            it** --- ``figures.aggregate`` derives its own interval from the
            Fisher-z weighted mean of ``rho`` and ``n_graphs`` and never looks
            at ``ci_lo``/``ci_hi`` --- so a figure-only run should pass False
            and skip what dominates the runtime. Leave it True for any table
            that quotes a per-stratum interval.
    """
    dist_dir = archive / "data/source/T06/distances" / suite
    arm_path = dist_dir / f"{dataset}__isalgraph_pruned__levenshtein.npz"
    if not arm_path.exists():
        LOGGER.warning("%s/%s: no arm matrix, skipping", suite, dataset)
        return []
    _, ids, _, node_counts = _load_square(arm_path, "distance_matrix", None)
    references = _references(archive, suite, dataset, ids, t28_root)
    if keep is not None:
        references = {k: v for k, v in references.items() if k in keep}
    if not references:
        LOGGER.warning("%s/%s: no references, skipping", suite, dataset)
        return []

    rows: list[StratumRow] = []
    sizes = sorted({int(n) for n in node_counts})
    for representation, metric in sorted(PRIMARY_DISTANCE.items()):
        if metric is None:
            continue
        path = dist_dir / f"{dataset}__{representation}__{metric}.npz"
        if not path.exists():
            continue
        distance, _, defined, _ = _load_square(path, "distance_matrix", ids)
        if defined is None:
            defined = np.ones_like(distance, dtype=bool)
        for n in sizes:
            take = np.flatnonzero(node_counts == n)
            if take.size < 3:
                continue
            block = distance[np.ix_(take, take)]
            valid_block = defined[np.ix_(take, take)] & np.isfinite(block)
            for name, reference in sorted(references.items()):
                ref_block = reference[np.ix_(take, take)]
                valid = valid_block & np.isfinite(ref_block)
                upper = np.triu(np.ones_like(valid), k=1).astype(bool) & valid
                n_pairs = int(upper.sum())
                if n_pairs < MIN_PAIRS:
                    continue
                x, y = block[upper], ref_block[upper]
                if np.all(x == x[0]) or np.all(y == y[0]):
                    continue
                result = stats.spearmanr(x, y)
                if bootstrap:
                    rng = np.random.default_rng(SEED + n)
                    ci_lo, ci_hi = _bootstrap_ci(block, ref_block, valid, rng)
                else:
                    ci_lo, ci_hi = None, None
                rows.append(
                    StratumRow(
                        suite=suite,
                        dataset=dataset,
                        representation=representation,
                        metric=metric,
                        reference=name,
                        n=int(n),
                        n_graphs=int(take.size),
                        n_pairs=n_pairs,
                        rho=float(result.statistic),
                        ci_lo=ci_lo,
                        ci_hi=ci_hi,
                        p_value=float(result.pvalue),
                        mean_distance=float(x.mean()),
                        mean_reference=float(y.mean()),
                        zero_fraction=float((x == 0.0).mean()),
                    )
                )
        LOGGER.info("%s/%-16s %-20s done", suite, dataset, representation)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--archive", type=Path, default=Path("/home/mpascual/research/data/isalgraph_archive")
    )
    ap.add_argument("--t28-root", type=Path, default=None, help="T-28 reference tree")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", default="all")
    ap.add_argument(
        "--references",
        default="",
        help="comma-separated reference filter; empty means every reference found",
    )
    ap.add_argument(
        "--no-bootstrap",
        action="store_true",
        help=(
            "skip the per-stratum interval. The figures never read it -- they "
            "aggregate their own from rho and n_graphs -- and it dominates runtime"
        ),
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    wanted = None if args.datasets == "all" else set(args.datasets.split(","))
    keep = frozenset(args.references.split(",")) if args.references else None
    rows: list[StratumRow] = []
    for suite, datasets in SUITES.items():
        for dataset in datasets:
            if wanted is not None and dataset not in wanted:
                continue
            rows.extend(
                profile_cell(
                    args.archive, suite, dataset, args.t28_root, keep, not args.no_bootstrap
                )
            )

    payload: dict[str, Any] = {
        "schema_version": "t06.size_profile.2",
        "ticket": "T-28",
        "descriptive": True,
        "note": (
            "Representation distances read from the T-06 cache, never recomputed. "
            "Statistics imported from eval_size_profile.size_profile so the two agree."
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "min_pairs": MIN_PAIRS,
        "n_bootstrap": 0 if args.no_bootstrap else N_BOOTSTRAP,
        "seed": SEED,
        "rows": [vars(r) for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    LOGGER.info("wrote %d rows -> %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
