"""Pair-accounting ladder, per dataset.

Acceptance criterion 9 / A9: emit
``raw -> connected -> GED-available -> GED > 0 -> Lev > 0 -> analysed``
for every dataset, so that the denominator behind every printed rho is visible
rather than implied.

**Per dataset, never pooled.** The rungs do not behave alike across datasets and
pooling hides exactly the differences that matter: ``aids`` and ``linux`` lose
~56 % of their pairs at *GED-available* because exact GED stops being computable
above ~12 nodes, while the three Letter datasets lose none; and the ``GED > 0``
rung is a no-op on those same two while removing 12.7-15.5 % on the Letter
datasets, which carry genuinely isomorphic pairs. A single pooled ladder would
report an average that describes no dataset.

**These are two rung families, not one chain, and reading them as a chain was a
defect in this module (corrected 2026-08-23).** ``raw -> connected ->
GED-available -> analysed`` accounts for the correlation, with ``analysed =
ged_available & defined_mask``. ``GED > 0`` and ``Lev > 0`` exist to define the
**collision denominator**: a collision is a pair with positive GED that the
encoding cannot separate, so it is only definable where GED > 0. They are not
steps towards ``analysed``, and ``analysed`` does not filter GED = 0 --- that is
what isomorphic graphs have, and dropping them would truncate the response at
its most informative end. Before this correction ``analysed`` was computed as
``ged_positive & defined_mask``, which disagreed with the pair count every rho
is actually computed over.

The ``Lev > 0`` rung is measured on the **IsalGraph reference arm**, since that
is the arm whose correlation the ladder is accounting for. A pair with a
zero Levenshtein distance and a positive GED is a *collision* --- two
non-isomorphic graphs the encoding cannot tell apart --- and its count is a
result in its own right, not merely a filter step.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

LOGGER: Final = logging.getLogger(__name__)

#: The arm the ladder accounts for.
REFERENCE_ARM: Final[str] = "isalgraph_pruned"


class LadderError(Exception):
    """Raised when an input needed for a rung is missing."""


@dataclass
class LadderRow:
    """One dataset's pair accounting.

    Attributes:
        suite: Suite key.
        dataset: Dataset key.
        reference: Which GED reference the rungs were computed against.
        n_raw_graphs: Graphs before the cohort filter.
        n_graphs: Graphs after it.
        raw: Pairs implied by ``n_raw_graphs``.
        connected: Pairs implied by ``n_graphs`` --- the cohort.
        ged_available: Pairs with a finite reference GED.
        ged_positive: **Collision denominator**, not a correlation rung. Pairs
            with GED > 0, i.e. the pairs on which a collision is even definable.
        lev_positive: **Collision denominator.** Of those, the pairs the
            reference arm separates.
        analysed: Pairs entering the correlation: ``ged_available`` intersected
            with this arm's ``defined_mask``. GED = 0 pairs are **retained** ---
            that is what isomorphic graphs have, 28.05 % of IAM Letter LOW is
            certified at exactly 0, and dropping them would truncate the
            response at its most informative end.
        collisions: GED > 0 but Levenshtein == 0 --- encoding collisions.
    """

    suite: str
    dataset: str
    reference: str
    n_raw_graphs: int | None
    n_graphs: int
    raw: int | None
    connected: int
    ged_available: int
    ged_positive: int
    lev_positive: int
    analysed: int
    collisions: int

    @property
    def collision_rate(self) -> float:
        """Fraction of GED-positive pairs the reference arm calls identical."""
        return self.collisions / self.ged_positive if self.ged_positive else 0.0


def _load_reference(
    suite: str, dataset: str, ged_root: Path, approx_root: Path
) -> tuple[str, np.ndarray, np.ndarray] | None:
    """Load the reference GED matrix and its ids.

    Args:
        suite: Suite key.
        dataset: Dataset key.
        ged_root: Suite-1 exact matrices.
        approx_root: ``APPROX_GED`` root.

    Returns:
        ``(name, matrix, graph_ids)`` or ``None``.
    """
    if suite == "suite1":
        path = ged_root / f"{dataset}.npz"
        key, name = "ged_matrix", "exact"
    else:
        path = approx_root / "LB" / f"{dataset}.npz"
        key, name = "lb_matrix", "lb"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        return name, np.asarray(z[key], dtype=np.float64), np.asarray(z["graph_ids"]).astype(str)


def ladder_row(
    suite: str,
    dataset: str,
    distances: Path,
    ged_root: Path,
    approx_root: Path,
    cohort_manifest: dict[str, Any] | None,
) -> LadderRow | None:
    """Compute one dataset's ladder.

    Args:
        suite: Suite key.
        dataset: Dataset key.
        distances: The ``distances/`` tree.
        ged_root: Suite-1 exact matrices.
        approx_root: ``APPROX_GED`` root.
        cohort_manifest: The export manifest, for the ``raw`` rung.

    Returns:
        The row, or ``None`` when the arm's distance matrix is absent.
    """
    arm = distances / suite / f"{dataset}__{REFERENCE_ARM}__levenshtein.npz"
    if not arm.exists():
        LOGGER.warning("%s/%s: no %s matrix, skipped", suite, dataset, REFERENCE_ARM)
        return None
    reference = _load_reference(suite, dataset, ged_root, approx_root)
    if reference is None:
        LOGGER.warning("%s/%s: no reference GED, skipped", suite, dataset)
        return None
    ref_name, ged, ref_ids = reference

    with np.load(arm, allow_pickle=True) as z:
        lev = np.asarray(z["distance_matrix"], dtype=np.float64)
        mask = np.asarray(z["defined_mask"], dtype=bool)
        ids = np.asarray(z["graph_ids"]).astype(str)

    position = {gid: j for j, gid in enumerate(ref_ids)}
    if not set(ids) <= set(position):
        raise LadderError(f"{suite}/{dataset}: graph_ids do not join onto the reference")
    # Join on graph_ids, never positionally (F-12).
    perm = np.array([position[g] for g in ids])
    ged = ged[np.ix_(perm, perm)]

    n = len(ids)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    available = upper & np.isfinite(ged)
    positive = available & (ged > 0)
    separated = positive & (lev > 0)
    # The analysed set is what the correlation actually runs on: every pair with
    # a finite reference GED that this arm encoded. GED = 0 is legitimate --- it
    # is what isomorphic graphs have --- so it is NOT filtered here. The two
    # rungs below are the collision denominator, not a step towards this one.
    analysed = available & mask
    collisions = int((positive & (lev == 0)).sum())

    raw_graphs: int | None = None
    if cohort_manifest and dataset in cohort_manifest:
        raw_graphs = cohort_manifest[dataset].get("n_raw")

    return LadderRow(
        suite=suite,
        dataset=dataset,
        reference=ref_name,
        n_raw_graphs=raw_graphs,
        n_graphs=n,
        raw=(raw_graphs * (raw_graphs - 1) // 2) if raw_graphs else None,
        connected=n * (n - 1) // 2,
        ged_available=int(available.sum()),
        ged_positive=int(positive.sum()),
        lev_positive=int(separated.sum()),
        analysed=int(analysed.sum()),
        collisions=collisions,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", type=Path, required=True)
    ap.add_argument("--ged-root", type=Path, required=True)
    ap.add_argument("--approx-root", type=Path, required=True)
    ap.add_argument(
        "--cohort-manifest", type=Path, default=None, help="exported_suite2/manifest.json"
    )
    ap.add_argument("--out", type=Path, required=True, help="ladder.json")
    ap.add_argument("--suite", choices=("suite1", "suite2"), default="suite2")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        0 when at least one row was emitted, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest: dict[str, Any] | None = None
    if args.cohort_manifest and args.cohort_manifest.exists():
        manifest = json.loads(args.cohort_manifest.read_text())

    directory = args.distances / args.suite
    datasets = sorted({p.stem.split("__")[0] for p in directory.glob("*.npz")})
    rows = [
        r
        for d in datasets
        if (
            r := ladder_row(
                args.suite, d, args.distances, args.ged_root, args.approx_root, manifest
            )
        )
    ]

    header = (
        f"{'dataset':16s} {'raw':>10s} {'connected':>10s} {'GED-avail':>10s} "
        f"{'GED>0':>10s} {'Lev>0':>10s} {'analysed':>10s} {'collisions':>10s}"
    )
    print(header)
    for r in rows:
        print(
            f"{r.dataset:16s} {str(r.raw or '-'):>10s} {r.connected:>10,} {r.ged_available:>10,} "
            f"{r.ged_positive:>10,} {r.lev_positive:>10,} {r.analysed:>10,} "
            f"{r.collisions:>10,} ({r.collision_rate:.4%})"
        )

    payload = {
        "schema_version": "t06.ladder.2",
        "ticket": "T-06",
        "reference_arm": REFERENCE_ARM,
        "suite": args.suite,
        "rung_semantics": {
            "correlation_ladder": ["raw", "connected", "ged_available", "analysed"],
            "collision_denominator": [
                "ged_positive",
                "lev_positive",
                "collisions",
                "collision_rate",
            ],
            "note": (
                "Two rung families, not one chain. 'analysed' = ged_available & defined_mask "
                "and is the pair set every rho in rho_table.json is computed over -- the two "
                "files agree on n_pairs. 'ged_positive' and 'lev_positive' exist to define the "
                "collision denominator: a collision is GED > 0 with Levenshtein == 0, so it is "
                "only definable on GED-positive pairs. GED = 0 pairs are RETAINED in the "
                "analysed set: that is what isomorphic graphs have, 28.05 % of IAM Letter LOW "
                "is certified at exactly 0, and filtering them would truncate the response at "
                "its most informative end (CONTRACTS.md 4.1)."
            ),
            "corrected": (
                "schema t06.ladder.1 computed analysed = ged_positive & defined_mask, which is "
                "not the set that gets analysed. Corrected 2026-08-23 under T-06."
            ),
        },
        "n_rows": len(rows),
        "rows": [asdict(r) | {"collision_rate": r.collision_rate} for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
