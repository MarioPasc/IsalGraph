"""Join the separate role campaigns into the bracket every output file carries.

Each of the three role campaigns of CONTRACTS §3 computes one number per pair
and knows nothing about the others: ``LB/{key}.npz`` holds ``BRANCH_FAST``'s
lower bound, ``UB/{key}.npz`` holds ``BIPARTITE``'s upper bound, and
``UB_SENSITIVITY/{key}.npz`` holds ``BP_BEAM``'s. Each is written by a separate
job, on a separate node, at a separate time. This module is the one step that
sees all three, and it writes the same ``lb_matrix``, ``ub_matrix`` and
``certified_mask`` into each of them, so that a reader who opens any one file
gets the whole bracket without having to open the other two.

Why ``certified_mask`` is computed here and not by a backend
------------------------------------------------------------
``GedlibBackend.pair()`` returns ``certified=False`` unconditionally and must
keep doing so. ``ANCHOR_AWARE_GED`` was measured on Picasso reporting
``LB == UB`` on values that exhaustive brute force showed to be wrong -- a false
optimality certificate, which is worse than a wrong number because it defeats
the check meant to catch one. ``T-03-design.md`` amendment 2 retired it and
withdrew self-certification from every GEDLIB method.

The mask written here is not a self-report and does not depend on one. It is the
derived statement

    a proven lower bound of k, and an exhibited edit path of cost k,
    together prove that GED = k,

which is a theorem about the two numbers, not a claim by the solver that
produced either. ``BRANCH_FAST``'s value is a valid lower bound under D6 by
Blumenthal and Gamper's construction; ``BIPARTITE``'s value is the cost of an
edit path it actually built, hence achievable, hence a valid upper bound. When
those two coincide the distance is pinned from both sides by two independent
computations that never saw each other. That is why the mask is legitimate, and
why it is computed by a step that reads two separate campaigns rather than by
either campaign alone.

What this step never touches
----------------------------
``ged_matrix`` and ``seconds_matrix`` are each file's own measurement -- its
role's value and its role's wall time. Cross-fill leaves both exactly as it
found them. Overwriting ``ged_matrix`` with a bracket end would silently make
``UB_SENSITIVITY`` a copy of ``UB``, and the sensitivity arm exists precisely to
be compared against it.

Usage
-----
::

    python -m benchmarks.real_data.eval_setup.approx_ged_crossfill \\
        --lb  $APPROX/LB/linux.npz \\
        --ub  $APPROX/UB/linux.npz \\
        --ubs $APPROX/UB_SENSITIVITY/linux.npz

References
----------
Blumenthal, D. B., & Gamper, J. (2018). On the exact computation of the graph
edit distance. *IEEE TKDE* 30(3), 503-516. doi:10.1109/TKDE.2017.2772243
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CROSSFILL_KEYS",
    "OUTPUT_KEYS",
    "CrossfillError",
    "CrossfillReport",
    "certified_mask",
    "crossfill",
    "main",
]

#: Tolerance at which a lower and an upper bound are treated as having met.
#: The same constant the backends certify at; both ends are integer-valued
#: under D6, so this is a float-equality guard rather than a real tolerance.
CERT_TOL = 1e-9

#: The ten keys of CONTRACTS §4, in the order the exact-GED census writes them.
OUTPUT_KEYS = (
    "ged_matrix",
    "lb_matrix",
    "ub_matrix",
    "certified_mask",
    "seconds_matrix",
    "node_counts",
    "edge_counts",
    "graph_ids",
    "labels",
    "metadata",
)

#: The three arrays cross-fill rewrites. Everything else in the file is passed
#: through byte-for-byte.
CROSSFILL_KEYS = ("lb_matrix", "ub_matrix", "certified_mask")


class CrossfillError(Exception):
    """Raised when the role files disagree or a required array is missing."""


@dataclass(slots=True)
class CrossfillReport:
    """What one cross-fill did.

    Attributes:
        n_graphs: Cohort size.
        n_pairs: Off-diagonal upper-triangle pairs.
        n_certified: Pairs whose bracket closed, i.e. proven distances.
        certification_rate: ``n_certified / n_pairs``.
        n_inverted: Pairs where ``lb > ub`` beyond tolerance. Always zero in a
            sound run; non-zero means one of the two campaigns is wrong.
        max_inversion: Largest ``lb - ub`` over those pairs.
        written: Paths rewritten.
    """

    n_graphs: int
    n_pairs: int
    n_certified: int
    certification_rate: float
    n_inverted: int
    max_inversion: float
    written: list[Path]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready view of the report."""
        return {
            "n_graphs": self.n_graphs,
            "n_pairs": self.n_pairs,
            "n_certified": self.n_certified,
            "certification_rate": self.certification_rate,
            "n_inverted": self.n_inverted,
            "max_inversion": self.max_inversion,
            "written": [str(p) for p in self.written],
        }


def _load(path: Path) -> dict[str, np.ndarray]:
    """Read every array of one role file into memory.

    Args:
        path: The ``.npz`` to read.

    Returns:
        A dict of the file's arrays, detached from the file handle.

    Raises:
        CrossfillError: If the file is missing or is not a CONTRACTS §4 file.
    """
    if not path.exists():
        raise CrossfillError(f"{path} does not exist")
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as exc:
        raise CrossfillError(f"cannot read {path}: {exc}") from exc
    for required in ("ged_matrix", "graph_ids"):
        if required not in arrays:
            raise CrossfillError(f"{path} is not a CONTRACTS section 4 file: missing {required!r}")
    return arrays


def certified_mask(lb: np.ndarray, ub: np.ndarray, *, tol: float = CERT_TOL) -> np.ndarray:
    """Return the pairs whose bracket closed, and are therefore proven.

    Args:
        lb: ``(N, N)`` lower bounds.
        ub: ``(N, N)`` upper bounds.
        tol: Tolerance at which the two ends count as equal.

    Returns:
        ``(N, N)`` boolean array, diagonal ``True``. The diagonal is set rather
        than measured: a graph's distance to itself is zero by definition and no
        solver is asked for it, so leaving it to a floating-point comparison
        against an unwritten cell would be an accident waiting to happen.

    Notes:
        Derived, never self-reported (CONTRACTS §4.1). A backend's own
        ``certified`` field is deliberately not consulted here and is
        unconditionally ``False`` in any case.
    """
    mask = np.abs(np.asarray(lb, dtype=np.float64) - np.asarray(ub, dtype=np.float64)) <= tol
    mask = np.asarray(mask, dtype=np.bool_)
    np.fill_diagonal(mask, True)
    return mask


def _check_agreement(files: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """Verify the role files describe the same cohort, and return its ids.

    Args:
        files: Role label to loaded arrays.

    Returns:
        The shared ``graph_ids`` array.

    Raises:
        CrossfillError: If ids, shapes or graph counts disagree. Two role files
            whose graph order differs would cross-fill a bound from one pair
            onto another, and every value would still look plausible, so this is
            a refusal rather than a warning.
    """
    labels = list(files)
    reference = labels[0]
    ids = np.asarray(files[reference]["graph_ids"]).astype(str)
    n = int(ids.size)
    for label in labels[1:]:
        other = np.asarray(files[label]["graph_ids"]).astype(str)
        if other.size != n:
            raise CrossfillError(
                f"{label} holds {other.size} graphs but {reference} holds {n}; "
                "these are not the same cohort"
            )
        if not np.array_equal(ids, other):
            first = int(np.flatnonzero(ids != other)[0])
            raise CrossfillError(
                f"graph_ids disagree between {reference} and {label}, first at position "
                f"{first}: {ids[first]!r} against {other[first]!r}. Pair indices are positions "
                "in this order, so cross-filling these files would attach a bound to the wrong "
                "pair without changing how any of it looks."
            )
    for label in labels:
        shape = np.asarray(files[label]["ged_matrix"]).shape
        if shape != (n, n):
            raise CrossfillError(f"{label}: ged_matrix is {shape}, expected {(n, n)}")
    return ids


def _role_values(arrays: dict[str, np.ndarray], label: str) -> np.ndarray:
    """Return one role's own values as a float64 matrix.

    Args:
        arrays: The role file's arrays.
        label: Role label, for the error message.

    Returns:
        The ``ged_matrix``, which under CONTRACTS §4 holds this role's own value.

    Raises:
        CrossfillError: If the matrix holds a non-finite entry. A bound campaign
            computes a number for every pair; ``inf`` would mean the campaign did
            not finish, and merging an unfinished campaign into a published
            bracket is the failure this refuses.
    """
    values = np.asarray(arrays["ged_matrix"], dtype=np.float64)
    if not np.isfinite(values).all():
        n_bad = int(np.count_nonzero(~np.isfinite(values)))
        raise CrossfillError(
            f"{label}: ged_matrix holds {n_bad} non-finite entries; a bound campaign produces a "
            "value for every pair, so this one did not finish"
        )
    return values


def _write_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Rewrite one role file in place, atomically.

    Args:
        path: The file to replace.
        arrays: Its complete new contents.

    Notes:
        Written to a sibling temporary and moved into place with ``os.replace``
        semantics, so a reader either sees the whole previous file or the whole
        new one. A partially rewritten matrix would be indistinguishable from a
        correct one on inspection.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)


def _updated_metadata(
    arrays: dict[str, np.ndarray],
    *,
    report_fields: dict[str, Any],
) -> np.ndarray:
    """Return the file's metadata with the cross-fill fields merged in.

    Args:
        arrays: The role file's arrays.
        report_fields: Fields to add or overwrite.

    Returns:
        A zero-dimensional ``<U`` array holding the JSON string.
    """
    meta: dict[str, Any] = {}
    if "metadata" in arrays:
        try:
            meta = dict(json.loads(str(arrays["metadata"])))
        except (ValueError, TypeError):
            logger.warning("metadata was not JSON; replacing it with the cross-fill fields alone")
            meta = {}
    meta.update(report_fields)
    return np.array(json.dumps(meta))


def crossfill(
    *,
    lb_path: Path,
    ub_path: Path,
    ubs_path: Path | None = None,
    dry_run: bool = False,
) -> CrossfillReport:
    """Write the shared bracket into every role file.

    Args:
        lb_path: ``LB/{key}.npz``, whose ``ged_matrix`` is the lower bound.
        ub_path: ``UB/{key}.npz``, whose ``ged_matrix`` is the upper bound.
        ubs_path: ``UB_SENSITIVITY/{key}.npz``. Optional: the sensitivity arm
            can be cross-filled later without recomputing anything.
        dry_run: Compute and check, write nothing.

    Returns:
        The :class:`CrossfillReport`.

    Raises:
        CrossfillError: If the files disagree on the cohort, if a matrix holds a
            non-finite entry, or if the bracket is inverted on any pair.
    """
    files: dict[str, dict[str, np.ndarray]] = {
        "lb": _load(lb_path),
        "ub": _load(ub_path),
    }
    paths: dict[str, Path] = {"lb": lb_path, "ub": ub_path}
    if ubs_path is not None:
        files["ubs"] = _load(ubs_path)
        paths["ubs"] = ubs_path

    ids = _check_agreement(files)
    n = int(ids.size)

    lb_m = _role_values(files["lb"], "lb")
    ub_m = _role_values(files["ub"], "ub")

    off = ~np.eye(n, dtype=bool)
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)

    inverted = off & (lb_m > ub_m + CERT_TOL)
    n_inverted = int(np.count_nonzero(inverted & triu))
    max_inversion = float((lb_m - ub_m)[inverted].max()) if bool(inverted.any()) else 0.0
    if n_inverted:
        raise CrossfillError(
            f"{n_inverted} pairs have lb > ub, by at most {max_inversion}. These come from two "
            "independent campaigns, so an inverted bracket means one of them is wrong -- a "
            "lower bound that exceeds an achievable edit path is not a lower bound. Refusing "
            "to write a bracket that cannot hold."
        )

    mask = certified_mask(lb_m, ub_m)
    n_pairs = int(np.count_nonzero(triu))
    n_certified = int(np.count_nonzero(mask & triu))
    rate = (n_certified / n_pairs) if n_pairs else 0.0

    written: list[Path] = []
    if not dry_run:
        stamp = datetime.now(timezone.utc).isoformat()
        for label, arrays in files.items():
            updated = dict(arrays)
            updated["lb_matrix"] = lb_m
            updated["ub_matrix"] = ub_m
            updated["certified_mask"] = mask
            updated["metadata"] = _updated_metadata(
                arrays,
                report_fields={
                    "n_certified": n_certified,
                    "certification_rate": rate,
                    "crossfilled_utc": stamp,
                    "crossfill_sources": {
                        "lb": str(lb_path),
                        "ub": str(ub_path),
                        "ubs": str(ubs_path) if ubs_path is not None else None,
                    },
                },
            )
            _write_atomic(paths[label], updated)
            written.append(paths[label])

    logger.info(
        "cross-fill: %d graphs, %d pairs, %d certified (%.4f), %d files rewritten",
        n,
        n_pairs,
        n_certified,
        rate,
        len(written),
    )
    return CrossfillReport(
        n_graphs=n,
        n_pairs=n_pairs,
        n_certified=n_certified,
        certification_rate=rate,
        n_inverted=n_inverted,
        max_inversion=max_inversion,
        written=written,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser.

    Returns:
        The configured parser.
    """
    p = argparse.ArgumentParser(
        prog="approx_ged_crossfill",
        description="Write the shared lb/ub/certified bracket into every role file.",
    )
    p.add_argument("--lb", required=True, help="LB/{key}.npz")
    p.add_argument("--ub", required=True, help="UB/{key}.npz")
    p.add_argument("--ubs", default=None, help="UB_SENSITIVITY/{key}.npz")
    p.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on any refusal.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        report = crossfill(
            lb_path=Path(args.lb),
            ub_path=Path(args.ub),
            ubs_path=Path(args.ubs) if args.ubs else None,
            dry_run=bool(args.dry_run),
        )
    except (CrossfillError, OSError) as exc:
        logger.error("cross-fill failed: %s", exc)
        return 1
    logger.info("%s", json.dumps(report.as_dict()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
