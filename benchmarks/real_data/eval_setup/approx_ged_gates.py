"""The independent validation gates of ticket T-05.

Every gate is a runnable command that writes a JSON record and returns 0 on pass,
1 on fail. They read the **finished** campaign files and decide whether a number
from them may be printed.

================= ========================================================
Gate              What it proves
================= ========================================================
``G2``            the campaign reproduces T-27's recorded ``BRANCH_FAST``
                  and ``BIPARTITE`` census exactly, on all 3,602,615 pairs
                  of the four datasets whose Suite-2 cohort is identical
                  to Suite 1
``G3``            ``lb <= ub`` on every Suite-2 pair, and
                  ``lb <= exact <= ub`` against T-03's certified values
``G4``            the written files are structurally what CONTRACTS §4
                  says they are: ten keys, stated dtypes, symmetric,
                  zero-diagonal, finite, non-negative
``lb-consistency`` re-running ``BRANCH_FAST`` on a seeded sample
                  reproduces ``LB/{key}.npz``
================= ========================================================

Independence is the point
-------------------------
These gates are written against ``CONTRACTS.md``, not against the code that
produces the files. In particular ``lb-consistency`` calls GEDLIB directly
rather than through :mod:`ged_backends`: routing the independent check through
the machinery it checks would make it circular, and would verify determinism
where the intent is to verify correctness.

Nothing here imports :mod:`isalgraph` (CONTRACTS §9) and nothing here imports
:mod:`ged_backends`. ``networkx`` and ``gklearn`` are imported lazily, inside
``lb-consistency`` only, so the module is usable with neither installed.

Tolerance, and why each gate uses the one it does
-------------------------------------------------
GED under cost model D6 is integer-valued and stored as ``float64``. T-03
recorded two successively tighter tolerance guesses (``1e-9``, then ``1e-6``)
both reporting storage noise as disagreement against a **third-party** file.
Against **our own** output there is no storage noise to absorb:

* **Exact equality** for G2 and ``lb-consistency``. Both compare our output
  against a value produced by the same deterministic method under the same
  cost model. A tolerance here would hide a real defect, and any non-zero
  difference is one.
* **``1e-9``** for the inequality comparisons in G3 and G4. An inequality
  needs a tolerance on the boundary case ``lb == ub``, where a float64
  round-trip of an integer is exact but the comparison should not depend on
  that being true.

Every JSON record names the tolerance it used and the reason.

Ordering: G4 runs AFTER cross-fill
----------------------------------
``--compute lb`` leaves ``ub = inf`` in the shard and ``--compute ub`` leaves
``lb = -inf`` (CONTRACTS §6). Those become the ``lb_matrix`` / ``ub_matrix`` of
a freshly merged role file, and only the CONTRACTS §4.2 cross-fill puts the
real arrays into all three. Running G4 before cross-fill therefore reports
``non-finite entries`` correctly and uselessly.

Usage
-----
``python -m benchmarks.real_data.eval_setup.approx_ged_gates --gate all
--lb-dir <LB> --ub-dir <UB> --ubs-dir <UB_SENSITIVITY>
--t27-cells <cells> --exact-dir <computed> --out <dir>``
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import socket
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("approx_ged_gates")

SCHEMA_VERSION = 1

#: The ten Suite-2 dataset keys, CONTRACTS §1.
SUITE2_DATASETS: tuple[str, ...] = (
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

#: The four datasets whose Suite-2 cohort is identical to Suite 1, and therefore the
#: only ones on which T-27's recorded census is comparable pair-for-pair.
G2_DATASETS: tuple[str, ...] = (
    "iam_letter_low",
    "iam_letter_med",
    "iam_letter_high",
    "linux",
)

#: Pair count G2 covers when all four datasets are present. Recorded so a partial run
#: cannot be mistaken for the full gate.
G2_FULL_PAIRS = 3_602_615

#: role -> (T-27 cell name, campaign directory argument).
G2_CELLS: tuple[tuple[str, str, str], ...] = (
    ("lb", "BRANCH_FAST", "lb_dir"),
    ("ub", "BIPARTITE", "ub_dir"),
)

#: CONTRACTS §4, the ten keys and the dtype each must have.
REQUIRED_KEYS: dict[str, str] = {
    "ged_matrix": "float64",
    "lb_matrix": "float64",
    "ub_matrix": "float64",
    "certified_mask": "bool",
    "seconds_matrix": "float32",
    "node_counts": "int32",
    "edge_counts": "int32",
    "graph_ids": "U",
    "labels": "U",
    "metadata": "U",
}

#: The three matrices that must behave as a distance matrix.
VALUE_MATRICES: tuple[str, ...] = ("ged_matrix", "lb_matrix", "ub_matrix")

#: CONTRACTS §4, the metadata JSON keys.
REQUIRED_METADATA: tuple[str, ...] = (
    "dataset",
    "role",
    "method",
    "options_string",
    "accessor",
    "cost_model",
    "n_graphs",
    "n_pairs",
    "n_zero_offdiag",
    "n_certified",
    "certification_rate",
    "seconds_total",
    "mean_seconds_per_pair",
    "filter",
    "splits_merged",
    "gedlib_source",
    "code_commit",
    "computed_utc",
    "schema_version",
)

#: Cost model D6: node ins/del = 1, edge ins/del = 1, substitutions free.
D6_COSTS: tuple[int, int, int, int, int, int] = (1, 1, 0, 1, 1, 0)

INEQUALITY_TOL = 1e-9
SYMMETRY_TOL = 1e-12
ZERO_OFFDIAG_MAX_FRACTION = 0.99


class GateError(Exception):
    """A gate could not be evaluated, as distinct from a gate that failed."""


@dataclass
class GateRecord:
    """The JSON record one gate writes.

    Attributes
    ----------
    gate : str
        Gate identifier.
    passed : bool
        Whether the gate passed.
    tolerance : str
        Human-readable statement of the comparison tolerance used.
    tolerance_rationale : str
        Why that tolerance and not another.
    n_compared : int
        Number of scalar comparisons the gate made.
    n_violations : int
        Number of comparisons that failed.
    violations : list of dict
        Up to ``max_violations`` failures, each naming its pair index.
    per_dataset : dict
        Per-dataset detail.
    notes : list of str
        Anything a reader must know to interpret the verdict.
    """

    gate: str
    passed: bool = True
    tolerance: str = ""
    tolerance_rationale: str = ""
    n_compared: int = 0
    n_violations: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)
    per_dataset: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0
    schema_version: int = SCHEMA_VERSION

    def fail(self, note: str) -> None:
        """Mark the gate failed and record why.

        Parameters
        ----------
        note : str
            The reason, recorded verbatim in ``notes``.
        """
        self.passed = False
        self.notes.append(note)


def environment_record() -> dict[str, Any]:
    """Describe the machine the gates ran on.

    Returns
    -------
    dict
        Host, platform, Python and NumPy versions.
    """
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


# --------------------------------------------------------------------------- helpers


def _load(path: Path) -> dict[str, np.ndarray]:
    """Load an ``.npz`` without pickle.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    dict
        Mapping from key to array.

    Raises
    ------
    GateError
        If the file does not exist or cannot be read.
    """
    if not path.exists():
        raise GateError(f"missing file: {path}")
    try:
        with np.load(path, allow_pickle=False) as handle:
            return {key: handle[key] for key in handle.files}
    except (OSError, ValueError) as exc:  # pragma: no cover - corrupt file
        raise GateError(f"unreadable: {path}: {exc}") from exc


def _upper(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the strict upper triangle in canonical order.

    Parameters
    ----------
    matrix : ndarray
        Square matrix.

    Returns
    -------
    values, rows, cols : ndarray
        ``matrix[triu_indices(N, k=1)]`` and the index arrays that produced it.
    """
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols], rows, cols


def _violations(
    mask: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    ours: np.ndarray,
    reference: np.ndarray,
    limit: int,
    dataset: str,
    what: str,
) -> list[dict[str, Any]]:
    """Describe up to ``limit`` failing comparisons, naming each pair index.

    Parameters
    ----------
    mask : ndarray of bool
        True where the comparison failed.
    rows, cols : ndarray
        Graph indices of every compared pair.
    ours, reference : ndarray
        The two compared value arrays.
    limit : int
        Maximum number of violations to describe.
    dataset : str
        Dataset key, recorded on every violation.
    what : str
        Short description of the comparison.

    Returns
    -------
    list of dict
        One record per described violation.
    """
    where = np.flatnonzero(mask)[:limit]
    return [
        {
            "dataset": dataset,
            "comparison": what,
            "flat_index": int(k),
            "pair": [int(rows[k]), int(cols[k])],
            "ours": float(ours[k]),
            "reference": float(reference[k]),
            "difference": float(ours[k] - reference[k]),
        }
        for k in where
    ]


def _graph_ids_match(
    record: GateRecord,
    dataset: str,
    ours: np.ndarray,
    references: dict[str, np.ndarray],
) -> bool:
    """Assert the campaign's graph order equals every available reference's.

    This is a **precondition**, checked before any value comparison. Every pair
    index downstream is ``numpy.triu_indices(N, k=1)`` over the graph order, so a
    cohort in a different order does not produce wrong bounds -- it produces
    correct bounds compared against the wrong pairs, over millions of entries.
    Reporting that as "graph order differs" rather than as three million
    disagreeing values is the difference between a diagnosable failure and an
    inexplicable one.

    The check is on **identity**, never on the form of an id. Graph ids are the
    source loader's native ids: Letter ships bare filename stems (``IP1_0000``),
    only GraphEdX ships ``linux_train_0000``. A gate that validated the shape of
    an id would fail on eight of the ten datasets.

    Parameters
    ----------
    record : GateRecord
        Record to annotate on failure.
    dataset : str
        Dataset key.
    ours : ndarray
        The campaign file's ``graph_ids``.
    references : dict
        Mapping from reference name to that reference's ``graph_ids``.

    Returns
    -------
    bool
        True if every reference agrees element-wise.
    """
    if not references:
        record.fail(
            f"{dataset}: no reference cohort available to check graph order against; "
            "the G2 precondition cannot be evaluated, so the gate cannot pass"
        )
        return False
    ok = True
    for name, ref in references.items():
        if ours.shape != ref.shape:
            record.fail(
                f"{dataset}: graph order differs from {name}: "
                f"{ours.shape[0]} graphs vs {ref.shape[0]}"
            )
            ok = False
            continue
        if not np.array_equal(ours.astype(str), ref.astype(str)):
            n_bad = int((ours.astype(str) != ref.astype(str)).sum())
            first = int(np.flatnonzero(ours.astype(str) != ref.astype(str))[0])
            record.fail(
                f"{dataset}: graph order differs from {name} at {n_bad} of "
                f"{ours.shape[0]} positions, first at index {first} "
                f"({ours[first]!r} vs {ref[first]!r}). Every pair index is "
                "misaligned; the value comparison is not attempted."
            )
            ok = False
    return ok


# --------------------------------------------------------------------------- G2


def gate2_t27_reproduction(
    lb_dir: Path,
    ub_dir: Path,
    t27_cells: Path,
    exact_dir: Path | None,
    t27_index: Path | None,
    datasets: Sequence[str],
    max_violations: int,
) -> GateRecord:
    """Reproduce T-27's recorded ``BRANCH_FAST`` and ``BIPARTITE`` census exactly.

    The strongest gate available: one comparison covers the loader, the cost
    model, the options string, symmetrisation and pair ordering against a census
    already on record, at 9,000x the sample size of the T-03 cross-check T-27
    discharged.

    T-27's arrays are flat, in canonical ``numpy.triu_indices(N, k=1)`` order;
    the campaign output is a dense symmetric matrix, so its upper triangle is
    taken in the same order.

    Parameters
    ----------
    lb_dir, ub_dir : Path
        Campaign output directories for roles ``lb`` and ``ub``.
    t27_cells : Path
        Directory of T-27 cell files, ``{ds}__{CELL}.npz``, key ``value``.
    exact_dir : Path or None
        T-03's exact census, used as a graph-order reference.
    t27_index : Path or None
        T-27's index files, used as a graph-order reference.
    datasets : sequence of str
        Datasets to compare; non-G2 datasets are skipped with a note.
    max_violations : int
        Cap on described violations.

    Returns
    -------
    GateRecord
        The verdict.
    """
    started = time.monotonic()
    record = GateRecord(
        gate="G2",
        tolerance="exact equality (numpy.array_equal)",
        tolerance_rationale=(
            "Both sides are the same deterministic method under the same cost model "
            "D6, and T-27's values are our own recorded output rather than a "
            "third-party file. There is no storage noise for a tolerance to absorb, "
            "so any non-zero difference is a real defect. T-03 twice recorded a "
            "tolerance guess (1e-9, then 1e-6) reporting storage noise as "
            "disagreement against a third-party file; that situation does not arise "
            "here and the tolerance is not carried over."
        ),
    )
    dirs = {"lb_dir": lb_dir, "ub_dir": ub_dir}

    skipped = [d for d in datasets if d not in G2_DATASETS]
    if skipped:
        record.notes.append(
            f"not compared (Suite-2 cohort differs from Suite 1, so pair indices are "
            f"not comparable): {', '.join(sorted(skipped))}"
        )
    wanted = [d for d in datasets if d in G2_DATASETS]
    if not wanted:
        record.fail("no G2 dataset requested; the gate compared nothing")
        record.seconds = time.monotonic() - started
        return record

    for dataset in wanted:
        detail: dict[str, Any] = {}
        record.per_dataset[dataset] = detail

        references: dict[str, np.ndarray] = {}
        for name, root in (("t27_index", t27_index), ("exact", exact_dir)):
            if root is None:
                continue
            candidate = root / f"{dataset}.npz"
            if candidate.exists():
                arrays = _load(candidate)
                if "graph_ids" in arrays:
                    references[name] = arrays["graph_ids"]
        detail["graph_order_references"] = sorted(references)

        first = _load(dirs["lb_dir"] / f"{dataset}.npz")
        if not _graph_ids_match(record, dataset, first["graph_ids"], references):
            detail["precondition"] = "FAILED: graph order"
            continue
        detail["precondition"] = "graph order matches every reference"

        for role, cell, dir_key in G2_CELLS:
            arrays = _load(dirs[dir_key] / f"{dataset}.npz")
            cell_path = t27_cells / f"{dataset}__{cell}.npz"
            reference = _load(cell_path)["value"]

            ours, rows, cols = _upper(arrays["ged_matrix"])
            if ours.shape != reference.shape:
                record.fail(
                    f"{dataset}/{cell}: {ours.shape[0]} pairs in the campaign file "
                    f"vs {reference.shape[0]} in T-27's cell"
                )
                detail[cell] = {"error": "pair count differs"}
                continue

            bad = ours != reference
            n_bad = int(bad.sum())
            record.n_compared += int(ours.size)
            record.n_violations += n_bad
            detail[cell] = {
                "role": role,
                "n_pairs": int(ours.size),
                "n_mismatched": n_bad,
                "reference": str(cell_path),
            }
            if n_bad:
                record.violations.extend(
                    _violations(
                        bad, rows, cols, ours, reference, max_violations, dataset, f"{cell} vs T-27"
                    )
                )
                record.fail(f"{dataset}/{cell}: {n_bad} of {ours.size} pairs differ from T-27")

    if record.n_compared and record.n_violations == 0:
        record.notes.append(
            f"{record.n_compared} value comparisons, 0 disagreements, over {len(wanted)} datasets"
        )
    covered = sum(d.get("BRANCH_FAST", {}).get("n_pairs", 0) for d in record.per_dataset.values())
    record.per_dataset["_coverage"] = {
        "pairs_per_cell": covered,
        "pairs_per_cell_when_complete": G2_FULL_PAIRS,
        "complete": covered == G2_FULL_PAIRS,
    }
    if covered != G2_FULL_PAIRS:
        record.notes.append(
            f"PARTIAL: {covered} pairs per cell against {G2_FULL_PAIRS} when all four "
            "G2 datasets are present. This is not the full gate."
        )
    record.seconds = time.monotonic() - started
    return record


# --------------------------------------------------------------------------- G3


def gate3_bracket(
    lb_dir: Path,
    ub_dir: Path,
    exact_dir: Path | None,
    datasets: Sequence[str],
    max_violations: int,
    tol: float = INEQUALITY_TOL,
) -> GateRecord:
    """Check ``lb <= ub`` everywhere, and ``lb <= exact <= ub`` where exact is known.

    ``LB <= GED <= UB`` is the only thing the paper's large-``n`` argument rests
    on. The exact-value arm uses T-03's ``certified_mask`` to select: T-03's
    ``ged_matrix`` carries ``NaN`` on censored pairs, so an unselected comparison
    would silently propagate ``NaN`` through every inequality, and ``NaN <= x``
    is ``False`` without raising.

    Parameters
    ----------
    lb_dir, ub_dir : Path
        Campaign output directories.
    exact_dir : Path or None
        T-03's exact census. Without it only the bracket arm runs.
    datasets : sequence of str
        Datasets to check.
    max_violations : int
        Cap on described violations.
    tol : float
        Slack on each inequality.

    Returns
    -------
    GateRecord
        The verdict.
    """
    started = time.monotonic()
    record = GateRecord(
        gate="G3",
        tolerance=f"inequalities compared at {tol:g}",
        tolerance_rationale=(
            "An inequality needs slack only on its boundary case lb == ub, where a "
            "float64 round-trip of an integer is exact but the verdict should not "
            "depend on that. 1e-9 is far below the D6 unit spacing of 1, so it "
            "cannot mask a real violation, which is always at least 1."
        ),
    )
    if exact_dir is None:
        record.notes.append("no --exact-dir: the lb <= exact <= ub arm did not run, only lb <= ub")

    for dataset in datasets:
        detail: dict[str, Any] = {}
        record.per_dataset[dataset] = detail
        lb_arrays = _load(lb_dir / f"{dataset}.npz")
        ub_arrays = _load(ub_dir / f"{dataset}.npz")

        lower, rows, cols = _upper(lb_arrays["lb_matrix"])
        upper, _, _ = _upper(ub_arrays["ub_matrix"])
        bad = ~(lower <= upper + tol)
        n_bad = int(bad.sum())
        record.n_compared += int(lower.size)
        record.n_violations += n_bad
        detail["bracket"] = {"n_pairs": int(lower.size), "n_inverted": n_bad}
        if n_bad:
            record.violations.extend(
                _violations(bad, rows, cols, lower, upper, max_violations, dataset, "lb <= ub")
            )
            record.fail(f"{dataset}: {n_bad} inverted brackets (lb > ub)")

        if exact_dir is None:
            continue
        exact_path = exact_dir / f"{dataset}.npz"
        if not exact_path.exists():
            detail["exact"] = "no T-03 file"
            continue
        exact_arrays = _load(exact_path)

        # Compare by index only when the cohorts are the same graphs in the same
        # order. aids_graphedx (819 graphs) is a DIFFERENT cohort from Suite 1's
        # aids (769) and must never be compared positionally.
        ours_ids = lb_arrays["graph_ids"].astype(str)
        ref_ids = exact_arrays["graph_ids"].astype(str)
        if ours_ids.shape != ref_ids.shape or not np.array_equal(ours_ids, ref_ids):
            detail["exact"] = (
                "SKIPPED: cohort differs from T-03's; a positional comparison would "
                "compare unrelated graphs"
            )
            record.notes.append(
                f"{dataset}: exact arm skipped, cohort differs from T-03's "
                f"({ours_ids.shape[0]} vs {ref_ids.shape[0]} graphs)"
            )
            continue

        certified, _, _ = _upper(exact_arrays["certified_mask"])
        exact, _, _ = _upper(exact_arrays["ged_matrix"])
        selected = certified & np.isfinite(exact)
        n_sel = int(selected.sum())
        detail["exact"] = {
            "n_certified": n_sel,
            "n_certified_but_nonfinite": int((certified & ~np.isfinite(exact)).sum()),
        }
        if n_sel == 0:
            continue

        lo_bad = ~(lower[selected] <= exact[selected] + tol)
        hi_bad = ~(exact[selected] <= upper[selected] + tol)
        sel_rows, sel_cols = rows[selected], cols[selected]
        record.n_compared += 2 * n_sel
        record.n_violations += int(lo_bad.sum()) + int(hi_bad.sum())
        detail["exact"]["n_lb_above_exact"] = int(lo_bad.sum())
        detail["exact"]["n_ub_below_exact"] = int(hi_bad.sum())
        if lo_bad.any():
            record.violations.extend(
                _violations(
                    lo_bad,
                    sel_rows,
                    sel_cols,
                    lower[selected],
                    exact[selected],
                    max_violations,
                    dataset,
                    "lb <= exact",
                )
            )
            record.fail(f"{dataset}: {int(lo_bad.sum())} pairs with lb > exact")
        if hi_bad.any():
            record.violations.extend(
                _violations(
                    hi_bad,
                    sel_rows,
                    sel_cols,
                    upper[selected],
                    exact[selected],
                    max_violations,
                    dataset,
                    "exact <= ub",
                )
            )
            record.fail(f"{dataset}: {int(hi_bad.sum())} pairs with ub < exact")

    record.seconds = time.monotonic() - started
    return record


# --------------------------------------------------------------------------- G4


def _check_one_file(record: GateRecord, path: Path, tol: float) -> dict[str, Any]:
    """Apply every structural check to one written file.

    Parameters
    ----------
    record : GateRecord
        Record to annotate on failure.
    path : Path
        File to check.
    tol : float
        Slack for the ``certified_mask`` consistency check.

    Returns
    -------
    dict
        Per-file detail.
    """
    label = f"{path.parent.name}/{path.name}"
    detail: dict[str, Any] = {}
    arrays = _load(path)

    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    detail["missing_keys"] = missing
    if missing:
        record.fail(f"{label}: missing key(s) {', '.join(missing)}")
        return detail

    wrong: list[str] = []
    for key, expected in REQUIRED_KEYS.items():
        actual = arrays[key].dtype
        ok = actual.kind == "U" if expected == "U" else actual.name == expected
        if not ok:
            wrong.append(f"{key}: {actual} != {expected}")
    detail["wrong_dtypes"] = wrong
    if wrong:
        record.fail(f"{label}: {'; '.join(wrong)}")

    # labels: presence and dtype only. Suite-2 class counts are NOT asserted --
    # they were raw dataset counts, not post-filter counts, and LINUX and
    # AIDS-GraphEdX legitimately ship an all-empty labels column.
    detail["labels_all_empty"] = bool((arrays["labels"].astype(str) == "").all())

    n = int(arrays["ged_matrix"].shape[0])
    detail["n_graphs"] = n
    record.n_compared += 1

    for key in VALUE_MATRICES:
        matrix = arrays[key]
        sub: dict[str, Any] = {}
        detail[key] = sub

        if matrix.shape != (n, n):
            record.fail(f"{label}/{key}: shape {matrix.shape} is not ({n}, {n})")
            continue

        asym = float(np.abs(matrix - matrix.T).max()) if n else 0.0
        sub["max_asymmetry"] = asym
        # NaN propagates through max() as NaN, and NaN <= tol is False, so a file
        # full of NaN fails here rather than passing a comparison silently.
        if not (asym <= SYMMETRY_TOL):
            record.fail(
                f"{label}/{key}: not symmetric, max |A - A.T| = {asym}. "
                "An upper-bound matrix filled in one orientation is not a "
                "distance matrix (decision §6.2)."
            )

        diag = np.diag(matrix)
        max_diag = float(np.abs(diag).max()) if n else 0.0
        sub["max_abs_diagonal"] = max_diag
        if not (max_diag <= SYMMETRY_TOL):
            record.fail(f"{label}/{key}: diagonal is not zero, max |d| = {max_diag}")

        n_nonfinite = int((~np.isfinite(matrix)).sum())
        sub["n_nonfinite"] = n_nonfinite
        if n_nonfinite:
            record.fail(
                f"{label}/{key}: {n_nonfinite} non-finite entries. If this file has "
                "not been cross-filled yet, that is expected and this gate is being "
                "run too early (CONTRACTS §4.2)."
            )

        finite = matrix[np.isfinite(matrix)]
        n_negative = int((finite < 0).sum())
        sub["n_negative"] = n_negative
        if n_negative:
            record.fail(f"{label}/{key}: {n_negative} negative entries")

        n_off = n * n - n
        n_zero_off = int((matrix == 0).sum()) - int((diag == 0).sum())
        frac = (n_zero_off / n_off) if n_off else 0.0
        sub["offdiag_zero_fraction"] = frac
        sub["n_zero_offdiag"] = n_zero_off
        if frac >= ZERO_OFFDIAG_MAX_FRACTION:
            record.fail(
                f"{label}/{key}: {frac:.4f} of off-diagonal entries are exactly zero. "
                "GEDLIB returns 0.00 through the wrong accessor and raises nothing; "
                "this is that signature."
            )

    mask = arrays["certified_mask"]
    diag_all_true = bool(np.all(np.diag(mask))) if n else True
    detail["certified_diagonal_all_true"] = diag_all_true
    if not diag_all_true:
        record.fail(f"{label}: certified_mask diagonal is not all True")

    # certified_mask is a DERIVED PROOF (CONTRACTS §4.1), never a backend self-report:
    # a proven lower bound of k and an exhibited edit path of cost k together prove
    # GED = k. Its definition is lb == ub at 1e-9, so it must equal that.
    if mask.shape == (n, n):
        expected_mask = np.abs(arrays["lb_matrix"] - arrays["ub_matrix"]) <= tol
        np.fill_diagonal(expected_mask, True)
        disagree = int((mask != expected_mask).sum())
        detail["certified_mask_disagreements"] = disagree
        if disagree:
            record.fail(
                f"{label}: certified_mask disagrees with |lb - ub| <= {tol:g} on "
                f"{disagree} entries; it must be derived, not self-reported"
            )

    try:
        metadata = json.loads(str(arrays["metadata"]))
    except ValueError as exc:
        record.fail(f"{label}: metadata is not valid JSON: {exc}")
        return detail
    missing_meta = [k for k in REQUIRED_METADATA if k not in metadata]
    detail["missing_metadata"] = missing_meta
    if missing_meta:
        record.fail(f"{label}: metadata missing {', '.join(missing_meta)}")
    options = str(metadata.get("options_string", ""))
    detail["options_string"] = options
    if not options.strip():
        record.fail(
            f"{label}: metadata.options_string is empty. The options string is part "
            "of the method name -- GEDLIB's upper bounds change on 91.5-93.6 % of "
            "pairs between runs at library defaults (T-27 §4.2)."
        )
    return detail


def gate4_structural(
    role_dirs: dict[str, Path],
    datasets: Sequence[str],
    tol: float = INEQUALITY_TOL,
) -> GateRecord:
    """Read every written file and check it is what CONTRACTS §4 says it is.

    This duplicates the merge's own structural gate deliberately: an independent
    reader of the finished file, written against the contract rather than against
    the writer, is what makes the merge's self-report checkable.

    Parameters
    ----------
    role_dirs : dict
        Mapping from role id to its output directory.
    datasets : sequence of str
        Datasets to check.
    tol : float
        Slack for the ``certified_mask`` consistency check.

    Returns
    -------
    GateRecord
        The verdict.
    """
    started = time.monotonic()
    record = GateRecord(
        gate="G4",
        tolerance=(
            f"symmetry and diagonal at {SYMMETRY_TOL:g}; certified_mask "
            f"consistency at {tol:g}; zero-fraction is an exact-equality count"
        ),
        tolerance_rationale=(
            "Symmetry and the zero diagonal are exact properties of a correctly "
            "written file, so the slack is machine-precision rather than "
            "value-scale: a real asymmetry from a one-orientation fill is at least "
            "1 under D6, ten orders of magnitude above 1e-12. The off-diagonal zero "
            "fraction is counted with exact equality on purpose -- it exists to "
            "catch GEDLIB's wrong-accessor 0.00, which is an exact zero."
        ),
    )
    for role, directory in sorted(role_dirs.items()):
        for dataset in datasets:
            path = directory / f"{dataset}.npz"
            key = f"{role}:{dataset}"
            if not path.exists():
                record.fail(f"{key}: missing file {path}")
                record.per_dataset[key] = {"error": "missing"}
                continue
            record.per_dataset[key] = _check_one_file(record, path, tol)
    record.n_violations = len(record.notes) if not record.passed else 0
    record.seconds = time.monotonic() - started
    return record


# --------------------------------------------------------------------------- lb


def _graphs_from_export(arrays: dict[str, np.ndarray]) -> list[Any]:
    """Rebuild NetworkX graphs from the CONTRACTS §2 CSR export.

    Node and edge attributes are set to the constant string ``"1"``. Under cost
    model D6 substitutions are free, so no label affects the cost; GEDLIB's GXL
    bindings require attributes to be strings.

    Parameters
    ----------
    arrays : dict
        Contents of an ``exported_suite2/{key}.npz``.

    Returns
    -------
    list
        One ``networkx.Graph`` per exported graph, in export order.
    """
    import networkx as nx

    n_nodes = arrays["n_nodes"]
    offsets = arrays["edge_offsets"]
    edges = arrays["edges"]
    graphs: list[Any] = []
    for index in range(int(n_nodes.shape[0])):
        graph = nx.Graph()
        for node in range(int(n_nodes[index])):
            graph.add_node(node, l="1")
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        for u, v in zip(edges[0, lo:hi], edges[1, lo:hi], strict=True):
            graph.add_edge(int(u), int(v), l="1")
        graphs.append(graph)
    return graphs


def _branch_fast(graphs: list[Any], pairs: np.ndarray) -> np.ndarray:
    """Compute ``BRANCH_FAST`` lower bounds directly through GEDLIB.

    Deliberately does **not** go through :mod:`ged_backends`. This function is
    the independent check on the campaign, and running it through the campaign's
    own backend would verify determinism where the intent is to verify
    correctness.

    Parameters
    ----------
    graphs : list
        Graphs in export order.
    pairs : ndarray of int, shape (P, 2)
        Index pairs to evaluate.

    Returns
    -------
    ndarray of float, shape (P,)
        The lower bound for each pair.

    Raises
    ------
    GateError
        If every read is zero, which is GEDLIB's wrong-accessor signature.
    """
    import importlib

    # 🔴 libraries_import dlopen()s libdoublefann/libsvm/libnomad and MUST load
    # before gedlibpy_gxl. isort/ruff reorder plain `from ... import` lines
    # alphabetically and break this; importlib.import_module cannot be reordered.
    importlib.import_module("gklearn.gedlib.libraries_import")
    gedlib = importlib.import_module("gklearn.gedlib.gedlibpy_gxl")

    env = gedlib.GEDEnvGXL()
    handles = [env.add_nx_graph(graph, "") for graph in graphs]
    env.set_edit_cost("CONSTANT", edit_cost_constant=list(D6_COSTS))
    env.init(init_option="EAGER_WITHOUT_SHUFFLED_COPIES")
    env.set_method("BRANCH_FAST", "--threads 1")
    env.init_method()

    out = np.empty(pairs.shape[0], dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        env.run_method(handles[int(i)], handles[int(j)])
        out[k] = float(env.get_lower_bound(handles[int(i)], handles[int(j)]))
    return out


def _assert_reads_sane(values: np.ndarray, label: str) -> list[str]:
    """Check a block of GEDLIB reads for the wrong-accessor signature.

    The project rule "assert ``0 < value < inf`` on every read" is right about
    ``inf`` and about the all-zero failure, and **wrong per pair on the lower
    end**: GED is legitimately zero for isomorphic graphs, and T-03's recorded
    ``iam_letter_low`` census has 215,968 exactly-zero off-diagonal entries out
    of 1,391,220. A per-pair ``value > 0`` assertion fails on 15.5 % of correct
    Letter LOW pairs.

    What actually distinguishes the wrong accessor is that it returns 0.00 for
    **every** pair. So the checks are: finite, non-negative, and not identically
    zero across the block.

    Parameters
    ----------
    values : ndarray
        The reads to check.
    label : str
        Description used in the returned messages.

    Returns
    -------
    list of str
        One message per failed check; empty if all pass.
    """
    problems: list[str] = []
    n_nonfinite = int((~np.isfinite(values)).sum())
    if n_nonfinite:
        problems.append(f"{label}: {n_nonfinite} non-finite reads")
    n_negative = int((values[np.isfinite(values)] < 0).sum())
    if n_negative:
        problems.append(f"{label}: {n_negative} negative reads")
    if values.size and not (values > 0).any():
        problems.append(
            f"{label}: every one of {values.size} reads is exactly 0.00, which is "
            "GEDLIB's wrong-accessor signature -- get_lower_bound() on an "
            "upper-bound method returns 0.00 and raises nothing"
        )
    return problems


def gate_lb_consistency(
    lb_dir: Path,
    input_dir: Path,
    datasets: Sequence[str],
    seed: int,
    sample_size: int,
    max_violations: int,
) -> GateRecord:
    """Re-run ``BRANCH_FAST`` on a seeded sample and compare to ``LB/{key}.npz``.

    The cross-check that the three separately-submitted role campaigns saw the
    same lower bound, at negligible cost.

    Parameters
    ----------
    lb_dir : Path
        Campaign output directory for role ``lb``.
    input_dir : Path
        Directory of ``exported_suite2/{key}.npz`` inputs.
    datasets : sequence of str
        Datasets to sample.
    seed : int
        RNG seed; the draw is reproducible from it alone.
    sample_size : int
        Pairs to draw per dataset.
    max_violations : int
        Cap on described violations.

    Returns
    -------
    GateRecord
        The verdict.
    """
    started = time.monotonic()
    record = GateRecord(
        gate="lb-consistency",
        tolerance="exact equality",
        tolerance_rationale=(
            "BRANCH_FAST is deterministic (T-27 recorded deterministic=True for the "
            "cell) and both sides run the same cost model D6 with the same verbatim "
            "options string. Any difference is a real defect, not noise."
        ),
    )
    record.notes.append(
        "GEDLIB is called directly, not through ged_backends: routing the "
        "independent check through the machinery it checks would make it circular."
    )
    for dataset in datasets:
        detail: dict[str, Any] = {}
        record.per_dataset[dataset] = detail
        lb_path = lb_dir / f"{dataset}.npz"
        in_path = input_dir / f"{dataset}.npz"
        if not lb_path.exists() or not in_path.exists():
            detail["error"] = "missing campaign or input file"
            record.fail(f"{dataset}: missing {lb_path if not lb_path.exists() else in_path}")
            continue

        lb_arrays = _load(lb_path)
        in_arrays = _load(in_path)
        matrix = lb_arrays["ged_matrix"]
        n = int(matrix.shape[0])
        rows, cols = np.triu_indices(n, k=1)
        if rows.size == 0:
            detail["error"] = "no pairs"
            continue

        rng = np.random.default_rng(seed)
        take = min(sample_size, int(rows.size))
        picked = rng.choice(rows.size, size=take, replace=False)
        pairs = np.stack([rows[picked], cols[picked]], axis=1)

        graphs = _graphs_from_export(in_arrays)
        if len(graphs) != n:
            record.fail(f"{dataset}: export has {len(graphs)} graphs, campaign file has {n}")
            detail["error"] = "cohort size differs"
            continue

        recomputed = _branch_fast(graphs, pairs)
        problems = _assert_reads_sane(recomputed, f"{dataset} recomputed BRANCH_FAST")
        for problem in problems:
            record.fail(problem)

        stored = matrix[pairs[:, 0], pairs[:, 1]]
        bad = recomputed != stored
        n_bad = int(bad.sum())
        record.n_compared += take
        record.n_violations += n_bad
        detail.update({"n_sampled": take, "n_mismatched": n_bad, "seed": seed})
        if n_bad:
            record.violations.extend(
                _violations(
                    bad,
                    pairs[:, 0],
                    pairs[:, 1],
                    stored,
                    recomputed,
                    max_violations,
                    dataset,
                    "stored vs recomputed BRANCH_FAST",
                )
            )
            record.fail(f"{dataset}: {n_bad} of {take} sampled pairs disagree")
    record.seconds = time.monotonic() - started
    return record


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = argparse.ArgumentParser(
        description="Independent validation gates for the T-05 bound campaigns."
    )
    parser.add_argument(
        "--gate",
        action="append",
        default=None,
        choices=["G2", "G3", "G4", "lb-consistency", "all"],
        help="Gate to run; repeatable. Default: all.",
    )
    parser.add_argument("--lb-dir", type=Path, default=None, help="LB/ directory.")
    parser.add_argument("--ub-dir", type=Path, default=None, help="UB/ directory.")
    parser.add_argument("--ubs-dir", type=Path, default=None, help="UB_SENSITIVITY/ directory.")
    parser.add_argument(
        "--t27-cells", type=Path, default=None, help="T-27 cells directory (G2 reference)."
    )
    parser.add_argument(
        "--t27-index", type=Path, default=None, help="T-27 index directory (graph-order reference)."
    )
    parser.add_argument(
        "--exact-dir",
        type=Path,
        default=None,
        help="T-03 extended_merged_exact_ged/computed (G3 reference).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="exported_suite2 directory (lb-consistency input).",
    )
    parser.add_argument(
        "--datasets", default=",".join(SUITE2_DATASETS), help="Comma-separated dataset keys."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("approx_ged_gate_reports"),
        help="Directory for the JSON records.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument(
        "--workers", type=int, default=1, help="Recorded for provenance; the gates are IO-bound."
    )
    parser.add_argument(
        "--max-violations",
        type=int,
        default=20,
        help="Cap on violations described in the JSON record.",
    )
    parser.add_argument(
        "--inequality-tol",
        type=float,
        default=INEQUALITY_TOL,
        help="Slack on G3's and G4's inequality comparisons.",
    )
    return parser


def _require(value: Path | None, flag: str, gate: str) -> Path:
    """Return a required path or explain which flag is missing.

    Parameters
    ----------
    value : Path or None
        The supplied value.
    flag : str
        The flag name, for the error message.
    gate : str
        The gate that needs it.

    Returns
    -------
    Path
        The value.

    Raises
    ------
    GateError
        If the value is None.
    """
    if value is None:
        raise GateError(f"gate {gate} needs {flag}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested gates.

    Parameters
    ----------
    argv : sequence of str, optional
        Command line, defaulting to :data:`sys.argv`.

    Returns
    -------
    int
        ``0`` if every requested gate passed, ``1`` otherwise.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    datasets = [d for d in (s.strip() for s in args.datasets.split(",")) if d]
    wanted = args.gate or ["all"]
    if "all" in wanted:
        wanted = ["G2", "G3", "G4", "lb-consistency"]

    args.out.mkdir(parents=True, exist_ok=True)
    records: list[GateRecord] = []

    for gate in wanted:
        try:
            if gate == "G2":
                records.append(
                    gate2_t27_reproduction(
                        _require(args.lb_dir, "--lb-dir", gate),
                        _require(args.ub_dir, "--ub-dir", gate),
                        _require(args.t27_cells, "--t27-cells", gate),
                        args.exact_dir,
                        args.t27_index,
                        datasets,
                        args.max_violations,
                    )
                )
            elif gate == "G3":
                records.append(
                    gate3_bracket(
                        _require(args.lb_dir, "--lb-dir", gate),
                        _require(args.ub_dir, "--ub-dir", gate),
                        args.exact_dir,
                        datasets,
                        args.max_violations,
                        args.inequality_tol,
                    )
                )
            elif gate == "G4":
                role_dirs = {
                    role: path
                    for role, path in (
                        ("lb", args.lb_dir),
                        ("ub", args.ub_dir),
                        ("ubs", args.ubs_dir),
                    )
                    if path is not None
                }
                if not role_dirs:
                    raise GateError("gate G4 needs at least one of --lb-dir/--ub-dir/--ubs-dir")
                records.append(gate4_structural(role_dirs, datasets, args.inequality_tol))
            elif gate == "lb-consistency":
                records.append(
                    gate_lb_consistency(
                        _require(args.lb_dir, "--lb-dir", gate),
                        _require(args.input_dir, "--input-dir", gate),
                        datasets,
                        args.seed,
                        args.sample_size,
                        args.max_violations,
                    )
                )
        except GateError as exc:
            failed = GateRecord(gate=gate, passed=False)
            failed.notes.append(f"could not evaluate: {exc}")
            records.append(failed)

    passed = True
    for record in records:
        payload = asdict(record)
        payload["environment"] = environment_record()
        payload["invocation"] = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        }
        destination = args.out / f"gate_{record.gate}.json"
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
        status = "PASS" if record.passed else "FAIL"
        LOGGER.info(
            "%s %s  compared=%d violations=%d  tolerance=%s  -> %s",
            status,
            record.gate,
            record.n_compared,
            record.n_violations,
            record.tolerance,
            destination,
        )
        for note in record.notes:
            LOGGER.info("  %s: %s", record.gate, note)
        passed = passed and record.passed
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
