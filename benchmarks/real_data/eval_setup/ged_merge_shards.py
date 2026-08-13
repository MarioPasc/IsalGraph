"""Merge CONTRACT C shards into one CONTRACT D matrix file, and run gate 4 (T-03).

Concatenates every shard for one dataset into a single ``.npz`` that
``eval_correlation.py``, ``method_comparator.py``, ``dataset_filter.py`` and
``validator.py`` consume **unchanged**: the six keys ``ged_matrix``,
``node_counts``, ``edge_counts``, ``graph_ids``, ``labels`` and ``metadata`` keep
exactly the names, dtypes and semantics they have in
``ged_computer.py::save_ged_matrix``. The four additions -- ``lb_matrix``,
``ub_matrix``, ``certified_mask``, ``seconds_matrix`` -- are ignored by that code
and carry the D11 censoring interval and the D12 timing.

Three assertions, in this order, all of them exit non-zero on failure:

1. **Coverage.** Every ``k`` in ``[0, C(N, 2))`` appears in at least one shard.
2. **Consistency.** Every ``k`` appearing in more than one shard carries identical
   values everywhere it appears. This is how stage-1 reuse is verified: the
   stage-2 census seeds itself from stage-1's results, both shards land in the
   same directory, and a disagreement means the two stages computed different
   answers for the same pair. That is a hard failure, not a rounding question.
3. **Gate 4** (CONTRACTS §7): symmetry, zero diagonal, every off-diagonal entry
   either finite and positive or ``inf`` with ``certified == False`` and a finite
   ``lb <= ub``, ``lb`` and ``ub`` matrices symmetric and finite, and
   ``lb <= ged <= ub`` wherever ``ged`` is finite.

``--delete-shards`` runs only after all three have passed and the written output
has been read back successfully.

**One documented reading of gate 4.** CONTRACTS §7 says every off-diagonal entry
must be ``0 < v < inf`` or censored, while CONTRACT B §5 invariant 1 says a value
of exactly ``0`` is legal when the two graphs are isomorphic. Both cannot hold on a
corpus containing isomorphic duplicates, which IAM Letter and AIDS do. This module
resolves it in the direction that keeps the trap closed: a zero off-diagonal is
accepted **only** when the pair is certified with ``lb == ub == 0``, is counted
separately in the metadata, and is rejected outright under ``--strict-nonzero``.
An uncertified zero -- the "matrix silently filled with zeros" failure that
CONTRACT B §5 invariant 1 exists to catch -- always fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .ged_pair_index import GedPairIndexError, n_pairs, pairs_from_indices
else:  # pragma: no cover - only when run as a bare script from eval_setup/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ged_pair_index import (  # noqa: E402
        GedPairIndexError,
        n_pairs,
        pairs_from_indices,
    )

logger = logging.getLogger(__name__)

__all__ = ["Gate4Report", "MergeError", "collect_shards", "gate4", "main", "merge_shards"]

SHARD_KEYS = ("pair_index", "ged", "lb", "ub", "certified", "seconds")

#: Off-diagonal exact-zero fraction at or above which gate 4 refuses the merge
#: (CONTRACTS §7). Set just under 1.0 rather than at some tuned threshold: the
#: failure this catches fills essentially the *whole* matrix with zeros, and a
#: real dataset that genuinely reached 99 % isomorphic pairs would deserve the
#: same look.
_ZERO_FRACTION_LIMIT = 0.99

#: Legacy vocabulary, so downstream metadata readers see a familiar string.
_COST_FUNCTION_NAMES = {
    "unit": "uniform_topology_only",
    "graphedx": "graphedx_edge_only",
}


class MergeError(Exception):
    """Raised on missing coverage, conflicting shards or a gate-4 violation."""


@dataclass(slots=True)
class Gate4Report:
    """Outcome of the structural gate.

    Attributes:
        passed: Whether every check held.
        violations: Human-readable descriptions of what failed.
        n_certified: Off-diagonal upper-triangle pairs with a certified optimum.
        n_censored: Off-diagonal upper-triangle pairs that are interval-censored.
        n_zero_offdiag: Certified off-diagonal zeros, i.e. proven isomorphic pairs.
        max_asymmetry: Largest ``|M - M.T|`` over the finite part of ``ged_matrix``.
        zero_offdiag_fraction: Off-diagonal entries of ``ged_matrix`` that are
            exactly zero, as a fraction of the off-diagonal upper triangle.
            Recorded whatever the source, because a matrix that is almost all
            zeros is the shape the wrong accessor produces, and GEDLIB reports
            that failure no other way.
    """

    passed: bool
    violations: list[str] = field(default_factory=list)
    n_certified: int = 0
    n_censored: int = 0
    n_zero_offdiag: int = 0
    max_asymmetry: float = 0.0
    zero_offdiag_fraction: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready view of the report."""
        return {
            "passed": self.passed,
            "violations": self.violations,
            "n_certified": self.n_certified,
            "n_censored": self.n_censored,
            "n_zero_offdiag_certified": self.n_zero_offdiag,
            "max_asymmetry": self.max_asymmetry,
            "zero_offdiag_fraction": self.zero_offdiag_fraction,
        }


def collect_shards(shard_dir: Path, key: str, exclude: set[Path]) -> list[Path]:
    """Find the shard files for one dataset.

    Selection is by *content*, not by filename alone: a candidate must carry all
    six CONTRACT C keys. That keeps a ``pair_list.npz`` or a stray export out of the
    merge without needing a brittle naming convention, and checkpoints are excluded
    by name as well because they are legitimately shard-shaped.

    Args:
        shard_dir: Directory to scan.
        key: Dataset key, e.g. ``aids``.
        exclude: Paths to skip, typically the output file.

    Returns:
        Sorted list of shard paths.

    Raises:
        MergeError: If the directory does not exist or holds no shards.
    """
    if not shard_dir.is_dir():
        raise MergeError(f"{shard_dir} is not a directory")
    found: list[Path] = []
    for path in sorted(shard_dir.glob(f"{key}_*.npz")):
        if path.resolve() in exclude or path.name.endswith(".ckpt.npz"):
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                if all(k in data for k in SHARD_KEYS):
                    found.append(path)
                else:
                    logger.info("skipping %s: not a CONTRACT C shard", path.name)
        except (OSError, ValueError) as exc:
            raise MergeError(f"cannot read {path}: {exc}") from exc
    if not found:
        raise MergeError(f"no CONTRACT C shards matching {key}_*.npz in {shard_dir}")
    return found


@dataclass(slots=True)
class _Accumulated:
    """Dense per-pair arrays over the whole upper triangle."""

    seen: np.ndarray
    ged: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    certified: np.ndarray
    seconds: np.ndarray
    n_duplicate: int = 0
    shard_meta: list[dict[str, Any]] = field(default_factory=list)


def _accumulate(shards: list[Path], total: int) -> _Accumulated:
    """Load every shard into dense arrays, checking duplicates for agreement.

    Args:
        shards: Shard paths.
        total: ``C(N, 2)``.

    Returns:
        The accumulated arrays.

    Raises:
        MergeError: On an out-of-range index, a NaN, or a duplicated pair whose
            values disagree between shards.
    """
    acc = _Accumulated(
        seen=np.zeros(total, dtype=bool),
        ged=np.full(total, np.nan, dtype=np.float64),
        lb=np.full(total, np.nan, dtype=np.float64),
        ub=np.full(total, np.nan, dtype=np.float64),
        certified=np.zeros(total, dtype=bool),
        seconds=np.zeros(total, dtype=np.float32),
    )
    conflicts: list[str] = []

    for path in shards:
        with np.load(path, allow_pickle=False) as data:
            k = np.asarray(data["pair_index"], dtype=np.int64)
            ged = np.asarray(data["ged"], dtype=np.float64)
            lb = np.asarray(data["lb"], dtype=np.float64)
            ub = np.asarray(data["ub"], dtype=np.float64)
            cert = np.asarray(data["certified"], dtype=np.bool_)
            sec = np.asarray(data["seconds"], dtype=np.float32)
            if "meta" in data:
                try:
                    acc.shard_meta.append(dict(json.loads(str(data["meta"]))))
                except (ValueError, TypeError):
                    acc.shard_meta.append({"file": path.name, "meta": "unparseable"})

        if k.size and (int(k.min()) < 0 or int(k.max()) >= total):
            raise MergeError(f"{path.name}: pair index outside [0, {total})")
        if np.isnan(ged).any() or np.isnan(lb).any() or np.isnan(ub).any():
            raise MergeError(f"{path.name}: NaN in ged/lb/ub -- refusing to merge")
        if k.size != np.unique(k).size:
            raise MergeError(f"{path.name}: repeats a pair index within a single shard")

        dup = acc.seen[k]
        if bool(dup.any()):
            acc.n_duplicate += int(dup.sum())
            dk = k[dup]
            # inf == inf compares True, which is what censored agreement needs.
            bad = (
                (acc.ged[dk] != ged[dup])
                | (acc.lb[dk] != lb[dup])
                | (acc.ub[dk] != ub[dup])
                | (acc.certified[dk] != cert[dup])
            )
            if bool(bad.any()):
                for pos in np.flatnonzero(bad)[:5]:
                    kk = int(dk[pos])
                    conflicts.append(
                        f"k={kk}: stored (ged={acc.ged[kk]}, lb={acc.lb[kk]}, ub={acc.ub[kk]}, "
                        f"cert={bool(acc.certified[kk])}) vs {path.name} "
                        f"(ged={float(ged[dup][pos])}, lb={float(lb[dup][pos])}, "
                        f"ub={float(ub[dup][pos])}, cert={bool(cert[dup][pos])})"
                    )
                raise MergeError(
                    f"{int(bad.sum())} pairs disagree between shards. The two stages computed "
                    f"different answers for the same pair; this is a hard failure. "
                    + " | ".join(conflicts)
                )

        fresh = ~dup
        kf = k[fresh]
        acc.ged[kf] = ged[fresh]
        acc.lb[kf] = lb[fresh]
        acc.ub[kf] = ub[fresh]
        acc.certified[kf] = cert[fresh]
        acc.seconds[kf] = sec[fresh]
        acc.seen[kf] = True

    return acc


def gate4(
    ged: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    certified: np.ndarray,
    *,
    strict_nonzero: bool = False,
    ged_from: str = "exact",
    computed: str = "both",
) -> Gate4Report:
    """Run the structural gate on the assembled matrices.

    Args:
        ged: ``(N, N)`` distance matrix, ``inf`` where censored.
        lb: ``(N, N)`` lower bounds, finite.
        ub: ``(N, N)`` upper bounds, finite.
        certified: ``(N, N)`` optimality certificate mask.
        strict_nonzero: Enforce the literal CONTRACTS §7 text, under which a zero
            off-diagonal entry fails even when it is a certified isomorphism.
        ged_from: Which quantity ``ged`` holds. ``'exact'`` is T-03's census and
            the default, and keeps every check exactly as it was. Under ``'lb'``
            or ``'ub'`` the matrix holds a *bound*, not a distance, and a zero
            entry no longer implies an isomorphism: ``BRANCH_FAST`` returns the
            trivial bound 0 on real pairs whose true distance is 2 or 6, so
            demanding a certificate for every zero would reject correct data.
        computed: Which ends the campaign evaluated. ``'both'`` is the default
            and requires both bound matrices finite. A single-role campaign
            leaves the other end at its sentinel, and only the end it computed
            is checked.

    Returns:
        The :class:`Gate4Report`.

    Raises:
        MergeError: If ``ged_from`` or ``computed`` is not a recognised value.
    """
    if ged_from not in ("exact", "lb", "ub"):
        raise MergeError(f"ged_from must be exact, lb or ub; got {ged_from!r}")
    if computed not in ("both", "lb", "ub"):
        raise MergeError(f"computed must be both, lb or ub; got {computed!r}")
    rep = Gate4Report(passed=True)
    n = ged.shape[0]
    off = ~np.eye(n, dtype=bool)
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)

    if np.isnan(ged).any() or np.isnan(lb).any() or np.isnan(ub).any():
        rep.violations.append("NaN present in ged, lb or ub")
    if not np.array_equal(ged, ged.T):
        rep.violations.append("ged_matrix is not symmetric")
    if not np.array_equal(lb, lb.T):
        rep.violations.append("lb_matrix is not symmetric")
    if not np.array_equal(ub, ub.T):
        rep.violations.append("ub_matrix is not symmetric")
    if not np.array_equal(certified, certified.T):
        rep.violations.append("certified_mask is not symmetric")

    finite = np.isfinite(ged)
    # Subtract only where both orientations are finite: inf - inf is NaN, and
    # np.where would evaluate it anyway before selecting.
    finite_both = finite & finite.T
    if bool(finite_both.any()):
        rep.max_asymmetry = float(np.abs(ged[finite_both] - ged.T[finite_both]).max())

    if not np.all(np.diag(ged) == 0.0):
        rep.violations.append("ged_matrix diagonal is not exactly zero")
    if computed != "ub" and not np.isfinite(lb).all():
        rep.violations.append("lb_matrix holds a non-finite entry")
    if computed != "lb" and not np.isfinite(ub).all():
        rep.violations.append("ub_matrix holds a non-finite entry")
    if bool(np.any(lb[off] > ub[off] + 1e-9)):
        rep.violations.append("some off-diagonal lb exceeds its ub")

    censored = np.isinf(ged) & off
    rep.n_censored = int(np.count_nonzero(censored & triu))
    rep.n_certified = int(np.count_nonzero(certified & triu))
    if bool(np.any(censored & certified)):
        rep.violations.append("a censored entry is marked certified")

    finite_off = finite & off
    if bool(np.any(ged[finite_off] < 0.0)):
        rep.violations.append("a negative distance is present")
    zeros = finite_off & (ged == 0.0)
    rep.n_zero_offdiag = int(np.count_nonzero(zeros & triu))
    n_offdiag = int(np.count_nonzero(triu))
    rep.zero_offdiag_fraction = (rep.n_zero_offdiag / n_offdiag) if n_offdiag else 0.0

    # CONTRACTS §7 addition. The failure a wrong accessor produces is not one
    # bad number, it is a matrix that is almost entirely zero, and it raises
    # nothing on the way there: get_lower_bound() on an upper-bound method
    # returns 0.00 silently. This is the shape check for that, and it applies to
    # every source -- an exact census that came out 99 % zeros would be just as
    # wrong, just as quietly.
    if rep.zero_offdiag_fraction >= _ZERO_FRACTION_LIMIT:
        rep.violations.append(
            f"{rep.zero_offdiag_fraction:.4f} of off-diagonal pairs are exactly zero, at or "
            f"above the {_ZERO_FRACTION_LIMIT} limit -- this is the shape of a matrix filled "
            "through the wrong accessor, which GEDLIB does not report as an error"
        )

    if rep.n_zero_offdiag and ged_from == "exact":
        # A zero *distance* is a claim of isomorphism and needs the certificate
        # that proves it. A zero *bound* claims nothing: BRANCH_FAST's trivial
        # lower bound is 0 on real pairs whose exact distance is 2 and 6
        # (measured on Picasso 2026-08-12), so this check would reject correct
        # data if it were applied to a bound matrix.
        uncertified_zero = zeros & ~certified
        if bool(uncertified_zero.any()):
            rep.violations.append(
                f"{int(np.count_nonzero(uncertified_zero & triu))} off-diagonal zeros are not "
                "certified -- this is the silently-zero-filled-matrix failure"
            )
        elif bool(np.any((lb[zeros] != 0.0) | (ub[zeros] != 0.0))):
            rep.violations.append("a certified zero does not have lb == ub == 0")
        elif strict_nonzero:
            rep.violations.append(
                f"{rep.n_zero_offdiag} certified off-diagonal zeros, rejected under "
                "--strict-nonzero (CONTRACTS section 7 literal reading)"
            )

    bracket = finite_off & ((ged < lb - 1e-9) | (ged > ub + 1e-9))
    if bool(bracket.any()):
        rep.violations.append(
            f"{int(np.count_nonzero(bracket & triu))} finite entries fall outside [lb, ub]"
        )
    if bool(np.any(np.isinf(ged) & ~off)):
        rep.violations.append("the diagonal holds an infinite entry")

    rep.passed = not rep.violations
    return rep


def _orientation_summary(shards: list[Path]) -> dict[str, Any]:
    """Pool the per-shard ``ub_fwd``/``ub_rev`` columns into four numbers.

    The upper bound is built from a *directed* assignment and is therefore not
    symmetric. The manuscript currently reports that asymmetry from our own
    bipartite implementation over 400 LINUX pairs at mean order 8.71; this
    measures it under the production method over the whole cohort.

    Args:
        shards: Shard paths.

    Returns:
        Counts and magnitudes, or ``{}`` when no shard carried the columns.

    Notes:
        Aggregate only, deliberately. Two more ``(N, N)`` matrices would add
        roughly 243 MB on COIL-DEL for a quantity that is only ever reported per
        dataset, so nothing dense is built here.
    """
    n = 0
    n_asym = 0
    n_rev_tighter = 0
    total_gap = 0.0
    max_gap = 0.0
    for path in shards:
        with np.load(path, allow_pickle=False) as data:
            if "ub_fwd" not in data or "ub_rev" not in data:
                continue
            fwd = np.asarray(data["ub_fwd"], dtype=np.float64)
            rev = np.asarray(data["ub_rev"], dtype=np.float64)
        usable = np.isfinite(fwd) & np.isfinite(rev)
        if not bool(usable.any()):
            continue
        gap = np.abs(fwd[usable] - rev[usable])
        n += int(usable.sum())
        n_asym += int(np.count_nonzero(gap > 1e-9))
        n_rev_tighter += int(np.count_nonzero(rev[usable] < fwd[usable] - 1e-9))
        total_gap += float(gap.sum())
        max_gap = max(max_gap, float(gap.max()))
    if not n:
        return {}
    return {
        "n_pairs": n,
        "n_asymmetric": n_asym,
        "asymmetry_rate": n_asym / n,
        "mean_abs_gap": total_gap / n,
        "max_abs_gap": max_gap,
        "reverse_tighter_rate": n_rev_tighter / n,
    }


def _repo_commit() -> str | None:
    """Return the repository HEAD sha, or ``None`` outside a checkout."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def _gedlib_source() -> str | None:
    """Return the directory the GEDLIB bindings were loaded from.

    Recorded because the bindings are an in-place build outside site-packages,
    so the merged file cannot otherwise say which build produced its values.
    Import failure is not an error here: the merge itself never calls GEDLIB.
    """
    try:
        import gklearn.gedlib as _g  # noqa: PLC0415
    except Exception:  # pragma: no cover - environment dependent
        return None
    path = getattr(_g, "__file__", None)
    return str(Path(path).resolve().parent) if path else None


def _optional_int(value: Any) -> int | None:
    """Coerce a metadata value to ``int``, preserving an explicit ``None``.

    Suite 2 applies no ``n_max`` and its exporter records that as JSON ``null``
    rather than by omitting the key, so a plain ``int(...)`` raises and a
    ``dict.get(key, default)`` default never fires. ``None`` is carried through
    because "no size cap" and "capped at 12" are different facts about the
    cohort and the metadata is what a reader consults to tell them apart.

    Args:
        value: The raw metadata value, possibly ``None``.

    Returns:
        The value as an ``int``, or ``None`` when it was ``None``.
    """
    return None if value is None else int(value)


def _agreed(shard_meta: list[dict[str, Any]], field_name: str) -> str | None:
    """Return the single value the shards carry for one metadata field.

    Args:
        shard_meta: Parsed ``meta`` dicts, one per shard.
        field_name: The key to read.

    Returns:
        The agreed value, ``None`` when no shard carries the key, or a
        ``'MIXED: ...'`` marker when they disagree. Disagreement is recorded
        rather than raised here because ``gate4`` owns the refusals; a mixed
        options string is caught by the campaign gate, which is where CONTRACTS
        §3 puts it.
    """
    values = {
        str(m[field_name]) for m in shard_meta if field_name in m and m[field_name] is not None
    }
    if not values:
        return None
    if len(values) > 1:
        return "MIXED: " + "|".join(sorted(values))
    return values.pop()


def _computed_mode(shard_meta: list[dict[str, Any]]) -> str:
    """Return the compute mode the shards agree on.

    Args:
        shard_meta: Parsed ``meta`` dicts, one per shard.

    Returns:
        ``'both'``, ``'lb'`` or ``'ub'``. Shards written before the flag existed
        carry no ``compute`` key and mean ``'both'``.

    Raises:
        MergeError: If the shards disagree. Two halves of one matrix computed
            under different modes would leave part of a bound matrix at its
            sentinel and part measured, with nothing in the file to say which.
    """
    modes = {str(m.get("compute", "both")) for m in shard_meta}
    if not modes:
        return "both"
    if len(modes) > 1:
        raise MergeError(f"shards mix compute modes {sorted(modes)}; refusing to merge")
    mode = modes.pop()
    if mode not in ("both", "lb", "ub"):
        raise MergeError(f"shards declare an unknown compute mode {mode!r}")
    return mode


def _load_cohort(path: Path) -> dict[str, Any]:
    """Read the per-graph arrays CONTRACT D needs from the CONTRACT A file.

    The frozen merge CLI does not name this input, but ``node_counts``,
    ``edge_counts``, ``graph_ids`` and ``labels`` exist nowhere in a shard, so
    CONTRACT D cannot be written without it. Reported to the orchestrator as a gap
    in the frozen CLI; the flag is optional and falls back to convention.

    Args:
        path: CONTRACT A ``.npz``.

    Returns:
        A dict of the four arrays plus the parsed source metadata.

    Raises:
        MergeError: If a required key is missing.
    """
    with np.load(path, allow_pickle=False) as data:
        for key in ("graph_ids", "n_nodes", "n_edges"):
            if key not in data:
                raise MergeError(f"{path} is not a CONTRACT A file: missing '{key}'")
        graph_ids = np.asarray(data["graph_ids"])
        n_nodes = np.asarray(data["n_nodes"], dtype=np.int32)
        n_edges = np.asarray(data["n_edges"], dtype=np.int32)
        labels = (
            np.asarray(data["labels"])
            if "labels" in data
            else np.array([""] * graph_ids.size, dtype=str)
        )
        meta: dict[str, Any] = {}
        if "metadata" in data:
            try:
                meta = dict(json.loads(str(data["metadata"])))
            except (ValueError, TypeError):
                meta = {}
    return {
        "graph_ids": graph_ids,
        "node_counts": n_nodes,
        "edge_counts": n_edges,
        "labels": labels,
        "source_metadata": meta,
    }


def _find_cohort(shard_dir: Path, key: str, explicit: str | None) -> Path:
    """Locate the CONTRACT A file for this dataset.

    Args:
        shard_dir: Directory holding the shards.
        key: Dataset key.
        explicit: Value of ``--input``, if given.

    Returns:
        Path to the CONTRACT A ``.npz``.

    Raises:
        MergeError: If no candidate exists.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise MergeError(f"--input {path} does not exist")
        return path
    for candidate in (shard_dir / f"{key}.npz", shard_dir.parent / f"{key}.npz"):
        if candidate.is_file():
            logger.info("using %s as the CONTRACT A cohort file", candidate)
            return candidate
    raise MergeError(
        f"cannot find a CONTRACT A file for {key!r}; pass --input. node_counts, edge_counts, "
        "graph_ids and labels do not exist in any shard, so CONTRACT D cannot be written "
        "without it."
    )


def merge_shards(
    *,
    shard_dir: Path,
    key: str,
    n_graphs: int,
    out: Path,
    cohort_path: str | None = None,
    strict_nonzero: bool = False,
    ged_from: str = "exact",
    role: str | None = None,
    seconds_role: str | None = None,
) -> tuple[Gate4Report, list[Path]]:
    """Merge every shard for one dataset and write the CONTRACT D file.

    Args:
        shard_dir: Directory holding the shards.
        key: Dataset key.
        n_graphs: Expected graph count; must match the cohort file.
        out: Output path.
        cohort_path: Optional explicit CONTRACT A path.
        strict_nonzero: Pass through to :func:`gate4`.
        ged_from: Which shard array becomes ``ged_matrix``. ``'exact'`` is
            T-03's behaviour and the default. ``'lb'`` and ``'ub'`` exist for
            the T-05 role campaigns, whose reported value *is* a bound
            (CONTRACTS §4).
        role: Role label written into the metadata, e.g. ``'lb'`` or ``'ubs'``.
        seconds_role: Provenance label for ``seconds_matrix``. The wall time in
            a role file is that role's own method's, and pooling it with
            another role's would be meaningless, so it is named.

    Returns:
        ``(gate4_report, shard_paths)``.

    Raises:
        MergeError: On any coverage, consistency or gate-4 failure.
    """
    if ged_from not in ("exact", "lb", "ub"):
        raise MergeError(f"--ged-from must be exact, lb or ub; got {ged_from!r}")
    cohort = _load_cohort(_find_cohort(shard_dir, key, cohort_path))
    n = int(np.asarray(cohort["graph_ids"]).size)
    if n != n_graphs:
        raise MergeError(f"--n-graphs {n_graphs} but the cohort file holds {n} graphs")
    total = n_pairs(n)

    shards = collect_shards(shard_dir, key, exclude={out.resolve()})
    logger.info("merging %d shards for %s (%d graphs, %d pairs)", len(shards), key, n, total)
    acc = _accumulate(shards, total)

    missing = int(total - np.count_nonzero(acc.seen))
    if missing:
        gaps = np.flatnonzero(~acc.seen)[:10]
        raise MergeError(
            f"{missing} of {total} pairs are absent from every shard "
            f"(first missing indices: {gaps.tolist()})"
        )

    cost_models = {str(m.get("cost_model")) for m in acc.shard_meta if "cost_model" in m}
    if len(cost_models) > 1:
        raise MergeError(f"shards mix cost models {sorted(cost_models)}; refusing to merge")
    backends = sorted({str(m.get("backend_name", m.get("backend", "?"))) for m in acc.shard_meta})
    if len(backends) > 1:
        logger.warning("shards were produced by more than one backend: %s", backends)

    i, j = pairs_from_indices(np.arange(total, dtype=np.int64), n)
    ged_m = np.zeros((n, n), dtype=np.float64)
    lb_m = np.zeros((n, n), dtype=np.float64)
    ub_m = np.zeros((n, n), dtype=np.float64)
    cert_m = np.zeros((n, n), dtype=np.bool_)
    sec_m = np.zeros((n, n), dtype=np.float32)
    for mat, vals in (
        (ged_m, acc.ged),
        (lb_m, acc.lb),
        (ub_m, acc.ub),
        (cert_m, acc.certified),
        (sec_m, acc.seconds),
    ):
        mat[i, j] = vals
        mat[j, i] = vals
    np.fill_diagonal(cert_m, True)

    if ged_from != "exact":
        # The role's own bound becomes the reported value. The diagonal is set
        # explicitly: a bound of a graph against itself is 0 by construction and
        # the solver is never asked for it.
        ged_m = (lb_m if ged_from == "lb" else ub_m).copy()
        np.fill_diagonal(ged_m, 0.0)

    computed = _computed_mode(acc.shard_meta)
    report = gate4(
        ged_m,
        lb_m,
        ub_m,
        cert_m,
        strict_nonzero=strict_nonzero,
        ged_from=ged_from,
        computed=computed,
    )
    if not report.passed:
        raise MergeError("gate 4 failed: " + "; ".join(report.violations))

    src = dict(cohort["source_metadata"])
    cost_model = next(iter(cost_models)) if cost_models else "unit"
    n_valid = int(np.isfinite(ged_m[np.triu_indices(n, k=1)]).sum())
    metadata = json.dumps(
        {
            # Existing schema, unchanged, so downstream readers keep working.
            "dataset": key,
            "ged_method": "+".join(backends) if backends else "unknown",
            "ged_cost_function": _COST_FUNCTION_NAMES.get(cost_model, cost_model),
            "source": str(src.get("source", "unknown")),
            "n_graphs": n,
            "n_valid_pairs": n_valid,
            # Suite 2 runs with NO size cap and its exporter writes `"n_max": null`
            # (CONTRACTS §2). The key is PRESENT and None, so a `.get(..., 12)`
            # default never fires and `int(None)` raises. None is the honest value
            # here -- "no cap" is not the same fact as "capped at 12" -- and the 12
            # fallback is kept for legacy files that omit the key entirely.
            "n_max_filter": _optional_int(dict(src.get("filter", {})).get("n_max", 12)),
            "n_dropped": int(
                src.get("n_dropped_size", 0)
                + src.get("n_dropped_disconnected", 0)
                + src.get("n_dropped_trivial", 0)
            ),
            # Additions.
            "cost_model": cost_model,
            "role": role,
            "ged_from": ged_from,
            "compute": computed,
            "seconds_role": seconds_role if seconds_role is not None else role,
            # CONTRACTS §3: the options string is part of the method name, so it
            # is carried from the shards into the merged file verbatim.
            "method": _agreed(acc.shard_meta, "lb_method" if ged_from == "lb" else "ub_method"),
            "options_string": _agreed(
                acc.shard_meta, "lb_options" if ged_from == "lb" else "ub_options"
            ),
            "accessor": {"exact": "exact", "lb": "lower", "ub": "upper"}[ged_from],
            "n_zero_offdiag": report.n_zero_offdiag,
            "zero_offdiag_fraction": report.zero_offdiag_fraction,
            # Aggregate, never two more (N,N) matrices. Empty unless the
            # campaign ran with --record-orientations.
            "ub_orientation": _orientation_summary(shards),
            "n_pairs": total,
            "n_certified": report.n_certified,
            "n_censored": report.n_censored,
            "censoring_rate": (report.n_censored / total) if total else 0.0,
            "n_zero_offdiag_certified": report.n_zero_offdiag,
            "n_shards": len(shards),
            "n_duplicate_pairs": acc.n_duplicate,
            "total_solver_seconds": float(np.sum(acc.seconds, dtype=np.float64)),
            "gate4": report.as_dict(),
            "merged_utc": datetime.now(timezone.utc).isoformat(),
            # CONTRACTS §4's required key list, added by fix(integration) after
            # the independent G4 gate reported all seven missing. Kept ALONGSIDE
            # the pre-existing near-synonyms rather than renaming them, so that
            # readers of `total_solver_seconds` / `merged_utc` keep working.
            "seconds_total": float(np.sum(acc.seconds, dtype=np.float64)),
            "mean_seconds_per_pair": (
                float(np.sum(acc.seconds, dtype=np.float64) / total) if total else 0.0
            ),
            "computed_utc": datetime.now(timezone.utc).isoformat(),
            "filter": dict(src.get("filter", {})),
            "splits_merged": bool(src.get("splits_merged", True)),
            "gedlib_source": _gedlib_source(),
            "code_commit": _repo_commit(),
            "schema_version": 1,
        }
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            ged_matrix=ged_m,
            node_counts=np.asarray(cohort["node_counts"], dtype=np.int32),
            edge_counts=np.asarray(cohort["edge_counts"], dtype=np.int32),
            graph_ids=np.asarray(cohort["graph_ids"], dtype=str),
            labels=np.asarray(cohort["labels"], dtype=str),
            metadata=np.array(metadata),
            lb_matrix=lb_m,
            ub_matrix=ub_m,
            certified_mask=cert_m,
            seconds_matrix=sec_m,
        )
    tmp.replace(out)

    with np.load(out, allow_pickle=False) as check:
        for required in ("ged_matrix", "node_counts", "edge_counts", "graph_ids", "labels"):
            if required not in check:
                raise MergeError(f"written file {out} is missing '{required}'")
    return report, shards


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser, frozen in CONTRACTS §6.

    Returns:
        The configured parser. ``--input`` and ``--strict-nonzero`` are additions;
        every flag named in the contract keeps its name and meaning.
    """
    p = argparse.ArgumentParser(
        prog="ged_merge_shards",
        description="Merge CONTRACT C shards into one CONTRACT D matrix and run gate 4.",
    )
    p.add_argument("--shards", required=True, help="directory holding the shards")
    p.add_argument("--key", required=True, help="dataset key, e.g. aids")
    p.add_argument("--n-graphs", type=int, required=True, help="expected graph count")
    p.add_argument("--out", required=True, help="output CONTRACT D .npz")
    p.add_argument(
        "--input", default=None, help="CONTRACT A .npz (needed for the per-graph arrays)"
    )
    p.add_argument("--delete-shards", action="store_true", help="only after every assertion passes")
    p.add_argument(
        "--strict-nonzero",
        action="store_true",
        help="reject certified off-diagonal zeros (literal CONTRACTS section 7)",
    )
    p.add_argument(
        "--ged-from",
        default="exact",
        choices=("exact", "lb", "ub"),
        help="which shard array becomes ged_matrix (default exact, T-03's behaviour)",
    )
    p.add_argument("--role", default=None, help="role label written into the metadata")
    p.add_argument("--seconds-role", default=None, help="provenance label for seconds_matrix")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` when the merge and gate 4 both pass, ``1`` otherwise.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    out = Path(args.out)
    try:
        report, shards = merge_shards(
            shard_dir=Path(args.shards),
            key=args.key,
            n_graphs=int(args.n_graphs),
            out=out,
            cohort_path=args.input,
            strict_nonzero=bool(args.strict_nonzero),
            ged_from=str(args.ged_from),
            role=args.role,
            seconds_role=args.seconds_role,
        )
    except (MergeError, GedPairIndexError, OSError) as exc:
        logger.error("merge failed: %s", exc)
        return 1

    logger.info(
        "gate 4 passed: %d certified, %d censored, %d certified zeros, max asymmetry %.3g",
        report.n_certified,
        report.n_censored,
        report.n_zero_offdiag,
        report.max_asymmetry,
    )
    logger.info("wrote %s", out)

    if args.delete_shards:
        for path in shards:
            path.unlink()
        logger.info("deleted %d shards", len(shards))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
