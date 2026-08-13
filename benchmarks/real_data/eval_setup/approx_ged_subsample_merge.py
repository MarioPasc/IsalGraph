"""Join the ``ubt`` role's shards onto the frozen subsample pair list.

The other three roles produce one dense ``(N, N)`` matrix per dataset, which is
what ``ged_merge_shards.py`` writes. The ``ubt`` role does not: its scope is the
size-stratified subsample of CONTRACTS §5, a **flat** list of roughly 28,000
pairs pooled across all ten datasets. ``--n-graphs`` is meaningless for it and
no single ``--key`` names its cohort, so the dense merge CLI cannot express it
and is deliberately not widened to try -- that path is load-bearing for T-03, a
closed ticket.

This is a join, not a merge. It matches shard rows to pair-list rows on
``(dataset_key, pair_i, pair_j)`` and writes the CONTRACTS §5 flat schema in the
pair list's own row order, so the output is reproducible from seed 42 alone.

Why the join is checked in both directions
------------------------------------------
A partial join here is undetectable downstream: the file would simply be shorter
than it should be, with every row it did contain perfectly valid. Nothing later
in the pipeline knows how many rows to expect, because the count depends on the
realised bin populations. So a pair present in the shards but absent from the
list, or present in the list but absent from the shards, is a hard failure
rather than a dropped row.

A note on ``value_fwd`` and ``value_rev``
-----------------------------------------
CONTRACTS §5 asks for the two orientations separately. The CONTRACT C shard
schema is frozen at six arrays by §6.2 and carries only ``ub``, which the
backend has **already** symmetrised to ``min(fwd, rev)``. The per-orientation
values therefore do not exist in the shard and cannot be recovered from it.
They are written as ``NaN`` and the metadata says so, rather than duplicating
the symmetrised value into both columns, which would read as a measurement that
the two orientations agreed. Reported to the orchestrator as a gap between §5
and §6.2.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .ged_pair_index import pairs_from_indices
else:  # pragma: no cover - only when run as a bare script from eval_setup/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ged_pair_index import pairs_from_indices  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = [
    "SUBSAMPLE_KEYS",
    "SubsampleMergeError",
    "collect_subsample_shards",
    "main",
    "merge_subsample",
]

#: The flat schema of CONTRACTS §5.
SUBSAMPLE_KEYS = (
    "dataset_key",
    "pair_i",
    "pair_j",
    "n_max",
    "bin_index",
    "value",
    "value_fwd",
    "value_rev",
    "seconds",
    "metadata",
)

#: Arrays a CONTRACT C shard must carry.
_SHARD_KEYS = ("pair_index", "ged", "lb", "ub", "certified", "seconds")


class SubsampleMergeError(Exception):
    """Raised on an incomplete join, a malformed shard, or a missing pair list."""


@dataclass(slots=True)
class _ShardRows:
    """One shard's rows, resolved to graph-index pairs."""

    dataset_key: str
    pair_i: np.ndarray
    pair_j: np.ndarray
    value: np.ndarray
    seconds: np.ndarray
    meta: dict[str, Any]


def _code_commit() -> str:
    """Return the current commit hash, or ``'unknown'`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - git absent
        return "unknown"
    return out.stdout.strip() or "unknown"


def collect_subsample_shards(shard_dir: Path) -> list[Path]:
    """Find every CONTRACT C shard in a directory, whatever its dataset.

    Unlike the dense merge, this does not filter by key: the subsample campaign
    spans all ten datasets and its shards land together.

    Args:
        shard_dir: Directory to scan.

    Returns:
        Sorted shard paths.

    Raises:
        SubsampleMergeError: If the directory is missing or holds no shards.
    """
    if not shard_dir.is_dir():
        raise SubsampleMergeError(f"{shard_dir} is not a directory")
    found: list[Path] = []
    for path in sorted(shard_dir.glob("*.npz")):
        if path.name.endswith(".ckpt.npz"):
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                if all(k in data for k in _SHARD_KEYS):
                    found.append(path)
        except (OSError, ValueError) as exc:
            raise SubsampleMergeError(f"cannot read {path}: {exc}") from exc
    if not found:
        raise SubsampleMergeError(f"no CONTRACT C shards in {shard_dir}")
    return found


def _read_shard(path: Path) -> _ShardRows:
    """Read one shard and resolve its linear pair indices to ``(i, j)``.

    Args:
        path: Shard ``.npz``.

    Returns:
        The shard's rows.

    Raises:
        SubsampleMergeError: If the metadata does not name the dataset and its
            graph count, without which a linear index cannot be inverted.
    """
    with np.load(path, allow_pickle=False) as data:
        pair_index = np.asarray(data["pair_index"], dtype=np.int64)
        ub = np.asarray(data["ub"], dtype=np.float64)
        seconds = np.asarray(data["seconds"], dtype=np.float64)
        raw_meta = str(data["meta"]) if "meta" in data else "{}"
    try:
        meta = dict(json.loads(raw_meta))
    except (ValueError, TypeError) as exc:
        raise SubsampleMergeError(f"{path}: meta is not JSON") from exc

    key = meta.get("dataset")
    n_graphs = meta.get("n_graphs")
    if not key or not isinstance(n_graphs, int):
        raise SubsampleMergeError(
            f"{path}: meta must name 'dataset' and an integer 'n_graphs'; a linear pair "
            "index means nothing without the cohort it indexes"
        )
    i, j = pairs_from_indices(pair_index, int(n_graphs))
    return _ShardRows(
        dataset_key=str(key),
        pair_i=np.asarray(i, dtype=np.int64),
        pair_j=np.asarray(j, dtype=np.int64),
        value=ub,
        seconds=seconds,
        meta=meta,
    )


def _load_pair_list(path: Path) -> dict[str, np.ndarray]:
    """Read the frozen subsample pair list.

    Args:
        path: ``UB_TIGHT/subsample_pairs.npz``.

    Returns:
        Its arrays.

    Raises:
        SubsampleMergeError: If a required column is missing.
    """
    if not path.exists():
        raise SubsampleMergeError(f"{path} does not exist")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    for required in ("dataset_key", "pair_i", "pair_j"):
        if required not in arrays:
            raise SubsampleMergeError(f"{path} is not a §5 pair list: missing {required!r}")
    return arrays


def _bin_counts(bin_index: np.ndarray) -> dict[str, int]:
    """Return the realised row count per stratum.

    Args:
        bin_index: Per-row stratum index.

    Returns:
        Counts keyed by bin index as a string, so the dict is JSON-safe.
    """
    counter = Counter(int(b) for b in np.asarray(bin_index).ravel())
    return {str(k): int(counter[k]) for k in sorted(counter)}


def merge_subsample(
    *,
    shard_dir: Path,
    pair_list: Path,
    out: Path,
    role: str = "ubt",
    method: str | None = None,
    options: str | None = None,
) -> dict[str, Any]:
    """Join the shards onto the pair list and write the §5 flat file.

    Args:
        shard_dir: Directory holding the campaign's shards.
        pair_list: The frozen ``subsample_pairs.npz``.
        out: Output ``subsample.npz``.
        role: Role label for the metadata.
        method: GEDLIB method. Read from the shards when omitted.
        options: Verbatim option string. Read from the shards when omitted.

    Returns:
        The metadata written.

    Raises:
        SubsampleMergeError: If the join is not exact in both directions, or a
            joined value is not a usable bound.
    """
    listed = _load_pair_list(pair_list)
    keys = np.asarray(listed["dataset_key"]).astype(str)
    li = np.asarray(listed["pair_i"], dtype=np.int64)
    lj = np.asarray(listed["pair_j"], dtype=np.int64)
    n_rows = int(keys.size)

    shards = collect_subsample_shards(shard_dir)
    logger.info("joining %d shards onto %d listed pairs", len(shards), n_rows)

    computed: dict[tuple[str, int, int], tuple[float, float]] = {}
    shard_meta: list[dict[str, Any]] = []
    for path in shards:
        rows = _read_shard(path)
        shard_meta.append(rows.meta)
        for i, j, value, secs in zip(
            rows.pair_i.tolist(), rows.pair_j.tolist(), rows.value, rows.seconds, strict=True
        ):
            computed[(rows.dataset_key, int(i), int(j))] = (float(value), float(secs))

    wanted = {(keys[t], int(li[t]), int(lj[t])) for t in range(n_rows)}
    missing = wanted - set(computed)
    extra = set(computed) - wanted
    if missing or extra:
        raise SubsampleMergeError(
            f"the join is not exact: {len(missing)} listed pairs have no computed value "
            f"(e.g. {sorted(missing)[:3]}), {len(extra)} computed pairs are not on the list "
            f"(e.g. {sorted(extra)[:3]}). A partial join is undetectable downstream, because "
            "nothing later knows how many rows this file should have."
        )

    # Pair-list order, so the file is reproducible from seed 42 alone.
    value = np.empty(n_rows, dtype=np.float64)
    seconds = np.empty(n_rows, dtype=np.float32)
    for t in range(n_rows):
        v, s = computed[(keys[t], int(li[t]), int(lj[t]))]
        value[t] = v
        seconds[t] = s

    if n_rows and not np.isfinite(value).all():
        n_bad = int(np.count_nonzero(~np.isfinite(value)))
        raise SubsampleMergeError(
            f"{n_bad} joined values are not finite; an upper bound of inf is the signature "
            "of a method that does not set that end, and GEDLIB does not raise on it"
        )

    list_meta: dict[str, Any] = {}
    if "metadata" in listed:
        try:
            list_meta = dict(json.loads(str(listed["metadata"])))
        except (ValueError, TypeError):
            list_meta = {}

    def _agreed(field_name: str) -> Any:
        values = {
            str(m[field_name]) for m in shard_meta if field_name in m and m[field_name] is not None
        }
        return values.pop() if len(values) == 1 else (sorted(values) or None)

    bin_index = np.asarray(
        listed.get("bin_index", np.full(n_rows, -1, dtype=np.int8)), dtype=np.int8
    )
    n_max = np.asarray(listed.get("n_max", np.zeros(n_rows, dtype=np.int32)), dtype=np.int32)

    metadata: dict[str, Any] = {
        "role": role,
        "method": method if method is not None else _agreed("ub_method"),
        "options_string": options if options is not None else _agreed("ub_options"),
        "accessor": "upper",
        "cost_model": _agreed("cost_model") or "unit",
        "bin_edges": list_meta.get("bin_edges"),
        "seed": list_meta.get("seed"),
        "n_per_bin": list_meta.get("n_per_bin"),
        "n_pairs": n_rows,
        "n_datasets": int(np.unique(keys).size),
        "realised_per_bin": _bin_counts(bin_index),
        "seconds_total": float(np.sum(seconds, dtype=np.float64)),
        "mean_seconds_per_pair": float(np.mean(seconds)) if n_rows else 0.0,
        # CONTRACTS §6.2 freezes the shard at six arrays and ub is already
        # min(fwd, rev); the two orientations are not recoverable from it.
        "orientation_detail": (
            "not retained: CONTRACT C carries only the symmetrised ub, so value_fwd and "
            "value_rev are NaN. value is min(fwd, rev), computed in the backend."
        ),
        "n_shards": len(shards),
        "code_commit": _code_commit(),
        "computed_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            dataset_key=keys.astype(str),
            pair_i=li.astype(np.int32),
            pair_j=lj.astype(np.int32),
            n_max=n_max,
            bin_index=bin_index,
            value=value,
            value_fwd=np.full(n_rows, np.nan, dtype=np.float64),
            value_rev=np.full(n_rows, np.nan, dtype=np.float64),
            seconds=seconds,
            metadata=np.array(json.dumps(metadata)),
        )
    tmp.replace(out)

    with np.load(out, allow_pickle=False) as check:
        for required in SUBSAMPLE_KEYS:
            if required not in check:
                raise SubsampleMergeError(f"written file {out} is missing {required!r}")
    logger.info("wrote %s: %d rows across %d datasets", out, n_rows, metadata["n_datasets"])
    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser.

    Returns:
        The configured parser.
    """
    p = argparse.ArgumentParser(
        prog="approx_ged_subsample_merge",
        description="Join the ubt role's shards onto the frozen §5 subsample pair list.",
    )
    p.add_argument("--shards", required=True, help="directory holding the campaign's shards")
    p.add_argument("--pair-list", required=True, help="UB_TIGHT/subsample_pairs.npz")
    p.add_argument("--out", required=True, help="output UB_TIGHT/subsample.npz")
    p.add_argument("--role", default="ubt", help="role label for the metadata")
    p.add_argument("--method", default=None, help="GEDLIB method; read from the shards if omitted")
    p.add_argument("--options", default=None, help="verbatim option string")
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
        merge_subsample(
            shard_dir=Path(args.shards),
            pair_list=Path(args.pair_list),
            out=Path(args.out),
            role=str(args.role),
            method=args.method,
            options=args.options,
        )
    except (SubsampleMergeError, OSError) as exc:
        logger.error("subsample merge failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
