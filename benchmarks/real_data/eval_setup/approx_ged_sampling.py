"""Emit the frozen seed-42 size-stratified subsample pair list -- CONTRACTS section 5.

Why this exists
---------------
`IPFP_MS` is the tight upper bound in T-05's sensitivity arm and costs roughly 808 ms/pair at
n_bar = 29.51 (T-27 section 5). Running it over all 21,710,892 Suite-2 pairs is not
affordable, so it runs on a subsample. The subsample answers exactly one question -- *how much
tightness does the frozen `BIPARTITE` gate cost, as a function of n, in the regime AE.1 disputes* --
and its design is frozen here, **before the run**, so the answer cannot be shaped by the result.

The design, verbatim from CONTRACTS section 5
----------------------------------------------
Stratum is the bin of ``max(n1, n2)`` under 14 right-open bins with edges
``[2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 99)``; bin membership is
``np.searchsorted(edges, n, side="right") - 1``. The draw is **uniform within bin, pooled across all
ten datasets, without replacement, seed 42**, taking ``min(2000, bin_population)`` per non-empty
bin. The ceiling is therefore 14 x 2000 = 28,000 pairs.

**This is a size-stratified sample, not a random sample of Suite-2 pairs.** It deliberately
over-weights large ``n``: a proportional sample would be dominated by Letter and COIL-DEL below
n = 25 and would say nothing about n = 98. Every figure derived from it is reported per bin and
never pooled into a cohort-level mean. That constraint is recorded in the emitted metadata so it
travels with the numbers.

What this module writes, and what it does not
----------------------------------------------
``subsample_pairs.npz``
    The pooled pair list -- ``dataset_key, pair_i, pair_j, n_max, bin_index, pair_index`` -- emitted
    ahead of the run. CONTRACTS section 5 amendment 3.
``pair_lists/{key}.npz``
    One per dataset with a single ``pair_index`` key of ascending linear indices, the format
    ``ged_exact_runner.py:794 --pair-list`` requires.

It does **not** write ``UB_TIGHT/subsample.npz``. That is the runner's output and carries the
``value``/``value_fwd``/``value_rev``/``seconds`` columns this module cannot know.

Determinism
-----------
Reproducibility is over **array content**, not file bytes: ``np.savez_compressed`` stamps each zip
member with the local time, so two byte-identical exports produce two different files.
:func:`content_digest` is the check that "the sample reproduces" actually means, and
``--verify-reproducible`` runs the whole draw twice in one process and compares it.

The RNG stream is consumed in a fixed order -- bins ascending, one ``Generator.choice`` per
non-empty bin, always called even when the whole bin is taken -- so the draw is a pure function of
the seed, the pool order and the bin populations. The pool order is the dataset order of
:data:`export_graphs_suite2.SUITE2_DATASETS` and, within a dataset, ``np.triu_indices(N, 1)`` order.

Usage
-----
``python -m benchmarks.real_data.eval_setup.approx_ged_sampling``
``python -m benchmarks.real_data.eval_setup.approx_ged_sampling --verify-only``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_graphs import _git_commit  # noqa: E402
from export_graphs_suite2 import (  # noqa: E402
    DEFAULT_EXPORT_DIR,
    SUITE2_DATASETS,
    TOTAL_EXPECTED_PAIRS,
)
from ged_pair_index import indices_of_pairs  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Right-open bin edges on ``max(n1, n2)``. 15 edges -> 14 bins; the last is ``[80, 99)``.
BIN_EDGES: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 99)
N_BINS = len(BIN_EDGES) - 1

#: Frozen by CONTRACTS section 5. Deliberately **not** exposed on the CLI: a sampling seed that can
#: be changed from the command line is a sampling seed that will be, and the whole point of freezing
#: the design before the run is that the draw cannot be re-rolled after seeing the result.
SEED = 42
MAX_PER_BIN = 2000
MAX_TOTAL_PAIRS = N_BINS * MAX_PER_BIN

_SANDISK = "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph"
DEFAULT_OUT_DIR = f"{_SANDISK}/data/source/APPROX_GED/UB_TIGHT"

PAIR_LIST_SUBDIR = "pair_lists"
SUBSAMPLE_NAME = "subsample_pairs.npz"

#: Explicit width for the ``dataset_key`` column. ``np.full(size, key, dtype=np.str_)`` silently
#: yields ``<U1`` and truncates every key to its first character, which merged the three Letter
#: datasets under ``'i'`` and both AIDS cohorts under ``'a'`` with no error raised. Measured on the
#: real export 2026-08-13. The width is derived from the registry so a longer key cannot reintroduce
#: it, and :func:`_check_dataset_keys` asserts the round trip.
_KEY_DTYPE = f"<U{max(len(key) for key in SUITE2_DATASETS)}"

#: Recorded in the metadata so the constraint travels with the numbers.
STRATIFICATION_WARNING = (
    "size-stratified sample, not a random sample of Suite-2 pairs; "
    "report per bin, never pooled into a cohort-level mean"
)


class SamplingError(Exception):
    """The subsample cannot be drawn, or violates CONTRACTS section 5."""


@dataclass(frozen=True, slots=True)
class DatasetPairs:
    """Every upper-triangle pair of one dataset, with its stratum.

    Attributes
    ----------
    key : str
        Dataset key.
    n_graphs : int
        Graph count, defining the ``triu_indices(n_graphs, 1)`` enumeration.
    pair_i, pair_j : np.ndarray
        ``int32 (P,)`` with ``pair_i < pair_j``, in ``triu_indices`` order.
    n_max : np.ndarray
        ``int32 (P,)`` equal to ``max(n_nodes[i], n_nodes[j])``.
    bin_index : np.ndarray
        ``int8 (P,)`` stratum per :func:`bin_of`.
    """

    key: str
    n_graphs: int
    pair_i: np.ndarray
    pair_j: np.ndarray
    n_max: np.ndarray
    bin_index: np.ndarray


@dataclass
class Subsample:
    """The drawn pair list, pooled across datasets.

    Attributes
    ----------
    dataset_key : np.ndarray
        ``<U (P,)`` dataset each pair belongs to.
    pair_i, pair_j : np.ndarray
        ``int32 (P,)`` indices into that dataset's exported graph order, ``pair_i < pair_j``.
    n_max : np.ndarray
        ``int32 (P,)``.
    bin_index : np.ndarray
        ``int8 (P,)``.
    pair_index : np.ndarray
        ``int64 (P,)`` linear upper-triangle index within its dataset, the key
        ``ged_exact_runner.py --pair-list`` consumes.
    bin_population : dict[int, int]
        Pool size per bin, over all ten datasets.
    bin_drawn : dict[int, int]
        Realised draw per bin.
    bin_population_by_dataset : dict[str, dict[int, int]]
        Pool size per dataset per bin. Requested by the orchestrator on 2026-08-13 to size the
        Picasso jobs from a measured distribution rather than an n_bar-based projection.
    """

    dataset_key: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    n_max: np.ndarray
    bin_index: np.ndarray
    pair_index: np.ndarray
    bin_population: dict[int, int] = field(default_factory=dict)
    bin_drawn: dict[int, int] = field(default_factory=dict)
    bin_population_by_dataset: dict[str, dict[int, int]] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.pair_i.shape[0])


def bin_of(n: np.ndarray | int) -> np.ndarray:
    """Return the stratum of ``max(n1, n2)`` values.

    Parameters
    ----------
    n : np.ndarray or int
        Node counts.

    Returns
    -------
    np.ndarray
        ``int8`` bin indices. A value below the first edge yields ``-1`` and a value at or above the
        last edge yields ``N_BINS``; both are outside the design and are rejected by
        :func:`_check_bins` rather than silently binned.
    """
    values = np.asarray(n)
    edges = np.asarray(BIN_EDGES)
    return (np.searchsorted(edges, values, side="right") - 1).astype(np.int8)


def _check_bins(key: str, bins: np.ndarray, n_max: np.ndarray) -> None:
    """Reject any pair whose ``max(n1, n2)`` falls outside the 14 bins.

    The design covers ``2 <= n < 99``. Suite 2's largest graph has 98 nodes and its smallest has 2,
    so every pair must land in a bin; one that does not means the cohort changed under the frozen
    design and the sample would silently omit it.

    Raises
    ------
    SamplingError
        If any bin index is outside ``[0, N_BINS)``.
    """
    bad = np.flatnonzero((bins < 0) | (bins >= N_BINS))
    if bad.size:
        offending = np.unique(n_max[bad])[:10]
        raise SamplingError(
            f"{key}: {bad.size} pair(s) fall outside the {N_BINS} frozen bins "
            f"[{BIN_EDGES[0]}, {BIN_EDGES[-1]}); offending max(n1,n2) values {offending.tolist()}"
        )


def _check_dataset_keys(dataset_key: np.ndarray) -> None:
    """Reject a ``dataset_key`` column that does not hold whole registry keys.

    A truncated unicode dtype does not raise; it silently rewrites ``"iam_letter_low"`` as ``"i"``
    and collapses three datasets into one label, which downstream reads as a valid pair list against
    the wrong graphs. This is a measured failure, not a hypothetical one -- see :data:`_KEY_DTYPE`.

    Raises
    ------
    SamplingError
        If any entry is not a key of :data:`export_graphs_suite2.SUITE2_DATASETS`.
    """
    observed = {str(value) for value in np.unique(dataset_key)}
    unknown = observed - set(SUITE2_DATASETS)
    if unknown:
        raise SamplingError(
            f"dataset_key holds {sorted(unknown)}, which are not Suite-2 keys; "
            f"column dtype is {dataset_key.dtype} (truncation?)"
        )


def read_node_counts(export_dir: str | Path, key: str) -> np.ndarray:
    """Read one exported dataset's ``n_nodes`` array.

    Only ``n_nodes`` is needed, so the graphs are not rebuilt -- :func:`export_graphs.load_exported`
    would reconstruct 16,370 ``nx.Graph`` objects this module never touches.

    Raises
    ------
    SamplingError
        If the file is missing or has no ``n_nodes``.
    """
    path = Path(export_dir) / f"{key}.npz"
    if not path.is_file():
        raise SamplingError(f"exported dataset not found: {path}")
    with np.load(path, allow_pickle=False) as handle:
        if "n_nodes" not in handle:
            raise SamplingError(f"{path} has no 'n_nodes' key")
        return handle["n_nodes"].astype(np.int32, copy=True)


def build_pairs(key: str, n_nodes: np.ndarray) -> DatasetPairs:
    """Enumerate every upper-triangle pair of one dataset with its stratum.

    Memory is the reason for the explicit casts: ``np.triu_indices`` returns ``int64``, which for
    Mutagenicity's 8,158,780 pairs is 130 MB for the two arrays alone. Graph counts are far below
    ``2**31`` so ``int32`` is lossless, and the ``int64`` originals are released immediately.

    Raises
    ------
    SamplingError
        If any pair falls outside the frozen bins.
    """
    n_graphs = int(n_nodes.shape[0])
    i64, j64 = np.triu_indices(n_graphs, k=1)
    pair_i = i64.astype(np.int32)
    pair_j = j64.astype(np.int32)
    del i64, j64

    n_max = np.maximum(n_nodes[pair_i], n_nodes[pair_j]).astype(np.int32)
    bins = bin_of(n_max)
    _check_bins(key, bins, n_max)

    return DatasetPairs(
        key=key,
        n_graphs=n_graphs,
        pair_i=pair_i,
        pair_j=pair_j,
        n_max=n_max,
        bin_index=bins,
    )


def draw(
    pools: list[DatasetPairs],
    seed: int = SEED,
    max_per_bin: int = MAX_PER_BIN,
) -> Subsample:
    """Draw the size-stratified subsample.

    For each bin in ascending order the pooled members are gathered in dataset order and, within a
    dataset, in ``triu_indices`` order; ``min(max_per_bin, population)`` of them are drawn uniformly
    without replacement; the chosen positions are then **sorted**, so the emitted order is pool
    order and carries no information about the draw.

    ``Generator.choice`` is called even when the whole bin is taken. Skipping it would make the RNG
    stream depend on which bins happen to be small, which is a reproducibility hazard for no
    benefit.

    Parameters
    ----------
    pools : list[DatasetPairs]
        One per dataset, in the frozen dataset order.
    seed : int, optional
        Frozen at :data:`SEED`. Parameterised for tests only.
    max_per_bin : int, optional
        Frozen at :data:`MAX_PER_BIN`. Parameterised for tests only.

    Returns
    -------
    Subsample
        Pairs sorted by ``(dataset order, pair_index)``, with realised and pooled bin counts.

    Raises
    ------
    SamplingError
        If the drawn total exceeds the design ceiling.
    """
    rng = np.random.default_rng(seed)

    population: dict[int, int] = {}
    drawn: dict[int, int] = {}
    by_dataset: dict[str, dict[int, int]] = {}
    for pool in pools:
        counts = np.bincount(pool.bin_index.astype(np.int64), minlength=N_BINS)
        by_dataset[pool.key] = {b: int(counts[b]) for b in range(N_BINS)}
    for b in range(N_BINS):
        population[b] = sum(by_dataset[pool.key][b] for pool in pools)

    # Per bin, the pooled member positions as (dataset ordinal, position within that dataset).
    chosen_rows: list[tuple[int, np.ndarray]] = []
    for b in range(N_BINS):
        pop = population[b]
        if pop == 0:
            drawn[b] = 0
            continue
        take = min(max_per_bin, pop)
        picks = np.sort(rng.choice(pop, size=take, replace=False))
        drawn[b] = int(take)

        # Map pooled ranks back to (dataset, within-dataset position) without materialising the
        # concatenated pool: the pool is the datasets' members laid end to end in dataset order.
        offset = 0
        for d, pool in enumerate(pools):
            members = np.flatnonzero(pool.bin_index == b)
            lo, hi = offset, offset + members.size
            local = picks[(picks >= lo) & (picks < hi)] - lo
            if local.size:
                chosen_rows.append((d, members[local]))
            offset = hi

    total = sum(int(rows.size) for _, rows in chosen_rows)
    if total > MAX_TOTAL_PAIRS:
        raise SamplingError(f"drew {total} pairs, design ceiling is {MAX_TOTAL_PAIRS}")
    if total != sum(drawn.values()):
        raise SamplingError(f"drew {total} pairs, per-bin counts sum to {sum(drawn.values())}")

    # Concatenate in (dataset order, ascending position) so the emitted file is canonical.
    per_dataset: dict[int, list[np.ndarray]] = {}
    for d, rows in chosen_rows:
        per_dataset.setdefault(d, []).append(rows)

    keys: list[np.ndarray] = []
    out_i: list[np.ndarray] = []
    out_j: list[np.ndarray] = []
    out_n: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    out_k: list[np.ndarray] = []
    for d, pool in enumerate(pools):
        if d not in per_dataset:
            continue
        rows = np.sort(np.concatenate(per_dataset[d]))
        i_sel = pool.pair_i[rows]
        j_sel = pool.pair_j[rows]
        keys.append(np.full(rows.size, pool.key, dtype=_KEY_DTYPE))
        out_i.append(i_sel)
        out_j.append(j_sel)
        out_n.append(pool.n_max[rows])
        out_b.append(pool.bin_index[rows])
        out_k.append(
            indices_of_pairs(i_sel.astype(np.int64), j_sel.astype(np.int64), pool.n_graphs)
        )

    empty_u = np.empty(0, dtype=_KEY_DTYPE)
    dataset_key = np.concatenate(keys) if keys else empty_u
    _check_dataset_keys(dataset_key)
    return Subsample(
        dataset_key=dataset_key,
        pair_i=np.concatenate(out_i).astype(np.int32) if out_i else np.empty(0, np.int32),
        pair_j=np.concatenate(out_j).astype(np.int32) if out_j else np.empty(0, np.int32),
        n_max=np.concatenate(out_n).astype(np.int32) if out_n else np.empty(0, np.int32),
        bin_index=np.concatenate(out_b).astype(np.int8) if out_b else np.empty(0, np.int8),
        pair_index=np.concatenate(out_k).astype(np.int64) if out_k else np.empty(0, np.int64),
        bin_population=population,
        bin_drawn=drawn,
        bin_population_by_dataset=by_dataset,
    )


def content_digest(sample: Subsample) -> str:
    """Return a sha256 over the drawn pair list, independent of zip framing.

    ``np.savez_compressed`` stamps each member with the local time, so two byte-identical draws
    produce two different files. This digest is a function of the data alone, which is what
    "the sample reproduces from seed 42" has to mean.
    """
    digest = hashlib.sha256()
    for value in sample.dataset_key:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\x00")
    for arr in (sample.pair_i, sample.pair_j, sample.n_max, sample.bin_index, sample.pair_index):
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def build_metadata(
    sample: Subsample,
    export_dir: str | Path,
    seed: int,
    max_per_bin: int,
) -> dict[str, object]:
    """Assemble the JSON metadata recorded beside the pair list."""
    return {
        "bin_edges": list(BIN_EDGES),
        "n_bins": N_BINS,
        "seed": seed,
        "max_per_bin": max_per_bin,
        "n_pairs": len(sample),
        "max_total_pairs": MAX_TOTAL_PAIRS,
        "n_per_bin": {str(b): sample.bin_drawn.get(b, 0) for b in range(N_BINS)},
        "bin_population": {str(b): sample.bin_population.get(b, 0) for b in range(N_BINS)},
        "bin_population_by_dataset": {
            key: {str(b): counts.get(b, 0) for b in range(N_BINS)}
            for key, counts in sample.bin_population_by_dataset.items()
        },
        "datasets": list(SUITE2_DATASETS),
        "pool_total_pairs": sum(sample.bin_population.values()),
        "stratification": STRATIFICATION_WARNING,
        "bin_rule": 'np.searchsorted(BIN_EDGES, max(n1, n2), side="right") - 1',
        "export_dir": str(export_dir),
        "content_sha256": content_digest(sample),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": _git_commit(),
        "schema_version": SCHEMA_VERSION,
    }


def write_subsample(
    sample: Subsample, metadata: dict[str, object], out_dir: str | Path
) -> tuple[Path, list[Path]]:
    """Write the pooled pair list and one runner-consumable list per dataset.

    Returns
    -------
    tuple
        ``(pooled_path, per_dataset_paths)``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pooled = out / SUBSAMPLE_NAME
    np.savez_compressed(
        pooled,
        dataset_key=sample.dataset_key,
        pair_i=sample.pair_i,
        pair_j=sample.pair_j,
        n_max=sample.n_max,
        bin_index=sample.bin_index,
        pair_index=sample.pair_index,
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    logger.info("Wrote %s (%d pairs)", pooled, len(sample))

    list_dir = out / PAIR_LIST_SUBDIR
    list_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key in SUITE2_DATASETS:
        mask = sample.dataset_key == key
        if not bool(mask.any()):
            continue
        path = list_dir / f"{key}.npz"
        np.savez_compressed(
            path,
            pair_index=np.sort(sample.pair_index[mask]).astype(np.int64),
            metadata=np.array(json.dumps({"dataset": key, **metadata}, sort_keys=True)),
        )
        written.append(path)
        logger.info("Wrote %s (%d pairs)", path, int(mask.sum()))
    return pooled, written


def run(
    export_dir: str | Path,
    seed: int = SEED,
    max_per_bin: int = MAX_PER_BIN,
) -> Subsample:
    """Read the exported datasets and draw the subsample.

    Raises
    ------
    SamplingError
        If a dataset is missing, a pair falls outside the frozen bins, or the pooled pair total
        disagrees with the locked cohort.
    """
    pools: list[DatasetPairs] = []
    for key in SUITE2_DATASETS:
        n_nodes = read_node_counts(export_dir, key)
        pool = build_pairs(key, n_nodes)
        logger.info("%-16s %d graphs, %d pairs", key, pool.n_graphs, pool.pair_i.shape[0])
        pools.append(pool)

    pooled_pairs = sum(int(p.pair_i.shape[0]) for p in pools)
    if pooled_pairs != TOTAL_EXPECTED_PAIRS:
        raise SamplingError(
            f"pool holds {pooled_pairs} pairs, locked cohort is {TOTAL_EXPECTED_PAIRS}"
        )
    return draw(pools, seed=seed, max_per_bin=max_per_bin)


def main(argv: list[str] | None = None) -> int:
    """Draw and write the subsample pair list. Returns a process exit status."""
    parser = argparse.ArgumentParser(description="Emit the frozen seed-42 subsample pair list.")
    parser.add_argument(
        "--export-dir", default=DEFAULT_EXPORT_DIR, help="exported_suite2 directory"
    )
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="destination directory")
    parser.add_argument("--verify-only", action="store_true", help="draw and report, write nothing")
    parser.add_argument(
        "--verify-reproducible",
        action="store_true",
        help="draw twice in one process and compare the content digests",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        sample = run(args.export_dir)
    except SamplingError as exc:
        logger.error("%s", exc)
        return 1

    metadata = build_metadata(sample, args.export_dir, SEED, MAX_PER_BIN)
    logger.info(
        "Drew %d pairs over %d non-empty bins (ceiling %d)",
        len(sample),
        sum(1 for b in range(N_BINS) if sample.bin_drawn.get(b, 0) > 0),
        MAX_TOTAL_PAIRS,
    )
    for b in range(N_BINS):
        logger.info(
            "  bin %2d [%3d, %3d): population %9d, drawn %5d",
            b,
            BIN_EDGES[b],
            BIN_EDGES[b + 1],
            sample.bin_population.get(b, 0),
            sample.bin_drawn.get(b, 0),
        )

    if args.verify_reproducible:
        try:
            again = run(args.export_dir)
        except SamplingError as exc:
            logger.error("%s", exc)
            return 1
        if content_digest(again) != content_digest(sample):
            logger.error("the draw is NOT reproducible: two runs disagree")
            return 1
        logger.info("draw reproduces: content_sha256 %s", content_digest(sample))

    if args.verify_only:
        logger.info("verify-only: nothing written")
        return 0

    write_subsample(sample, metadata, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
