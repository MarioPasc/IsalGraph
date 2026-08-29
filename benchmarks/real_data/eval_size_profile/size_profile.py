"""Compute Spearman rho against GED within equal-``n`` strata.

One row per ``(suite, dataset, representation, reference, n)``. The figures in
:mod:`benchmarks.real_data.eval_size_profile.figures` consume this table and add
no statistics of their own.

The bootstrap resamples **graphs**, never pairs: rho moved by up to 0.07 between
two independent 200-graph draws, so the effective sample size is governed by
graphs and a pair-level interval is wrong by construction.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy import stats

LOGGER: Final = logging.getLogger(__name__)

#: T-28's alternative similarity references, loaded only when this points at a
#: built tree. Unset -- the default -- and :func:`_load_reference` returns exactly
#: what it returned for T-06, so the published size profile reproduces.
#:
#: Layout: ``{T28_REFERENCE_ROOT}/{suite}/{dataset}__{key}.npz``, dense CONTRACTS
#: section 4 schema. Same contract as ``t06_f2_inputs.T28_REFERENCE_ROOT``.
T28_REFERENCE_ROOT: Final[str] = os.environ.get("T28_REFERENCE_ROOT", "")

FloatArray = npt.NDArray[np.float64]

#: Each representation's selected primary distance, from T-04a's ``grid.py``.
#: ``None`` means no admissible distance passed F1-F4, so the representation is
#: k-excluded and carries no Claim-B cell.
PRIMARY_DISTANCE: Final[dict[str, str | None]] = {
    "adjacency": None,
    "graph6": None,
    "sparse6": None,
    "size_null": None,
    "agm_cam": "levenshtein",
    "isalgraph_canonical": "levenshtein",
    "isalgraph_pruned": "levenshtein",
    "min_dfs": "levenshtein",
    "nauty_graph6": "levenshtein",
    "sparse6_nauty": "levenshtein",
    "wl_subtree": "kernel",
}

#: A stratum needs this many usable pairs before a rho is reported. Below it the
#: rank correlation is dominated by which handful of graphs happened to land in
#: the size bin.
MIN_PAIRS: Final[int] = 30

#: Graph-level bootstrap replicates per stratum.
N_BOOTSTRAP: Final[int] = 1000

SEED: Final[int] = 42


class SizeProfileError(Exception):
    """Raised when an input violates a frozen contract."""


@dataclass
class StratumRow:
    """One ``(suite, dataset, representation, reference, n)`` result.

    Attributes:
        suite: ``suite1`` or ``suite2``.
        dataset: Dataset key.
        representation: Backend name.
        metric: The primary distance actually used.
        reference: ``exact``, ``lb`` or ``ub``.
        n: The common node count defining the stratum.
        n_graphs: Graphs in the stratum with a usable encoding and reference.
        n_pairs: Usable pairs.
        rho: Spearman rho of distance against reference GED.
        ci_lo: Lower 95 % graph-level bootstrap bound.
        ci_hi: Upper 95 % graph-level bootstrap bound.
        p_value: Two-sided Spearman p-value, uncorrected.
        mean_distance: Mean representation distance over the stratum's pairs.
        mean_reference: Mean reference GED over the same pairs.
        zero_fraction: Fraction of pairs the representation calls identical.
    """

    suite: str
    dataset: str
    representation: str
    metric: str
    reference: str
    n: int
    n_graphs: int
    n_pairs: int
    rho: float | None
    ci_lo: float | None
    ci_hi: float | None
    p_value: float | None
    mean_distance: float | None
    mean_reference: float | None
    zero_fraction: float | None
    arm: str = "primary"
    n_censored_in_stratum: int = 0


def _symbols(encoding: npt.NDArray[Any], separator: str) -> list[tuple[str, ...]]:
    """Split stored encodings into symbol sequences per CONTRACTS 3.1.

    Args:
        encoding: Stored encoding strings.
        separator: ``metadata.symbol_sep``; empty when symbols are characters.

    Returns:
        One symbol tuple per graph.
    """
    if separator:
        return [tuple(text.split(separator)) for text in encoding]
    return [tuple(text) for text in encoding]


def _levenshtein_block(seqs: list[tuple[str, ...]]) -> FloatArray:
    """Symbol-level Levenshtein distances within one stratum.

    Args:
        seqs: Symbol sequences for the stratum's graphs.

    Returns:
        Square distance matrix.
    """
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    return np.asarray(
        cdist(seqs, seqs, scorer=Levenshtein.distance, dtype=np.float64), dtype=np.float64
    )


def _wl_counts(seqs: list[tuple[str, ...]]) -> FloatArray:
    """Rebuild the WL count matrix from stored encodings.

    The stored encoding is the ``symbol_sep``-joined multiset of WL colours, so
    counting occurrences over the dataset-wide colour set reproduces the count
    vectors ``wl_driver.feature_table`` fits, provided the vocabulary is taken
    over the whole dataset --- which it is, because the encoding campaign fitted
    per dataset.

    Args:
        seqs: Symbol sequences for every graph in the dataset.

    Returns:
        ``(G, V)`` count matrix over the dataset vocabulary.
    """
    vocabulary = sorted({symbol for seq in seqs for symbol in seq})
    index = {colour: j for j, colour in enumerate(vocabulary)}
    counts = np.zeros((len(seqs), len(vocabulary)), dtype=np.float64)
    for i, seq in enumerate(seqs):
        for colour, count in Counter(seq).items():
            counts[i, index[colour]] = float(count)
    return counts


def _kernel_block(counts: FloatArray) -> FloatArray:
    """Kernel-induced distance ``sqrt(K_ii + K_jj - 2 K_ij)``.

    Mirrors ``wl_driver.kernel_distance_matrix`` exactly, including the refusal
    to normalise: normalisation would divide out the graph-size signal GED
    depends on.

    Args:
        counts: ``(G, V)`` WL feature counts for the stratum.

    Returns:
        Symmetric distance matrix with an exactly zero diagonal.
    """
    gram = counts @ counts.T
    diagonal = np.diagonal(gram)
    squared = diagonal[:, None] + diagonal[None, :] - 2.0 * gram
    np.maximum(squared, 0.0, out=squared)
    matrix = np.sqrt(squared)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _bootstrap_ci(
    distance: FloatArray,
    reference: FloatArray,
    valid: npt.NDArray[np.bool_],
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval for rho, resampling graphs.

    Args:
        distance: Square distance matrix for the stratum.
        reference: Square reference-GED matrix for the same graphs.
        valid: Square mask of usable pairs.
        rng: Seeded generator.

    Returns:
        ``(ci_lo, ci_hi)``, or ``(None, None)`` when too few replicates are
        defined.
    """
    size = distance.shape[0]
    replicates: list[float] = []
    for _ in range(N_BOOTSTRAP):
        take = rng.integers(0, size, size)
        sub_valid = valid[np.ix_(take, take)]
        # Drop the resampled diagonal: a graph paired with its own duplicate is
        # a zero-distance pair the original stratum never contained.
        same = take[:, None] == take[None, :]
        sub_valid = sub_valid & ~same
        upper = np.triu(np.ones_like(sub_valid), k=1).astype(bool) & sub_valid
        if int(upper.sum()) < 3:
            continue
        x = distance[np.ix_(take, take)][upper]
        y = reference[np.ix_(take, take)][upper]
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        replicates.append(float(stats.spearmanr(x, y).statistic))
    if len(replicates) < N_BOOTSTRAP // 10:
        return None, None
    return (
        float(np.percentile(replicates, 2.5)),
        float(np.percentile(replicates, 97.5)),
    )


def _load_reference(
    suite: str, dataset: str, ged_root: Path, approx_root: Path
) -> dict[str, tuple[FloatArray, npt.NDArray[Any]]]:
    """Load the reference GED matrices for one dataset.

    Args:
        suite: ``suite1`` or ``suite2``.
        dataset: Dataset key.
        ged_root: Directory of Suite-1 exact matrices.
        approx_root: ``APPROX_GED`` root holding ``LB/``.

    Returns:
        Mapping from reference name to ``(matrix, graph_ids)``.
    """
    out: dict[str, tuple[FloatArray, npt.NDArray[Any]]] = {}
    if suite == "suite1":
        path = ged_root / f"{dataset}.npz"
        if path.exists():
            with np.load(path, allow_pickle=True) as z:
                out["exact"] = (
                    np.asarray(z["ged_matrix"], dtype=np.float64),
                    np.asarray(z["graph_ids"]).astype(str),
                )
    else:
        path = approx_root / "LB" / f"{dataset}.npz"
        if path.exists():
            with np.load(path, allow_pickle=True) as z:
                ids = np.asarray(z["graph_ids"]).astype(str)
                out["lb"] = (np.asarray(z["lb_matrix"], dtype=np.float64), ids)
                out["ub"] = (np.asarray(z["ub_matrix"], dtype=np.float64), ids)
    out.update(_load_t28_references(suite, dataset))
    return out


def _load_t28_references(
    suite: str, dataset: str
) -> dict[str, tuple[FloatArray, npt.NDArray[Any]]]:
    """Load T-28's alternative similarity references, if a tree is configured.

    Returns an empty mapping when :data:`T28_REFERENCE_ROOT` is unset, which is
    the default and reproduces the T-06 size profile exactly.

    Args:
        suite: Suite key.
        dataset: Dataset key.

    Returns:
        Mapping from reference name to ``(matrix, graph_ids)``.
    """
    if not T28_REFERENCE_ROOT:
        return {}
    out: dict[str, tuple[FloatArray, npt.NDArray[Any]]] = {}
    for path in sorted((Path(T28_REFERENCE_ROOT) / suite).glob(f"{dataset}__*.npz")):
        key = path.stem.split("__", 1)[1]
        with np.load(path, allow_pickle=True) as z:
            out[key] = (
                np.asarray(z["distance_matrix"], dtype=np.float64),
                np.asarray(z["graph_ids"]).astype(str),
            )
    return out


def profile_cell(
    encodings: Path,
    references: dict[str, tuple[FloatArray, npt.NDArray[Any]]],
    *,
    arm: str = "primary",
) -> list[StratumRow]:
    """Compute every stratum row for one ``(dataset, representation)`` cell.

    Args:
        encodings: Path to ``{dataset}__{representation}.npz``.
        references: Reference matrices keyed by name, from :func:`_load_reference`.
        arm: ``primary`` keeps the D14 censored graphs, which enter with a
            greedy-min fallback string rather than the canonical one;
            ``complete_case`` keeps only ``status == "ok"``. The pair exists
            because a fallback string is **not canonical** and sits outside the
            completeness theorem, so a stratum where half the arm is fallback is
            measuring the 300 s budget as much as the representation. Comparing
            the two arms on identical strata is what separates the two.

    Returns:
        One row per stratum and reference.

    Raises:
        SizeProfileError: If the encoding file is missing a required column.
    """
    suite = encodings.parent.name
    dataset, representation = encodings.stem.split("__", 1)
    metric = PRIMARY_DISTANCE.get(representation)
    if metric is None:
        return []

    with np.load(encodings, allow_pickle=True) as z:
        missing = {"graph_ids", "encoding", "length", "status", "node_counts", "metadata"} - set(
            z.files
        )
        if missing:
            raise SizeProfileError(f"{encodings} lacks {sorted(missing)}")
        meta = json.loads(str(z["metadata"]))
        ids = np.asarray(z["graph_ids"]).astype(str)
        status = np.asarray(z["status"]).astype(str)
        length = np.asarray(z["length"]).astype(np.int64)
        node_counts = np.asarray(z["node_counts"]).astype(np.int64)
        seqs = _symbols(np.asarray(z["encoding"]).astype(str), str(meta.get("symbol_sep", "")))

    censored = status == "censored"
    if arm == "complete_case":
        usable = (status == "ok") & (length >= 0)
    else:
        usable = ((status == "ok") | censored) & (length >= 0)
    counts = _wl_counts(seqs) if metric == "kernel" else None

    rows: list[StratumRow] = []
    rng = np.random.default_rng(SEED)
    for reference_name, (matrix, ref_ids) in sorted(references.items()):
        position = {gid: j for j, gid in enumerate(ref_ids)}
        if not set(ids) <= set(position):
            LOGGER.warning(
                "%s/%s: graph_ids do not join onto %s", dataset, representation, reference_name
            )
            continue
        # Join on graph_ids, never positionally: aids is 769 in Suite 1 against
        # aids_graphedx's 819 in Suite 2 (F-12).
        perm = np.array([position[g] for g in ids])
        for n in sorted({int(v) for v in node_counts}):
            idx = np.flatnonzero(usable & (node_counts == n))
            if idx.size < 2 or idx.size * (idx.size - 1) // 2 < MIN_PAIRS:
                continue
            ref_block = matrix[np.ix_(perm[idx], perm[idx])]
            valid = np.isfinite(ref_block)
            upper = np.triu(np.ones_like(valid), k=1).astype(bool) & valid
            if int(upper.sum()) < MIN_PAIRS:
                continue
            if metric == "kernel":
                assert counts is not None
                dist = _kernel_block(counts[idx])
            else:
                dist = _levenshtein_block([seqs[i] for i in idx])
            x = dist[upper]
            y = ref_block[upper]
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                rho = ci_lo = ci_hi = p_value = None
            else:
                result = stats.spearmanr(x, y)
                rho = float(result.statistic)
                p_value = float(result.pvalue)
                ci_lo, ci_hi = _bootstrap_ci(dist, ref_block, valid, rng)
            rows.append(
                StratumRow(
                    suite=suite,
                    dataset=dataset,
                    representation=representation,
                    metric=metric,
                    reference=reference_name,
                    n=n,
                    n_graphs=int(idx.size),
                    n_pairs=int(upper.sum()),
                    rho=rho,
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                    p_value=p_value,
                    mean_distance=float(x.mean()),
                    mean_reference=float(y.mean()),
                    zero_fraction=float((x == 0).mean()),
                    arm=arm,
                    n_censored_in_stratum=int(censored[idx].sum()),
                )
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encodings", type=Path, required=True, help="the encodings/ tree")
    ap.add_argument("--ged-root", type=Path, required=True, help="Suite-1 exact matrices")
    ap.add_argument("--approx-root", type=Path, required=True, help="APPROX_GED root")
    ap.add_argument("--out", type=Path, required=True, help="size_profile.json")
    ap.add_argument("--suite", choices=("suite1", "suite2"), default=None)
    ap.add_argument(
        "--arm",
        choices=("primary", "complete_case", "both"),
        default="primary",
        help="both emits each stratum twice, so the D14 fallback confound is measurable",
    )
    ap.add_argument("--dataset", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows: list[StratumRow] = []
    suites = [args.suite] if args.suite else ["suite1", "suite2"]
    for suite in suites:
        directory = args.encodings / suite
        if not directory.is_dir():
            continue
        datasets = sorted({p.stem.split("__", 1)[0] for p in directory.glob("*.npz")})
        for dataset in datasets:
            if args.dataset and dataset != args.dataset:
                continue
            references = _load_reference(suite, dataset, args.ged_root, args.approx_root)
            if not references:
                LOGGER.warning("%s/%s: no reference GED, skipped", suite, dataset)
                continue
            for path in sorted(directory.glob(f"{dataset}__*.npz")):
                arms = ("primary", "complete_case") if args.arm == "both" else (args.arm,)
                new: list[StratumRow] = []
                for arm in arms:
                    new.extend(profile_cell(path, references, arm=arm))
                rows.extend(new)
                if new:
                    LOGGER.info(
                        "%s/%-16s %-20s %3d strata",
                        suite,
                        dataset,
                        path.stem.split("__", 1)[1],
                        len(new),
                    )

    payload = {
        "schema_version": "t06.size_profile.2",
        "ticket": "T-06",
        "descriptive": True,
        "note": (
            "Equal-n strata: within a stratum |n_i - n_j| is identically 0, so the size null is "
            "undefined and raw rho is the structural signal. NOT a pre-registered family."
        ),
        "arm": args.arm,
        "arm_note": (
            "A censored graph enters the primary arm with its greedy-min FALLBACK string, "
            "which is not canonical and sits outside the completeness theorem. Where the "
            "censoring rate is high the primary arm therefore measures the 300 s budget as "
            "well as the representation. The complete_case arm removes those graphs; the "
            "difference between the arms on identical strata is the budget's contribution."
        ),
        "min_pairs": MIN_PAIRS,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "rows": [asdict(r) for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    LOGGER.info("wrote %s (%d rows)", args.out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
