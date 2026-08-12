"""Stage-1 pair sampling for the AIDS exact-GED run (T-03), implemented verbatim.

The design is **pre-registered**: it is fixed in
``.claude/notes/2026-08-12-exact-ged/CONTRACTS.md`` §8 and
``.claude/notes/review/tasks/T-03-design.md`` §4, both written before any
production pair was computed, so the choice between a stage-1 and a stage-2
estimate cannot be made after seeing either. This module implements that text and
nothing else. A change here is a change to a pre-registered analysis and needs a
dated changelog entry in the design note.

Three components, in this order::

    core    simple random sample of K = 180 graphs (seed 42); ALL C(180, 2) pairs
    halo    for each of the N - K non-core graphs, q = 10 partners drawn
            uniformly without replacement from the other N - 1 graphs
    top-up  every NON-EMPTY pair-stratum holding fewer than f = 30 sampled pairs
            is filled to min(f, |stratum|) by uniform draw without replacement
            from that stratum's not-yet-sampled members

The core is a **simple** random sample, not a stratified one. That is deliberate:
it makes the core-block Spearman rho exactly unbiased for the population rho and
makes the D2 graph-level cluster bootstrap exact on a complete induced submatrix.
Stratification applies only to the halo and the top-up, whose job is coverage
rather than estimation.

Pair strata, AIDS-internal:

* size cell -- unordered pair of node-count bins over ``{2-5, 6-9, 10-12}``, 6 cells;
* density cell -- unordered pair of density quintiles, 15 cells;
* stratum -- the cross product, 90 cells.

Quintile edges come from ``np.quantile(density, [.2, .4, .6, .8])`` and are applied
with ``np.searchsorted(..., side="right")``, so ties fall consistently and a density
exactly equal to an edge lands in the **upper** bin. CONTRACTS §8 fixes the quantiles
and the use of ``searchsorted`` but left the side unspecified; it was disambiguated to
``"right"`` on 2026-08-12, before any production pair was computed, because under
``"left"`` the top quintile is unreachable whenever the 80th percentile equals the
maximum density -- which happens on the real AIDS cohort, where every ``n = 2`` graph
has density exactly 1.0. The mirror case is worth knowing: ``"right"`` would empty the
*bottom* quintile if the 20th percentile equalled the minimum density, which needs at
least a fifth of the corpus sharing one exact density value and does not occur here.

Whether a stratum is empty is judged on the **population**, never on the sample; empty
strata are reported as empty and are never topped up.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Importable both as ``python -m benchmarks.real_data.eval_setup.ged_sampling`` from
# the repository root (the way the SLURM worker invokes it) and as a bare module
# from inside ``eval_setup/``. The package branch is preferred so that
# ``ged_pair_index`` is not loaded twice under two different module identities.
if __package__:
    from .ged_pair_index import (
        GedPairIndexError,
        index_of_pair,
        indices_of_pairs,
        n_pairs,
        pairs_from_indices,
    )
else:  # pragma: no cover - only when run as a bare script from eval_setup/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ged_pair_index import (  # noqa: E402
        GedPairIndexError,
        index_of_pair,
        indices_of_pairs,
        n_pairs,
        pairs_from_indices,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_F",
    "DEFAULT_K",
    "DEFAULT_Q",
    "DEFAULT_SEED",
    "GedSamplingError",
    "PairStrata",
    "Stage1Sample",
    "main",
    "stage1_sample",
]

DEFAULT_K = 180
DEFAULT_Q = 10
DEFAULT_F = 30
DEFAULT_SEED = 42

#: Node-count bin edges. ``np.searchsorted(_SIZE_EDGES, n, side="right")`` sends
#: 2-5 to 0, 6-9 to 1 and 10-12 to 2.
_SIZE_EDGES = np.array([6, 10], dtype=np.int64)
_N_SIZE_BINS = 3
_N_DENSITY_BINS = 5
_SIZE_MIN = 2
_SIZE_MAX = 12

#: Quantile cut points for the density quintiles, per CONTRACTS §8.
_QUANTILES = (0.2, 0.4, 0.6, 0.8)


class GedSamplingError(Exception):
    """Raised when the sampling parameters or the population are inadmissible."""


def _unordered_pair_id(a: np.ndarray, b: np.ndarray, m: int) -> np.ndarray:
    """Index an unordered pair of bins ``{a, b}`` drawn from ``m`` bins.

    Enumerates the upper triangle *including* the diagonal, so ``{a, a}`` is a
    valid cell: for ``m = 3`` the ids run ``(0,0)=0 (0,1)=1 (0,2)=2 (1,1)=3
    (1,2)=4 (2,2)=5``.

    Args:
        a: Integer array of bin indices.
        b: Integer array of bin indices, same shape.
        m: Number of bins.

    Returns:
        ``int64`` array of cell ids in ``[0, m*(m+1)//2)``.
    """
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    return lo * np.int64(m) - lo * (lo - np.int64(1)) // np.int64(2) + (hi - lo)


@dataclass(frozen=True, slots=True)
class PairStrata:
    """The pair-stratum assignment of an entire population of graphs.

    Attributes:
        n_graphs: Number of graphs in the population.
        size_bin: ``int64 (n_graphs,)`` node-count bin per graph, in ``[0, 3)``.
        density: ``float64 (n_graphs,)`` graph density ``2m / (n(n-1))``.
        density_bin: ``int64 (n_graphs,)`` density quintile per graph, in ``[0, 5)``.
        quantile_edges: ``float64 (4,)`` the quintile cut points actually used.
        stratum: ``int64 (C(n_graphs, 2),)`` stratum id per pair, indexed by the
            linear upper-triangle index ``k``.
        population_counts: ``int64 (90,)`` number of population pairs per stratum.
    """

    n_graphs: int
    size_bin: np.ndarray
    density: np.ndarray
    density_bin: np.ndarray
    quantile_edges: np.ndarray
    stratum: np.ndarray
    population_counts: np.ndarray

    @property
    def n_strata(self) -> int:
        """Total number of strata, empty ones included."""
        return int(self.population_counts.size)


def _describe_stratum(stratum_id: int) -> dict[str, object]:
    """Decompose a stratum id back into its size and density cells.

    Args:
        stratum_id: Id in ``[0, 90)``.

    Returns:
        A JSON-ready description naming both cells.
    """
    n_density_cells = _N_DENSITY_BINS * (_N_DENSITY_BINS + 1) // 2
    size_cell, density_cell = divmod(stratum_id, n_density_cells)
    size_pair = _cell_to_pair(size_cell, _N_SIZE_BINS)
    density_pair = _cell_to_pair(density_cell, _N_DENSITY_BINS)
    labels = ["2-5", "6-9", "10-12"]
    return {
        "stratum_id": stratum_id,
        "size_cell": size_cell,
        "size_bins": list(size_pair),
        "size_labels": [labels[size_pair[0]], labels[size_pair[1]]],
        "density_cell": density_cell,
        "density_quintiles": list(density_pair),
    }


def _cell_to_pair(cell: int, m: int) -> tuple[int, int]:
    """Invert :func:`_unordered_pair_id` for a single cell id.

    Args:
        cell: Cell id in ``[0, m*(m+1)//2)``.
        m: Number of bins.

    Returns:
        The ``(lo, hi)`` bin pair with ``lo <= hi``.

    Raises:
        GedSamplingError: If ``cell`` does not decode.
    """
    for lo in range(m):
        start = lo * m - lo * (lo - 1) // 2
        width = m - lo
        if start <= cell < start + width:
            return lo, lo + (cell - start)
    raise GedSamplingError(f"cell id {cell} is out of range for m={m}")


def build_pair_strata(n_nodes: np.ndarray, n_edges: np.ndarray) -> PairStrata:
    """Assign every population pair to its stratum.

    Args:
        n_nodes: ``(N,)`` node count per graph. Every entry must lie in ``[2, 12]``,
            the range the frozen size bins cover.
        n_edges: ``(N,)`` edge count per graph.

    Returns:
        The :class:`PairStrata` for the whole population.

    Raises:
        GedSamplingError: If the arrays disagree in length, if fewer than two
            graphs are supplied, or if a node count falls outside ``[2, 12]``.
    """
    nn = np.asarray(n_nodes, dtype=np.int64).ravel()
    ne = np.asarray(n_edges, dtype=np.int64).ravel()
    if nn.shape != ne.shape:
        raise GedSamplingError(f"n_nodes and n_edges disagree: {nn.shape} vs {ne.shape}")
    n = int(nn.size)
    if n < 2:
        raise GedSamplingError(f"need at least 2 graphs, got {n}")
    if int(nn.min()) < _SIZE_MIN or int(nn.max()) > _SIZE_MAX:
        raise GedSamplingError(
            f"node counts must lie in [{_SIZE_MIN}, {_SIZE_MAX}] -- the frozen size bins "
            f"{{2-5, 6-9, 10-12}} cover nothing else -- got "
            f"[{int(nn.min())}, {int(nn.max())}]"
        )

    size_bin = np.searchsorted(_SIZE_EDGES, nn, side="right").astype(np.int64)
    density = 2.0 * ne.astype(np.float64) / (nn.astype(np.float64) * (nn.astype(np.float64) - 1.0))
    edges = np.quantile(density, list(_QUANTILES))
    # side="right" sends a density equal to an edge into the UPPER bin, which keeps
    # the top quintile reachable when q80 equals the maximum density. That case is
    # not hypothetical: after min_nodes=2 the AIDS cohort contains n=2 graphs, whose
    # density is exactly 1.0 by construction, so under side="left" every one of them
    # would fall to bin 3 and quintile 4 would be empty on the real cohort.
    density_bin = np.searchsorted(edges, density, side="right").astype(np.int64)

    i, j = np.triu_indices(n, k=1)
    # np.triu_indices enumerates the triangle in exactly the order the linear
    # index defines, so position p in these arrays is pair k = p. Assert it rather
    # than rely on it: everything downstream is indexed by k.
    if n <= 4096:
        expected = np.arange(n_pairs(n), dtype=np.int64)
        if not np.array_equal(indices_of_pairs(i, j, n), expected):
            raise GedSamplingError("np.triu_indices order does not match the linear pair index")

    size_cell = _unordered_pair_id(size_bin[i], size_bin[j], _N_SIZE_BINS)
    density_cell = _unordered_pair_id(density_bin[i], density_bin[j], _N_DENSITY_BINS)
    n_density_cells = _N_DENSITY_BINS * (_N_DENSITY_BINS + 1) // 2
    n_size_cells = _N_SIZE_BINS * (_N_SIZE_BINS + 1) // 2
    stratum = size_cell * np.int64(n_density_cells) + density_cell
    counts = np.bincount(stratum, minlength=n_size_cells * n_density_cells).astype(np.int64)

    return PairStrata(
        n_graphs=n,
        size_bin=size_bin,
        density=density,
        density_bin=density_bin,
        quantile_edges=np.asarray(edges, dtype=np.float64),
        stratum=stratum,
        population_counts=counts,
    )


@dataclass(frozen=True, slots=True)
class Stage1Sample:
    """The frozen stage-1 pair sample and everything the report has to record.

    Attributes:
        pair_index: ``int64`` linear pair indices, ascending and deduplicated.
        core_graphs: ``int64`` the ``K`` graphs of the simple random core sample.
        core_pairs: ``int64`` the ``C(K, 2)`` core-block pair indices, ascending.
        halo_pairs: ``int64`` pair indices contributed by the halo and not already
            in the core block.
        topup_pairs: ``int64`` pair indices contributed by the top-up.
        strata: The population stratum assignment used.
        sampled_counts: ``int64 (90,)`` sampled pairs per stratum.
        graphs_covered: Number of distinct graphs appearing in at least one pair.
        params: The ``K``, ``q``, ``f`` and seed actually used.
    """

    pair_index: np.ndarray
    core_graphs: np.ndarray
    core_pairs: np.ndarray
    halo_pairs: np.ndarray
    topup_pairs: np.ndarray
    strata: PairStrata
    sampled_counts: np.ndarray
    graphs_covered: int
    params: dict[str, int] = field(default_factory=dict)

    @property
    def n_pairs_sampled(self) -> int:
        """Total number of distinct pairs selected."""
        return int(self.pair_index.size)


def stage1_sample(
    n_nodes: np.ndarray,
    n_edges: np.ndarray,
    *,
    k_core: int = DEFAULT_K,
    q_halo: int = DEFAULT_Q,
    f_topup: int = DEFAULT_F,
    seed: int = DEFAULT_SEED,
) -> Stage1Sample:
    """Draw the frozen stage-1 sample.

    Args:
        n_nodes: ``(N,)`` node count per graph.
        n_edges: ``(N,)`` edge count per graph.
        k_core: Size of the simple random core sample of graphs.
        q_halo: Partners drawn for each non-core graph.
        f_topup: Floor on sampled pairs per non-empty stratum.
        seed: RNG seed. The design fixes it at 42.

    Returns:
        The :class:`Stage1Sample`.

    Raises:
        GedSamplingError: If the parameters are inadmissible for this population,
            or if the resulting sample fails to cover every graph -- the coverage
            requirement is what distinguishes this design from a plain core sample
            and a silent failure of it would invalidate the stage-1 claim.
    """
    strata = build_pair_strata(n_nodes, n_edges)
    n = strata.n_graphs
    if not (2 <= k_core <= n):
        raise GedSamplingError(f"K must satisfy 2 <= K <= N = {n}, got {k_core}")
    if not (0 <= q_halo <= n - 1):
        raise GedSamplingError(f"q must satisfy 0 <= q <= N - 1 = {n - 1}, got {q_halo}")
    if f_topup < 0:
        raise GedSamplingError(f"f must be non-negative, got {f_topup}")

    total = n_pairs(n)
    rng = np.random.default_rng(seed)
    selected = np.zeros(total, dtype=bool)

    # --- core: simple random sample of graphs, complete induced pair block ----
    core_graphs = np.sort(rng.choice(n, size=k_core, replace=False).astype(np.int64))
    ci, cj = np.triu_indices(k_core, k=1)
    core_pairs = np.sort(indices_of_pairs(core_graphs[ci], core_graphs[cj], n))
    selected[core_pairs] = True

    # --- halo: q partners for every non-core graph, so all N graphs appear -----
    in_core = np.zeros(n, dtype=bool)
    in_core[core_graphs] = True
    non_core = np.flatnonzero(~in_core).astype(np.int64)
    halo_hits: list[int] = []
    all_graphs = np.arange(n, dtype=np.int64)
    for u in non_core:  # ascending, so the RNG stream is reproducible
        others = np.delete(all_graphs, u)
        partners = rng.choice(others, size=q_halo, replace=False)
        for v in partners:
            lo, hi = (int(u), int(v)) if u < v else (int(v), int(u))
            halo_hits.append(index_of_pair(lo, hi, n))
    halo_arr = np.asarray(sorted(set(halo_hits)), dtype=np.int64)
    halo_pairs = halo_arr[~selected[halo_arr]] if halo_arr.size else halo_arr
    selected[halo_arr] = True

    # --- top-up: raise every non-empty stratum to min(f, |stratum|) ------------
    order = np.argsort(strata.stratum, kind="stable")
    bounds = np.concatenate(([0], np.cumsum(strata.population_counts))).astype(np.int64)
    topup_hits: list[np.ndarray] = []
    for s in range(strata.n_strata):  # ascending stratum id, for reproducibility
        pop = int(strata.population_counts[s])
        if pop == 0:
            continue  # empty on the POPULATION: reported as empty, never topped up
        members = order[bounds[s] : bounds[s + 1]]
        have = int(np.count_nonzero(selected[members]))
        target = min(f_topup, pop)
        if have >= target:
            continue
        free = members[~selected[members]]
        picked = rng.choice(free, size=target - have, replace=False)
        selected[picked] = True
        topup_hits.append(np.sort(picked.astype(np.int64)))
    topup_pairs = np.concatenate(topup_hits) if topup_hits else np.empty(0, dtype=np.int64)

    pair_index = np.flatnonzero(selected).astype(np.int64)
    sampled_counts = np.bincount(strata.stratum[pair_index], minlength=strata.n_strata).astype(
        np.int64
    )

    pi, pj = pairs_from_indices(pair_index, n)
    covered = int(np.unique(np.concatenate([pi, pj])).size)
    if covered != n:
        raise GedSamplingError(
            f"stage-1 sample covers {covered} of {n} graphs; the design requires all of them. "
            f"With q={q_halo} every non-core graph should receive at least one partner."
        )

    return Stage1Sample(
        pair_index=pair_index,
        core_graphs=core_graphs,
        core_pairs=core_pairs,
        halo_pairs=halo_pairs,
        topup_pairs=topup_pairs,
        strata=strata,
        sampled_counts=sampled_counts,
        graphs_covered=covered,
        params={"K": k_core, "q": q_halo, "f": f_topup, "seed": seed},
    )


def sampling_report(sample: Stage1Sample, *, dataset: str) -> dict[str, object]:
    """Build the ``sampling_report.json`` payload.

    Args:
        sample: The drawn sample.
        dataset: Dataset key, e.g. ``"aids"``.

    Returns:
        A JSON-serialisable dict recording the parameters, the per-stratum
        population and sampled counts, the distinct-graph coverage and the totals.
    """
    strata = sample.strata
    rows: list[dict[str, object]] = []
    for s in range(strata.n_strata):
        pop = int(strata.population_counts[s])
        row = _describe_stratum(s)
        row["population_pairs"] = pop
        row["sampled_pairs"] = int(sample.sampled_counts[s])
        row["empty"] = pop == 0
        row["floor"] = min(int(sample.params["f"]), pop)
        row["meets_floor"] = pop == 0 or int(sample.sampled_counts[s]) >= min(
            int(sample.params["f"]), pop
        )
        rows.append(row)

    non_empty = [r for r in rows if not r["empty"]]
    return {
        "dataset": dataset,
        "design": "T-03 stage-1, pre-registered in CONTRACTS.md section 8",
        "K": int(sample.params["K"]),
        "q": int(sample.params["q"]),
        "f": int(sample.params["f"]),
        "seed": int(sample.params["seed"]),
        "n_graphs": strata.n_graphs,
        "n_population_pairs": n_pairs(strata.n_graphs),
        "n_sampled_pairs": sample.n_pairs_sampled,
        "n_core_graphs": int(sample.core_graphs.size),
        "n_core_pairs": int(sample.core_pairs.size),
        "n_halo_pairs_new": int(sample.halo_pairs.size),
        "n_topup_pairs": int(sample.topup_pairs.size),
        "graphs_covered": sample.graphs_covered,
        "graphs_covered_is_complete": sample.graphs_covered == strata.n_graphs,
        "n_strata": strata.n_strata,
        "n_strata_non_empty": len(non_empty),
        "n_strata_empty": strata.n_strata - len(non_empty),
        "all_non_empty_strata_meet_floor": all(bool(r["meets_floor"]) for r in non_empty),
        "density_quantile_edges": [float(x) for x in strata.quantile_edges],
        "strata": rows,
    }


def _load_counts(path: str | Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Read node and edge counts from a CONTRACT A ``.npz``.

    A deliberately minimal reader for the three keys the sampler needs. It will be
    replaced by ``export_graphs.load_exported`` once that module lands; until then
    the sampler must not depend on a module that does not exist yet.

    Args:
        path: Path to the CONTRACT A ``.npz``.

    Returns:
        ``(n_nodes, n_edges, dataset_key)``.

    Raises:
        GedSamplingError: If a required key is missing.
    """
    with np.load(path, allow_pickle=False) as data:
        for key in ("n_nodes", "n_edges"):
            if key not in data:
                raise GedSamplingError(f"{path} is not a CONTRACT A file: missing '{key}'")
        n_nodes = np.asarray(data["n_nodes"], dtype=np.int64)
        n_edges = np.asarray(data["n_edges"], dtype=np.int64)
        dataset = ""
        if "metadata" in data:
            try:
                dataset = str(json.loads(str(data["metadata"])).get("dataset", ""))
            except (ValueError, TypeError):
                dataset = ""
    return n_nodes, n_edges, dataset


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        prog="ged_sampling",
        description="Draw the frozen T-03 stage-1 pair sample (CONTRACTS section 8).",
    )
    p.add_argument("--input", required=True, help="CONTRACT A .npz for the dataset")
    p.add_argument("--out-pairs", required=True, help="output pair_list.npz (key 'pair_index')")
    p.add_argument("--out-report", required=True, help="output sampling_report.json")
    p.add_argument("-K", "--k-core", type=int, default=DEFAULT_K, help="core graphs (default 180)")
    p.add_argument("-q", "--q-halo", type=int, default=DEFAULT_Q, help="halo partners (default 10)")
    p.add_argument(
        "-f", "--f-topup", type=int, default=DEFAULT_F, help="stratum floor (default 30)"
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="RNG seed (design fixes it at 42)"
    )
    p.add_argument(
        "--expect-graphs",
        type=int,
        default=None,
        help="assert the population has exactly this many graphs (AIDS: 769)",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on any inadmissible input or failed
        invariant.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        n_nodes, n_edges, dataset = _load_counts(args.input)
        if args.expect_graphs is not None and int(n_nodes.size) != int(args.expect_graphs):
            raise GedSamplingError(
                f"population has {int(n_nodes.size)} graphs, expected {int(args.expect_graphs)}"
            )
        sample = stage1_sample(
            n_nodes,
            n_edges,
            k_core=args.k_core,
            q_halo=args.q_halo,
            f_topup=args.f_topup,
            seed=args.seed,
        )
        report = sampling_report(sample, dataset=dataset or Path(args.input).stem)
    except (GedSamplingError, GedPairIndexError) as exc:
        logger.error("stage-1 sampling failed: %s", exc)
        return 1

    out_pairs = Path(args.out_pairs)
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with out_pairs.open("wb") as fh:
        np.savez_compressed(fh, pair_index=sample.pair_index)
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "stage 1: %d pairs over %d/%d graphs (core %d, halo +%d, top-up +%d), "
        "%d/%d strata non-empty",
        sample.n_pairs_sampled,
        sample.graphs_covered,
        sample.strata.n_graphs,
        int(sample.core_pairs.size),
        int(sample.halo_pairs.size),
        int(sample.topup_pairs.size),
        report["n_strata_non_empty"],
        sample.strata.n_strata,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
