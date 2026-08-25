"""How far does the *exhaustive* canonical string reach on the C++ engine?

``isalgraph_canonical`` carries ``Capability.SUITE1_ONLY`` and refuses above
``n = 12``. That guard was set from a measurement at a **2 s** budget --
``isalgraph_ref.py`` records 207/400 COIL-DEL, 118/400 Mutagenicity and
300/400 Protein timing out there. T-06's production budget is **300 s**, and
the campaign runs on the C++ engine. So the guard may be far more conservative
than the encoder now requires, and the paper's main arm
(``isalgraph_pruned``) is a length-suboptimal canonical form
(``t06_pruned_vs_exhaustive.py``): every stratum where the exhaustive form is
reachable is a stratum where the compactness figures could be tightened.

This script measures the real ceiling. For each node count it draws graphs
from the frozen T-06 cohorts, runs ``canonical_string`` (exhaustive, whichever
engine is active) under a per-graph timeout in a **subprocess**, and reports
the completion rate and the time distribution. A killed subprocess is the only
safe way to bound a recursive search: the encoder does not poll a deadline
between branches, so an in-process timeout cannot interrupt it.

The pruned arm is timed on the same graphs, so the cost ratio is measured
rather than inferred.

Usage::

    python .claude/notes/review/tasks/t06_exhaustive_ceiling.py \\
        --budget 30 --per-n 12 --n-max 30
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
import statistics
import sys
import time
from multiprocessing.queues import Queue
from typing import Any, Final

LOGGER: Final = logging.getLogger(__name__)

#: Cohorts to draw from. Suite 2 is where the interesting sizes live; Suite 1
#: is included so the n <= 12 rows can be checked against the campaign, which
#: already encoded every one of them.
SOURCES: Final[tuple[tuple[str, str], ...]] = (
    ("suite1", "aids"),
    ("suite2", "aids_iam"),
    ("suite2", "mutagenicity"),
    ("suite2", "protein"),
    ("suite2", "coil_del"),
    ("suite2", "grec"),
)


def _worker(edges: list[tuple[int, int]], n: int, variant: str, out: Queue[Any]) -> None:
    """Encode one graph and put ``(length, seconds)`` on *out*.

    Args:
        edges: Edge list on ``range(n)``.
        n: Node count.
        variant: ``canonical`` or ``pruned``.
        out: A queue the parent reads.
    """
    from isalgraph import SparseGraph, canonical_string, pruned_canonical_string

    graph = SparseGraph(n, False)
    for _ in range(n):
        graph.add_node()
    for u, v in edges:
        graph.add_edge(u, v)
    encoder = canonical_string if variant == "canonical" else pruned_canonical_string
    start = time.perf_counter()
    text = encoder(graph)
    out.put((len(text), time.perf_counter() - start))


def encode_with_budget(
    edges: list[tuple[int, int]], n: int, variant: str, budget: float
) -> tuple[int, float] | None:
    """Run one encode in a subprocess killed after *budget* seconds.

    Args:
        edges: Edge list.
        n: Node count.
        variant: ``canonical`` or ``pruned``.
        budget: Wall-clock seconds before the child is killed.

    Returns:
        ``(symbol count, seconds)``, or ``None`` when the budget was exceeded.
    """
    queue: Queue[Any] = mp.Queue()
    child = mp.Process(target=_worker, args=(edges, n, variant, queue), daemon=True)
    child.start()
    child.join(budget)
    if child.is_alive():
        child.terminate()
        child.join()
        return None
    return queue.get() if not queue.empty() else None


def sample(per_n: int, n_max: int, seed: int) -> dict[int, list[list[tuple[int, int]]]]:
    """Draw up to *per_n* graphs per node count from the frozen cohorts.

    Args:
        per_n: Graphs to keep per node count.
        n_max: Largest node count to sample.
        seed: RNG seed; the draw is reproducible.

    Returns:
        ``{n: [edge list, ...]}``.
    """
    from benchmarks.real_data.eval_encoding.t06_cohort import load_cohort

    rng = random.Random(seed)
    pools: dict[int, list[list[tuple[int, int]]]] = {}
    for suite, dataset in SOURCES:
        try:
            cohort = load_cohort(suite, dataset)
        except Exception as exc:  # noqa: BLE001 - a missing export is not fatal here
            LOGGER.warning("skipping %s/%s: %s", suite, dataset, exc)
            continue
        for count, edges in zip(cohort.node_counts, cohort.edge_lists, strict=True):
            n = int(count)
            if not 3 <= n <= n_max:
                continue
            pools.setdefault(n, []).append([(int(u), int(v)) for u, v in edges])
    return {n: rng.sample(v, min(per_n, len(v))) for n, v in sorted(pools.items())}


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=30.0, help="seconds per graph")
    ap.add_argument("--per-n", type=int, default=12, help="graphs per node count")
    ap.add_argument("--n-max", type=int, default=30, help="largest node count to try")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None, help="write results as JSON here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import isalgraph

    LOGGER.info("engine=%s build=%s", isalgraph.engine(), isalgraph.build_info().get("build_hash"))
    if isalgraph.engine() != "cpp":
        LOGGER.error("engine is not cpp; a ceiling measured on pure Python is not the one we want")
        return 1

    drawn = sample(args.per_n, args.n_max, args.seed)
    rows: list[dict[str, Any]] = []
    print(
        f"\n{'n':>4} {'graphs':>7} {'exh ok':>7} {'exh %':>7} "
        f"{'med s':>9} {'max s':>9} {'pruned med s':>13} {'ratio':>7} "
        f"{'len exh/pruned':>15} {'hybrid saving':>14}"
    )
    for n, edge_lists in drawn.items():
        exhaustive = [encode_with_budget(e, n, "canonical", args.budget) for e in edge_lists]
        pruned = [encode_with_budget(e, n, "pruned", args.budget) for e in edge_lists]
        paired = list(zip(exhaustive, pruned, strict=True))
        ok = [(a, b) for a, b in paired if a and b]
        # The production arm would be a D14 hybrid: exhaustive where it lands
        # inside the budget, pruned where it does not. That keeps the column
        # complete -- which is the whole reason the SUITE1_ONLY guard exists --
        # and, since the pruned form is never shorter, it is a conservative
        # upper bound on the true canonical length. Measure what it buys over
        # the pure pruned arm on the SAME graphs, timeouts included.
        hybrid = [(a[0] if a else b[0], b[0]) for a, b in paired if b]
        saving = (
            100.0 * (1.0 - sum(h for h, _ in hybrid) / sum(q for _, q in hybrid)) if hybrid else 0.0
        )
        done = len(ok)
        rate = 100.0 * done / len(edge_lists)
        if done:
            e_times = [a[1] for a, _ in ok]
            p_times = [b[1] for _, b in ok]
            e_len = [a[0] for a, _ in ok]
            p_len = [b[0] for _, b in ok]
            print(
                f"{n:>4} {len(edge_lists):>7} {done:>7} {rate:>6.0f}% "
                f"{statistics.median(e_times):>9.3f} {max(e_times):>9.3f} "
                f"{statistics.median(p_times):>13.4f} "
                f"{statistics.median(e_times) / max(statistics.median(p_times), 1e-9):>7.0f} "
                f"{statistics.mean(e_len):>7.1f}/{statistics.mean(p_len):<7.1f} "
                f"{saving:>13.1f}%"
            )
            rows.append(
                {
                    "n": n,
                    "graphs": len(edge_lists),
                    "exhaustive_ok": done,
                    "exhaustive_rate": rate,
                    "median_s": statistics.median(e_times),
                    "max_s": max(e_times),
                    "pruned_median_s": statistics.median(p_times),
                    "mean_len_exhaustive": statistics.mean(e_len),
                    "mean_len_pruned": statistics.mean(p_len),
                    "hybrid_saving_pct": saving,
                }
            )
        else:
            print(f"{n:>4} {len(edge_lists):>7} {0:>7} {0:>6.0f}%   -- all exceeded the budget --")
            rows.append(
                {"n": n, "graphs": len(edge_lists), "exhaustive_ok": 0, "exhaustive_rate": 0.0}
            )
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"budget_s": args.budget, "rows": rows}, handle, indent=1)
        LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
