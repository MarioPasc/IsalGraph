"""``python -m isalgraph.competitors.smoke`` -- the frozen smoke CLI.

One command, run identically by all three agents locally and by the
orchestrator once on Picasso's loginexa node, so the timings are
attributable: one process produced them.

::

    python -m isalgraph.competitors.smoke \\
        --backends <comma-separated> --dataset <name> --n-graphs 200 \\
        --seed 42 --out <path.json>

The JSON schema is frozen in wave 0 and the entry point dispatches through
the registry, so an agent adding a backend extends this without editing it.

**What the Picasso run gates**: that ``pynauty`` builds from source under
gcc 12.2.0 where production will run.  A failure there takes
``nauty_graph6``, ``sparse6_nauty`` and AGM's orbit pruning down together.
The ``.so`` does not rsync, for the same reason the C++ engine does not.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

from isalgraph.competitors import datasets, fixtures
from isalgraph.competitors.base import Budget, ReprBackend, VectorBackend
from isalgraph.competitors.registry import (
    available_backends,
    get_backend,
    registered_backends,
)
from isalgraph.errors import CompetitorError

#: F3 protocol, frozen: 50 real graphs x 20 genuine relabellings, seed 42.
F3_GRAPHS = 50
F3_RELABELLINGS = 20


def _run_header(dataset: str, seed: int, n_graphs: int) -> dict[str, Any]:
    """Provenance every consumer of this JSON needs and nobody remembers to add."""
    import isalgraph

    versions: dict[str, str] = {}
    for module in ("networkx", "numpy", "rapidfuzz", "pynauty", "grakel"):
        try:
            versions[module] = getattr(__import__(module), "__version__", "unknown")
        except ImportError:
            versions[module] = "absent"
    return {
        "dataset": dataset,
        "seed": seed,
        "n_graphs_requested": n_graphs,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "packages": versions,
        # Recorded unconditionally: a timing quoted without the engine that
        # produced it is R1.1's complaint reproduced inside our answer to it.
        "isalgraph_engine": isalgraph.engine(),
        "isalgraph_build_hash": isalgraph.build_info().get("build_hash"),
        "registered_backends": list(registered_backends()),
    }


def _f3(
    backend: ReprBackend | VectorBackend,
    graphs: list[nx.Graph],
    seed: int,
    budget: Budget | None = None,
) -> tuple[int, int]:
    """How many of *graphs* encode identically under 20 fresh relabellings.

    The relabeller **rebuilds** each copy with a new insertion order.
    ``nx.relabel_nodes(copy=True)`` alone preserves insertion order and makes
    order-dependent formats look invariant, which would make this harness
    incapable of failing -- and a harness that cannot fail is worthless.

    A ``VectorBackend`` is fitted **once per graph family** -- the graph and
    all twenty of its relabellings together -- and never once per graph.
    Fitting per graph is the per-batch trap at batch size one: it builds a
    different colour vocabulary for each member and makes the comparison
    depend on the batching rather than on the structure.  It happens to be
    harmless for a WL whose features are fit-independent, and it would
    silently corrupt any future ``VectorBackend`` whose features are not.
    Reported by track C, 2026-08-15.

    Returns:
        ``(invariant, skipped)``.  **The skip count is not optional.**  A graph
        that fails to encode is not evidence of invariance, so it is skipped --
        but a skip and a genuine non-invariance both push the ratio down, and
        without the second number ``0/50`` reads identically whether the format
        is order-dependent or the harness never managed to call it.  That is
        not hypothetical: this function ignored *budget* on its first outing
        and reported ``isalgraph_pruned`` at ``0/50`` on Picasso, where the
        default 2 s budget is unenforceable and every single encode raised.
        The number was plausible, wrong, and silent, which is the exact failure
        the whole ticket is written against.
    """
    import random

    rng = random.Random(seed)
    invariant = 0
    skipped = 0
    for graph in graphs:
        try:
            copies = [fixtures.shuffled_copy(graph, rng) for _ in range(F3_RELABELLINGS)]
            if isinstance(backend, VectorBackend):
                backend.fit([graph, *copies])
            codes = {_signature(backend, g, budget) for g in (graph, *copies)}
        except Exception:  # noqa: BLE001 - see below
            # Broad by design. min-DFS raises a plain ValueError on a
            # disconnected graph, which is not a CompetitorError, so a
            # narrower clause let it escape and abort the whole smoke run --
            # every other backend's numbers lost to one graph the cohort
            # filter was supposed to exclude. run_backend records the failure
            # separately, and `skipped` makes it visible here too.
            skipped += 1
            continue
        if len(codes) == 1:
            invariant += 1
    return invariant, skipped


def _signature(
    backend: ReprBackend | VectorBackend, graph: nx.Graph, budget: Budget | None = None
) -> str:
    """A comparable string for one graph.  **Does not fit** -- see :func:`_f3`."""
    if isinstance(backend, VectorBackend):
        return json.dumps(sorted(backend.features(graph).items()))
    return "".join(backend.encode(graph, budget=budget).symbols)


def run_backend(
    name: str,
    graphs: list[nx.Graph],
    *,
    seed: int,
    dataset: str,
    unbudgeted_reference: bool = False,
) -> dict[str, Any]:
    """Encode every graph, timing each, recording every failure.

    **Every failure is recorded, never dropped.**  The failure *rate* is a
    reported number -- AGM's 24 % on GREC and min-DFS's 24/400 on
    Mutagenicity are results the paper prints.  A stated ceiling is a
    result; a silent one is a defect.
    """
    record: dict[str, Any] = {"backend": name, "available": False}
    try:
        backend = get_backend(name)
    except CompetitorError as exc:
        record["import_error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["available"] = True
    record["capabilities"] = sorted(c.value for c in backend.capabilities)

    if isinstance(backend, VectorBackend):
        # Fitted PER DATASET, never per batch: a per-batch fit produces a
        # different colour vocabulary and makes the distance matrix depend
        # on batching order.
        t0 = time.perf_counter()
        backend.fit(graphs)
        record["fit_s"] = time.perf_counter() - t0

    # The override is narrowed to the reference arm ON PURPOSE. Handing every
    # backend a Budget whose fields are None would read as "unbounded" to AGM
    # and min-DFS, silently discarding the 200k/100k and 50,000 caps that their
    # published failure rates were measured at. Only isalgraph_* reads
    # `timeout_s`, and only it is affected here.
    budget: Budget | None = None
    if unbudgeted_reference and name.startswith("isalgraph_"):
        budget = Budget(timeout_s=None)
        record["budget"] = "timeout_s=None -- UNBUDGETED, engine cannot enforce one"

    elapsed: list[float] = []
    failures: list[dict[str, Any]] = []
    n_encoded = 0
    bits_entropy: list[float] = []
    bits_realised: list[int] = []
    for idx, graph in enumerate(graphs):
        t0 = time.perf_counter()
        try:
            if isinstance(backend, VectorBackend):
                backend.features(graph)
            else:
                encoding = backend.encode(graph, budget=budget)
                try:
                    counted = backend.bits(encoding)
                    bits_entropy.append(counted.entropy_bits)
                    bits_realised.append(counted.realised_bits)
                except CompetitorError:
                    pass  # wl_subtree / size_null have no bit count, by design
            n_encoded += 1
        except Exception as exc:  # noqa: BLE001 - every failure is a datum
            failures.append(
                {
                    "graph_id": idx,
                    "n_nodes": graph.number_of_nodes(),
                    "exception": type(exc).__name__,
                    "elapsed_s": time.perf_counter() - t0,
                }
            )
            continue
        finally:
            elapsed.append(time.perf_counter() - t0)

    record["n_encoded"] = n_encoded
    record["n_failed"] = len(failures)
    record["failures"] = failures[:50]
    record["n_failures_recorded"] = len(failures)
    if elapsed:
        ms = sorted(x * 1e3 for x in elapsed)
        record["ms_per_graph"] = {
            "p50": statistics.median(ms),
            "p90": ms[min(len(ms) - 1, int(0.9 * len(ms)))],
            "max": ms[-1],
        }
    record["bits"] = {
        "entropy_p50": statistics.median(bits_entropy) if bits_entropy else None,
        "realised_p50": statistics.median(bits_realised) if bits_realised else None,
    }
    f3_graphs = graphs[:F3_GRAPHS]
    invariant, skipped = _f3(backend, f3_graphs, seed, budget)
    record["f3_invariant_of_50"] = f"{invariant}/{len(f3_graphs)}"
    # Never omitted. Without it a 0/50 cannot be told apart from a harness that
    # failed to call the backend at all -- which is how a wrong F3 shipped once.
    record["f3_skipped"] = skipped
    return record


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a non-zero code only on a harness failure."""
    parser = argparse.ArgumentParser(prog="python -m isalgraph.competitors.smoke")
    parser.add_argument(
        "--backends",
        default="",
        help="comma-separated backend names; empty means every available one",
    )
    parser.add_argument("--dataset", required=True, choices=list(datasets.ALL_DATASETS))
    parser.add_argument("--n-graphs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--unbudgeted-reference",
        action="store_true",
        help=(
            "run isalgraph_* with timeout_s=None. Needed where the C++ engine is "
            "absent, since timeout_s is cpp-only and the backend refuses a budget "
            "it cannot enforce rather than dropping it. Recorded in the JSON."
        ),
    )
    args = parser.parse_args(argv)

    cohort = datasets.load(args.dataset)
    indices = cohort.sample(args.n_graphs, seed=args.seed)
    graphs = [cohort.graphs[i] for i in indices]

    names = (
        [n.strip() for n in args.backends.split(",") if n.strip()]
        if args.backends
        else list(available_backends(include_baseline=True))
    )

    result: dict[str, Any] = {
        "header": _run_header(args.dataset, args.seed, args.n_graphs),
        "n_graphs_drawn": len(graphs),
        "suite": cohort.suite,
        "backends": {},
    }
    for name in names:
        result["backends"][name] = run_backend(
            name,
            graphs,
            seed=args.seed,
            dataset=args.dataset,
            unbudgeted_reference=args.unbudgeted_reference,
        )

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    for name, rec in result["backends"].items():
        if not rec.get("available"):
            print(f"{name:18s} UNAVAILABLE  {rec.get('import_error', '')}")
            continue
        ms = rec.get("ms_per_graph", {})
        print(
            f"{name:18s} ok={rec['n_encoded']:4d} failed={rec['n_failed']:3d} "
            f"p50={ms.get('p50', float('nan')):8.3f}ms F3={rec['f3_invariant_of_50']}"
            + (f"  SKIPPED={rec['f3_skipped']}" if rec.get("f3_skipped") else "")
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
