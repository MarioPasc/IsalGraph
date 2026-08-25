"""The canonicalisation ablation: what dropping the canonical search costs.

Answers R1.2's *"why must a representation be canonical"* **using our own
encoder** rather than a competitor's. Three arms, one instruction set:

| arm | canonical search | expected |
|---|---|---|
| ``isalgraph_exhaustive`` | full, over every start | invariant everywhere |
| ``isalgraph_pruned`` | pruned | invariant everywhere |
| ``isalgraph_greedy`` | **none** | not invariant |

Two measurements, and the difference between them matters.

**Part A -- exhaustive over the connected atlas.** Reuses
``admissibility.e1_invariance.exhaustive_invariance``, which enumerates *every*
permutation of *every* connected graph up to ``n = 7`` and is therefore a
**proof on that population**, not a sample. A sampled 89 % non-invariance rate
can be dismissed as a draw artefact; "invariant on 0 of the 853 connected graphs
on 7 nodes, every permutation enumerated" cannot.

**Part B -- a sampled rate at cohort scale.** The atlas stops at ``n = 7`` and
the paper's cohort does not, so Part B draws relabellings at ``n = 5-9`` to show
the failure does not vanish as graphs grow. It is the weaker measurement and is
reported as such.

The ablation's force is that greedy loses on **both** axes at once: it destroys
invariance and it is not shorter. The length half is measured here too, paired
per graph, because "no cheaper and not invariant" is a much stronger sentence
than either half alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from isalgraph.competitors.base import ReprBackend

LOGGER = logging.getLogger(__name__)

CANONICAL_ARMS = ("isalgraph_exhaustive", "isalgraph_pruned")
GREEDY_ARM = "isalgraph_greedy"
ARMS = (*CANONICAL_ARMS, GREEDY_ARM)

SEED = 42
#: Part B draw. ``n`` range and permutations per graph.
SAMPLE_NODE_RANGE = range(5, 10)
SAMPLE_PERMUTATIONS = 6
SAMPLE_GRAPHS_PER_N = 24


@dataclass(frozen=True)
class SampledRow:
    """One arm's Part-B relabelling result at one node count."""

    backend: str
    n_nodes: int
    n_draws: int
    n_changed: int
    non_invariance_rate: float


def _connected(n: int, extra: int, seed: int) -> nx.Graph:
    """A connected graph: a random tree plus *extra* chords."""
    graph = nx.random_labeled_tree(n, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(extra):
        u, v = (int(x) for x in rng.integers(0, n, size=2))
        if u != v:
            graph.add_edge(u, v)
    return graph


def _encode(backend: ReprBackend, graph: nx.Graph) -> str:
    """Encode unbudgeted. These graphs are tiny; a budget would only add noise."""
    from isalgraph.competitors.base import Budget

    text: str = backend.encode(graph, budget=Budget(timeout_s=None)).text
    return text


def part_a_exhaustive(max_n: int) -> dict[str, list[dict[str, Any]]]:
    """Exhaustive invariance over the connected atlas, per arm.

    Args:
        max_n: Largest node count; the atlas reaches 7.

    Returns:
        Arm -> one row per node count, as plain dicts.
    """
    from isalgraph.competitors.admissibility import e1_invariance as e1

    out: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        LOGGER.info("part A: %s", arm)
        rows = e1.exhaustive_invariance(arm, "levenshtein", max_n=max_n, seed=SEED)
        out[arm] = [asdict(row) for row in rows]
    return out


def part_b_sampled() -> list[SampledRow]:
    """Sampled relabelling at cohort-adjacent sizes, per arm and node count."""
    from isalgraph.competitors.registry import get_repr_backend

    rows: list[SampledRow] = []
    for arm in ARMS:
        backend = get_repr_backend(arm)
        for n in SAMPLE_NODE_RANGE:
            rng = random.Random(SEED + n)
            draws = changed = 0
            for g in range(SAMPLE_GRAPHS_PER_N):
                base = _connected(n, n // 3, seed=SEED + 1000 * n + g)
                reference = _encode(backend, base)
                for _ in range(SAMPLE_PERMUTATIONS):
                    perm = list(range(n))
                    rng.shuffle(perm)
                    relabelled = nx.relabel_nodes(base, dict(zip(range(n), perm, strict=True)))
                    draws += 1
                    if _encode(backend, relabelled) != reference:
                        changed += 1
            rows.append(
                SampledRow(
                    backend=arm,
                    n_nodes=n,
                    n_draws=draws,
                    n_changed=changed,
                    non_invariance_rate=changed / draws if draws else 0.0,
                )
            )
            LOGGER.info("part B: %s n=%d -> %d/%d changed", arm, n, changed, draws)
    return rows


def part_c_length() -> dict[str, Any]:
    """Paired symbol counts: greedy against each canonical arm, same graphs.

    Returns:
        Arm -> paired comparison against ``isalgraph_greedy``.
    """
    from isalgraph.competitors.registry import get_repr_backend

    greedy = get_repr_backend(GREEDY_ARM)
    graphs = [
        _connected(n, n // 3, seed=SEED + 1000 * n + g)
        for n in SAMPLE_NODE_RANGE
        for g in range(SAMPLE_GRAPHS_PER_N)
    ]
    greedy_len = np.array([len(_encode(greedy, g)) for g in graphs], dtype=np.int64)

    out: dict[str, Any] = {"n_graphs": len(graphs), "greedy_mean": float(greedy_len.mean())}
    for arm in CANONICAL_ARMS:
        backend = get_repr_backend(arm)
        other = np.array([len(_encode(backend, g)) for g in graphs], dtype=np.int64)
        out[arm] = {
            "mean": float(other.mean()),
            "greedy_shorter": int((greedy_len < other).sum()),
            "equal": int((greedy_len == other).sum()),
            "greedy_longer": int((greedy_len > other).sum()),
            "median_greedy_excess_pct": float(100.0 * (np.median(greedy_len / other) - 1.0)),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Canonicalisation ablation for T06_exhaustive.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/"
            "data/source/T06_exhaustive/invariance_ablation.json"
        ),
    )
    ap.add_argument("--max-n", type=int, default=7, help="atlas ceiling for part A")
    ap.add_argument("--skip-part-a", action="store_true")
    args = ap.parse_args(argv)

    import isalgraph

    payload: dict[str, Any] = {
        "seed": SEED,
        "engine": isalgraph.engine(),
        "build_hash": str(isalgraph.build_info().get("build_hash", "")),
        "arms": list(ARMS),
        "part_b_sampled": [asdict(row) for row in part_b_sampled()],
        "part_c_length": part_c_length(),
    }
    if not args.skip_part_a:
        payload["part_a_exhaustive"] = part_a_exhaustive(args.max_n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
