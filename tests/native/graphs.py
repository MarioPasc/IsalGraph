"""Shared graph generators for the native-engine differential suite.

Everything here is stdlib-only and deterministic: every generator takes an
explicit seed, and the corpus functions return the same graphs on every run so
a reported mismatch count is reproducible.

The generators cover the families named in the acceptance criteria: paths,
cycles, stars, trees, complete graphs, grids, Barabasi-Albert, Erdos-Renyi,
self-loops, directed and undirected.
"""

from __future__ import annotations

import itertools
import random

from isalgraph.core.sparse_graph import SparseGraph
from isalgraph.errors import IsalGraphError

Edge = tuple[int, int]

# The frozen reference raises bare ValueError/RuntimeError; the dispatch layer
# raises the isalgraph.errors classes. Until `main` lands the builtin mixins at
# integration the two families are disjoint, so any "skip the graphs that
# cannot be encoded" guard must name all three or it will let a legitimate
# error escape and be misreported as a parity failure.
ENCODING_ERRORS = (ValueError, RuntimeError, IsalGraphError)


def build(
    n: int, edges: list[Edge], *, directed: bool = False, max_nodes: int | None = None
) -> SparseGraph:
    """Assemble a SparseGraph from an explicit edge list."""
    g = SparseGraph(max_nodes if max_nodes is not None else n, directed)
    for _ in range(n):
        g.add_node()
    for a, b in edges:
        g.add_edge(a, b)
    return g


# ----------------------------------------------------------------------
# Structured families
# ----------------------------------------------------------------------


def path(n: int, *, directed: bool = False) -> SparseGraph:
    return build(n, [(i, i + 1) for i in range(n - 1)], directed=directed)


def cycle(n: int, *, directed: bool = False) -> SparseGraph:
    return build(n, [(i, (i + 1) % n) for i in range(n)], directed=directed)


def star(n: int, *, directed: bool = False) -> SparseGraph:
    return build(n, [(0, i) for i in range(1, n)], directed=directed)


def complete(n: int, *, directed: bool = False) -> SparseGraph:
    pairs = (
        list(itertools.permutations(range(n), 2))
        if directed
        else list(itertools.combinations(range(n), 2))
    )
    return build(n, list(pairs), directed=directed)


def grid(rows: int, cols: int) -> SparseGraph:
    edges: list[Edge] = []
    idx = lambda r, c: r * cols + c  # noqa: E731
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return build(rows * cols, edges)


def random_tree(n: int, seed: int, *, directed: bool = False) -> SparseGraph:
    rng = random.Random(seed)
    return build(n, [(i, rng.randrange(i)) for i in range(1, n)], directed=directed)


def barabasi_albert(n: int, m: int, seed: int) -> SparseGraph:
    """Preferential attachment. Reimplemented so tests stay stdlib-only."""
    rng = random.Random(seed)
    m = max(1, min(m, n - 1))
    edges: list[Edge] = []
    targets = list(range(m))
    repeated: list[int] = list(range(m))
    for i in range(m, n):
        chosen: set[int] = set()
        while len(chosen) < m:
            chosen.add(repeated[rng.randrange(len(repeated))] if repeated else rng.randrange(i))
        for t in sorted(chosen):
            edges.append((i, t))
            repeated.extend((i, t))
    # Connect the initial clique so the graph is connected for m == 1 too.
    for a, b in itertools.combinations(targets, 2):
        edges.append((a, b))
    return build(n, edges)


def erdos_renyi_connected(n: int, p: float, seed: int, *, directed: bool = False) -> SparseGraph:
    """G(n, p) forced connected by first laying down a random spanning tree.

    Connectivity is a precondition of the encoders, so an unconditioned
    Erdos-Renyi sample would spend most of its draws on the error path.
    """
    rng = random.Random(seed)
    # The spanning tree is oriented parent -> child when directed, so node 0 is
    # the root of a spanning out-tree and the graph is always encodable. A
    # child -> parent orientation would make node 0 a sink, and most of the
    # directed corpus would land on the error path instead of being compared.
    edges: list[Edge] = [
        (rng.randrange(i), i) if directed else (i, rng.randrange(i)) for i in range(1, n)
    ]
    present = {frozenset(e) for e in edges}
    for a, b in itertools.combinations(range(n), 2):
        if frozenset((a, b)) not in present and rng.random() < p:
            edges.append((a, b) if not directed or rng.random() < 0.5 else (b, a))
    return build(n, edges, directed=directed)


def with_self_loop(g: SparseGraph, node: int) -> SparseGraph:
    """Return a copy of *g* carrying a self-loop at *node*.

    Self-loops are the case that broke the naive edge-count marshalling: in an
    undirected graph a self-loop occupies one adjacency slot but increments
    the stored edge count twice.
    """
    n = g.node_count()
    edges = [(u, v) for u in range(n) for v in sorted(g.neighbors(u)) if g.directed() or u <= v]
    edges.append((node, node))
    return build(n, edges, directed=g.directed(), max_nodes=g.max_nodes())


# ----------------------------------------------------------------------
# Corpora
# ----------------------------------------------------------------------


def structured_corpus() -> list[tuple[str, SparseGraph]]:
    """Named structured graphs, small enough for the exhaustive canonical search."""
    out: list[tuple[str, SparseGraph]] = []
    for n in range(2, 9):
        out.append((f"path{n}", path(n)))
    for n in range(3, 9):
        out.append((f"cycle{n}", cycle(n)))
    for n in range(2, 9):
        out.append((f"star{n}", star(n)))
    for n in range(2, 7):
        out.append((f"complete{n}", complete(n)))
    for rows, cols in ((2, 2), (2, 3), (3, 3), (2, 4)):
        out.append((f"grid{rows}x{cols}", grid(rows, cols)))
    for seed in range(6):
        out.append((f"tree7_s{seed}", random_tree(7, seed)))
    for seed in range(4):
        out.append((f"ba8m2_s{seed}", barabasi_albert(8, 2, seed)))
    for n in (3, 5, 7):
        out.append((f"pathdir{n}", path(n, directed=True)))
        out.append((f"cycledir{n}", cycle(n, directed=True)))
        out.append((f"stardir{n}", star(n, directed=True)))
    out.append(("selfloop_path4", with_self_loop(path(4), 0)))
    out.append(("selfloop_cycle4", with_self_loop(cycle(4), 2)))
    out.append(("selfloop_star5", with_self_loop(star(5), 0)))
    out.append(("selfloop_dirpath4", with_self_loop(path(4, directed=True), 1)))
    out.append(("singleton", build(1, [])))
    out.append(("edge", build(2, [(0, 1)])))
    return out


def random_corpus(count: int, *, max_n: int = 8, directed: bool = False) -> list[SparseGraph]:
    """Deterministic Erdos-Renyi corpus spanning 2..max_n nodes."""
    out: list[SparseGraph] = []
    for seed in range(count):
        n = 2 + seed % (max_n - 1)
        p = 0.15 + 0.6 * ((seed // (max_n - 1)) % 4) / 3.0
        out.append(erdos_renyi_connected(n, p, seed, directed=directed))
    return out


def sized_corpus(
    spec: dict[int, int], *, directed: bool = False, seed0: int = 0
) -> list[SparseGraph]:
    """Erdos-Renyi corpus with an explicit node-count distribution.

    The differential tests run the *Python* reference on every graph, and that
    reference is exponential: one 8-node graph costs ~1.2 s where a 5-node one
    costs ~3 ms.  A uniform spread over 2..8 would therefore spend 95% of the
    wall clock on 5% of the corpus.  Weighting the distribution toward small
    graphs buys the required sample size at a tractable cost while still
    covering the expensive tail.

    Args:
        spec: Mapping of node count to how many graphs of that size.
        directed: Whether to build directed graphs.
        seed0: Base seed, so several corpora can be disjoint.

    Returns:
        Connected graphs, deterministic across runs.
    """
    out: list[SparseGraph] = []
    seed = seed0
    for n in sorted(spec):
        for k in range(spec[n]):
            p = 0.15 + 0.6 * ((k % 4) / 3.0)
            out.append(erdos_renyi_connected(n, p, seed, directed=directed))
            seed += 1
    return out


def relabel(g: SparseGraph, perm: list[int]) -> SparseGraph:
    """Return *g* with node i renamed to perm[i]."""
    n = g.node_count()
    edges = [(perm[u], perm[v]) for u in range(n) for v in sorted(g.neighbors(u))]
    return build(n, edges, directed=g.directed(), max_nodes=g.max_nodes())


def edge_set(g: SparseGraph) -> set[tuple[int, int]]:
    """Canonical comparable form of a graph's edges."""
    n = g.node_count()
    if g.directed():
        return {(u, v) for u in range(n) for v in g.neighbors(u)}
    return {(min(u, v), max(u, v)) for u in range(n) for v in g.neighbors(u)}


def graphs_equal(a: SparseGraph, b: SparseGraph) -> bool:
    """Structural equality: same node count, direction and edge set."""
    return (
        a.node_count() == b.node_count()
        and a.directed() == b.directed()
        and edge_set(a) == edge_set(b)
    )
