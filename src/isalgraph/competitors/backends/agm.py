"""AGM canonical adjacency-matrix code (CAM), branch and bound.

Inokuchi, Washio & Motoda, *An Apriori-Based Algorithm for Mining Frequent
Substructures from Graph Data*, PKDD 2000, LNCS 1910:13-23,
doi:10.1007/3-540-45372-5_2; extended as *Complete Mining of Frequent
Patterns from Graphs*, Machine Learning 50:321-354, 2003,
doi:10.1023/A:1021726221443.  Neither ships code, and no maintained package
exposes "the AGM canonical code of one graph" -- the canonical form lives
inside frequent-subgraph miners, applied to mined patterns.  So this is our
implementation, validated against brute force over all ``n!`` permutations
on 327 graphs (``tests/unit/test_agm_cam.py``).

The convention, stated once
---------------------------

**AGM takes the minimum; FFSM (Huan, Wang & Prins, ICDM 2003) takes the
maximum.**  They are mirror images and neither is more canonical than the
other, so the convention must be stated in the paper or the numbers are
unreproducible.

We use AGM's **minimum**, over the **strict lower triangle read row by
row**.  Following Jiang, Coenen & Zito, *A survey of frequent subgraph
mining algorithms*, Knowledge Engineering Review 28(1):75-105, 2013,
doi:10.1017/S0269888912000331, §3.1: the encoding concatenates the
triangular entries of the adjacency matrix and the canonical form is the
extremal encoding.  For an unlabelled simple graph the diagonal is all-zero
and every vertex label is equal, so::

    code(pi) = x_{1,0} | x_{2,0} x_{2,1} | x_{3,0} x_{3,1} x_{3,2} | ...

which is the same bit sequence as the strict **upper** triangle read
column-wise -- i.e. byte-identical to graph6's payload and to the
``adjacency`` backend's symbol order.  A test asserts that agreement.

That reading order is not cosmetic.  The first ``k(k-1)/2`` bits depend
only on the first ``k`` vertices of the permutation, and **that prefix
property is the only reason branch and bound is possible at all**.

The ceiling, and why it must raise
----------------------------------

Exact minimisation is a lex-leader problem, NP-hard in general (Crawford,
Ginsberg, Luks & Roy, *Symmetry-breaking predicates for search problems*,
KR 1996, 148-159; framework: Babai & Luks, STOC 1983, 171-183).  We do not
claim the graph-restricted case inherits that bound; we report the measured
behaviour, which is consistent with it.

Measured on the real cohort at the frozen budgets (``agm.md`` §2.2b): all
three Letter sets and LINUX **100 %** exact at 200k, AIDS **99.6 %** (3 of
769 fail), then GREC **76 %** and AIDS-IAM **82 %** at 100k, collapsing to
**2 %** on Mutagenicity.  All of Suite 1 is computable; none of Suite 2 is.
The failure is driven by the tail, not the mean -- Protein (``n_bar =
31.9``) fails *less* often than Mutagenicity (``n_bar = 27.9``) -- so the
ceiling cannot be stated as a single ``n``.

Two consequences that are enforced here rather than documented:

1. :func:`agm_canonical_code` **raises** :class:`AGMBudgetExceeded` and
   never returns the incumbent.  The greedy initialisation's incumbent is
   not canonical, would fail F3, and would put a non-invariant code into a
   column headed canonical -- precisely the error ``graph6`` is in the pool
   to expose.
2. ``agm_cam`` carries :attr:`Capability.SUITE1_ONLY` and refuses above
   :data:`SUITE1_MAX_NODES` rather than silently producing a 76 %-complete
   column whose bit counts are conditioned on the graphs that happened to
   finish.

**AGM contributes nothing new to Claim A.**  Its bit count is ``n(n-1)/2``,
identical to the raw adjacency matrix by construction, because
canonicalisation permutes bits without changing their number.  Print one
``n^2`` row naming its four members; AGM earns its place on Claim B and on
the AE.3 properties table.

**nauty cannot supply this labelling.**  ``competitors.md`` §2 budgets AGM
at "1 d, derive from nauty labelling"; the premise is wrong.  nauty
produces *a* canonical labelling, not the one minimising AGM's code.  On
the running example nauty gives ``001110010011100`` and AGM gives
``000001110011110`` -- both canonical, different bit strings.
``pynauty.autgrp``'s orbits *can* prune this search, and
``backends/nauty.py`` exposes them, but the pruning is **deliberately not
wired in**: it changes how many search nodes are expanded, and the frozen
budgets are the values behind published failure rates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

from isalgraph.competitors.base import (
    Budget,
    Capability,
    Encoding,
    PositionalFrame,
    ReprBackend,
)
from isalgraph.competitors.registry import register_backend
from isalgraph.errors import AGMBudgetExceeded, SuiteScopeError

#: Frozen search-node budgets.  **These are the values behind published
#: failure rates -- 99.6 % on AIDS, 76 % on GREC, 82 % on AIDS-IAM.
#: Changing one silently rewrites a number the plan already prints.**
SUITE1_NODE_BUDGET = 200_000
SUITE2_NODE_BUDGET = 100_000

#: Used when no :class:`Budget` is supplied.  ``agm_cam`` is Suite-1 only,
#: so the Suite-1 value is the default.
DEFAULT_NODE_BUDGET = SUITE1_NODE_BUDGET

#: Suite 1's true maximum node count is ``n = 12`` on AIDS, so this bound
#: admits every Suite-1 graph and rejects everything above.  The same
#: constant and the same rule as ``backends/isalgraph_ref.py``.
SUITE1_MAX_NODES = 12

#: ``{0, 1}``.
ALPHABET_SIZE = 2


def _adjacency_sets(graph: nx.Graph) -> tuple[list[set[int]], int]:
    """Neighbour sets on ``0..n-1``, in the graph's own node order.

    **Deliberately not normalised.**  The canonical code is a minimum over
    every permutation, so it does not depend on this order at all -- but the
    greedy incumbent and therefore the *number of search nodes expanded* do,
    and the frozen 200k/100k budgets are the values behind published failure
    rates.  Normalising here would move them.  :func:`identity_code` is the
    one place order is pinned, because that is the one place it is observed.
    """
    nodes = list(graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    return [{index[w] for w in graph.neighbors(v)} for v in nodes], len(nodes)


def identity_code(graph: nx.Graph) -> str:
    """The code at the **identity** permutation, on the pinned labelling.

    This is AGM's contribution to the reading-order identity: it must equal
    ``adjacency.symbols`` and graph6's unpacked payload for the same graph,
    bit for bit (CONTRACTS.md §9).  It is not canonical and is not what
    ``encode`` returns; it exists so the family identity is executable.

    The labelling is pinned by **rebuilding** the graph with nodes added in
    ascending order.  ``nx.convert_node_labels_to_integers(ordering=
    "sorted")`` does *not* do this -- it renames values and leaves insertion
    order alone, and ``to_graph6_bytes`` re-derives its labelling from
    insertion order, so the two disagree on 290 of 300 scrambled graphs
    (measured in wave 1 by track A and by the orchestrator independently).
    ``graph6.md`` §7 prescribes the broken call; this does not use it.
    """
    import networkx as nx

    pinned = nx.Graph()
    pinned.add_nodes_from(sorted(graph.nodes()))
    pinned.add_edges_from(graph.edges())
    adjacency, n = _adjacency_sets(pinned)
    return _code_from_perm(adjacency, list(range(n)))


def _code_from_perm(adjacency: list[set[int]], perm: list[int]) -> str:
    """The code *perm* induces: strict lower triangle, row by row."""
    bits: list[str] = []
    for k in range(1, len(perm)):
        row = adjacency[perm[k]]
        for j in range(k):
            bits.append("1" if perm[j] in row else "0")
    return "".join(bits)


def _greedy_incumbent(adjacency: list[set[int]], n: int) -> str:
    """A starting upper bound: low degree first, fewest links to the prefix.

    **Never returned to a caller.**  It is a bound for the search, not an
    answer; an incumbent is not canonical and would fail F3.
    """
    order: list[int] = []
    placed: set[int] = set()
    for _ in range(n):
        candidate = min(
            (v for v in range(n) if v not in placed),
            key=lambda v: (len(adjacency[v] & placed), len(adjacency[v]), v),
        )
        order.append(candidate)
        placed.add(candidate)
    return _code_from_perm(adjacency, order)


def agm_canonical_code(
    graph: nx.Graph,
    *,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> tuple[str, int]:
    """Minimum CAM code of an unlabelled simple graph.

    This is **the algorithm**, with no suite policy attached: the
    ``SUITE1_ONLY`` refusal belongs to :class:`AGMBackend`, so that the
    ceiling itself stays measurable at any scale.  ``agm.md`` §2.2b's
    per-dataset exact-rate table is produced by calling this directly.

    Args:
        graph: a ``networkx.Graph``.  Disconnected graphs and isolated
            vertices are fine -- AGM was designed for them.
        node_budget: branch-and-bound search nodes.  See
            :data:`SUITE1_NODE_BUDGET` / :data:`SUITE2_NODE_BUDGET`.

    Returns:
        ``(code, nodes_expanded)``.  *code* has ``n(n-1)/2`` bits.

    Raises:
        AGMBudgetExceeded: when the budget runs out before the search
            closes.  **The incumbent is discarded, not returned.**  A stated
            ceiling is a result; a silent one is a defect.
    """
    adjacency, n = _adjacency_sets(graph)
    if n <= 1:
        return "", 0

    best = _greedy_incumbent(adjacency, n)
    expanded = 0
    prefix: list[int] = []
    used = [False] * n

    def extend(position: int, code_so_far: str) -> None:
        nonlocal best, expanded
        expanded += 1
        if expanded > node_budget:
            raise AGMBudgetExceeded(
                f"AGM canonical code: budget of {node_budget} search nodes exhausted "
                f"on a graph with n={n}, m={graph.number_of_edges()}. The incumbent is "
                f"NOT canonical and is deliberately not returned"
            )
        if position == n:
            if code_so_far < best:
                best = code_so_far
            return
        candidates: list[tuple[str, int]] = []
        for v in range(n):
            if used[v]:
                continue
            row = adjacency[v]
            candidates.append(
                ("".join("1" if prefix[j] in row else "0" for j in range(position)), v)
            )
        candidates.sort()
        for bits, v in candidates:
            extended = code_so_far + bits
            # Prefix pruning: the first k(k-1)/2 bits depend only on the
            # first k vertices, so a prefix already above the incumbent
            # cannot be completed below it.
            if extended > best[: len(extended)]:
                continue
            used[v] = True
            prefix.append(v)
            extend(position + 1, extended)
            prefix.pop()
            used[v] = False

    extend(0, "")
    return best, expanded


def code_to_graph(code: str, n: int) -> nx.Graph:
    """Inverse map: the code plus ``n`` rebuilds the graph exactly.

    Reversible in the strong sense -- not merely up to isomorphism, since
    the code *is* the canonical labelling's adjacency triangle.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    bits = iter(code)
    for k in range(1, n):
        for j in range(k):
            if next(bits) == "1":
                graph.add_edge(k, j)
    return graph


def upper_triangle_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """``(i, j)``, ``i < j``, strict upper triangle **column-wise**.

    Identical to the order :func:`_code_from_perm` walks, read the other
    way round: its ``(j, k)`` with ``j < k`` is this ``(i, j)`` with
    ``i < j``.  That identity is what makes the ``adjacency`` / ``graph6``
    / ``agm_cam`` family share one reading order.
    """
    return tuple((i, j) for j in range(1, n) for i in range(j))


class AGMBackend(ReprBackend):
    """The AGM canonical adjacency-matrix code.  **Suite 1 only.**"""

    name = "agm_cam"
    capabilities = frozenset(
        {
            Capability.POSITIONAL_FRAME,
            Capability.CANONICAL,
            Capability.COMPLETE_INVARIANT,
            Capability.REVERSIBLE,
            Capability.HANDLES_DISCONNECTED,
            Capability.SUITE1_ONLY,
        }
    )

    def _check_scope(self, graph: nx.Graph) -> None:
        n = graph.number_of_nodes()
        if n > SUITE1_MAX_NODES:
            raise SuiteScopeError(
                f"{self.name!r} is Suite-1 only and was asked for n={n}. Above "
                f"n={SUITE1_MAX_NODES} the branch and bound closes on a minority of "
                f"real graphs (76 % GREC, 82 % AIDS-IAM, 2 % Mutagenicity at a "
                f"{SUITE2_NODE_BUDGET} node budget), so its column would be conditioned "
                f"on the graphs symmetric enough to finish -- a biased sample"
            )

    def encode(self, graph: nx.Graph, *, budget: Budget | None = None) -> Encoding:
        """Encode *graph* as its minimum CAM code.

        Args:
            graph: a ``networkx.Graph``.
            budget: ``search_nodes`` is read; the default is
                :data:`DEFAULT_NODE_BUDGET`.

        Raises:
            AGMBudgetExceeded: budget exhausted.  A recorded failure, never
                a heuristic labelling.
            SuiteScopeError: above :data:`SUITE1_MAX_NODES`.
        """
        self._check_scope(graph)
        node_budget = DEFAULT_NODE_BUDGET
        if budget is not None and budget.search_nodes is not None:
            node_budget = budget.search_nodes
        code, _expanded = agm_canonical_code(graph, node_budget=node_budget)
        n = graph.number_of_nodes()
        return Encoding(
            backend=self.name,
            symbols=tuple(code),
            alphabet_size=ALPHABET_SIZE,
            n_nodes=n,
            n_edges=graph.number_of_edges(),
            text=code,
            payload_bits=n * (n - 1) // 2,
            frame=PositionalFrame(
                n_nodes=n,
                pairs=upper_triangle_pairs(n),
                bits=tuple(code),
            ),
        )

    def decode(self, encoding: Encoding) -> nx.Graph:
        """Rebuild the canonically labelled graph from the code and ``n``."""
        return code_to_graph("".join(encoding.symbols), encoding.n_nodes)

    @classmethod
    def is_available(cls) -> bool:
        """``networkx`` only -- the search itself is pure standard library."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            return False
        return True


register_backend("agm_cam", AGMBackend)

__all__ = [
    "ALPHABET_SIZE",
    "DEFAULT_NODE_BUDGET",
    "SUITE1_MAX_NODES",
    "SUITE1_NODE_BUDGET",
    "SUITE2_NODE_BUDGET",
    "AGMBackend",
    "agm_canonical_code",
    "code_to_graph",
    "identity_code",
    "upper_triangle_pairs",
]
