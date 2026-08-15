"""The minimum DFS code of a single graph -- gSpan's canonical form.

Yan & Han, *gSpan: Graph-Based Substructure Pattern Mining*, ICDM 2002,
721-724, doi:10.1109/ICDM.2002.1184038.

This is the closest competitor IsalGraph has: canonical, a string,
edit-distance-comparable, and the same problem setting.  Reviewer 1 named
gSpan by name.

**Nothing is vendored.**  Three repositories were tested and all three
rejected: ``LasseRegin/gSpan`` does not run on numpy >= 1.24 and its
``G2DFS`` reads insertion order rather than producing the minimum;
``betterenvi/gSpan``'s ``_is_min`` is a private method needing a miner, a
graph database and a ``min_support``; and ``kaviniitm/DFSCode`` builds
cleanly, claims exactly this in its README, agrees with us on the running
example, and is **wrong on half of all 6-node graphs and not
isomorphism-invariant** (46 of 90 graphs).  ``T-04-design.md`` decision 8's
instruction to vendor ``LasseRegin/gSpan`` is superseded by that
measurement.  ``tests/unit/test_min_dfs.py`` keeps the ``kaviniitm``
verdict as the acceptance gate any future third-party candidate must pass,
**K2 (isomorphism invariance) first** -- K2 needs no oracle and it is where
that implementation died.

Construction
------------
A DFS code is a sequence of 5-tuples ``(i, j, l_i, l_ij, l_j)`` with ``i``,
``j`` DFS discovery indices.  Our corpus is topology-only, so every label is
constant and the tuple degenerates to ``(i, j)``.  ``|code| = m`` exactly,
always -- the only pool member whose length is deterministic.

The construction is the standard one **with correct tie branching**: hold
the set of embeddings realising the current minimal prefix, take the
globally minimal rightmost-path extension, and keep only the embeddings
achieving it.  Tie branching is not optional here.  For an unlabelled graph
every vertex and edge label is equal, so *every* step is a tie, and the
greedy no-branch construction that ``LasseRegin/gSpan`` uses is not
guaranteed to reach the minimum.

Under the DFS lexicographic order (Yan & Han, Def. 5), backward edges
precede forward edges, backward edges are ordered by increasing target
index, and forward extensions prefer the deepest point of the rightmost
path.

Bit accounting -- and why the bound is not tightened
----------------------------------------------------
``bits.py`` reports ``entropy_bits = m * 2*ceil(log2 n)``.  **That is a
fixed-width upper bound and a reviewer can say so**: DFS indices are not
uniform on ``[0, n)`` -- a forward extension always introduces index
``max + 1`` and a backward edge targets a vertex on the rightmost path, so
a tighter bound exists.

We report the fixed-width bound, state that it is an upper bound, and state
why we did not tighten it: the same fixed-width convention is applied to
``B_GED``'s ``2M ceil(log2 N)`` endpoint addressing (``statistics.md`` §2),
so tightening one and not the other would be exactly the asymmetry R3.6a
objects to.  **Consistency is the defence; silence is not.**

``realised_bits`` is ``8 * len(text)`` over the character rendering and is
flagged ``inflated=True`` precisely so it cannot be quoted unlabelled.

Two conventions that are decided here, not in the analysis
-----------------------------------------------------------
1. **One symbol is one DFS tuple.**  ``Encoding.symbols`` is tuple-level;
   ``Encoding.text`` carries the character rendering ``'0-1 1-2 2-0 ...'``
   for figures only.  Mixing them produced a **2x** difference in measured
   Levenshtein: character level charges four edits for one deleted tuple
   (``' 5-2'``), tuple level charges one and is the semantically correct
   unit -- the like-for-like comparison against IsalGraph, whose symbols are
   also single operations.
2. **The budget is on MEMORY, not time.**  The construction holds every
   embedding realising the current minimal prefix and that set is worst-case
   exponential in the number of ties.  The first Suite-2 run was
   **OOM-killed (exit 137)** partway through Mutagenicity, not slow --
   *killed*.  A wall-clock cap does not prevent this.
   :data:`MAX_PROJECTIONS` is **50_000, frozen**: at that cap the cost is
   24 of 400 Mutagenicity graphs and zero elsewhere in the cohort, and that
   failure rate is published.

There is **no positional frame**: the code is a sequence of index pairs, not
a bit vector over a fixed cell ordering, so ``padded_hamming`` is
*undefined* here.  That is a reported F1 result, not an error.

Disconnected graphs raise :class:`ValueError` by construction -- a DFS code
spans one component.  Both suites are ``require_connected = True``, so it
never fires on the cohort, but it is still a documented row in the AE.3
table because AGM, graph6 and sparse6 handle disconnection and IsalGraph
does not either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isalgraph.competitors.base import (
    Budget,
    Capability,
    Encoding,
    ReprBackend,
)
from isalgraph.competitors.registry import register_backend
from isalgraph.errors import MinDfsBudgetExceeded

if TYPE_CHECKING:
    import networkx as nx

#: A DFS edge ``(i, j, l_i, l_ij, l_j)``.  The full 5-tuple is kept rather
#: than collapsed to ``(i, j)`` so the implementation stays a literal
#: transcription of Yan & Han and so the brute-force oracle in
#: ``tests/unit/test_min_dfs.py`` compares the same object the paper
#: defines.  For an unlabelled corpus every label is ``0``.
DfsEdge = tuple[int, int, int, int, int]

#: **Frozen.**  A cap on simultaneously live embeddings, i.e. on memory.
#: Behind the published ``24 / 400`` Mutagenicity failure rate -- changing it
#: silently rewrites a number the plan already prints.
MAX_PROJECTIONS = 50_000


def rightmost_path(code: list[DfsEdge]) -> list[int]:
    """DFS indices on the rightmost path of *code*, root first.

    Args:
        code: a DFS code prefix.

    Returns:
        The DFS indices of the rightmost path, from the root to the
        rightmost vertex.  Empty for an empty code.
    """
    if not code:
        return []
    path: list[int] = []
    prev = -1
    for i, j, *_ in reversed(code):
        if i < j and (prev == -1 or j == prev):  # forward edge on the rightmost path
            if prev == -1:
                path.append(j)
            path.append(i)
            prev = i
    return list(reversed(path))


def extension_key(edge: DfsEdge) -> tuple[int, ...]:
    """Sort key realising the DFS lexicographic order on one extension step.

    Backward extensions precede forward ones and are ordered by increasing
    target index; forward extensions share the new index ``j``, so they are
    ordered by *decreasing* source index -- the deepest point of the
    rightmost path first.

    Args:
        edge: a candidate extension.

    Returns:
        A tuple that sorts ascending in the DFS lexicographic order.
    """
    i, j, _li, lij, lj = edge
    if i > j:  # backward: smallest first, ordered by target j
        return (0, j, lij)
    return (1, -i, lij, lj)  # forward: a deeper source i is smaller


class _Projection:
    """One partial embedding: DFS index -> graph vertex, plus consumed edges."""

    __slots__ = ("g_of", "used", "v_of")

    def __init__(
        self,
        v_of: list[int],
        g_of: dict[int, int],
        used: frozenset[frozenset[int]],
    ) -> None:
        self.v_of = v_of  # DFS index -> graph node
        self.g_of = g_of  # graph node -> DFS index
        self.used = used  # frozenset({u, v}) per consumed graph edge


def _roots(graph: nx.Graph) -> tuple[DfsEdge, list[_Projection]]:
    """The minimal root tuple and every embedding realising it.

    On a labelled corpus the root is the minimal ``(l_u, l_uv, l_v)`` over
    every *oriented* edge, and only the embeddings achieving that minimum
    survive.  Our corpus is topology-only, so every label is ``0``, the
    minimum is ``(0, 1, 0, 0, 0)``, and **every oriented edge realises it**
    -- which is why tie branching is the whole algorithm here rather than a
    corner case.
    """
    root: DfsEdge = (0, 1, 0, 0, 0)
    projections = [
        _Projection([a, b], {a: 0, b: 1}, frozenset({frozenset((a, b))}))
        for u, v in graph.edges()
        for a, b in ((u, v), (v, u))
    ]
    return root, projections


def min_dfs_code(
    graph: nx.Graph,
    *,
    max_projections: int | None = MAX_PROJECTIONS,
) -> list[DfsEdge]:
    """Minimum DFS code of a connected graph with at least one edge.

    Args:
        graph: a connected ``networkx.Graph`` with at least one edge.
        max_projections: cap on the number of simultaneously live
            embeddings.  ``None`` is unbounded, which is what the
            exhaustive validation suite uses.  The shipped default is
            :data:`MAX_PROJECTIONS`.

    Returns:
        The minimum DFS code, ``m`` tuples long.

    Raises:
        ValueError: if *graph* is disconnected or has no edge.  The DFS code
            is undefined in both cases and the caller must supply a
            convention rather than receive a silent one.
        MinDfsBudgetExceeded: if the live embedding set exceeds
            *max_projections*.  Carries no incumbent: a partial answer in a
            column headed canonical is the error this pool exists to expose.
    """
    import networkx as nx

    if graph.number_of_edges() == 0:
        raise ValueError("DFS code undefined: no edges (an isolated vertex carries no tuple)")
    if not nx.is_connected(graph):
        raise ValueError("DFS code undefined: graph is disconnected")

    m = graph.number_of_edges()
    root, projections = _roots(graph)
    code: list[DfsEdge] = [root]

    while len(code) < m:
        if max_projections is not None and len(projections) > max_projections:
            raise MinDfsBudgetExceeded(
                f"min-DFS: {len(projections)} live embeddings at edge "
                f"{len(code)}/{m} exceeds max_projections={max_projections}. "
                f"The budget is on memory, not time: the first Suite-2 run "
                f"was OOM-killed rather than slow."
            )
        candidates = _extensions(graph, code, projections)
        if not candidates:  # pragma: no cover - unreachable on a connected graph
            raise RuntimeError("no extension available; graph traversal is inconsistent")
        chosen = min(candidates, key=extension_key)
        code.append(chosen)
        projections = candidates[chosen]

    return code


def _extensions(
    graph: nx.Graph,
    code: list[DfsEdge],
    projections: list[_Projection],
) -> dict[DfsEdge, list[_Projection]]:
    """Every rightmost-path extension, grouped by the tuple it produces."""
    rmpath = rightmost_path(code)
    rm_idx = rmpath[-1]
    candidates: dict[DfsEdge, list[_Projection]] = {}

    for projection in projections:
        rm_v = projection.v_of[rm_idx]
        # Backward extensions: the rightmost vertex to an ancestor on the
        # rightmost path.  These precede every forward extension.
        for ancestor in rmpath[:-1]:
            ancestor_v = projection.v_of[ancestor]
            edge = frozenset((rm_v, ancestor_v))
            if graph.has_edge(rm_v, ancestor_v) and edge not in projection.used:
                tup: DfsEdge = (rm_idx, ancestor, 0, 0, 0)
                candidates.setdefault(tup, []).append(
                    _Projection(projection.v_of, projection.g_of, projection.used | {edge})
                )
        # Forward extensions: any rightmost-path vertex to a fresh vertex.
        new_idx = len(projection.v_of)
        for src in rmpath:
            src_v = projection.v_of[src]
            for w in graph.neighbors(src_v):
                if w in projection.g_of:
                    continue
                edge = frozenset((src_v, w))
                if edge in projection.used:
                    continue
                tup = (src, new_idx, 0, 0, 0)
                candidates.setdefault(tup, []).append(
                    _Projection(
                        [*projection.v_of, w],
                        {**projection.g_of, w: new_idx},
                        projection.used | {edge},
                    )
                )
    return candidates


def code_symbols(code: list[DfsEdge]) -> tuple[str, ...]:
    """**The comparison unit**: one symbol per DFS tuple.

    Args:
        code: a DFS code.

    Returns:
        ``('0-1', '1-2', '2-0', ...)`` -- one entry per tuple, so one edit
        is one tuple.  Never the character rendering.
    """
    return tuple(f"{i}-{j}" for i, j, *_ in code)


def render(code: list[DfsEdge]) -> str:
    """Character rendering, **for figures and debugging only**.

    Args:
        code: a DFS code.

    Returns:
        ``'0-1 1-2 2-0 ...'``.  Never measured with an edit distance except
        by the one sanctioned reader ``levenshtein_char``, which exists to
        *report* the character-level answer as a supplementary number.
    """
    return " ".join(code_symbols(code))


def code_to_graph(code: list[DfsEdge]) -> nx.Graph:
    """Rebuild a graph from a DFS code, up to isomorphism.

    Args:
        code: a DFS code.

    Returns:
        A graph on the DFS indices.  Isomorphic to the encoded graph;
        vertices carry DFS indices, not the original labels.
    """
    import networkx as nx

    out = nx.Graph()
    for i, j, _li, _lij, _lj in code:
        out.add_node(i)
        out.add_node(j)
        out.add_edge(i, j)
    return out


class MinDfsBackend(ReprBackend):
    """gSpan's minimum DFS code as a :class:`ReprBackend`."""

    name = "min_dfs"
    capabilities = frozenset(
        {
            Capability.CANONICAL,
            Capability.COMPLETE_INVARIANT,
            Capability.REVERSIBLE,
        }
    )

    def encode(self, graph: nx.Graph, *, budget: Budget | None = None) -> Encoding:
        """Encode *graph* as its minimum DFS code.

        Args:
            graph: a connected ``networkx.Graph`` with at least one edge.
            budget: ``None`` uses the frozen :data:`MAX_PROJECTIONS`.  A
                supplied :class:`~isalgraph.competitors.base.Budget` with
                ``max_projections=None`` runs unbounded, which is what the
                exhaustive oracles do.

        Returns:
            The encoding.  ``symbols`` is tuple-level; ``text`` is the
            character rendering and is never measured.

        Raises:
            ValueError: on a disconnected or edgeless graph.
            MinDfsBudgetExceeded: when the live embedding set outgrows the
                budget.
        """
        cap = MAX_PROJECTIONS if budget is None else budget.max_projections
        code = min_dfs_code(graph, max_projections=cap)
        symbols = code_symbols(code)
        n = graph.number_of_nodes()
        return Encoding(
            backend=self.name,
            symbols=symbols,
            # The alphabet is the set of ordered index pairs, so it grows as
            # O(n^2).  This is the one *conceptual* difference R1.2 asks
            # about that is not a matter of degree: IsalGraph's alphabet is
            # nine construction operations regardless of n.
            alphabet_size=max(n * (n - 1), 2),
            n_nodes=n,
            n_edges=graph.number_of_edges(),
            text=" ".join(symbols),
        )

    def decode(self, encoding: Encoding) -> nx.Graph:
        """Rebuild the graph from *encoding*, up to isomorphism.

        Args:
            encoding: an encoding produced by :meth:`encode`.

        Returns:
            A graph isomorphic to the encoded one.
        """
        import networkx as nx

        out = nx.Graph()
        for symbol in encoding.symbols:
            i, j = symbol.split("-")
            out.add_edge(int(i), int(j))
        return out

    @classmethod
    def is_available(cls) -> bool:
        """``networkx`` is a genuine runtime dependency here, not just a type.

        ``encode`` calls ``nx.is_connected``: the DFS code is undefined on a
        disconnected graph and refusing it is part of the contract.  Without
        this the backend reported itself available with ``networkx`` absent
        and then raised a bare ``ImportError`` from inside ``encode`` -- the
        wrong exception, from the wrong place, after the caller had already
        been told the backend was usable.  A missing dependency must surface
        as ``BackendUnavailableError`` **on request**.

        ``wl_subtree`` deliberately does *not* override this: it needs no
        ``networkx`` at all and reporting itself available is honest.
        """
        try:
            import networkx  # noqa: F401
        except ImportError:
            return False
        return True


register_backend("min_dfs", MinDfsBackend)

__all__ = [
    "MAX_PROJECTIONS",
    "DfsEdge",
    "MinDfsBackend",
    "code_symbols",
    "code_to_graph",
    "extension_key",
    "min_dfs_code",
    "render",
    "rightmost_path",
]
