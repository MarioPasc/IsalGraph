"""The reference arm: IsalGraph's own canonical and pruned-canonical strings.

Two registered backends, because they are **not interchangeable** -- they
produce different strings and different bit counts, and their ceilings are
in different places:

- ``isalgraph_canonical`` is fine on Suite 1 (0 failures) and unusable on
  Suite 2: at a 2 s budget it times out on 207/400 COIL-DEL, 118/400
  Mutagenicity and 300/400 Protein.  It carries
  :attr:`Capability.SUITE1_ONLY` and **raises** above that scale rather than
  silently producing a 76 %-complete column.
- ``isalgraph_pruned`` has a ceiling too, and the correction matters: zero
  failures through AIDS-IAM, then **24/400 on Mutagenicity** (149 ms/graph)
  and **4/400 on Protein** (66 ms/graph) at a 2 s budget.  An earlier note
  claimed ``pruned`` was fine to ``n = 98``; on real graphs it is not.

**Timing note, which is a plan-level instruction.**  These run on the C++
engine.  Timing a pure-Python competitor against it reproduces R1.1's own
complaint inside our answer to it, so a language-matched mode is provided:
:func:`encode_python` forces the pure-Python reference.  Fig. 2 must have
both arms in the same language.  ``engine()`` is recorded in every smoke
run header so a timing can never be quoted without it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

    from isalgraph import SparseGraph

from isalgraph.competitors.base import Budget, Capability, Encoding, ReprBackend
from isalgraph.competitors.registry import register_backend
from isalgraph.errors import BackendError, SuiteScopeError

#: Sigma = {N, n, P, p, V, v, C, c, W}.  Fixed, and the fixedness is an
#: argument the paper makes against min-DFS's O(n^2) alphabet.
ALPHABET_SIZE = 9

#: Reproduction-gate budget only.  T-06 sets the production value; changing
#: it here would move a published failure rate.
DEFAULT_TIMEOUT_S = 2.0

#: Above this node count ``canonical_string`` is not merely slow but useless
#: in aggregate, so the SUITE1_ONLY backend refuses rather than producing a
#: partially complete column whose bit counts are conditioned on the graphs
#: that happened to finish -- a biased sample.
SUITE1_MAX_NODES = 12


def _to_sparse_graph(graph: nx.Graph) -> SparseGraph:
    """Convert a ``networkx.Graph`` to a ``SparseGraph`` on sorted labels."""
    import networkx as nx

    from isalgraph import SparseGraph

    normalised = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    n = normalised.number_of_nodes()
    out = SparseGraph(n, False)
    for _ in range(n):
        out.add_node()
    for u, v in normalised.edges():
        out.add_edge(u, v)
    return out


class _IsalGraphBackend(ReprBackend):
    """Shared machinery for the two IsalGraph arms."""

    #: ``"pruned"`` or ``"canonical"``.
    variant = "pruned"

    def _encoder(self, *, backend: str | None = None) -> Callable[[SparseGraph, float | None], str]:
        from isalgraph import canonical_string, pruned_canonical_string

        fn = pruned_canonical_string if self.variant == "pruned" else canonical_string

        def call(graph: SparseGraph, timeout_s: float | None) -> str:
            return fn(graph, timeout_s=timeout_s, backend=backend)

        return call

    def _check_budget_enforceable(self, timeout: float | None, engine: str | None) -> None:
        """Refuse a timeout the active engine cannot honour.

        ``timeout_s`` is a ``cpp``-only parameter: the pure-Python reference has
        no interruption point, so the C++ path raises ``BackendError`` when
        asked for one.  Two wrong answers are available here and both are taken
        elsewhere in the literature -- drop the timeout silently and run
        unbounded, or catch and report a spurious failure.  The first turns a
        2 s budget into an unbounded Suite-2 run whose bit counts are then
        quoted as if budgeted; the second invents a ceiling that is not real.

        So: state it.  A budget that cannot be enforced refuses, exactly as a
        budget that runs out refuses.  Pass ``Budget(timeout_s=None)`` to opt
        out deliberately -- which is what the language-matched Fig. 2 timing
        does, since it runs small graphs where no budget is needed.
        """
        import isalgraph

        if timeout is None:
            return
        effective = engine or isalgraph.engine()
        if effective != "cpp":
            raise BackendError(
                f"{self.name!r} was asked for a {timeout} s budget but the active "
                f"engine is {effective!r}, which has no interruption point and "
                f"cannot enforce one. Build the C++ extension, or pass "
                f"Budget(timeout_s=None) to run unbounded on purpose"
            )

    def _check_scope(self, graph: nx.Graph) -> None:
        if Capability.SUITE1_ONLY not in self.capabilities:
            return
        n = graph.number_of_nodes()
        if n > SUITE1_MAX_NODES:
            raise SuiteScopeError(
                f"{self.name!r} is Suite-1 only and was asked for n={n}. Above "
                f"n={SUITE1_MAX_NODES} it times out on the majority of real graphs, "
                f"so its bit counts would be conditioned on the graphs fast enough "
                f"to finish. Use 'isalgraph_pruned'"
            )

    def encode(
        self,
        graph: nx.Graph,
        *,
        budget: Budget | None = None,
        engine: str | None = None,
    ) -> Encoding:
        """Encode *graph* as an IsalGraph instruction string.

        Args:
            graph: a ``networkx.Graph``.
            budget: ``timeout_s`` is read; the default is
                :data:`DEFAULT_TIMEOUT_S`.
            engine: ``"cpp"`` or ``"python"``.  ``None`` uses the active
                engine.  Pass ``"python"`` for a language-matched timing
                against a pure-Python competitor.

        Raises:
            CanonicalizationTimeoutError: when the budget runs out.  A
                recorded failure, never a degraded string.
            SuiteScopeError: for ``isalgraph_canonical`` above Suite 1.
            BackendError: when a finite budget is requested but the active
                engine cannot enforce it.  See
                :meth:`_check_budget_enforceable`.
        """
        self._check_scope(graph)
        timeout = DEFAULT_TIMEOUT_S if budget is None else budget.timeout_s
        self._check_budget_enforceable(timeout, engine)
        text = self._encoder(backend=engine)(_to_sparse_graph(graph), timeout)
        return Encoding(
            backend=self.name,
            symbols=tuple(text),
            alphabet_size=ALPHABET_SIZE,
            n_nodes=graph.number_of_nodes(),
            n_edges=graph.number_of_edges(),
            text=text,
        )

    def decode(self, encoding: Encoding) -> nx.Graph:
        """Rebuild the graph by running S2G on the instruction string.

        Reversible **up to isomorphism**, which is what the round-trip
        theorem states: ``S2G(w)`` is isomorphic to ``S2G(G2S(S2G(w), v0))``.
        """
        import networkx as nx

        from isalgraph import StringToGraph

        graph, _trace = StringToGraph(encoding.text, directed=False).run()
        out = nx.Graph()
        out.add_nodes_from(range(graph.node_count()))
        for u in range(graph.node_count()):
            for v in graph.neighbors(u):
                if u < v:
                    out.add_edge(u, v)
        return out

    @classmethod
    def is_available(cls) -> bool:
        """``networkx`` is the only optional dependency; the core is stdlib."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            return False
        return True


class IsalGraphPruned(_IsalGraphBackend):
    """``pruned_canonical_string``.  The arm used on both suites."""

    name = "isalgraph_pruned"
    variant = "pruned"
    capabilities = frozenset(
        {
            Capability.CANONICAL,
            Capability.COMPLETE_INVARIANT,
            Capability.REVERSIBLE,
        }
    )


class IsalGraphCanonical(_IsalGraphBackend):
    """``canonical_string``.  **Suite 1 only** -- see the module docstring."""

    name = "isalgraph_canonical"
    variant = "canonical"
    capabilities = frozenset(
        {
            Capability.CANONICAL,
            Capability.COMPLETE_INVARIANT,
            Capability.REVERSIBLE,
            Capability.SUITE1_ONLY,
        }
    )


register_backend("isalgraph_pruned", IsalGraphPruned)
register_backend("isalgraph_canonical", IsalGraphCanonical)

__all__ = ["ALPHABET_SIZE", "IsalGraphCanonical", "IsalGraphPruned"]
