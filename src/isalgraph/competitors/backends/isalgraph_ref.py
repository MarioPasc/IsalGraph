"""The reference arm: IsalGraph's own canonical and pruned-canonical strings.

Three registered backends, because they are **not interchangeable** -- they
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
- ``isalgraph_exhaustive`` computes the same string as
  ``isalgraph_canonical`` but **accepts both suites**, because the ceiling
  the ``SUITE1_ONLY`` guard encodes is not the ceiling the engine has.

**Why a third arm rather than lifting the guard on the second.**  The
``n = 12`` refusal in ``isalgraph_canonical`` was calibrated at a **2 s
budget on the pure-Python path** and is far too conservative on the C++
engine.  Measured (``.claude/notes/review/tasks/t06_exhaustive_ceiling.py``):
100 % completion through ``n = 20`` at a 60 s budget, median 9 ms against a
33 s maximum; 75-100 % through ``n = 26`` at 20 s.  The exhaustive form is
**8-12 % shorter** than the pruned form at ``n = 13-20`` and 12-22 % at
``n = 23-26``, and over all 5,350 Suite-1 graphs the pruned form is
**never** shorter (0 of 5,350,
``.claude/notes/review/tasks/t06_pruned_vs_exhaustive.py``).

``isalgraph_canonical`` keeps its guard: it is the frozen T-04 arm that
already carries published numbers, and moving a published refusal would
move a published failure rate.  The new arm carries the corrected ceiling.

**The budget is not this module's to enforce.**  A graph that exhausts its
budget raises, and the *campaign driver* substitutes a fallback string --
see ``benchmarks/real_data/eval_encoding/t06_encode.py``, which applies D14
in one place so that a censored graph cannot be dropped down one code path
and retained down another.  :attr:`ReprBackend.fallback_variant` names the
substitute this arm wants; it does not perform the substitution.

**Timing note, which is a plan-level instruction.**  These run on the C++
engine.  Timing a pure-Python competitor against it reproduces R1.1's own
complaint inside our answer to it, so a language-matched mode is provided:
passing ``engine="python"`` forces the pure-Python reference.  Fig. 2 must
have both arms in the same language.  ``engine()`` is recorded in every
smoke run header so a timing can never be quoted without it.
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


def to_sparse_graph(graph: nx.Graph) -> SparseGraph:
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
        text = self._encoder(backend=engine)(to_sparse_graph(graph), timeout)
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


class IsalGraphExhaustive(_IsalGraphBackend):
    """``canonical_string`` on **both** suites, with a declared fallback.

    The same string as :class:`IsalGraphCanonical` -- the true ``w*_G``, so
    the same completeness theorem and the same alphabet -- without the
    ``SUITE1_ONLY`` refusal, whose ``n = 12`` threshold was calibrated on
    the pure-Python path at a 2 s budget and does not describe the engine.

    A graph that exhausts its wall clock raises
    ``CanonicalizationTimeoutError`` exactly as the other two arms do.  The
    campaign driver then records it as ``status="censored"`` with the
    :attr:`fallback_variant` string, so the column is never conditioned on
    the graphs that happened to finish.  Substituting here instead would
    report the row as ``ok``, which is the bias the refusal was written to
    avoid in the first place.

    ``pruned`` rather than the greedy-min string the other arms fall back
    to: the pruned form is still a canonical form, so a substituted row
    stays inside the completeness theorem, whereas a greedy-min row does
    not.  The driver cascades to greedy-min only if pruned also exhausts
    the budget, which is what keeps "never drop a graph" true.
    """

    name = "isalgraph_exhaustive"
    variant = "canonical"
    fallback_variant = "pruned"
    capabilities = frozenset(
        {
            Capability.CANONICAL,
            Capability.COMPLETE_INVARIANT,
            Capability.REVERSIBLE,
        }
    )


register_backend("isalgraph_pruned", IsalGraphPruned)
register_backend("isalgraph_canonical", IsalGraphCanonical)
register_backend("isalgraph_exhaustive", IsalGraphExhaustive)

__all__ = [
    "ALPHABET_SIZE",
    "IsalGraphCanonical",
    "IsalGraphExhaustive",
    "IsalGraphPruned",
    "to_sparse_graph",
]
