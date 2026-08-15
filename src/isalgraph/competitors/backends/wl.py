"""Weisfeiler-Lehman subtree kernel -- a :class:`VectorBackend`, Claim B only.

Shervashidze, Schweitzer, van Leeuwen, Mehlhorn & Borgwardt,
*Weisfeiler-Lehman graph kernels*, JMLR 12:2539-2561, 2011.

The manuscript cites ``weisfeiler1968reduction`` at ``introduction.tex:27``.
**The kernel paper is a different citation and is missing.**

Why this is not a ``ReprBackend``
---------------------------------
WL is not a serialisation.  It has ``distance(a, b)`` and **no**
``encode() -> str`` and **no** bit count.  A feature-vector "bit cost"
(dimension x counter width) would measure our choice of container rather
than the encoding, and would be indefensible next to a reversible format.
``preregistration.md`` §4.1 already excludes WL from Claim A.  There is
deliberately no ``bits()`` on :class:`VectorBackend`, so fabricating one is
*unreachable* rather than merely forbidden, and ``bits.py`` raises
:class:`~isalgraph.errors.BitCountUndefined` for this backend name.

Role in the argument
--------------------
WL is the completeness witness.  ``K_{3,3}`` and the triangular prism are
both connected and 3-regular on six vertices and are not isomorphic; 1-WL
cannot separate two 3-regular graphs of the same order at **any** number of
rounds, because the colouring is constant after round 1 and refinement
never starts.  They receive kernel distance **exactly 0.0000** while every
other pool member separates them.  That 6-node witness is the cleanest
evidence in the folder for R1.2's uniqueness axis and it costs one small
figure.

The distance is therefore a **pseudometric**: identity of indiscernibles
fails.  ``competitors.md`` §3.3 F2 requires that be declared rather than
inferred; :class:`~isalgraph.competitors.metrics.kernel.KernelDistance`
declares ``is_pseudometric = True``.

Frozen parameters
-----------------
``h = 2``, ``normalize = False``, **fitted per dataset**.

* **``h = 2``, and do not tune it.**  ``h = 3`` is below ``h = 2`` on all
  five Suite-1 datasets.  Tuning ``h`` on the correlation with GED would be
  selecting a baseline on the outcome -- the same error ``competitors.md``
  §3.4 forbids for our own distances.  ``h`` is a constructor keyword only
  so the identity check can instantiate ``h in {1, 2, 3}`` without mutating
  a shared object.
* **``normalize = False``.**  ``normalize=True`` divides by
  ``sqrt(K(x,x) K(y,y))`` and removes the graph-size signal GED depends on,
  so a normalised kernel would look worse for reasons unrelated to WL.
  Passing ``normalize=True`` raises rather than silently producing a
  different measurement.

The convention, measured rather than assumed
---------------------------------------------
**There is no grakel off-by-one.**  ``grakel/kernels/weisfeiler_lehman.py``
sets ``self._n_iter = self.n_iter + 1`` and loops
``for i in range(1, self._n_iter)``, i.e. ``k`` refinement rounds *plus*
the base histogram, so **``grakel(n_iter=k) == ours(h=k)``**.  Measured on
the running example under GraKeL 0.1.10: ``n_iter=1 -> 2.000000``,
``n_iter=2 -> 5.830952``, ``n_iter=3 -> 7.211103``, all reproduced exactly
by :class:`WLSubtree`.  ``wl-subtree-kernel.md`` §1's warning that
``grakel(n_iter=3) == ours(h=2)`` is **wrong** and is superseded.

The off-by-one that does exist is ours, and it must not be ported.
``scratch/backends.py::wl_features`` compresses colours to small integers
**per graph, per round**, then builds the next round's signature from those
compressed labels.  The compression table comes from one graph's own
signature set, so **features from rounds >= 2 are not comparable across
graphs**.

This module avoids that class of bug structurally rather than by
convention: a colour is the ``blake2b`` digest of the round index and the
canonical signature built from *parent colours*, so a colour is a
deterministic function of local structure alone.  Two consequences:

1. :meth:`WLSubtree.features` does not depend on which graphs were fitted.
   A per-batch fit therefore **cannot** change a distance, which is the
   silent-corruption bug ``wl-subtree-kernel.md`` §7 warns about, made
   impossible instead of merely discouraged.
2. Two independent implementations -- this one and grakel -- agree to
   machine precision, so the WL row is auditable without a third-party
   version pin.

Digest collisions are the one residual risk and they are bounded: a 64-bit
digest over the few thousand distinct colours a Suite-2 dataset produces
has collision probability below ``1e-10``.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from isalgraph.competitors.base import Capability, VectorBackend
from isalgraph.competitors.registry import register_backend
from isalgraph.errors import BackendUnavailableError

if TYPE_CHECKING:
    import networkx as nx

    # Untyped third party; only ever a return annotation.  Importing it here
    # keeps ``grakel`` out of the runtime import path, which the subpackage's
    # dependency contract requires.
    from grakel import Graph as GrakelGraph

#: **Frozen.**  Refinement rounds.  ``grakel(n_iter=WL_ROUNDS)`` is the same
#: kernel.  Do not tune this on a correlation with GED.
WL_ROUNDS = 2

#: The colour every vertex starts with.  Our corpus is topology-only, so
#: there are no vertex labels and round 0 is a single constant -- which is
#: why the base histogram of a graph on ``n`` vertices contributes ``n^2``
#: to ``K(G, G)``.
BASE_COLOUR = "b"

#: Bytes of ``blake2b`` output per colour.
_DIGEST_BYTES = 8


def _refine(colour: str, neighbours: list[str], round_index: int) -> str:
    """One vertex's colour for the next round.

    Args:
        colour: this vertex's current colour.
        neighbours: the current colours of its neighbours, any order.
        round_index: the round being produced, ``1``-based.

    Returns:
        A digest that is a deterministic function of the signature, and
        therefore comparable across graphs.
    """
    signature = f"{round_index}#{colour}|" + ",".join(sorted(neighbours))
    return hashlib.blake2b(signature.encode("utf-8"), digest_size=_DIGEST_BYTES).hexdigest()


def wl_colours(graph: nx.Graph, h: int) -> list[dict[Any, str]]:
    """Vertex colours after each of rounds ``0 .. h``.

    Args:
        graph: any ``networkx.Graph``.  Disconnected graphs and isolated
            vertices are fine -- WL is defined on them, unlike the DFS code.
        h: number of refinement rounds.

    Returns:
        ``h + 1`` dictionaries, ``rounds[r][v]`` being ``v``'s colour after
        ``r`` refinements.
    """
    colour: dict[Any, str] = dict.fromkeys(graph.nodes(), BASE_COLOUR)
    rounds = [colour]
    for r in range(1, h + 1):
        colour = {
            v: _refine(colour[v], [colour[u] for u in graph.neighbors(v)], r) for v in graph.nodes()
        }
        rounds.append(colour)
    return rounds


def wl_features(graph: nx.Graph, h: int = WL_ROUNDS) -> dict[str, int]:
    """The WL subtree feature multiset of *graph*.

    Args:
        graph: any ``networkx.Graph``.
        h: refinement rounds.

    Returns:
        ``{'h<r>:<colour>': count}`` over rounds ``0 .. h``.  The round
        index is in the key as well as in the digest, so histograms from
        different rounds never merge.
    """
    counts: Counter[str] = Counter()
    for r, colour in enumerate(wl_colours(graph, h)):
        counts.update(f"h{r}:{c}" for c in colour.values())
    return dict(counts)


class WLSubtree(VectorBackend):
    """The WL subtree kernel's feature map, at a fixed number of rounds."""

    name = "wl_subtree"
    capabilities = frozenset(
        {
            Capability.CANONICAL,
            Capability.HANDLES_DISCONNECTED,
        }
    )

    def __init__(self, h: int = WL_ROUNDS, *, normalize: bool = False) -> None:
        """Build a WL feature map.

        Args:
            h: refinement rounds.  Defaults to the frozen
                :data:`WL_ROUNDS`.  **Do not tune this on a correlation
                with GED**; the keyword exists so the grakel identity check
                can instantiate ``h in {1, 2, 3}`` without mutating a
                shared object.
            normalize: must be ``False``.

        Raises:
            ValueError: if *h* is negative, or if *normalize* is ``True``.
        """
        if h < 0:
            raise ValueError(f"h must be non-negative, got {h!r}")
        if normalize:
            raise ValueError(
                "normalize=True is not supported and is not the frozen choice: "
                "dividing by sqrt(K(x,x) K(y,y)) removes the graph-size signal "
                "GED depends on, so a normalised kernel looks worse for reasons "
                "unrelated to WL. See wl-subtree-kernel.md §3."
            )
        self.h = h
        self.normalize = normalize
        self._vocabulary: tuple[str, ...] = ()
        self._n_fitted = 0

    def fit(self, graphs: Sequence[nx.Graph]) -> None:
        """Record the colour vocabulary of *graphs*.

        Fitting is **per dataset, never per batch** by contract.  Here it is
        also *inert with respect to the distance*: colours are digests of
        canonical signatures, so :meth:`features` returns the same multiset
        whatever was fitted.  That is deliberate -- it makes the
        distance matrix independent of batching order by construction rather
        than by discipline, and a test asserts it.

        Args:
            graphs: the whole dataset.
        """
        vocabulary: set[str] = set()
        for graph in graphs:
            vocabulary.update(wl_features(graph, self.h))
        self._vocabulary = tuple(sorted(vocabulary))
        self._n_fitted = len(graphs)

    def features(self, graph: nx.Graph) -> Mapping[str, int]:
        """The fitted feature multiset for *graph*.

        Args:
            graph: any ``networkx.Graph``.

        Returns:
            ``{'h<r>:<colour>': count}``.  **Not** restricted to the fitted
            vocabulary: dropping unseen colours would reintroduce the
            batching dependence this design exists to remove.
        """
        return wl_features(graph, self.h)

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Colours seen during :meth:`fit`, sorted.  Reporting only."""
        return self._vocabulary

    @property
    def n_fitted(self) -> int:
        """How many graphs the last :meth:`fit` saw.  Reporting only."""
        return self._n_fitted

    def distance(self, a: nx.Graph, b: nx.Graph) -> float:
        """RKHS distance ``sqrt(K(a,a) + K(b,b) - 2 K(a,b))``.

        Args:
            a: a graph.
            b: a graph.

        Returns:
            The kernel distance.  **A pseudometric**: identity of
            indiscernibles fails, ``d(K_{3,3}, prism) == 0``.
        """
        from isalgraph.competitors.metrics.kernel import KernelDistance

        return KernelDistance().distance(self.features(a), self.features(b))


# ---------------------------------------------------------------------------
# The second, independent implementation
# ---------------------------------------------------------------------------


def grakel_available() -> bool:
    """Whether ``grakel`` imports."""
    try:
        import grakel  # noqa: F401
    except ImportError:
        return False
    return True


def _to_grakel(graph: nx.Graph) -> GrakelGraph:
    """One ``grakel.Graph`` with a single constant vertex label.

    The **edge-dictionary** form, not an edge list: ``grakel.Graph`` reads an
    empty edge list as an empty adjacency matrix and dies with
    ``IndexError: tuple index out of range``, so an edgeless or
    isolated-vertex graph could not be converted at all.  The dictionary
    form carries the vertex set explicitly and handles both.
    """
    from grakel import Graph

    adjacency: dict[Any, dict[Any, float]] = {v: {} for v in graph.nodes()}
    for u, v in graph.edges():
        adjacency[u][v] = 1.0
        adjacency[v][u] = 1.0
    return Graph(
        adjacency,
        node_labels=dict.fromkeys(graph.nodes(), BASE_COLOUR),
        graph_format="all",
    )


def grakel_gram(graphs: Sequence[nx.Graph], *, h: int = WL_ROUNDS) -> list[list[float]]:
    """Gram matrix from ``grakel``, the independent implementation.

    ``n_iter = h``: there is no off-by-one.  ``grakel`` runs ``h``
    refinement rounds plus the base histogram, which is exactly what
    :func:`wl_features` accumulates.

    Args:
        graphs: the graphs to compare, fitted together.
        h: refinement rounds.

    Returns:
        The ``len(graphs) x len(graphs)`` unnormalised Gram matrix.

    Raises:
        BackendUnavailableError: if ``grakel`` is not installed.
    """
    try:
        from grakel import VertexHistogram, WeisfeilerLehman
    except ImportError as exc:  # pragma: no cover - exercised by the dep test
        raise BackendUnavailableError(
            "grakel is required for the independent WL cross-check; the shipped "
            "backend 'wl_subtree' needs no third-party library"
        ) from exc

    kernel = WeisfeilerLehman(n_iter=h, base_graph_kernel=VertexHistogram, normalize=False)
    gram = kernel.fit_transform([_to_grakel(g) for g in graphs])
    return [[float(x) for x in row] for row in gram]


def grakel_distance(a: nx.Graph, b: nx.Graph, *, h: int = WL_ROUNDS) -> float:
    """Kernel distance between two graphs via ``grakel``.

    Args:
        a: a graph.
        b: a graph.
        h: refinement rounds.

    Returns:
        ``sqrt(K(a,a) + K(b,b) - 2 K(a,b))``.
    """
    import math

    gram = grakel_gram([a, b], h=h)
    squared = gram[0][0] + gram[1][1] - 2.0 * gram[0][1]
    return math.sqrt(max(squared, 0.0))


register_backend("wl_subtree", WLSubtree)

__all__ = [
    "BASE_COLOUR",
    "WL_ROUNDS",
    "WLSubtree",
    "grakel_available",
    "grakel_distance",
    "grakel_gram",
    "wl_colours",
    "wl_features",
]
