"""Constructed graph families with closed-form ``|Aut|``, for T-13's controlled arm.

Why constructed graphs at all.  On the real IAM cohort ``n``, ``m``, density and
``|Aut|`` all move together, so the marginal correlation between ``log|Aut|`` and
``log t`` is only ``+0.189`` against ``+0.326`` for ``log n``; only a
within-fixed-``(n, m)`` contrast recovers the true effect (``+0.655``, positive
in 12 of 13 strata).  An observational study on that cohort therefore cannot
establish the cost law, and the primary evidence has to come from graphs where
one factor moves at a time.  These are those graphs.

Every deterministic family carries its ``|Aut|`` in closed form, and
:func:`build` **verifies the closed form against nauty on every single graph it
returns** -- a family whose measured ``|Aut|`` disagrees with its formula is a
bug in this file, not a datum, and raises :class:`FamilyVerificationError`
(`T-13-design.md` §3 rule 6).

The matched designs: ``symmetry_ladder`` and ``spider_ladder``
--------------------------------------------------------------

The other nine families each break one confound.  The two ladders break all of
them at once and are the arms the primary analysis rule (§3 rule 7) is written
against: along a ladder ``n``, ``m`` and the whole degree sequence are held
exactly while ``|Aut|`` falls.  They reach that the same guarantee by different
routes and, critically, **at different densities** -- ``spider_ladder`` is a
tree (``m = n - 1``), ``Q_d`` has ``m/n = d/2``, ``K_{a,a}`` has ``m = n^2/4``.
Three ladders at matched ``n`` and three densities turn *"is the effect really
density?"* from an objection into a measured answer, which matters because the
real cohort is sparse (IAM mean density 0.094-0.607) and a law shown only on
dense graphs would not transfer.

``symmetry_ladder``
~~~~~~~~~~~~~~~~~~~

A ladder starts from a ``d``-regular base with a large closed-form
``|Aut|`` and applies ``k = 0, 1, 2, 4, 8, 16, 32`` **degree-preserving double
edge swaps**, rejecting any swap that disconnects the graph.  Along a ladder
``n``, ``m`` **and the entire degree sequence** are exactly constant while
``|Aut|`` falls, which closes the objection that the effect is really maximum
degree -- an objection none of the other families can close on its own.  It
terminates in a rigid graph because random ``d``-regular graphs with ``d >= 3``
are a.a.s. rigid (Bollobás, *European J. Combin.* 1(4):311-316, 1980).

Two consequences of that, both deliberate:

- **A 2-regular base is unusable.**  On ``C_n`` a connectivity-preserving
  degree-preserving swap either disconnects the graph or returns a cycle, so the
  ladder has no rungs.  ``cycle`` is a family here but never a ladder base.
- **Only super-polynomial ``|Aut|`` spans the range.**  For a sparse regular base
  ``|Aut|`` is polynomial in ``n`` -- the prism's is ``4a`` -- so ``log10|Aut|``
  never exceeds ``2.2`` at ``n <= 64`` and the ladder cannot fall three orders of
  magnitude however many swaps it takes.  The bases used are therefore
  ``K_{a,a}`` (``2(a!)^2``, at every size) and ``Q_d`` (``2^d d!``, where ``n`` is
  a power of two), the two ``d``-regular graphs whose group is factorial.
  :func:`ladder_spans` reports the realised span of every ladder.

**Non-monotone rungs are kept.**  A rung whose ``|Aut|`` failed to drop is
recorded as measured.  Discarding it would select on the outcome, which is the
one thing this ticket cannot afford.

``spider_ladder`` -- the sparse matched design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A spider is a hub with ``k >= 3`` disjoint paths ("legs") of lengths
``L_1..L_k`` summing to ``n - 1``.  It is the sparsest connected graph there is
and still reaches a factorial group, which no regular base does at these orders.

The three quantities are fixed **by construction, not by a check**, which is
what makes this ladder stronger than a swap-driven one:

- ``n = 1 + sum L_i``, and :func:`spider_legs` displaces leg lengths
  antisymmetrically, so the sum is invariant;
- ``m = n - 1``, because every spider is a tree;
- the **degree sequence** is ``(k, 2^(n-1-k), 1^k)`` at every rung: the hub
  always has degree ``k``, there are always exactly ``k`` leaves, and the
  degree-2 count is ``sum_i (L_i - 1) = (n - 1) - k`` **whatever the
  partition**.

``|Aut| = prod_d (m_d)!`` where ``m_d`` counts the legs of length ``d``: for
``k >= 3`` the hub is the unique vertex of degree ``>= 3`` and hence fixed, each
leg is rigid once the hub is, so an automorphism is exactly a length-preserving
permutation of legs.  Rung ``j`` leaves ``k - 2j`` legs equal, giving
``|Aut| = (k - 2j)!`` and a span of ``log10(k!)``: 4.61 at ``k = 8``, 6.56 at
``k = 10``.  ``k = 2`` is excluded -- the hub would have degree 2, the spider is
a path, and ``|Aut| = 2`` however unequal the legs are.

Determinism
-----------

Every random draw is seeded from ``(seed, family, n, replicate, params)`` through
a BLAKE2b digest, so :func:`build` is a pure function of its arguments and a
shard can rebuild any graph from its spec alone without shipping the graph.
Every returned graph is relabelled onto ``range(n)``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx

from benchmarks.real_data.eval_t13_complexity import symmetry

log = logging.getLogger(__name__)

__all__ = [
    "AUT_TOLERANCE",
    "FAMILIES",
    "LADDER_BASES",
    "LADDER_SWAPS",
    "SIZES",
    "SPIDER_CELLS",
    "FamilySpec",
    "FamilyVerificationError",
    "build",
    "enumerate_grid",
    "ladder_spans",
    "spider_legs",
    "spider_rungs",
]


class FamilyVerificationError(RuntimeError):
    """A constructed graph failed the invariant its family guarantees.

    Raised when a measured ``|Aut|`` disagrees with the family's closed form,
    when a ``rigid_er`` draw cannot be made rigid, when a ladder rung moved
    ``n`` or ``m``, or when a graph came out disconnected.  It is an abort, not a
    datum: `T-13-design.md` §3 rule 6 requires the run to stop rather than record
    a graph that is not what its spec says it is.
    """


#: The families, in the order the design note tabulates them.  ``spider_ladder``
#: is not in `T-13-design.md` §2.3's table: it was added mid-wave, on the
#: orchestrator's instruction, to give the matched design a **tree-density** arm
#: (see the module docstring).
FAMILIES: tuple[str, ...] = (
    "path",
    "cycle",
    "star",
    "complete",
    "complete_bipartite",
    "hypercube",
    "prism",
    "caterpillar",
    "rigid_er",
    "symmetry_ladder",
    "spider_ladder",
)

#: The requested orders of the factorial grid (`T-13-design.md` §2.3).  Each
#: family snaps these to its nearest realisable order.
SIZES: tuple[int, ...] = (8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64)

#: Ladder bases, indexed by the integer stored under ``params["base"]``.
#: ``params`` values must be ints so that a :class:`FamilySpec` stays hashable
#: under its declared type, so the base is carried as its index here.
LADDER_BASES: tuple[str, ...] = ("complete_bipartite", "hypercube")

#: Swap counts per ladder.  Geometric rather than consecutive because ``|Aut|``
#: falls fast at first and then plateaus, so the informative rungs are early.
LADDER_SWAPS: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)

#: ``(legs, leg_length)`` of each ``spider_ladder``, realising ``n = k * L + 1``.
#: A **fixed design**, not a size sweep: a spider's order is ``k * L + 1``, so
#: snapping it to a requested size would have to move ``k`` (which changes the
#: span, ``log10(k!)``) or ``L`` (which changes the density).  Neither is a free
#: parameter here, so :func:`enumerate_grid` emits these cells whatever *sizes*
#: it is given.  ``(3, 3)`` realises ``n = 10`` and is the ``n <= 12``
#: consistency gate, where exhaustive search and exact GED are both tractable.
SPIDER_CELLS: tuple[tuple[int, int], ...] = ((3, 3), (10, 3), (8, 4), (10, 6), (8, 8))

#: Edge probability for ``rigid_er``, in percent (``params`` carries ints).
#: ``G(n, 0.5)`` is the maximum-entropy draw at fixed ``n`` and is rigid with
#: high probability for ``n >= 8``; :func:`build` resamples until it is.
RIGID_ER_P_PERCENT = 50

#: Resampling budget for ``rigid_er`` before the run aborts (`CONTRACTS` §3).
RIGID_ER_MAX_DRAWS = 50

#: Attempts to find a connectivity-preserving swap for one ladder rung.
_SWAP_ATTEMPTS = 200

#: Absolute tolerance on ``log10|Aut|`` when checking a closed form.  nauty
#: reports a 15-significant-digit mantissa, so the true agreement is ~1e-12 and
#: this is three orders looser than the mechanism needs.
AUT_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """One cell of the constructed grid: enough to rebuild the graph exactly.

    Attributes:
        family: one of :data:`FAMILIES`.
        n: the **realised** order, after the family snapped the requested size to
            something it can build (``hypercube`` to a power of two, and so on).
            There is no separate ``n_target`` field: two requested sizes that
            snap to the same realisable order are the same graph, and carrying
            the request would defeat the de-duplication in
            :func:`enumerate_grid`.
        replicate: replicate index for the random families, ``0`` for the
            deterministic ones and for rung 0 of a ladder, which has no random
            component.
        params: family-specific integer parameters, e.g. ``(("swaps", 4),
            ("base", 0))``.  Integers only, so the dataclass stays hashable and
            the spec serialises to JSON without a custom encoder.
        log10_aut_expected: the closed form, or ``None`` where none exists.
            ``rigid_er`` is ``None`` but is verified as ``0.0``;
            ``symmetry_ladder`` is ``None`` and its ``|Aut|`` is measured.
    """

    family: str
    n: int
    replicate: int
    params: tuple[tuple[str, int], ...]
    log10_aut_expected: float | None


# ----------------------------------------------------------------------------
# Closed forms
# ----------------------------------------------------------------------------


def _log10_factorial(k: int) -> float:
    """``log10(k!)`` exactly.

    ``math.factorial`` is arbitrary-precision and ``math.log10`` accepts a
    Python ``int`` of any size, so this stays exact where a ``lgamma``
    approximation would drift into the 1e-6 tolerance at large ``k``.

    Args:
        k: a non-negative integer.

    Returns:
        ``log10(k!)``.  ``0.0`` for ``k <= 1``.
    """
    if k <= 1:
        return 0.0
    return math.log10(math.factorial(k))


def _params(**kwargs: int) -> tuple[tuple[str, int], ...]:
    """Build a ``params`` tuple with a deterministic key order."""
    return tuple(sorted(kwargs.items()))


def _param(spec: FamilySpec, key: str) -> int:
    """Read one integer parameter off a spec.

    Args:
        spec: the spec to read.
        key: the parameter name.

    Returns:
        The parameter value.

    Raises:
        KeyError: if the spec does not carry that parameter.
    """
    for name, value in spec.params:
        if name == key:
            return value
    raise KeyError(f"{spec.family} spec carries no parameter {key!r}: {spec.params!r}")


# ----------------------------------------------------------------------------
# Deterministic constructors
# ----------------------------------------------------------------------------


def _relabel(graph: nx.Graph) -> nx.Graph:
    """Return *graph* on ``range(n)`` with integer labels.

    ``nx.hypercube_graph`` labels vertices with bit tuples and ``nx`` graph
    products label with pairs; every consumer downstream wants ``0..n-1``
    because that is what ``SparseGraph`` indexes with.
    """
    relabelled: nx.Graph = nx.convert_node_labels_to_integers(
        graph, first_label=0, ordering="default"
    )
    return relabelled


def _build_caterpillar(spine: int, doubles: int) -> nx.Graph:
    """A caterpillar: a spine path with two leaves on each of its first *doubles* nodes.

    The automorphism group of a caterpillar with spine leaf-counts
    ``(l_1, ..., l_s)`` is ``(prod_i l_i!)`` extended by ``Z_2`` when the
    sequence is a palindrome.  This construction puts ``l_i = 2`` on a
    contiguous prefix and ``0`` elsewhere, which makes the sequence
    non-palindromic for ``0 < doubles < spine`` and hence gives exactly
    ``|Aut| = 2^doubles`` with no mirror factor.  Verified against nauty at all
    twelve grid sizes.

    Args:
        spine: number of spine vertices.
        doubles: number of spine vertices carrying two leaves.

    Returns:
        A caterpillar on ``spine + 2 * doubles`` vertices.
    """
    graph = nx.path_graph(spine)
    nxt = spine
    for i in range(doubles):
        for _ in range(2):
            graph.add_edge(i, nxt)
            nxt += 1
    return graph


def spider_legs(k: int, leg: int, rung: int) -> tuple[int, ...]:
    """Leg lengths of one ``spider_ladder`` rung.

    Rung ``j`` lengthens the first ``j`` legs to ``L+j, ..., L+1`` and shortens
    the next ``j`` to ``L-1, ..., L-j``, leaving ``k - 2j`` legs at ``L``.  The
    displacement is antisymmetric, so ``sum(lengths) = k * L`` at every rung and
    ``n`` is fixed by construction rather than by a check.  ``|Aut|`` falls as
    ``(k - 2j)!`` because exactly that many legs remain interchangeable.

    Args:
        k: number of legs.  Must be ``>= 3``: at ``k = 2`` the hub has degree 2,
            the spider is a path, and ``|Aut| = 2`` regardless of the partition.
        leg: the balanced leg length, i.e. rung 0's.
        rung: the rung index ``j``.

    Returns:
        The ``k`` leg lengths, all ``>= 1``.

    Raises:
        FamilyVerificationError: if the parameters are out of range, in
            particular if ``rung`` would drive a leg to length 0.
    """
    if k < 3:
        raise FamilyVerificationError(f"a spider needs k >= 3 legs; k = {k} is a path")
    if leg < 1:
        raise FamilyVerificationError(f"spider leg length must be >= 1, got {leg}")
    if rung < 0 or 2 * rung > k or rung > leg - 1:
        raise FamilyVerificationError(
            f"spider rung {rung} is out of range for k={k}, leg={leg}; "
            f"the ladder has rungs 0..{min(k // 2, leg - 1)}"
        )
    lengths = [leg] * k
    for i in range(rung):
        lengths[i] = leg + (rung - i)
        lengths[rung + i] = leg - (i + 1)
    return tuple(lengths)


def spider_rungs(k: int, leg: int) -> tuple[int, ...]:
    """The rung indices a ``(k, leg)`` spider ladder admits."""
    return tuple(range(min(k // 2, leg - 1) + 1))


def _log10_spider_aut(lengths: Sequence[int]) -> float:
    """``log10|Aut|`` of a spider, in closed form.

    For ``k >= 3`` the hub is the unique vertex of degree ``>= 3`` and is fixed
    by every automorphism; each leg is a path attached at one end, so it has no
    internal automorphism once the hub is fixed.  An automorphism is therefore
    exactly a permutation of the legs that preserves length, giving
    ``|Aut| = prod_d (m_d)!`` where ``m_d`` counts the legs of length ``d``.
    Verified against nauty on every rung of every cell.

    Args:
        lengths: the leg lengths.

    Returns:
        ``log10(prod_d m_d!)``.
    """
    counts = Counter(lengths)
    return sum(_log10_factorial(m) for m in counts.values())


def _build_spider(lengths: Sequence[int]) -> nx.Graph:
    """A spider: a hub with disjoint paths of the given lengths attached."""
    graph = nx.Graph()
    graph.add_node(0)
    nxt = 1
    for length in lengths:
        previous = 0
        for _ in range(length):
            graph.add_edge(previous, nxt)
            previous = nxt
            nxt += 1
    return graph


def _ladder_base(base: str, n: int) -> nx.Graph:
    """The rung-0 graph of a ladder.

    Args:
        base: one of :data:`LADDER_BASES`.
        n: the realised order.

    Returns:
        A connected ``d``-regular graph on ``n`` vertices with ``d >= 3``.

    Raises:
        FamilyVerificationError: if *base* cannot realise *n*.
    """
    if base == "complete_bipartite":
        if n % 2 or n < 8:
            raise FamilyVerificationError(f"K_a,a ladder base needs even n >= 8, got {n}")
        return _relabel(nx.complete_bipartite_graph(n // 2, n // 2))
    if base == "hypercube":
        d = round(math.log2(n))
        if 2**d != n or d < 3:
            raise FamilyVerificationError(f"Q_d ladder base needs n a power of two >= 8, got {n}")
        return _relabel(nx.hypercube_graph(d))
    raise FamilyVerificationError(f"unknown ladder base {base!r}; known: {LADDER_BASES}")


# ----------------------------------------------------------------------------
# Randomised constructors
# ----------------------------------------------------------------------------


def _spec_rng(spec: FamilySpec, seed: int) -> random.Random:
    """A deterministic RNG for one spec.

    Derived by digest rather than by arithmetic on the seed so that two specs
    differing only in, say, ``replicate`` cannot land on correlated streams.

    Args:
        spec: the spec being built.
        seed: the campaign seed.

    Returns:
        A ``random.Random`` seeded from ``(seed, family, n, replicate, params)``.
    """
    key = f"{seed}|{spec.family}|{spec.n}|{spec.replicate}|{spec.params}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _build_rigid_er(n: int, p_percent: int, rng: random.Random) -> nx.Graph:
    """A connected, rigid ``G(n, p)`` draw.

    "Rigid" means ``|Aut| = 1``, i.e. ``log10|Aut| == 0``.  This is the control
    that pins the low end of the ``|Aut|`` axis at every density: whatever the
    order and edge count, its search has no symmetry to explore.

    Args:
        n: order.
        p_percent: edge probability in percent.
        rng: the spec's RNG.

    Returns:
        A connected rigid graph on ``range(n)``.

    Raises:
        FamilyVerificationError: if no rigid connected draw appeared within
            :data:`RIGID_ER_MAX_DRAWS` attempts.
    """
    p = p_percent / 100.0
    for draw in range(RIGID_ER_MAX_DRAWS):
        graph = nx.gnp_random_graph(n, p, seed=rng.randint(0, 2**31 - 1))
        if not nx.is_connected(graph):
            continue
        if abs(symmetry.log10_aut(graph)) <= AUT_TOLERANCE:
            log.debug("rigid_er n=%d found on draw %d", n, draw + 1)
            return graph
    raise FamilyVerificationError(
        f"rigid_er could not draw a connected rigid G({n}, {p}) in "
        f"{RIGID_ER_MAX_DRAWS} attempts; the density is wrong for this order"
    )


def _apply_swaps(base: nx.Graph, swaps: int, rng: random.Random) -> nx.Graph:
    """Apply *swaps* connectivity-preserving double edge swaps.

    ``nx.double_edge_swap`` is degree-preserving but not connectivity-preserving,
    so each swap is applied to a copy and accepted only if the result is still
    connected.  Degree preservation is what makes the ladder a matched design:
    ``n``, ``m`` and the whole degree sequence are identical at every rung.

    Args:
        base: the rung-0 graph.  Not modified.
        swaps: number of accepted swaps to apply.
        rng: the spec's RNG.

    Returns:
        A fresh connected graph with the same degree sequence as *base*.

    Raises:
        FamilyVerificationError: if no connectivity-preserving swap was found
            within :data:`_SWAP_ATTEMPTS` attempts for some rung.
    """
    graph = base.copy()
    for step in range(swaps):
        for _ in range(_SWAP_ATTEMPTS):
            candidate = graph.copy()
            try:
                nx.double_edge_swap(candidate, nswap=1, max_tries=100, seed=rng)
            except (nx.NetworkXAlgorithmError, nx.NetworkXError):
                continue
            if nx.is_connected(candidate):
                graph = candidate
                break
        else:
            raise FamilyVerificationError(
                f"no connectivity-preserving double edge swap found for swap {step + 1} "
                f"of {swaps} on an n={base.number_of_nodes()} m={base.number_of_edges()} base "
                f"in {_SWAP_ATTEMPTS} attempts"
            )
    return graph


# ----------------------------------------------------------------------------
# The public constructor
# ----------------------------------------------------------------------------


def _construct(spec: FamilySpec, rng: random.Random) -> nx.Graph:
    """Dispatch to the family constructor.  No verification here."""
    n = spec.n
    if spec.family == "path":
        return nx.path_graph(n)
    if spec.family == "cycle":
        return nx.cycle_graph(n)
    if spec.family == "star":
        return nx.star_graph(n - 1)
    if spec.family == "complete":
        return nx.complete_graph(n)
    if spec.family == "complete_bipartite":
        return _relabel(nx.complete_bipartite_graph(n // 2, n // 2))
    if spec.family == "hypercube":
        return _relabel(nx.hypercube_graph(_param(spec, "dimension")))
    if spec.family == "prism":
        return nx.circular_ladder_graph(n // 2)
    if spec.family == "caterpillar":
        return _build_caterpillar(_param(spec, "spine"), _param(spec, "doubles"))
    if spec.family == "rigid_er":
        return _build_rigid_er(n, _param(spec, "p_percent"), rng)
    if spec.family == "symmetry_ladder":
        base = _ladder_base(LADDER_BASES[_param(spec, "base")], n)
        return _apply_swaps(base, _param(spec, "swaps"), rng)
    if spec.family == "spider_ladder":
        return _build_spider(
            spider_legs(_param(spec, "legs"), _param(spec, "leg"), _param(spec, "rung"))
        )
    raise FamilyVerificationError(f"unknown family {spec.family!r}; known: {FAMILIES}")


def build(spec: FamilySpec, *, seed: int) -> nx.Graph:
    """Build the graph *spec* describes, and verify everything it promises.

    Verification is not optional and not deferred to a test.  A constructed
    graph whose measured ``|Aut|`` disagrees with its closed form is a
    construction bug, and recording it would put a mislabelled point on the
    primary axis of the experiment; `T-13-design.md` §3 rule 6 requires the run
    to abort instead.  What is checked, for every spec:

    - the realised order equals ``spec.n``;
    - the graph is connected (every family here is);
    - ``|log10|Aut| - spec.log10_aut_expected| <= AUT_TOLERANCE`` where a closed
      form exists;
    - ``rigid_er`` is rigid, which is its closed form in all but name;
    - a ladder rung has the same ``n``, ``m`` and degree sequence as its rung-0
      base, which is the property the matched design rests on.

    Args:
        spec: the cell to build.
        seed: the campaign seed.  Combined with the spec's own fields, so the
            same ``(spec, seed)`` always yields the same graph.

    Returns:
        A fresh ``networkx.Graph`` on ``range(spec.n)``.

    Raises:
        FamilyVerificationError: on any of the failures above.
    """
    rng = _spec_rng(spec, seed)
    graph = _relabel(_construct(spec, rng))

    n = int(graph.number_of_nodes())
    if n != spec.n:
        raise FamilyVerificationError(
            f"{spec.family} built {n} vertices for a spec declaring n={spec.n}"
        )
    if n and not nx.is_connected(graph):
        raise FamilyVerificationError(f"{spec.family} n={n} came out disconnected")

    if spec.family == "symmetry_ladder":
        base = _ladder_base(LADDER_BASES[_param(spec, "base")], spec.n)
        expected_degrees = sorted(d for _v, d in base.degree())
        actual_degrees = sorted(d for _v, d in graph.degree())
        if actual_degrees != expected_degrees:
            raise FamilyVerificationError(
                f"symmetry_ladder n={spec.n} swaps={_param(spec, 'swaps')} changed the degree "
                f"sequence; the swap is supposed to preserve it"
            )
        if int(graph.number_of_edges()) != int(base.number_of_edges()):
            raise FamilyVerificationError(
                f"symmetry_ladder n={spec.n} moved m from {base.number_of_edges()} to "
                f"{graph.number_of_edges()}"
            )

    if spec.family == "spider_ladder":
        k = _param(spec, "legs")
        degrees = sorted((d for _v, d in graph.degree()), reverse=True)
        # (k, 2^(n-1-k), 1^k) at every rung, whatever the leg partition.
        expected_degrees = [k] + [2] * (spec.n - 1 - k) + [1] * k
        if degrees != expected_degrees:
            raise FamilyVerificationError(
                f"spider_ladder n={spec.n} params={spec.params} has degree sequence "
                f"{degrees}, not the invariant {expected_degrees}"
            )
        if int(graph.number_of_edges()) != spec.n - 1:
            raise FamilyVerificationError(
                f"spider_ladder n={spec.n} has {graph.number_of_edges()} edges, not the "
                f"{spec.n - 1} a tree must have"
            )

    if spec.family == "rigid_er":
        measured = symmetry.log10_aut(graph)
        if abs(measured) > AUT_TOLERANCE:
            raise FamilyVerificationError(
                f"rigid_er n={spec.n} replicate={spec.replicate} is not rigid: "
                f"log10|Aut| = {measured!r}"
            )
    elif spec.log10_aut_expected is not None:
        measured = symmetry.log10_aut(graph)
        if abs(measured - spec.log10_aut_expected) > AUT_TOLERANCE:
            raise FamilyVerificationError(
                f"{spec.family} n={spec.n} params={spec.params}: measured log10|Aut| = "
                f"{measured!r} but the closed form says {spec.log10_aut_expected!r} "
                f"(difference {measured - spec.log10_aut_expected!r})"
            )

    return graph


# ----------------------------------------------------------------------------
# The grid
# ----------------------------------------------------------------------------


def _spec_path(n: int) -> FamilySpec | None:
    if n < 2:
        return None
    return FamilySpec("path", n, 0, (), math.log10(2.0))


def _spec_cycle(n: int) -> FamilySpec | None:
    if n < 3:
        return None
    return FamilySpec("cycle", n, 0, (), math.log10(2.0 * n))


def _spec_star(n: int) -> FamilySpec | None:
    # n = 2 is K_2, whose |Aut| is 2 rather than (n-1)! = 1; excluded so the
    # closed form holds without a special case.  The grid starts at n = 8.
    if n < 3:
        return None
    return FamilySpec("star", n, 0, (), _log10_factorial(n - 1))


def _spec_complete(n: int) -> FamilySpec | None:
    if n < 2:
        return None
    return FamilySpec("complete", n, 0, (), _log10_factorial(n))


def _spec_complete_bipartite(n: int) -> FamilySpec | None:
    a = n // 2
    if a < 2 or 2 * a != n:
        return None
    return FamilySpec("complete_bipartite", n, 0, (), math.log10(2.0) + 2.0 * _log10_factorial(a))


def _spec_hypercube(n: int) -> FamilySpec | None:
    """Snap *n* to the nearest power of two in log space, ``d >= 3``."""
    d = max(3, round(math.log2(n)))
    realised = 2**d
    expected = d * math.log10(2.0) + _log10_factorial(d)
    return FamilySpec("hypercube", realised, 0, _params(dimension=d), expected)


def _spec_prism(n: int) -> FamilySpec | None:
    """``C_a x K_2``.  ``|Aut| = 4a`` **except at a = 4**, where the prism is ``Q_3``.

    ``CONTRACTS`` §3 flags ``a = 3`` as the exception and requires ``a >= 4``.
    That is backwards, and measurement settles it: the 3-prism has
    ``|Aut| = 12 = 4 * 3`` exactly, while the 4-prism *is* the cube ``Q_3`` and
    has ``|Aut| = 48``, not ``16``.  Both are in the grid; ``a = 4`` gets the
    hypercube formula.
    """
    a = n // 2
    if a < 3 or 2 * a != n:
        return None
    expected = 3 * math.log10(2.0) + _log10_factorial(3) if a == 4 else math.log10(4.0 * a)
    return FamilySpec("prism", n, 0, _params(a=a), expected)


def _spec_caterpillar(n: int) -> FamilySpec | None:
    """Spine of ``n - 2k`` vertices with two leaves on each of the first ``k``.

    ``k = n // 4`` keeps ``0 < k < spine`` at every grid size, which is exactly
    the condition under which the leaf sequence is non-palindromic and
    ``|Aut| = 2^k`` holds with no mirror factor.
    """
    doubles = max(1, n // 4)
    spine = n - 2 * doubles
    if spine < 3 or doubles >= spine:
        return None
    return FamilySpec(
        "caterpillar", n, 0, _params(doubles=doubles, spine=spine), doubles * math.log10(2.0)
    )


def _spec_rigid_er(n: int, replicate: int) -> FamilySpec | None:
    if n < 4:
        return None
    return FamilySpec("rigid_er", n, replicate, _params(p_percent=RIGID_ER_P_PERCENT), None)


def _spec_ladder(n: int, base_index: int, swaps: int, replicate: int) -> FamilySpec | None:
    base = LADDER_BASES[base_index]
    if base == "complete_bipartite" and (n % 2 or n < 8):
        return None
    if base == "hypercube" and (n < 8 or 2 ** round(math.log2(n)) != n):
        return None
    # Rung 0 has no random component, so all replicates of it are the same
    # graph; forcing replicate 0 lets de-duplication collapse them.
    return FamilySpec(
        "symmetry_ladder",
        n,
        0 if swaps == 0 else replicate,
        _params(base=base_index, swaps=swaps),
        None,
    )


def _spec_spider(k: int, leg: int, rung: int) -> FamilySpec:
    """One ``spider_ladder`` rung, with its closed-form ``|Aut|``.

    Unlike ``symmetry_ladder``, whose ``|Aut|`` can only be measured, every
    spider rung has an exact formula, so :func:`build` verifies this ladder the
    same way it verifies ``complete`` or ``hypercube``.
    """
    lengths = spider_legs(k, leg, rung)
    return FamilySpec(
        "spider_ladder",
        1 + sum(lengths),
        0,
        _params(leg=leg, legs=k, rung=rung),
        _log10_spider_aut(lengths),
    )


def enumerate_grid(*, sizes: Sequence[int], replicates: int, seed: int) -> tuple[FamilySpec, ...]:
    """The full constructed grid, de-duplicated and in a deterministic order.

    Deterministic families contribute one spec per realisable order;
    ``rigid_er`` and ``symmetry_ladder`` contribute *replicates* each, except at
    ladder rung 0, which is deterministic and collapses.  De-duplication is on
    the whole spec, which is what makes the twelve requested sizes collapse to
    four hypercubes without leaving eight duplicate rows in the campaign.

    Args:
        sizes: requested orders.  Each family snaps them to what it can build.
        replicates: replicate count for the random families.
        seed: the campaign seed.  It does not change *which* specs exist -- the
            grid is a fixed design -- and is accepted so that the signature says
            so and so that a future stochastic design choice has a home.

    Returns:
        A tuple of unique :class:`FamilySpec`, ordered by family (in
        :data:`FAMILIES` order), then ``n``, then ``params``, then ``replicate``.

    Raises:
        ValueError: if *replicates* is not positive.
    """
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    del seed  # documented above: the design is fixed, not sampled

    specs: list[FamilySpec] = []
    for n in sizes:
        specs.extend(
            spec
            for spec in (
                _spec_path(n),
                _spec_cycle(n),
                _spec_star(n),
                _spec_complete(n),
                _spec_complete_bipartite(n),
                _spec_hypercube(n),
                _spec_prism(n),
                _spec_caterpillar(n),
            )
            if spec is not None
        )
        for replicate in range(replicates):
            spec = _spec_rigid_er(n, replicate)
            if spec is not None:
                specs.append(spec)
            for base_index in range(len(LADDER_BASES)):
                for swaps in LADDER_SWAPS:
                    ladder = _spec_ladder(n, base_index, swaps, replicate)
                    if ladder is not None:
                        specs.append(ladder)

    # A spider's order is k * L + 1, so its cells are a fixed design rather than
    # a snap of `sizes`; see SPIDER_CELLS.
    for k, leg in SPIDER_CELLS:
        specs.extend(_spec_spider(k, leg, rung) for rung in spider_rungs(k, leg))

    unique = dict.fromkeys(specs)
    family_rank = {name: i for i, name in enumerate(FAMILIES)}
    return tuple(sorted(unique, key=lambda s: (family_rank[s.family], s.n, s.params, s.replicate)))


def ladder_spans(grid: Sequence[FamilySpec], *, seed: int) -> dict[tuple[int, str], float]:
    """Realised ``log10|Aut|`` span of every ladder in *grid*.

    The orchestrator's acceptance bar for the matched design is that a ladder
    falls at least three orders of magnitude in ``|Aut|`` between its first and
    last rung; below that the base is too symmetric-poor to separate the
    hypotheses.  This measures it rather than predicting it, because only rung 0
    has a closed form.

    Args:
        grid: specs to scan.  Non-ladder specs are ignored.
        seed: the campaign seed, passed through to :func:`build`.

    Returns:
        ``{(n, base_name): max(log10|Aut|) - min(log10|Aut|)}`` over the rungs
        present in *grid*, taking the best replicate at each rung (a
        ``symmetry_ladder`` swap is a random search for asymmetry, so the
        *lowest* ``|Aut|`` reached at a rung is the one that ladder attains; a
        ``spider_ladder`` rung is deterministic and the minimum is the value).
        ``base_name`` is the ladder base for ``symmetry_ladder`` and
        ``"spider_k<k>"`` for ``spider_ladder``.
    """
    reached: dict[tuple[int, str], dict[int, float]] = {}
    for spec in grid:
        if spec.family == "symmetry_ladder":
            key = (spec.n, LADDER_BASES[_param(spec, "base")])
            rung = _param(spec, "swaps")
        elif spec.family == "spider_ladder":
            key = (spec.n, f"spider_k{_param(spec, 'legs')}")
            rung = _param(spec, "rung")
        else:
            continue
        value = symmetry.log10_aut(build(spec, seed=seed))
        rungs = reached.setdefault(key, {})
        rungs[rung] = min(rungs.get(rung, value), value)
    return {key: max(rungs.values()) - min(rungs.values()) for key, rungs in reached.items()}
